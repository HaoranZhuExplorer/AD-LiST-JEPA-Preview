# v3: spatio-temporal 3D voxel representation
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.cuda.amp import autocast

import spconv.pytorch as spconv


# =====================================================================================
# (UNMODIFIED) Your default 3D sparse encoder from OpenPCDet-style code
# =====================================================================================

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class Encoder(nn.Module):
    def __init__(self, input_channels, grid_size):
        super().__init__()
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [512, 256, 33] <- [256, 128, 16]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [256, 128, 16] <- [128, 64, 8]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [128, 64, 8] <- [64, 32, 4]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        self.conv_out = spconv.SparseSequential(
            # [[64, 32, 4] -> [64, 32, 2]
            spconv.SparseConv3d(64, 128, (2, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )

    def forward(self, batch_dict, voxel_features_partial=None, voxel_coords_partial=None):
        """
        Returns:
            feats: dense tensor (B, 128, 2, 32, 64)  # [B, C, Z, Y, X]
        """
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features_partial,
            indices=voxel_coords_partial.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        out = self.conv_out(x_conv4)
        feats = out.dense()
        return feats


# =====================================================================================
# Our additions: time-as-feature point encoder + dense decoder + top-level model
# =====================================================================================

def deconv3x3_ani(in_channels, out_channels, stride):  # stride: (sz, sy, sx)
    out_pad = tuple(max(s - 1, 0) for s in stride)
    return nn.ConvTranspose3d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        output_padding=out_pad,
        bias=False,
    )

class DenseDecoderFromSparse(nn.Module):
    """
    Input : (B, 128,  2,  32,  64)  -- (C, Z, Y, X)
    Stage1: (B, 128,  4,  64, 128)  stride=(2, 2, 2)
    Stage2: (B,  64, 32, 256, 512)  stride=(8, 4, 4)
    Output: (B, out_channels, 32, 256, 512)  -- (Z, Y, X)
    """
    def __init__(self, in_channels=128, mid1=128, mid2=64, out_channels=1):
        super().__init__()
        self.block = nn.Sequential(
            # Deconv #1
            deconv3x3_ani(in_channels, mid1, stride=(2, 2, 2)),
            nn.BatchNorm3d(mid1),
            nn.ReLU(inplace=True),

            nn.Conv3d(mid1, mid1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid1),
            nn.ReLU(inplace=True),

            # Deconv #2 (anisotropic to jump to full res)
            deconv3x3_ani(mid1, mid2, stride=(8, 4, 4)),
            nn.BatchNorm3d(mid2),
            nn.ReLU(inplace=True),

            nn.Conv3d(mid2, mid2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid2),
            nn.ReLU(inplace=True),

            # Prediction head
            nn.Conv3d(mid2, out_channels, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, x):
        return self.block(x)


class SinusoidalTimeEmbedding(nn.Module):
    """
    t (N,1) -> (N, F), F must be even. Uses sin/cos at exponentially spaced frequencies.
    """
    def __init__(self, F=16, base_freq=1.0):
        super().__init__()
        assert F % 2 == 0, "F must be even for sin/cos positional encoding"
        half = F // 2
        # frequencies in radians; log-spaced: base_freq * 2^k * pi
        freqs = (2.0 ** torch.arange(half, dtype=torch.float32)) * math.pi * base_freq
        self.register_buffer("freqs", freqs, persistent=False)
        self.F = F

    def forward(self, t):
        """
        t: (N,) or (N,1), typically in seconds in [-1.0, 0.0]
        returns: (N, F)
        """
        if t.dim() == 1:
            t = t.unsqueeze(1)
        ang = t * self.freqs[None, :]  # (N, half)
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)  # (N, F)
        return emb


class SpatialPointMLP(nn.Module):
    """
    Encode (z, y, x) -> F with a small MLP. Coordinates are normalized to [-1, 1].
    """
    def __init__(self, F=16, grid_size=(512, 256, 32)):
        super().__init__()
        # Expect grid_size in (X, Y, Z)
        X, Y, Z = grid_size
        denom = torch.tensor([Z - 1, Y - 1, X - 1], dtype=torch.float32)  # for (z,y,x)
        self.register_buffer("denom_zyx", denom, persistent=False)
        self.mlp = nn.Sequential(
            nn.Linear(3, 32, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(32, F, bias=True),
        )

    def forward(self, zyx_int):
        """
        zyx_int: (N, 3) integer or float coordinates (z, y, x)
        returns: (N, F)
        """
        zyx = zyx_int.to(dtype=torch.float32)
        # normalize to [-1, 1]
        zyx_norm = 2.0 * (zyx / (self.denom_zyx + 1e-6)) - 1.0
        return self.mlp(zyx_norm)


class PointTimeAggregator(nn.Module):
    """
    Build point features:
      - spatial MLP on (z, y, x) -> (N, F)
      - sinusoidal time embedding on t -> (N, F)
      - sum: (N, F)
      - aggregate over identical (b, z, y, x) by *summing over time* -> (N', F), (N', 4)
    """
    def __init__(self, F=16, grid_size=(512, 256, 32)):
        super().__init__()
        self.F = F
        self.spatial = SpatialPointMLP(F=F, grid_size=grid_size)
        self.time_pe = SinusoidalTimeEmbedding(F=F, base_freq=1.0)

        # cache multipliers for sorting key (for a stable order)
        X, Y, Z = grid_size
        mult = torch.tensor([(Z + 1) * (Y + 1) * (X + 1),
                             (Y + 1) * (X + 1),
                             (X + 1),
                             1], dtype=torch.long)
        self.register_buffer("sort_multipliers", mult, persistent=False)

    def forward(self, voxel_features_N4, voxel_coords_N4):
        """
        voxel_features_N4: (N, 4) float32 [z, y, x, t_seconds]
        voxel_coords_N4 : (N, 4) int32   [b, z, y, x]

        returns:
          features_agg: (N', F) float32
          coords_uniq : (N', 4) int32   [b, z, y, x]
        """
        if isinstance(voxel_features_N4, np.ndarray):
            voxel_features_N4 = torch.from_numpy(voxel_features_N4)
        if isinstance(voxel_coords_N4, np.ndarray):
            voxel_coords_N4 = torch.from_numpy(voxel_coords_N4)

        device = self.spatial.denom_zyx.device
        vf = voxel_features_N4.to(device=device, dtype=torch.float32)  # (N,4)
        vc = voxel_coords_N4.to(device=device, dtype=torch.int32)      # (N,4)

        if vf.numel() == 0 or vc.numel() == 0:
            # return empty tensors on correct device/dtype
            return (torch.zeros((0, self.F), device=device, dtype=torch.float32),
                    torch.zeros((0, 4), device=device, dtype=torch.int32))

        # split inputs
        zyx = vc[:, 1:4].to(torch.float32)          # authoritative coords for space
        t   = vf[:, 3].to(torch.float32)            # time seconds in [-1, 0]
        # encode
        f_sp = self.spatial(zyx)                    # (N, F)
        f_tm = self.time_pe(t)                      # (N, F)
        f_pt = f_sp + f_tm                          # (N, F)

        # unique by (b,z,y,x) and sum features across time
        coords_uniq, inv = torch.unique(vc, dim=0, return_inverse=True)
        features_agg = torch.zeros((coords_uniq.size(0), self.F),
                                   device=device, dtype=torch.float32)
        features_agg.index_add_(0, inv, f_pt)

        # optional: stable sort by (b,z,y,x) for better kernel locality
        key = (coords_uniq.to(torch.long) * self.sort_multipliers).sum(dim=1)
        order = torch.argsort(key)
        coords_uniq = coords_uniq[order]
        features_agg = features_agg[order]

        return features_agg, coords_uniq


class Conv3DSparseForecasting(nn.Module):
    """
    Top-level model:
      - INPUT: batch dict from your dataset/loader:
            {
              'voxel_features': [N, 4] (z, y, x, t_seconds),
              'voxel_coords'  : [N, 4] (b, z, y, x),
              'labels'        : [B, T_post, X, Y, Z],
              'invalid'       : [B, T_post, X, Y, Z],
              'batch_size'    : int
            }
      - PIPELINE:
            (z,y,x,t) --MLP/PE--> (N,16) --sum over time @ (b,z,y,x)--> (N',16)
            -> (Encoder w/ input_channels=16) -> (B,128,2,32,64)
            -> Dense decoder -> logits (B, T_post, 512, 256, 32)
      - TRAIN: returns loss
      - EVAL : returns logits
    """
    def __init__(self, p_post, feat_dim=16, use_soft_iou=False):
        super().__init__()
        self.grid_size = np.array([512, 256, 32])
        self.F = feat_dim
        self.p_post = p_post
        self.use_soft_iou = use_soft_iou

        # Point encoder/aggregator
        self.point_enc = PointTimeAggregator(F=self.F, grid_size=self.grid_size)

        # Unmodified sparse encoder (expects input_channels = F)
        self.encoder = Encoder(input_channels=self.F, grid_size=np.array(self.grid_size, dtype=np.int32))

        # Dense decoder to full resolution
        self.decoder = DenseDecoderFromSparse(in_channels=128, out_channels=self.p_post)

        # Losses
        self.bce = nn.BCEWithLogitsLoss()
        if self.use_soft_iou:
            self.soft_iou = SoftIoUWithSigmoidLoss()

    def _to_device(self, x, device):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, non_blocking=True)
        return torch.as_tensor(x, device=device)

    @autocast(enabled=False)
    def forward(self, batch):
        """
        Args:
            batch: dict from your Dataset.collate_batch
        Returns:
            loss (train) or logits (eval) with shape (B, p_post, 512, 256, 32)
        """
        device = next(self.parameters()).device

        # ------------------------------------------------------------------
        # 1) Build aggregated sparse inputs (N', F) and coords (N', 4)
        # ------------------------------------------------------------------
        vf = batch['voxel_features']   # N x 4
        vc = batch['voxel_coords']     # N x 4
        feats_agg, coords_uniq = self.point_enc(vf, vc)  # (N', F), (N',4 [b,z,y,x])

        B = int(batch['batch_size'])

        # ------------------------------------------------------------------
        # 2) Run sparse encoder (or fallback to zeros if empty)
        # ------------------------------------------------------------------
        if feats_agg.numel() == 0 or coords_uniq.numel() == 0:
            # No active sites in the whole batch: produce zeros at encoder output shape
            enc_out = torch.zeros((B, 128, 2, 32, 64), device=device, dtype=torch.float32)
        else:
            enc_out = self.encoder(
                batch_dict={'batch_size': B},
                voxel_features_partial=feats_agg.to(device=device, dtype=torch.float32),
                voxel_coords_partial=coords_uniq.to(device=device, dtype=torch.int32)
            )  # (B,128,2,32,64)

        # ------------------------------------------------------------------
        # 3) Dense decode to full resolution, permute to (B, T, X, Y, Z)
        # ------------------------------------------------------------------
        dec_zyx = self.decoder(enc_out)                   # (B, p_post, 32, 256, 512)
        logits = dec_zyx.permute(0, 1, 4, 3, 2).contiguous()  # (B, p_post, 512, 256, 32)

        # ------------------------------------------------------------------
        # 4) Train/Eval branches
        # ------------------------------------------------------------------
        if self.training:
            labels  = self._to_device(batch['labels'], device)   # (B,T,X,Y,Z)
            invalid = self._to_device(batch['invalid'], device).bool()

            gt = labels.clone()
            gt[gt > 0] = 1
            gt = gt.to(dtype=torch.float32)

            valid_logits = logits[~invalid]
            valid_gt     = gt[~invalid]

            bce_loss = self.bce(valid_logits, valid_gt)
            if self.use_soft_iou:
                soft_iou_loss = self.soft_iou(valid_logits, valid_gt)
                return 0.5 * bce_loss + 0.5 * soft_iou_loss
            return bce_loss

        return logits
