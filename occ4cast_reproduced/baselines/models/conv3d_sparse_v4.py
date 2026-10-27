# v4: the proposed method

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
# Additions: MLP+time PE point encoder, multi-scale fusion + residuals, dense decoder
# =====================================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """t (N,) in seconds → (N, F), F must be even."""
    def __init__(self, F=16, base_freq=1.0):
        super().__init__()
        assert F % 2 == 0
        half = F // 2
        freqs = (2.0 ** torch.arange(half, dtype=torch.float32)) * math.pi * base_freq
        self.register_buffer("freqs", freqs, persistent=False)
        self.F = F

    def forward(self, t):  # t: (N,) or (N,1)
        if t.dim() == 1:
            t = t.unsqueeze(1)
        ang = t * self.freqs[None, :]
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)  # (N, F)


class SpatialPointMLP(nn.Module):
    """Encode (z,y,x) → F using a tiny MLP; normalize coords to [-1,1]."""
    def __init__(self, F=16, grid_size=(512, 256, 32)):
        super().__init__()
        X, Y, Z = grid_size
        denom = torch.tensor([Z - 1, Y - 1, X - 1], dtype=torch.float32)  # order (z,y,x)
        self.register_buffer("denom_zyx", denom, persistent=False)
        self.mlp = nn.Sequential(
            nn.Linear(3, 32, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(32, F, bias=True),
        )

    def forward(self, zyx_int):
        zyx = zyx_int.to(dtype=torch.float32)
        zyx_norm = 2.0 * (zyx / (self.denom_zyx + 1e-6)) - 1.0
        return self.mlp(zyx_norm)


class PointTimeFeatureBuilder(nn.Module):
    """
    Build per-voxel features from (z,y,x,t):
        - f_sp = MLP(z,y,x)
        - f_tm = sinusoidal PE(t)
        - f_pt = f_sp + f_tm   (keep your current design)
        - aggregate by summing f_pt over identical (b,z,y,x)
    """
    def __init__(self, F=16, grid_size=(512, 256, 32)):
        super().__init__()
        self.F = F
        self.spatial = SpatialPointMLP(F=F, grid_size=grid_size)
        self.time_pe = SinusoidalTimeEmbedding(F=F, base_freq=1.0)

        # for stable ordering (optional)
        X, Y, Z = grid_size
        mult = torch.tensor([(Z + 1) * (Y + 1) * (X + 1),
                             (Y + 1) * (X + 1),
                             (X + 1),
                             1], dtype=torch.long)
        self.register_buffer("sort_multipliers", mult, persistent=False)

    def forward(self, voxel_features, voxel_coords):
        """
        voxel_features: (N, 4) float32 [z, y, x, t_sec]
        voxel_coords  : (N, 4) int32   [b, z, y, x]
        returns:
            feats_agg : (N', F) float32
            coords_uniq: (N', 4) int32 [b, z, y, x]
            current_coords: (M, 4) int32 coords for t≈0 (for skip head)
        """
        if isinstance(voxel_features, np.ndarray):
            voxel_features = torch.from_numpy(voxel_features)
        if isinstance(voxel_coords, np.ndarray):
            voxel_coords = torch.from_numpy(voxel_coords)

        device = self.spatial.denom_zyx.device
        vf = voxel_features.to(device=device, dtype=torch.float32)  # (N,4)
        vc = voxel_coords.to(device=device, dtype=torch.int32)      # (N,4)

        if vf.numel() == 0 or vc.numel() == 0:
            emptyF = torch.zeros((0, self.F), device=device, dtype=torch.float32)
            empty4 = torch.zeros((0, 4), device=device, dtype=torch.int32)
            return emptyF, empty4, empty4

        zyx = vc[:, 1:4].to(torch.float32)
        t   = vf[:, 3].to(torch.float32)

        f_sp = self.spatial(zyx)    # (N, F)
        f_tm = self.time_pe(t)      # (N, F)
        f_pt = f_sp + f_tm          # (N, F)

        coords_uniq, inv = torch.unique(vc, dim=0, return_inverse=True)
        feats_agg = torch.zeros((coords_uniq.size(0), self.F),
                                device=device, dtype=torch.float32)
        feats_agg.index_add_(0, inv, f_pt)

        # optional stable order
        key = (coords_uniq.to(torch.long) * self.sort_multipliers).sum(dim=1)
        order = torch.argsort(key)
        coords_uniq = coords_uniq[order]
        feats_agg = feats_agg[order]

        # t≈0 (current) coords for residual skip-to-output
        cur_mask = torch.isclose(t, torch.zeros_like(t), atol=1e-4)
        cur_coords = vc[cur_mask]
        if cur_coords.numel() > 0:
            cur_coords = torch.unique(cur_coords, dim=0)

        return feats_agg, coords_uniq, cur_coords


def deconv3x3_ani(in_channels, out_channels, stride):  # stride: (sz, sy, sx)
    out_pad = tuple(max(s - 1, 0) for s in stride)
    return nn.ConvTranspose3d(
        in_channels, out_channels, kernel_size=3,
        stride=stride, padding=1, output_padding=out_pad, bias=False
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
            deconv3x3_ani(in_channels, mid1, stride=(2, 2, 2)),
            nn.BatchNorm3d(mid1),
            nn.ReLU(inplace=True),

            nn.Conv3d(mid1, mid1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid1),
            nn.ReLU(inplace=True),

            deconv3x3_ani(mid1, mid2, stride=(8, 4, 4)),
            nn.BatchNorm3d(mid2),
            nn.ReLU(inplace=True),

            nn.Conv3d(mid2, mid2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid2),
            nn.ReLU(inplace=True),

            nn.Conv3d(mid2, out_channels, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, x):
        return self.block(x)


class MSFusion(nn.Module):
    """
    Fuse multi-scale encoder features (densified & resized) to 128ch,
    then add a residual from the encoder's final output (conv_out.dense()).
    """
    def __init__(self, chs_in=(16, 32, 64, 64), ch_out=128):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv3d(sum(chs_in), ch_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, c_list_zyx, enc_out_zyx):
        """
        c_list_zyx: [c1, c2, c3, c4] all resized to enc_out_zyx spatial size (Z,Y,X)
        enc_out_zyx: (B, 128, Z, Y, X)
        """
        x = torch.cat(c_list_zyx, dim=1)
        x = self.fuse(x)
        # residual connection with final encoder output
        if enc_out_zyx.shape[1] == x.shape[1]:
            x = x + enc_out_zyx
        else:
            # channel mismatch shouldn't happen here, but guard just in case
            x = x
        return x  # (B, 128, Z, Y, X)


class Conv3DSparseForecasting(nn.Module):
    """
    Takes the dataset dict (unchanged):
        {
          'voxel_features': [N, 4] (z,y,x,t),
          'voxel_coords'  : [N, 4] (b,z,y,x),
          'labels'        : [B, T_post, X, Y, Z],
          'invalid'       : [B, T_post, X, Y, Z],
          'batch_size'    : int
        }

    Pipeline:
        (z,y,x,t) -- MLP+timePE --sum over t--> (N',F) + coords (N',4)
        -> build SparseConvTensor
        -> run encoder stages (conv_input/1/2/3/4/conv_out) to get multi-scales
        -> densify & resize c1..c4 to (2,32,64), concat, 3D conv fuse -> 128ch
        -> residual add with conv_out.dense()
        -> dense decoder to full res
        -> output residual (copy head) from current-frame sparse occupancy
        -> logits (B, T_post, 512, 256, 32)

    Train: returns loss; Eval: returns logits.
    """
    def __init__(self, p_post, grid_size=(512, 256, 32), feat_dim=16, use_soft_iou=False):
        super().__init__()
        self.grid_size = np.array([512, 256, 32])
        self.F = feat_dim
        self.p_post = p_post
        self.use_soft_iou = use_soft_iou

        # 1) Point feature builder (keeps f_sp + f_tm and sum over time)
        self.pt_builder = PointTimeFeatureBuilder(F=self.F, grid_size=self.grid_size)

        # 2) Unmodified Encoder (we will run its submodules to get multi-scales)
        self.encoder = Encoder(input_channels=self.F, grid_size=np.array(self.grid_size, dtype=np.int32))

        # 3) Multi-scale fusion (to 128 ch) + residual with encoder's final output
        self.ms_fuse = MSFusion(chs_in=(16, 32, 64, 64), ch_out=128)

        # 4) Dense decoder to full res
        self.decoder = DenseDecoderFromSparse(in_channels=128, out_channels=self.p_post)

        # 5) Output residual: sparse skip head from current frame t≈0
        self.skip_head_sparse = spconv.SparseSequential(
            spconv.SparseConv3d(1, self.p_post, kernel_size=3, stride=1, padding=1, bias=True,
                                indice_key='skip_k3_fullres')
        )

        # Losses
        self.bce = nn.BCEWithLogitsLoss()
        if self.use_soft_iou:
            self.soft_iou = SoftIoUWithSigmoidLoss()

    # --- helpers ---
    @staticmethod
    def _to_zyx(x_lwh):
        # (B,C,L,W,H) -> (B,C,Z=H,Y=W,X=L)
        return x_lwh.permute(0, 1, 4, 3, 2).contiguous()

    @staticmethod
    def _to_lwh(x_zyx):
        # (B,C,Z,Y,X) -> (B,C,L=X,W=Y,H=Z)
        return x_zyx.permute(0, 1, 4, 3, 2).contiguous()

    def _build_sparse_from_coords(self, coords_bzyx, batch_size, feature_channels, feature_value=1.0):
        """
        Build a SparseConvTensor at the encoder's full resolution (Z,Y,X) with a constant feature.
        coords_bzyx: (N,4) int32 [b,z,y,x]
        feature_channels: C_in
        """
        device = next(self.parameters()).device
        if coords_bzyx.numel() == 0:
            feats = torch.zeros((0, feature_channels), device=device, dtype=torch.float32)
            idxs  = torch.zeros((0, 4), device=device, dtype=torch.int32)
        else:
            N = coords_bzyx.shape[0]
            feats = torch.full((N, feature_channels), float(feature_value), device=device, dtype=torch.float32)
            idxs  = coords_bzyx.to(device=device, dtype=torch.int32)

        sp = spconv.SparseConvTensor(
            features=feats,
            indices=idxs,
            spatial_shape=(32, 256, 512),  # (Z,Y,X) with trailing [1,0,0] as defined in Encoder
            batch_size=batch_size
        )
        return sp

    @autocast(enabled=False)
    def forward(self, batch):
        device = next(self.parameters()).device

        # =========================
        # 1) Point features (MLP+time PE) and aggregation over time
        # =========================
        vf = batch['voxel_features']   # (N,4) z,y,x,t
        vc = batch['voxel_coords']     # (N,4) b,z,y,x
        feats_agg, coords_uniq, cur_coords = self.pt_builder(vf, vc)  # (N',F), (N',4), (M,4)
        B = int(batch['batch_size'])

        # early exit (empty)
        if feats_agg.numel() == 0 or coords_uniq.numel() == 0:
            enc_final_zyx = torch.zeros((B, 128, 2, 32, 64), device=device, dtype=torch.float32)
            dec_zyx = self.decoder(enc_final_zyx)
            logits = self._to_lwh(dec_zyx)  # (B,T,X,Y,Z)
            if self.training:
                labels  = batch['labels'].to(device=device).clone()
                labels[labels > 0] = 1
                labels = labels.to(torch.float32)
                invalid = batch['invalid'].to(device=device).bool()
                return self.bce(logits[~invalid], labels[~invalid])
            return logits

        # =========================
        # 2) Build input sparse tensor with aggregated features
        # =========================
        sp_in = spconv.SparseConvTensor(
            features=feats_agg.to(device=device, dtype=torch.float32),
            indices=coords_uniq.to(device=device, dtype=torch.int32),
            spatial_shape=self.encoder.sparse_shape,
            batch_size=B
        )

        # =========================
        # 3) Run encoder submodules to collect multi-scales
        # =========================
        s_in  = self.encoder.conv_input(sp_in)     # 16ch @ full res
        s1    = self.encoder.conv1(s_in)           # 16ch
        s2    = self.encoder.conv2(s1)             # 32ch, down
        s3    = self.encoder.conv3(s2)             # 64ch, down
        s4    = self.encoder.conv4(s3)             # 64ch, same res as s3 downed further
        sout  = self.encoder.conv_out(s4)          # 128ch, final encoder output

        # Densify to (B,C,Z,Y,X)
        c1_zyx  = s1.dense()
        c2_zyx  = s2.dense()
        c3_zyx  = s3.dense()
        c4_zyx  = s4.dense()
        out_zyx = sout.dense()                     # (B,128,2,32,64)

        # Resize all c1..c4 to match out_zyx spatial size (Z=2,Y=32,X=64)
        target_size = out_zyx.shape[2:]            # (Z,Y,X)
        c1r = F.interpolate(c1_zyx, size=target_size, mode='trilinear', align_corners=True)
        c2r = F.interpolate(c2_zyx, size=target_size, mode='trilinear', align_corners=True)
        c3r = F.interpolate(c3_zyx, size=target_size, mode='trilinear', align_corners=True)
        c4r = F.interpolate(c4_zyx, size=target_size, mode='trilinear', align_corners=True)

        # =========================
        # 4) Multi-scale fusion + residual with encoder final output
        # =========================
        enc_final_zyx = self.ms_fuse([c1r, c2r, c3r, c4r], out_zyx)  # (B,128,2,32,64)

        # =========================
        # 5) Dense decode to full res
        # =========================
        dec_zyx = self.decoder(enc_final_zyx)      # (B,p_post,32,256,512)
        logits = self._to_lwh(dec_zyx)             # (B,p_post,512,256,32)

        # =========================
        # 6) Output residual (“copy head”) from current frame t≈0
        # =========================
        # Build a 1-channel sparse tensor with only current-frame coords, then conv -> p_post, densify
        if cur_coords.numel() > 0:
            sp_cur = self._build_sparse_from_coords(cur_coords, B, feature_channels=1, feature_value=1.0)
            skip_sp = self.skip_head_sparse(sp_cur)               # (sparse, p_post)
            skip_zyx = skip_sp.dense()                            # (B, p_post, Z=32, Y=256, X=512)
            skip_lwh = self._to_lwh(skip_zyx)                     # (B,p_post,512,256,32)
            logits = logits + skip_lwh

        # =========================
        # 7) Loss / Eval
        # =========================
        if self.training:
            labels  = batch['labels'].to(device=device).clone()
            labels[labels > 0] = 1
            labels = labels.to(torch.float32)
            invalid = batch['invalid'].to(device=device).bool()

            valid_logits = logits[~invalid]
            valid_gt     = labels[~invalid]
            bce_loss = self.bce(valid_logits, valid_gt)
            if self.use_soft_iou:
                # compute soft IoU on valid set
                pred = valid_logits
                tgt  = valid_gt
                pred_sig = torch.sigmoid(pred)
                inter = (pred_sig * tgt).sum()
                union = (pred_sig + tgt - pred_sig * tgt).sum().clamp_min(1e-6)
                soft_iou = 1.0 - inter / union
                return 0.5 * bce_loss + 0.5 * soft_iou
            return bce_loss

        return logits
