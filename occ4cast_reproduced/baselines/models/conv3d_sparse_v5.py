# v5

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.cuda.amp import autocast

import spconv.pytorch as spconv


# =====================================================================================
# (UNMODIFIED) OpenPCDet-style sparse encoder you provided earlier
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
# Point encoders: (2) Separate space & time before aggregation, (4) higher capacity
# =====================================================================================

class SpatialPointMLP(nn.Module):
    """
    Encode integer (z,y,x) -> F_sp using a wider MLP.
    We normalize coords to [-1,1] with known grid size.
    """
    def __init__(self, F_sp=128, grid_size=(512, 256, 32)):
        super().__init__()
        X, Y, Z = grid_size
        denom = torch.tensor([Z - 1, Y - 1, X - 1], dtype=torch.float32)  # (z,y,x)
        self.register_buffer("denom_zyx", denom, persistent=False)
        self.mlp = nn.Sequential(
            nn.Linear(3, 128, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(128, F_sp, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, zyx_int):
        zyx = zyx_int.to(dtype=torch.float32)
        zyx_norm = 2.0 * (zyx / (self.denom_zyx + 1e-6)) - 1.0
        return self.mlp(zyx_norm)  # (N', F_sp)


class TimePointMLP(nn.Module):
    """
    Encode scalar time t (seconds) -> F_tm with a small MLP (learned, no sinusoid).
    """
    def __init__(self, F_tm=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 64, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(64, F_tm, bias=True),
        )

    def forward(self, t):
        if t.dim() == 1:
            t = t.unsqueeze(1)
        return self.mlp(t)  # (N, F_tm)


class SepAggFeatureBuilder(nn.Module):
    """
    Build per-voxel features with separated space/time:
        - Compute coords_uniq, inv from voxel_coords (N,4) [b,z,y,x]
        - Spatial MLP once per unique voxel -> (N', F_sp)
        - Time MLP per point on t -> (N, F_tm); aggregate (mean) per voxel via inv -> (N', F_tm)
        - Concat -> (N', F_sp + F_tm) and (N', 4) coords
    """
    def __init__(self, F_sp=128, F_tm=64, grid_size=(512, 256, 32), proj_out=None):
        super().__init__()
        self.spatial = SpatialPointMLP(F_sp=F_sp, grid_size=grid_size)
        self.time_mlp = TimePointMLP(F_tm=F_tm)
        in_cat = F_sp + F_tm
        self.use_proj = proj_out is not None
        if self.use_proj:
            self.proj = nn.Sequential(
                nn.Linear(in_cat, proj_out, bias=False),
                nn.LayerNorm(proj_out),
            )
        else:
            self.proj = nn.Identity()

    def forward(self, voxel_features, voxel_coords):
        """
        voxel_features: (N,4) float32 [z, y, x, t_sec]
        voxel_coords  : (N,4) int32   [b, z, y, x]
        returns:
            feats_voxel: (N', C_in) float32
            coords_uniq: (N', 4) int32 [b, z, y, x]
        """
        if isinstance(voxel_features, np.ndarray):
            voxel_features = torch.from_numpy(voxel_features)
        if isinstance(voxel_coords, np.ndarray):
            voxel_coords = torch.from_numpy(voxel_coords)

        device = self.spatial.denom_zyx.device
        vf = voxel_features.to(device=device, dtype=torch.float32)
        vc = voxel_coords.to(device=device, dtype=torch.int32)

        if vf.numel() == 0 or vc.numel() == 0:
            emptyC = self.proj[-1].normalized_shape[0] if self.use_proj else (self.spatial.mlp[-2].out_features + self.time_mlp.mlp[-1].out_features)
            feats_voxel = torch.zeros((0, emptyC), device=device, dtype=torch.float32)
            coords_uniq = torch.zeros((0, 4), device=device, dtype=torch.int32)
            return feats_voxel, coords_uniq

        # Unique voxel coords
        coords_uniq, inv = torch.unique(vc, dim=0, return_inverse=True)

        # Spatial once per voxel
        zyx = coords_uniq[:, 1:4].to(torch.float32)
        f_sp = self.spatial(zyx)  # (N', F_sp)

        # Time per point -> aggregate mean per voxel
        t = vf[:, 3].to(torch.float32)
        f_tm_pt = self.time_mlp(t)  # (N, F_tm)
        num_vox = coords_uniq.size(0)
        F_tm = f_tm_pt.size(1)
        sum_tm = torch.zeros((num_vox, F_tm), device=device, dtype=torch.float32)
        cnt = torch.zeros((num_vox, 1), device=device, dtype=torch.float32)
        sum_tm.index_add_(0, inv, f_tm_pt)
        ones = torch.ones((f_tm_pt.size(0), 1), device=device, dtype=torch.float32)
        cnt.index_add_(0, inv, ones)
        mean_tm = sum_tm / cnt.clamp_min(1.0)

        feats_cat = torch.cat([f_sp, mean_tm], dim=1)
        feats_voxel = self.proj(feats_cat)

        return feats_voxel, coords_uniq


# =====================================================================================
# Dense decoder that outputs features (not logits); time-conditioned head (UNO-style)
# =====================================================================================

def conv3x3(in_channels, out_channels, bias=False):
    return nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

def deconv3x3_ani(in_channels, out_channels, stride):  # stride: (sz, sy, sx)
    out_pad = tuple(max(s - 1, 0) for s in stride)
    return nn.ConvTranspose3d(
        in_channels, out_channels, kernel_size=3,
        stride=stride, padding=1, output_padding=out_pad, bias=False
    )

class DenseDecoderFeatures(nn.Module):
    """
    Input  : (B, 128,  2,  32,  64)
    Stage1 : (B, 128,  4,  64, 128)   stride=(2,2,2)
    Stage2 : (B,  64, 32, 256, 512)   stride=(8,4,4)
    Output : (B, C_feat, 32, 256, 512)
    """
    def __init__(self, in_channels=128, mid1=128, mid2=64, out_feat=64):
        super().__init__()
        self.block = nn.Sequential(
            deconv3x3_ani(in_channels, mid1, stride=(2, 2, 2)),
            nn.BatchNorm3d(mid1),
            nn.ReLU(inplace=True),

            conv3x3(mid1, mid1),
            nn.BatchNorm3d(mid1),
            nn.ReLU(inplace=True),

            deconv3x3_ani(mid1, mid2, stride=(8, 4, 4)),
            nn.BatchNorm3d(mid2),
            nn.ReLU(inplace=True),

            conv3x3(mid2, out_feat),
            nn.BatchNorm3d(out_feat),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)  # (B, out_feat, 32, 256, 512)


class TimeConditionedHead(nn.Module):
    """
    FiLM-modulated per-horizon head:
      - Learn an embedding e_k for each step k in [0..p_post-1]
      - Map e_k -> (gamma_k, beta_k) (per-channel FiLM on decoder features)
      - Shared 1x1x1 conv to 1 logit channel
      - Loop over k (p_post is small: ~11)

    Input : (B, C_feat, Z=32, Y=256, X=512)
    Output: (B, p_post, Z, Y, X)
    """
    def __init__(self, in_channels, p_post, cond_dim=64):
        super().__init__()
        self.p_post = p_post
        self.time_embed = nn.Embedding(p_post, cond_dim)
        self.to_film = nn.Sequential(
            nn.Linear(cond_dim, 2 * in_channels),
            nn.SiLU(inplace=True)
        )
        self.pred = nn.Conv3d(in_channels, 1, kernel_size=1, bias=True)

    def forward(self, dec_feats):
        B, C, Z, Y, X = dec_feats.shape
        logits_list = []
        for k in range(self.p_post):
            e = self.time_embed.weight[k]                  # (cond_dim,)
            film = self.to_film(e)                         # (2*C,)
            gamma, beta = film[:C], film[C:]
            gamma = gamma.view(1, C, 1, 1, 1)
            beta  = beta.view(1, C, 1, 1, 1)
            h = dec_feats * (1.0 + gamma) + beta           # FiLM
            logit_k = self.pred(h)                         # (B,1,Z,Y,X)
            logits_list.append(logit_k)
        return torch.cat(logits_list, dim=1)               # (B,p_post,Z,Y,X)


class SoftIoUWithSigmoidLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        inter = (pred * target).sum()
        union = (pred + target - pred * target).sum().clamp_min(1e-6)
        return 1.0 - inter / union


# =====================================================================================
# Top-level model (keeps dataset & encoder contracts)
# =====================================================================================

class Conv3DSparseForecasting(nn.Module):
    """
    Consumes the *unchanged* dataset dict:
      'voxel_features': [N,4] (z,y,x,t),
      'voxel_coords'  : [N,4] (b,z,y,x),
      'labels'        : [B,T_post,X,Y,Z],
      'invalid'       : [B,T_post,X,Y,Z],
      'batch_size'    : int

    Pipeline:
      - Separate space & time encoding for points, mean-pool time per voxel, concat -> per-voxel features
      - Build sparse tensor and run the provided Encoder (unchanged)
      - Dense decoder -> feature volume
      - Time-conditioned head (FiLM per step) -> logits (B, p_post, Z, Y, X)
      - Permute to (B, p_post, X, Y, Z) to match labels
    """
    def __init__(self, p_post, grid_size=(512, 256, 32),
                 F_sp=128, F_tm=64, proj_out=None,  # proj_out -> channels fed to Encoder (if None, uses concat size)
                 use_soft_iou=False):
        super().__init__()
        self.grid_size = np.array([512, 256, 32])
        self.p_post = p_post
        self.use_soft_iou = use_soft_iou

        # (2) Separate space & time before sum; (4) higher capacity
        self.sepagg = SepAggFeatureBuilder(F_sp=F_sp, F_tm=F_tm, grid_size=self.grid_size,
                                           proj_out=proj_out)
        C_in = proj_out if proj_out is not None else (F_sp + F_tm)

        # Unmodified Encoder (we only pass different input_channels)
        self.encoder = Encoder(input_channels=C_in, grid_size=np.array(self.grid_size, dtype=np.int32))

        # Dense decoder to feature volume, then time-conditioned head
        self.decoder = DenseDecoderFeatures(in_channels=128, out_feat=64)
        self.time_head = TimeConditionedHead(in_channels=64, p_post=self.p_post, cond_dim=64)

        # Losses
        self.bce = nn.BCEWithLogitsLoss()
        if self.use_soft_iou:
            self.soft_iou = SoftIoUWithSigmoidLoss()

    # --- utility permutations ---
    @staticmethod
    def _to_lwh(x_zyx):
        # (B,C,Z,Y,X) -> (B,C,L=X,W=Y,H=Z)
        return x_zyx.permute(0, 1, 4, 3, 2).contiguous()

    @autocast(enabled=False)
    def forward(self, batch):
        device = next(self.parameters()).device

        # ===== 1) build per-voxel features (separate space & time) =====
        vf = batch['voxel_features']   # (N,4) z,y,x,t
        vc = batch['voxel_coords']     # (N,4) b,z,y,x
        feats_voxel, coords_uniq = self.sepagg(vf, vc)  # (N', C_in), (N',4)
        B = int(batch['batch_size'])

        # Handle empty samples gracefully
        if feats_voxel.numel() == 0 or coords_uniq.numel() == 0:
            enc_final_zyx = torch.zeros((B, 128, 2, 32, 64), device=device, dtype=torch.float32)
            dec_feat_zyx = self.decoder(enc_final_zyx)             # (B,64,32,256,512)
            logits_zyx = self.time_head(dec_feat_zyx)              # (B,p_post,32,256,512)
            logits_lwh = self._to_lwh(logits_zyx)                  # (B,p_post,512,256,32)
            if self.training:
                labels  = batch['labels'].to(device=device).clone()
                labels[labels > 0] = 1
                labels = labels.to(torch.float32)
                invalid = batch['invalid'].to(device=device).bool()
                return self.bce(logits_lwh[~invalid], labels[~invalid])
            return logits_lwh

        # ===== 2) sparse tensor -> encoder =====
        sp_in = spconv.SparseConvTensor(
            features=feats_voxel.to(device=device, dtype=torch.float32),
            indices=coords_uniq.to(device=device, dtype=torch.int32),
            spatial_shape=self.encoder.sparse_shape,
            batch_size=B
        )
        # run encoder stages
        s_in  = self.encoder.conv_input(sp_in)
        s1    = self.encoder.conv1(s_in)
        s2    = self.encoder.conv2(s1)
        s3    = self.encoder.conv3(s2)
        s4    = self.encoder.conv4(s3)
        sout  = self.encoder.conv_out(s4)
        enc_final_zyx = sout.dense()                                 # (B,128,2,32,64)

        # ===== 3) dense decoder -> features; time-conditioned head -> logits =====
        dec_feat_zyx = self.decoder(enc_final_zyx)                    # (B,64,32,256,512)
        logits_zyx = self.time_head(dec_feat_zyx)                     # (B,p_post,32,256,512)
        logits_lwh = self._to_lwh(logits_zyx)                         # (B,p_post,512,256,32)

        # ===== 4) loss / eval =====
        if self.training:
            labels  = batch['labels'].to(device=device).clone()
            labels[labels > 0] = 1
            labels = labels.to(torch.float32)
            invalid = batch['invalid'].to(device=device).bool()
            valid_logits = logits_lwh[~invalid]
            valid_gt     = labels[~invalid]
            bce_loss = self.bce(valid_logits, valid_gt)
            if self.use_soft_iou:
                pred_sig = torch.sigmoid(valid_logits)
                inter = (pred_sig * valid_gt).sum()
                union = (pred_sig + valid_gt - pred_sig * valid_gt).sum().clamp_min(1e-6)
                soft_iou = 1.0 - inter / union
                return 0.5 * bce_loss + 0.5 * soft_iou
            return bce_loss

        return logits_lwh
