# v5-fixed: time-conditioned dense head (stable init) + pre-encoder proj+LN + dynamic pos_weight
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import spconv.pytorch as spconv


# ============================
# (UNMODIFIED) OpenPCDet-style sparse encoder (as in your code)
# ============================
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
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        self.conv_out = spconv.SparseSequential(
            spconv.SparseConv3d(64, 128, (2, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )

    def forward(self, batch_dict, voxel_features_partial=None, voxel_coords_partial=None):
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
        feats = out.dense()   # (B,128,2,32,64)
        return feats


# ============================
# Point encoders: separate space/time + more headroom
# ============================
class SpatialPointMLP(nn.Module):
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
        return self.mlp(zyx_norm)


class TimePointMLP(nn.Module):
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
        return self.mlp(t)


class SepAggFeatureBuilder(nn.Module):
    """
    Separate space & time, mean-pool time per voxel, concat -> proj+LayerNorm (always).
    """
    def __init__(self, F_sp=128, F_tm=64, grid_size=(512, 256, 32), proj_out=128):
        super().__init__()
        self.grid_size = tuple(grid_size)
        self.spatial = SpatialPointMLP(F_sp=F_sp, grid_size=self.grid_size)
        self.time_mlp = TimePointMLP(F_tm=F_tm)
        in_cat = F_sp + F_tm
        self.out_dim = proj_out if proj_out is not None else in_cat
        self.proj = nn.Sequential(
            nn.Linear(in_cat, self.out_dim, bias=False),
            nn.LayerNorm(self.out_dim),
        )

    def forward(self, voxel_features, voxel_coords):
        if isinstance(voxel_features, np.ndarray):
            voxel_features = torch.from_numpy(voxel_features)
        if isinstance(voxel_coords, np.ndarray):
            voxel_coords = torch.from_numpy(voxel_coords)

        device = self.proj[0].weight.device
        vf = voxel_features.to(device=device, dtype=torch.float32)  # (N,4) z,y,x,t
        vc = voxel_coords.to(device=device, dtype=torch.int32)      # (N,4) b,z,y,x

        if vf.numel() == 0 or vc.numel() == 0:
            feats_voxel = torch.zeros((0, self.out_dim), device=device, dtype=torch.float32)
            coords_uniq = torch.zeros((0, 4), device=device, dtype=torch.int32)
            return feats_voxel, coords_uniq

        coords_uniq, inv = torch.unique(vc, dim=0, return_inverse=True)  # (N',4)
        zyx = coords_uniq[:, 1:4].to(torch.float32)
        f_sp = self.spatial(zyx)  # (N', F_sp)

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

        feats_cat = torch.cat([f_sp, mean_tm], dim=1)    # (N', F_sp+F_tm)
        feats_voxel = self.proj(feats_cat)               # (N', out_dim) + LN
        return feats_voxel, coords_uniq


# ============================
# Dense decoder to features + time-conditioned (FiLM) head (stable init)
# ============================
def conv3x3(in_channels, out_channels, bias=False):
    return nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

def deconv3x3_ani(in_channels, out_channels, stride):
    out_pad = tuple(max(s - 1, 0) for s in stride)
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, output_padding=out_pad, bias=False)

class DenseDecoderFeatures(nn.Module):
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
    Stable FiLM: start at identity (gamma=0, beta=0). Output prior-biased logits.
    """
    def __init__(self, in_channels, p_post, cond_dim=64, prior=0.01):
        super().__init__()
        self.p_post = p_post
        self.time_embed = nn.Embedding(p_post, cond_dim)
        self.to_gamma = nn.Linear(cond_dim, in_channels)
        self.to_beta  = nn.Linear(cond_dim, in_channels)
        # init FiLM to identity
        nn.init.zeros_(self.to_gamma.weight); nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight);  nn.init.zeros_(self.to_beta.bias)

        self.pred = nn.Conv3d(in_channels, 1, kernel_size=1, bias=True)
        # head starts near prior probability π -> bias = logit(π)
        nn.init.zeros_(self.pred.weight)
        with torch.no_grad():
            self.pred.bias.fill_(math.log(prior / max(1.0 - prior, 1e-6)))

    def forward(self, dec_feats):
        B, C, Z, Y, X = dec_feats.shape
        outs = []
        for k in range(self.p_post):
            e = self.time_embed.weight[k]      # (cond_dim,)
            gamma = self.to_gamma(e).view(1, C, 1, 1, 1)  # starts at 0
            beta  = self.to_beta(e).view(1, C, 1, 1, 1)   # starts at 0
            h = dec_feats * (1.0 + gamma) + beta
            outs.append(self.pred(h))           # (B,1,Z,Y,X)
        return torch.cat(outs, dim=1)           # (B, p_post, Z, Y, X)


class SoftIoUWithSigmoidLoss(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        inter = (pred * target).sum()
        union = (pred + target - pred * target).sum().clamp_min(1e-6)
        return 1.0 - inter / union


# ============================
# Top-level model (dataset & encoder contracts preserved)
# ============================
class Conv3DSparseForecasting(nn.Module):
    """
    Input batch dict (unchanged):
      voxel_features: [N,4] float32 (z,y,x,t)
      voxel_coords:   [N,4] int32   (b,z,y,x)
      labels:         [B,T_post,X,Y,Z]
      invalid:        [B,T_post,X,Y,Z]
      batch_size:     int
    Output: if training -> loss; else logits [B,T_post,X,Y,Z]
    """
    def __init__(self, p_pre, p_post, grid_size=(512, 256, 32),
                 F_sp=128, F_tm=64, proj_out=128, use_soft_iou=False,
                 prior=0.01):
        super().__init__()
        self.grid_size = np.array(grid_size, dtype=np.int32)
        self.p_post = p_post
        self.use_soft_iou = use_soft_iou

        self.sepagg = SepAggFeatureBuilder(F_sp=F_sp, F_tm=F_tm,
                                           grid_size=tuple(grid_size), proj_out=proj_out)
        C_in = proj_out if proj_out is not None else (F_sp + F_tm)

        self.encoder = Encoder(input_channels=C_in, grid_size=self.grid_size)
        self.decoder = DenseDecoderFeatures(in_channels=128, out_feat=64)
        self.time_head = TimeConditionedHead(in_channels=64, p_post=self.p_post, cond_dim=64, prior=prior)

    @staticmethod
    def _to_lwh(x_zyx):
        # (B,C,Z,Y,X) -> (B,C,L=X,W=Y,H=Z)
        return x_zyx.permute(0, 1, 4, 3, 2).contiguous()

    def forward(self, batch):
        device = next(self.parameters()).device
        B = int(batch['batch_size'])

        vf = batch['voxel_features']   # (N,4)
        vc = batch['voxel_coords']     # (N,4)

        feats_voxel, coords_uniq = self.sepagg(vf, vc)  # (N', C_in), (N',4)

        # Empty safeguard (rare but possible)
        if feats_voxel.numel() == 0 or coords_uniq.numel() == 0:
            enc = torch.zeros((B, 128, 2, 32, 64), device=device, dtype=torch.float32)
        else:
            sp_in = spconv.SparseConvTensor(
                features=feats_voxel.to(device=device, dtype=torch.float32),
                indices=coords_uniq.to(device=device, dtype=torch.int32),
                spatial_shape=self.encoder.sparse_shape,
                batch_size=B
            )
            x = self.encoder.conv_input(sp_in)
            x = self.encoder.conv1(x)
            x = self.encoder.conv2(x)
            x = self.encoder.conv3(x)
            x = self.encoder.conv4(x)
            x = self.encoder.conv_out(x)
            enc = x.dense()  # (B,128,2,32,64)

        dec = self.decoder(enc)             # (B,64,32,256,512)
        logits_zyx = self.time_head(dec)    # (B,p_post,32,256,512)
        logits = self._to_lwh(logits_zyx)   # (B,p_post,512,256,32)

        if self.training:
            labels  = batch['labels'].to(device=device).clone()
            labels[labels > 0] = 1
            labels = labels.to(torch.float32)
            invalid = batch['invalid'].to(device=device).bool()

            valid_logits = logits[~invalid]
            valid_gt     = labels[~invalid]

            # dynamic positive reweighting (on valid voxels)
            with torch.no_grad():
                pos = valid_gt.sum()
                neg = valid_gt.numel() - pos
                pos_w = (neg / (pos + 1e-6)).clamp(1.0, 200.0)   # cap to avoid exploding grads

            bce = F.binary_cross_entropy_with_logits(valid_logits, valid_gt, pos_weight=pos_w)

            if self.use_soft_iou:
                pred_sig = torch.sigmoid(valid_logits)
                inter = (pred_sig * valid_gt).sum()
                union = (pred_sig + valid_gt - pred_sig * valid_gt).sum().clamp_min(1e-6)
                soft_iou = 1.0 - inter / union
                return 0.5 * bce + 0.5 * soft_iou
            return bce

        return logits
