# v1: sparse convolution by stacking the time
# spconv_forecast_timechannels_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

import numpy as np
import spconv.pytorch as spconv
from functools import partial


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size,
                                 bias=False, indice_key=indice_key, padding=padding)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size,
                                   stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size,
                                          indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    return spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(inplace=True),
    )


class Encoder(nn.Module):
    """
    Sparse 3D encoder (spconv) that downsamples from full resolution (Z=32, Y=256, X=512)
    to a compact grid (Z=2, Y=32, X=64), returning a dense tensor (B, 128, 2, 32, 64).

    The input features are time-channels per spatial site: C_in = p_pre.
    """
    def __init__(self, input_channels, grid_size_xyz):
        super().__init__()

        # grid_size_xyz is (X, Y, Z); spconv expects spatial_shape as (Z, Y, X)
        self.sparse_shape = list(grid_size_xyz[::-1])  # [Z, Y, X] = [32, 256, 512]
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        # Stem
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, kernel_size=3, padding=1,
                              bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(inplace=True),
        )

        block = post_act_block

        # Stage 1 (keep resolution)
        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1', conv_type='subm'),
        )

        # Stage 2 (down x2, mostly in Y/X; padding=(0,1,1) to reduce Z slower)
        self.conv2 = spconv.SparseSequential(
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1),
                  indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2', conv_type='subm'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2', conv_type='subm'),
        )

        # Stage 3 (down x2)
        self.conv3 = spconv.SparseSequential(
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1,
                  indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3', conv_type='subm'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3', conv_type='subm'),
        )

        # Stage 4 (down x2)
        self.conv4 = spconv.SparseSequential(
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=1,
                  indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4', conv_type='subm'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4', conv_type='subm'),
        )

        # Final Z-downsampling to reach Z=2 (keep Y/X)
        self.conv_out = spconv.SparseSequential(
            spconv.SparseConv3d(64, 128, kernel_size=(2, 1, 1), stride=(2, 1, 1),
                                padding=0, bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, batch_dict, voxel_features=None, voxel_coords=None):
        """
        Args:
            batch_dict: dict with 'batch_size' for building SparseConvTensor
            voxel_features: (N, C_in) np.float32 or torch.float32
            voxel_coords:   (N, 4) or (N, 3):
                            - If (N,4): [b,z,y,x] already padded (preferred).
                            - If (N,3): (z,y,x) (then batch_dict must set 'batch_size'=1).
        Returns:
            feats: dense tensor (B, 128, 2, 32, 64)  # (B, C, Z, Y, X)
        """
        batch_size = batch_dict['batch_size']
        # Allow numpy inputs
        if isinstance(voxel_features, np.ndarray):
            voxel_features = torch.from_numpy(voxel_features)
        if isinstance(voxel_coords, np.ndarray):
            voxel_coords = torch.from_numpy(voxel_coords)

        # Ensure indices have batch column
        if voxel_coords.size(-1) == 3:
            if batch_size != 1:
                raise ValueError("voxel_coords without batch column only valid for batch_size=1")
            bcol = torch.zeros((voxel_coords.shape[0], 1), dtype=voxel_coords.dtype, device=voxel_coords.device)
            voxel_coords = torch.cat([bcol, voxel_coords], dim=1)  # (N, 4) [b,z,y,x]

        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features.to(torch.float32),
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,  # [Z, Y, X] = [32, 256, 512]
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        out = self.conv_out(x)

        feats = out.dense()  # (B, 128, Z=2, Y=32, X=64)
        return feats


def conv3x3(in_channels, out_channels, bias=False):
    return nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias)


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
    Stage1: (B, 128,  4,  64, 128)  stride=(2,2,2)
    Stage2: (B,  64, 32, 256, 512)  stride=(8,4,4)
    Output: (B, out_channels, 32, 256, 512)  -- still (Z, Y, X)
    """
    def __init__(self, in_channels=128, mid1=128, mid2=64, out_channels=1):
        super().__init__()
        self.block = nn.Sequential(
            # Deconv #1
            deconv3x3_ani(in_channels, mid1, stride=(2, 2, 2)),
            nn.BatchNorm3d(mid1),
            nn.ReLU(inplace=True),

            conv3x3(mid1, mid1),
            nn.BatchNorm3d(mid1),
            nn.ReLU(inplace=True),

            # Deconv #2 (anisotropic to jump to full res)
            deconv3x3_ani(mid1, mid2, stride=(8, 4, 4)),
            nn.BatchNorm3d(mid2),
            nn.ReLU(inplace=True),

            conv3x3(mid2, mid2),
            nn.BatchNorm3d(mid2),
            nn.ReLU(inplace=True),

            # Prediction head
            conv3x3(mid2, out_channels, bias=True),
        )

    def forward(self, x):
        return self.block(x)


class SoftIoUWithSigmoidLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        inter = (pred * target).sum()
        union = (pred + target - pred * target).sum().clamp(min=1e-6)
        return (1.0 - inter / union).mean()


class Conv3DSparseForecasting(nn.Module):
    """
    Option A model: spatially sparse encoder (time as channels) + dense decoder.
    Forward API matches your previous model:
        - train: returns loss (scalar)
        - eval:  returns logits of shape (B, p_post, X, Y, Z) = (B, p_post, 512, 256, 32)
    """
    def __init__(self, p_pre, p_post, use_soft_iou=False):
        super().__init__()

        # Keep the canonical grid for Occ4D-Lyft
        self.grid_size_xyz = np.array([512, 256, 32], dtype=np.int32)  # (X, Y, Z)

        self.p_pre = p_pre
        self.p_post = p_post

        # Encoder now consumes p_pre time channels
        self.encoder = Encoder(input_channels=self.p_pre, grid_size_xyz=self.grid_size_xyz)

        # Dense decoder to full resolution
        self.decoder = DenseDecoderFromSparse(in_channels=128, out_channels=self.p_post)

        # Losses (same semantics as your baseline)
        self.bce = nn.BCEWithLogitsLoss()
        self.use_soft_iou = use_soft_iou
        if self.use_soft_iou:
            self.soft_iou = SoftIoUWithSigmoidLoss()

    @autocast(enabled=False)
    def forward(self, batch_dict):
        """
        Args:
            batch_dict = {
                'voxel_features': np/torch float32 [N, p_pre],
                'voxel_coords':   np/torch int32   [N, 4]  (b, z, y, x) after collate,
                'labels':         torch.Tensor     [B, p_post, X, Y, Z],
                'invalid':        torch.BoolTensor [B, p_post, X, Y, Z],
                'batch_size':     int
            }
        """
        voxel_features = batch_dict['voxel_features']
        voxel_coords = batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        # The encoder builds its own SparseConvTensor and returns (B,128,2,32,64)
        encoder_output = self.encoder(batch_dict, voxel_features, voxel_coords)

        # Decode to full resolution (B, p_post, Z, Y, X)
        logits = self.decoder(encoder_output)        # (B, p_post, 32, 256, 512) in (Z, Y, X)

        # Reorder to match labels/invalid layout: (B, p_post, X, Y, Z)
        logits = logits.permute(0, 1, 4, 3, 2)       # -> (B, p_post, 512, 256, 32)

        # Training branch: compute masked loss
        if self.training:
            gt_occ = batch_dict['labels'].clone()
            gt_occ[gt_occ > 0] = 1
            gt_occ = gt_occ.to(dtype=torch.float32)

            invalid_mask = batch_dict['invalid'].bool()

            # Mask out invalid voxels
            valid_logits = logits[~invalid_mask]
            valid_gt     = gt_occ[~invalid_mask]

            bce_loss = self.bce(valid_logits, valid_gt)
            if self.use_soft_iou:
                soft_iou_loss = self.soft_iou(valid_logits, valid_gt)
                return 0.5 * bce_loss + 0.5 * soft_iou_loss
            else:
                return bce_loss

        # Eval: return raw logits (B, p_post, X, Y, Z)
        return logits
