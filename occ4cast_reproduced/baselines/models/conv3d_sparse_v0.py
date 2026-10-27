# v0, ignore timestamp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import spconv.pytorch as spconv
from functools import partial
import numpy as np


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
    Input : (B, 128,  2,  32,  64)
    Stage1: (B, 128,  4,  64, 128)   stride=(2,2,2)
    Stage2: (B,  64, 32, 256, 512)   stride=(8,4,4)
    Output: (B, p_post, 32, 256, 512)
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
        union = (pred + target - pred * target).sum()
        iou = inter / union
        return (1 - iou).mean()


# ========= top-level module with decoder and training loss (encoder untouched) =========
class Conv3DSparseForecasting(nn.Module):
    def __init__(self, p_pre, p_post, use_soft_iou=False):
        super().__init__()

        self.sparse_shape = np.array([512, 256, 32]) # TODO: hard code for now 
 
        self.p_pre = p_pre
        self.p_post = p_post

        self.encoder = Encoder(input_channels=4, grid_size=self.sparse_shape)

        self.decoder = DenseDecoderFromSparse(in_channels=128, out_channels=self.p_post)

        # (ADDED) training losses matching original script behavior
        self.bce = nn.BCEWithLogitsLoss()
        self.use_soft_iou = use_soft_iou
        if self.use_soft_iou:
            self.soft_iou = SoftIoUWithSigmoidLoss()

    def forward(self, batch_dict):
        voxel_features, voxel_coords, batch_size = batch_dict['voxel_features'], batch_dict['voxel_coords'], batch_dict['batch_size']
        _ = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        # Encoder output: (B, 128, 2, 32, 64)
        encoder_output = self.encoder(batch_dict, voxel_features, voxel_coords)
        
        # Dense decoder to full res: (B, p_post, 512, 256, 32)
        logits = self.decoder(encoder_output)
        logits = logits.permute(0, 1, 4, 3, 2) # (B, p_post, 32, 256, 512)

        gt_occ = batch_dict['labels']
        invalid_mask =  batch_dict['invalid']

        # --- Training branch (same semantics as original script) ---
        if self.training:
            # binarize targets like original
            gt_occ = gt_occ.clone()
            gt_occ[gt_occ > 0] = 1
            gt_occ = gt_occ.float()

            # mask out invalid voxels
            valid_logits = logits[~invalid_mask]
            valid_gt     = gt_occ[~invalid_mask]

            bce_loss = self.bce(valid_logits, valid_gt)
            if self.use_soft_iou:
                soft_iou_loss = self.soft_iou(valid_logits, valid_gt)
                return 0.5 * bce_loss + 0.5 * soft_iou_loss
            else:
                return bce_loss

        # --- Eval branch: return logits ---
        return logits
