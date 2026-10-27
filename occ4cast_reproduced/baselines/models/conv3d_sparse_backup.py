from typing import Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import spconv.pytorch as spconv

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


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, indice_key=None, norm_fn=None):
        super(SparseBasicBlock, self).__init__()
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x.features

        assert x.features.dim() == 2, 'x.features.dim()=%d' % x.features.dim()

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity)
        out = replace_feature(out, self.relu(out.features))

        return out

class Encoder(nn.Module):
    # openpcdet's standard sparse 3D convolution backbone, 
    # the only difference is last conv_out layer's z dimension kernel size=2 instead 3
    # so that the output shape  is [B, 128, 2, 32, 64]
    def __init__(self, input_channels, grid_size, **kwargs):
        super().__init__()
        self.sparse_shape = [grid_size[2]+1, grid_size[1], grid_size[0]] # similar to openpcdet's standard backbone, pad z by 1
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv_input = spconv.SparseSequential(
            # [512, 256, 33] -> [512, 256, 33]
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            # [512, 256, 33] -> [512, 256, 33]
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [512, 256, 33] -> [256, 128, 17]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [256, 128, 17] -> [128, 64, 9]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [128, 64, 9] -> [64, 32, 4] 
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        
        last_pad = 0
        self.conv_out = spconv.SparseSequential(
            # [64, 32, 4] -> [64, 32, 2] 
            spconv.SparseConv3d(64, 128, (2, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )

    def forward(self, batch_dict, voxel_features=None, voxel_coords=None):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded voxel feature: sparse tensor (B, 128, 2, 32, 64)
        """
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_dict['batch_size_time']
        )

        x = self.conv_input(input_sp_tensor) #  sparse, (B, 16, 33, 256, 512)  
        x_conv1 = self.conv1(x)       # sparse, (B, 16, 33, 256, 512)  
        x_conv2 = self.conv2(x_conv1) # sparse, (B, 32, 17, 128, 256)  
        x_conv3 = self.conv3(x_conv2) # sparse, (B, 65, 9, 64, 128)
        x_conv4 = self.conv4(x_conv3) # sparse, (B, 64, 4, 32, 64)
        out = self.conv_out(x_conv4)  # dense,  (B, 128, 2, 32, 64)
        feats = out.dense() 
        return feats



class Predictor(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv3d(256, 256, 3, padding=1, stride=1),
            nn.BatchNorm3d(256),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(256, 256, 3, padding=1, stride=1),
            nn.BatchNorm3d(256),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(256, 256, 3, padding=1, stride=1),
            nn.BatchNorm3d(256),
            nn.ReLU()
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        feats = out
        return feats



def conv3x3(in_channels, out_channels, bias=False):
    return nn.Conv3d(
        in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias
    )


# def deconv3x3(in_channels, out_channels, stride):
#     return nn.ConvTranspose3d(
#         in_channels,
#         out_channels,
#         kernel_size=3,
#         stride=stride,
#         padding=1,
#         output_padding=1,
#         bias=False,
#     )

def deconv3x3(in_channels, out_channels, stride=(1,2,2)):
    return nn.ConvTranspose3d(
        in_channels, out_channels,
        kernel_size=3,
        stride=stride,            # (1,2,2) keeps T, upsamples H,W
        padding=1,
        output_padding=(0,1,1),   # match the spatial upsample
        bias=False,
    )


def relu(inplace=True):
    return nn.ReLU(inplace=inplace)

def bn(num_features):
    return nn.BatchNorm3d(num_features=num_features)

class Decoder(nn.Module):
    """
    Lift BEV features (B, C_bev, T, H_bev, W_bev) to 3D occupancy logits (B, T, X, Y, Z).
    Assumes H_bev=Y/8, W_bev=X/8 from your sparse backbone.

    Default target grid (X, Y, Z) = (512, 256, 32).
    """
    def __init__(
        self,
    ):
        super().__init__()

        self.block = nn.Sequential(
            deconv3x3(256, 128, stride=(1,2,2)),
            bn(128),
            relu(),
            conv3x3(128, 128),
            bn(128),
            relu(),
            deconv3x3(128, 128, stride=(1,2,2)),
            bn(128),
            relu(),
            conv3x3(128, 128),
            bn(128),
            relu(),
            deconv3x3(128, 64, stride=(1,2,2)),
            bn(64),
            relu(),
            conv3x3(64, 64),
            bn(64),
            relu(),
            conv3x3(64, 32, bias=True),
        )


    def forward(self, x):
        """
        Args:
            bev_feat: (B, C_bev=256, T, H_bev=Y/8, W_bev=X/8)
        Returns:
            occ_logits: (B, T, X, Y, Z)  [matches your label layout]
        """
        
        return self.block(x)


# ---------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------
class SoftIoUWithSigmoidLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(pred_logits)
        inter = (p * target).sum()
        union = (p + target - p * target).sum().clamp_min(self.eps)
        return 1.0 - inter / union


# ---------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------
class Conv3DSparseForecasting(nn.Module):
    """
    """
    def __init__(
        self,
        p_pre: int,                              # history length (dataset includes current)
        p_post: int,                             # future steps (labels include current)
        grid_size_xyz: Tuple[int, int, int] = (512, 256, 32),  # (X, Y, Z)
        in_feat_dim: int = 3,                    # one feature per point (e.g., ones)
        use_soft_iou: bool = False,
    ):
        super().__init__()
        self.grid_size_xyz = tuple(grid_size_xyz)
        self.p_pre = int(p_pre)
        self.p_post = int(p_post)
        self.use_soft_iou = use_soft_iou

        X, Y, Z = self.grid_size_xyz

        self.backbone = Encoder(input_channels=in_feat_dim, grid_size=[X, Y, Z])
        self.predictor = Predictor()
        self.decoder = Decoder()
        
        # self.linear = torch.nn.Conv3d(
        #     self.p_pre, self.p_post, 3, stride=1, padding=1, bias=True
        # )

        # Losses
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        if use_soft_iou:
            self.soft_iou = SoftIoUWithSigmoidLoss()


    def forward(self, batch_dict, compute_loss: bool = True):
        """
        batch:
          - input_feats:     [N, 1] float
          - input_coords_bt: [N, 4] int32 (bt, z, y, x)
          - batch_size:      B
          - batch_size_time: B * p_pre
          - labels:          [B, T_post, X, Y, Z]
          - invalid:         [B, T_post, X, Y, Z]
        """
        device = next(self.parameters()).device
        B  = int(batch_dict['batch_size'])
        BT = int(batch_dict['batch_size_time'])
        X, Y, Z = self.grid_size_xyz
        T_hist = self.p_pre

        # Ensure features/coords on device for spconv
        voxel_features = batch_dict['input_feats'].to(device, non_blocking=True)
        voxel_coords = batch_dict['input_coords_bt'].to(device=device, dtype=torch.int32, non_blocking=True)
        

        encoder_output = self.backbone(batch_dict, voxel_features, voxel_coords)
        BT, C, D, H, W = encoder_output.shape
        encoder_output_bev = encoder_output.reshape(BT, -1, H, W) # (B*T, 128*2, 32, 64)
        encoder_output_bev = encoder_output_bev.reshape(B, T_hist, -1, H, W) # (B, T, 128*2, 32, 64)
        encoder_output_bev = encoder_output_bev.permute(0, 2, 1, 3, 4) # (B, 128*2, T, 32, 64)
        predictor_output_bev = self.predictor(encoder_output_bev) # (B, 128*2, T, 32, 64)

        output = self.decoder(predictor_output_bev) # [B, 32, T, 256, 512]
        output = output.permute(0,2,4,3,1)

        gt_occ = batch_dict['labels'].to(device) # [B, T, 512, 256, 32]   
        invalid_mask = batch_dict['invalid'].to(device, dtype=torch.bool) 

        # input_dense = batch_dict['input_dense'].to(device) #
        # identity = self.linear(input_dense)
        # output = output+identity

        #print("debug gt_occ", gt_occ.shape, torch.sum(gt_occ))

        nums = voxel_features.shape[0]
        voxel_features_all_one = torch.ones(nums, 1).to(voxel_features.device)
        input_sp_tensor_ones = spconv.SparseConvTensor(
            features=voxel_features_all_one,
            indices=voxel_coords.int(),
            spatial_shape=[32, 256, 512],
            batch_size=batch_dict['batch_size_time']
        )
        input_sp_tensor_ones = input_sp_tensor_ones.dense().permute(0,1,4,3,2)
        gt_occ = input_sp_tensor_ones
        print("debug gt_occ", gt_occ.shape, torch.sum(gt_occ), voxel_coords.shape)
        print("debug input_dense", torch.sum(batch_dict['input_dense']), torch.equal(batch_dict['input_dense'].to(voxel_features.device), gt_occ))

        if self.training:
            valid_output = output[~invalid_mask]
            gt_occ[gt_occ > 0] = 1
            valid_gt = gt_occ[~invalid_mask]
            bce_loss = self.bce(valid_output, valid_gt.to(torch.float32))
            if self.use_soft_iou:
                soft_iou_loss = self.soft_iou(valid_output, valid_gt)
                return 0.5 * bce_loss + 0.5 * soft_iou_loss
            else:
                return bce_loss
        else:
            return output

