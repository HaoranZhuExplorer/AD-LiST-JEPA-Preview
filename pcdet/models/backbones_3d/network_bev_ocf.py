from functools import partial
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from ...utils.spconv_utils import replace_feature, spconv
from ...utils import common_utils
from .spconv_backbone import post_act_block


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
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
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
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        if self.model_cfg.get('RETURN_ENCODED_TENSOR', True):
            last_pad = self.model_cfg.get('last_pad', 0)

            self.conv_out = spconv.SparseSequential(
                # [200, 175, 5] -> [200, 176, 2]
                spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                    bias=False, indice_key='spconv_down2'),
                norm_fn(128),
                nn.ReLU(),
            )
        else:
            self.conv_out = None



    def forward(self, batch_size, voxel_features_partial=None, voxel_coords_partial=None):
        """
        Args:
            batch_size: int
            vfe_features: (num_voxels, C)
            voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded voxel feature: sparse tensor (B, 128, 2, 200, 176)
        """
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

        # Only this layer reduces the num_frames dimension by 1
        self.conv3 = nn.Sequential(
            nn.Conv3d(
                256, 256,
                kernel_size=(2,1,1),
                padding=(0,0,0),
                stride=1
            ),
            nn.BatchNorm3d(256),
            nn.ReLU()
        )

        # add final binary classifier layer that map feature of 256 -> 1
        self.conv4 = nn.Conv3d(
            256,  # in_channels from conv3
            1,    # 1 channel output for binary classification
            kernel_size=1,  # 1x1x1 conv, no change in T/H/W
            stride=1,
            padding=0
        )


    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        feats = out
        class_logits = self.conv4(feats)
        return class_logits


class Network_BEV_OCF(nn.Module):
    """
    pre-trained model
    """

    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
    
        self.num_point_features = 16 # for building the model

        self.encoder = Encoder(model_cfg, input_channels, grid_size, voxel_size, point_cloud_range)
        self.predictor = Predictor()

        # dict to save results, used for visualization or debugging
        self.forward_re_dict = {}

        down_factor = 8
        self.down_factor = down_factor
        self.grid = 1
        self.bev_x_shape = round((self.point_cloud_range[4] - self.point_cloud_range[1])/self.voxel_size[1]/self.down_factor)
        self.bev_y_shape = round((self.point_cloud_range[3] - self.point_cloud_range[0])/self.voxel_size[0]/self.down_factor)
        
        
    def get_loss(self, tb_dict=None):
        
        pred_bev_occ =  self.forward_re_dict['pred_bev_occ'] # [n, 2, 188, 188]
        future_bev_occ = self.forward_re_dict['future_bev_occ'] # [n, 2, 188, 188]

        # binary cross entropy loss to make pred_bev_occ match ground truth future_bev_occ
        import torch.nn.functional as F

        loss = F.binary_cross_entropy_with_logits(
            pred_bev_occ,     # raw logits, NOT passed through sigmoid
            future_bev_occ.float()
        )

        # with torch.no_grad():
        #     # predicted probabilities for foreground (class 1)
        #     pred_probs = torch.sigmoid(pred_bev_occ[:, 1])

        #     # convert to predicted labels (0 or 1)
        #     pred_labels = (pred_probs >= 0.5).long()

        #     # ground truth labels (0 or 1)
        #     true_labels = future_bev_occ[:, 1].long()

        #     # Boolean masks
        #     fg_mask = (true_labels == 1)
        #     bg_mask = (true_labels == 0)

        #     # Foreground accuracy
        #     if fg_mask.sum() > 0:
        #         fg_correct = (pred_labels[fg_mask] == 1).sum().item()
        #         fg_total = fg_mask.sum().item()
        #         fg_acc = fg_correct / fg_total
        #     else:
        #         fg_acc = float('nan')

        #     # Background accuracy
        #     if bg_mask.sum() > 0:
        #         bg_correct = (pred_labels[bg_mask] == 0).sum().item()
        #         bg_total = bg_mask.sum().item()
        #         bg_acc = bg_correct / bg_total
        #     else:
        #         bg_acc = float('nan')

        #     print(f"Foreground accuracy: {fg_acc:.4f}")
        #     print(f"Background accuracy: {bg_acc:.4f}")

            


        tb_dict = {
            'loss_bce': loss.item(),
        }

        return loss, tb_dict

    def get_voxel_feature(self):
        return self.forward_re_dict['pred_bev_occ'], self.forward_re_dict['future_bev_occ']


    def forward(self, batch_dict):
        # unpack inputs
        voxel_features = batch_dict['voxel_features']
        coors = batch_dict['voxel_coords']
        total_frames = batch_dict['batch_size']  # = N_samples * num_frames
        num_frames = batch_dict['num_frames']             # number of offsets per sample
        N = total_frames // num_frames                   # original batch size

        #print("debug ad_list_jepa forward total_frames", voxel_features.shape, coors.shape, torch.unique(coors[:,0]), total_frames, num_frames, N)
        #print("debug", voxel_features[0,:], coors[0,:])

        from tools.visual_utils import evaluate_features_utils
        import os
        t1, t2 = -3, 3
        mid = 3
        prev_ids = (coors[:, 0]%(t2-t1+1) <=mid)
        future_ids = (coors[:, 0]%(t2-t1+1) >mid)
        voxel_features_prev = voxel_features[prev_ids]
        coors_prev = coors[prev_ids]
        voxel_features_future = voxel_features[future_ids]
        coors_future = coors[future_ids]
        #print("debug ad_list_jepa forward prev/future", voxel_features_prev.shape, coors_prev.shape, voxel_features_future.shape, coors_future.shape)
        
        total_frames_prev = N * (mid+1)
        context = self.encoder(total_frames_prev, voxel_features_prev, coors_prev)
        bs, c, d, h, w = context.shape
        context = context.reshape(bs, -1, h, w)
        context_chunks = torch.chunk(context, N, dim=0)
        context_chunks_stacked = torch.stack(context_chunks, dim=0) # [n, num_frames, d, h, w]
        context_reorder = context_chunks_stacked.permute(0, 2, 1, 3, 4) # [n, d, num_frames, h, w]

        total_frames_future = total_frames - total_frames_prev
        #print("debug context_reorder", context_reorder.shape)
        prediction_reorder = self.predictor(context_reorder) # [n, 2, num_frames, h, w]
        #print("debug prediction_reorder", prediction_reorder.shape)
        prediction_chunks_stacked = prediction_reorder.permute(0, 2, 1, 3, 4) # [n, num_frames, 2, h, w]
        prediction = prediction_chunks_stacked.reshape(total_frames_future, 1, h, w) # [total_frames, 2, h, w]

        #print("debug", context.shape, context_reorder.shape, context_chunks_stacked.shape, prediction_reorder.shape, prediction.shape)

        # 1) down‐sample coords into BEV grid
        coor_future_ds = coors_future.detach().clone().long()  # [pts, 4]
        coor_future_ds[:, 1:] = torch.div(
            coor_future_ds[:, 1:], 
            self.down_factor * self.grid, 
            rounding_mode='floor'
        )
        coor_future_ds[:, 1] = torch.div(
            coor_future_ds[:, 1], 
            coor_future_ds[:, 1].max() * 2, 
            rounding_mode='floor'
        )

        # create dense future bev tensor
        future_timestamps = torch.unique(coor_future_ds[:, 0])
        future_timestamps, _ = torch.sort(future_timestamps)  # just to be safe / ordered
        num_future_frames = future_timestamps.shape[0]

        device = coor_future_ds.device

        bev_occ = torch.zeros(num_future_frames, 1, 188, 188,
                            device=device, dtype=torch.float32)

        for i, t in enumerate(future_timestamps):
            # points belonging to this timestamp
            mask = (coor_future_ds[:, 0] == t)

            # get BEV indices (adjust indices if your order is different)
            ys = coor_future_ds[mask, 2].clamp(0, h - 1)
            xs = coor_future_ds[mask, 3].clamp(0, w - 1)

            # mark occupied in channel 1
            bev_occ[i, 0, ys, xs] = 1.0

           
        self.forward_re_dict['pred_bev_occ'] = prediction
        self.forward_re_dict['future_bev_occ'] = bev_occ

        return batch_dict

        