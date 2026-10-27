# future occpuancy prediction
# usage: waymo_dataset_multiframe_ocf.py 
# from line 217
# uncomment:
# load all points
# sample_idx_pre_list = np.clip(sample_idx + np.arange(sequence_cfg.SAMPLE_OFFSET[0]*sequence_cfg.FRAME_SKIP, (sequence_cfg.SAMPLE_OFFSET[1]+1)*sequence_cfg.FRAME_SKIP, sequence_cfg.FRAME_SKIP), 0, 0x7FFFFFFF)

# comment:
# only load previous points
#sample_idx_pre_list

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



    def forward(self, input_sp_tensor):
        """
        Args:
            batch_size: int
            vfe_features: (num_voxels, C)
            voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded voxel feature: sparse tensor (B, 128, 2, 200, 176)
        """

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

        self.conv3 = nn.Sequential(
            nn.Conv3d(256, 256, 3, padding=1, stride=1),
            nn.BatchNorm3d(256),
            nn.ReLU()
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.conv3(out)
        return out
    
    
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(
                128, 32, 3,
                padding=1,
                stride=(3, 2, 2),
                output_padding=(1, 1, 1),
                bias=False
            ),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose3d(
                32, 8, 3,
                padding=1,
                stride=(2, 2, 2),
                output_padding=(1, 1, 1),
                bias=False
            ),
            nn.BatchNorm3d(8),
            nn.ReLU()
        )

        # changed: out_channels=5, kernel_size=(5,3,3), output_padding=(2,1,1)
        self.deconv3 = nn.ConvTranspose3d(
            8, 1, kernel_size=(5, 3, 3),
            padding=(1, 1, 1),
            stride=(4, 2, 2),
            output_padding=(2, 1, 1),
            bias=False
        )

    def forward(self, x):
        x = self.deconv1(x)   # [11, 32, 5, 376, 376]
        x = self.deconv2(x)   # [11,  8,10, 752, 752]
        x = self.deconv3(x)   # [11,  1,41,1504,1504]
        return x


class Network_OCF(nn.Module):
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
        self.decoder = Decoder()

        # dict to save results, used for visualization or debugging
        self.forward_re_dict = {}

        down_factor = 8
        self.down_factor = down_factor
        self.grid = 1
        self.bev_x_shape = round((self.point_cloud_range[4] - self.point_cloud_range[1])/self.voxel_size[1]/self.down_factor)
        self.bev_y_shape = round((self.point_cloud_range[3] - self.point_cloud_range[0])/self.voxel_size[0]/self.down_factor)
        
        
    def get_loss(self, tb_dict=None):
        
        tb_dict = {} if tb_dict is None else tb_dict
        pred = self.forward_re_dict['pred']
        target = self.forward_re_dict['target']
        loss = F.binary_cross_entropy_with_logits(pred, target)
        print('OCF loss:', loss.item())
        tb_dict = {
            'loss': loss.item()
        }

        return loss, tb_dict

    def get_voxel_feature(self):
        return self.forward_re_dict['pred_bev_occ'], self.forward_re_dict['future_bev_occ']


    def forward(self, batch_dict):
        # unpack inputs
        voxel_features = batch_dict['voxel_features']
        coors = batch_dict['voxel_coords']
        total_frames_pre = batch_dict['batch_size_pre']  # = N_samples * num_frames
        num_frames_pre = batch_dict['num_frames_pre']             # number of offsets per sample
        N = total_frames_pre // num_frames_pre                   # original batch size

        total_frames_post = batch_dict['batch_size_post']  # = N_samples * num_frames
        num_frames_post = batch_dict['num_frames_post']             # number of offsets per sample
        N_post = total_frames_post // num_frames_post                   # original batch size

        print("debug", voxel_features.shape, coors.shape, total_frames_pre, num_frames_pre, N)
        from tools.visual_utils import evaluate_features_utils
        import os
        print("debug total frames", total_frames_pre)
        t1, t2 = 0, 10
        t_curr = 5
        final_output_dir = "<your_path>/AD-LiST-JEPA/output/waymo_models/debug"
        if not os.path.exists(final_output_dir):
            os.makedirs(final_output_dir)
        else:
            # clear the all files in the folder before ploting new visualization
            filelist = os.listdir(final_output_dir)
            for file in filelist:
                os.remove(os.path.join(final_output_dir, file))
        for t in range(t1, t2+1):
            coors_t =  coors[coors[:, 0]==t]
            print("debug t", t, torch.unique(coors_t[:,0]), coors_t.shape)
            evaluate_features_utils.get_bev_map_waymo_2(coors_t[:, 1:].cpu().numpy(), final_output_dir+"/bev_{}_lidar_data.png".format(t))


        voxel_features_pre = voxel_features[coors[:, 0]<=t_curr]
        coors_pre = coors[coors[:, 0]<=t_curr]
        voxel_features_post = voxel_features[coors[:, 0]>=t_curr]
        coors_post = coors[coors[:, 0]>=t_curr]
        coors_post[:,0] = coors_post[:,0] - t_curr  # shift the time index to start from 0

        print("debug pre/post", voxel_features_pre.shape, coors_pre.shape, voxel_features_post.shape, coors_post.shape)

        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features_pre,
            indices=coors_pre.int(),
            spatial_shape=self.sparse_shape,
            batch_size=total_frames_pre
        )

        nums = voxel_features_post.shape[0]
        voxel_features_all_one = torch.ones(nums,1).to(voxel_features_post.device)
        input_sp_tensor_ones = spconv.SparseConvTensor(
            features=voxel_features_all_one,
            indices=coors_post.int(),
            spatial_shape=self.sparse_shape,
            batch_size=total_frames_post
        )

        print("debug input_sp_tensor_ones", input_sp_tensor.dense().shape, input_sp_tensor_ones.dense().shape)

        future_ocf = batch_dict['ocf']
        print("debug ocf", future_ocf.shape)
        for t in range(t_curr, t2+1):
            future_ocf_t = future_ocf[0, t - t_curr, :, :, :].squeeze(0).cpu().numpy()  # [752, 752, 20]
            # visualize the sum over z axis
            future_ocf_bev_t = np.sum(future_ocf_t, axis=-1)  # [752, 752]
            future_ocf_bev_t[future_ocf_bev_t > 0] = 1 # [752, 752]
            print("debug", t, future_ocf_bev_t.shape)
            
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 10))
            plt.imshow(future_ocf_bev_t, aspect='auto', cmap='gray')
            plt.axis('off')  
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
            plt.savefig(final_output_dir+"/bev_{}_future_ocf_bev.png".format(t), bbox_inches='tight', pad_inches=0)
        
       
        for t in range(t_curr, t2+1):
            future_pcd_t = input_sp_tensor_ones.dense()[t - t_curr, 0, :, :, :].squeeze(1).cpu().numpy() 
            print("debug future_pcd_t", np.sum(future_pcd_t))
            print("debug future_pcd_t", future_pcd_t.shape)
            future_pcd_bev_t = np.sum(future_pcd_t, axis=0) 
            print("debug sum over z axis", future_pcd_bev_t.shape)
            future_pcd_bev_t[future_pcd_bev_t > 0] = 1
            print("debug", t, future_pcd_bev_t.shape)
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 10))
            plt.imshow(future_pcd_bev_t, aspect='auto', cmap='gray')
            plt.axis('off')  
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
            plt.savefig(final_output_dir+"/bev_{}_future_pcd_bev.png".format(t), bbox_inches='tight', pad_inches=0)


        context = self.encoder(
            input_sp_tensor
        )  # (B, 128, 2, 188, 188)

        bs, c, d, h, w = context.shape
        context = context.reshape(bs, -1, h, w)

        context_chunks = torch.chunk(context, N, dim=0)
        context_chunks_stacked = torch.stack(context_chunks, dim=0) # [n, num_frames, d, h, w]
        context = context_chunks_stacked.permute(0, 2, 1, 3, 4) # [n, d, num_frames, h, w]

        print("debug context", context.shape)

        pred_bev = self.predictor(context)
        pred_bev = pred_bev.reshape(bs, 128, 2, h, w)

        print("debug pred_bev", pred_bev.shape)

        output = self.decoder(pred_bev)  # (B, 1, 20, 752, 752)
        self.forward_re_dict['target'] = input_sp_tensor_ones.dense()
        self.forward_re_dict['pred'] = output

        print("debug output", output.shape, input_sp_tensor_ones.dense().shape, future_ocf.shape)
        exit()
        return batch_dict

        