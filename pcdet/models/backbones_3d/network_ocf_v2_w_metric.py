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
    
    
class Decoder(nn.Module):
    # 3 layer conv decoder that upsample per frame BEV features back to 3D voxel binary occupancy (B, 128, 2, 188, 188) -> (B, 1, 20, 752, 752)
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
        self.deconv3 = nn.Sequential(
            nn.ConvTranspose3d(
                8, 1, 3,
                padding=1,
                stride=(2, 1, 1),
                output_padding=(1, 0, 0),
                bias=False
            ),
        )

    def forward(self, x):
        x = self.deconv1(x)  # [24, 32,  5, 376, 376]
        x = self.deconv2(x)  # [24,  8, 10, 752, 752]
        x = self.deconv3(x)  # [24,  1, 20, 752, 752]
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
        
        prediction_ocf = self.forward_re_dict['prediction_ocf']  # [N, num_frames_future, 752, 752, 20]
        future_ocf = self.forward_re_dict['future_occ']          # [N, num_frames_future, 752, 752, 20]
        future_ocf_invalid = self.forward_re_dict['future_ocf_invalid']  # [N, num_frames_future, 752,
        future_ocf_invalid = future_ocf_invalid.to(torch.bool)

        future_ocf[future_ocf > 0] = 1
        future_ocf = future_ocf.to(torch.long)
        output = prediction_ocf[~future_ocf_invalid]
        label = future_ocf[~future_ocf_invalid]
        
        import torchmetrics
        precision = torchmetrics.functional.classification.binary_precision(output, label)
        recall = torchmetrics.functional.classification.binary_recall(output, label)
        f1 = 2 * precision * recall / (precision + recall)
        iou = torchmetrics.functional.classification.binary_jaccard_index(output, label)
        ap = torchmetrics.functional.classification.binary_average_precision(output, label)

        print("debug loss metrics: precision {}, recall {}, f1 {}, iou {}, ap {}".format(precision.item(), recall.item(), f1.item(), iou.item(), ap.item()))
        
        
        loss = F.binary_cross_entropy_with_logits(output, label.to(torch.float32))

        tb_dict = {
            'loss_bce': loss.item(),
        }
        print("debug loss", loss.item())

        exit()
        return loss, tb_dict

    def get_voxel_feature(self):
        pass


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
        t1, t2 = -5, 5
        mid = 5
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

        total_frames_future = total_frames - total_frames_prev + N
        #print("debug context_reorder", context_reorder.shape)
        prediction_reorder = self.predictor(context_reorder) # [n, 2, num_frames, h, w]
        #print("debug prediction_reorder", prediction_reorder.shape)
        prediction_chunks_stacked = prediction_reorder.permute(0, 2, 1, 3, 4) # [n, num_frames, 2, h, w]
        prediction = prediction_chunks_stacked.reshape(total_frames_future, -1, h, w) # [total_frames, 256, h, w]
        prediction = prediction.reshape(total_frames_future, c, d, h, w) # [total_frames, 128, 2, h, w]

        future_ocf = batch_dict['ocf']
        future_ocf_invalid = batch_dict['ocf_invalid']

        prediction_ocf = self.decoder(prediction) # [total_frames, 1, 20, 752, 752]
        # reshape to [N, num_frames_future, 1, 20, 752, 752]
        num_frames_future = num_frames - mid
        print("debug num_frames_future", num_frames_future, num_frames, mid)
        prediction_ocf = prediction_ocf.reshape(N, num_frames_future, 1, 20, 752, 752)
        prediction_ocf = prediction_ocf.squeeze(2) # [N, num_frames_future, 20, 752, 752]
        prediction_ocf = prediction_ocf.permute(0, 1, 3, 4, 2) # [N, num_frames_future, 752, 752, 20]

        self.forward_re_dict['prediction_ocf'] = prediction_ocf
        self.forward_re_dict['future_occ'] = future_ocf
        self.forward_re_dict['future_ocf_invalid'] = future_ocf_invalid

        '''
        from tools.visual_utils import evaluate_features_utils
        import os
        print("debug total frames", total_frames)

        t1, t2 = 0, 5
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
        
        print("debug ocf", future_ocf.shape)
        for t in range(t1, t2+1):
            future_ocf_t = future_ocf[0, t - t1, :, :, :].squeeze(0).cpu().numpy()  # [752, 752, 20]
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
        '''


        return batch_dict

        