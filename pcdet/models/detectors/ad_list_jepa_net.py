from .detector3d_template_ad_list_jepa import Detector3DTemplate_AD_LiST_JEPA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import torch
from tools.visual_utils import evaluate_features_utils

class AD_LiST_JEPA(Detector3DTemplate_AD_LiST_JEPA):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict, plot=False, final_output_dir=None):
        input_batch = batch_dict
        
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            loss, tb_dict, disp_dict = self.get_training_loss()

            if plot:
                print("debug frame_id", input_batch['frame_id'])
                print("debug loss", loss)
                print("debug tb_dict", tb_dict)
                
                '''
                gt_boxes = input_batch['gt_boxes'] # [B, M, 8]
                num_classes = len(torch.unique(gt_boxes[:,:,7]))
                empty_token = self.backbone_3d.empty_token
                print("debug", input_batch['voxels_multi_frame_prev_curr'].shape, input_batch['voxels_multi_frame_future'].shape,)
                evaluate_features_utils.get_bev_map_waymo(input_batch['voxels_multi_frame_future'][:,0,0:4], final_output_dir+"/bev_{}_lidar_data.png".format(str(batch_dict['frame_id'])))
                
                prediction_future = self.backbone_3d.forward_re_dict['prediction_future'][0]
                target_future = self.backbone_3d.forward_re_dict['target_future'][0]
                print("debug prediction_future", prediction_future.shape, target_future.shape)
                loss_jepa = evaluate_features_utils.viz_bev_cosine_similarity(prediction_future, target_future, empty_token, final_output_dir+"/bev_{}_jepa_loss.png".format(str(batch_dict['frame_id'])))
                evaluate_features_utils.viz_histogram(loss_jepa.reshape(-1), final_output_dir+"/bev_{}_jepa_loss_histogram.png".format(str(batch_dict['frame_id'])))
                '''

            return loss
        

    def get_training_loss(self):
        disp_dict = {} 

        loss, tb_dict = self.backbone_3d.get_loss()
        tb_dict = {
            'loss_pretrain': loss.item(),
            **tb_dict
        }

        return loss, tb_dict, disp_dict