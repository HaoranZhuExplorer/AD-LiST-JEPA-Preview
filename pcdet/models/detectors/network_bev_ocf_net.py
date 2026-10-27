from .detector3d_template_ad_list_jepa import Detector3DTemplate_AD_LiST_JEPA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import torch
from tools.visual_utils import evaluate_features_utils

class Network_BEV_OCF(Detector3DTemplate_AD_LiST_JEPA):
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
            pred_bev_occ, future_bev_occ = self.backbone_3d.get_voxel_feature()

            
            with torch.no_grad():
                # predicted probabilities for foreground (class 1)
                pred_probs = torch.sigmoid(pred_bev_occ[:, 1])

                # convert to predicted labels (0 or 1)
                pred_labels = (pred_probs >= 0.5).long()

                # ground truth labels (0 or 1)
                true_labels = future_bev_occ[:, 1].long()

                # Boolean masks
                fg_mask = (true_labels == 1)
                bg_mask = (true_labels == 0)

                # Foreground accuracy
                if fg_mask.sum() > 0:
                    fg_correct = (pred_labels[fg_mask] == 1).sum().item()
                    fg_total = fg_mask.sum().item()
                    fg_acc = fg_correct / fg_total
                else:
                    fg_acc = float('nan')

                # Background accuracy
                if bg_mask.sum() > 0:
                    bg_correct = (pred_labels[bg_mask] == 0).sum().item()
                    bg_total = bg_mask.sum().item()
                    bg_acc = bg_correct / bg_total
                else:
                    bg_acc = float('nan')
            
            return loss, fg_acc, bg_acc

                


          
    def get_training_loss(self):
        disp_dict = {} 

        loss, tb_dict = self.backbone_3d.get_loss()
        tb_dict = {
            'loss_pretrain': loss.item(),
            **tb_dict
        }

        return loss, tb_dict, disp_dict