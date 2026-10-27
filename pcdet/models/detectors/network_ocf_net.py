from .detector3d_template_ad_list_jepa import Detector3DTemplate_AD_LiST_JEPA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import torch
from tools.visual_utils import evaluate_features_utils
import torchmetrics

class Network_OCF(Detector3DTemplate_AD_LiST_JEPA):
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
            future_ocf, future_ocf_invalid, prediction_ocf = self.backbone_3d.get_voxel_feature()
            
            prediction_ocf = prediction_ocf.squeeze(1)  # [N*num_frames_future, 20, 752, 752]
            future_ocf = future_ocf.reshape(-1, 20, 752, 752)
            future_ocf_invalid = future_ocf_invalid.reshape(-1, 20, 752, 752)
            future_ocf_invalid = future_ocf_invalid.to(torch.bool)

            future_ocf[future_ocf > 0] = 1
            future_ocf = future_ocf.to(torch.long)
            output = prediction_ocf[~future_ocf_invalid]
            label = future_ocf[~future_ocf_invalid]

            with torch.no_grad():
                precision = torchmetrics.functional.classification.binary_precision(output, label)
                recall = torchmetrics.functional.classification.binary_recall(output, label)
                f1 = 2 * precision * recall / (precision + recall)
                iou = torchmetrics.functional.classification.binary_jaccard_index(output, label)
                ap = torchmetrics.functional.classification.binary_average_precision(output, label)

            return loss, precision, recall, f1, iou, ap

                


          
    def get_training_loss(self):
        disp_dict = {} 

        loss, tb_dict = self.backbone_3d.get_loss()
        tb_dict = {
            'loss_pretrain': loss.item(),
            **tb_dict
        }

        return loss, tb_dict, disp_dict