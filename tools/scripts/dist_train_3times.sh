#!/bin/bash


export PYTHONPATH="${PYTHONPATH}:<your_path>/AD-LiST-JEPA/tools"

cd "<your_path>/AD-LiST-JEPA/tools"

# Script to run training multiple times with different run IDs

# Define the base command
base_command="bash ./scripts/dist_train.sh 4  --cfg_file cfgs/waymo_models/centerpoint_without_resnet.yaml  --pretrained_model <your_path>/AD-LiST-JEPA/output/waymo_models/ad_list_jepa_waymo/ad_list_jepa_v4_0507_4gpu_3dConv_effective_bs16_each_gpu_grad_accumulation/ckpt/checkpoint_epoch_30.pth --extra_tag centerpoint_wo_resnet_ad_list_jepa_v4_0507_4gpu_3dConv_effective_bs16_each_gpu_grad_accumulation_run_"

# Loop to run the command three times with run_id from 1 to 3
for run_id in {1..3}
do
    echo "Running training with run_id: $run_id"
    $base_command$run_id
done
