#!/usr/bin/env bash

# Define the arrays of run_id and epoch values
# run_ids=("bev_ocf_ad_list_jepa_1206_sampled-interval-2" 
#         )
# run_ids=("bev_ocf_ad_list_jepa_1206_sampled-interval-2_run_2" "bev_ocf_ad_list_jepa_1206_sampled-interval-2_run_3" "bev_ocf_scratch_1206_sampled-interval-2_run_2" "bev_ocf_scratch_1206_sampled-interval-2_run_3")
run_ids=("bev_ocf_scratch_skip-connection_1211_sampled-interval-2")

epochs=("4")
#epochs=("10")

#run_ids=("second_ad_l_jepa_exp_14_2_new_epoch_30_run_3")
#epochs=("80")

# Loop through each run_id
for run_id in "${run_ids[@]}"; do
    # Loop through each epoch
    for epoch in "${epochs[@]}"; do
        echo "Testing run_id: ${run_id}, epoch: ${epoch}"
        # Execute the command with current run_id and epoch
        bash ./scripts/dist_test_multi_frame_bev_ocf.sh 1 \
            --cfg_file cfgs/waymo_models/network_bev_ocf.yaml \
            --ckpt ../output/waymo_models/network_bev_ocf/${run_id}/ckpt/checkpoint_epoch_${epoch}.pth \
            --extra_tag ${run_id}
    done
done