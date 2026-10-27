# AD-LiST-JEPA

AD-LiST-JEPA is a self-supervised world model that learns spatiotemporal representations from LiDAR sequences using a joint-embedding predictive architecture (JEPA). The learned representations are evaluated through downstream occupancy completion and forecasting.

**Paper:** [Self-Supervised JEPA-based World Models for LiDAR Occupancy Completion and Forecasting](https://arxiv.org/abs/2602.12540)

Haoran Zhu and Anna Choromanska

> **Preview release:** This repository contains an early code release accompanying the arXiv preprint. The implementation and documentation are under development, and a more complete release is planned. Some configuration settings require adaptation to your environment.

## Installation

Follow the environment setup instructions in the related [AD-L-JEPA repository](https://github.com/HaoranZhuExplorer/adljepa).

## Data Preparation

Review the dataset paths and split settings in these configurations to match your local data:

- [Self-supervised learning configuration](tools/cfgs/dataset_configs/waymo_dataset_multiframe_ssl.yaml)
- [Occupancy completion and forecasting configuration](tools/cfgs/dataset_configs/waymo_dataset_multiframe_ocf.yaml)

### Self-Supervised Representation Learning

```bash
bash scripts/dist_train_multi_frame_ssl.sh 4 \
    --cfg_file cfgs/waymo_models/ad_list_jepa_waymo.yaml
```

### Occupancy Completion and Forecasting

```bash
bash scripts/dist_train_multi_frame_ocf.sh 4 \
    --cfg_file cfgs/waymo_models/network_ocf.yaml
```

To initialize training from a pretrained checkpoint, append `--pretrained_model /path/to/pretrained_checkpoint.pth` to the command above.

## Evaluation

Evaluate an occupancy completion and forecasting checkpoint from the `tools/` directory:

```bash
bash scripts/dist_test_multi_frame_ocf.sh 1 \
    --cfg_file cfgs/waymo_models/network_ocf.yaml \
    --ckpt /path/to/ocf_checkpoint.pth
```

Replace `/path/to/ocf_checkpoint.pth` with the checkpoint you want to evaluate. Training outputs are saved under `output/` in the repository root.
