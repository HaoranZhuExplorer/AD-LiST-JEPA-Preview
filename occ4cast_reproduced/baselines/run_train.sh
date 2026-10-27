#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python train.py -d lyft \
    -r /Occ4D-Lyft \
    -m conv3d \
    --p_pre 5 \
    --p_post 5 \
    --batch-size 3 \
    --num-workers 8 \
    --num-epoch 15 \
    --amp
