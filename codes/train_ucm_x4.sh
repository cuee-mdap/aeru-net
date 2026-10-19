#!/bin/bash
# Train AERU-Net on UCMerced, scale x4 (runs in this terminal; closing it stops training).
# Usage:  bash train_ucm_x4.sh                -> start a new training
#         bash train_ucm_x4.sh --resume 1     -> continue from the last saved epoch
# Logs/checkpoints go to ../experiment/AERUNet_UCMerced_x4/
# patch_size 192 -> LR patch 48x48, as in the paper

cd "$(dirname "$0")"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ${PYTHON:-python} demo_train.py \
    --model areunet \
    --dataset UCMerced \
    --scale 4 \
    --patch_size 192 \
    --batch_size 16 \
    --n_threads 4 \
    --save AERUNet_UCMerced_x4 \
    "$@"
