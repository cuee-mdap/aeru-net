#!/bin/bash
# Train AERU-Net on UCMerced, scale x3.
# Usage:  cd codes && bash train_ucm_x3.sh
# Logs/checkpoints go to ../experiment/AERUNet_UCMerced_x3/

cd "$(dirname "$0")"
PY=${PYTHON:-python}

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup $PY demo_train.py \
    --model areunet \
    --dataset UCMerced \
    --scale 3 \
    --patch_size 144 \
    --batch_size 8 \
    --n_threads 4 \
    --save AERUNet_UCMerced_x3 \
    > train_ucm_x3.out 2>&1 &

echo "Started (PID $!). Follow progress with:  tail -f $(pwd)/train_ucm_x3.out"
