#!/bin/bash
# Test the best AERU-Net checkpoint on the UCMerced x4 test set (1050 images).
# Prints PSNR/SSIM per image and the averages, and saves all SR images to
#   ../experiment/AERUNet_UCMerced_x4/results/   (same folder as the training files;
#   test output is appended to log.txt there, model files are not touched)
# Note: no x4 checkpoint is shipped in pretrained/; train one first with train_ucm_x4.sh.
# Usage:  bash test_ucm_x4.sh

cd "$(dirname "$0")"

${PYTHON:-python} demo_train.py \
    --model areunet \
    --dataset UCMerced \
    --scale 4 \
    --patch_size 192 \
    --test_only \
    --save_results \
    --pre_train ../experiment/AERUNet_UCMerced_x4/model/model_best.pt \
    --save AERUNet_UCMerced_x4 \
    "$@"
