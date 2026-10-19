#!/bin/bash
# Test the released AERU-Net checkpoint (../pretrained/aerunet_ucmerced_x3.pt) on the UCMerced x3 test set (1050 images).
# Prints PSNR/SSIM per image and the averages, and saves all SR images to
#   ../experiment/AERUNet_UCMerced_x3/results/   (same folder as the training files;
#   test output is appended to log.txt there, model files are not touched)
# Usage:  bash test_ucm_x3.sh

cd "$(dirname "$0")"

${PYTHON:-python} demo_train.py \
    --model areunet \
    --dataset UCMerced \
    --scale 3 \
    --patch_size 144 \
    --test_only \
    --save_results \
    --pre_train ../pretrained/aerunet_ucmerced_x3.pt \
    --save AERUNet_UCMerced_x3 \
    "$@"
