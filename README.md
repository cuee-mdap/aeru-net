# AERU-Net: Adaptive Edge Recovery and Attention U-Shaped Network for Remote Sensing Image Super-Resolution

Official PyTorch implementation of the paper:

> **AERU-Net: Adaptive Edge Recovery and Attention U-Shaped Network for Remote Sensing Image Super-Resolution**
> Amir Hajian and Supavadee Aramvith
> *IEEE Access*, vol. 13, pp. 59177–59197, 2025. [doi:10.1109/ACCESS.2025.3556312](https://doi.org/10.1109/ACCESS.2025.3556312)

## Overview

AERU-Net is a three-level encoder–decoder (U-shaped) network for remote sensing image super-resolution (RSISR). It combines:

- **Edge Recovery Block (ERB)**: multi-scale Laplacian edge extraction with edge-guided attention.
- **Channel Attention Module (CAM)** and **Spatial Attention Module (SAM)** inside every encoder/decoder block.
- **Adaptive Cross-scale Interaction (CSI)**: links encoder and decoder features across the three levels.
- **Laplacian Edge Preserver (LEP)**: keeps high-frequency edge detail between network levels.

The model was evaluated on UCMerced, RSSCN7, AID and WHU-RS19 at ×2, ×3 and ×4. See the paper for the full results.

The network is defined in [`codes/model/areunet.py`](codes/model/areunet.py). Use it with `--model areunet`.

## Repository structure

```
aeru-net/
├── codes/
│   ├── demo_train.py          # training / testing entry point
│   ├── demo_deploy.py         # super-resolve a folder of images
│   ├── option.py              # all command-line options
│   ├── trainer.py, utils.py   # training loop, checkpointing, PSNR/SSIM
│   ├── data/                  # UCMerced dataset loader
│   ├── model/areunet.py       # AERU-Net
│   ├── loss/                  # L1 (default), MSE, VGG, GAN losses
│   ├── calculate_metric.py    # PSNR / SSIM / SAM / QI / SCC on saved SR images
│   ├── metric_scripts/        # additional metric scripts (Python / MATLAB)
│   └── train_ucm_x*.sh, test_ucm_x*.sh
└── pretrained/
    └── aerunet_ucmerced_x3.pt # AERU-Net weights, UCMerced ×3
```

## Requirements

- Python 3.8+
- PyTorch with CUDA (training was done on an NVIDIA RTX 4090)

```bash
pip install -r requirements.txt
```

`sewar` is only needed for `calculate_metric.py`, and `thop` only for the FLOPs benchmark in `model/areunet.py`.

## Dataset preparation

LR images are generated from the HR images by bicubic downsampling. By default the UCMerced loader reads `.tif` images from `codes/datasets/UCMerced-dataset`, laid out as follows:

```
codes/datasets/UCMerced-dataset/
├── train/  HR_x3/  LR_x3/  HR_x4/  LR_x4/ ...
├── val/    HR_x3/  LR_x3/  ...
└── test/   HR_x3/  LR_x3/  ...
```

An LR image must have the same file name as its HR image. UCMerced is split 50/50 into train and test (1,050 images each), and 20% of the training images are used for validation. To keep the data somewhere else, pass `--dir_data /path/to/UCMerced-dataset`.

## Training

All commands are run from the `codes/` directory:

```bash
cd codes
bash train_ucm_x3.sh                 # ×3 (runs in the background, log in train_ucm_x3.out)
bash train_ucm_x4.sh                 # ×4
bash train_ucm_x4.sh --resume 1      # continue the ×4 run from its last epoch
```

Or call the entry point directly:

```bash
python demo_train.py --model areunet --dataset UCMerced --scale 4 --patch_size 192 \
    --batch_size 16 --save AERUNet_UCMerced_x4
```

`--patch_size` is the HR patch size, so the LR patch is `patch_size / scale`. Training uses the L1 loss and Adam (β1 = 0.9, β2 = 0.999, ε = 1e-8), with a cosine-annealed learning rate that starts at 8e-4 and decays to 1e-7. Run `python demo_train.py --help` for all options. Logs, plots and checkpoints are written to `experiment/<save>/`.

The scripts set the interpreter to `python`. To use a different one, set `PYTHON`, for example `PYTHON=~/miniconda3/envs/aerunet/bin/python bash train_ucm_x3.sh`.

## Testing

```bash
cd codes
bash test_ucm_x3.sh    # evaluates pretrained/aerunet_ucmerced_x3.pt on the UCMerced ×3 test set
```

The script prints PSNR and SSIM for each image and the averages over the test set, all computed on RGB with a border of `scale` pixels cropped. SR images are saved to `experiment/AERUNet_UCMerced_x3/results/`. `test_ucm_x4.sh` does the same for ×4 using a checkpoint trained with `train_ucm_x4.sh`.

To compute PSNR, SSIM, SAM, QI and SCC on the saved SR images, edit the folders at the top of `calculate_metric.py` and run:

```bash
python calculate_metric.py
```

## Super-resolving your own images

```bash
cd codes
python demo_deploy.py --model areunet --scale 3 --patch_size 144 \
    --pre_train ../pretrained/aerunet_ucmerced_x3.pt \
    --dir_data /path/to/lr_pngs --dir_out ../experiment/results/my_sr
```

The model runs on LR tiles of size `patch_size / scale`, so each input image must be at least that size.

## Citation

If you use this code, please cite:

```bibtex
@article{hajian2025aerunet,
  author  = {Hajian, Amir and Aramvith, Supavadee},
  title   = {{AERU-Net}: Adaptive Edge Recovery and Attention U-Shaped Network for Remote Sensing Image Super-Resolution},
  journal = {IEEE Access},
  volume  = {13},
  pages   = {59177--59197},
  year    = {2025},
  doi     = {10.1109/ACCESS.2025.3556312}
}
```

## Acknowledgements

This code builds on [HAUNet](https://github.com/likakakaka/HAUNet_RSISR) and uses components from [Omni-SR](https://github.com/Francis0625/Omni-SR) (`codes/model/ops/`) and [BasicSR](https://github.com/XPixelGroup/BasicSR) (metric scripts). We thank the authors for releasing their code.

This research was supported by the Ratchadapisek Somphot Fund for Postdoctoral Fellowship, Chulalongkorn University, and in part by the Thailand Science Research and Innovation Fund, Chulalongkorn University (Grant IND_FF_68_280_2100_039).
