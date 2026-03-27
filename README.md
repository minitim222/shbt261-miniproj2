# SHBT261 Mini Project 2

Clean training-focused repository for semantic segmentation on Pascal VOC 2007.

## Models

- U-Net (from scratch)
- DeepLabV3+ (ImageNet-pretrained backbone)
- SAM2 (zero-shot baseline)

## Repository Contents

- `main.ipynb`: Main end-to-end notebook for data loading, model training, validation, and comparison.

## Quick Start

1. Open `main.ipynb`.
2. Run cells top-to-bottom.
3. Ensure dataset paths are set correctly in the notebook.

## Core Results

| Model | Pixel Accuracy | mIoU | Dice |
|---|---:|---:|---:|
| U-Net (scratch) | 0.7398 | 0.0470 | 0.0580 |
| DeepLabV3+ (pretrained) | 0.6966 | 0.2417 | 0.3636 |
| SAM2 (zero-shot) | 0.9063 | 0.6392 | 0.7625 |

## Notes

- Experiments were run in Google Colab on NVIDIA A100.
- Validation follows Pascal VOC mask conventions (ignore label 255).
