# SHBT261 Mini Project 2

Semantic segmentation comparison on Pascal VOC 2007 using:
- U-Net (trained from scratch)
- DeepLabV3+ (ImageNet-pretrained backbone)
- SAM2 (zero-shot)

## Repository Contents

- `main.ipynb`: Main notebook with full experimental pipeline (training, evaluation, plots, and report bundle export).
- `project2_report.pdf`: Final compiled report.
- `project2_report.tex`: LaTeX source for the final report.
- `generate_project2_report.py`: Markdown report draft generator from `report_assets`.
- `generate_project2_report_tex.py`: LaTeX report generator from `report_assets`.
- `report_assets/`: Figures, metrics, and report JSON/CSV artifacts used in the report.

## Core Results

| Model | Pixel Accuracy | mIoU | Dice |
|---|---:|---:|---:|
| U-Net (scratch) | 0.7398 | 0.0470 | 0.0580 |
| DeepLabV3+ (pretrained) | 0.6966 | 0.2417 | 0.3636 |
| SAM2 (zero-shot) | 0.9063 | 0.6392 | 0.7625 |

## Reproduce Report PDF

From the repository root:

```bash
pdflatex -interaction=nonstopmode project2_report.tex
pdflatex -interaction=nonstopmode project2_report.tex
```

## Notes

- Experiments were run in Google Colab on NVIDIA A100.
- Validation protocol follows Pascal VOC mask conventions with ignore label 255.
