from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


def fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def latest_csv(reports_dir: Path, pattern: str) -> Path | None:
    candidates = sorted(reports_dir.glob(pattern))
    return candidates[-1] if candidates else None


def load_bundle(bundle_path: Path) -> dict:
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle JSON not found: {bundle_path}")
    with bundle_path.open("r") as file:
        return json.load(file)


def load_top_classes(per_class_csv: Path | None, top_k: int = 5) -> list[tuple[str, float]]:
    if per_class_csv is None or not per_class_csv.exists():
        return []

    rows: list[tuple[str, float]] = []
    with per_class_csv.open("r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                class_name = row.get("Class", "")
                score = float(row.get("DeepLabV3+", "nan"))
                if class_name:
                    rows.append((class_name, score))
            except ValueError:
                continue

    rows.sort(key=lambda item: item[1], reverse=True)
    return rows[:top_k]


def build_report(bundle: dict, top_classes: list[tuple[str, float]]) -> str:
    dataset = bundle["dataset"]
    results = bundle["results"]
    ablation = bundle["ablation"]

    unet = results["unet_scratch_30ep"]
    deeplab = results["deeplabv3_pretrained_30ep"]
    sam2 = results["sam2_zeroshot"]

    gain_dl_vs_unet = deeplab["miou"] / unet["miou"]
    gain_sam2_vs_dl = sam2["miou"] / deeplab["miou"]
    gain_sam2_vs_unet = sam2["miou"] / unet["miou"]

    abl_a = ablation["A_baseline_aug_ce_pretrain_30ep"]
    abl_b = ablation["B_no_augmentation"]
    abl_c = ablation["C_dice_loss"]
    abl_d = ablation["D_scratch_backbone"]

    aug_delta_pp = (abl_a - abl_b) * 100
    dice_delta_pp = (abl_c - abl_a) * 100

    report_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    top_classes_text = ""
    if top_classes:
        lines = [f"- {class_name}: {score:.4f}" for class_name, score in top_classes]
        top_classes_text = "\n".join(lines)
    else:
        top_classes_text = "- Per-class CSV unavailable; see `figures/per_class_iou.png`."

    return f"""# Comparative Analysis of Supervised Segmentation vs Zero-Shot Foundation Models on Pascal VOC 2007

Author: Tim Cao  
Generated: {report_time}  
Project Folder: `mini-project-2`

## Abstract
This report presents a comparative semantic segmentation study on the Pascal VOC 2007 segmentation subset, using three paradigms: (1) U-Net trained from scratch, (2) DeepLabV3+ with an ImageNet-pretrained backbone, and (3) SAM2 zero-shot inference without task-specific training. Experiments were performed on a constrained labeled dataset of {dataset['train_images']} training and {dataset['val_images']} validation images over {dataset['num_classes']} classes (ignore index {dataset['ignore_index']}). Results show a clear hierarchy of performance: SAM2 achieved the strongest mIoU ({fmt_pct(sam2['miou'])}), DeepLabV3+ was best among trainable models ({fmt_pct(deeplab['miou'])}), and U-Net from scratch underperformed ({fmt_pct(unet['miou'])}) due to severe data scarcity. Ablation analysis indicates that pretrained initialization is the dominant contributor, while augmentation provided minimal marginal gain in this low-data setting.

## 1. Introduction
Semantic segmentation has evolved from architecture-focused supervised learning toward large-scale foundation models with strong zero-shot transfer. This study quantifies that transition in a controlled educational setting by comparing classical fully supervised pipelines against a modern foundation model under the same evaluation protocol.

## 2. Methods
### 2.1 Dataset and Splits
- Dataset: {dataset['name']}
- Training set: {dataset['train_images']} images
- Validation set: {dataset['val_images']} images
- Classes: {dataset['num_classes']} (with ignore label {dataset['ignore_index']})

### 2.2 Models
- **U-Net (scratch)**: supervised encoder-decoder with random initialization.
- **DeepLabV3+ (pretrained)**: supervised model with pretrained ResNet backbone.
- **SAM2 (zero-shot)**: foundation model inference without fine-tuning.

### 2.3 Metrics
- Pixel Accuracy (PA)
- Mean Intersection-over-Union (mIoU)
- Dice score

## 3. Main Results
| Model | Pixel Accuracy | mIoU | Dice |
|---|---:|---:|---:|
| U-Net (scratch, 30 ep) | {unet['pixel_accuracy']:.4f} | {unet['miou']:.4f} | {unet['dice']:.4f} |
| DeepLabV3+ (pretrained, 30 ep) | {deeplab['pixel_accuracy']:.4f} | {deeplab['miou']:.4f} | {deeplab['dice']:.4f} |
| SAM2 (zero-shot) | {sam2['pixel_accuracy']:.4f} | {sam2['miou']:.4f} | {sam2['dice']:.4f} |

Key relative gains (mIoU):
- DeepLabV3+ vs U-Net: {gain_dl_vs_unet:.2f}×
- SAM2 vs DeepLabV3+: {gain_sam2_vs_dl:.2f}×
- SAM2 vs U-Net: {gain_sam2_vs_unet:.2f}×

## 4. Ablation Study (DeepLabV3+)
| Variant | mIoU |
|---|---:|
| A: Baseline (augmentation + CE + pretrained) | {abl_a:.4f} |
| B: No augmentation | {abl_b:.4f} |
| C: Dice loss | {abl_c:.4f} |
| D: Scratch backbone | {abl_d:.4f} |

Interpretation:
- Removing augmentation changes mIoU by {aug_delta_pp:+.2f} percentage points.
- Switching CE → Dice changes mIoU by {dice_delta_pp:+.2f} percentage points.
- Removing pretrained initialization causes a large collapse, confirming transfer initialization as the primary driver.

## 5. Per-Class Behavior
Top classes by DeepLabV3+ IoU:
{top_classes_text}

Detailed visual evidence is available in:
- `figures/per_class_iou.png`
- `figures/confusion_matrix.png`
- `figures/model_comparison.png`
- `figures/training_curves.png`
- `figures/predictions_grid.png`
- `figures/ablation_study.png`

## 6. Conclusion
On small labeled segmentation datasets, randomly initialized supervised models are strongly data-limited. Pretraining significantly improves supervised performance, but zero-shot foundation models can still dominate when they transfer rich priors learned from large-scale pretraining corpora. For this project, the best practical path is either pretrained supervised fine-tuning or direct foundation-model inference, with architecture changes being secondary to initialization and data scale.

---

### Reproducibility Artifacts
- Bundle JSON: `report_assets/reports/final_report_bundle_latest.json`
- Per-class IoU CSV: `report_assets/reports/per_class_iou_*.csv`
- Model summary CSV: `report_assets/reports/model_summary_*.csv`
- Ablation CSV: `report_assets/reports/ablation_summary_*.csv`
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Project 2 report draft from exported assets.")
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=Path("report_assets"),
        help="Directory containing extracted report assets.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("project2_report_draft.md"),
        help="Output markdown report path.",
    )
    args = parser.parse_args()

    reports_dir = args.assets_dir / "reports"
    bundle_path = reports_dir / "final_report_bundle_latest.json"
    per_class_path = latest_csv(reports_dir, "per_class_iou_*.csv")

    bundle = load_bundle(bundle_path)
    top_classes = load_top_classes(per_class_path)
    report_text = build_report(bundle, top_classes)

    args.out.write_text(report_text)
    print(f"Wrote: {args.out}")
    if per_class_path:
        print(f"Used per-class CSV: {per_class_path}")
    else:
        print("Per-class CSV not found; report generated without top-class table.")


if __name__ == "__main__":
    main()
