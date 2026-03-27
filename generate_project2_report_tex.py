from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


def latest_csv(reports_dir: Path, pattern: str) -> Path | None:
    files = sorted(reports_dir.glob(pattern))
    return files[-1] if files else None


def load_bundle(bundle_path: Path) -> dict:
    if not bundle_path.exists():
        raise FileNotFoundError(f"Bundle JSON not found: {bundle_path}")
    with bundle_path.open("r") as file:
        return json.load(file)


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = text
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def load_top_classes(per_class_csv: Path | None, top_k: int = 8) -> list[tuple[str, float, float | None]]:
    if per_class_csv is None or not per_class_csv.exists():
        return []

    rows: list[tuple[str, float, float | None]] = []
    with per_class_csv.open("r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                class_name = row.get("Class", "").strip()
                dl = float(row.get("DeepLabV3+", "nan"))
                unet_val: float | None = None
                u = row.get("U-Net", "")
                if u not in (None, ""):
                    unet_val = float(u)
                if class_name:
                    rows.append((class_name, dl, unet_val))
            except ValueError:
                continue

    rows.sort(key=lambda item: item[1], reverse=True)
    return rows[:top_k]


def build_tex(bundle: dict, top_classes: list[tuple[str, float, float | None]]) -> str:
    dataset = bundle["dataset"]
    results = bundle["results"]
    ablation = bundle["ablation"]

    unet = results["unet_scratch_30ep"]
    deeplab = results["deeplabv3_pretrained_30ep"]
    sam2 = results["sam2_zeroshot"]

    a = ablation["A_baseline_aug_ce_pretrain_30ep"]
    b = ablation["B_no_augmentation"]
    c = ablation["C_dice_loss"]
    d = ablation["D_scratch_backbone"]

    gain_dl_unet = deeplab["miou"] / unet["miou"]
    gain_sam2_dl = sam2["miou"] / deeplab["miou"]

    aug_delta_pp = (a - b) * 100.0
    dice_delta_pp = (c - a) * 100.0

    now = datetime.now().strftime("%Y-%m-%d")

    top_rows = []
    for cls, dl_iou, unet_iou in top_classes:
        u_str = "--" if unet_iou is None else f"{unet_iou:.4f}"
        top_rows.append(
            rf"{latex_escape(cls)} & {u_str} & {dl_iou:.4f} \\" 
        )
    top_rows_tex = "\n".join(top_rows) if top_rows else r"N/A & -- & -- \\" 

    return rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{float}}
\usepackage{{hyperref}}
\usepackage{{caption}}
\usepackage{{subcaption}}
\usepackage{{siunitx}}
\usepackage{{xcolor}}
\hypersetup{{colorlinks=true,linkcolor=blue,citecolor=blue,urlcolor=blue}}

\title{{Comparative Analysis of Supervised Segmentation vs. Zero-Shot Foundation Models on Pascal VOC 2007}}
\author{{Tim Cao\\\small tim\_cao@hsph.harvard.edu}}
\date{{{now}}}

\begin{{document}}
\maketitle

\begin{{abstract}}
This report presents a comparative semantic segmentation study on the Pascal VOC 2007 segmentation subset using three paradigms: (1) U-Net trained from scratch, (2) DeepLabV3+ with a pretrained backbone, and (3) SAM2 zero-shot inference without task-specific fine-tuning. Experiments were conducted on a constrained labeled dataset ({dataset['train_images']} train / {dataset['val_images']} validation images, {dataset['num_classes']} classes). Results show a clear performance hierarchy: SAM2 achieves the strongest mIoU ({sam2['miou']:.4f}), DeepLabV3+ is best among trainable supervised models ({deeplab['miou']:.4f}), and U-Net from scratch underperforms ({unet['miou']:.4f}) due to data scarcity. Ablation confirms pretrained initialization as the dominant factor, while augmentation contributes only marginal gains in this low-data regime.
\end{{abstract}}

\section{{Introduction}}
Semantic segmentation has rapidly transitioned from purely supervised architecture-centric pipelines toward foundation-model-based transfer. This project quantifies that transition on Pascal VOC 2007 by comparing supervised baselines against a modern zero-shot model under a common evaluation framework.

\section{{Methods}}
\subsection{{Dataset and Protocol}}
\begin{{itemize}}
  \item Dataset: {latex_escape(dataset['name'])}
  \item Training split: {dataset['train_images']} images
  \item Validation split: {dataset['val_images']} images
  \item Number of classes: {dataset['num_classes']} (ignore index = {dataset['ignore_index']})
  \item Evaluation metrics: Pixel Accuracy (PA), mIoU, Dice
\end{{itemize}}

\subsection{{Models}}
\begin{{itemize}}
  \item \textbf{{U-Net (scratch)}}: supervised encoder-decoder with random initialization.
  \item \textbf{{DeepLabV3+ (pretrained)}}: supervised segmentation with pretrained ResNet backbone.
  \item \textbf{{SAM2 (zero-shot)}}: foundation model inference without task-specific training.
\end{{itemize}}

\section{{Main Results}}
\begin{{table}}[H]
\centering
\caption{{Overall validation performance.}}
\begin{{tabular}}{{lccc}}
\toprule
Model & Pixel Accuracy & mIoU & Dice \\
\midrule
U-Net (scratch, 30 ep) & {unet['pixel_accuracy']:.4f} & {unet['miou']:.4f} & {unet['dice']:.4f} \\
DeepLabV3+ (pretrained, 30 ep) & {deeplab['pixel_accuracy']:.4f} & {deeplab['miou']:.4f} & {deeplab['dice']:.4f} \\
SAM2 (zero-shot) & {sam2['pixel_accuracy']:.4f} & {sam2['miou']:.4f} & {sam2['dice']:.4f} \\
\bottomrule
\end{{tabular}}
\end{{table}}

Relative mIoU gains: DeepLabV3+ vs U-Net = {gain_dl_unet:.2f}$\times$; SAM2 vs DeepLabV3+ = {gain_sam2_dl:.2f}$\times$.

\section{{Ablation Study}}
\begin{{table}}[H]
\centering
\caption{{Ablation on DeepLabV3+ configuration choices.}}
\begin{{tabular}}{{lc}}
\toprule
Variant & mIoU \\
\midrule
A: Baseline (augmentation + CE + pretrained) & {a:.4f} \\
B: No augmentation & {b:.4f} \\
C: Dice loss & {c:.4f} \\
D: Scratch backbone & {d:.4f} \\
\bottomrule
\end{{tabular}}
\end{{table}}

\noindent
Effect sizes: removing augmentation changes mIoU by {aug_delta_pp:+.2f} percentage points; switching CE to Dice changes mIoU by {dice_delta_pp:+.2f} percentage points.

\section{{Per-Class Behavior}}
\begin{{table}}[H]
\centering
\caption{{Top classes by DeepLabV3+ IoU.}}
\begin{{tabular}}{{lcc}}
\toprule
Class & U-Net IoU & DeepLabV3+ IoU \\
\midrule
{top_rows_tex}
\bottomrule
\end{{tabular}}
\end{{table}}

\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.9\textwidth]{{report_assets/figures/model_comparison.png}}
  \caption{{Model-level comparison across PA, mIoU, and Dice.}}
\end{{figure}}

\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.9\textwidth]{{report_assets/figures/training_curves.png}}
  \caption{{Training dynamics for supervised models.}}
\end{{figure}}

\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.9\textwidth]{{report_assets/figures/per_class_iou.png}}
  \caption{{Per-class IoU comparison between U-Net and DeepLabV3+.}}
\end{{figure}}

\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.9\textwidth]{{report_assets/figures/confusion_matrix.png}}
  \caption{{Row-normalized confusion matrix for DeepLabV3+ on validation set.}}
\end{{figure}}

\begin{{figure}}[H]
  \centering
  \includegraphics[width=0.9\textwidth]{{report_assets/figures/ablation_study.png}}
  \caption{{Ablation study summary.}}
\end{{figure}}

\section{{Conclusion}}
On this small labeled segmentation dataset, random-initialization supervised learning is strongly data-limited. Pretrained supervised transfer substantially improves performance, but a foundation model with zero-shot transfer still achieves the strongest segmentation quality. The dominant practical lever is initialization and pretraining scale, with architecture-only tuning being secondary under limited labeled data.

\section*{{Reproducibility Artifacts}}
\begin{{itemize}}
  \item Bundle JSON: \texttt{{report\_assets/reports/final\_report\_bundle\_latest.json}}
  \item Per-class IoU CSV: \texttt{{report\_assets/reports/per\_class\_iou\_*.csv}}
  \item Model summary CSV: \texttt{{report\_assets/reports/model\_summary\_*.csv}}
  \item Ablation CSV: \texttt{{report\_assets/reports/ablation\_summary\_*.csv}}
\end{{itemize}}

\end{{document}}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Project 2 LaTeX report from exported assets.")
    parser.add_argument("--assets-dir", type=Path, default=Path("report_assets"))
    parser.add_argument("--out", type=Path, default=Path("project2_report.tex"))
    args = parser.parse_args()

    reports_dir = args.assets_dir / "reports"
    bundle_path = reports_dir / "final_report_bundle_latest.json"
    per_class_csv = latest_csv(reports_dir, "per_class_iou_*.csv")

    bundle = load_bundle(bundle_path)
    top_classes = load_top_classes(per_class_csv, top_k=8)
    tex = build_tex(bundle, top_classes)

    args.out.write_text(tex)
    print(f"Wrote: {args.out}")
    print(f"Assets dir: {args.assets_dir}")
    if per_class_csv:
        print(f"Per-class CSV: {per_class_csv}")


if __name__ == "__main__":
    main()
