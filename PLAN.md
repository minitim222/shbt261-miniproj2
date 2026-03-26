# Mini-Project 2: Semantic Segmentation — Execution Plan

## Project Overview
- **Course**: SHBT261
- **Task**: Semantic segmentation on Pascal VOC 2007 (21 classes: 20 objects + background)
- **Deadline**: April 15, 2026
- **Contact**: mengyu_wang@meei.harvard.edu; mohammad_eslami@meei.harvard.edu
- **Environment**: Google Colab (GPU runtime — T4 minimum, A100 preferred)

---

## Repository Structure

```
mini-proj-2/
├── PLAN.md                  ← this file
├── 01_dataset.ipynb         ← dataset setup, exploration, class distribution
├── 02_unet.ipynb            ← U-Net training + ablation toggles
├── 03_sam2.ipynb            ← SAM2 zero-shot inference
└── 04_evaluation.ipynb      ← metrics, visualizations, ablation table
```

---

## Storage Layout

| Location | Purpose |
|---|---|
| `/content/drive/MyDrive/SHBT261_mini_proj2/` | Persistent Drive folder |
| `/content/drive/MyDrive/SHBT261_mini_proj2/VOCtrainval_06-Nov-2007.zip` | Dataset zip (downloaded once, persists) |
| `/content/drive/MyDrive/SHBT261_mini_proj2/mini_proj2_checkpoints/` | Model checkpoints, plots, predictions |
| `/content/VOCdata/` | Local SSD extraction (fast I/O during training) |
| `/content/project_config.json` | Shared config (VOC_ROOT, IMG_SIZE, classes) |
| `/root/.kaggle/kaggle.json` | Kaggle credentials (upload manually once per session) |

---

## Execution Order

### Step 1 — Run `01_dataset.ipynb`
**Goal**: Download dataset, set up DataLoaders, explore data.

```
Cell 1: pip install dependencies
Cell 2: Mount Google Drive
Cell 3: Upload kaggle.json (manual step — run once)
Cell 4: Download VOC 2007 zip from Kaggle → save to Drive (skipped if zip exists)
Cell 5: Copy zip from Drive → extract to /content/VOCdata/ (skipped if already extracted)
Cell 6: Define VOC_CLASSES (21), transforms (resize 256x256, normalize ImageNet stats)
Cell 7: Load VOCSegmentation train + val datasets; val is used as test set per assignment
Cell 8: Visualize 4 random samples (image + colour mask)
Cell 9: Plot class frequency bar chart
Cell 10: Save /content/project_config.json (shared by all other notebooks)
```

**Key config values saved**:
- `voc_root`: `/content/VOCdata`
- `img_size`: `[256, 256]`
- `num_classes`: `21`
- `voc_classes`: list of 21 class name strings

---

### Step 2 — Run `02_unet.ipynb` (multiple times for ablation)
**Goal**: Train U-Net, save best checkpoint per variant.

```
Cell 1: pip install
Cell 2: Load config, set DEVICE, set SAVE_DIR = Drive checkpoints folder
Cell 3: Build DataLoaders (toggle USE_AUGMENTATION)
Cell 4: Define U-Net from scratch (DoubleConv, encoder, bottleneck, decoder)
Cell 5: Define loss function (toggle LOSS_FN)
Cell 6: Training loop — 30 epochs, AdamW, CosineAnnealingLR
         → saves best checkpoint as: unet_aug{True/False}_loss{ce/dice}_best.pth
Cell 7: Plot training curves (loss, mIoU, pixel accuracy)
Cell 8: Ablation notes table
```

**Ablation variants to run** (change toggles, re-run notebook each time):

| Run | `USE_AUGMENTATION` | `LOSS_FN` | Checkpoint name |
|---|---|---|---|
| 1 (baseline) | `True`  | `'ce'`   | `unet_augTrue_lossce_best.pth` |
| 2 | `False` | `'ce'`   | `unet_augFalse_lossce_best.pth` |
| 3 | `True`  | `'dice'` | `unet_augTrue_lossdice_best.pth` |

**Architecture**: Standard U-Net, features=[64, 128, 256, 512], ~31M params  
**Training**: 30 epochs, batch_size=8, LR=1e-3, weight_decay=1e-4, CosineAnnealingLR  
**Augmentation**: RandomHorizontalFlip + ColorJitter  
**Normalization**: ImageNet mean/std

---

### Step 3 — Run `03_sam2.ipynb`
**Goal**: Zero-shot SAM2 inference using GT bounding box prompts.

```
Cell 1: pip install + install SAM2 from GitHub (facebookresearch/sam2)
Cell 2: Load config, set DEVICE
Cell 3: Download SAM2 checkpoint (vit_b, ~375MB) from Meta CDN → /content/sam2_checkpoints/
Cell 4: Build SAM2 model + SAM2ImagePredictor
Cell 5: Load val dataset WITHOUT ImageNet normalization (SAM2 takes raw uint8 RGB)
Cell 6: For each image, extract GT bounding boxes per class → pass to SAM2 predictor
         → combine per-class masks into single prediction mask
Cell 7: Compute quick mIoU on N_EVAL=50 images
Cell 8: Visualize 3 sample comparisons
Cell 9: Save predictions to Drive as sam2_preds.pkl
```

**Prompt strategy**: GT bounding boxes per class instance (upper bound on prompt quality)  
**Model**: `sam2.1_hiera_base_plus` (balance of speed and accuracy)  
**Checkpoint URL**: `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt`

---

### Step 4 — Run `04_evaluation.ipynb`
**Goal**: Full evaluation, comparison, visualizations, ablation table.

```
Cell 1: pip install
Cell 2: Load config + colormap
Cell 3: Rebuild U-Net class, load best checkpoint (unet_augTrue_lossce_best.pth)
Cell 4: Run full U-Net inference on entire val set
Cell 5: Compute metrics — pixel accuracy, mIoU, per-class IoU, per-class accuracy
Cell 6: Compute HD95 per class (on 50-image subset for speed)
Cell 7: Per-class IoU bar chart
Cell 8: Best 3 and worst 3 predictions mosaic (per image mIoU ranking)
Cell 9: Load sam2_preds.pkl, compute SAM2 metrics
Cell 10: U-Net vs SAM2 comparison table
Cell 11: Ablation results table (fill in after all 02_unet runs)
Cell 12: Confusion matrix (normalised, all 21 classes)
```

**Metrics required by assignment**:
- [x] Mean Dice Coefficient & mIoU — overall pixel-wise agreement
- [x] 95th percentile Hausdorff Distance (HD95)
- [x] Pixel Accuracy — fraction correctly labelled
- [x] Per-class IoU and Accuracy
- [ ] Confusion Matrix (optional — implemented)

**Visualizations required**:
- [x] Qualitative segmentation maps — mosaic style
- [x] Side-by-side GT vs prediction — top 3 best, top 3 worst
- [x] Consider human/person class specifically

---

## Ablation Studies (≥2 required)

| Study | Variable | Values |
|---|---|---|
| 1. Data augmentation | `USE_AUGMENTATION` in `02_unet.ipynb` | `True` vs `False` |
| 2. Loss function | `LOSS_FN` in `02_unet.ipynb` | `'ce'` vs `'dice'` |

Optional additional ablations (not yet implemented):
- Encoder backbone: ResNet-18 vs ResNet-50 (swap U-Net encoder)
- Pre-training: random init vs ImageNet-pretrained encoder
- Image resolution: 256×256 vs 512×512

---

## Manual Steps Required (one-time)

1. **Kaggle credentials**: Before running `01_dataset.ipynb` cell 4:
   ```python
   from google.colab import files
   files.upload()  # upload kaggle.json
   !mv kaggle.json /root/.kaggle/
   !chmod 600 /root/.kaggle/kaggle.json
   ```

2. **Colab runtime**: Set to GPU before running any notebook:
   `Runtime → Change runtime type → T4 GPU (or A100)`

---

## Deliverables Checklist

- [ ] `01_dataset.ipynb` — executed with outputs
- [ ] `02_unet.ipynb` — run ×3 for ablation variants
- [ ] `03_sam2.ipynb` — executed with outputs
- [ ] `04_evaluation.ipynb` — executed with all metrics and plots
- [ ] Project report: methods, results, observations, ablation studies, lessons learned
