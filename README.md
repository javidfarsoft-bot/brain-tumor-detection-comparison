# Comparative Analysis of Object Detectors for Medical Images

**Course:** Deep Learning — Imam Khomeini International University
**Instructor:** Dr. Bahaghighat
**Task:** Compare Faster R-CNN vs. YOLOv8n for brain tumor detection on MRI images.

## Dataset

[MRI for Brain Tumor with Bounding Boxes](https://www.kaggle.com/datasets/ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes) (Kaggle, CC0-1.0)
- 5,246 MRI images after removing 1 corrupt (empty-label) sample
- Annotations in YOLO format, merged from per-class subfolders
- 4 tumor classes (project brief mentions 5 — verified and documented as a
  dataset-level discrepancy in `notebooks/01_dataset_exploration.ipynb`)
- Split 80/10/10 (train/valid/test), stratified per class, seed=42

## Project Structure

```
brain-tumor-detection-comparison/
├── notebooks/              # one notebook per project phase
├── src/
│   ├── data/merge_dataset.py     # raw Kaggle download -> unified YOLO split
│   ├── models/                    # Faster R-CNN / YOLO build helpers (reference only,
│   │                               # see note below — actual training code is in the notebooks)
│   ├── evaluation/                 # metrics / robustness / statistics helpers (reference only)
│   ├── explainability/gradcam.py   # Grad-CAM helpers (reference only)
│   └── utils/
├── configs/config.yaml     # documents the actual hyperparameters used
├── results/                # figures / tables (tracked)
├── outputs/weights/        # YOLOv8n best checkpoint (~6MB, tracked);
│                            # Faster R-CNN weights (~330MB) are NOT tracked — see Note below
└── requirements.txt
```

> **Note on `src/`:** these modules were written as an initial scaffold before
> training began. The actual training loops (with validation, AMP, and
> early stopping on mAP@0.5) evolved during interactive Colab sessions and
> live directly in the phase notebooks below, which are the authoritative,
> reproducible source of truth for this project. The `src/` modules are kept
> for reference and reuse but may not be byte-identical to the notebook cells.

## Workflow (matches project phases)

| Phase | Notebook | Description |
|---|---|---|
| 1 | — | Literature review (YOLOv8, XAI, Grad-CAM) — see final report Section 2 |
| 2 | `01_dataset_exploration.ipynb` | Dataset download, merge, verification, EDA |
| 3 | `02_baseline_fasterrcnn.ipynb` | Faster R-CNN baseline training |
| 4 | `03_yolo_implementation.ipynb` | YOLOv8n training (identical conditions) |
| 5 | `04_performance_comparison.ipynb` | Precision/Recall/F1/IoU/Dice/mAP + compute metrics |
| 6 | `05_robustness_evaluation.ipynb` | Corruption robustness (blur, noise, brightness, JPEG) |
| 7 | `06_explainability_gradcam.ipynb` | Grad-CAM analysis (good/bad/failure cases) |
| 8 | `07_statistical_analysis.ipynb` | McNemar, Wilcoxon, permutation test, bootstrap CI |

## Setup (Google Colab)

```python
!git clone https://github.com/javidfarsoft-bot/brain-tumor-detection-comparison.git
%cd brain-tumor-detection-comparison
!pip install kaggle pycocotools ultralytics torchmetrics scipy statsmodels grad-cam -q
```

> Package notes: `pycocotools` and `torchmetrics` are needed by notebook 02
> (Faster R-CNN); `ultralytics` by notebook 03; `grad-cam` by notebook 06;
> `scipy` and `statsmodels` by notebook 07. Installing all of them up front
> (as above) covers every notebook.

Kaggle API access (needed by notebook 01, and re-downloaded at the start of
notebooks 02–07 since each Colab session starts with an empty disk):

```python
!mkdir -p ~/.kaggle
!echo YOUR_KAGGLE_TOKEN > ~/.kaggle/access_token
!chmod 600 ~/.kaggle/access_token
```

## Reproducing Results

Both detectors are trained on the **same** train/val/test split, an equivalent
augmentation policy, and the same optimizer settings, per the project
requirement: "Replace only the detector. Do NOT modify: dataset, augmentation,
optimizer." All actual hyperparameters are documented in `configs/config.yaml`.

Training used Google Colab's free-tier GPU (Tesla T4), which imposes session
time/quota limits; both notebooks checkpoint to Google Drive every epoch so
training can resume across multiple sessions if disconnected.

> **Note on exact figures:** because inference (not training) involves some
> GPU/cuDNN non-determinism and the robustness corruption functions are not
> seeded, exact metric values can shift slightly (typically <2%) between
> re-runs of notebooks 04, 05, and 07. The overall pattern and conclusions are
> stable across runs; see each notebook's discussion for details. The table
> below reflects the values embedded in the currently committed notebooks
> and `results/tables/`.

## Results Summary

| Model | Precision | Recall | F1 | IoU | Dice | mAP@0.5 | mAP@0.5:0.95 | Params (M) | GFLOPs | FPS | Model size (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Faster R-CNN | 0.8863 | 0.9813 | 0.9314 | 0.8747 | 0.9316 | 0.9527 | 0.6711 | 43.27 | 280.81 | 7.93 | 329.69 |
| YOLOv8n | 0.9493 | 0.9558 | 0.9525 | 0.8837 | 0.9366 | 0.9790 | 0.7350 | 3.01 | 8.10 | 83.83 | 6.20 |

Full tables: `results/tables/`. Full write-up, including robustness,
explainability, and statistical-significance analysis: final report.

## License / Academic Note

This repository is submitted as coursework for the Deep Learning course.
Dataset credit: Ahmed Sorour (Kaggle), derived from Nickparvar and Bhuvaji
brain tumor datasets.
