# Comparative Analysis of Object Detectors for Medical Images

**Course:** Deep Learning — Imam Khomeini International University
**Instructor:** Dr. Bahaghighat
**Task:** Compare Faster R-CNN vs. YOLOv8 (or newer) for brain tumor detection on MRI images.

## Dataset

[MRI for Brain Tumor with Bounding Boxes](https://www.kaggle.com/datasets/ahmedsorour1/mri-for-brain-tumor-with-bounding-boxes)
- 5,249 MRI images (sagittal, axial, coronal views)
- Annotations in YOLO format
- 4 tumor classes (note: project brief mentions 5 classes — verified and documented as a discrepancy in `notebooks/01_dataset_exploration.ipynb`)

## Project Structure

```
brain-tumor-detection-comparison/
├── notebooks/            # One notebook per project phase (run in Colab)
├── src/
│   ├── data/              # Download + annotation conversion + PyTorch Dataset
│   ├── models/             # Faster R-CNN and YOLO wrappers
│   ├── evaluation/         # Metrics + robustness corruption pipeline
│   ├── explainability/     # Grad-CAM
│   └── utils/               # Visualization helpers
├── configs/               # config.yaml (paths, hyperparameters)
├── results/                # figures / tables / training logs (tracked)
├── outputs/weights/        # trained model weights (NOT tracked, see .gitignore)
└── report/                  # final report source/PDF
```

## Workflow (matches project phases)

| Phase | Notebook | Description |
|---|---|---|
| 1 | — | Literature review (YOLOv8, XAI, Grad-CAM) |
| 2 | `01_dataset_exploration.ipynb` | Dataset download, verification, EDA |
| 3 | `02_baseline_fasterrcnn.ipynb` | Faster R-CNN baseline training |
| 4 | `03_yolo_implementation.ipynb` | YOLOv8 training (identical conditions) |
| 5 | `04_performance_comparison.ipynb` | Precision/Recall/F1/IoU/Dice/mAP + compute metrics |
| 6 | `05_robustness_evaluation.ipynb` | Corruption robustness (blur, noise, brightness, JPEG) |
| 7 | `06_explainability_gradcam.ipynb` | Grad-CAM analysis |
| 8 | `07_statistical_analysis.ipynb` | Bootstrap CI, permutation tests, Wilcoxon, McNemar |

## Setup (Google Colab)

```python
!git clone https://github.com/<your-username>/brain-tumor-detection-comparison.git
%cd brain-tumor-detection-comparison
!pip install -r requirements.txt -q
```

Then upload your `kaggle.json` and run `src/data/download.py` (see notebook 01).

## Reproducing Results

All hyperparameters and paths live in `configs/config.yaml`. Both detectors are trained
on the **same** train/val/test split, augmentation pipeline, and optimizer settings so
that the comparison in Phase 5 is fair (per project requirement: "Replace only the
detector. Do NOT modify: dataset, augmentation, optimizer").

## Results Summary

_(fill in after running the pipeline)_

| Model | mAP@0.5 | mAP@0.5:0.95 | FPS | Params (M) | GFLOPs |
|---|---|---|---|---|---|
| Faster R-CNN | | | | | |
| YOLOv8 | | | | | |

## License / Academic Note

This repository is submitted as coursework for the Deep Learning course. Dataset credit:
Ahmed Sorour (Kaggle), derived from Nickparvar and Bhuvaji brain tumor datasets.
