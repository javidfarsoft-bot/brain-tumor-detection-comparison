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
