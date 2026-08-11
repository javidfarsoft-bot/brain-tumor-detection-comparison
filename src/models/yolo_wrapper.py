"""Thin wrapper around ultralytics YOLO so training calls stay config-driven
and directly comparable to the Faster R-CNN run (same epochs/batch/optimizer).

Note: the actual Colab training run also disabled Ultralytics' extra default
augmentations (mosaic, mixup, scale, translate, shear, perspective) and the
albumentations package, and used AMP + validation-based early stopping — see
notebooks/03_yolo_implementation.ipynb for the exact reproducible call."""
from ultralytics import YOLO


def build_yolo(variant: str = "yolov8n.pt"):
    return YOLO(variant)


def train_yolo(model, data_yaml: str, cfg: dict, project="results/logs", name="yolo_run"):
    """
    data_yaml: path to a YOLO-format data.yaml (train/val/test paths + class names)
    cfg: the loaded configs/config.yaml dict — keeps hyperparams identical to Faster R-CNN
    """
    return model.train(
        data=data_yaml,
        epochs=cfg["training"]["epochs"],
        batch=cfg["training"]["batch_size"],
        imgsz=cfg["dataset"]["img_size"],
        optimizer=cfg["training"]["optimizer"],
        lr0=cfg["training"]["lr"],
        seed=cfg["project"]["seed"],
        patience=cfg["training"]["early_stopping_patience"],
        project=project,
        name=name,
    )
