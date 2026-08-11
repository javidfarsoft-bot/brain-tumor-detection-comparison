"""Thin wrapper around Ultralytics YOLO (reference; full training loop lives in notebooks/03)."""
from ultralytics import YOLO


def build_yolo(variant: str = "yolov8n.pt"):
    return YOLO(variant)


def train_yolo(model, data_yaml: str, cfg: dict, project="results/logs", name="yolo_run"):
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
