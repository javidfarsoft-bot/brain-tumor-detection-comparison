"""Detection metrics: IoU, Dice, Precision/Recall/F1, and mAP (via torchmetrics)."""
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision


def box_iou(box1, box2):
    """box format: [x_min, y_min, x_max, y_max]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def dice_score(box1, box2):
    iou = box_iou(box1, box2)
    return (2 * iou) / (1 + iou) if iou > 0 else 0.0


def precision_recall_f1(tp: int, fp: int, fn: int):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compute_map(preds: list, targets: list):
    """
    preds / targets: lists of dicts in torchmetrics format, e.g.
      preds = [{"boxes": Tensor[N,4], "scores": Tensor[N], "labels": Tensor[N]}, ...]
      targets = [{"boxes": Tensor[M,4], "labels": Tensor[M]}, ...]
    Returns a dict with map, map_50, map_75, etc.
    """
    metric = MeanAveragePrecision(iou_type="bbox")
    metric.update(preds, targets)
    return metric.compute()
