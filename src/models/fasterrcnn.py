"""Builds a Faster R-CNN model for transfer learning (reference; full training loop lives in notebooks/02)."""
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_fasterrcnn(num_classes: int, pretrained: bool = True):
    weights = "DEFAULT" if pretrained else None
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
