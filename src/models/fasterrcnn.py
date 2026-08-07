"""Faster R-CNN baseline builder (transfer learning on a torchvision backbone)."""
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def build_fasterrcnn(num_classes: int, pretrained: bool = True):
    """
    num_classes must include the background class, i.e. len(class_names) + 1.
    """
    weights = "DEFAULT" if pretrained else None
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
