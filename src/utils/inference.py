"""Shared model-loading, prediction, and IoU helpers used by notebooks 04-07."""
import torch
from PIL import Image
import torchvision.transforms.functional as F
from ultralytics import YOLO
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

CLASS_NAMES = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

YOLO_WEIGHTS_PATH = "outputs/weights/yolo_best.pt"
FASTERRCNN_CKPT_PATH = "/content/drive/MyDrive/brain-tumor-project/checkpoints/fasterrcnn_best.pth"


def box_iou_np(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_models(device, yolo_path=YOLO_WEIGHTS_PATH, frcnn_ckpt_path=FASTERRCNN_CKPT_PATH):
    model_yolo = YOLO(yolo_path)

    num_classes = len(CLASS_NAMES) + 1
    model_frcnn = fasterrcnn_resnet50_fpn_v2(weights=None)
    in_features = model_frcnn.roi_heads.box_predictor.cls_score.in_features
    model_frcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    ckpt = torch.load(frcnn_ckpt_path, map_location=device)
    model_frcnn.load_state_dict(ckpt["model_state"])
    model_frcnn.to(device).eval()

    return model_yolo, model_frcnn


def make_predict_fns(model_yolo, model_frcnn, device, conf_threshold=0.5):
    def yolo_predict_fn(img_path):
        result = model_yolo.predict(img_path, conf=conf_threshold, verbose=False)[0]
        return result.boxes.xyxy.cpu().numpy().tolist() if len(result.boxes) > 0 else []

    def frcnn_predict_fn(img_path):
        img = Image.open(img_path).convert("RGB")
        img_tensor = F.to_tensor(img).to(device)
        with torch.no_grad():
            output = model_frcnn([img_tensor])[0]
        keep = output["scores"] >= conf_threshold
        return output["boxes"][keep].cpu().numpy().tolist()

    return yolo_predict_fn, frcnn_predict_fn
