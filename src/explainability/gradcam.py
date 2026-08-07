"""
Grad-CAM wrappers for both detectors (Phase 7).

Uses the `grad-cam` (pytorch-grad-cam) package. Faster R-CNN and YOLO expose
different internal graphs, so each needs its own target-layer + reshape_transform.
"""
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


def run_gradcam_fasterrcnn(model, target_layers, input_tensor, rgb_img_float01):
    """
    model: eval()'d Faster R-CNN
    target_layers: e.g. [model.backbone.body.layer4[-1]]
    input_tensor: normalized image tensor, shape [1, C, H, W]
    rgb_img_float01: original image as float32 HxWx3 in [0,1] (for overlay)
    """
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    visualization = show_cam_on_image(rgb_img_float01, grayscale_cam, use_rgb=True)
    return visualization, grayscale_cam


def run_gradcam_yolo(yolo_model, target_layers, input_tensor, rgb_img_float01):
    """
    yolo_model: the underlying nn.Module from an ultralytics YOLO() instance (yolo_model.model.model)
    target_layers: e.g. [yolo_model.model.model[-2]]  # verify index for the chosen variant
    """
    cam = GradCAM(model=yolo_model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    visualization = show_cam_on_image(rgb_img_float01, grayscale_cam, use_rgb=True)
    return visualization, grayscale_cam


def save_comparison_grid(images: list, titles: list, out_path: str):
    """Save [original | heatmap] rows for good/bad/failure cases, per Phase 7 deliverable."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(images), 2, figsize=(8, 4 * len(images)))
    if len(images) == 1:
        axes = np.expand_dims(axes, 0)
    for i, (orig, heat) in enumerate(images):
        axes[i, 0].imshow(orig)
        axes[i, 0].set_title(f"{titles[i]} — Original")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(heat)
        axes[i, 1].set_title(f"{titles[i]} — Grad-CAM")
        axes[i, 1].axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
