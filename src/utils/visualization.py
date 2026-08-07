"""Visualization helpers: sample images with bounding boxes, class distribution, training curves."""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter


def plot_sample_with_boxes(image, boxes, labels, class_names, ax=None):
    """boxes: list of [x_min, y_min, x_max, y_max] in absolute pixels."""
    if ax is None:
        fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(image)
    for box, label in zip(boxes, labels):
        x_min, y_min, x_max, y_max = box
        rect = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor="lime", facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x_min, max(y_min - 5, 0), class_names[label],
            color="white", fontsize=9,
            bbox=dict(facecolor="green", alpha=0.6, pad=1),
        )
    ax.axis("off")
    return ax


def plot_class_distribution(all_labels: list, class_names: list, out_path: str = None):
    counts = Counter(all_labels)
    names = [class_names[i] for i in sorted(counts)]
    values = [counts[i] for i in sorted(counts)]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, values, color="#4C72B0")
    ax.set_ylabel("Number of instances")
    ax.set_title("Class distribution")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
    return fig


def plot_training_curves(history: dict, out_path: str = None):
    """history: {'loss': [...], 'precision': [...], 'recall': [...], 'mAP': [...]}"""
    fig, axes = plt.subplots(1, len(history), figsize=(5 * len(history), 4))
    if len(history) == 1:
        axes = [axes]
    for ax, (key, values) in zip(axes, history.items()):
        ax.plot(values)
        ax.set_title(key)
        ax.set_xlabel("epoch")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150)
    return fig
