"""Colab environment setup: mounts Google Drive and detects the compute device."""
import torch


def setup_environment(mount_drive: bool = True):
    if mount_drive:
        from google.colab import drive
        drive.mount('/content/drive')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device
