"""Generates corrupted copies of test images for robustness evaluation (Phase 6)."""
import cv2
import numpy as np


def apply_brightness(img: np.ndarray, delta: int):
    return np.clip(img.astype(np.int16) + delta, 0, 255).astype(np.uint8)


def apply_gaussian_noise(img: np.ndarray, std: float):
    noise = np.random.normal(0, std, img.shape).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def apply_salt_pepper(img: np.ndarray, amount: float = 0.02):
    out = img.copy()
    n_salt = int(amount * img.size * 0.5)
    n_pepper = int(amount * img.size * 0.5)

    coords = [np.random.randint(0, i, n_salt) for i in img.shape[:2]]
    out[coords[0], coords[1]] = 255

    coords = [np.random.randint(0, i, n_pepper) for i in img.shape[:2]]
    out[coords[0], coords[1]] = 0
    return out


def apply_gaussian_blur(img: np.ndarray, kernel: int = 5):
    return cv2.GaussianBlur(img, (kernel, kernel), 0)


def apply_motion_blur(img: np.ndarray, kernel: int = 9):
    k = np.zeros((kernel, kernel))
    k[(kernel - 1) // 2, :] = np.ones(kernel)
    k /= kernel
    return cv2.filter2D(img, -1, k)


def apply_jpeg_compression(img: np.ndarray, quality: int = 20):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, enc = cv2.imencode(".jpg", img, encode_param)
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


CORRUPTIONS = {
    "darker": lambda img: apply_brightness(img, -40),
    "brighter": lambda img: apply_brightness(img, 40),
    "gaussian_noise": lambda img: apply_gaussian_noise(img, 25),
    "salt_pepper": lambda img: apply_salt_pepper(img, 0.02),
    "gaussian_blur": lambda img: apply_gaussian_blur(img, 9),
    "motion_blur": lambda img: apply_motion_blur(img, 9),
    "jpeg_20": lambda img: apply_jpeg_compression(img, 20),
    "jpeg_50": lambda img: apply_jpeg_compression(img, 50),
}
