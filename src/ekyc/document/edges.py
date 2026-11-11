"""Edge detection for card localisation. MATLAB: `edge(I, 'sobel' | 'prewitt' | 'canny')`."""

from __future__ import annotations

import cv2
import numpy as np

from ..io import to_float

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
PREWITT_X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32)


def gradient_magnitude(gray: np.ndarray, kx: np.ndarray) -> np.ndarray:
    """sqrt(Gx^2 + Gy^2) with Gx = I * kx, Gy = I * kx^T (correlation, `imfilter`)."""
    f = to_float(gray)
    gx = cv2.filter2D(f, cv2.CV_32F, kx)
    gy = cv2.filter2D(f, cv2.CV_32F, kx.T)
    return np.hypot(gx, gy)


def threshold_magnitude(mag: np.ndarray, scale: float = 4.0) -> np.ndarray:
    """MATLAB's automatic cutoff for `edge(...,'sobel')`: mag^2 > scale * mean(mag^2)."""
    return (mag**2 > scale * (mag**2).mean()).astype(np.uint8) * 255


def sobel(gray: np.ndarray, scale: float = 4.0) -> np.ndarray:
    return threshold_magnitude(gradient_magnitude(gray, SOBEL_X), scale)


def prewitt(gray: np.ndarray, scale: float = 4.0) -> np.ndarray:
    return threshold_magnitude(gradient_magnitude(gray, PREWITT_X), scale)


def canny(gray: np.ndarray, lo: int = 50, hi: int = 150) -> np.ndarray:
    """Gradient + non-maximum suppression + hysteresis thresholding."""
    return cv2.Canny(gray, lo, hi)


def detect(gray: np.ndarray, method: str, cfg: dict) -> np.ndarray:
    """Binary uint8 {0, 255} edge map with the chosen operator."""
    d = cfg["document"]
    if method == "canny":
        return canny(gray, *d["canny"])
    if method == "sobel":
        return sobel(gray, d["edge_scale"])
    if method == "prewitt":
        return prewitt(gray, d["edge_scale"])
    raise ValueError(method)
