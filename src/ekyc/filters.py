"""Spatial filters (filtering labs, Lab 7). MATLAB: `imfilter`, `fspecial`, `medfilt2`."""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from skimage.filters.rank import modal
from skimage.morphology import footprint_rectangle

from .io import to_float, to_uint8


def mean(img: np.ndarray, k: int = 3) -> np.ndarray:
    """Box filter, `fspecial('average', k)`."""
    return cv2.blur(img, (k, k))


def gaussian(img: np.ndarray, k: int = 3) -> np.ndarray:
    """Gaussian smoothing, `fspecial('gaussian', k, sigma)` with OpenCV's sigma rule for size k."""
    return cv2.GaussianBlur(img, (k | 1, k | 1), 0)


def median(img: np.ndarray, k: int = 3) -> np.ndarray:
    """Order-statistic median filter, `medfilt2`. Best for salt-and-pepper noise."""
    return cv2.medianBlur(img, k | 1)


def mode(img: np.ndarray, k: int = 3) -> np.ndarray:
    """Mode (most frequent value) filter, per channel."""
    fp = footprint_rectangle((k, k))
    if img.ndim == 2:
        return modal(img, fp)
    return np.dstack([modal(img[..., c], fp) for c in range(img.shape[2])])


def laplacian(img: np.ndarray, k: int = 3) -> np.ndarray:
    """Second-derivative edge response, `fspecial('laplacian')`. Returns float32."""
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(to_float(g), cv2.CV_32F, ksize=k | 1)


def log(img: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Laplacian of Gaussian, `fspecial('log', ..., sigma)`. Returns float32."""
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return ndimage.gaussian_laplace(to_float(g), sigma)


def unsharp(img: np.ndarray, k: int = 5, amount: float = 1.0) -> np.ndarray:
    """Unsharp masking: I + amount * (I - blur(I))."""
    f = to_float(img)
    return to_uint8(np.clip(f + amount * (f - to_float(gaussian(img, k))), 0, 1))


DENOISE = {"none": lambda x, k: x, "mean": mean, "gaussian": gaussian, "median": median, "mode": mode}
ALL = {**DENOISE, "laplacian": laplacian, "log": lambda x, k: log(x, k / 2), "unsharp": unsharp}


def apply(img: np.ndarray, name: str, k: int = 3) -> np.ndarray:
    return ALL[name](img, k)
