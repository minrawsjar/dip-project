"""Intensity transformations and histogram processing (Lab 6, 17 Sep HSV lecture).

Each method works on a single uint8 channel; `apply` lifts it to colour images in one of three
modes: 'gray', 'rgb' (per channel: shifts hue, experiment E5) or 'hsv_v' (V channel only:
preserves hue).
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

from .io import to_float, to_uint8


def contrast_stretch(ch: np.ndarray, low_pct: float = 1, high_pct: float = 99) -> np.ndarray:
    """Linear stretch of the [p_low, p_high] percentile range to [0, 255] (`imadjust(I, stretchlim(I))`)."""
    lo, hi = np.percentile(ch, [low_pct, high_pct])
    if hi - lo < 1:
        return ch.copy()
    return to_uint8(np.clip((ch.astype(np.float32) - lo) / (hi - lo), 0, 1))


def log_transform(ch: np.ndarray) -> np.ndarray:
    """s = c * log(1 + r), c chosen so 255 -> 255. Expands dark tones."""
    f = to_float(ch)
    return to_uint8(np.log1p(f) / np.log(2.0))


def gamma(ch: np.ndarray, g: float = 0.7) -> np.ndarray:
    """Power law s = r^g. g < 1 brightens shadows (`imadjust(I, [], [], g)`)."""
    return to_uint8(to_float(ch) ** g)


def histeq(ch: np.ndarray) -> np.ndarray:
    """Global histogram equalization via the CDF, implemented from scratch (`histeq`).

    T(r_k) = round(255 * (cdf(r_k) - cdf_min) / (N - cdf_min)). Matches cv2.equalizeHist.
    """
    hist = np.bincount(ch.ravel(), minlength=256)
    cdf = hist.cumsum()
    cdf_min = cdf[cdf > 0][0]
    n = ch.size
    if n == cdf_min:
        return ch.copy()
    lut = np.clip(np.rint((cdf - cdf_min) * 255.0 / (n - cdf_min)), 0, 255).astype(np.uint8)
    return lut[ch]


def clahe(ch: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    """Contrast-limited adaptive histogram equalization (`adapthisteq`)."""
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(ch)


def methods(cfg: dict | None = None) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    e = (cfg or {}).get("enhance", {})
    lo, hi = e.get("stretch_pct", [1, 99])
    return {
        "none": lambda c: c,
        "stretch": lambda c: contrast_stretch(c, lo, hi),
        "log": log_transform,
        "gamma": lambda c: gamma(c, e.get("gamma", 0.7)),
        "histeq": histeq,
        "clahe": lambda c: clahe(c, e.get("clahe_clip", 2.0), e.get("clahe_tile", 8)),
    }


def apply(img: np.ndarray, method: str, mode: str = "hsv_v", cfg: dict | None = None) -> np.ndarray:
    """Enhance `img` with `method` in colour `mode` ('gray' | 'rgb' | 'hsv_v').

    'rgb' equalizes each BGR channel independently (MATLAB: histeq per channel) and shifts hue.
    'hsv_v' converts to HSV (`rgb2hsv`), enhances V only and converts back (`hsv2rgb`).
    """
    fn = methods(cfg)[method]
    if method == "none":
        return img.copy()
    if img.ndim == 2 or mode == "gray":
        g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return fn(g)
    if mode == "rgb":
        return cv2.merge([fn(c) for c in cv2.split(img)])
    if mode == "hsv_v":
        h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        return cv2.cvtColor(cv2.merge([h, s, fn(v)]), cv2.COLOR_HSV2BGR)
    raise ValueError(mode)


def hue_shift(a: np.ndarray, b: np.ndarray, s_min: int = 25, v_min: int = 25) -> float:
    """Mean absolute circular hue difference in degrees between two BGR images.

    Only pixels that are chromatic in the reference (S, V above thresholds) count, since hue is
    undefined for greys.
    """
    ha, sa, va = cv2.split(cv2.cvtColor(a, cv2.COLOR_BGR2HSV))
    hb = cv2.cvtColor(b, cv2.COLOR_BGR2HSV)[..., 0]
    m = (sa > s_min) & (va > v_min)
    if not m.any():
        return 0.0
    d = np.abs(ha[m].astype(np.int16) - hb[m].astype(np.int16)) * 2  # OpenCV hue is degrees / 2
    return float(np.minimum(d, 360 - d).mean())


def histogram(img: np.ndarray) -> list[int]:
    """256-bin grey-level histogram (`imhist`)."""
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.bincount(g.ravel(), minlength=256).tolist()
