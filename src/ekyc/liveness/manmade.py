"""Man-made vs natural cue detector (17 Sep lecture).

Paper edges and phone bezels add long straight lines around the face; screens add periodic moiré
(high-frequency FFT energy) and reduced saturation spread. Features are computed on the band around
the face box (lines, edges, entropy, texture) and on the face crop (FFT, HSV).
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.measure import shannon_entropy

FEATURES = [
    "line_count",
    "line_length",
    "edge_density",
    "entropy",
    "texture_var",
    "grad_mean",
    "hf_energy",
    "sat_mean",
    "sat_std",
    "val_mean",
    "val_std",
]


def band(img: np.ndarray, box, scale: float = 1.8) -> tuple[np.ndarray, np.ndarray]:
    """(band crop, mask of band pixels): the box scaled by `scale` minus the box itself."""
    x, y, w, h = box
    H, W = img.shape[:2]
    cx, cy = x + w / 2, y + h / 2
    x0, y0 = int(max(0, cx - w * scale / 2)), int(max(0, cy - h * scale / 2))
    x1, y1 = int(min(W, cx + w * scale / 2)), int(min(H, cy + h * scale / 2))
    crop = img[y0:y1, x0:x1]
    m = np.ones(crop.shape[:2], bool)
    m[max(0, y - y0) : y - y0 + h, max(0, x - x0) : x - x0 + w] = False
    return crop, m


def hf_energy(gray: np.ndarray, cutoff: float = 0.5) -> float:
    """Fraction of spectral energy above `cutoff` x Nyquist (radial), DC removed."""
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32) - gray.mean()))
    p = np.abs(f) ** 2
    h, w = gray.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot((yy - h / 2) / (h / 2), (xx - w / 2) / (w / 2))
    return float(p[r > cutoff].sum() / (p.sum() + 1e-9))


def features(img: np.ndarray, box, cfg: dict) -> dict:
    crop, m = band(img, box, cfg["liveness"]["band_scale"])
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(g, 50, 150)
    edges_band = np.where(m, edges, 0).astype(np.uint8)
    lines = cv2.HoughLinesP(edges_band, 1, np.pi / 180, 40, minLineLength=0.3 * min(g.shape), maxLineGap=5)
    lines = np.zeros((0, 4)) if lines is None else lines.reshape(-1, 4)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    x, y, w, h = box
    face = img[max(0, y) : y + h, max(0, x) : x + w]
    fg = cv2.resize(cv2.cvtColor(face, cv2.COLOR_BGR2GRAY), (128, 128), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    return {
        "line_count": float(len(lines)),
        "line_length": float(
            np.hypot(lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1]).sum() / max(g.shape)
        ),
        "edge_density": float((edges_band > 0)[m].mean()) if m.any() else 0.0,
        "entropy": float(shannon_entropy(g)),
        "texture_var": float(lap[m].var()) if m.any() else 0.0,
        "grad_mean": float(np.hypot(gx, gy)[m].mean()) if m.any() else 0.0,
        "hf_energy": hf_energy(fg),
        "sat_mean": float(hsv[..., 1].mean()),
        "sat_std": float(hsv[..., 1].std()),
        "val_mean": float(hsv[..., 2].mean()),
        "val_std": float(hsv[..., 2].std()),
    }


def rule_classify(f: dict, cfg: dict) -> tuple[bool, list[str]]:
    """Baseline hand-written rules; returns (looks_live, reasons for suspicion)."""
    r = cfg["liveness"]["rules"]
    reasons = []
    if f["line_count"] > r["line_count_max"]:
        reasons.append(f"{int(f['line_count'])} straight lines around face (paper/bezel edges)")
    if f["hf_energy"] > r["hf_energy_max"]:
        reasons.append(f"high-frequency energy {f['hf_energy']:.2f} (moire / halftone)")
    if f["sat_std"] < r["sat_std_min"]:
        reasons.append(f"flat saturation (std {f['sat_std']:.1f})")
    return not reasons, reasons
