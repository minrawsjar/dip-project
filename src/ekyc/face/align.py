"""Geometric normalisation: 5-point similarity alignment and interpolation-based upscaling.

MATLAB: `fitgeotrans(moving, fixed, 'nonreflectivesimilarity')` + `imwarp`; `imresize(..., method)`.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..degrade import INTERP

# ArcFace canonical landmark positions in a 112x112 crop (image-left eye first, as YuNet emits them).
TEMPLATE_112 = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    np.float32,
)


def align_5pt(img: np.ndarray, pts5: np.ndarray, size: int = 112) -> np.ndarray:
    """Similarity transform (rotation + uniform scale + translation) mapping pts5 onto the template."""
    M, _ = cv2.estimateAffinePartial2D(pts5.astype(np.float32), TEMPLATE_112 * (size / 112), method=cv2.LMEDS)
    if M is None:
        raise ValueError("degenerate landmarks")
    return cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def crop_box(img: np.ndarray, box, size: int = 112, margin: float = 0.15) -> np.ndarray:
    """Unaligned baseline: square crop around the detector box, resized (no rotation correction)."""
    x, y, w, h = box
    s = int(max(w, h) * (1 + 2 * margin))
    cx, cy = x + w // 2, y + h // 2
    pad = cv2.copyMakeBorder(img, s, s, s, s, cv2.BORDER_REPLICATE)
    crop = pad[cy - s // 2 + s : cy + s // 2 + s, cx - s // 2 + s : cx + s // 2 + s]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


def upscale(img: np.ndarray, min_side: int, interp: str = "bicubic") -> tuple[np.ndarray, float]:
    """Upscale so the short side is at least `min_side` using the chosen interpolation kernel."""
    s = min_side / min(img.shape[:2])
    if s <= 1:
        return img, 1.0
    out = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=INTERP[interp])
    return out, s
