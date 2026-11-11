"""Capture quality gate (CLAUDE.md §4.1): blur, exposure, glare, face size."""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import remove_small_objects

from .io import Debug, dbg, to_gray


def blur_score(img: np.ndarray) -> float:
    """Variance of the Laplacian (Lab 7). Low = blurry: blur removes the high frequencies it measures."""
    return float(cv2.Laplacian(to_gray(img), cv2.CV_64F).var())


def exposure_stats(img: np.ndarray, extreme_frac: float = 0.02) -> dict:
    """Histogram statistics (Lab 6): mean, mass in the bottom/top 2% of bins, p1-p99 dynamic range."""
    g = to_gray(img)
    hist = np.bincount(g.ravel(), minlength=256) / g.size
    nb = max(1, round(256 * extreme_frac))
    p1, p99 = np.percentile(g, [1, 99])
    return {
        "mean": float(g.mean()),
        "dark_frac": float(hist[:nb].sum()),
        "bright_frac": float(hist[-nb:].sum()),
        "dynamic_range": float(p99 - p1),
    }


def glare_mask(img: np.ndarray, v_min=230, s_max=40, smooth_sigma=2.0, min_area=30) -> np.ndarray:
    """Specular glare: HSV intensity slicing V > t_v AND S < t_s, then cleanup.

    Lab 3 cleanup chain: Gaussian smoothing of the mask, `bwareaopen` (remove_small_objects),
    `imfill` (binary_fill_holes).
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = (hsv[..., 2] > v_min) & (hsv[..., 1] < s_max)
    m = cv2.GaussianBlur(m.astype(np.float32), (0, 0), smooth_sigma) > 0.5
    m = remove_small_objects(m, max_size=min_area) if min_area > 0 else m
    return ndimage.binary_fill_holes(m)


def box_mask(shape, box) -> np.ndarray:
    x, y, w, h = box
    m = np.zeros(shape[:2], bool)
    m[max(0, y) : y + h, max(0, x) : x + w] = True
    return m


def quad_mask(shape, quad) -> np.ndarray:
    m = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(m, np.asarray(quad, np.int32), 1)
    return m.astype(bool)


def quality_gate(
    img: np.ndarray,
    cfg: dict,
    roi: np.ndarray | None = None,
    face_box: tuple[int, int, int, int] | None = None,
    debug: Debug | None = None,
) -> dict:
    """Run all checks. `roi` is an optional bool mask restricting the glare ratio; `face_box` is (x,y,w,h).

    Returns {passed, scores, reasons}.
    """
    q = cfg["quality"]
    g = q["glare"]
    scores = {"blur": blur_score(img), **exposure_stats(img, q["extreme_bin_frac"])}
    gm = glare_mask(img, g["v_min"], g["s_max"], g["smooth_sigma"], g["min_area"])
    region = roi if roi is not None else np.ones(gm.shape, bool)
    scores["glare_ratio"] = float(gm[region].mean()) if region.any() else 0.0
    dbg(debug, "laplacian", np.abs(cv2.Laplacian(to_gray(img), cv2.CV_32F)))
    dbg(debug, "glare_mask", gm)

    reasons = []
    if scores["blur"] < q["blur_min"]:
        reasons.append(f"blurry (var of Laplacian {scores['blur']:.0f} < {q['blur_min']})")
    if scores["dark_frac"] > q["dark_frac_max"]:
        reasons.append(f"underexposed ({scores['dark_frac']:.0%} pixels clipped dark)")
    if scores["bright_frac"] > q["bright_frac_max"]:
        reasons.append(f"overexposed ({scores['bright_frac']:.0%} pixels clipped bright)")
    if scores["dynamic_range"] < q["dynamic_range_min"]:
        reasons.append(f"low contrast (range {scores['dynamic_range']:.0f})")
    if scores["glare_ratio"] > g["ratio_max"]:
        reasons.append(f"glare ({scores['glare_ratio']:.1%} of ROI)")
    if face_box is not None:
        scores["face_area"] = face_box[2] * face_box[3] / (img.shape[0] * img.shape[1])
        if scores["face_area"] < q["face_area_min"]:
            reasons.append(f"face too small ({scores['face_area']:.1%} of frame)")
    return {"passed": not reasons, "scores": scores, "reasons": reasons}
