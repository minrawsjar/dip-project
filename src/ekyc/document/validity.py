"""Card validity rule classifier on region properties (Lab 5 shape rules; MATLAB `regionprops`, `bwconvhull`)."""

from __future__ import annotations

import cv2
import numpy as np


def region_features(contour: np.ndarray, image_area: float) -> dict:
    """Aspect ratio, rectangularity, solidity, perimeter^2/area and relative area of a blob contour."""
    area = float(cv2.contourArea(contour))
    peri = float(cv2.arcLength(contour, True))
    (_, _), (rw, rh), _ = cv2.minAreaRect(contour)
    hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
    return {
        "aspect": max(rw, rh) / max(min(rw, rh), 1e-6),
        "rectangularity": area / max(rw * rh, 1e-6),
        "solidity": area / max(hull_area, 1e-6),
        "p2a": peri**2 / max(area, 1e-6),
        "area_frac": area / image_area,
    }


def classify(features: dict, cfg: dict) -> tuple[bool, list[str]]:
    """Hand-written rules; every rule that fails adds a reason."""
    v = cfg["document"]["validity"]
    reasons = []
    if not v["aspect"][0] <= features["aspect"] <= v["aspect"][1]:
        reasons.append(f"aspect {features['aspect']:.2f} outside {v['aspect']}")
    if features["rectangularity"] < v["rectangularity_min"]:
        reasons.append(f"rectangularity {features['rectangularity']:.2f} < {v['rectangularity_min']}")
    if features["solidity"] < v["solidity_min"]:
        reasons.append(f"solidity {features['solidity']:.2f} < {v['solidity_min']}")
    if not v["p2a"][0] <= features["p2a"] <= v["p2a"][1]:
        reasons.append(f"perimeter^2/area {features['p2a']:.1f} outside {v['p2a']}")
    if features["area_frac"] < v["min_area_frac"]:
        reasons.append(f"card covers only {features['area_frac']:.0%} of frame")
    return not reasons, reasons


def check(contour: np.ndarray | None, shape: tuple[int, ...], cfg: dict) -> dict:
    if contour is None or len(contour) < 3:
        return {"valid": False, "features": {}, "reasons": ["no card region found"]}
    f = region_features(contour, float(np.prod(shape[:2])))
    ok, reasons = classify(f, cfg)
    return {"valid": ok, "features": f, "reasons": reasons}
