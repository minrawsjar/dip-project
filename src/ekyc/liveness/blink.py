"""Blink detection via the eye aspect ratio (Soukupova & Cech 2016).

EAR = (|p2 - p6| + |p3 - p5|) / (2 |p1 - p4|); a blink is EAR below `ratio * median(EAR)` for at least
`consec` consecutive frames. A relative threshold adapts to each subject's eye shape.
"""

from __future__ import annotations

import numpy as np

from ..face.landmarks import EAR_IDX


def ear(mesh: np.ndarray) -> float:
    vals = []
    for p1, p2, p3, p4, p5, p6 in EAR_IDX:
        P = mesh[[p1, p2, p3, p4, p5, p6]]
        vals.append(
            (np.linalg.norm(P[1] - P[5]) + np.linalg.norm(P[2] - P[4]))
            / (2 * np.linalg.norm(P[0] - P[3]) + 1e-6)
        )
    return float(np.mean(vals))


def series(meshes: list[np.ndarray | None]) -> np.ndarray:
    """EAR per frame; NaN where no face was found."""
    return np.array([ear(m) if m is not None else np.nan for m in meshes], np.float32)


def blinks(ears: np.ndarray, ratio: float = 0.75, consec: int = 2) -> dict:
    valid = ears[~np.isnan(ears)]
    if len(valid) < 3:
        return {"blink_count": 0, "ear_median": 0.0, "ear_dip": 0.0, "ear_threshold": 0.0}
    med = float(np.median(valid))
    thr = ratio * med
    below = np.nan_to_num(ears, nan=med) < thr
    count, run = 0, 0
    for b in below:
        run = run + 1 if b else 0
        if run == consec:
            count += 1
    return {
        "blink_count": count,
        "ear_median": med,
        "ear_dip": float((med - valid.min()) / (med + 1e-6)),  # continuous score: depth of the deepest dip
        "ear_threshold": thr,
    }
