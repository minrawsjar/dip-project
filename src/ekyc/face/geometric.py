"""Baseline face descriptor: normalised landmark distances (course facial-landmark line diagram).

Distances are divided by face width so the descriptor is invariant to image scale.
"""

from __future__ import annotations

import numpy as np

# MediaPipe mesh indices
L_EYE = (33, 133)  # image-left eye corners
R_EYE = (362, 263)  # image-right eye corners
L_LIP, R_LIP = 61, 291
L_BOUND, R_BOUND, TOP, BOTTOM = 234, 454, 10, 152

FEATURES = ["eye_eye", "leye_llip", "reye_rlip", "leye_bound", "reye_bound", "eyes_top", "lips_bottom"]


def descriptor(mesh: np.ndarray) -> np.ndarray:
    """7 distances / face width: eye-eye, eye-lip (L, R), eye-boundary (L, R), eye-top, lip-bottom."""
    le, re = mesh[list(L_EYE)].mean(0), mesh[list(R_EYE)].mean(0)
    ll, rl = mesh[L_LIP], mesh[R_LIP]
    d = lambda a, b: float(np.linalg.norm(a - b))
    width = d(mesh[L_BOUND], mesh[R_BOUND]) or 1.0
    eyes, lips = (le + re) / 2, (ll + rl) / 2
    v = [
        d(le, re),
        d(le, ll),
        d(re, rl),
        d(le, mesh[L_BOUND]),
        d(re, mesh[R_BOUND]),
        d(eyes, mesh[TOP]),
        d(lips, mesh[BOTTOM]),
    ]
    return np.array(v, np.float32) / width


def score(a: np.ndarray, b: np.ndarray) -> float:
    """Negative Euclidean distance (higher = more similar)."""
    return -float(np.linalg.norm(a - b))


def draw(img: np.ndarray, mesh: np.ndarray) -> np.ndarray:
    import cv2

    out = img.copy()
    le, re = mesh[list(L_EYE)].mean(0), mesh[list(R_EYE)].mean(0)
    pts = {
        "le": le,
        "re": re,
        "ll": mesh[L_LIP],
        "rl": mesh[R_LIP],
        "lb": mesh[L_BOUND],
        "rb": mesh[R_BOUND],
        "top": mesh[TOP],
        "bot": mesh[BOTTOM],
        "eyes": (le + re) / 2,
        "lips": (mesh[L_LIP] + mesh[R_LIP]) / 2,
    }
    for a, b in [
        ("le", "re"),
        ("le", "ll"),
        ("re", "rl"),
        ("le", "lb"),
        ("re", "rb"),
        ("eyes", "top"),
        ("lips", "bot"),
    ]:
        cv2.line(out, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), (0, 255, 255), 1, cv2.LINE_AA)
    for p in pts.values():
        cv2.circle(out, tuple(p.astype(int)), 2, (0, 0, 255), -1)
    return out
