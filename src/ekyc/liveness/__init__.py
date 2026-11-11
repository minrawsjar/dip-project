"""Passive liveness from a short selfie clip: motion + EAR blink + man-made cues (+ optional CNN).

`analyze` returns every handcrafted feature plus per-signal scores in [0, 1] and a combined
`liveness_score`. If a trained logistic-regression model exists (`models/liveness_lr.joblib`, written by
`ekyc eval liveness`), it replaces the rule-based combination.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from ..face import detect, landmarks
from ..io import Debug, dbg, path
from ..models import load_trained
from . import blink, manmade, motion

log = logging.getLogger(__name__)
FEATURES = [
    "motion_eyes_mouth",
    "motion_face",
    "motion_bg",
    "motion_ratio",
    "motion_rigidity",
    "blink_count",
    "ear_dip",
] + manmade.FEATURES


def _lr_model(cfg: dict):
    return load_trained(path(cfg, "models") / cfg["models"]["liveness_lr"])


def _ear_plot(ears: np.ndarray, thr: float) -> np.ndarray:
    """EAR time series as an image (for the debug dump / web UI)."""
    h, w = 160, 480
    img = np.full((h, w, 3), 255, np.uint8)
    v = np.nan_to_num(ears, nan=np.nanmedian(ears) if np.isfinite(ears).any() else 0)
    if len(v) < 2 or v.max() <= 0:
        return img
    top = max(v.max(), thr) * 1.15
    pts = np.stack([np.linspace(10, w - 10, len(v)), h - 10 - v / top * (h - 20)], 1).astype(np.int32)
    yt = int(h - 10 - thr / top * (h - 20))
    cv2.line(img, (10, yt), (w - 10, yt), (0, 0, 220), 1, cv2.LINE_AA)
    cv2.polylines(img, [pts], False, (180, 90, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "EAR", (12, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
    return img


def analyze(
    frames: list[np.ndarray], cfg: dict, debug: Debug | None = None, thresh: int | None = None
) -> dict:
    """Liveness features + scores for a list of BGR frames (a single frame skips motion and blink)."""
    lc = cfg["liveness"]
    meshes = [landmarks.mesh(f, cfg) for f in frames]
    mid = len(frames) // 2
    face = detect.detect(frames[mid], cfg)
    feats: dict = {"frames": len(frames), "faces_found": int(sum(m is not None for m in meshes))}

    if face is not None:
        feats.update(manmade.features(frames[mid], face["box"], cfg))
        crop, m = manmade.band(frames[mid], face["box"], lc["band_scale"])
        vis = crop.copy()
        vis[~m] = (vis[~m] * 0.35).astype(np.uint8)
        dbg(debug, "manmade_band", vis)
    else:
        feats.update({k: 0.0 for k in manmade.FEATURES})

    if len(frames) > 2:
        t = thresh if thresh is not None else lc["diff_threshold"]
        feats.update(motion.features(frames, meshes, t, cfg))
        ears = blink.series(meshes)
        feats.update(blink.blinks(ears, lc["ear_ratio"], lc["ear_consec"]))
        if debug is not None:
            masks = motion.change_masks(frames, t, lc["blur_ksize"], lc["min_blob"])
            k = int(np.argmax([mk.sum() for mk in masks]))
            dbg(debug, "frame_diff", cv2.absdiff(frames[k + 1], frames[k]))
            dbg(
                debug,
                "motion_mask",
                motion.visualize(frames[k + 1], masks[k], meshes[k + 1], lc["band_scale"]),
            )
            dbg(debug, "ear_series", _ear_plot(ears, feats["ear_threshold"]))
    else:
        feats.update({k: 0.0 for k in FEATURES if k.startswith("motion") or k in ("blink_count", "ear_dip")})

    live_rules, reasons = manmade.rule_classify(feats, cfg) if face is not None else (False, ["no face"])
    scores = {
        "manmade": 1.0 if live_rules else 0.0,
        "motion": float(np.clip(feats["motion_ratio"] / (2 * lc["motion_ratio_min"]), 0, 1)),
        "blink": (
            1.0
            if feats["blink_count"] > 0
            else float(np.clip(feats["ear_dip"] / (1 - lc["ear_ratio"]), 0, 1) * 0.5)
        ),
    }
    model = _lr_model(cfg) if len(frames) > 2 else None  # trained on clips; motion features are 0 for stills
    if model is not None:
        x = np.array([[feats[k] for k in FEATURES]], np.float32)
        combined, method = float(model.predict_proba(x)[0, 1]), "logistic_regression"
    elif len(frames) > 2:
        combined, method = float(np.mean(list(scores.values()))), "rules_mean"
    else:
        combined, method = scores["manmade"], "rules_single_frame"
    return {
        "features": feats,
        "scores": scores,
        "liveness_score": combined,
        "method": method,
        "reasons": reasons,
    }
