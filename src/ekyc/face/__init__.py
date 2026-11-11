"""Face match: detect -> (upscale) -> align -> photometric normalisation -> descriptor -> score.

Baseline (CLAUDE.md §4.3): Haar + geometric landmark distances, negative Euclidean distance.
Improved: YuNet + 5-point alignment + HSV-V contrast stretch + SFace embedding, cosine similarity.
Every choice is a config key under `face:` so ablations flip one at a time via `io.with_overrides`.
"""

from __future__ import annotations

import cv2
import numpy as np

from .. import enhance
from ..io import Debug, dbg
from . import align, detect, embed, geometric, landmarks

# photometric option -> (enhance method, colour mode)
PHOTOMETRIC = {
    "none": ("none", "hsv_v"),
    "histeq_v": ("histeq", "hsv_v"),
    "clahe_v": ("clahe", "hsv_v"),
    "histeq_rgb": ("histeq", "rgb"),
    "clahe_rgb": ("clahe", "rgb"),
    "stretch_v": ("stretch", "hsv_v"),
    "gamma_v": ("gamma", "hsv_v"),
}

BASELINE = {
    "face.detector": "haar",
    "face.descriptor": "geometric",
    "face.align": False,
    "face.photometric": "none",
    "face.descreen_sigma": 0,
}


def represent(img: np.ndarray, cfg: dict, debug: Debug | None = None) -> dict:
    """Detect the largest face and compute its descriptor. Returns {found, box, vector, ...}."""
    f = cfg["face"]
    up, s = align.upscale(img, f["portrait_min_side"], f["upscale_interp"])
    face = detect.detect(up, cfg)
    descreened = False
    if face is None and f.get("descreen_sigma", 0) > 0:
        # Printed ID portraits carry a halftone dot screen that defeats the detector; a Gaussian
        # low-pass removes it (descreening). Only on failure, so clean images are untouched.
        up = cv2.GaussianBlur(up, (0, 0), f["descreen_sigma"])
        face, descreened = detect.detect(up, cfg), True
        dbg(debug, "descreened", up)
    if face is None:
        return {"found": False, "reason": "no face detected"}
    dbg(debug, "detection", detect.draw(up, face))
    out = {
        "found": True,
        "box": [round(v / s) for v in face["box"]],
        "det_score": face["score"],
        "descreened": descreened,
    }

    if f["descriptor"] == "geometric":
        crop = align.crop_box(up, face["box"], size=256, margin=0.4)
        m = landmarks.mesh(crop, cfg)
        if m is None:
            return {"found": False, "reason": "no landmarks on face crop"}
        dbg(debug, "landmark_distances", geometric.draw(crop, m))
        out["vector"] = geometric.descriptor(m)
        return out

    if f["align"] and face["pts5"] is not None:
        crop = align.align_5pt(up, face["pts5"])
    else:
        crop = align.crop_box(up, face["box"])
    dbg(debug, "aligned", crop)
    method, mode = PHOTOMETRIC[f["photometric"]]
    crop = enhance.apply(crop, method, mode, cfg)
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    dbg(debug, "normalized", crop)
    out["vector"] = embed.embed(crop, cfg)
    out["crop"] = crop
    return out


def score(a: dict, b: dict, cfg: dict) -> float | None:
    """Similarity of two representations (None if either face is missing)."""
    if not (a.get("found") and b.get("found")):
        return None
    if cfg["face"]["descriptor"] == "geometric":
        return geometric.score(a["vector"], b["vector"])
    return embed.cosine(a["vector"], b["vector"])


def threshold(cfg: dict) -> float:
    return cfg["face"][
        "geometric_threshold" if cfg["face"]["descriptor"] == "geometric" else "match_threshold"
    ]
