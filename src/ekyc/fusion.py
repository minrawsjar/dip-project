"""Decision fusion (CLAUDE.md §4.5).

Baseline: fixed per-module thresholds with AND logic. Improved: logistic regression on module scores,
mapped to ACCEPT / REVIEW / REJECT by two probability thresholds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .io import path
from .models import load_trained

FEATURES = ["face_score", "liveness_score", "ocr_conf", "selfie_blur", "id_blur", "id_glare"]


def fixed(s: dict, cfg: dict) -> dict:
    """AND of per-module thresholds. Face or liveness failing -> REJECT; only OCR/quality failing -> REVIEW."""
    t = cfg["fusion"]["fixed"]
    face_ok = s.get("face_score") is not None and s["face_score"] >= t["face"]
    live_ok = s.get("liveness_score", 0.0) >= t["liveness"]
    ocr_ok = s.get("ocr_conf", 0.0) >= t["ocr"]
    quality_ok = s.get("quality_passed", True)
    if face_ok and live_ok and ocr_ok and quality_ok:
        decision = "ACCEPT"
    elif not face_ok or not live_ok:
        decision = "REJECT"
    else:
        decision = "REVIEW"
    checks = {"face": face_ok, "liveness": live_ok, "ocr": ocr_ok, "quality": quality_ok}
    return {"method": "fixed_and", "decision": decision, "checks": checks, "prob": None}


def vector(s: dict) -> np.ndarray:
    return np.array([[s.get(k) if s.get(k) is not None else -1.0 for k in FEATURES]], np.float32)


def train(X: np.ndarray, y: np.ndarray, seed: int = 0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        StandardScaler(), LogisticRegression(class_weight="balanced", random_state=seed)
    ).fit(X, y)


def decide(prob: float, cfg: dict) -> str:
    f = cfg["fusion"]
    return "ACCEPT" if prob >= f["accept"] else "REJECT" if prob < f["reject"] else "REVIEW"


def model_file(cfg: dict) -> Path:
    return path(cfg, "models") / cfg["models"]["fusion"]


def fuse(s: dict, cfg: dict) -> dict:
    """Logistic regression if a trained model exists (written by `ekyc eval fusion`), else fixed AND."""
    base = fixed(s, cfg)
    model = load_trained(model_file(cfg))
    if model is None:
        return base
    prob = float(model.predict_proba(vector(s))[0, 1])
    return {
        "method": "logistic_regression",
        "decision": decide(prob, cfg),
        "prob": prob,
        "checks": base["checks"],
        "baseline_decision": base["decision"],
    }
