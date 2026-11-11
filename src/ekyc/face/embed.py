"""Learned face embedding: SFace (OpenCV Zoo), a ResNet-style CNN trained with a margin-based
hypersphere loss in the ArcFace family; 128-d, compared by cosine similarity.

Chosen over insightface ArcFace because it runs through OpenCV DNN with no extra runtime. Input is the
112x112 crop produced by `align.align_5pt` (same template SFace was trained with).
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from ..models import model_path


@lru_cache(2)
def _sface(p: str):
    return cv2.FaceRecognizerSF.create(p, "")


def embed(aligned_112: np.ndarray, cfg: dict) -> np.ndarray:
    v = _sface(str(model_path(cfg, "sface"))).feature(aligned_112).ravel().astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))
