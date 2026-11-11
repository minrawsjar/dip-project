"""Face detection: Haar cascade (baseline, Viola-Jones) and YuNet CNN (improved, 5 landmarks)."""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from ..models import model_path


@lru_cache(4)
def _haar(p: str):
    return cv2.CascadeClassifier(p)


@lru_cache(4)
def _yunet(p: str, score: float, nms: float):
    return cv2.FaceDetectorYN.create(p, "", (320, 320), score, nms, 5000)


def haar(img: np.ndarray, cfg: dict) -> list[dict]:
    """Viola-Jones boosted cascade of Haar-like features on the grey image."""
    f = cfg["face"]
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    det = _haar(str(model_path(cfg, "haar")))
    min_side = max(24, min(g.shape) // 10)
    boxes = det.detectMultiScale(g, f["haar_scale"], f["haar_neighbors"], minSize=(min_side, min_side))
    return [
        {"box": tuple(int(v) for v in b), "pts5": None, "score": 1.0} for b in (boxes if len(boxes) else [])
    ]


def yunet(img: np.ndarray, cfg: dict) -> list[dict]:
    """YuNet: box, 5 landmarks (image-left eye, image-right eye, nose, mouth corners) and score."""
    f = cfg["face"]
    det = _yunet(str(model_path(cfg, "yunet")), f["yunet_score"], f["yunet_nms"])
    h, w = img.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    out = []
    for r in faces if faces is not None else []:
        x, y, bw, bh = r[:4]
        out.append(
            {
                "box": (int(x), int(y), int(bw), int(bh)),
                "pts5": r[4:14].reshape(5, 2).astype(np.float32),
                "score": float(r[14]),
            }
        )
    return out


def detect(img: np.ndarray, cfg: dict, method: str | None = None) -> dict | None:
    """Largest face found by the configured detector, or None."""
    faces = (haar if (method or cfg["face"]["detector"]) == "haar" else yunet)(img, cfg)
    return max(faces, key=lambda d: d["box"][2] * d["box"][3]) if faces else None


def draw(img: np.ndarray, face: dict | None, mesh: np.ndarray | None = None) -> np.ndarray:
    out = img.copy()
    if mesh is not None:
        for x, y in mesh.astype(int):
            cv2.circle(out, (x, y), 1, (255, 200, 0), -1)
    if face is not None:
        x, y, w, h = face["box"]
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if face["pts5"] is not None:
            for px, py in face["pts5"].astype(int):
                cv2.circle(out, (px, py), 3, (0, 0, 255), -1)
    return out
