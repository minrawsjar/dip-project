"""MediaPipe Face Landmarker (478-point mesh) via the Tasks API, plus the index sets used downstream."""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np

from ..models import model_path

# Eye aspect ratio points p1..p6 (Soukupova & Cech 2016) on the MediaPipe mesh, image-left then image-right eye.
EAR_IDX = [(33, 160, 158, 133, 153, 144), (362, 385, 387, 263, 373, 380)]
EYE_RING = [
    [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
FACE_OVAL = [
    10,
    338,
    297,
    332,
    284,
    251,
    389,
    356,
    454,
    323,
    361,
    288,
    397,
    365,
    379,
    378,
    400,
    377,
    152,
    148,
    176,
    149,
    150,
    136,
    172,
    58,
    132,
    93,
    234,
    127,
    162,
    21,
    54,
    103,
    67,
    109,
]


@lru_cache(2)
def _landmarker(p: str):
    from mediapipe.tasks.python import BaseOptions, vision

    opts = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=p), num_faces=1)
    return vision.FaceLandmarker.create_from_options(opts)


def mesh(img: np.ndarray, cfg: dict) -> np.ndarray | None:
    """(478, 2) pixel coordinates of the face mesh, or None if no face is found."""
    import mediapipe as mp

    rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    res = _landmarker(str(model_path(cfg, "face_landmarker"))).detect(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    )
    if not res.face_landmarks:
        return None
    h, w = img.shape[:2]
    return np.array([(p.x * w, p.y * h) for p in res.face_landmarks[0]], np.float32)
