"""Motion-based liveness (Lab 3 change detection). MATLAB: `imabsdiff`, `imbinarize`, `bwareaopen`, `imfill`.

A live face shows localised, non-rigid motion (blinks, lips); a print shows almost none or moves
rigidly as a whole, so eye/mouth motion is not larger than the rest of the face.
"""

from __future__ import annotations

from itertools import pairwise

import cv2
import numpy as np
from scipy import ndimage
from skimage.morphology import remove_small_objects

from ..face.landmarks import EYE_RING, FACE_OVAL, LIPS


def change_masks(
    frames: list[np.ndarray], thresh: int, blur_k: int = 5, min_blob: int = 20
) -> list[np.ndarray]:
    """|I_t - I_{t-1}| > thresh on blurred grey frames, then bwareaopen + imfill."""
    g = [cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (blur_k | 1, blur_k | 1), 0) for f in frames]
    out = []
    for a, b in pairwise(g):
        m = cv2.absdiff(b, a) > thresh
        m = remove_small_objects(m, max_size=min_blob) if min_blob else m
        out.append(ndimage.binary_fill_holes(m))
    return out


def region_masks(mesh: np.ndarray, shape: tuple[int, ...], band_scale: float = 1.8) -> dict[str, np.ndarray]:
    """Eye+mouth, face-rest and background masks from mesh landmarks (logical mask ops)."""
    h, w = shape[:2]

    def poly(idx, dilate=0):
        m = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(m, cv2.convexHull(mesh[idx].astype(np.int32)), 1)
        if dilate:
            m = cv2.dilate(m, np.ones((dilate, dilate), np.uint8))
        return m.astype(bool)

    face = poly(FACE_OVAL)
    fw = max(4, int(np.ptp(mesh[:, 0]) * 0.06))
    em = poly(EYE_RING[0], fw) | poly(EYE_RING[1], fw) | poly(LIPS, fw)
    x, y, bw, bh = cv2.boundingRect(mesh.astype(np.int32))
    cx, cy = x + bw / 2, y + bh / 2
    outer = np.zeros((h, w), bool)
    sw, sh = bw * band_scale / 2, bh * band_scale / 2
    outer[max(0, int(cy - sh)) : int(cy + sh), max(0, int(cx - sw)) : int(cx + sw)] = True
    return {"eyes_mouth": em & face, "face_rest": face & ~em, "background": ~outer}


def features(frames: list[np.ndarray], meshes: list[np.ndarray | None], thresh: int, cfg: dict) -> dict:
    """Mean fraction of changed pixels per region, their ratio, and face/background rigidity."""
    lc = cfg["liveness"]
    masks = change_masks(frames, thresh, lc["blur_ksize"], lc["min_blob"])
    e = {"eyes_mouth": [], "face_rest": [], "background": []}
    for i, m in enumerate(masks):
        mesh = meshes[i + 1]
        if mesh is None:
            continue
        for k, r in region_masks(mesh, m.shape, lc["band_scale"]).items():
            e[k].append(float(m[r].mean()) if r.any() else 0.0)
    if not e["face_rest"]:
        return {
            "motion_eyes_mouth": 0.0,
            "motion_face": 0.0,
            "motion_bg": 0.0,
            "motion_ratio": 0.0,
            "motion_rigidity": 0.0,
        }
    em, fr, bg = (np.array(e[k]) for k in ("eyes_mouth", "face_rest", "background"))
    rig = float(np.corrcoef(fr, bg)[0, 1]) if fr.std() > 0 and bg.std() > 0 else 0.0
    return {
        "motion_eyes_mouth": float(em.mean()),
        "motion_face": float(fr.mean()),
        "motion_bg": float(bg.mean()),
        "motion_ratio": float(em.mean() / (fr.mean() + 1e-4)) if em.mean() > 1e-4 else 0.0,
        "motion_rigidity": rig,
    }


def visualize(frame: np.ndarray, mask: np.ndarray, mesh: np.ndarray | None, band_scale: float) -> np.ndarray:
    out = frame.copy()
    out[mask] = (0.4 * out[mask] + 0.6 * np.array([0, 0, 255])).astype(np.uint8)
    if mesh is not None:
        for k, col in zip(("eyes_mouth", "face_rest"), ((0, 255, 0), (255, 200, 0))):
            r = region_masks(mesh, frame.shape, band_scale)[k].astype(np.uint8)
            cnts, _ = cv2.findContours(r, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1, col, 1)
    return out
