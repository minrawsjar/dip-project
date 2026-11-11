"""Corner localisation alternatives and perspective rectification.

- contour path: largest external contour -> `approxPolyDP` to 4 vertices (MATLAB `bwboundaries` + simplification)
- Harris path: snap each contour vertex to the strongest Harris response nearby (`detectHarrisFeatures`)
- warp: homography to the canonical ID-1 rectangle (`fitgeotrans(..., 'projective')` + `imwarp`)
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from ..io import to_float


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points TL, TR, BR, BL (clockwise in image coordinates, starting top-left)."""
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    c = pts.mean(0)
    pts = pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]
    return np.roll(pts, -int(np.argmin(pts.sum(1))), axis=0)


def region_mask(edges: np.ndarray, dilate: int = 1) -> tuple[np.ndarray, np.ndarray | None]:
    """Largest filled blob from a closed edge map (MATLAB: `imclose`, `imfill`, keep largest)."""
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.dilate(edges, k, iterations=dilate) if dilate else edges
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, k, iterations=2)
    filled = ndimage.binary_fill_holes(closed > 0).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return np.zeros_like(edges), None
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros_like(edges)
    cv2.drawContours(mask, [c], -1, 255, -1)
    return mask, c


def quad_from_contour(contour: np.ndarray) -> np.ndarray:
    """Simplify a contour to 4 vertices; fall back to its min-area rectangle."""
    peri = cv2.arcLength(contour, True)
    for eps in (0.02, 0.03, 0.05, 0.08):
        approx = cv2.approxPolyDP(contour, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return order_quad(approx.reshape(4, 2))
    return order_quad(cv2.boxPoints(cv2.minAreaRect(contour)))


def harris_response(gray: np.ndarray, cfg: dict) -> np.ndarray:
    """R = det(M) - k trace(M)^2 of the structure tensor M."""
    h = cfg["document"]["harris"]
    return cv2.cornerHarris(to_float(gray), h["block"], h["ksize"], h["k"])


def harris_refine(gray: np.ndarray, quad: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Move each corner to the max Harris response within `snap_radius_frac * max(h, w)` px."""
    R = harris_response(gray, cfg)
    r = max(2, int(cfg["document"]["harris"]["snap_radius_frac"] * max(gray.shape)))
    out = quad.copy()
    for i, (x, y) in enumerate(quad.astype(int)):
        x0, y0 = max(0, x - r), max(0, y - r)
        win = R[y0 : y + r + 1, x0 : x + r + 1]
        if win.size and win.max() > 0:
            dy, dx = np.unravel_index(int(np.argmax(win)), win.shape)
            out[i] = (x0 + dx, y0 + dy)
    return out, R


def canonical_size(cfg: dict) -> tuple[int, int]:
    w = cfg["document"]["canonical_width"]
    return w, round(w / cfg["document"]["aspect"])


def homography(quad: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    return cv2.getPerspectiveTransform(order_quad(quad), dst)


def warp(img: np.ndarray, quad: np.ndarray, cfg: dict, size: tuple[int, int] | None = None) -> np.ndarray:
    """Rectify the card to the canonical ID-1 rectangle (or `size`) via the 4-point homography."""
    size = size or canonical_size(cfg)
    return cv2.warpPerspective(img, homography(quad, size), size, flags=cv2.INTER_CUBIC)


def canonical_corner_error(pred: np.ndarray, gt: np.ndarray, size: tuple[int, int]) -> float:
    """Mean corner error in px of the rectified card: map predicted corners through the GT homography."""
    H = homography(gt, size)
    p = cv2.perspectiveTransform(order_quad(pred).reshape(-1, 1, 2), H).reshape(4, 2)
    w, h = size
    ref = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    return float(np.linalg.norm(p - ref, axis=1).mean())
