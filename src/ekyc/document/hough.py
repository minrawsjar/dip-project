"""Hough-transform card corners. MATLAB: `hough`, `houghpeaks`, `houghlines`.

Probabilistic Hough segments are split into near-horizontal and near-vertical groups; each group is
split into two halves (top/bottom, left/right) and the dominant (longest) line of each half, refit
with its collinear neighbours, becomes a card side. The four sides intersect in the corners.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.transform import hough_line, hough_line_peaks


def accumulator(edges: np.ndarray, n_peaks: int = 8):
    """Standard (rho, theta) accumulator and its peaks, for the report figure."""
    theta = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False)
    h, theta, rho = hough_line(edges > 0, theta=theta)
    peaks = hough_line_peaks(h, theta, rho, num_peaks=n_peaks)
    return h, theta, rho, peaks


def segments(edges: np.ndarray, cfg: dict) -> np.ndarray:
    """(N, 4) array of segments x1, y1, x2, y2 from `cv2.HoughLinesP`."""
    hc = cfg["document"]["hough"]
    min_len = hc["min_line_frac"] * edges.shape[1]
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, hc["threshold"], minLineLength=min_len, maxLineGap=hc["max_gap"]
    )
    return np.zeros((0, 4), np.float32) if lines is None else lines.reshape(-1, 4).astype(np.float32)


def _fit(segs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares line (point, unit direction) through segment endpoints, weighted by length."""
    pts = []
    for x1, y1, x2, y2 in segs:
        n = max(2, int(np.hypot(x2 - x1, y2 - y1) // 10))
        pts.append(np.linspace([x1, y1], [x2, y2], n))
    vx, vy, x0, y0 = cv2.fitLine(np.concatenate(pts).astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    return np.array([x0, y0]), np.array([vx, vy])


def _intersect(l1, l2) -> np.ndarray | None:
    (p, d), (q, e) = l1, l2
    A = np.array([d, -e]).T
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    t = np.linalg.solve(A, q - p)[0]
    return p + t * d


def _refine(line, a: np.ndarray, b: np.ndarray, pts: np.ndarray, band: float):
    """Least-squares refit of a side on the edge pixels within `band` px of it, between its corners a, b.

    Hough localises lines only to its (rho, theta) bin size; the refit recovers sub-bin accuracy.
    """
    p, d = line
    n = np.array([-d[1], d[0]])
    ab = b - a
    t = ((pts - a) * ab).sum(1) / max(float((ab * ab).sum()), 1e-6)
    dist = np.abs(((pts - p) * n).sum(1))
    near = pts[(dist < band) & (t > 0.1) & (t < 0.9)]  # skip the cluttered corners
    if len(near) < 20:
        return line
    vx, vy, x0, y0 = cv2.fitLine(near.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    return np.array([x0, y0]), np.array([vx, vy])


def _sides(segs: np.ndarray, offset: np.ndarray, tol: float, long_frac: float = 0.7):
    """Split a group by length-weighted 1-D 2-means on offset; return the refit dominant line per half."""
    length = np.hypot(segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1])
    c = np.array([offset.min(), offset.max()])
    for _ in range(10):
        far = np.abs(offset - c[0]) > np.abs(offset - c[1])
        if far.all() or (~far).all():
            return None
        c = np.array(
            [np.average(offset[~far], weights=length[~far]), np.average(offset[far], weights=length[far])]
        )
    out = []
    for half, pick in ((~far, np.argmin), (far, np.argmax)):
        idx = np.where(half)[0]
        # outermost of the long lines: inner card features (header bars, text rows) are long too
        long_ = idx[length[idx] >= long_frac * length[idx].max()]
        best = long_[pick(offset[long_])]
        near = idx[np.abs(offset[idx] - offset[best]) < tol]
        out.append(_fit(segs[near]))
    return out


def quad_from_segments(
    segs: np.ndarray, shape: tuple[int, int], cfg: dict, edges: np.ndarray | None = None
) -> np.ndarray | None:
    """Four corners (TL, TR, BR, BL) from Hough segments, or None if the geometry is not a quad.

    With `edges`, each side is refit by least squares on nearby edge pixels before intersecting.
    """
    if len(segs) < 4:
        return None
    h, w = shape[:2]
    dx, dy = segs[:, 2] - segs[:, 0], segs[:, 3] - segs[:, 1]
    ang = np.degrees(np.arctan2(dy, dx))
    ang = (ang + 90) % 180 - 90  # [-90, 90)
    tol = cfg["document"]["hough"]["angle_tol_deg"]
    horiz, vert = np.abs(ang) < tol, np.abs(ang) > 90 - tol  # diagonal text strokes fall in neither
    if horiz.sum() < 2 or vert.sum() < 2:
        return None
    H, V = segs[horiz], segs[vert]
    # offset of each line at the image centre: y at x = w/2 for horizontals, x at y = h/2 for verticals
    hy = H[:, 1] + (w / 2 - H[:, 0]) * (H[:, 3] - H[:, 1]) / np.where(
        H[:, 2] == H[:, 0], 1e-6, H[:, 2] - H[:, 0]
    )
    vx = V[:, 0] + (h / 2 - V[:, 1]) * (V[:, 2] - V[:, 0]) / np.where(
        V[:, 3] == V[:, 1], 1e-6, V[:, 3] - V[:, 1]
    )
    tol = 0.02 * max(h, w)
    lf = cfg["document"]["hough"]["long_frac"]
    hs, vs = _sides(H, hy, tol, lf), _sides(V, vx, tol, lf)
    if hs is None or vs is None:
        return None
    (top, bottom), (left, right) = hs, vs

    def corners_of(top, bottom, left, right):
        return [
            _intersect(top, left),
            _intersect(top, right),
            _intersect(bottom, right),
            _intersect(bottom, left),
        ]

    c = corners_of(top, bottom, left, right)
    if any(v is None for v in c):
        return None
    if edges is not None:
        pts = np.column_stack(np.nonzero(edges))[:, ::-1].astype(np.float64)
        band = cfg["document"]["hough"]["refine_band_px"]
        tl, tr, br, bl = c
        top, bottom = _refine(top, tl, tr, pts, band), _refine(bottom, bl, br, pts, band)
        left, right = _refine(left, tl, bl, pts, band), _refine(right, tr, br, pts, band)
        c = corners_of(top, bottom, left, right)
        if any(v is None for v in c):
            return None
    q = np.array(c, np.float32)
    margin = 0.1 * max(h, w)
    if (
        (q[:, 0] < -margin).any()
        or (q[:, 0] > w + margin).any()
        or (q[:, 1] < -margin).any()
        or (q[:, 1] > h + margin).any()
    ):
        return None
    return q


def draw(img: np.ndarray, segs: np.ndarray, quad: np.ndarray | None) -> np.ndarray:
    out = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in segs.astype(int):
        cv2.line(out, (x1, y1), (x2, y2), (0, 200, 255), 1)
    if quad is not None:
        cv2.polylines(out, [quad.astype(np.int32)], True, (0, 255, 0), 2)
        for x, y in quad.astype(int):
            cv2.circle(out, (x, y), 5, (0, 0, 255), -1)
    return out
