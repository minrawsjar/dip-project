"""Document module (CLAUDE.md §4.2): locate, validate, rectify, crop portrait, OCR fields.

`process_baseline`: grey -> Canny -> largest contour -> bounding-box crop -> Otsu -> Tesseract.
`process`: blur -> edges (sobel|prewitt|canny) -> Hough quad (contour fallback, optional Harris snap)
-> validity rules -> homography warp -> portrait (face detector on the card) -> per-field OCR chain.
"""

from __future__ import annotations

import cv2
import numpy as np

from .. import face as face_mod
from ..io import Debug, dbg, to_gray
from . import corners, edges, hough, ocr, validity


def _work(img: np.ndarray, cfg: dict) -> tuple[np.ndarray, float]:
    """Resize to the working width; returns (small, scale small->original)."""
    s = cfg["document"]["work_width"] / img.shape[1]
    if s >= 1:
        return img, 1.0
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), 1 / s


def _agrees(quad: np.ndarray, mask: np.ndarray, cfg: dict) -> bool:
    """Hough quad overlaps the region blob (IoU) and has a card-like side-length aspect."""
    q = np.zeros_like(mask)
    cv2.fillConvexPoly(q, quad.astype(np.int32), 255)
    a, b = q > 0, mask > 0
    iou = (a & b).sum() / max((a | b).sum(), 1)
    tl, tr, br, bl = corners.order_quad(quad)
    w = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    h = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    lo, hi = cfg["document"]["validity"]["aspect"]
    return iou >= cfg["document"]["hough"]["min_region_iou"] and lo <= max(w, h) / max(min(w, h), 1) <= hi


def locate(
    img: np.ndarray, cfg: dict, edge: str = "canny", corner: str = "hough", debug: Debug | None = None
) -> dict:
    """Find the card quad in original-image coordinates.

    corner: 'hough' (line intersections, contour fallback) | 'contour' | 'harris' (contour + Harris snap).
    """
    d = cfg["document"]
    small, back = _work(img, cfg)
    blur = cv2.GaussianBlur(small, (d["blur_ksize"], d["blur_ksize"]), 0)
    g = to_gray(blur)
    e = edges.detect(g, edge, cfg)
    if d.get("color_edges") and blur.ndim == 3:
        for ch in cv2.split(blur):
            e |= edges.detect(ch, edge, cfg)
    dbg(debug, f"edges_{edge}", e)
    mask, contour = corners.region_mask(e, d["dilate_edges"])
    dbg(debug, "region", mask)
    valid = validity.check(contour, small.shape, cfg)

    quad, used = None, corner
    if corner == "hough":
        segs = hough.segments(e, cfg)
        quad = hough.quad_from_segments(segs, small.shape, cfg, e)
        dbg(debug, "hough_lines", hough.draw(small, segs, quad))
        if quad is None:
            used = "contour_fallback"
        elif valid["valid"] and not _agrees(quad, mask, cfg):
            # Hough proposes, the region blob verifies: a low-contrast side lets Hough lock onto an
            # inner edge (portrait border, header bar); the filled blob does not have that failure mode
            quad, used = None, "contour_fallback (hough disagreed with region)"
    if quad is None and contour is not None:
        quad = corners.quad_from_contour(contour)
        if corner == "harris":
            quad, R = corners.harris_refine(g, quad, cfg)
            dbg(debug, "harris_response", np.log1p(np.maximum(R, 0) * 1e4))
    if quad is not None:
        quad = corners.order_quad(quad)
        dbg(debug, "quad", hough.draw(small, np.zeros((0, 4)), quad))
        quad = quad * back
    return {"quad": quad, "corner_method": used, "validity": valid}


def portrait(card: np.ndarray, cfg: dict) -> tuple[np.ndarray | None, tuple | None]:
    """Crop the portrait by running the face detector on the rectified card; layout slot as fallback."""
    f = face_mod.detect.detect(card, cfg, "yunet")
    if f is not None:
        x, y, w, h = f["box"]
        m = int(0.35 * max(w, h))
        x0, y0 = max(0, x - m), max(0, y - m)
        rect = (x0, y0, min(card.shape[1], x + w + m) - x0, min(card.shape[0], y + h + m) - y0)
    else:
        W, H = card.shape[1], card.shape[0]
        px, py, pw, ph = cfg["document"]["layout"]["portrait"]
        rect = (round(px * W), round(py * H), round(pw * W), round(ph * H))
    x, y, w, h = rect
    crop = card[y : y + h, x : x + w]
    return (crop if crop.size else None), rect


def process(
    img: np.ndarray,
    cfg: dict,
    field_rects: dict | None = None,
    edge: str = "canny",
    corner: str = "hough",
    debug: Debug | None = None,
) -> dict:
    """Improved document path. `field_rects`: name -> (x,y,w,h) in canonical-card pixels (default: config layout)."""
    loc = locate(img, cfg, edge, corner, debug)
    out = {
        "quad": None if loc["quad"] is None else loc["quad"].tolist(),
        "corner_method": loc["corner_method"],
        "validity": loc["validity"],
        "fields": {},
        "ocr_conf": 0.0,
        "portrait_rect": None,
        "found": loc["quad"] is not None,
    }
    if loc["quad"] is None:
        return out
    card = corners.warp(img, loc["quad"], cfg)
    dbg(debug, "warped", card)
    out["card"] = card
    crop, rect = portrait(card, cfg)
    out["portrait"], out["portrait_rect"] = crop, rect
    if crop is not None:
        dbg(debug, "portrait", crop)
    rects = field_rects or ocr.layout_rects(cfg["document"]["layout"], corners.canonical_size(cfg))
    if debug is not None:
        vis = card.copy()
        for x, y, w, h in rects.values():
            cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if rect:
            cv2.rectangle(vis, rect[:2], (rect[0] + rect[2], rect[1] + rect[3]), (0, 200, 0), 2)
        dbg(debug, "field_rois", vis)
    out["fields"] = ocr.read_fields(card, rects, cfg, debug.child("ocr/") if debug else None)
    confs = [f["conf"] for f in out["fields"].values()]
    out["ocr_conf"] = float(np.mean(confs)) if confs else 0.0
    return out


def process_baseline(img: np.ndarray, cfg: dict, debug: Debug | None = None) -> dict:
    """Baseline: no perspective correction, global Otsu, full-crop OCR."""
    small, back = _work(img, cfg)
    g = to_gray(small)
    e = edges.canny(g, *cfg["document"]["canny"])
    dbg(debug, "edges", e)
    cnts, _ = cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return {"found": False, "quad": None, "text": "", "ocr_conf": 0.0}
    x, y, w, h = (np.array(cv2.boundingRect(max(cnts, key=cv2.contourArea))) * back).astype(int)
    crop = img[y : y + h, x : x + w]
    dbg(debug, "bbox_crop", crop)
    _, b = cv2.threshold(to_gray(crop), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dbg(debug, "otsu", b)
    text, conf = ocr.tesseract(b, cfg["document"]["ocr"]["psm_full"], cfg["document"]["ocr"]["lang"])
    quad = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    return {"found": True, "quad": quad.tolist(), "crop": crop, "text": text, "ocr_conf": conf}
