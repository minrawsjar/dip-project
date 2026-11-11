"""Field OCR chain: enhance -> denoise -> threshold -> morphology -> Tesseract.

MATLAB equivalents: `adapthisteq`, `medfilt2`, `imbinarize(I, 'global' | 'adaptive')`, `imopen`/`imclose`.
"""

from __future__ import annotations

import logging
import shutil

import cv2
import numpy as np

from .. import enhance, filters
from ..io import Debug, dbg, to_gray

log = logging.getLogger(__name__)
HAVE_TESSERACT = shutil.which("tesseract") is not None
if not HAVE_TESSERACT:
    log.warning("tesseract binary not found: OCR returns empty text (brew/apt install tesseract)")


def threshold(gray: np.ndarray, method: str, block: int = 31, c: int = 10) -> np.ndarray:
    """Global Otsu (`graythresh`) or adaptive mean/Gaussian local thresholds. Text ends up black on white."""
    if method == "none":
        return gray
    if method == "otsu":
        _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method in ("adaptive_mean", "adaptive_gaussian"):
        kind = cv2.ADAPTIVE_THRESH_MEAN_C if method == "adaptive_mean" else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        b = cv2.adaptiveThreshold(gray, 255, kind, cv2.THRESH_BINARY, block | 1, c)
    else:
        raise ValueError(method)
    return b if b.mean() > 127 else 255 - b  # Tesseract wants dark text on light paper


def morph(binary: np.ndarray, op: str, k: int = 2) -> np.ndarray:
    """Opening removes specks, closing bridges broken strokes. Applied to the text (inverted) layer."""
    if op == "none" or k < 1:
        return binary
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    flag = cv2.MORPH_OPEN if op == "open" else cv2.MORPH_CLOSE
    return 255 - cv2.morphologyEx(255 - binary, flag, se)


def preprocess(roi: np.ndarray, cfg: dict, debug: Debug | None = None) -> np.ndarray:
    """Run the configured chain on one field ROI; every step is dumped for figures."""
    o = cfg["document"]["ocr"]
    g = to_gray(roi)
    h = o["upscale_height"]
    if g.shape[0] and g.shape[0] != h:
        g = cv2.resize(g, (max(1, round(g.shape[1] * h / g.shape[0])), h), interpolation=cv2.INTER_CUBIC)
    dbg(debug, "gray", g)
    g = enhance.apply(g, o["enhance"], "gray", cfg)
    dbg(debug, "enhanced", g)
    g = filters.DENOISE[o["denoise"]](g, cfg["filters"]["ksize"])
    dbg(debug, "denoised", g)
    b = threshold(g, o["threshold"], o["adaptive_block"], o["adaptive_c"])
    dbg(debug, "binary", b)
    b = morph(b, o["morph"], o["morph_ksize"])
    dbg(debug, "morph", b)
    return cv2.copyMakeBorder(
        b, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255 if b.ndim == 2 else (255, 255, 255)
    )


def tesseract(img: np.ndarray, psm: int = 7, lang: str = "eng") -> tuple[str, float]:
    """(text, mean word confidence 0-100). Empty when Tesseract is unavailable."""
    if not HAVE_TESSERACT:
        return "", 0.0
    import pytesseract

    d = pytesseract.image_to_data(img, lang=lang, config=f"--psm {psm}", output_type=pytesseract.Output.DICT)
    words = [(w, float(c)) for w, c in zip(d["text"], d["conf"]) if str(w).strip() and float(c) >= 0]
    if not words:
        return "", 0.0
    return " ".join(w for w, _ in words), float(np.mean([c for _, c in words]))


def read_fields(
    card: np.ndarray, rects: dict[str, tuple[int, int, int, int]], cfg: dict, debug: Debug | None = None
) -> dict:
    """OCR each named pixel rect (x, y, w, h) of the rectified card."""
    o = cfg["document"]["ocr"]
    out = {}
    for name, (x, y, w, h) in rects.items():
        roi = card[max(0, y) : y + h, max(0, x) : x + w]
        if roi.size == 0:
            out[name] = {"text": "", "conf": 0.0}
            continue
        prep = preprocess(roi, cfg, debug.child(f"{name}/") if debug else None)
        text, conf = tesseract(prep, o["psm_field"], o["lang"])
        out[name] = {"text": text, "conf": conf}
    return out


def layout_rects(
    layout: dict, size: tuple[int, int], skip=("portrait",)
) -> dict[str, tuple[int, int, int, int]]:
    """Normalized (x, y, w, h) layout -> pixel rects for a card of size (w, h)."""
    W, H = size
    return {
        k: (round(x * W), round(y * H), round(w * W), round(h * H))
        for k, (x, y, w, h) in layout.items()
        if k not in skip
    }
