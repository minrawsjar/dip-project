"""Document evaluation: card localisation (quad IoU, canonical corner error) and field OCR (CER, exact match).

Data: synthetic captured cards (always available) and MIDV-2020 photos (if downloaded; capped by --midv-n).
Baseline: Canny + largest contour bounding box, no rectification, Otsu, Tesseract.
Improved: colour edges + Hough quad + homography + enhance/denoise/threshold chain + Tesseract.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from .. import metrics, viz
from ..document import corners, locate, ocr, process_baseline, template
from ..io import Debug, imread, load_config, path, with_overrides, write_results
from .datasets import midv_root, synthetic_pairs

log = logging.getLogger(__name__)
BASELINE_OCR = {
    "document.ocr.enhance": "none",
    "document.ocr.denoise": "none",
    "document.ocr.threshold": "otsu",
    "document.ocr.morph": "none",
}


def items(cfg: dict, n_synth: int | None = None, n_midv: int = 200) -> list[dict]:
    """Uniform item dicts: id, source, image, quad (gt), size (rectified w, h), fields {name: {rect, value}}."""
    out = []
    try:
        df = synthetic_pairs(cfg)
        size = corners.canonical_size(cfg)
        for r in df.head(n_synth).itertuples() if n_synth else df.itertuples():
            out.append(
                {
                    "id": r.pair_id,
                    "source": "synthetic",
                    "image": r.scene,
                    "quad": np.float32(r.quad),
                    "size": size,
                    "fields": r.fields,
                    "severity": r.severity,
                }
            )
    except FileNotFoundError as e:
        log.warning("%s", e)
    root = midv_root(cfg)
    if root is not None:
        midv = list(template.midv_photos(root))
        rng = np.random.default_rng(cfg["seed"])
        for i in sorted(rng.permutation(len(midv))[:n_midv]):
            m = midv[i]
            with Image.open(m["template"]) as t:
                size = t.size
            out.append(
                {
                    "id": m["id"],
                    "source": "midv2020",
                    "image": m["image"],
                    "quad": m["quad"],
                    "size": size,
                    "fields": m["fields"],
                    "severity": np.nan,
                }
            )
    return out


def field_cer(card: np.ndarray, fields: dict, cfg: dict) -> dict:
    rects = {k: tuple(int(v) for v in f["rect"]) for k, f in fields.items()}
    read = ocr.read_fields(card, rects, cfg)
    cers = [metrics.cer(f["value"], read[k]["text"]) for k, f in fields.items()]
    exact = [f["value"].strip() == read[k]["text"].strip() for k, f in fields.items()]
    return {
        "cer": float(np.mean(cers)) if cers else np.nan,
        "exact": float(np.mean(exact)) if exact else np.nan,
        "ocr_conf": float(np.mean([r["conf"] for r in read.values()])) if read else 0.0,
    }


def run_item(it: dict, cfg: dict, edge="canny", corner="hough", do_ocr=True, img=None) -> dict:
    img = imread(it["image"]) if img is None else img
    t0 = time.perf_counter()
    loc = locate(img, cfg, edge, corner)
    row = {
        "id": it["id"],
        "source": it["source"],
        "edge": edge,
        "corner": corner,
        "found": loc["quad"] is not None,
        "valid": loc["validity"]["valid"],
        "corner_method_used": loc["corner_method"],
    }
    if loc["quad"] is not None:
        row["iou"] = metrics.quad_iou(loc["quad"], it["quad"])
        row["corner_err"] = corners.canonical_corner_error(loc["quad"], it["quad"], it["size"])
        if do_ocr:
            row.update(field_cer(corners.warp(img, loc["quad"], cfg, it["size"]), it["fields"], cfg))
    else:
        row.update({"iou": 0.0, "corner_err": np.nan, "cer": 1.0 if do_ocr else np.nan, "exact": 0.0})
    row["ms"] = (time.perf_counter() - t0) * 1000
    return row


def run_baseline(it: dict, cfg: dict, img=None) -> dict:
    """Bounding box only: crop is resized to the card size without perspective correction."""
    import cv2

    img = imread(it["image"]) if img is None else img
    t0 = time.perf_counter()
    b = process_baseline(img, cfg)
    row = {"id": it["id"], "source": it["source"], "edge": "canny", "corner": "bbox", "found": b["found"]}
    if b["found"]:
        q = np.float32(b["quad"])
        row["iou"] = metrics.quad_iou(q, it["quad"])
        row["corner_err"] = corners.canonical_corner_error(q, it["quad"], it["size"])
        card = cv2.resize(b["crop"], it["size"], interpolation=cv2.INTER_CUBIC)
        row.update(field_cer(card, it["fields"], with_overrides(cfg, BASELINE_OCR)))
    else:
        row.update({"iou": 0.0, "corner_err": np.nan, "cer": 1.0, "exact": 0.0})
    row["ms"] = (time.perf_counter() - t0) * 1000
    return row


def summarize(df: pd.DataFrame, by: list[str], cfg: dict) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by):
        key = key if isinstance(key, tuple) else (key,)
        r = dict(zip(by, key))
        for m in ("iou", "corner_err", "cer"):
            v = g[m].dropna().values if m in g else []  # cer absent when OCR is skipped (E6, E7)
            if len(v):
                r[m], r[f"{m}_lo"], r[f"{m}_hi"] = metrics.bootstrap_ci(
                    np.mean, v, n=cfg["eval"]["bootstrap"], seed=cfg["seed"]
                )
        r.update(
            {
                "exact": g.get("exact", pd.Series(dtype=float)).mean(),
                "found": g["found"].mean(),
                "iou>0.9": (g["iou"] > 0.9).mean(),
                "ms": g["ms"].mean(),
                "n": len(g),
            }
        )
        if "valid" in g:
            r["valid"] = g["valid"].mean()
        rows.append(r)
    return pd.DataFrame(rows)


def main(cfg: dict | None = None, n_midv: int = 200) -> pd.DataFrame:
    cfg = cfg or load_config()
    its = items(cfg, n_midv=n_midv)
    if not its:
        raise SystemExit(
            "no document data: run scripts/make_degraded_pairs.py (and/or scripts/download_midv2020.py)"
        )
    rows = []
    for it in tqdm(its, desc="document"):
        img = imread(it["image"])
        rows.append({**run_item(it, cfg, img=img), "variant": "improved"})
        rows.append({**run_baseline(it, cfg, img=img), "variant": "baseline"})
    df = pd.DataFrame(rows)
    write_results(df, "document_scores", cfg)
    s = summarize(df, ["source", "variant"], cfg)
    write_results(s, "document_summary", cfg)
    d = Debug()
    from ..document import process

    process(
        imread(its[0]["image"]),
        cfg,
        field_rects={k: tuple(f["rect"]) for k, f in its[0]["fields"].items()},
        debug=d,
    )
    keep = {k: v for k, v in d.images.items() if "/" not in k or k.startswith("ocr/surname")}
    viz.panel(
        keep,
        path(cfg, "results") / "figures" / "document_intermediates.png",
        cols=4,
        title="Document module intermediates",
    )
    print(s.round(4).to_string(index=False))
    return s
