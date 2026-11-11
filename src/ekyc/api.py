"""REST API + static web UI.

    GET  /api/health           models / tesseract availability
    GET  /api/config           active configuration
    POST /api/verify           selfie + ID image (+ optional clip) -> decision, scores, timings, intermediate images
    POST /api/lab              apply one DIP operation to an uploaded image (interactive lab)
    GET  /api/experiments      E1-E14 + headline tables and figures from results/
    /results/*                 result figures
    /*                         built frontend (web/dist)

Images go back as base64 data URLs, so the server keeps no per-request state.
"""

from __future__ import annotations

import base64
import json
import math
import os
import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import degrade, document, enhance, face, filters, quality
from .document import edges, hough, ocr
from .io import ROOT, Debug, imdecode, load_config, path, read_video, to_gray, with_overrides
from .pipeline import verify as run_verify

CFG = load_config(os.environ.get("EKYC_CONFIG"))
MAX_IMAGE_MB, MAX_VIDEO_MB, LAB_MAX_SIDE, OUT_MAX_SIDE = 15, 80, 1280, 720
# ponytail: one global lock (MediaPipe/OpenCV DNN objects are shared and not thread-safe); per-model locks if throughput matters
LOCK = threading.Lock()

app = FastAPI(title="eKYC DIP", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------------------------------------------- helpers


def _read(f: UploadFile, limit_mb: int) -> bytes:
    data = f.file.read(limit_mb * 2**20 + 1)
    if len(data) > limit_mb * 2**20:
        raise HTTPException(413, f"{f.filename}: larger than {limit_mb} MB")
    if not data:
        raise HTTPException(400, f"{f.filename}: empty upload")
    return data


def _image(f: UploadFile, max_side: int | None = None) -> np.ndarray:
    try:
        img = imdecode(_read(f, MAX_IMAGE_MB))
    except ValueError:
        raise HTTPException(400, f"{f.filename}: not a decodable image") from None
    if max_side and max(img.shape[:2]) > max_side:
        s = max_side / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img


def _src(img: np.ndarray) -> str:
    """Downscaled data URL: PNG for single-channel (masks, edges), JPEG for colour."""
    if max(img.shape[:2]) > OUT_MAX_SIDE:
        s = OUT_MAX_SIDE / max(img.shape[:2])
        img = cv2.resize(
            img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if img.ndim == 3 else cv2.INTER_NEAREST
        )
    ext, mime, params = (
        (".png", "image/png", []) if img.ndim == 2 else (".jpg", "image/jpeg", [cv2.IMWRITE_JPEG_QUALITY, 88])
    )
    _, buf = cv2.imencode(ext, img, params)
    return f"data:{mime};base64,{base64.b64encode(buf).decode()}"


def _clean(o):
    """Recursively convert numpy / NaN values into JSON-safe Python."""
    if isinstance(o, dict):
        return {
            str(k): _clean(v)
            for k, v in o.items()
            if not isinstance(v, np.ndarray) or v.ndim <= 2 and v.size <= 64
        }
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return o


def _gallery(d: Debug) -> list[dict]:
    out = []
    for key, img in d.images.items():
        parts = key.split("/")
        out.append(
            {
                "key": key,
                "stage": parts[0],
                "group": "/".join(parts[1:-1]),
                "label": parts[-1],
                "src": _src(img),
            }
        )
    return out


# ----------------------------------------------------------------------------- endpoints


@app.get("/api/health")
def health():
    models = {
        k: (path(CFG, "models") / v["file"]).exists() for k, v in CFG["models"].items() if isinstance(v, dict)
    }
    trained = {
        k: (path(CFG, "models") / CFG["models"][k]).exists()
        for k in ("fusion", "liveness_lr", "liveness_cnn")
    }
    return {"status": "ok", "tesseract": ocr.HAVE_TESSERACT, "models": models, "trained": trained}


@app.get("/api/config")
def config():
    return CFG


@app.post("/api/verify")
def verify(
    selfie: UploadFile = File(...), id_image: UploadFile = File(...), video: UploadFile | None = File(None)
):
    s, i = _image(selfie), _image(id_image)
    frames = None
    if video is not None and video.filename:
        data = _read(video, MAX_VIDEO_MB)
        suffix = Path(video.filename).suffix.lower() or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(data)
            tmp.flush()
            try:
                lc = CFG["liveness"]
                frames = read_video(tmp.name, lc["max_frames"], lc["frame_stride"])
            except ValueError:
                raise HTTPException(400, f"{video.filename}: could not decode video") from None
    d = Debug()
    with LOCK:
        res = run_verify(s, i, CFG, frames, d)
    res["images"] = _gallery(d)
    res["inputs"] = {"selfie": _src(s), "id": _src(i), "frames": len(frames) if frames else 0}
    return _clean(res)


def _lab(img: np.ndarray, op: str, p: dict) -> dict:
    """One DIP operation; returns {output, extras: [{label, src}], info}."""
    g = to_gray(img)
    extras, info = [], {}
    if op == "enhance":
        out = enhance.apply(img, p.get("method", "histeq"), p.get("mode", "hsv_v"), CFG)
        if out.ndim == 3:
            info["hue_shift_deg"] = enhance.hue_shift(img, out)
    elif op == "filter":
        out = filters.apply(img, p.get("name", "median"), int(p.get("k", 3)))
    elif op == "edges":
        out = edges.detect(cv2.GaussianBlur(g, (5, 5), 0), p.get("method", "canny"), CFG)
        info["edge_density"] = float((out > 0).mean())
    elif op == "threshold":
        out = ocr.threshold(g, p.get("method", "otsu"), int(p.get("block", 31)), int(p.get("c", 10)))
    elif op == "morphology":
        _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(p.get("k", 3)),) * 2)
        ops = {
            "erode": cv2.MORPH_ERODE,
            "dilate": cv2.MORPH_DILATE,
            "open": cv2.MORPH_OPEN,
            "close": cv2.MORPH_CLOSE,
            "gradient": cv2.MORPH_GRADIENT,
            "tophat": cv2.MORPH_TOPHAT,
        }
        out = cv2.morphologyEx(b, ops[p.get("op", "open")], k)
        extras.append({"label": "otsu binary", "src": _src(b)})
    elif op == "degrade":
        name = p.pop("op", "jpeg")
        if name not in degrade.OPS:
            raise ValueError(f"unknown degradation {name}")
        out, info = degrade.apply(img, name, seed=int(p.pop("seed", 0)), **p)
        info.pop("H", None)
    elif op == "quality":
        q = quality.quality_gate(img, CFG)
        g_ = CFG["quality"]["glare"]
        out = (
            quality.glare_mask(img, g_["v_min"], g_["s_max"], g_["smooth_sigma"], g_["min_area"]).astype(
                np.uint8
            )
            * 255
        )
        extras.append(
            {"label": "|Laplacian|", "src": _src(cv2.convertScaleAbs(cv2.Laplacian(g, cv2.CV_32F), alpha=4))}
        )
        info = {**q["scores"], "passed": q["passed"], "reasons": q["reasons"]}
    elif op == "hough":
        d = Debug()
        loc = document.locate(img, CFG, p.get("edge", "canny"), p.get("corner", "hough"), d)
        e = next(v for k, v in d.images.items() if k.startswith("edges"))
        h, *_ = hough.accumulator(e)
        acc = cv2.applyColorMap(
            cv2.normalize(np.log1p(h).astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            cv2.COLORMAP_MAGMA,
        )
        acc = cv2.resize(
            acc, (540, 360), interpolation=cv2.INTER_AREA
        )  # rho x theta is tall and thin; stretch theta
        extras = [{"label": k, "src": _src(v)} for k, v in d.images.items()] + [
            {"label": "rho-theta accumulator", "src": _src(acc)}
        ]
        out = document.corners.warp(img, loc["quad"], CFG) if loc["quad"] is not None else img
        info = {"corner_method": loc["corner_method"], **loc["validity"]}
    elif op == "document":
        d = Debug()
        r = document.process(img, CFG, debug=d)
        extras = [{"label": k, "src": _src(v)} for k, v in d.images.items()]
        out = r.get("card", img)
        info = {"fields": r["fields"], "ocr_conf": r["ocr_conf"], "validity": r["validity"]}
    elif op == "face":
        d = Debug()
        c = CFG if p.get("variant", "improved") == "improved" else with_overrides(CFG, face.BASELINE)
        r = face.represent(img, c, d)
        extras = [{"label": k, "src": _src(v)} for k, v in d.images.items()]
        out = d.images.get("detection", img)
        info = {
            "found": r["found"],
            "box": r.get("box"),
            "det_score": r.get("det_score"),
            "reason": r.get("reason"),
        }
    else:
        raise ValueError(f"unknown op {op}")
    return {
        "output": _src(out),
        "extras": extras,
        "info": info,
        "hist_in": enhance.histogram(img),
        "hist_out": enhance.histogram(out),
    }


@app.post("/api/lab")
def lab(image: UploadFile = File(...), op: str = Form(...), params: str = Form("{}")):
    img = _image(image, LAB_MAX_SIDE)
    try:
        p = json.loads(params)
        if not isinstance(p, dict):
            raise TypeError("params must be a JSON object")
        with LOCK:
            res = _lab(img, op, p)
    except (ValueError, KeyError, TypeError, cv2.error) as e:
        raise HTTPException(400, f"{op}: {e}") from None
    res["input"] = _src(img)
    return _clean(res)


@app.get("/api/experiments")
def experiments():
    from .eval.ablations import EXPERIMENTS

    res = path(CFG, "results")
    figs = res / "figures"
    entries = [
        (
            "face_summary",
            "Face verification: baseline vs improved",
            "EER / AUC / TAR on LFW and synthetic doc-vs-selfie pairs.",
        ),
        (
            "document_summary",
            "Document: baseline vs improved",
            "Quad IoU, corner error and field CER on synthetic + MIDV-2020.",
        ),
    ]
    entries += [(k, (fn.__doc__ or k).strip().split("\n")[0], "") for k, fn in EXPERIMENTS.items()]
    out = []
    for key, title, desc in entries:
        csv = res / f"{key}.csv"
        rows, cols = [], []
        if csv.exists():
            df = pd.read_csv(csv).drop(columns=["commit"], errors="ignore")
            df = df[[c for c in df.columns if not c.endswith(("_lo", "_hi"))]].head(60)
            cols, rows = list(df.columns), _clean(df.to_dict("records"))
        stem = key.removesuffix("_summary")
        fig_list = sorted(
            f"/results/figures/{p.name}"
            for p in figs.glob("*.png")
            if p.stem == stem or p.stem.startswith(stem + "_")
        )
        out.append(
            {
                "id": key,
                "title": title,
                "description": desc,
                "columns": cols,
                "rows": rows,
                "figures": fig_list,
                "available": csv.exists(),
            }
        )
    return out


@app.get("/api/samples")
def samples(n: int = 4):
    """A few genuine and impostor synthetic pairs for one-click demos (empty until make_degraded_pairs ran)."""
    m = path(CFG, "synthetic") / "pairs" / "manifest.csv"
    if not m.exists():
        return []
    df = pd.read_csv(m, dtype={"pair_id": str})
    pick = pd.concat([df[df.label == 1].head(n), df[df.label == 0].head(n)])
    return [
        {
            "pair_id": r.pair_id,
            "label": "genuine" if r.label == 1 else "impostor",
            "name": r.name_b,
            "selfie": f"/api/samples/{r.pair_id}/selfie",
            "id": f"/api/samples/{r.pair_id}/scene",
        }
        for r in pick.itertuples()
    ]


@app.get("/api/samples/{pair_id}/{kind}")
def sample_file(pair_id: str, kind: str):
    from fastapi.responses import FileResponse

    if not pair_id.isdigit() or kind not in ("selfie", "scene"):
        raise HTTPException(404)
    p = path(CFG, "synthetic") / "pairs" / f"{pair_id}_{kind}.jpg"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/jpeg")


path(CFG, "results").joinpath("figures").mkdir(parents=True, exist_ok=True)
app.mount("/results", StaticFiles(directory=path(CFG, "results")), name="results")
_dist = ROOT / "web" / "dist"
if _dist.exists():
    app.mount("/", StaticFiles(directory=_dist, html=True), name="web")
