"""Experiments E1-E14 (CLAUDE.md §8). Each writes results/<E>.csv and results/figures/<E>.png.

Face experiments degrade only the document side (A) of LFW pairs and keep the selfie (B) clean, which
mirrors the deployment setting: the ID portrait is the degraded capture. Document OCR experiments
rectify with the ground-truth quad so that only the studied stage varies.
"""

from __future__ import annotations

import logging
from functools import partial

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from .. import degrade, enhance, face, filters, metrics, quality, viz
from ..document import corners, edges, hough, locate, ocr, portrait
from ..io import Debug, imread, load_config, path, with_overrides, write_results
from . import eval_document, eval_face, eval_fusion, eval_liveness
from .datasets import lfw_pairs, synthetic_pairs

log = logging.getLogger(__name__)


def _fig(cfg, exp):
    return path(cfg, "results") / "figures" / f"{exp}.png"


def _face_sweep(cfg, conditions: list[tuple[dict, callable, dict]], n=None) -> pd.DataFrame:
    """conditions: (row labels, transform for side A, cfg overrides). B embeddings cached per override set."""
    pairs = lfw_pairs(cfg, n or cfg["eval"]["lfw_pairs"])
    a, b, y = list(pairs["a"]), list(pairs["b"]), pairs["label"].values
    cache, rows = {}, []
    for labels, tf, ov in tqdm(conditions, desc="conditions"):
        c = with_overrides(cfg, ov)
        key = repr(sorted(ov.items()))
        if key not in cache:
            cache[key] = eval_face.represent_many(b, c, desc="B")
        s = eval_face.score_pairs(a, b, c, tf, cache[key])
        rows.append({**labels, **eval_face.summarize(s, y, cfg)})
    return pd.DataFrame(rows)


def _ocr_items(cfg, n=120):
    """(GT-rectified card, fields) from synthetic captures; OCR experiments vary only the OCR chain."""
    df = synthetic_pairs(cfg).head(n)
    size = corners.canonical_size(cfg)
    return [(corners.warp(imread(r.scene), np.float32(r.quad), cfg, size), r.fields) for r in df.itertuples()]


def _ocr_cer(items, cfg, tf=None) -> tuple[float, float]:
    cers = []
    for card, fields in items:
        c = tf(card) if tf else card
        cers += [eval_document.field_cer(c, {k: f}, cfg)["cer"] for k, f in fields.items()]
    v = np.array(cers)
    _, lo, hi = metrics.bootstrap_ci(np.mean, v, n=300, seed=cfg["seed"])
    return float(v.mean()), lo, hi


# ----------------------------------------------------------------------------- face degradations


def e1(cfg):
    """Face EER vs bit depth of the document portrait."""
    conds = [
        ({"bits": b}, partial(lambda im, b: degrade.quantize_bits(im, b)[0], b=b), {})
        for b in cfg["degrade"]["bits"]
    ]
    df = _face_sweep(cfg, conds)
    viz.lines(df, "bits", "EER", None, _fig(cfg, "E1"), "E1: face EER vs bit depth", invert_x=True)
    sample = imread(lfw_pairs(cfg, 2)["a"][0])
    viz.panel(
        {f"{b} bit": degrade.quantize_bits(sample, b)[0] for b in cfg["degrade"]["bits"]},
        _fig(cfg, "E1_samples"),
        cols=6,
        title="Quantization / false contouring",
    )
    return df


def e2(cfg):
    """Face EER vs portrait resolution x interpolation used to bring it back up."""
    conds = [
        ({"scale": s, "interp": i}, partial(lambda im, s, i: degrade.downsample(im, s, i)[0], s=s, i=i), {})
        for s in cfg["degrade"]["scales"]
        for i in cfg["degrade"]["interps"]
    ]
    df = _face_sweep(cfg, conds)
    viz.lines(
        df, "scale", "EER", "interp", _fig(cfg, "E2"), "E2: face EER vs resolution x interpolation", logx=True
    )
    sample = imread(lfw_pairs(cfg, 2)["a"][0])
    viz.panel(
        {i: degrade.downsample(sample, 0.125, i)[0] for i in cfg["degrade"]["interps"]},
        _fig(cfg, "E2_samples"),
        cols=4,
        title="1/8 resolution restored with each kernel",
    )
    return df


def e3(cfg):
    """Face EER and OCR CER vs JPEG quality."""
    conds = [
        ({"quality": q}, partial(lambda im, q: degrade.jpeg(im, q)[0], q=q), {})
        for q in cfg["degrade"]["jpeg"]
    ]
    df = _face_sweep(cfg, conds)
    items = _ocr_items(cfg, 80)
    cer = [
        _ocr_cer(items, cfg, partial(lambda im, q: degrade.jpeg(im, q)[0], q=q))
        for q in cfg["degrade"]["jpeg"]
    ]
    df["CER"], df["CER_lo"], df["CER_hi"] = zip(*cer)
    long = pd.concat(
        [df.assign(metric="face EER", value=df["EER"]), df.assign(metric="OCR CER", value=df["CER"])]
    )
    viz.lines(long, "quality", "value", "metric", _fig(cfg, "E3"), "E3: JPEG quality", invert_x=True)
    return df


def e4(cfg):
    """Enhancement method vs face EER (low-light portraits) and OCR CER (low-light cards)."""
    dark = lambda im: degrade.gamma(degrade.brightness_contrast(im, 0.6, -20)[0], 1.6)[0]
    methods = ["none", "stretch", "gamma", "histeq", "clahe"]
    conds = [
        ({"method": m}, dark, {"face.photometric": f"{m}_v" if m != "none" else "none"}) for m in methods
    ]
    df = _face_sweep(cfg, conds)
    items = _ocr_items(cfg, 80)
    cer = [_ocr_cer(items, with_overrides(cfg, {"document.ocr.enhance": m}), dark) for m in methods]
    df["CER"], df["CER_lo"], df["CER_hi"] = zip(*cer)
    viz.bars(
        df.melt(id_vars="method", value_vars=["EER", "CER"]),
        "method",
        "value",
        _fig(cfg, "E4"),
        "E4: enhancement under low light",
        hue="variable",
    )
    sample = dark(imread(lfw_pairs(cfg, 2)["a"][0]))
    viz.panel(
        {m: enhance.apply(sample, m, "hsv_v", cfg) for m in methods},
        _fig(cfg, "E4_panel"),
        cols=5,
        hist=True,
        title="Low-light portrait: enhancement + histograms",
    )
    return df


def e5(cfg):
    """Colour preservation: per-RGB-channel vs HSV-V equalisation (hue shift + face EER)."""
    pairs = lfw_pairs(cfg, 200)
    rows = []
    for mode, ph in [("rgb", "histeq_rgb"), ("hsv_v", "histeq_v")]:
        shifts = [
            enhance.hue_shift(im, enhance.apply(im, "histeq", mode)) for im in map(imread, pairs["a"][:200])
        ]
        rows.append({"mode": mode, "hue_shift_deg": float(np.mean(shifts)), "photometric": ph})
    df = pd.DataFrame(rows)
    ev = _face_sweep(
        cfg,
        [
            ({"photometric": r.photometric}, None, {"face.photometric": r.photometric})
            for r in df.itertuples()
        ],
    )
    df = df.merge(ev, on="photometric")
    im = imread(pairs["a"][0])
    viz.panel(
        {
            "original": im,
            "histeq per RGB channel": enhance.apply(im, "histeq", "rgb"),
            "histeq on HSV V": enhance.apply(im, "histeq", "hsv_v"),
        },
        _fig(cfg, "E5"),
        cols=3,
        hist=True,
        title="E5: colour preservation",
    )
    return df


# ----------------------------------------------------------------------------- document


def e6(cfg):
    """Edge operator (Sobel / Prewitt / Canny) for card localisation: quad IoU."""
    its = eval_document.items(cfg, n_synth=200, n_midv=100)
    rows = [
        eval_document.run_item(it, cfg, e, "hough", do_ocr=False)
        for it in tqdm(its)
        for e in ("sobel", "prewitt", "canny")
    ]
    df = eval_document.summarize(pd.DataFrame(rows), ["source", "edge"], cfg)
    viz.bars(df, "edge", "iou", _fig(cfg, "E6"), "E6: quad IoU by edge operator", hue="source")
    g = cv2.GaussianBlur(cv2.cvtColor(imread(its[0]["image"]), cv2.COLOR_BGR2GRAY), (5, 5), 0)
    viz.panel(
        {e: edges.detect(g, e, cfg) for e in ("sobel", "prewitt", "canny")},
        _fig(cfg, "E6_edges"),
        cols=3,
        title="Edge maps",
    )
    return df


def e7(cfg):
    """Corner localisation: Hough line intersections vs contour vs contour + Harris snap."""
    its = eval_document.items(cfg, n_synth=200, n_midv=100)
    rows = [
        eval_document.run_item(it, cfg, "canny", c, do_ocr=False)
        for it in tqdm(its)
        for c in ("hough", "contour", "harris")
    ]
    df = eval_document.summarize(pd.DataFrame(rows), ["source", "corner"], cfg)
    viz.bars(
        df, "corner", "corner_err", _fig(cfg, "E7"), "E7: mean corner error (canonical px)", hue="source"
    )
    img = imread(its[0]["image"])
    small = cv2.resize(
        img,
        None,
        fx=cfg["document"]["work_width"] / img.shape[1],
        fy=cfg["document"]["work_width"] / img.shape[1],
    )
    e = edges.canny(
        cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0), *cfg["document"]["canny"]
    )
    h, th, rho, peaks = hough.accumulator(e)
    viz.accumulator(h, th, rho, peaks, small, _fig(cfg, "E7_accumulator"))
    return df


def e8(cfg):
    """Denoising filter vs OCR CER under Gaussian and salt-and-pepper noise."""
    items = _ocr_items(cfg, 60)
    rows = []
    for kind, level in [("gaussian", 0.08), ("salt_pepper", 0.05)]:
        tf = partial(lambda im, k, lv: degrade.add_noise(im, k, lv, seed=cfg["seed"])[0], k=kind, lv=level)
        for f in ("none", "mean", "gaussian", "median"):
            c = with_overrides(cfg, {"document.ocr.denoise": f})
            m, lo, hi = _ocr_cer(items, c, tf)
            rows.append({"noise": kind, "level": level, "filter": f, "CER": m, "CER_lo": lo, "CER_hi": hi})
    df = pd.DataFrame(rows)
    viz.bars(df, "filter", "CER", _fig(cfg, "E8"), "E8: OCR CER by denoising filter", hue="noise")
    roi = degrade.add_noise(items[0][0], "salt_pepper", 0.05, seed=cfg["seed"])[0][100:220, 300:820]
    viz.panel(
        {f: filters.DENOISE[f](roi, 3) for f in ("none", "mean", "gaussian", "median")},
        _fig(cfg, "E8_panel"),
        cols=2,
        title="Salt-and-pepper field ROI",
    )
    return df


def e9(cfg):
    """Global (Otsu) vs adaptive thresholding, clean and under uneven illumination (glare + vignetting)."""
    items = _ocr_items(cfg, 60)

    def uneven(im):
        w = im.shape[1]
        ramp = (0.45 + 0.55 * np.arange(w) / w)[None, :, None]
        return degrade.glare((im * ramp).astype(np.uint8), (0.75, 0.4), 0.25, 0.7, cfg["seed"])[0]

    rows = []
    for cond, tf in [("clean", None), ("uneven light", uneven)]:
        for t in ("otsu", "adaptive_mean", "adaptive_gaussian"):
            m, lo, hi = _ocr_cer(items, with_overrides(cfg, {"document.ocr.threshold": t}), tf)
            rows.append({"condition": cond, "threshold": t, "CER": m, "CER_lo": lo, "CER_hi": hi})
    df = pd.DataFrame(rows)
    viz.bars(df, "threshold", "CER", _fig(cfg, "E9"), "E9: OCR CER by thresholding", hue="condition")
    roi = uneven(items[0][0])[95:160, 300:820]
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    o = cfg["document"]["ocr"]
    viz.panel(
        {
            "input": roi,
            **{
                t: ocr.threshold(g, t, o["adaptive_block"], o["adaptive_c"])
                for t in ("otsu", "adaptive_mean", "adaptive_gaussian")
            },
        },
        _fig(cfg, "E9_panel"),
        cols=2,
        title="Binarised field under uneven light",
    )
    return df


# ----------------------------------------------------------------------------- liveness


def e10(cfg):
    out = eval_liveness.main(cfg)
    return None if out is None else pd.read_csv(path(cfg, "results") / "E10.csv")


def e11(cfg):
    return eval_liveness.main(cfg)


# ----------------------------------------------------------------------------- headline / system


def e12(cfg):
    """Face headline: geometric baseline vs SFace with/without alignment and enhancement, LFW + synthetic."""
    variants = {
        "haar + geometric (baseline)": face.BASELINE,
        "yunet + geometric": {"face.descriptor": "geometric"},
        "sface, no align, no enh": {"face.align": False, "face.photometric": "none"},
        "sface + align": {"face.photometric": "none"},
        "sface + align + CLAHE-V": {"face.photometric": "clahe_v"},
        "sface + align + stretch-V, no descreen": {"face.descreen_sigma": 0},
        "sface + align + stretch-V + descreen (default)": {},
    }
    df = _face_sweep(cfg, [({"variant": k, "dataset": "lfw"}, None, v) for k, v in variants.items()])
    syn = synthetic_pairs(cfg)
    rows = []
    for k, v in variants.items():
        c = with_overrides(cfg, v)
        s = eval_face.score_pairs(list(syn["doc_portrait"]), list(syn["selfie"]), c)
        rows.append(
            {"variant": k, "dataset": "synthetic_doc", **eval_face.summarize(s, syn["label"].values, cfg)}
        )
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    viz.bars(df, "variant", "EER", _fig(cfg, "E12"), "E12: face verification EER", hue="dataset")
    return df


def e13(cfg):
    """Quality gate on/off: FRR of genuine pairs and OCR CER when half the captures are badly degraded."""
    syn = synthetic_pairs(cfg).head(300)
    rng = np.random.default_rng(cfg["seed"])
    rows = []
    for r in tqdm(syn.itertuples(), total=len(syn)):
        sel, scn = imread(r.selfie), imread(r.scene)
        bad = rng.random() < 0.5
        if bad:
            kind = rng.choice(["blur", "dark", "glare"])
            if kind == "blur":
                sel, scn = degrade.gaussian_blur(sel, 3.0)[0], degrade.gaussian_blur(scn, 4.0)[0]
            elif kind == "dark":
                sel = degrade.brightness_contrast(sel, 0.25, -10)[0]
                scn = degrade.brightness_contrast(scn, 0.25, -10)[0]
            else:
                scn = degrade.glare(scn, None, 0.3, 1.0, int(rng.integers(1 << 30)))[0]
        else:
            kind = "none"
        f = face.detect.detect(sel, cfg, "yunet")
        q_sel = quality.quality_gate(sel, cfg, roi=quality.box_mask(sel.shape, f["box"]) if f else None)
        q_id = quality.quality_gate(scn, cfg, roi=quality.quad_mask(scn.shape, r.quad))
        rs = face.represent(sel, cfg)
        loc = locate(scn, cfg)
        if loc["quad"] is not None:
            card = corners.warp(scn, loc["quad"], cfg)
            p, _ = portrait(card, cfg)
            rp = face.represent(p, cfg) if p is not None else {"found": False}
            cer = eval_document.field_cer(card, r.fields, cfg)["cer"]
        else:
            rp, cer = {"found": False}, 1.0
        s = face.score(rs, rp, cfg)
        rows.append(
            {
                "degraded": kind,
                "label": r.label,
                "gate_pass": q_sel["passed"] and q_id["passed"],
                "score": np.nan if s is None else s,
                "cer": cer,
            }
        )
    d = pd.DataFrame(rows)
    thr = cfg["face"]["match_threshold"]
    out = []
    for gate in (False, True):
        g = d[d.gate_pass] if gate else d
        gen = g[g.label == 1]
        out.append(
            {
                "quality_gate": "on" if gate else "off",
                "processed_frac": len(g) / len(d),
                "FRR": float((gen["score"].fillna(-1) < thr).mean()),
                "CER": float(g["cer"].mean()),
                "retake_rate": 1 - len(g) / len(d),
            }
        )
    df = pd.DataFrame(out)
    write_results(d, "E13_items", cfg)
    viz.bars(
        df.melt(id_vars="quality_gate", value_vars=["FRR", "CER"]),
        "quality_gate",
        "value",
        _fig(cfg, "E13"),
        "E13: with vs without quality gate",
        hue="variable",
    )
    return df


def e14(cfg):
    return eval_fusion.main(cfg)


def failures(cfg, k: int = 6):
    """Failure-case galleries: worst face pairs and worst document localisations, with intermediates."""
    from ..document import process

    res = path(cfg, "results")
    out = []
    f = pd.read_csv(res / "face_scores.csv")
    f = f[(f.dataset == "synthetic_doc") & (f.variant == "improved")]
    worst = pd.concat(
        [
            f[f.label == 1].sort_values("score", na_position="first").head(k).assign(kind="false reject"),
            f[f.label == 0].sort_values("score", ascending=False).head(k).assign(kind="false accept"),
        ]
    )
    imgs = {}
    for r in worst.itertuples():
        tag = f"{r.kind} s={r.score:.2f}" if r.score == r.score else f"{r.kind} (no face)"
        imgs[f"{tag} #{len(imgs) // 2} doc"] = imread(r.a)
        imgs[f"{tag} #{len(imgs) // 2} selfie"] = imread(r.b)
    viz.panel(
        imgs,
        _fig(cfg, "FAILURES_face"),
        cols=4,
        title="Face: worst genuine (false reject) and impostor pairs",
    )
    out += [{"module": "face", "kind": r.kind, "item": r.a, "score": r.score} for r in worst.itertuples()]

    d = pd.read_csv(res / "document_scores.csv", dtype={"id": str})
    d = d[(d.variant == "improved") & (d.source == "synthetic")].sort_values("iou").head(k)
    syn = synthetic_pairs(cfg).set_index("pair_id")
    imgs = {}
    for r in d.itertuples():
        pid = r.id.zfill(5)
        dbg_ = Debug()
        process(imread(syn.loc[pid, "scene"]), cfg, debug=dbg_)
        imgs[f"#{pid} IoU={r.iou:.2f} hough"] = dbg_.images.get("hough_lines", dbg_.images.get("quad"))
        imgs[f"#{pid} region"] = dbg_.images["region"]
        imgs[f"#{pid} warped"] = dbg_.images.get("warped", np.zeros((10, 10), np.uint8))
        out.append({"module": "document", "kind": "worst IoU", "item": pid, "score": r.iou})
    viz.panel(imgs, _fig(cfg, "FAILURES_document"), cols=3, title="Document: worst localisations")
    return pd.DataFrame(out)


EXPERIMENTS = {
    **{f"E{i}": fn for i, fn in enumerate([e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14], 1)},
    "FAILURES": failures,
}


def run(exp: str, cfg: dict | None = None) -> pd.DataFrame | None:
    cfg = cfg or load_config()
    exp = exp.upper()
    df = EXPERIMENTS[exp](cfg)
    if df is not None and exp not in ("E10", "E11", "E14"):  # those write their own CSVs
        write_results(df, exp, cfg)
    if df is not None:
        cols = [c for c in df.columns if not c.endswith(("_lo", "_hi")) and c != "commit"]
        print(df[cols].round(4).to_string(index=False))
    return df
