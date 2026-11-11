"""Face verification: baseline (Haar + landmark distances) vs improved (YuNet + align + HSV-V stretch + SFace)
on LFW pairs and synthetic doc-portrait-vs-selfie pairs.

Failure to acquire (no face) is scored as a reject (below every real score) and reported as FTA.
Outputs: results/face_scores.csv, results/face_summary.csv, results/figures/face_roc.png,
results/figures/face_intermediates.png.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from .. import face, metrics, viz
from ..io import Debug, imread, load_config, path, with_overrides, write_results
from .datasets import lfw_pairs, synthetic_pairs

log = logging.getLogger(__name__)

VARIANTS = {
    "baseline": face.BASELINE,
    "improved": {},
}


def represent_many(paths, cfg: dict, transform: Callable | None = None, desc: str = "") -> dict:
    out = {}
    for p in tqdm(sorted(set(paths)), desc=desc, leave=False):
        img = imread(p)
        if transform is not None:
            img = transform(img)
        out[p] = face.represent(img, cfg)
    return out


def score_pairs(
    a_paths,
    b_paths,
    cfg: dict,
    transform_a: Callable | None = None,
    b_cache: dict | None = None,
    desc: str = "",
) -> np.ndarray:
    """Similarity per pair (NaN = failure to acquire). `transform_a` degrades the A (document) side only."""
    ra = represent_many(a_paths, cfg, transform_a, desc + " A")
    rb = b_cache if b_cache is not None else represent_many(b_paths, cfg, None, desc + " B")
    s = [face.score(ra[a], rb[b], cfg) for a, b in zip(a_paths, b_paths)]
    return np.array([np.nan if v is None else v for v in s], float)


def fill_fta(scores: np.ndarray) -> np.ndarray:
    ok = ~np.isnan(scores)
    low = (np.nanmin(scores) - 1) if ok.any() else -1.0
    return np.where(ok, scores, low)


def summarize(scores: np.ndarray, labels: np.ndarray, cfg: dict) -> dict:
    s, y = fill_fta(scores), np.asarray(labels)
    e, lo, hi = metrics.bootstrap_ci(
        lambda a, b: metrics.eer(a, b)[0], s, y, n=cfg["eval"]["bootstrap"], seed=cfg["seed"]
    )
    fpr, tpr, _ = metrics.roc(s, y)
    return {
        "EER": e,
        "EER_lo": lo,
        "EER_hi": hi,
        "EER_threshold": metrics.eer(s, y)[1],
        "AUC": metrics.auc(fpr, tpr),
        "TAR@FAR=1e-2": metrics.tar_at_far(s, y, 1e-2),
        "TAR@FAR=1e-3": metrics.tar_at_far(s, y, 1e-3),
        "FTA": float(np.isnan(scores).mean()),
        "n": len(y),
    }


def intermediates_figure(cfg: dict, a, b, out) -> None:
    """One pair through the improved and baseline face paths, every intermediate image in one panel."""
    imgs = {}
    for name, ov in VARIANTS.items():
        c = with_overrides(cfg, ov)
        for side, p in (("doc", a), ("selfie", b)):
            d = Debug()
            face.represent(imread(p), c, d)
            imgs.update({f"{name}/{side}/{k}": v for k, v in d.images.items()})
    viz.panel(imgs, out, cols=4, title="Face module intermediates (baseline vs improved)")


def main(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    sets = {"lfw": lfw_pairs(cfg, cfg["eval"]["lfw_pairs"])}
    try:
        syn = synthetic_pairs(cfg)
        sets["synthetic_doc"] = pd.DataFrame(
            {"a": syn["doc_portrait"], "b": syn["selfie"], "label": syn["label"], "fold": syn["fold"]}
        )
    except FileNotFoundError as e:
        log.warning("%s", e)
    rows, summary, curves, thresholds = [], [], {}, []
    for ds, df in sets.items():
        for vname, ov in VARIANTS.items():
            c = with_overrides(cfg, ov)
            s = score_pairs(list(df["a"]), list(df["b"]), c, desc=f"{ds}/{vname}")
            rows += [
                {"dataset": ds, "variant": vname, "a": str(a), "b": str(b), "label": int(y), "score": v}
                for a, b, y, v in zip(df["a"], df["b"], df["label"], s)
            ]
            summary.append({"dataset": ds, "variant": vname, **summarize(s, df["label"].values, cfg)})
            fpr, tpr, _ = metrics.roc(fill_fta(s), df["label"].values)
            curves[f"{ds} / {vname}"] = (fpr, tpr)
            log.info("%s %s EER=%.3f", ds, vname, summary[-1]["EER"])
            if "fold" in df:  # operating threshold = EER point on val folds 0-6 only (folds 7-9 are test)
                val = (df["fold"] <= 6).values
                e, thr = metrics.eer(fill_fta(s[val]), df["label"].values[val])
                key = "geometric_threshold" if c["face"]["descriptor"] == "geometric" else "match_threshold"
                thresholds.append(
                    {
                        "dataset": ds,
                        "variant": vname,
                        "config_key": f"face.{key}",
                        "val_EER": e,
                        "threshold": thr,
                    }
                )
    write_results(pd.DataFrame(rows), "face_scores", cfg)
    out = pd.DataFrame(summary)
    write_results(out, "face_summary", cfg)
    if thresholds:
        t = pd.DataFrame(thresholds)
        write_results(t, "face_thresholds", cfg)
        for r in t.itertuples():
            log.info(
                "val-tuned %s = %.3f (configs/default.yaml has %s)",
                r.config_key,
                r.threshold,
                cfg["face"][r.config_key.split(".")[1]],
            )
    figs = path(cfg, "results") / "figures"
    viz.roc(curves, figs / "face_roc.png", "Face verification ROC")
    last = sets.get("synthetic_doc", sets["lfw"])
    g = last[last["label"] == 1].iloc[0]
    intermediates_figure(cfg, g["a"], g["b"], figs / "face_intermediates.png")
    print(out.round(4).to_string(index=False))
    return out
