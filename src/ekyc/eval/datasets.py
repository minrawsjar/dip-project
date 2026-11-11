"""Dataset indexes: LFW pairs (official 10-fold file), synthetic pairs manifest, own-capture liveness clips."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from ..io import path

log = logging.getLogger(__name__)
CLIP_EXT = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}


def lfw_pairs(cfg: dict, n: int | None = None) -> pd.DataFrame:
    """Rows: a, b (image paths), label (1 = same person), fold. `n` takes a balanced subset spread over folds."""
    home = path(cfg, "lfw_home")
    lines = (home / "pairs.txt").read_text().splitlines()
    folds, per = map(int, lines[0].split())
    img = lambda name, i: home / "lfw_funneled" / name / f"{name}_{int(i):04d}.jpg"
    rows, k = [], 1
    for f in range(folds):
        for _ in range(per):
            name, i, j = lines[k].split()
            rows.append(
                {"a": img(name, i), "b": img(name, j), "label": 1, "fold": f, "name_a": name, "name_b": name}
            )
            k += 1
        for _ in range(per):
            n1, i, n2, j = lines[k].split()
            rows.append({"a": img(n1, i), "b": img(n2, j), "label": 0, "fold": f, "name_a": n1, "name_b": n2})
            k += 1
    df = pd.DataFrame(rows)
    if n and n < len(df):
        q = max(1, n // (2 * folds))
        df = df.groupby(["fold", "label"], group_keys=False).head(q).reset_index(drop=True)
    return df


def synthetic_pairs(cfg: dict) -> pd.DataFrame:
    p = path(cfg, "synthetic") / "pairs" / "manifest.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: run `python scripts/make_degraded_pairs.py` first")
    df = pd.read_csv(p, dtype={"pair_id": str})
    root = p.parent
    for c in ("doc_portrait", "selfie", "scene", "card"):
        df[c] = df[c].map(lambda s: root / s)
    for c in ("quad", "fields"):
        df[c] = df[c].map(json.loads)
    return df


def own_capture(cfg: dict) -> pd.DataFrame:
    """Index `data/own_capture/<subject>/{live,print,replay}/*` clips. Empty frame if none recorded yet."""
    root = path(cfg, "own_capture")
    rows = []
    for clip in sorted(root.glob("*/*/*")) if root.exists() else []:
        kind = clip.parent.name
        if clip.suffix.lower() in CLIP_EXT and kind in ("live", "print", "replay"):
            rows.append(
                {"subject": clip.parent.parent.name, "kind": kind, "label": int(kind == "live"), "path": clip}
            )
    return pd.DataFrame(rows, columns=["subject", "kind", "label", "path"])


def subject_split(df: pd.DataFrame, seed: int, frac=(0.6, 0.2, 0.2)) -> pd.DataFrame:
    """Subject-disjoint train/val/test assignment (CLAUDE.md §5)."""
    import numpy as np

    subs = np.array(sorted(df["subject"].unique()))
    np.random.default_rng(seed).shuffle(subs)
    n = len(subs)
    a, b = round(frac[0] * n), round((frac[0] + frac[1]) * n)
    split = {s: "train" if i < a else "val" if i < b else "test" for i, s in enumerate(subs)}
    return df.assign(split=df["subject"].map(split))


def midv_root(cfg: dict) -> Path | None:
    p = path(cfg, "midv")
    return (
        p if (p / "photo" / "annotations").exists() and (p / "templates" / "annotations").exists() else None
    )
