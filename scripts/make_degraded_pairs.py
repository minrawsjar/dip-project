"""Synthetic doc-portrait vs selfie pairs (CLAUDE.md §6).

For each LFW pair (A, B): A is printed into the portrait slot of a mock ID-1 card, which is "photographed"
(halftone print -> perspective onto a desk -> glare -> low light -> blur -> JPEG). The portrait is cropped
back out through the ground-truth homography and paired with B as the selfie. Genuine pairs come from
LFW matched pairs (same identity), impostors from mismatched pairs. Every parameter is logged to
data/synthetic/pairs/manifest.csv; the whole set is reproducible from the config seed.

Usage: python scripts/make_degraded_pairs.py [--n 600]
"""

import argparse
import json
import logging
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm

from ekyc.document import corners, template
from ekyc.eval.datasets import lfw_pairs
from ekyc.io import imread, imwrite, load_config, path

logging.basicConfig(level=logging.INFO)
ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=None)
args = ap.parse_args()

cfg = load_config()
n = args.n or cfg["eval"]["synthetic_pairs"]
out = path(cfg, "synthetic") / "pairs"
if out.exists():
    shutil.rmtree(out)
out.mkdir(parents=True)

pairs = lfw_pairs(cfg, n)
rows = []
for i, r in tqdm(pairs.iterrows(), total=len(pairs)):
    seed = cfg["seed"] + i
    rng = np.random.default_rng(seed)
    values = template.random_identity(rng)
    card, meta = template.render_card(imread(r.a), values, cfg, seed=seed)
    severity = float(rng.uniform(0.5, 1.5))
    scene, quad, log = template.capture(card, seed=seed, severity=severity)
    px, py, pw, ph = meta["portrait"]
    portrait = corners.warp(scene, quad, cfg)[py : py + ph, px : px + pw]
    pid = f"{i:05d}"
    imwrite(out / f"{pid}_scene.jpg", scene)
    imwrite(out / f"{pid}_card.jpg", card)
    imwrite(out / f"{pid}_doc_portrait.png", portrait)
    shutil.copy(r.b, out / f"{pid}_selfie.jpg")
    rows.append(
        {
            "pair_id": pid,
            "label": r.label,
            "fold": r.fold,
            "name_a": r.name_a,
            "name_b": r.name_b,
            "seed": seed,
            "severity": round(severity, 3),
            "scene": f"{pid}_scene.jpg",
            "card": f"{pid}_card.jpg",
            "doc_portrait": f"{pid}_doc_portrait.png",
            "selfie": f"{pid}_selfie.jpg",
            "quad": json.dumps(quad.round(2).tolist()),
            "fields": json.dumps(meta["fields"]),
            "portrait_rect": json.dumps(meta["portrait"]),
            "degradations": json.dumps(log),
        }
    )
pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
print(f"wrote {len(rows)} pairs to {out}")
