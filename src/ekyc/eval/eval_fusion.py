"""Fusion (E14): fixed per-module thresholds with AND logic vs logistic regression.

There is no dataset with matched selfie + ID + liveness clip for the same attempt, so attempts are
assembled assuming module independence (standard practice for multimodal fusion without joint data):
each synthetic pair (face + document scores) is combined with a liveness score drawn from bona fide
clips, plus spoof attempts (genuine face + attack clip). Without own-capture liveness results, the
liveness score is fixed at 1 and the experiment covers face + document + quality only (stated in output).
Train on LFW folds 0-6, test on folds 7-9. Writes results/E14.csv, figure, models/fusion_lr.joblib.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .. import fusion, metrics, quality, viz
from ..io import imread, load_config, path, write_results
from .datasets import synthetic_pairs

log = logging.getLogger(__name__)


def attempts(cfg: dict) -> tuple[pd.DataFrame, str]:
    res = path(cfg, "results")
    face = pd.read_csv(res / "face_scores.csv")
    face = face[(face.dataset == "synthetic_doc") & (face.variant == "improved")].reset_index(drop=True)
    doc = pd.read_csv(res / "document_scores.csv")
    doc = doc[(doc.source == "synthetic") & (doc.variant == "improved")]
    syn = synthetic_pairs(cfg)
    syn["face_score"] = face["score"].fillna(face["score"].min() - 1).values
    syn = syn.merge(
        doc[["id", "cer"]].assign(id=lambda d: d["id"].astype(int).map("{:05d}".format)),
        left_on="pair_id",
        right_on="id",
        how="left",
    )
    q = []
    for r in syn.itertuples():
        sel, scn = imread(r.selfie), imread(r.scene)
        q.append(
            {
                "selfie_blur": quality.blur_score(sel),
                "id_blur": quality.blur_score(scn),
                "id_glare": quality.quality_gate(scn, cfg, roi=quality.quad_mask(scn.shape, r.quad))[
                    "scores"
                ]["glare_ratio"],
            }
        )
    syn = pd.concat([syn, pd.DataFrame(q)], axis=1)
    syn["ocr_conf"] = (1 - syn["cer"].fillna(1).clip(0, 1)) * 100  # readable-document proxy from field CER

    rng = np.random.default_rng(cfg["seed"])
    lv = res / "liveness_features.csv"
    if lv.exists():
        L = pd.read_csv(lv)
        col = "handcrafted fused: LR" if "handcrafted fused: LR" in L else "rules_live"
        live, attack = L.loc[L.label == 1, col].values, L.loc[L.label == 0, col].values
        a = syn.assign(
            liveness_score=rng.choice(live, len(syn)), kind=np.where(syn.label == 1, "genuine", "impostor")
        )
        g = syn[syn.label == 1]
        b = g.assign(liveness_score=rng.choice(attack, len(g)), kind="spoof")
        df, note = pd.concat([a, b], ignore_index=True), f"liveness from own-capture ({col})"
    else:
        df = syn.assign(liveness_score=1.0, kind=np.where(syn.label == 1, "genuine", "impostor"))
        note = "NO liveness data: liveness fixed at 1, fusion covers face + document + quality only"
    df["target"] = (df["kind"] == "genuine").astype(int)
    return df, note


def main(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    df, note = attempts(cfg)
    print(note)
    train, test = df[df.fold <= 6], df[df.fold > 6]
    X = lambda d: np.vstack([fusion.vector(r) for r in d[fusion.FEATURES].to_dict("records")])
    model = fusion.train(X(train), train["target"].values, cfg["seed"])
    import joblib

    p = fusion.model_file(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, p)
    prob = model.predict_proba(X(test))[:, 1]
    fixed = np.array(
        [
            fusion.fixed({**r, "quality_passed": True}, cfg)["decision"] == "ACCEPT"
            for r in test.to_dict("records")
        ]
    )
    y = test["target"].values
    far_fixed, frr_fixed = float(fixed[y == 0].mean()), float((~fixed[y == 1]).mean())
    e, _ = metrics.eer(prob, y)
    lr_acc = prob >= cfg["fusion"]["accept"]
    rows = [
        {"method": "fixed AND thresholds", "FAR": far_fixed, "FRR": frr_fixed, "EER": np.nan, "AUC": np.nan},
        {
            "method": "logistic regression @ accept",
            "FAR": float(lr_acc[y == 0].mean()),
            "FRR": float((~lr_acc[y == 1]).mean()),
            "EER": e,
            "AUC": metrics.auc(*metrics.roc(prob, y)[:2]),
        },
        {  # same false-accept budget as the fixed rule: the fair comparison
            "method": "logistic regression @ fixed-rule FAR",
            "FAR": far_fixed,
            "FRR": 1 - metrics.tar_at_far(prob, y, far_fixed),
            "EER": e,
            "AUC": metrics.auc(*metrics.roc(prob, y)[:2]),
        },
    ]
    out = pd.DataFrame(rows).assign(note=note, n_test=len(test))
    write_results(out, "E14", cfg)
    fpr, tpr, _ = metrics.roc(prob, y)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    ax.plot(fpr, tpr, label=f"logistic regression (EER {e:.3f})")
    ax.plot([far_fixed], [1 - frr_fixed], "rs", label="fixed AND thresholds")
    ax.set_xlabel("FAR")
    ax.set_ylabel("TAR")
    ax.set_title("E14 fusion ROC")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    viz.save(fig, path(cfg, "results") / "figures" / "E14.png")
    print(out.round(4).to_string(index=False))
    return out
