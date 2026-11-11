"""Liveness evaluation on own-capture clips (subject-disjoint splits; ISO/IEC 30107-3 metrics).

Signals compared (E11): motion ratio, EAR blink, man-made cue rules, man-made cue LR/SVM, CNN, fused
(handcrafted + CNN embedding). Decision thresholds are fitted on val (EER point), reported on test.
E10: motion-signal APCER/BPCER vs frame-difference threshold.
Writes results/liveness_features.csv, results/E10.csv, results/E11.csv and models/liveness_lr.joblib.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

from .. import metrics, viz
from ..face import detect, landmarks
from ..io import load_config, path, read_video, write_results
from ..liveness import FEATURES, blink, manmade, motion
from .datasets import own_capture, subject_split

log = logging.getLogger(__name__)
HOWTO = """No liveness clips found under data/own_capture/<subject_id>/{live,print,replay}/*.mp4.
Record >= 10 consented subjects (5-10 s each, varied lighting), then rerun. See README "Liveness data"."""


def clip_features(frames, cfg: dict, sweep: list[int]) -> dict:
    """All handcrafted features for one clip; motion features recomputed per threshold in `sweep`."""
    lc = cfg["liveness"]
    meshes = [landmarks.mesh(f, cfg) for f in frames]
    mid = len(frames) // 2
    f = detect.detect(frames[mid], cfg)
    row = manmade.features(frames[mid], f["box"], cfg) if f else {k: 0.0 for k in manmade.FEATURES}
    row["rules_live"] = float(manmade.rule_classify(row, cfg)[0]) if f else 0.0
    row.update(blink.blinks(blink.series(meshes), lc["ear_ratio"], lc["ear_consec"]))
    for t in sweep:
        m = motion.features(frames, meshes, t, cfg)
        if t == lc["diff_threshold"]:
            row.update(m)
        row[f"motion_ratio@{t}"] = m["motion_ratio"]
    return row


def face_crops(frames, cfg: dict, every: int = 10) -> list[np.ndarray]:
    from ..face.align import crop_box

    out = []
    for fr in frames[::every]:
        f = detect.detect(fr, cfg)
        if f:
            out.append(crop_box(fr, f["box"], size=224, margin=0.25))
    return out


def _pad(scores, labels, kinds, thr) -> dict:
    return metrics.pad_metrics(scores, labels, thr, np.where(np.asarray(labels) == 1, "bona_fide", kinds))


def evaluate_signal(name: str, val: pd.DataFrame, test: pd.DataFrame, col: str) -> dict:
    _, thr = metrics.eer(val[col], val["label"])
    e, _ = metrics.eer(test[col], test["label"])
    return {
        "signal": name,
        "threshold": thr,
        "EER_test": e,
        **_pad(test[col], test["label"], test["kind"], thr),
    }


def main(cfg: dict | None = None) -> pd.DataFrame | None:
    cfg = cfg or load_config()
    lc = cfg["liveness"]
    clips = own_capture(cfg)
    if clips.empty:
        print(HOWTO)
        return None
    clips = subject_split(clips, cfg["seed"])
    rows, crops = [], {}
    for r in tqdm(clips.itertuples(), total=len(clips), desc="clips"):
        frames = read_video(r.path, lc["max_frames"], lc["frame_stride"])
        rows.append(
            {
                "subject": r.subject,
                "kind": r.kind,
                "label": r.label,
                "split": r.split,
                "path": str(r.path),
                **clip_features(frames, cfg, lc["diff_sweep"]),
            }
        )
        crops[str(r.path)] = face_crops(frames, cfg)
    df = pd.DataFrame(rows)
    write_results(df, "liveness_features", cfg)
    tr, va, te = (df[df.split == s] for s in ("train", "val", "test"))
    if tr["label"].nunique() < 2 or va["label"].nunique() < 2 or te["label"].nunique() < 2:
        print("need live and attack clips in every split; record more subjects")
        return None

    # E10: threshold sweep for the motion signal
    e10 = []
    for t in lc["diff_sweep"]:
        r = evaluate_signal(f"motion@{t}", va, te, f"motion_ratio@{t}")
        e10.append({"diff_threshold": t, **r})
    e10 = pd.DataFrame(e10)
    write_results(e10, "E10", cfg)
    viz.lines(
        e10.melt(id_vars="diff_threshold", value_vars=["APCER", "BPCER", "ACER"]),
        "diff_threshold",
        "value",
        "variable",
        path(cfg, "results") / "figures" / "E10.png",
        "Motion liveness vs frame-diff threshold",
        "error rate",
    )

    # E11: signals
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    feats = [k for k in FEATURES if k in df]
    mm = [k for k in manmade.FEATURES]
    out = [
        evaluate_signal("motion (Lab 3 frame diff)", va, te, "motion_ratio"),
        evaluate_signal("blink (EAR dip)", va, te, "ear_dip"),
        evaluate_signal("man-made cues: rules", va, te, "rules_live"),
    ]
    df = df.copy()
    for name, model, cols in [
        ("man-made cues: LR", LogisticRegression(class_weight="balanced", max_iter=1000), mm),
        ("man-made cues: SVM", SVC(probability=True, class_weight="balanced"), mm),
        ("handcrafted fused: LR", LogisticRegression(class_weight="balanced", max_iter=1000), feats),
    ]:
        pipe = make_pipeline(StandardScaler(), model).fit(tr[cols], tr["label"])
        df[name] = pipe.predict_proba(df[cols])[:, 1]
        out.append(evaluate_signal(name, df[df.split == "val"], df[df.split == "test"], name))
        if name == "handcrafted fused: LR":
            import joblib

            p = path(cfg, "models") / cfg["models"]["liveness_lr"]
            p.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(pipe, p)

    try:
        from ..liveness import cnn

        X = [c for p in tr["path"] for c in crops[p]]
        y = [lab for p, lab in zip(tr["path"], tr["label"]) for _ in crops[p]]
        model = cnn.train(X, y, cfg, path(cfg, "models") / cfg["models"]["liveness_cnn"], seed=cfg["seed"])
        model.eval()
        df["cnn"] = [
            float(np.mean(cnn.predict(model, crops[p], cfg))) if crops[p] else 0.0 for p in df["path"]
        ]
        out.append(
            evaluate_signal("CNN (MobileNetV3-S)", df[df.split == "val"], df[df.split == "test"], "cnn")
        )
        emb = {p: cnn.embed(model, crops[p], cfg).mean(0) if crops[p] else None for p in df["path"]}
        dim = next(v for v in emb.values() if v is not None).shape[0]
        E = np.stack([emb[p] if emb[p] is not None else np.zeros(dim) for p in df["path"]])
        Z = np.hstack([E, df[feats].values])
        trm = (df.split == "train").values
        fused = make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, C=0.1)
        ).fit(Z[trm], df["label"][trm])
        df["cnn+handcrafted"] = fused.predict_proba(Z)[:, 1]
        out.append(
            evaluate_signal(
                "CNN + handcrafted", df[df.split == "val"], df[df.split == "test"], "cnn+handcrafted"
            )
        )
    except ImportError:
        log.warning("torch not installed (uv sync --extra cnn): skipping CNN signals")

    e11 = pd.DataFrame(out)
    write_results(e11, "E11", cfg)
    viz.bars(
        e11, "signal", "ACER", path(cfg, "results") / "figures" / "E11.png", "Liveness signals (test ACER)"
    )
    print(e11.round(4).to_string(index=False))
    return e11
