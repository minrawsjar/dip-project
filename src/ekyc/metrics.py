"""Evaluation metrics implemented from scratch (CLAUDE.md §7); cross-checked against sklearn in tests.

Score convention: higher score = more likely genuine / bona fide. Labels: 1 = genuine / bona fide.
"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np


def roc(scores, labels) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(fpr, tpr, thresholds), thresholds descending; accept when score >= threshold."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    distinct = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tp = np.cumsum(y)[distinct]
    fp = np.cumsum(1 - y)[distinct]
    P, N = max(y.sum(), 1), max((1 - y).sum(), 1)
    tpr = np.r_[0.0, tp / P]
    fpr = np.r_[0.0, fp / N]
    return fpr, tpr, np.r_[np.inf, s[distinct]]


def auc(fpr, tpr) -> float:
    return float(np.trapezoid(tpr, fpr))


def far_frr(scores, labels, threshold: float) -> tuple[float, float]:
    """FAR = impostors accepted / impostors; FRR = genuines rejected / genuines."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    acc = s >= threshold
    far = acc[y == 0].mean() if (y == 0).any() else 0.0
    frr = (~acc[y == 1]).mean() if (y == 1).any() else 0.0
    return float(far), float(frr)


def eer(scores, labels) -> tuple[float, float]:
    """Equal error rate and its threshold, linearly interpolated where FAR(t) crosses FRR(t)."""
    fpr, tpr, thr = roc(scores, labels)
    fnr = 1 - tpr
    i = int(np.argmax(fpr >= fnr))  # first ROC point where FAR >= FRR
    if i == 0:
        return float(fnr[0]), float(thr[1] if len(thr) > 1 else thr[0])
    d0, d1 = fnr[i - 1] - fpr[i - 1], fnr[i] - fpr[i]
    a = d0 / (d0 - d1) if d0 != d1 else 0.0
    e = fpr[i - 1] + a * (fpr[i] - fpr[i - 1])
    t = thr[i] if np.isinf(thr[i - 1]) else thr[i - 1] + a * (thr[i] - thr[i - 1])
    return float(e), float(t)


def tar_at_far(scores, labels, far: float) -> float:
    """True accept rate at the largest operating point with FAR <= `far`."""
    fpr, tpr, _ = roc(scores, labels)
    ok = fpr <= far
    return float(tpr[ok].max()) if ok.any() else 0.0


def pad_metrics(scores, labels, threshold: float, attack_types=None) -> dict:
    """ISO/IEC 30107-3 PAD metrics. labels: 1 = bona fide. Score >= threshold -> classified bona fide.

    APCER is computed per attack type and the worst case is reported (as the standard specifies);
    BPCER = bona fide rejected; ACER = (APCER + BPCER) / 2.
    """
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    live = s >= threshold
    bpcer = float((~live[y == 1]).mean()) if (y == 1).any() else 0.0
    out = {"BPCER": bpcer}
    if attack_types is None:
        attack_types = np.where(y == 0, "attack", "bona_fide")
    at = np.asarray(attack_types)
    apcers = {}
    for t in sorted(set(at[y == 0])):
        m = (y == 0) & (at == t)
        apcers[t] = float(live[m].mean())
        out[f"APCER_{t}"] = apcers[t]
    out["APCER"] = max(apcers.values()) if apcers else 0.0
    out["ACER"] = (out["APCER"] + bpcer) / 2
    return out


def levenshtein(a, b) -> int:
    """Edit distance (insert/delete/substitute = 1) between two sequences."""
    a, b = list(a), list(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    """Character error rate = edits / len(ref)."""
    return levenshtein(ref, hyp) / max(len(ref), 1)


def wer(ref: str, hyp: str) -> float:
    return levenshtein(ref.split(), hyp.split()) / max(len(ref.split()), 1)


def quad_iou(q1, q2) -> float:
    """IoU of two convex quadrilaterals (4x2 point arrays)."""
    a = np.asarray(q1, np.float32).reshape(-1, 2)
    b = np.asarray(q2, np.float32).reshape(-1, 2)
    a, b = cv2.convexHull(a), cv2.convexHull(b)
    inter, _ = cv2.intersectConvexConvex(a, b)
    union = cv2.contourArea(a) + cv2.contourArea(b) - inter
    return float(inter / union) if union > 0 else 0.0


def corner_error(pred, gt) -> float:
    """Mean Euclidean distance between corresponding corners (both ordered TL, TR, BR, BL)."""
    return float(np.linalg.norm(np.asarray(pred, float) - np.asarray(gt, float), axis=1).mean())


def bootstrap_ci(
    fn: Callable, *arrays, n: int = 1000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float, float]:
    """(point, lo, hi) percentile bootstrap CI of fn(*arrays) resampling rows jointly."""
    arrays = [np.asarray(a) for a in arrays]
    rng = np.random.default_rng(seed)
    m = len(arrays[0])
    stats = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        try:
            stats.append(fn(*[a[idx] for a in arrays]))
        except (ValueError, ZeroDivisionError):
            continue
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(fn(*arrays)), float(lo), float(hi)
