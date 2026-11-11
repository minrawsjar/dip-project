import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from ekyc import metrics


def test_eer_perfect_and_inverted():
    s, y = [0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]
    assert metrics.eer(s, y)[0] == pytest.approx(0.0)
    assert metrics.eer([-v for v in s], y)[0] == pytest.approx(1.0)


def test_eer_known_value():
    # 4 genuine, 4 impostor; one of each on the wrong side -> FAR = FRR = 0.25 at the crossing
    s = [0.9, 0.8, 0.7, 0.3, 0.6, 0.2, 0.15, 0.1]
    y = [1, 1, 1, 1, 0, 0, 0, 0]
    assert metrics.eer(s, y)[0] == pytest.approx(0.25)


def test_roc_auc_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = y + rng.normal(0, 1, 500)
    fpr, tpr, _ = metrics.roc(s, y)
    assert metrics.auc(fpr, tpr) == pytest.approx(roc_auc_score(y, s), abs=1e-9)


def test_far_frr_and_tar():
    s, y = np.array([0.9, 0.4, 0.6, 0.1]), np.array([1, 1, 0, 0])
    assert metrics.far_frr(s, y, 0.5) == (0.5, 0.5)
    assert metrics.tar_at_far(s, y, 0.0) == 0.5


def test_pad_metrics_worst_case_apcer():
    s = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.1])
    y = np.array([1, 1, 0, 0, 0, 0])
    kinds = np.array(["bona_fide", "bona_fide", "print", "print", "replay", "replay"])
    m = metrics.pad_metrics(s, y, 0.5, kinds)
    assert m["APCER_print"] == 0.5 and m["APCER_replay"] == 0.5 and m["BPCER"] == 0.0
    assert m["ACER"] == pytest.approx(0.25)


def test_levenshtein_cer_wer():
    assert metrics.levenshtein("kitten", "sitting") == 3
    assert metrics.cer("abcd", "abcd") == 0
    assert metrics.cer("abcd", "") == 1
    assert metrics.wer("a b c", "a x c") == pytest.approx(1 / 3)


def test_quad_iou_and_corner_error():
    sq = [[0, 0], [2, 0], [2, 2], [0, 2]]
    shifted = [[1, 0], [3, 0], [3, 2], [1, 2]]
    assert metrics.quad_iou(sq, sq) == pytest.approx(1.0)
    assert metrics.quad_iou(sq, shifted) == pytest.approx(1 / 3)
    assert metrics.corner_error(sq, shifted) == pytest.approx(1.0)


def test_bootstrap_ci_brackets_point():
    v = np.random.default_rng(1).normal(5, 1, 200)
    p, lo, hi = metrics.bootstrap_ci(np.mean, v, n=300)
    assert lo < p < hi and hi - lo < 0.5
