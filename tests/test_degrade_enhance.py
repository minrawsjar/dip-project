import cv2
import numpy as np
import pytest

from ekyc import degrade, enhance

IMG = (np.random.default_rng(0).random((64, 80, 3)) * 255).astype(np.uint8)
SEEDED = {"add_noise": {"kind": "gaussian", "level": 0.1}, "glare": {}, "perspective": {"max_shift": 0.1}}


@pytest.mark.parametrize("op,params", SEEDED.items())
def test_seeded_ops_are_deterministic(op, params):
    a, pa = degrade.apply(IMG, op, seed=7, **params)
    b, pb = degrade.apply(IMG, op, seed=7, **params)
    c, _ = degrade.apply(IMG, op, seed=8, **params)
    assert np.array_equal(a, b) and pa == pb
    assert not np.array_equal(a, c)


@pytest.mark.parametrize(
    "op,params",
    [
        ("quantize_bits", {"bits": 3}),
        ("downsample", {"scale": 0.25}),
        ("jpeg", {"quality": 30}),
        ("gaussian_blur", {"sigma": 2}),
        ("motion_blur", {"length": 9, "angle": 30}),
        ("halftone", {"cell": 4}),
        ("brightness_contrast", {"alpha": 0.5, "beta": -10}),
        ("gamma", {"k": 1.8}),
    ],
)
def test_ops_keep_shape_and_dtype(op, params):
    out, info = degrade.apply(IMG, op, **params)
    assert out.shape == IMG.shape and out.dtype == np.uint8 and info["op"] == op


def test_quantize_levels():
    out, _ = degrade.quantize_bits(IMG, 2)
    assert len(np.unique(out)) <= 4
    assert np.array_equal(degrade.quantize_bits(IMG, 8)[0], IMG)


def test_histeq_matches_opencv():
    g = cv2.cvtColor(IMG, cv2.COLOR_BGR2GRAY)
    assert np.abs(enhance.histeq(g).astype(int) - cv2.equalizeHist(g).astype(int)).max() <= 1


def test_hsv_v_preserves_hue_rgb_does_not():
    img = np.zeros((32, 32, 3), np.uint8)
    img[..., 2] = np.linspace(60, 200, 32, dtype=np.uint8)[None, :]  # reddish ramp
    img[..., 1] = np.linspace(20, 120, 32, dtype=np.uint8)[:, None]
    assert enhance.hue_shift(img, enhance.apply(img, "histeq", "hsv_v")) < 2.0
    assert enhance.hue_shift(img, enhance.apply(img, "histeq", "rgb")) > 2.0
