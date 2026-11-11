import cv2
import numpy as np
import pytest

from ekyc.document import corners, hough, validity
from ekyc.io import load_config

CFG = load_config()


def test_order_quad_any_input_order():
    q = np.float32([[10, 10], [100, 12], [98, 60], [8, 58]])
    for perm in ([2, 0, 3, 1], [3, 2, 1, 0], [1, 3, 0, 2]):
        assert np.allclose(corners.order_quad(q[perm]), q)


def test_canonical_corner_error_zero_for_gt():
    q = np.float32([[50, 40], [400, 60], [390, 300], [40, 280]])
    assert corners.canonical_corner_error(q, q, (856, 540)) == pytest.approx(0, abs=1e-3)


def _card_scene():
    img = np.full((480, 640, 3), (40, 90, 160), np.uint8)
    quad = np.int32([[120, 100], [520, 130], [500, 380], [100, 350]])
    cv2.fillConvexPoly(img, quad, (235, 235, 235))
    return img, quad.astype(np.float32)


def test_hough_recovers_card_corners():
    img, quad = _card_scene()
    e = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0), 50, 150)
    segs = hough.segments(e, CFG)
    coarse = hough.quad_from_segments(segs, img.shape, CFG)
    refined = hough.quad_from_segments(segs, img.shape, CFG, edges=e)  # least-squares refit on edge pixels
    assert coarse is not None and refined is not None
    assert np.abs(corners.order_quad(coarse) - quad).max() < 4
    assert np.abs(corners.order_quad(refined) - quad).max() < 2.5


def test_validity_rules_accept_card_reject_blob():
    _, quad = _card_scene()
    ok = validity.check(quad.reshape(-1, 1, 2).astype(np.int32), (480, 640), CFG)
    assert ok["valid"], ok["reasons"]
    circle = cv2.ellipse2Poly((320, 240), (150, 150), 0, 0, 360, 5).reshape(-1, 1, 2)
    assert not validity.check(circle, (480, 640), CFG)["valid"]
