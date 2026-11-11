"""End-to-end smoke test through the HTTP API on a synthetic card (needs model weights; downloads on first run)."""

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from ekyc.api import app
from ekyc.document import template
from ekyc.io import load_config

client = TestClient(app)


def _png(img):
    return cv2.imencode(".png", img)[1].tobytes()


@pytest.fixture(scope="module")
def scene():
    cfg = load_config()
    face = np.full((250, 250, 3), 180, np.uint8)  # no real face: exercises the "not found" paths
    card, _ = template.render_card(face, template.random_identity(np.random.default_rng(0)), cfg)
    img, _, _ = template.capture(card, seed=1)
    return img


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_lab_ops(scene):
    for op, params in [
        ("enhance", '{"method":"clahe","mode":"hsv_v"}'),
        ("edges", '{"method":"sobel"}'),
        ("degrade", '{"op":"jpeg","quality":10}'),
        ("hough", "{}"),
    ]:
        r = client.post(
            "/api/lab", files={"image": ("s.png", _png(scene))}, data={"op": op, "params": params}
        )
        assert r.status_code == 200, r.text
        assert r.json()["output"].startswith("data:image/") and len(r.json()["hist_out"]) == 256


def test_lab_rejects_bad_input(scene):
    assert (
        client.post("/api/lab", files={"image": ("x.png", b"nope")}, data={"op": "edges"}).status_code == 400
    )
    r = client.post(
        "/api/lab", files={"image": ("s.png", _png(scene))}, data={"op": "rm -rf", "params": "{}"}
    )
    assert r.status_code == 400


def test_verify_returns_decision_and_images(scene):
    r = client.post(
        "/api/verify", files={"selfie": ("a.png", _png(scene)), "id_image": ("b.png", _png(scene))}
    )
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["decision"] in ("ACCEPT", "REJECT", "REVIEW")
    assert j["document"]["found"] and j["images"]
    assert j["face"]["score"] is None  # blank portrait: no face -> must not accept
    assert j["decision"] != "ACCEPT"
