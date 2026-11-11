"""End-to-end verification: quality gate -> document -> face match -> liveness -> fusion.

Each stage is timed and dumps intermediate images into a shared `Debug` (namespaced per module), which
the CLI writes to `--debug-dir` and the web API returns to the browser.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

import numpy as np

from . import document, face, fusion, liveness, quality
from .io import Debug, with_overrides


@contextmanager
def _timed(timings: dict, name: str):
    t0 = time.perf_counter()
    yield
    timings[name] = round((time.perf_counter() - t0) * 1000, 1)


def verify(
    selfie: np.ndarray,
    id_img: np.ndarray,
    cfg: dict,
    frames: list[np.ndarray] | None = None,
    debug: Debug | None = None,
    compare_baseline: bool = True,
) -> dict:
    """Verify that the person in `selfie` (and optional video `frames`) is the holder of `id_img`."""
    debug = debug or Debug()
    t: dict[str, float] = {}
    res: dict = {}

    with _timed(t, "document"):
        doc = document.process(id_img, cfg, debug=debug.child("document/"))
    res["document"] = {k: v for k, v in doc.items() if k not in ("card", "portrait")}

    with _timed(t, "quality"):
        # glare only matters on the face / card, not on white walls or desks around them
        f = face.detect.detect(selfie, cfg, "yunet")
        res["quality"] = {
            "selfie": quality.quality_gate(
                selfie,
                cfg,
                roi=quality.box_mask(selfie.shape, f["box"]) if f else None,
                face_box=f["box"] if f else None,
                debug=debug.child("quality/selfie/"),
            ),
            "id": quality.quality_gate(
                id_img,
                cfg,
                roi=quality.quad_mask(id_img.shape, doc["quad"]) if doc["quad"] else None,
                debug=debug.child("quality/id/"),
            ),
        }

    with _timed(t, "face"):
        sel = face.represent(selfie, cfg, debug.child("face/selfie/"))
        por = (
            face.represent(doc["portrait"], cfg, debug.child("face/portrait/"))
            if doc.get("portrait") is not None
            else {"found": False, "reason": "no portrait crop"}
        )
        res["face"] = {
            "score": face.score(sel, por, cfg),
            "threshold": face.threshold(cfg),
            "selfie_found": sel["found"],
            "portrait_found": por["found"],
            "method": f"{cfg['face']['detector']}+{cfg['face']['descriptor']}",
        }
    if compare_baseline:
        with _timed(t, "face_baseline"):
            bcfg = with_overrides(cfg, face.BASELINE)
            bs, bp = face.represent(selfie, bcfg, debug.child("face_baseline/selfie/")), (
                face.represent(doc["portrait"], bcfg, debug.child("face_baseline/portrait/"))
                if doc.get("portrait") is not None
                else {"found": False}
            )
            res["face_baseline"] = {
                "score": face.score(bs, bp, bcfg),
                "threshold": face.threshold(bcfg),
                "method": "haar+geometric",
            }

    with _timed(t, "liveness"):
        live = liveness.analyze(frames if frames else [selfie], cfg, debug.child("liveness/"))
    res["liveness"] = live

    scores = {
        "face_score": res["face"]["score"],
        "liveness_score": live["liveness_score"],
        "ocr_conf": doc["ocr_conf"],
        "selfie_blur": res["quality"]["selfie"]["scores"]["blur"],
        "id_blur": res["quality"]["id"]["scores"]["blur"],
        "id_glare": res["quality"]["id"]["scores"]["glare_ratio"],
        "quality_passed": res["quality"]["selfie"]["passed"] and res["quality"]["id"]["passed"],
    }
    with _timed(t, "fusion"):
        res["fusion"] = fusion.fuse(scores, cfg)
    res["scores"] = scores
    res["decision"] = res["fusion"]["decision"]
    res["timings_ms"] = t
    return res
