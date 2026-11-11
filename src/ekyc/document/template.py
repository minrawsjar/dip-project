"""MIDV-2020 annotation loading and synthetic ID-card rendering.

MIDV-2020 (Bulatov et al., Computer Optics 2022; CC BY-SA 2.5) annotations are VGG Image Annotator
v2 JSON: `templates/annotations/<code>.json` holds field rects + values in template coordinates,
`photo/annotations/<code>.json` holds the `doc_quad` polygon of each photo. Template and photo with the
same file name are the same physical document.

The synthetic renderer produces mock ID-1 cards with known field values, portrait slot and (after
`capture`) ground-truth quad, so document experiments run even without MIDV-2020.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from .. import degrade
from .corners import canonical_size
from .ocr import layout_rects

# --------------------------------------------------------------------------- MIDV-2020


def load_via(p: str | Path) -> dict[str, dict]:
    """filename -> {"quad": 4x2 | None, "fields": {name: {"rect": (x,y,w,h), "value": str}}}."""
    data = json.loads(Path(p).read_text(encoding="utf-8"))
    meta = data.get("_via_img_metadata", data)
    out = {}
    for entry in meta.values():
        quad, fields = None, {}
        for r in entry.get("regions", []):
            s, a = r["shape_attributes"], r["region_attributes"]
            name = a.get("field_name", "")
            if s["name"] == "polygon":
                pts = np.stack([s["all_points_x"], s["all_points_y"]], 1).astype(np.float32)
                if name == "doc_quad":
                    quad = pts
                else:
                    x, y, w, h = cv2.boundingRect(pts)
                    fields[name] = {"rect": (x, y, w, h), "value": a.get("value", "")}
            elif s["name"] == "rect":
                fields[name] = {
                    "rect": (s["x"], s["y"], s["width"], s["height"]),
                    "value": a.get("value", ""),
                }
        out[entry["filename"]] = {"quad": quad, "fields": fields}
    return out


def midv_photos(root: str | Path, text_only: bool = True):
    """Yield dicts (image path, gt quad, template fields, template size) for every MIDV-2020 photo."""
    root = Path(root)
    for ann in sorted((root / "photo" / "annotations").glob("*.json")):
        code = ann.stem
        photos = load_via(ann)
        tmpl = load_via(root / "templates" / "annotations" / f"{code}.json")
        for fname, p in photos.items():
            t = tmpl.get(fname)
            img = root / "photo" / "images" / code / fname
            if p["quad"] is None or t is None or not img.exists():
                continue
            tpath = root / "templates" / "images" / code / fname
            fields = {k: v for k, v in t["fields"].items() if not text_only or v["value"]}
            yield {
                "code": code,
                "id": f"{code}/{fname}",
                "image": img,
                "template": tpath,
                "quad": p["quad"],
                "fields": fields,
            }


# --------------------------------------------------------------------------- synthetic cards

SURNAMES = [
    "NOVAK",
    "SILVA",
    "KOWALSKI",
    "MERCER",
    "HALVORSEN",
    "OKAFOR",
    "DUBOIS",
    "TANAKA",
    "IVANOVA",
    "FERREIRA",
    "LINDQVIST",
    "MORENO",
    "BRANDT",
    "SATO",
    "KAPOOR",
    "QUINN",
    "VOSS",
    "MARLOWE",
]
GIVEN = [
    "ANNA",
    "LUKA",
    "MAYA",
    "OMAR",
    "IRIS",
    "TEO",
    "NORA",
    "JONAS",
    "LENA",
    "RAVI",
    "ELIF",
    "MILO",
    "SARA",
    "IVAN",
    "ZOE",
    "ADAM",
    "NINA",
    "KAI",
]


@lru_cache(1)
def _font_path() -> str:
    return font_manager.findfont("DejaVu Sans:bold")


def random_identity(rng: np.random.Generator) -> dict[str, str]:
    d, m, y = rng.integers(1, 29), rng.integers(1, 13), rng.integers(1950, 2006)
    return {
        "surname": str(rng.choice(SURNAMES)),
        "given_name": str(rng.choice(GIVEN)),
        "birth_date": f"{d:02d}.{m:02d}.{y}",
        "id_number": "".join(rng.choice(list("ABCDEFGHJKLMNPRSTUVWXYZ"), 2))
        + "".join(map(str, rng.integers(0, 10, 7))),
        "expiry_date": f"{rng.integers(1, 29):02d}.{rng.integers(1, 13):02d}.{rng.integers(2027, 2036)}",
    }


def render_card(
    face: np.ndarray, values: dict[str, str], cfg: dict, seed: int = 0
) -> tuple[np.ndarray, dict]:
    """Mock ID-1 card (canonical size) with guilloche background, portrait and text fields.

    Returns (card_bgr, {"fields": {name: {"rect", "value"}}, "portrait": rect}). Text is drawn in the
    lower part of each layout cell so field rects fully contain it.
    """
    rng = np.random.default_rng(seed)
    W, H = canonical_size(cfg)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    base = np.array(rng.uniform(205, 240, 3), np.float32)
    bg = base + 6 * np.sin(xx / rng.uniform(9, 15) + 3 * np.sin(yy / rng.uniform(25, 40)))[..., None]
    bg += 4 * np.sin((xx + yy) / rng.uniform(15, 25))[..., None]
    card = np.clip(bg, 0, 255).astype(np.uint8)
    cv2.rectangle(card, (0, 0), (W, round(0.12 * H)), tuple(int(c) for c in rng.uniform(60, 140, 3)), -1)

    rects = layout_rects(cfg["document"]["layout"], (W, H), skip=())
    px, py, pw, ph = rects.pop("portrait")
    fh, fw = face.shape[:2]
    s = max(pw / fw, ph / fh)  # cover the slot, centre-crop the overflow
    f = cv2.resize(face, (round(fw * s), round(fh * s)), interpolation=cv2.INTER_AREA)
    oy, ox = (f.shape[0] - ph) // 2, (f.shape[1] - pw) // 2
    card[py : py + ph, px : px + pw] = f[oy : oy + ph, ox : ox + pw]

    pil = Image.fromarray(cv2.cvtColor(card, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font_path = _font_path()
    head = ImageFont.truetype(font_path, round(0.07 * H))
    draw.text((round(0.03 * W), round(0.02 * H)), "IDENTITY CARD", fill=(255, 255, 255), font=head)
    fields = {}
    for name, (x, y, w, h) in rects.items():
        val = values.get(name, "")
        label = ImageFont.truetype(font_path, max(8, round(h * 0.22)))
        draw.text((x, y - round(h * 0.25)), name.replace("_", " ").upper(), fill=(90, 90, 110), font=label)
        font = ImageFont.truetype(font_path, max(10, round(h * 0.62)))
        draw.text((x + 4, y + round(h * 0.12)), val, fill=(20, 20, 30), font=font)
        fields[name] = {"rect": (x, y, w, h), "value": val}
    card = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return card, {"fields": fields, "portrait": (px, py, pw, ph)}


def capture(
    card: np.ndarray, seed: int = 0, out_size=(1024, 768), severity: float = 1.0
) -> tuple[np.ndarray, np.ndarray, list]:
    """Photograph a printed card: halftone print, place on a textured desk under a random homography,
    add glare, low light and JPEG. Returns (scene, gt_quad TL-TR-BR-BL, degradation log)."""
    rng = np.random.default_rng(seed)
    W, H = out_size
    ch, cw = card.shape[:2]
    dots, _ = degrade.halftone(card, 2)
    printed = cv2.addWeighted(dots, 0.35, card, 0.65, 0)  # dot screen is mostly below camera resolution
    # desk background: smooth random colour field + wood-ish stripes
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    col = rng.uniform(40, 200, 3)
    desk = (
        col
        + 18 * np.sin(yy / rng.uniform(6, 20) + 2 * np.sin(xx / 90))[..., None]
        + rng.normal(0, 6, (H, W, 1))
    )
    scene = np.clip(desk, 0, 255).astype(np.uint8)
    # target quad: card covers 45-70% of width, random tilt + perspective jitter
    scale = rng.uniform(0.45, 0.7) * W / cw
    cx, cy = W / 2 + rng.uniform(-0.1, 0.1) * W, H / 2 + rng.uniform(-0.1, 0.1) * H
    ang = np.radians(rng.uniform(-12, 12) * severity)
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    base = np.array([[-cw, -ch], [cw, -ch], [cw, ch], [-cw, ch]], np.float32) / 2 * scale
    quad = (base @ R.T + [cx, cy] + rng.uniform(-0.06, 0.06, (4, 2)) * cw * scale * severity).astype(
        np.float32
    )
    src = np.float32([[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]])
    Hm = cv2.getPerspectiveTransform(src, quad)
    warped = cv2.warpPerspective(printed, Hm, (W, H), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.full((ch, cw), 255, np.uint8), Hm, (W, H)) > 127
    scene[mask] = warped[mask]
    steps = [
        ("glare", {"radius": rng.uniform(0.1, 0.25), "strength": rng.uniform(0.2, 0.7) * severity}),
        ("gamma", {"k": rng.uniform(0.9, 1.6)}),
        ("gaussian_blur", {"sigma": rng.uniform(0.3, 1.3) * severity}),
        ("jpeg", {"quality": int(rng.integers(40, 90))}),
    ]
    scene, info = degrade.chain(scene, steps, seed)
    return scene, quad, info
