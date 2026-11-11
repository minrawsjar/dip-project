"""Image/video I/O, dtype conversions, config loading and intermediate-image dumps.

Course link: the lab lesson on `im2double` / display scaling. Arithmetic happens in float32 [0, 1]
and results are clipped back to uint8 explicitly.
"""

from __future__ import annotations

import copy
import logging
import subprocess
from pathlib import Path

import cv2
import numpy as np
import yaml

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    """Load the YAML config (default: configs/default.yaml)."""
    path = Path(path) if path else ROOT / "configs" / "default.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def with_overrides(cfg: dict, overrides: dict | None) -> dict:
    """Deep copy of cfg with dotted-key overrides, e.g. {"face.align": False}. Used by ablations."""
    out = copy.deepcopy(cfg)
    for key, val in (overrides or {}).items():
        node = out
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = val
    return out


def path(cfg: dict, key: str) -> Path:
    """Resolve a `paths:` entry relative to the repo root."""
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else ROOT / p


def imread(p: str | Path) -> np.ndarray:
    """Read an image as uint8 BGR (unicode-safe). Raises on failure."""
    img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read image: {p}")
    return img


def imdecode(buf: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("not a decodable image")
    return img


def imwrite(p: str | Path, img: np.ndarray) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(p.suffix or ".png", to_uint8(img))
    if not ok:
        raise ValueError(f"cannot encode {p}")
    buf.tofile(str(p))


def read_video(p: str | Path, max_frames: int = 90, stride: int = 1) -> list[np.ndarray]:
    """Read up to `max_frames` BGR frames, keeping every `stride`-th one."""
    cap = cv2.VideoCapture(str(p))
    frames, i = [], 0
    while len(frames) < max_frames:
        ok, f = cap.read()
        if not ok:
            break
        if i % stride == 0:
            frames.append(f)
        i += 1
    cap.release()
    if not frames:
        raise ValueError(f"no frames decoded from {p}")
    return frames


def to_float(img: np.ndarray) -> np.ndarray:
    """uint8 -> float32 in [0, 1] (MATLAB `im2double`)."""
    return img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img.astype(np.float32)


def to_uint8(img: np.ndarray) -> np.ndarray:
    """float [0, 1] (or any numeric) -> uint8 with clipping (MATLAB `im2uint8`)."""
    if img.dtype == np.uint8:
        return img
    if img.dtype == bool:
        return img.astype(np.uint8) * 255
    if np.issubdtype(img.dtype, np.floating) and img.max(initial=0) <= 1.0 + 1e-6:
        img = img * 255.0
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def to_gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def normalize_display(img: np.ndarray) -> np.ndarray:
    """Min-max scale any array to uint8 for display (MATLAB `imshow(I, [])`)."""
    img = img.astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    return np.zeros(img.shape, np.uint8) if hi - lo < 1e-9 else to_uint8((img - lo) / (hi - lo))


class Debug:
    """Collects named intermediate images for report figures and the web UI.

    Every pipeline stage calls `debug.add(name, img)`; pass `None` to disable. If `out_dir` is set,
    each image is also written to `<out_dir>/<NN>_<name>.png` in stage order.
    """

    def __init__(self, out_dir: str | Path | None = None, prefix: str = ""):
        self.out_dir = Path(out_dir) if out_dir else None
        self.prefix = prefix
        self.images: dict[str, np.ndarray] = {}

    def add(self, name: str, img: np.ndarray) -> None:
        name = f"{self.prefix}{name}"
        if img.dtype == bool:
            img = img.astype(np.uint8) * 255
        elif img.dtype != np.uint8:
            img = to_uint8(img) if img.min() >= 0 and img.max() <= 1 else normalize_display(img)
        self.images[name] = img
        if self.out_dir:
            imwrite(self.out_dir / f"{len(self.images):02d}_{name.replace('/', '__')}.png", img)

    def child(self, prefix: str) -> Debug:
        """A view that shares storage but namespaces names, e.g. 'doc/'."""
        c = Debug.__new__(Debug)
        c.out_dir, c.prefix, c.images = self.out_dir, self.prefix + prefix, self.images
        return c


def dbg(debug: Debug | None, name: str, img: np.ndarray) -> None:
    if debug is not None:
        debug.add(name, img)


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, OSError):  # no commits yet, or git missing
        return "uncommitted"


def write_results(df, exp_id: str, cfg: dict) -> Path:
    """Write a results CSV with the git commit hash column (CLAUDE.md §9)."""
    out = path(cfg, "results") / f"{exp_id}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["commit"] = git_hash()
    df.to_csv(out, index=False)
    log.info("wrote %s (%d rows)", out, len(df))
    return out
