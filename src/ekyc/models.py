"""Model weight paths; downloads on first use into `paths.models` (gitignored)."""

from __future__ import annotations

import logging
import urllib.request
from functools import lru_cache
from pathlib import Path

from .io import path

log = logging.getLogger(__name__)


def model_path(cfg: dict, name: str) -> Path:
    """Local path of a model listed under `models:` in the config, downloading it if absent."""
    spec = cfg["models"][name]
    p = path(cfg, "models") / spec["file"]
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s -> %s", spec["url"], p)
        tmp = p.with_suffix(p.suffix + ".part")
        urllib.request.urlretrieve(spec["url"], tmp)
        tmp.rename(p)
    return p


@lru_cache(8)
def _joblib(p: str, mtime: float):
    import joblib

    return joblib.load(p)


def load_trained(p: Path):
    """A trained sklearn model, cached until the file changes (retraining is picked up)."""
    return _joblib(str(p), p.stat().st_mtime) if p.exists() else None


def download_all(cfg: dict) -> None:
    for name, spec in cfg["models"].items():
        if isinstance(spec, dict) and "url" in spec:
            print(name, model_path(cfg, name))
