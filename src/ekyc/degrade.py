"""Deterministic image degradations for ablations (CLAUDE.md §6).

Every function returns `(degraded_uint8, params)` so the parameters can be logged to a manifest.
Stochastic functions take a `seed` and are bit-exact reproducible.
"""

from __future__ import annotations

import cv2
import numpy as np

from .io import to_float, to_uint8

INTERP = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
    "bicubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
    "area": cv2.INTER_AREA,
}


def quantize_bits(img: np.ndarray, bits: int) -> tuple[np.ndarray, dict]:
    """Reduce grey-level resolution to `bits` per channel (quantization / false contouring).

    Equivalent to the lab bit-shift `bitshift(bitshift(I, -(8-b)), 8-b)`, reconstructed at the
    centre of each quantization bin so mean brightness is preserved.
    """
    step = 1 << (8 - bits)
    out = (img // step) * step + step // 2 if bits < 8 else img.copy()
    return out.astype(np.uint8), {"op": "quantize_bits", "bits": bits}


def downsample(img: np.ndarray, scale: float, interp: str = "bicubic") -> tuple[np.ndarray, dict]:
    """Lose spatial resolution: area-downsample by `scale`, then upsample back with `interp`.

    MATLAB: `imresize(imresize(I, s), size(I), method)`.
    """
    h, w = img.shape[:2]
    if scale >= 1:
        return img.copy(), {"op": "downsample", "scale": scale, "interp": interp}
    small = cv2.resize(
        img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA
    )
    out = cv2.resize(small, (w, h), interpolation=INTERP[interp])
    return out, {"op": "downsample", "scale": scale, "interp": interp}


def jpeg(img: np.ndarray, quality: int) -> tuple[np.ndarray, dict]:
    """Lossy JPEG round trip (DCT quantization blocking)."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    assert ok
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED), {"op": "jpeg", "quality": quality}


def gaussian_blur(img: np.ndarray, sigma: float) -> tuple[np.ndarray, dict]:
    out = img.copy() if sigma <= 0 else cv2.GaussianBlur(img, (0, 0), sigma)
    return out, {"op": "gaussian_blur", "sigma": sigma}


def motion_blur(img: np.ndarray, length: int, angle: float) -> tuple[np.ndarray, dict]:
    """Linear motion PSF (MATLAB `fspecial('motion', len, theta)`)."""
    length = max(1, int(length))
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D(((length - 1) / 2, (length - 1) / 2), angle, 1.0)
    k = cv2.warpAffine(k, rot, (length, length))
    k /= k.sum()
    return cv2.filter2D(img, -1, k), {"op": "motion_blur", "length": length, "angle": angle}


def add_noise(img: np.ndarray, kind: str, level: float, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Gaussian (sigma in [0,1] units) or salt-and-pepper (density) noise (MATLAB `imnoise`)."""
    rng = np.random.default_rng(seed)
    f = to_float(img)
    if kind == "gaussian":
        f = f + rng.normal(0, level, f.shape).astype(np.float32)
    elif kind in ("salt_pepper", "s&p"):
        u = rng.random(f.shape[:2])
        f = f.copy()
        f[u < level / 2] = 0.0
        f[(u >= level / 2) & (u < level)] = 1.0
    else:
        raise ValueError(kind)
    return to_uint8(np.clip(f, 0, 1)), {"op": "add_noise", "kind": kind, "level": level, "seed": seed}


def glare(
    img: np.ndarray,
    center: tuple[float, float] | None = None,
    radius: float = 0.25,
    strength: float = 0.6,
    seed: int = 0,
) -> tuple[np.ndarray, dict]:
    """Additive elliptical specular highlight. `center` and `radius` are fractions of the image size."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    cx, cy = center if center is not None else (rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    rx, ry = radius * w, radius * h * rng.uniform(0.5, 1.0)
    d2 = ((xx - cx * w) / rx) ** 2 + ((yy - cy * h) / ry) ** 2
    hi = strength * np.exp(-d2 * 2.0)
    f = to_float(img)
    f = f + hi[..., None] if f.ndim == 3 else f + hi
    params = {
        "op": "glare",
        "cx": float(cx),
        "cy": float(cy),
        "radius": radius,
        "strength": strength,
        "seed": seed,
    }
    return to_uint8(np.clip(f, 0, 1)), params


def halftone(img: np.ndarray, cell: int = 4) -> tuple[np.ndarray, dict]:
    """Clustered-dot ordered dither per channel: simulates the dot screen of a printed photo."""
    if cell <= 1:
        return img.copy(), {"op": "halftone", "cell": cell}
    h, w = img.shape[:2]
    c = (np.arange(cell) + 0.5) / cell - 0.5
    d = np.sqrt(c[None, :] ** 2 + c[:, None] ** 2)
    order = d.ravel().argsort().argsort().reshape(cell, cell)  # centre-out threshold ranks
    t = (order + 0.5) / (cell * cell)
    T = np.tile(t, (h // cell + 1, w // cell + 1))[:h, :w].astype(np.float32)
    ink = 1.0 - to_float(cv2.blur(img, (cell, cell)))  # local darkness = dot size
    out = np.where(ink > (T[..., None] if img.ndim == 3 else T), 0.0, 1.0)
    # soften: paper never hits pure black/white
    out = cv2.GaussianBlur(out.astype(np.float32), (0, 0), cell / 4) * 0.85 + 0.08
    return to_uint8(out), {"op": "halftone", "cell": cell}


def perspective(img: np.ndarray, max_shift: float = 0.08, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Random homography: each corner moves by up to `max_shift` of the image size."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = src + rng.uniform(-max_shift, max_shift, (4, 2)).astype(np.float32) * np.float32([w, h])
    H = cv2.getPerspectiveTransform(src, dst)
    out = cv2.warpPerspective(img, H, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return out, {"op": "perspective", "max_shift": max_shift, "seed": seed, "H": H.round(5).tolist()}


def brightness_contrast(img: np.ndarray, alpha: float, beta: float) -> tuple[np.ndarray, dict]:
    """out = alpha * in + beta (beta in grey levels). Low light: alpha < 1, beta < 0."""
    out = to_uint8(np.clip(img.astype(np.float32) * alpha + beta, 0, 255))
    return out, {"op": "brightness_contrast", "alpha": alpha, "beta": beta}


def gamma(img: np.ndarray, k: float) -> tuple[np.ndarray, dict]:
    """Power-law darkening (k > 1) or brightening (k < 1)."""
    return to_uint8(to_float(img) ** k), {"op": "gamma", "k": k}


OPS = {
    "quantize_bits": quantize_bits,
    "downsample": downsample,
    "jpeg": jpeg,
    "gaussian_blur": gaussian_blur,
    "motion_blur": motion_blur,
    "add_noise": add_noise,
    "glare": glare,
    "halftone": halftone,
    "perspective": perspective,
    "brightness_contrast": brightness_contrast,
    "gamma": gamma,
}
SEEDED = {"add_noise", "glare", "perspective"}


def apply(img: np.ndarray, op: str, seed: int = 0, **params) -> tuple[np.ndarray, dict]:
    """Apply a degradation by name; seeded ops receive `seed`."""
    if op in SEEDED:
        params["seed"] = seed
    return OPS[op](img, **params)


def chain(img: np.ndarray, steps: list[tuple[str, dict]], seed: int = 0) -> tuple[np.ndarray, list[dict]]:
    """Apply a list of (op, params) in order; step i uses seed + i."""
    log = []
    for i, (op, p) in enumerate(steps):
        img, info = apply(img, op, seed=seed + i, **p)
        log.append(info)
    return img, log


def print_capture(img: np.ndarray, cfg: dict, seed: int = 0) -> tuple[np.ndarray, list[dict]]:
    """Print-then-capture chain from CLAUDE.md §6: halftone -> perspective -> glare -> JPEG."""
    pc = cfg["degrade"]["print_capture"]
    steps = [
        ("halftone", {"cell": pc["halftone_cell"]}),
        ("perspective", {"max_shift": pc["perspective_max_shift"]}),
        ("glare", {"radius": 0.3, "strength": pc["glare_strength"]}),
        ("jpeg", {"quality": pc["jpeg_quality"]}),
    ]
    return chain(img, steps, seed)
