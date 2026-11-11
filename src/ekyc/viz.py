"""Figure helpers for report panels (matplotlib, Agg backend)."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _rgb(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save(fig, p: str | Path) -> Path:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return p


def panel(
    images: dict[str, np.ndarray], p: str | Path, cols: int = 4, title: str | None = None, hist: bool = False
) -> Path:
    """Grid of titled images; with `hist`, each image gets its grey-level histogram underneath."""
    n = len(images)
    cols = min(cols, n)
    rows = int(np.ceil(n / cols)) * (2 if hist else 1)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.8 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for i, (name, img) in enumerate(images.items()):
        r, c = divmod(i, cols)
        r *= 2 if hist else 1
        ax = axes[r, c]
        ax.imshow(_rgb(img), cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=255)
        ax.set_title(name, fontsize=9)
        if hist:
            g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hx = axes[r + 1, c]
            hx.axis("on")
            counts = np.bincount(g.ravel(), minlength=256)
            hx.bar(np.arange(256), counts, width=1.0, color="0.3")
            hx.set_xlim(0, 255)
            hx.set_ylim(
                0, np.percentile(counts, 99) * 1.15 + 1
            )  # one clipped bin (0/255) must not flatten the rest
            hx.set_yticks([])
            hx.tick_params(labelsize=7)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return save(fig, p)


def lines(
    df,
    x: str,
    y: str,
    hue: str | None,
    p: str | Path,
    title: str = "",
    ylabel: str | None = None,
    logx: bool = False,
    invert_x: bool = False,
) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    groups = df.groupby(hue) if hue else [(y, df)]
    for name, g in groups:
        g = g.sort_values(x)
        ax.plot(g[x], g[y], marker="o", label=str(name))
        if f"{y}_lo" in g:
            ax.fill_between(g[x], g[f"{y}_lo"], g[f"{y}_hi"], alpha=0.15)
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel or y)
    ax.set_title(title)
    if logx:
        ax.set_xscale("log")
        xs = sorted(df[x].unique())
        ax.set_xticks(xs, [f"{v:g}" for v in xs])
        ax.minorticks_off()
    if invert_x:
        ax.invert_xaxis()
    ax.grid(alpha=0.3)
    if hue:
        ax.legend(fontsize=8)
    fig.tight_layout()
    return save(fig, p)


def bars(df, x: str, y: str, p: str | Path, title: str = "", hue: str | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    if hue:
        piv = df.pivot_table(index=x, columns=hue, values=y)
        piv.plot.bar(ax=ax, rot=0)
    else:
        ax.bar(df[x].astype(str), df[y], color="0.35")
        if f"{y}_lo" in df:
            ax.errorbar(
                df[x].astype(str),
                df[y],
                yerr=[df[y] - df[f"{y}_lo"], df[f"{y}_hi"] - df[y]],
                fmt="none",
                c="k",
            )
    ax.set_ylabel(y)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    return save(fig, p)


def roc(curves: dict[str, tuple[np.ndarray, np.ndarray]], p: str | Path, title: str = "ROC") -> Path:
    fig, ax = plt.subplots(figsize=(4.5, 4.2))
    for name, (fpr, tpr) in curves.items():
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xlim(0, 1)
    ax.set_xlabel("FAR (false accept rate)")
    ax.set_ylabel("TAR (1 - FRR)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return save(fig, p)


def accumulator(h, theta, rho, peaks, img: np.ndarray, p: str | Path) -> Path:
    """Hough rho-theta accumulator with peaks next to the source image."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    a.imshow(_rgb(img))
    a.set_title("input")
    a.axis("off")
    b.imshow(
        np.log1p(h),
        extent=[np.degrees(theta[0]), np.degrees(theta[-1]), rho[-1], rho[0]],
        aspect="auto",
        cmap="magma",
    )
    _, angles, dists = peaks
    b.plot(np.degrees(angles), dists, "c+", ms=12, mew=2)
    b.set_xlabel("theta (deg)")
    b.set_ylabel("rho (px)")
    b.set_title("Hough accumulator (log) + peaks")
    fig.tight_layout()
    return save(fig, p)
