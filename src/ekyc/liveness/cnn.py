"""Learned liveness: ImageNet-pretrained MobileNetV3-Small fine-tuned on face crops (live vs attack).

Course link: fixed hand-designed kernels (Sobel, Laplacian, LoG) -> learned convolution kernels.
Requires the optional `cnn` extra (torch, torchvision). Imported lazily so the rest of the pipeline
runs without torch.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)
MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


def _tensor(crops: list[np.ndarray], size: int):
    import torch

    x = np.stack(
        [
            cv2.resize(cv2.cvtColor(c, cv2.COLOR_BGR2RGB), (size, size), interpolation=cv2.INTER_AREA)
            for c in crops
        ]
    )
    x = (x.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(0, 3, 1, 2).copy())


def build(pretrained: bool = True):
    from torch import nn
    from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

    m = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 2)
    return m


def train(crops: list[np.ndarray], labels: list[int], cfg: dict, out: Path, seed: int = 0, val=None):
    """Fine-tune and save the state dict. labels: 1 = live. Light augmentation: flips + brightness jitter."""
    import torch
    from torch import nn

    c = cfg["liveness"]["cnn"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = build()
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"])
    y = torch.tensor(labels)
    for epoch in range(c["epochs"]):
        model.train()
        order = rng.permutation(len(crops))
        losses = []
        for i in range(0, len(order), c["batch"]):
            idx = order[i : i + c["batch"]]
            batch = [crops[j][:, ::-1] if rng.random() < 0.5 else crops[j] for j in idx]
            batch = [
                np.clip(b.astype(np.float32) * rng.uniform(0.8, 1.2), 0, 255).astype(np.uint8) for b in batch
            ]
            loss = nn.functional.cross_entropy(model(_tensor(batch, c["img_size"])), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        log.info("epoch %d loss %.4f", epoch, float(np.mean(losses)))
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    return model


def load(p: Path):
    import torch

    m = build(pretrained=False)
    m.load_state_dict(torch.load(p, map_location="cpu"))
    return m.eval()


def predict(model, crops: list[np.ndarray], cfg: dict) -> np.ndarray:
    """P(live) per crop."""
    import torch

    with torch.no_grad():
        return torch.softmax(model(_tensor(crops, cfg["liveness"]["cnn"]["img_size"])), 1)[:, 1].numpy()


def embed(model, crops: list[np.ndarray], cfg: dict) -> np.ndarray:
    """Penultimate-layer features, for the fused CNN + handcrafted model."""
    import torch

    with torch.no_grad():
        x = model.avgpool(model.features(_tensor(crops, cfg["liveness"]["cnn"]["img_size"]))).flatten(1)
        return x.numpy()
