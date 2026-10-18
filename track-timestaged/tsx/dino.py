"""Object features from DINOv2 (a pretrained image model), read at several depths.

Shallow layers describe shapes and parts, the last layer describes whole objects.
Model: timm `vit_small_patch14_dinov2.lvd142m` (Meta DINOv2 ViT-S/14, public, 2023).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"
LAYERS = (3, 7, 11)                              # blocks 4, 8 and 12 of 12 (0-based)
N_FRAMES = 8
FRAME_TIMES = (np.arange(N_FRAMES) + 0.5) * 3.0 / N_FRAMES   # 0.19 s ... 2.81 s
IMG = 224
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def read_frames_at(path: str | Path, times: np.ndarray = FRAME_TIMES, size: int = IMG) -> np.ndarray:
    """Return the frames shown at `times` (seconds), RGB uint8, shape (k, size, size, 3)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)[:, :, ::-1])
    cap.release()
    if not frames:
        raise IOError(f"no frames read from {path}")
    if not fps or not np.isfinite(fps) or fps <= 0:
        fps = len(frames) / 3.0
    idx = np.clip(np.round(np.asarray(times) * fps).astype(int), 0, len(frames) - 1)
    return np.stack([frames[i] for i in idx])

class DinoExtractor:
    """For each frame and layer: [class token, mean of patch tokens] -> 768 numbers."""

    def __init__(self, device: str = "cpu") -> None:
        import timm
        self.model = timm.create_model(MODEL_NAME, pretrained=True, img_size=IMG).eval().to(device)
        self.device = device
        self.n_prefix = getattr(self.model, "num_prefix_tokens", 1)
        self._out: dict[int, torch.Tensor] = {}
        for i in LAYERS:
            self.model.blocks[i].register_forward_hook(self._hook(i))

    def _hook(self, i: int):
        def fn(_module, _inp, out):
            self._out[i] = out
        return fn

    @torch.no_grad()
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        """frames (k, 224, 224, 3) uint8 -> features (k, len(LAYERS), 768) float32."""
        x = (frames.astype(np.float32) / 255.0 - _MEAN) / _STD
        x = torch.from_numpy(x).permute(0, 3, 1, 2).to(self.device)
        self.model.forward_features(x)
        feats = []
        for i in LAYERS:
            tok = self._out[i]
            cls = tok[:, 0]
            patches = tok[:, self.n_prefix:].mean(dim=1)
            feats.append(torch.cat([cls, patches], dim=1))
        return torch.stack(feats, dim=1).cpu().numpy().astype(np.float32)
