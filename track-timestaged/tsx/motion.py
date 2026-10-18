"""Action features from MViT-v2-S, a video model (torchvision, trained on Kinetics-400).

Unlike the image models, it looks at 16 frames at once, so it can describe
movement and actions. Each video is read as 3 clips: the whole 3 s, the first
half and the second half. Features are taken at 4 depths of the network.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .dino import read_frames_at

MODEL_NAME = "torchvision mvit_v2_s, weights MViT_V2_S_Weights.KINETICS400_V1"
BLOCKS = (2, 8, 13, 15)        # 0-based, of 16 blocks: end of stage 2, middle and end of stage 3, last
CLIPS = {"full": (0.0, 3.0), "first": (0.0, 1.5), "second": (1.5, 3.0)}
CLIP_FRAMES = 16
IMG = 224
_MEAN = np.array([0.45, 0.45, 0.45], dtype=np.float32)
_STD = np.array([0.225, 0.225, 0.225], dtype=np.float32)

def clip_times() -> np.ndarray:
    """Frame times (s) of all clips back to back: 16 evenly spaced frames per clip."""
    return np.concatenate([
        t0 + (np.arange(CLIP_FRAMES) + 0.5) * (t1 - t0) / CLIP_FRAMES for t0, t1 in CLIPS.values()
    ])

def read_clips(path: str | Path) -> np.ndarray:
    """Return (3 clips, 16 frames, 224, 224, 3) uint8 RGB."""
    frames = read_frames_at(path, clip_times(), IMG)
    return frames.reshape(len(CLIPS), CLIP_FRAMES, IMG, IMG, 3)

class MotionExtractor:
    """For each clip and block: [class token, mean of the other tokens]."""

    def __init__(self, device: str = "cpu") -> None:
        from torchvision.models.video import MViT_V2_S_Weights, mvit_v2_s
        self.model = mvit_v2_s(weights=MViT_V2_S_Weights.KINETICS400_V1).eval().to(device)
        self.device = device
        self._out: dict[int, torch.Tensor] = {}
        for i in BLOCKS:
            self.model.blocks[i].register_forward_hook(self._hook(i))

    def _hook(self, i: int):
        def fn(_module, _inp, out):
            self._out[i] = out[0] if isinstance(out, tuple) else out
        return fn

    @torch.no_grad()
    def __call__(self, clips: np.ndarray) -> dict[int, np.ndarray]:
        """clips (k, 16, 224, 224, 3) uint8 -> {block: (k, 2 * channels) float32}."""
        out: dict[int, list[np.ndarray]] = {i: [] for i in BLOCKS}
        for clip in clips:  # one clip at a time keeps memory low
            x = (clip.astype(np.float32) / 255.0 - _MEAN) / _STD
            x = torch.from_numpy(x).permute(3, 0, 1, 2)[None].to(self.device)   # (1, 3, 16, H, W)
            self.model(x)
            for i in BLOCKS:
                tok = self._out[i]
                out[i].append(torch.cat([tok[:, 0], tok[:, 1:].mean(dim=1)], dim=1)[0].cpu().numpy())
        return {i: np.stack(v).astype(np.float32) for i, v in out.items()}
