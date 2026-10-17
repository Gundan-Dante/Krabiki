"""Frame sampling and frozen pretrained stimulus features, cached to ``features/<name>_f<n>.npz``.

Pretrained models (all public, released well before Oct 16, 2026):
    alexnet   torchvision AlexNet, weights IMAGENET1K_V1, layer ``features.3`` (conv2, pre-ReLU)
    clip_b32  open_clip ViT-B-32 / laion2b_s34b_b79k, image embedding (``encode_image``)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
from tqdm import tqdm

from eegmr.data import ALL_IDS, DEFAULT_ROOT, video_path

DEFAULT_FEATURE_DIR = Path("features")
MAX_PER_FRAME_DIM = 4096  # per-frame features are cached only when they are this small

Extractor = Callable[[list[np.ndarray]], np.ndarray]


def read_frames(path: Path, n_frames: int) -> list[np.ndarray]:
    """Return ``n_frames`` RGB frames at the centres of ``n_frames`` equal segments of the video.

    ``n_frames=1`` gives the middle frame.
    """
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"could not decode {path}")
    idx = ((np.arange(n_frames) + 0.5) * len(frames) / n_frames).astype(int)
    return [cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB) for i in idx]


def _alexnet(device: str) -> Extractor:
    from torchvision.models import AlexNet_Weights, alexnet

    net = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1).features[:4].eval().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def run(frames: list[np.ndarray]) -> np.ndarray:
        x = np.stack([cv2.resize(f, (224, 224), interpolation=cv2.INTER_AREA) for f in frames])
        x = torch.from_numpy(x).to(device).permute(0, 3, 1, 2).float() / 255.0
        return net((x - mean) / std).flatten(1).cpu().numpy()

    return run


def _clip_b32(device: str) -> Extractor:
    import open_clip
    from PIL import Image

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    model = model.eval().to(device)

    @torch.no_grad()
    def run(frames: list[np.ndarray]) -> np.ndarray:
        x = torch.stack([preprocess(Image.fromarray(f)) for f in frames]).to(device)
        return model.encode_image(x).float().cpu().numpy()

    return run


EXTRACTORS: dict[str, Callable[[str], Extractor]] = {"alexnet": _alexnet, "clip_b32": _clip_b32}


def feature_path(name: str, n_frames: int, feature_dir: Path = DEFAULT_FEATURE_DIR) -> Path:
    return Path(feature_dir) / f"{name}_f{n_frames}.npz"


def extract_features(
    name: str,
    n_frames: int,
    root: Path = DEFAULT_ROOT,
    feature_dir: Path = DEFAULT_FEATURE_DIR,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Path:
    """Compute features for all 1,102 videos and cache ``ids``, ``mean`` and (if small) ``frames``."""
    extractor = EXTRACTORS[name](device)
    means, per_frame = [], []
    for vid in tqdm(ALL_IDS, desc=f"{name} f{n_frames}"):
        feats = extractor(read_frames(video_path(int(vid), root), n_frames))
        means.append(feats.mean(axis=0))
        if feats.shape[1] <= MAX_PER_FRAME_DIM:
            per_frame.append(feats.astype(np.float16))
    arrays = {"ids": ALL_IDS, "mean": np.stack(means).astype(np.float32)}
    if per_frame:
        arrays["frames"] = np.stack(per_frame)
    out = feature_path(name, n_frames, feature_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arrays)
    return out


def load_features(name: str, n_frames: int, ids: np.ndarray, feature_dir: Path = DEFAULT_FEATURE_DIR) -> np.ndarray:
    """Frame-averaged features ``(len(ids), D)`` for the requested video IDs."""
    with np.load(feature_path(name, n_frames, feature_dir)) as f:
        all_ids, mean = f["ids"], f["mean"]
    rows = np.searchsorted(all_ids, ids)
    assert np.array_equal(all_ids[rows], ids), "requested IDs missing from feature cache"
    return mean[rows]
