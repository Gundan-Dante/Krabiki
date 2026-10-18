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


# ---------------------------------------------------------------------------------------------------------------
# All-frames extraction: decode each video once (DataLoader workers), preprocess + embed every frame on the GPU.
# Any n-frame variant is then an exact index subset (same indices as read_frames), so it needs no re-extraction.
# Cache: features/<name>_all.npz with ids, n_video_frames (1102,), frames (1102, max_frames, D) float16 (zero-padded).
# Derived caches are written as features/<name>_all_f<n>.npz, so load_features(f"{name}_all", n, ...) works.
# ---------------------------------------------------------------------------------------------------------------

GpuEncoder = Callable[[torch.Tensor], torch.Tensor]  # (B, 3, H, W) float in [0, 1] on device -> (B, D)


def _clip_b32_gpu(device: str) -> GpuEncoder:
    """open_clip ViT-B-32 / laion2b_s34b_b79k with its preprocessing (bicubic resize to 224, CLIP mean/std) on GPU.

    The videos are square, so open_clip's resize + centre crop reduces to a plain resize.
    """
    import open_clip

    model = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")[0].eval().to(device)
    mean = torch.tensor(open_clip.OPENAI_DATASET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(open_clip.OPENAI_DATASET_STD, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def run(x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bicubic", antialias=True, align_corners=False)
        return model.encode_image((x.clamp(0, 1) - mean) / std).float()

    return run


def _dinov2_b_gpu(device: str) -> GpuEncoder:
    """timm ``vit_base_patch14_dinov2.lvd142m`` (DINOv2 ViT-B/14) at 224 px (16 x 16 patches), ImageNet mean/std.

    Returns ``[CLS token, mean of patch tokens]`` from the final normed layer, i.e. 768 + 768 = 1536 dims per frame.
    Use :func:`split_dinov2` to get the two halves as separate feature caches.
    """
    import timm

    model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, img_size=224).eval().to(device)
    cfg = model.pretrained_cfg
    mean = torch.tensor(cfg["mean"], device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device=device).view(1, 3, 1, 1)
    n_prefix = model.num_prefix_tokens

    @torch.no_grad()
    def run(x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bicubic", antialias=True, align_corners=False)
        tokens = model.forward_features((x.clamp(0, 1) - mean) / std)  # (B, prefix + 256, 768)
        return torch.cat([tokens[:, 0], tokens[:, n_prefix:].mean(dim=1)], dim=1).float()

    return run


GPU_ENCODERS: dict[str, Callable[[str], GpuEncoder]] = {"clip_b32": _clip_b32_gpu, "dinov2_b": _dinov2_b_gpu}


def split_dinov2(n_frames: int, feature_dir: Path = DEFAULT_FEATURE_DIR, name: str = "dinov2_b") -> list[Path]:
    """Split ``<name>_all_f<n>.npz`` ([CLS, patch mean]) into ``<name>_cls_all_f<n>`` and ``<name>_patch_all_f<n>``."""
    with np.load(feature_path(f"{name}_all", n_frames, feature_dir)) as f:
        ids, mean, frames = f["ids"], f["mean"], f["frames"]
    half = mean.shape[1] // 2
    outs = []
    for part, cols in (("cls", slice(None, half)), ("patch", slice(half, None))):
        out = feature_path(f"{name}_{part}_all", n_frames, feature_dir)
        np.savez(out, ids=ids, mean=mean[:, cols], frames=frames[..., cols])
        outs.append(out)
    return outs


class _VideoFrames(torch.utils.data.Dataset):
    """All RGB frames of one video as uint8 ``(F, H, W, 3)``."""

    def __init__(self, ids: np.ndarray, root: Path) -> None:
        self.ids, self.root = ids, root

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> np.ndarray:
        cv2.setNumThreads(1)
        cap = cv2.VideoCapture(str(video_path(int(self.ids[i]), self.root)))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not frames:
            raise RuntimeError(f"could not decode video {self.ids[i]}")
        return np.stack(frames)


def all_frames_path(name: str, feature_dir: Path = DEFAULT_FEATURE_DIR) -> Path:
    return Path(feature_dir) / f"{name}_all.npz"


def extract_all_frames(
    name: str,
    root: Path = DEFAULT_ROOT,
    feature_dir: Path = DEFAULT_FEATURE_DIR,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_workers: int = 4,
) -> Path:
    """Embed every frame of all 1,102 videos and cache them (see the section comment above)."""
    encode = GPU_ENCODERS[name](device)
    loader = torch.utils.data.DataLoader(_VideoFrames(ALL_IDS, root), batch_size=None, num_workers=num_workers)
    per_video = []
    for frames in tqdm(loader, desc=f"{name} all frames"):
        x = frames.to(device).permute(0, 3, 1, 2).float() / 255.0
        per_video.append(encode(x).cpu().numpy().astype(np.float16))
    counts = np.array([len(v) for v in per_video])
    out_frames = np.zeros((len(per_video), counts.max(), per_video[0].shape[1]), dtype=np.float16)
    for i, v in enumerate(per_video):
        out_frames[i, : len(v)] = v
    out = all_frames_path(name, feature_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, ids=ALL_IDS, n_video_frames=counts, frames=out_frames)
    return out


def derive_frame_subset(name: str, n_frames: int, feature_dir: Path = DEFAULT_FEATURE_DIR) -> Path:
    """Write ``<name>_all_f<n>.npz`` (ids, mean, frames) from the all-frames cache, with read_frames' indices."""
    with np.load(all_frames_path(name, feature_dir)) as f:
        ids, counts, frames = f["ids"], f["n_video_frames"], f["frames"]
    picked = np.stack([frames[i, ((np.arange(n_frames) + 0.5) * c / n_frames).astype(int)]
                       for i, c in enumerate(counts)])
    out = feature_path(f"{name}_all", n_frames, feature_dir)
    np.savez(out, ids=ids, mean=picked.astype(np.float32).mean(axis=1), frames=picked)
    return out


VIDEO_SECONDS = 3.0


def frame_times(n_frames: int, duration: float = VIDEO_SECONDS) -> np.ndarray:
    """Centre time (s) of each sampled frame, matching :func:`read_frames`."""
    return (np.arange(n_frames) + 0.5) * duration / n_frames


def pool_frames(frames: np.ndarray, pooling: str) -> np.ndarray:
    """Pool per-frame features ``(n, F, D)`` into one vector per video.

    ``first:<s>``      mean of the frames centred before ``s`` seconds (onset-weighted)
    ``concat``         all frames side by side, ``(n, F*D)``
    ``segments:<k>``   mean within k equal consecutive segments, concatenated, ``(n, k*D)``
    """
    kind, _, arg = pooling.partition(":")
    if kind == "first":
        keep = frame_times(frames.shape[1]) < float(arg)
        assert keep.any(), f"no frame centred before {arg} s"
        return frames[:, keep].mean(axis=1)
    if kind == "concat":
        return frames.reshape(len(frames), -1)
    if kind == "segments":
        return np.concatenate([seg.mean(axis=1) for seg in np.array_split(frames, int(arg), axis=1)], axis=1)
    raise ValueError(f"unknown pooling {pooling!r}")


def load_features(name: str, n_frames: int, ids: np.ndarray, feature_dir: Path = DEFAULT_FEATURE_DIR,
                  pooling: str = "mean") -> np.ndarray:
    """Pooled features ``(len(ids), D')`` for the requested video IDs (``pooling="mean"``: the cached frame mean)."""
    with np.load(feature_path(name, n_frames, feature_dir)) as f:
        all_ids = f["ids"]
        values = f["mean"] if pooling == "mean" else pool_frames(f["frames"].astype(np.float32), pooling)
    rows = np.searchsorted(all_ids, ids)
    assert np.array_equal(all_ids[rows], ids), "requested IDs missing from feature cache"
    return values[rows]
