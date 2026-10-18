"""Simple ("low-level") visual features of a video, measured over time.

Every video is cut into 30 bins of 0.1 s. For each bin we measure brightness,
contrast, color, saturation, edges, motion, and where on screen brightness and
motion are (a 4x4 grid). Result per video: an array of shape (30, 40).
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

N_BINS = 30          # 3 s / 0.1 s
BIN_SEC = 0.1
FRAME_SIZE = 64      # frames are shrunk to 64x64 pixels before measuring
GRID = 4             # 4x4 screen regions

FEATURE_NAMES = (
    ["brightness", "contrast", "red", "green", "blue", "saturation"]
    + [f"brightness_r{r}c{c}" for r in range(GRID) for c in range(GRID)]
    + ["edges", "motion"]
    + [f"motion_r{r}c{c}" for r in range(GRID) for c in range(GRID)]
)
N_FEATURES = len(FEATURE_NAMES)  # 40

def read_video(path: str | Path) -> tuple[np.ndarray, float]:
    """Return (frames, fps). frames: (n, 64, 64, 3) RGB floats in [0, 1]."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA)
        frames.append(frame[:, :, ::-1].astype(np.float32) / 255.0)  # BGR -> RGB
    cap.release()
    if not frames:
        raise IOError(f"no frames read from {path}")
    if not fps or not np.isfinite(fps) or fps <= 0:
        fps = len(frames) / 3.0
    return np.stack(frames), float(fps)

def lowlevel_features(frames: np.ndarray, fps: float) -> np.ndarray:
    """Measure the 40 features in each 0.1 s bin. Returns (30, 40) float32."""
    gray = frames @ np.array([0.299, 0.587, 0.114], dtype=np.float32)   # (n, H, W)
    diff = np.zeros_like(gray)
    diff[1:] = np.abs(gray[1:] - gray[:-1])                              # frame-to-frame change
    times = np.arange(len(frames)) / fps

    out = np.zeros((N_BINS, N_FEATURES), dtype=np.float32)
    for b in range(N_BINS):
        sel = np.flatnonzero((times >= b * BIN_SEC) & (times < (b + 1) * BIN_SEC))
        if sel.size == 0:  # low frame rate or video shorter than 3 s: use nearest frame
            sel = np.array([min(int(round((b + 0.5) * BIN_SEC * fps)), len(frames) - 1)])
        out[b] = _bin_features(frames[sel], gray[sel], diff[sel])
    return out

def _grid_means(img: np.ndarray) -> np.ndarray:
    """(k, H, W) -> average over frames, then mean of each 4x4 screen region (16,)."""
    m = img.mean(axis=0)
    h, w = m.shape
    return m.reshape(GRID, h // GRID, GRID, w // GRID).mean(axis=(1, 3)).ravel()

def _bin_features(rgb: np.ndarray, gray: np.ndarray, diff: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(gray, axis=(1, 2))
    edges = np.sqrt(gx**2 + gy**2).mean()
    mx, mn = rgb.max(axis=-1), rgb.min(axis=-1)
    saturation = ((mx - mn) / np.maximum(mx, 1e-6)).mean()
    return np.concatenate([
        [gray.mean(), gray.std(axis=(1, 2)).mean()],
        rgb.mean(axis=(0, 1, 2)),
        [saturation],
        _grid_means(gray),
        [edges, diff.mean()],
        _grid_means(diff),
    ])
