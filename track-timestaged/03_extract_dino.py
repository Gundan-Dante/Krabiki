"""Step 2a: DINOv2 object features for all 1,102 videos (run once, ~10-20 min on CPU).

    python 03_extract_dino.py            # skips if the cache already exists
    python 03_extract_dino.py --force    # recompute

The first run downloads the model weights (~90 MB) from Hugging Face.
Output: cache/dino.npz with
    ids       (1102,)
    features  (1102, 8, 3, 768) float16   video x frame x layer x feature
    frame_times, layers, model, timm_version
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from tsx.data import video_path
from tsx.dino import FRAME_TIMES, LAYERS, MODEL_NAME, DinoExtractor, read_frames_at

OUT = Path("cache/dino.npz")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="recompute even if cached")
    args = parser.parse_args()
    if OUT.exists() and not args.force:
        print(f"{OUT} already exists (use --force to recompute)")
        return

    import timm
    torch.set_num_threads(os.cpu_count() or 1)
    extractor = DinoExtractor()
    ids = np.arange(1, 1103)
    feats = np.zeros((len(ids), len(FRAME_TIMES), len(LAYERS), 768), dtype=np.float16)
    for k, vid in enumerate(tqdm(ids, desc="videos")):
        feats[k] = extractor(read_frames_at(video_path(vid)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, ids=ids, features=feats, frame_times=FRAME_TIMES, layers=np.array(LAYERS),
             model=MODEL_NAME, timm_version=timm.__version__)
    print(f"Saved {OUT}: features {feats.shape}  (model {MODEL_NAME}, timm {timm.__version__})")

if __name__ == "__main__":
    main()
