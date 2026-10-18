"""Step 1a: measure simple visual features for all 1,102 videos (run once).

    python 01_extract_lowlevel.py            # skips if the cache already exists
    python 01_extract_lowlevel.py --force    # recompute

Output: cache/lowlevel.npz with
    ids       (1102,)        video IDs 1..1102
    features  (1102, 30, 40) features per 0.1 s bin
    names     (40,)          feature names
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from tsx.data import video_path
from tsx.video_features import FEATURE_NAMES, lowlevel_features, read_video

OUT = Path("cache/lowlevel.npz")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="recompute even if cached")
    args = parser.parse_args()
    if OUT.exists() and not args.force:
        print(f"{OUT} already exists (use --force to recompute)")
        return

    ids = np.arange(1, 1103)
    feats, failed = [], []
    for vid in tqdm(ids, desc="videos"):
        try:
            frames, fps = read_video(video_path(vid))
            feats.append(lowlevel_features(frames, fps))
        except IOError as err:
            failed.append((vid, str(err)))
    if failed:
        raise RuntimeError(f"{len(failed)} videos failed, first: {failed[:3]}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, ids=ids, features=np.stack(feats), names=np.array(FEATURE_NAMES))
    print(f"Saved {OUT}: features {np.stack(feats).shape}")

if __name__ == "__main__":
    main()
