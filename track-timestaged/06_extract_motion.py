"""Step 4a: MViT-v2 action features for all 1,102 videos (run once, about 1-2 h on CPU).

    python 06_extract_motion.py            # skips if the cache already exists
    python 06_extract_motion.py --force    # recompute from scratch

Progress is saved every 25 videos. If the run is interrupted (Ctrl+C, laptop
sleeps), just start it again and it continues where it stopped.
The first run downloads the model weights (~140 MB).

Output: cache/motion.npz with
    ids               (1102,)
    block3, block9, block14, block16   (1102, 3 clips, 2 x channels) float16
    clips             names of the 3 clips (full, first, second)
    blocks, model, torchvision_version
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from tsx.data import video_path
from tsx.motion import BLOCKS, CLIPS, MODEL_NAME, MotionExtractor, read_clips

OUT = Path("cache/motion.npz")
SAVE_EVERY = 25

def extract(ids: np.ndarray, out: Path, extractor, save_every: int = SAVE_EVERY) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.stem + "_partial.npz")
    feats, start = None, 0
    if partial.exists():
        with np.load(partial) as z:
            if np.array_equal(z["ids"], ids):
                start = int(z["n_done"])
                feats = {k: z[k].copy() for k in z.files if k.startswith("block")}
                print(f"Resuming after {start} videos")

    for k in tqdm(range(start, len(ids)), initial=start, total=len(ids), desc="videos"):
        f = extractor(read_clips(video_path(int(ids[k]))))
        if feats is None:
            feats = {f"block{i + 1}": np.zeros((len(ids), *v.shape), np.float16) for i, v in f.items()}
        for i, v in f.items():
            feats[f"block{i + 1}"][k] = v
        if (k + 1) % save_every == 0 and k + 1 < len(ids):
            tmp = partial.with_name(partial.stem + "_tmp.npz")
            np.savez(tmp, ids=ids, n_done=k + 1, **feats)
            os.replace(tmp, partial)       # never leaves a half-written checkpoint

    import torchvision
    np.savez(out, ids=ids, clips=np.array(list(CLIPS)), blocks=np.array(BLOCKS),
             model=MODEL_NAME, torchvision_version=torchvision.__version__, **feats)
    partial.unlink(missing_ok=True)
    print(f"Saved {out}: " + ", ".join(f"{k} {v.shape}" for k, v in feats.items()))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="recompute from scratch")
    args = parser.parse_args()
    if OUT.exists() and not args.force:
        print(f"{OUT} already exists (use --force to recompute)")
        return
    if args.force:
        OUT.with_name(OUT.stem + "_partial.npz").unlink(missing_ok=True)

    torch.set_num_threads(os.cpu_count() or 1)
    extract(np.arange(1, 1103), OUT, MotionExtractor())

if __name__ == "__main__":
    main()
