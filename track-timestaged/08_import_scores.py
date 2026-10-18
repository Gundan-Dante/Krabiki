"""Step 5: import score matrices from another approach, so 05_combine.py can mix them in.

The other approach must save, for every subject 1..6, one array of shape (102, 102):
    rows    = test recordings in the ORIGINAL order of tst/sub-XX_tst.npy (never sorted)
    columns = candidate videos 1001, 1002, ..., 1102 in this order
    values  = higher means "more likely this video"

    python 08_import_scores.py --name proven --pattern "/path/to/sims/sub-{s:02d}.npy"

{s:02d} in the pattern is replaced by 01..06. The scores are saved as
outputs/sims/<name>_sub-XX.npz and printed for sub-01..04, so you can check
they match the numbers the other approach reports (if not, rows or columns
are in the wrong order).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tsx.data import DEV_SUBJECTS, SUBJECTS, TEST_IDS, load_test_labels
from tsx.evaluate import score_similarity

SIMS = Path("outputs/sims")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="short name, e.g. proven (no ':' or '|')")
    parser.add_argument("--pattern", required=True, help='e.g. "../other/sims/sub-{s:02d}.npy"')
    args = parser.parse_args()
    if ":" in args.name or "|" in args.name or "_sub-" in args.name:
        raise ValueError("name must not contain ':', '|' or '_sub-'")

    n = len(TEST_IDS)
    loaded = {}
    for s in SUBJECTS:
        path = Path(args.pattern.format(s=s))
        sim = np.load(path).astype(np.float64)
        if sim.shape != (n, n):
            raise ValueError(f"{path}: expected shape ({n}, {n}), got {sim.shape}")
        if not np.isfinite(sim).all():
            raise ValueError(f"{path}: contains NaN or inf")
        loaded[s] = sim

    SIMS.mkdir(parents=True, exist_ok=True)
    print(f"{'subject':<10}{'top-1':>8}{'top-5':>8}{'median':>8}")
    for s, sim in loaded.items():
        np.savez(SIMS / f"{args.name}_sub-{s:02d}.npz", **{"main|all": sim})
        if s in DEV_SUBJECTS:
            r = score_similarity(sim, load_test_labels(s))
            print(f"sub-{s:02d}    {r['top1']:>8.1%}{r['top5']:>8.1%}{r['median_rank']:>8.1f}")
        else:
            print(f"sub-{s:02d}    (held out)")
    print(f"Saved to {SIMS}/{args.name}_sub-XX.npz. Now run: python 05_combine.py")

if __name__ == "__main__":
    main()
