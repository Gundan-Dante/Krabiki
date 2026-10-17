"""Cache frozen stimulus features for all 1,102 videos.

Example:
    python scripts/extract_features.py --model clip_b32 --frames 1 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eegmr.data import DEFAULT_ROOT
from eegmr.features import DEFAULT_FEATURE_DIR, EXTRACTORS, extract_features


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=sorted(EXTRACTORS))
    p.add_argument("--frames", type=int, nargs="+", default=[8], help="frames per video (one cache file each)")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    p.add_argument("--device", default=None, help="cuda / cpu (default: cuda if available)")
    args = p.parse_args()

    kwargs = {"device": args.device} if args.device else {}
    for n in args.frames:
        print("wrote", extract_features(args.model, n, args.root, args.feature_dir, **kwargs))


if __name__ == "__main__":
    main()
