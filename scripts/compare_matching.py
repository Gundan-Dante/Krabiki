"""Per-subject and mean scores of logged experiments, with a paired Wilcoxon test on per-trial ranks vs a baseline.

Reads the similarity matrices that predict.py saves under results/similarity/<experiment>/ and scores them against
the dev subjects' test labels (validation only).

Example:
    python scripts/compare_matching.py --baseline match_base match_erp match_weighted --subjects 1 2 3 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from eegmr.data import DEFAULT_ROOT, load_test_labels
from eegmr.evaluate import true_ranks


def trial_ranks(experiment: str, subjects: list[int], root: Path, sim_dir: Path) -> dict[int, np.ndarray]:
    return {s: true_ranks(np.load(sim_dir / experiment / f"sub-{s:02d}.npy"), load_test_labels(s, root))
            for s in subjects}


def compare_table(baseline: str, experiments: list[str], subjects: list[int], root: Path = DEFAULT_ROOT,
                  sim_dir: Path = Path("results/similarity")) -> str:
    """Markdown table: per-subject top-1 / top-5 / median rank, means, and the paired Wilcoxon vs ``baseline``."""
    base = trial_ranks(baseline, subjects, root, sim_dir)
    base_all = np.concatenate([base[s] for s in subjects])
    header = "| experiment | " + " | ".join(f"sub-{s:02d} top1/top5/med" for s in subjects) + \
             " | mean top-1 | mean top-5 | mean median rank | better/worse | Wilcoxon p |"
    lines = [header, "|" + "---|" * (header.count("|") - 1)]
    for name in [baseline, *experiments]:
        ranks = base if name == baseline else trial_ranks(name, subjects, root, sim_dir)
        cells = [f"{np.mean(ranks[s] <= 1):.1%} / {np.mean(ranks[s] <= 5):.1%} / {np.median(ranks[s]):.1f}"
                 for s in subjects]
        top1 = np.mean([np.mean(ranks[s] <= 1) for s in subjects])
        top5 = np.mean([np.mean(ranks[s] <= 5) for s in subjects])
        med = np.mean([np.median(ranks[s]) for s in subjects])
        if name == baseline:
            cmp = "– | –"
        else:
            all_ranks = np.concatenate([ranks[s] for s in subjects])
            diff = base_all - all_ranks  # > 0: the experiment ranks the true video higher
            p = wilcoxon(base_all, all_ranks).pvalue if np.any(diff != 0) else float("nan")
            cmp = f"{int((diff > 0).sum())} / {int((diff < 0).sum())} | {p:.3g}"
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {top1:.1%} | {top5:.1%} | {med:.1f} | {cmp} |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--baseline", required=True)
    p.add_argument("experiments", nargs="+", help="experiments to compare with the baseline")
    p.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--sim-dir", type=Path, default=Path("results/similarity"))
    args = p.parse_args()
    print(compare_table(args.baseline, args.experiments, args.subjects, args.root, args.sim_dir))


if __name__ == "__main__":
    main()
