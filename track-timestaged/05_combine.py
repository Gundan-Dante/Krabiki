"""Step 3: combine the experts into one ranking.

Every expert saved one score matrix per (feature set, brain window). This
script tries four ways of adding them up, from "choose nothing" to "choose a
lot", and measures each one honestly:

  equal weights        add every matrix with weight 1 (nothing is fitted)
  per-window weights   average the matrices inside each window, then give each
                       window a weight of 0, 1 or 2 (3 numbers are fitted)
  expert x window      average the matrices of each expert inside each window,
                       then give each (expert, window) pair a weight of 0, 1 or 2
                       (one number per pair; experts with many feature sets
                       can no longer drown out the others)
  greedy selection     pick individual matrices one by one (many choices)

Choices are made with the test labels of sub-01..04, which the rules allow for
validation / model selection. Two numbers per strategy:
  in-sample  choose and score on the same 4 subjects (optimistic)
  LOSO       choose on 3 subjects, score on the 4th, for each of the 4
             (leave-one-subject-out; the honest estimate for sub-05/06)

The submission files use the strategy with the best LOSO result.

    python 05_combine.py                          # all experts in outputs/sims
    python 05_combine.py --experts lowlevel dino  # only some
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from tsx.data import DEV_SUBJECTS, SUBJECTS, TEST_IDS, load_test_labels
from tsx.encoding import combine
from tsx.evaluate import score_ranking, score_similarity
from tsx.submission import write_submission

OUT = Path("outputs")
SIMS = OUT / "sims"
MAX_STEPS = 12
WINDOW_NAMES = ("early", "middle", "late")
METRICS = ("top1", "top5", "median_rank")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experts", nargs="*", help="default: every expert in outputs/sims")
    args = parser.parse_args()
    experts = args.experts or sorted({p.name.rsplit("_sub-", 1)[0] for p in SIMS.glob("*_sub-*.npz")})
    if not experts:
        raise FileNotFoundError(f"no score files in {SIMS}; run the expert scripts first")

    cells = load_cells(experts)
    wins = group_averages(cells, lambda n: window_of(n))
    pairs = group_averages(cells, lambda n: f"{n.split(':')[0]}|{window_of(n)}")
    labels = {s: load_test_labels(s) for s in DEV_SUBJECTS}
    print(f"Experts: {', '.join(experts)}  ({len(cells)} score matrices, windows: {', '.join(wins)})")

    # Each strategy: given the subjects to choose on -> (description, subject -> scores).
    def equal(_sel):
        return f"all {len(cells)} matrices, weight 1", lambda s: ensemble(cells, list(cells), s)

    def per_window(sel):
        wt = select_weights(wins, sel, labels)
        desc = ", ".join(f"{w} {k}" for w, k in wt.items())
        return desc, lambda s: weighted(wins, wt, s)

    def per_pair(sel):
        wt = select_weights(pairs, sel, labels)
        desc = ", ".join(f"{p} {k}" for p, k in wt.items())
        return desc, lambda s: weighted(pairs, wt, s)

    def greedy(sel):
        chosen = greedy_select(cells, sel, labels)
        desc = "; ".join(f"{k}x {n}" for n, k in Counter(chosen).most_common())
        return desc, lambda s: ensemble(cells, chosen, s)

    strategies = {"equal weights": equal, "per-window weights": per_window,
                  "expert x window": per_pair, "greedy selection": greedy}

    results = {}
    for name, fit in strategies.items():
        desc, sim_of = fit(DEV_SUBJECTS)
        in_sample = {s: score_similarity(sim_of(s), labels[s]) for s in DEV_SUBJECTS}
        loso, loso_mrr = {}, []
        for held in DEV_SUBJECTS:
            _, held_sim = fit([s for s in DEV_SUBJECTS if s != held])
            loso[held] = score_similarity(held_sim(held), labels[held])
            loso_mrr.append(mean_mrr({held: held_sim(held)}, [held], labels))
        results[name] = {"desc": desc, "sim_of": sim_of, "in_sample": in_sample,
                         "loso": loso, "loso_mrr": float(np.mean(loso_mrr))}

    best = max(results, key=lambda n: results[n]["loso_mrr"])
    print_report(results, best)

    out_dir = OUT / "submissions" / "ensemble"
    sim_of = results[best]["sim_of"]
    for s in SUBJECTS:
        path = write_submission(sim_of(s), s, out_dir)
        if s in labels:  # re-score from the saved file, exactly as the organizers will
            assert score_ranking(np.load(path), labels[s]) == results[best]["in_sample"][s]
    (OUT / "ensemble.json").write_text(json.dumps(
        {"experts": experts, "strategy": best, "choice": results[best]["desc"]}, indent=2))
    print(f"Submissions ({best}) -> {out_dir}/   choice saved to {OUT / 'ensemble.json'}")
    append_csv(results)

def load_cells(experts: list[str]) -> dict[str, dict[int, np.ndarray]]:
    """'expert:features|window' -> {subject: row-z-scored (102, 102) scores}.
    Each expert's own 'combined' matrices are skipped."""
    cells: dict[str, dict[int, np.ndarray]] = {}
    for expert in experts:
        for s in SUBJECTS:
            path = SIMS / f"{expert}_sub-{s:02d}.npz"
            if not path.exists():
                raise FileNotFoundError(f"{path} missing; re-run the {expert} expert")
            with np.load(path) as z:
                for key in z.files:
                    if key.split("|")[-1] != "combined":
                        cells.setdefault(f"{expert}:{key}", {})[s] = combine([z[key]])
    return cells

def window_of(name: str) -> str:
    return name.split("|")[-1].split(":")[-1]

def window_rank(window: str) -> int:
    """Sort order: early, middle, late, then anything else (e.g. 'all' from imported scores)."""
    return WINDOW_NAMES.index(window) if window in WINDOW_NAMES else len(WINDOW_NAMES)

def group_averages(cells, group_of) -> dict[str, dict[int, np.ndarray]]:
    """Average the matrices in each group, z-scored again so groups are comparable.
    Groups are ordered by window (early, middle, late)."""
    groups: dict[str, list[str]] = {}
    for n in cells:
        groups.setdefault(group_of(n), []).append(n)
    order = sorted(groups, key=lambda g: (window_rank(window_of(g)), g))
    return {g: {s: combine([sum(cells[n][s] for n in groups[g]) / len(groups[g])]) for s in SUBJECTS}
            for g in order}

def weighted(groups, weights: dict[str, int], subject: int) -> np.ndarray:
    return sum(k * groups[g][subject] for g, k in weights.items() if k)

def select_weights(groups, subjects, labels) -> dict[str, int]:
    """Weight 0, 1 or 2 per group, chosen by mean reciprocal rank on `subjects`.
    Up to 6 groups: try every combination. More: change one weight at a time
    (starting from all 1) until no single change helps."""
    def score(wt):
        return mean_mrr({s: weighted(groups, wt, s) for s in subjects}, subjects, labels)

    keys = list(groups)
    if len(keys) <= 6:
        grid = [dict(zip(keys, ks)) for ks in itertools.product((0, 1, 2), repeat=len(keys)) if any(ks)]
        return max(grid, key=score)
    wt = {k: 1 for k in keys}
    best, improved = score(wt), True
    while improved:
        improved = False
        for k in keys:
            for v in (0, 1, 2):
                trial = {**wt, k: v}
                if v == wt[k] or not any(trial.values()):
                    continue
                sc = score(trial)
                if sc > best + 1e-12:
                    wt, best, improved = trial, sc, True
    return wt

def true_ranks(sim: np.ndarray, true_ids: np.ndarray) -> np.ndarray:
    """Rank (1 = best) of the true video in each row."""
    cols = np.searchsorted(TEST_IDS, true_ids)
    true_score = sim[np.arange(len(sim)), cols]
    return 1 + (sim > true_score[:, None]).sum(axis=1)

def mean_mrr(sims: dict[int, np.ndarray], subjects, labels) -> float:
    """Mean reciprocal rank: smoother than top-1, but still rewards being near the top."""
    return float(np.mean([np.mean(1.0 / true_ranks(sims[s], labels[s])) for s in subjects]))

def ensemble(cells, names: list[str], subject: int) -> np.ndarray:
    return sum(cells[n][subject] for n in names)

def greedy_select(cells, subjects, labels, max_steps: int = MAX_STEPS) -> list[str]:
    chosen: list[str] = []
    total = {s: np.zeros((len(TEST_IDS), len(TEST_IDS))) for s in subjects}
    best = -np.inf
    for _ in range(max_steps):
        gains = {n: mean_mrr({s: total[s] + c[s] for s in subjects}, subjects, labels)
                 for n, c in cells.items()}
        name = max(gains, key=gains.get)
        if gains[name] <= best:
            break
        best = gains[name]
        chosen.append(name)
        for s in subjects:
            total[s] += cells[name][s]
    return chosen

def _mean(per_s: dict) -> dict[str, float]:
    return {k: float(np.mean([r[k] for r in per_s.values()])) for k in METRICS}

def print_report(results: dict, best: str) -> None:
    print()
    print("Mean over sub-01..04   (chance: top-1 1.0%, top-5 4.9%, median rank 51.5)")
    print("Note: the used strategy is picked by its LOSO score, so that one number is slightly optimistic.")
    print(f"{'strategy':<22}{'in-sample (optimistic)':>26}{'LOSO (honest)':>26}")
    print(f"{'':<22}" + f"{'top-1':>8}{'top-5':>8}{'median':>8}  " * 2)
    for name, r in results.items():
        a, b = _mean(r["in_sample"]), _mean(r["loso"])
        cells = "".join(f"{m['top1']:>8.1%}{m['top5']:>8.1%}{m['median_rank']:>8.1f}  " for m in (a, b))
        print(f"{name:<22}{cells}" + ("<- used" if name == best else ""))
    print()
    for name, r in results.items():
        print(f"{name}: {r['desc']}")
    print()

def append_csv(results: dict) -> None:
    path = OUT / "results.csv"
    new = not path.exists()
    stamp = datetime.now().isoformat(timespec="seconds")
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["time", "expert", "subject", "window", *METRICS])
        if new:
            w.writeheader()
        for name, r in results.items():
            for kind in ("in_sample", "loso"):
                for s, sc in r[kind].items():
                    w.writerow({"time": stamp, "expert": "ensemble", "subject": s,
                                "window": f"{name} ({kind})", **sc})
    print(f"Appended results to {path}")

if __name__ == "__main__":
    main()
