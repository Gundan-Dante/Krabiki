"""Run any expert: try every feature set in every brain window, report the grid,
save the scores, and write submissions for the best combination."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from .data import (
    DEV_SUBJECTS, SFREQ, SUBJECTS, TEST_IDS, downsample, load_test, load_test_labels, load_train,
)
from .encoding import WINDOWS, WindowEncoder, combine, correlation_similarity, normalize_eeg, window_slice
from .evaluate import score_ranking, score_similarity
from .submission import write_submission

DECIM = 5                       # 500 Hz -> 100 Hz
OUT = Path("outputs")

def run_expert(expert: str, feature_sets: dict[str, np.ndarray], ids: np.ndarray) -> None:
    """feature_sets: name -> (n_videos, d) array, rows aligned with `ids` (video IDs)."""
    row_of = {int(v): k for k, v in enumerate(ids)}
    cand_rows = [row_of[v] for v in TEST_IDS]
    sfreq = SFREQ / DECIM
    sims: dict[int, dict[str, np.ndarray]] = {}

    for s in SUBJECTS:
        trn_ids, trn = load_train(s)
        trn, tst = normalize_eeg(downsample(trn, DECIM), downsample(load_test(s), DECIM))
        trn_rows = [row_of[v] for v in trn_ids]
        sims[s] = {}
        for fs, X in feature_sets.items():
            for win, (t0, t1) in WINDOWS.items():
                sl = window_slice(t0, t1, sfreq)
                enc = WindowEncoder().fit(X[trn_rows], trn[:, :, sl])
                sims[s][f"{fs}|{win}"] = correlation_similarity(tst[:, :, sl], enc.predict(X[cand_rows]))
            sims[s][f"{fs}|combined"] = combine([sims[s][f"{fs}|{w}"] for w in WINDOWS])
        print(f"sub-{s:02d} done")

    # Score everything on the dev subjects.
    labels = {s: load_test_labels(s) for s in DEV_SUBJECTS}
    rows, table = [], {}
    for key in sims[DEV_SUBJECTS[0]]:
        per_sub = [score_similarity(sims[s][key], labels[s]) for s in DEV_SUBJECTS]
        table[key] = {m: float(np.mean([r[m] for r in per_sub])) for m in per_sub[0]}
        fs, win = key.split("|")
        rows += [{"expert": f"{expert}:{fs}", "subject": s, "window": win, **r}
                 for s, r in zip(DEV_SUBJECTS, per_sub)]
    _print_grid(feature_sets, table)

    # Best combined feature set (by mean top-5 on dev subjects) -> submission files.
    best = max(feature_sets, key=lambda fs: table[f"{fs}|combined"]["top5"])
    print(f"Best combined: {best}  ->  outputs/submissions/{expert}/")
    (OUT / "sims").mkdir(parents=True, exist_ok=True)
    for s in SUBJECTS:
        np.savez(OUT / "sims" / f"{expert}_sub-{s:02d}.npz", **sims[s])
        path = write_submission(sims[s][f"{best}|combined"], s, OUT / "submissions" / expert)
        if s in labels:  # re-score from the saved file, exactly as the organizers will
            assert np.isclose(score_ranking(np.load(path), labels[s])["top5"],
                              score_similarity(sims[s][f"{best}|combined"], labels[s])["top5"])
    _append_csv(rows)

def _print_grid(feature_sets: dict, table: dict) -> None:
    cols = list(WINDOWS) + ["combined"]
    print()
    print("Mean over sub-01..04:  top-1 / top-5 / median rank   (chance: 1.0% / 4.9% / 51.5)")
    print(f"{'features':<14}" + "".join(f"{c:>24}" for c in cols))
    for fs in feature_sets:
        cells = "".join(
            f"{table[f'{fs}|{c}']['top1']:>8.1%} /{table[f'{fs}|{c}']['top5']:>6.1%} /{table[f'{fs}|{c}']['median_rank']:>5.1f}"
            for c in cols
        )
        print(f"{fs:<14}{cells}")
    print()

def _append_csv(rows: list[dict]) -> None:
    path = OUT / "results.csv"
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["time", "expert", "subject", "window", "top1", "top5", "median_rank"])
        if new:
            w.writeheader()
        stamp = datetime.now().isoformat(timespec="seconds")
        for r in rows:
            w.writerow({"time": stamp, **r})
    print(f"Appended results to {path}")
