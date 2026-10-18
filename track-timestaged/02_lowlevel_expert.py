"""Step 1b: the "simple visuals" expert, one model per brain window.

For every subject and window: learn video features -> EEG on the training
data, predict the EEG for each of the 102 candidate videos, and score each
test recording against every prediction. Windows are then combined.

    python 02_lowlevel_expert.py

Prints top-1/top-5 per window for sub-01..04, writes submission files for
all subjects to outputs/submissions/lowlevel/, and appends to outputs/results.csv.
"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from tsx.data import (
    SFREQ, SUBJECTS, TEST_IDS, downsample, load_test, load_test_labels, load_train,
)
from tsx.encoding import (
    WINDOWS, WindowEncoder, combine, correlation_similarity, normalize_eeg, window_slice,
)
from tsx.evaluate import score_ranking, score_similarity
from tsx.submission import write_submission

EXPERT = "lowlevel"
CACHE = Path("cache/lowlevel.npz")
DECIM = 5                                  # 500 Hz -> 100 Hz
OUT = Path("outputs")

def load_features() -> tuple[np.ndarray, dict[int, int]]:
    z = np.load(CACHE)
    row_of = {int(v): k for k, v in enumerate(z["ids"])}
    return z["features"].reshape(len(z["ids"]), -1), row_of   # flatten time x features

def main() -> None:
    feats, row_of = load_features()
    X_cand = feats[[row_of[v] for v in TEST_IDS]]
    sfreq = SFREQ / DECIM
    rows = []

    for s in SUBJECTS:
        trn_ids, trn = load_train(s)
        trn, tst = normalize_eeg(downsample(trn, DECIM), downsample(load_test(s), DECIM))
        X_trn = feats[[row_of[v] for v in trn_ids]]

        sims = {}
        for name, (t0, t1) in WINDOWS.items():
            sl = window_slice(t0, t1, sfreq)
            enc = WindowEncoder().fit(X_trn, trn[:, :, sl])
            sims[name] = correlation_similarity(tst[:, :, sl], enc.predict(X_cand))
            print(f"sub-{s:02d} {name:<7} alpha={enc.alpha:.0e}")
        sims["combined"] = combine([sims[w] for w in WINDOWS])

        (OUT / "sims").mkdir(parents=True, exist_ok=True)
        np.savez(OUT / "sims" / f"{EXPERT}_sub-{s:02d}.npz", **sims)
        path = write_submission(sims["combined"], s, OUT / "submissions" / EXPERT)

        labels = load_test_labels(s)
        if labels is None:
            print(f"sub-{s:02d} held out -> {path}")
            continue
        for name, sim in sims.items():
            # "combined" is scored from the saved file, exactly as the organizers will.
            r = score_ranking(np.load(path), labels) if name == "combined" else score_similarity(sim, labels)
            rows.append({"subject": s, "window": name, **r})

    _report(rows)

def _report(rows: list[dict]) -> None:
    print()
    print(f"{'window':<9}" + "".join(f"  sub-{s:02d} top1/top5" for s in (1, 2, 3, 4)) + "     MEAN top1/top5")
    for name in list(WINDOWS) + ["combined"]:
        rs = [r for r in rows if r["window"] == name]
        cells = "".join(f"     {r['top1']:5.1%}/{r['top5']:5.1%}" for r in rs)
        m1, m5 = np.mean([r["top1"] for r in rs]), np.mean([r["top5"] for r in rs])
        print(f"{name:<9}{cells}     {m1:5.1%}/{m5:5.1%}")
    print("chance: top-1 1.0%, top-5 4.9%")

    csv_path = OUT / "results.csv"
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["time", "expert", "subject", "window", "top1", "top5", "median_rank"])
        if new:
            w.writeheader()
        stamp = datetime.now().isoformat(timespec="seconds")
        for r in rows:
            w.writerow({"time": stamp, "expert": EXPERT, **r})
    print(f"Appended results to {csv_path}")

if __name__ == "__main__":
    main()
