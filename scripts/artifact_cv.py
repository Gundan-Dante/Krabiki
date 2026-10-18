"""S3 artifact-aware training: train-CV of dropping / down-weighting outlier *training* trials (``eegmr.artifacts``).

Same folds (KFold 5, seed 0), alphas (read from the ensemble ``--sim-cache``), test-time options (``--center-test``,
member-stage CSLS) and criteria (mean percentile rank, expected top-1 / top-5 of 102) as ``scripts/ensemble.py``.
Outliers are flagged per subject from the raw training EEG only (no labels). In every fold they are dropped from, or
down-weighted in, the *fitting* part only; all held-out trials are still scored. The "none" row recomputes the baseline
and must equal the cached ensemble CV (checked).

Example (the final model's settings, one member):
    PYTHONPATH=src:scripts python scripts/artifact_cv.py --member decw:clip_b32 --max-seconds 1.5 --center-test --csls 5 \\
        --sim-cache results/similarity/_cache
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

from ensemble import Evaluator, STEPS, cache_path, center_like, parse_member, row_z
from eegmr.artifacts import WeightedDecoding, flag_outliers, outlier_scores, trial_features
from eegmr.data import DEFAULT_ROOT, SFREQ, load_train
from eegmr.features import DEFAULT_FEATURE_DIR, load_features
from eegmr.retrieval import csls
from predict import preprocess


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--member", action="append", required=True, help="decw members only, e.g. decw:clip_b32")
    p.add_argument("--fracs", type=float, nargs="+", default=[0.05, 0.10, 0.15])
    p.add_argument("--modes", nargs="+", default=["drop", "weight"], choices=["drop", "weight"])
    p.add_argument("--down-weight", type=float, default=0.25, help="ridge weight of flagged trials in 'weight' mode")
    p.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    p.add_argument("--window", type=int, nargs=2, default=[5, 60])
    p.add_argument("--max-seconds", type=float, default=1.5)
    p.add_argument("--center-test", action="store_true")
    p.add_argument("--csls", type=int, default=0, help="member-stage CSLS k on each held-out fold matrix (0 = off)")
    p.add_argument("--sim-cache", type=Path, default=Path("results/similarity/_cache"))
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    p.add_argument("--out", type=Path, default=None, help="default: results/artifact_cv_<settings>.csv")
    args = p.parse_args()
    args.con_seeds = range(0, 3)  # only used by cache_path for con members
    window = slice(*args.window)
    members = [parse_member(s, 8) for s in args.member]
    assert all(m.head == "decw" for m in members), "artifact_cv supports decw members"
    configs = [("none", 0.0)] + [(mode, f) for mode in args.modes for f in args.fracs]
    tag = f"{'_ctr' if args.center_test else ''}{f'_csls{args.csls}' if args.csls else ''}"
    out = args.out or Path("results") / f"artifact_cv{tag}.csv"

    rows = []
    for m in members:
        for s in args.subjects:
            X, ids = load_train(s, args.root)
            eeg = preprocess(X, args.max_seconds)
            raw = X[..., : int(round(args.max_seconds * SFREQ))]
            scores = outlier_scores(trial_features(raw))
            feats = load_features(m.features, m.frames, ids, args.feature_dir)
            folds = list(KFold(5, shuffle=True, random_state=0).split(ids))
            path = cache_path(args, s, m, "cv")
            with np.load(path, allow_pickle=False) as z:
                dec_alpha, enc_alpha = (float(a) for a in z["alphas"])
                cached = [z[f"f{f}"] for f in range(5)]
            for mode, frac in configs:
                flags = flag_outliers(scores, frac)
                sims = []
                for trn, val in folds:
                    e_val = center_like(eeg[val], eeg[trn]) if args.center_test else eeg[val]
                    if mode == "drop":
                        trn = trn[~flags[trn]]
                    w = np.where(flags[trn], args.down_weight, 1.0) if mode == "weight" else None
                    dec = WeightedDecoding(dec_alpha, whiten_input=True, enc_alpha=enc_alpha, window=window,
                                           sample_weight=w)
                    sims.append(dec.fit(feats[trn], eeg[trn]).similarity(e_val, feats[val]))
                if mode == "none":
                    diff = max(np.abs(a - b).max() for a, b in zip(sims, cached))
                    assert diff < 1e-4, f"baseline differs from the ensemble cache by {diff}"
                if args.csls:
                    sims = [csls(x, args.csls) for x in sims]
                c = Evaluator({s: [[row_z(x)] for x in sims]}).criteria(s, list(range(5)), (STEPS,))
                rows.append({"member": m.name, "subject": s, "mode": mode, "frac": frac, "n_flagged": int(flags.sum()),
                             "pct_rank": c[0], "exp_top1": c[1], "exp_top5": c[2]})
                print(f"{m.name} sub-{s:02d} {mode:6s} {frac:.2f}: pct {c[0]:.4f}  E[top1] {c[1]:.4f}  "
                      f"E[top5] {c[2]:.4f}", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"\nwrote {out}\nmean over subjects (lower pct rank / higher E[top-k] is better; vs 'none'):")
    for m in members:
        base = [r for r in rows if r["member"] == m.name and r["mode"] == "none"]
        b = {k: np.mean([r[k] for r in base]) for k in ("pct_rank", "exp_top1", "exp_top5")}
        print(f"{m.name}\n  {'mode':6s} {'frac':>5s} {'pct_rank':>9s} {'E[top1]':>8s} {'E[top5]':>8s}  subjects better (avg crit)")
        for mode, frac in configs:
            rs = [r for r in rows if r["member"] == m.name and r["mode"] == mode and r["frac"] == frac]
            a = {k: np.mean([r[k] for r in rs]) for k in b}
            better = sum((r["exp_top1"] + r["exp_top5"]) > (q["exp_top1"] + q["exp_top5"]) for r, q in zip(rs, base))
            print(f"  {mode:6s} {frac:5.2f} {a['pct_rank']:9.4f} {a['exp_top1']:8.4f} {a['exp_top5']:8.4f}  "
                  f"{better}/{len(rs)}" + ("" if mode == "none" else
                                           f"  (dE[top1] {a['exp_top1'] - b['exp_top1']:+.4f}, dE[top5] {a['exp_top5'] - b['exp_top5']:+.4f})"))


if __name__ == "__main__":
    main()
