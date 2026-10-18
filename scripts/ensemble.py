"""CV-weighted ensemble of the decoding head and the noise-whitened (Euclid) encoding head.

S = w * z(dec) + (1 - w) * z(enc), z = per-row z-score over the candidates.

1. Training-only CV (5 folds, seed 0): out-of-fold similarity matrices of both heads among the held-out videos. The
   decoder's input whiteners (``--decoder whitened``) and the encoder's matcher are fitted inside each fold (nested,
   see ``eegmr.matching.nested_fold_similarities``); alphas are the ones chosen on the full training set by CV (as
   predict.py does). For every w on the grid the
   criteria are the mean percentile rank, the expected top-1 and the expected top-5 of 102 candidates
   (hypergeometric: the exact chance of the true video being in the top k if 101 distractors were drawn from the fold).
2. w is picked by the mean of expected top-1 and top-5 (global: over all subjects; per subject: that subject's own
   folds). A leave-one-fold-out check compares global vs per-subject selection honestly.
3. Test: both heads are fit on all training trials, the chosen w blends the test similarities, and the ranked lists
   are written (and scored for sub-01..04, appended to results.csv).

Example:
    python scripts/ensemble.py --features clip_b32 --frames 8 --max-seconds 1.5
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom
from sklearn.model_selection import KFold

from eegmr.data import DEFAULT_ROOT, TEST_IDS, load_test, load_test_labels, load_train
from eegmr.evaluate import evaluate
from eegmr.features import DEFAULT_FEATURE_DIR, load_features
from eegmr.decoding import DecodingVariant, nested_cv_scores
from eegmr.linear import DEFAULT_ALPHAS, DecodingModel, EncodingModel
from eegmr.matching import Matcher, nested_fold_similarities, oof_predictions
from eegmr.submission import score_submission_file, write_submission
from predict import RESULT_FIELDS, preprocess  # scripts/ is on sys.path when run as a script

W_GRID = np.round(np.arange(0.0, 1.0001, 0.1), 1)
N_CANDIDATES = 102  # test set size: 1 target + 101 distractors
ENC_MATCHER = dict(kind="noise", noise_cov="per_time", noise_similarity="euclid")


def row_z(sim: np.ndarray) -> np.ndarray:
    """Z-score every row over the candidates."""
    return (sim - sim.mean(axis=1, keepdims=True)) / sim.std(axis=1, keepdims=True)


def blend(dec: np.ndarray, enc: np.ndarray, w: float) -> np.ndarray:
    return w * row_z(dec) + (1 - w) * row_z(enc)


def decoder_alpha(features: np.ndarray, eeg: np.ndarray, enc_alpha: float, window: slice, kind: str) -> float:
    """Decoder alpha by training-only CV: nested (whiteners re-estimated per fold) for ``whitened``, plain for ``raw``."""
    if kind == "raw":
        return DecodingModel(window=window).fit(features, eeg).alpha
    curve = nested_cv_scores(features, eeg, DEFAULT_ALPHAS, enc_alpha, powers=(), window=window,
                             inputs=("whitened",))["whitened"]
    return float(DEFAULT_ALPHAS[curve.argmin()])


def make_decoder(kind: str, dec_alpha: float, enc_alpha: float, window: slice) -> DecodingVariant:
    """``whitened``: EEG input noise-whitened per timepoint (whiteners come from the data passed to ``fit``)."""
    return DecodingVariant(dec_alpha, whiten_input=kind == "whitened", enc_alpha=enc_alpha, window=window)


def fold_criteria(dec: np.ndarray, enc: np.ndarray) -> np.ndarray:
    """Per-w sums over the fold's trials of [percentile rank, expected top-1, expected top-5]: ``(len(W_GRID), 3)``."""
    n_distractors = len(dec) - 1
    out = np.zeros((len(W_GRID), 3))
    for i, w in enumerate(W_GRID):
        s = blend(dec, enc, w)
        better = (s >= np.diag(s)[:, None]).sum(axis=1) - 1  # held-out distractors scoring >= the true video
        out[i] = [(better / n_distractors).sum(),
                  hypergeom.cdf(0, n_distractors, better, N_CANDIDATES - 1).sum(),
                  hypergeom.cdf(4, n_distractors, better, N_CANDIDATES - 1).sum()]
    return out


def pick(crit: np.ndarray) -> dict[str, float]:
    """``crit``: ``(len(W_GRID), 3)`` per-trial means. The w each criterion alone would pick, and the combined pick."""
    return {"avg": float(W_GRID[np.argmax((crit[:, 1] + crit[:, 2]) / 2)]), "pct": float(W_GRID[np.argmin(crit[:, 0])]),
            "top1": float(W_GRID[np.argmax(crit[:, 1])]), "top5": float(W_GRID[np.argmax(crit[:, 2])])}


def subject_cv(subject: int, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Per-fold criterion sums ``(n_folds, len(W_GRID), 3)``, fold sizes, dec alpha, enc alpha."""
    X, ids = load_train(subject, args.root)
    eeg = preprocess(X, args.max_seconds)
    feats = load_features(args.features, args.frames, ids, args.feature_dir)
    window = slice(*args.window)
    enc_alpha = EncodingModel(alpha_selection="retrieval", window=window).fit(feats, eeg).alpha
    dec_alpha = decoder_alpha(feats, eeg, enc_alpha, window, args.decoder)
    matcher = Matcher(window=window, **ENC_MATCHER)
    folds = list(KFold(5, shuffle=True, random_state=0).split(feats))
    crit, sizes = [], []
    for (trn, val), (val2, enc_sim) in zip(folds, nested_fold_similarities(feats, eeg, enc_alpha, matcher)):
        assert np.array_equal(val, val2)
        dec_sim = make_decoder(args.decoder, dec_alpha, enc_alpha, window).fit(feats[trn], eeg[trn]).similarity(
            eeg[val], feats[val])  # whiteners (if any) are re-estimated from this fold's training part only
        crit.append(fold_criteria(dec_sim, enc_sim))
        sizes.append(len(val))
    return np.stack(crit), np.array(sizes), dec_alpha, enc_alpha


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", required=True)
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    p.add_argument("--window", type=int, nargs=2, default=[5, 60])
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--decoder", choices=["whitened", "raw"], default="whitened",
                   help="decoding member: noise-whitened input (headline) or the plain decoder")
    p.add_argument("--select", choices=["auto", "global", "subject"], default="auto",
                   help="which w to use on the test sets; auto: the better one by leave-one-fold-out CV")
    p.add_argument("--cv-only", action="store_true", help="stop after the training-only CV (touches no test data)")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    p.add_argument("--cv-out", type=Path, default=Path("results/ensemble_cv.csv"))
    p.add_argument("--results", type=Path, default=Path("results/results.csv"))
    p.add_argument("--experiment", default="ens_dec_encnoise", help="prefix; _global / _subject / _equal / _dec / _enc are appended")
    args = p.parse_args()
    window = slice(*args.window)

    # 1. training-only CV
    folds_crit, alphas = {}, {}
    for s in args.subjects:
        crit, sizes, dec_a, enc_a = subject_cv(s, args)
        folds_crit[s], alphas[s] = (crit, sizes), (dec_a, enc_a)
        full = crit.sum(0) / sizes.sum()
        print(f"sub-{s:02d} CV  dec(w=1) pct={full[-1, 0]:.4f} e1/e5={full[-1, 1]:.3f}/{full[-1, 2]:.3f}  "
              f"enc(w=0) pct={full[0, 0]:.4f} e1/e5={full[0, 1]:.3f}/{full[0, 2]:.3f}  "
              f"equal(w=.5) pct={full[5, 0]:.4f} e1/e5={full[5, 1]:.3f}/{full[5, 2]:.3f}", flush=True)
    per_subject = {s: c.sum(0) / n.sum() for s, (c, n) in folds_crit.items()}  # (w, 3)
    global_crit = np.mean([per_subject[s] for s in args.subjects], axis=0)
    w_subject = {s: pick(per_subject[s]) for s in args.subjects}
    w_global = pick(global_crit)
    print("\nglobal w (mean over subjects):", w_global)
    for s in args.subjects:
        print(f"sub-{s:02d} w:", w_subject[s])
    args.cv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.cv_out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["subject", "w", "pct_rank", "exp_top1", "exp_top5", "dec_alpha", "enc_alpha"])
        for s in args.subjects:
            for i, w in enumerate(W_GRID):
                writer.writerow([s, w, *(f"{v:.6f}" for v in per_subject[s][i]), *alphas[s]])
        for i, w in enumerate(W_GRID):
            writer.writerow(["global", w, *(f"{v:.6f}" for v in global_crit[i]), "", ""])

    # leave-one-fold-out: choose w on the other 4 folds, score on the held-out fold (average criterion)
    loo = {"dec only": [], "enc only": [], "equal": [], "global w": [], "per-subject w": []}
    for f in range(5):
        keep = [i for i in range(5) if i != f]
        sub_other = {s: folds_crit[s][0][keep].sum(0) / folds_crit[s][1][keep].sum() for s in args.subjects}
        w_g = pick(np.mean(list(sub_other.values()), axis=0))["avg"]
        for name, ws in [("dec only", dict.fromkeys(args.subjects, 1.0)), ("enc only", dict.fromkeys(args.subjects, 0.0)),
                         ("equal", dict.fromkeys(args.subjects, 0.5)), ("global w", dict.fromkeys(args.subjects, w_g)),
                         ("per-subject w", {s: pick(sub_other[s])["avg"] for s in args.subjects})]:
            loo[name].append(np.mean([(lambda c: (c[1] + c[2]) / 2)(folds_crit[s][0][f, list(W_GRID).index(ws[s])]
                                                                 / folds_crit[s][1][f]) for s in args.subjects]))
    loo = {k: float(np.mean(v)) for k, v in loo.items()}
    print("\nleave-one-fold-out mean of (E[top1] + E[top5]) / 2:", {k: round(v, 4) for k, v in loo.items()})
    use = args.select if args.select != "auto" else ("global" if loo["global w"] >= loo["per-subject w"] else "subject")
    print("using:", use)
    if args.cv_only:
        return

    # 2. test
    variants = {"global": {s: w_global["avg"] for s in args.subjects}, "subject": {s: w_subject[s]["avg"] for s in args.subjects},
                "equal": dict.fromkeys(args.subjects, 0.5), "dec": dict.fromkeys(args.subjects, 1.0),
                "enc": dict.fromkeys(args.subjects, 0.0)}
    rows = {name: [] for name in variants}
    sim_dir = Path("results/similarity")
    for s in args.subjects:
        X, ids = load_train(s, args.root)
        eeg = preprocess(X, args.max_seconds)
        measured = preprocess(load_test(s, args.root), args.max_seconds)
        F_trn = load_features(args.features, args.frames, ids, args.feature_dir)
        F_tst = load_features(args.features, args.frames, TEST_IDS, args.feature_dir)
        enc_model = EncodingModel(alpha_selection="retrieval", window=window).fit(F_trn, eeg)
        dec = make_decoder(args.decoder, alphas[s][0], enc_model.alpha, window).fit(F_trn, eeg).similarity(measured, F_tst)
        matcher = Matcher(window=window, **ENC_MATCHER).fit(eeg, oof_predictions(F_trn, eeg, enc_model.alpha))
        enc = matcher.similarity(measured, enc_model.predict(F_tst))
        for name, comp in (("dec", dec), ("enc", enc)):
            (sim_dir / "ens_components").mkdir(parents=True, exist_ok=True)
            np.save(sim_dir / "ens_components" / f"{name}_sub-{s:02d}.npy", comp)
        true_ids = load_test_labels(s, args.root)
        for name, ws in variants.items():
            sim = blend(dec, enc, ws[s])
            exp = f"{args.experiment}_{name}"
            (sim_dir / exp).mkdir(parents=True, exist_ok=True)
            np.save(sim_dir / exp / f"sub-{s:02d}.npy", sim)
            path = write_submission(sim, s, Path("submissions") / exp)
            if true_ids is None:
                continue
            scores = score_submission_file(path, true_ids)
            assert np.isclose(scores["top1"], evaluate(sim, true_ids)["top1"])
            rows[name].append({"subject": s, **scores, "alpha": enc_model.alpha, "n_pcs": 0})
            print(f"sub-{s:02d} {name:7s} w={ws[s]:.1f} top1={scores['top1']:.3f} top5={scores['top5']:.3f} "
                  f"median_rank={scores['median_rank']:.1f}", flush=True)
        if true_ids is not None:
            d = evaluate(dec, true_ids)
            print(f"sub-{s:02d} dec only top1={d['top1']:.3f} top5={d['top5']:.3f} median_rank={d['median_rank']:.1f}")
    print(f"\nCV-best variant for sub-05/06: {args.experiment}_{use}")
    now = dt.datetime.now().isoformat(timespec="seconds")
    new_file = not args.results.exists()
    with args.results.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_FIELDS)
        if new_file:
            writer.writeheader()
        for name, rs in rows.items():
            for r in rs:
                writer.writerow({"time": now, "experiment": f"{args.experiment}_{name}", "features": args.features,
                                 "frames": args.frames, "window": f"{args.window[0]}:{args.window[1]}", **r})


if __name__ == "__main__":
    main()
