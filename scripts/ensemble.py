"""CV-weighted ensemble of N members (head x feature set), weights on the simplex.

S = sum_i w_i * z(sim_i), z = per-row z-score over the candidates, w_i >= 0, sum w_i = 1 (step 0.1).

Members are given as ``--member HEAD:FEATURES[:FRAMES]`` (``con:FEATURES:CONFIG[:FRAMES]`` for contrastive) with HEAD in
    dec         raw decoder: EEG window -> standardized features, cosine
    decw        decoder with noise-whitened EEG input (``eegmr.decoding.DecodingVariant``)
    enc         encoder, plain correlation matching
    enc_noise   encoder, per-timepoint noise whitening + negative squared distance (``eegmr.matching``)
    con         linear InfoNCE head on the whitened-decoder input (``eegmr.contrastive``); CONFIG is one of
                ``warm | linear | two_tower_d256`` (the ``selected`` grid of scripts/train_contrastive.py). The member's
                similarity is the mean of row-z similarities over ``--con-seeds`` SGD seeds (0, 1, ...), in the CV folds
                and in the final fit. Whiteners are the decw member's (re-estimated on each fold's training part) and the
                ridge alpha is the decoder alpha chosen for these features.
and FEATURES the name of a cache in ``features/`` (``<name>_f<FRAMES>.npz``, default FRAMES = ``--frames``).

1. Training-only CV (5 folds, seed 0): out-of-fold similarity matrices of every member among the held-out videos.
   Whiteners and matchers are re-estimated inside each fold from that fold's training part (nested); alphas are the ones
   chosen on the full training set by CV. For a weight vector the criteria are the mean percentile rank, the expected
   top-1 and the expected top-5 of 102 candidates (hypergeometric: the exact chance of the true video being in the top k
   if 101 distractors were drawn from the fold).
2. Weights are picked by the mean of expected top-1 and top-5: exhaustively on the simplex lattice for N <= 3, by
   steepest-ascent pair transfers (0.1 from one member to another) for N > 3. Strategies compared by a
   leave-one-fold-out check (weights picked on 4 folds, scored on the 5th): best single member, equal weights, one global
   weight vector (all subjects), and one weight vector per subject. The simplest strategy within 0.0005 of the best wins.
3. Test: members are fit on all training trials, the chosen weights blend the test similarities, the ranked lists are
   written and (sub-01..04) scored, and a paired Wilcoxon on per-trial ranks is computed against the best single member.

Example:
    python scripts/ensemble.py --member decw:clip_b32 --member decw:dinov2_b_cls_all --member enc_noise:clip_b32 \\
        --max-seconds 1.5 --experiment ens_clip_dino
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom
from sklearn.model_selection import KFold

from compare_matching import compare_table  # scripts/ is on sys.path when run as a script
from eegmr import gpu
from eegmr.contrastive import ContrastiveConfig, prepare, train_head
from eegmr.data import DEFAULT_ROOT, TEST_IDS, load_test, load_test_labels, load_train
from eegmr.decoding import DecodingVariant, nested_cv_scores, noise_whiteners
from eegmr.evaluate import evaluate
from eegmr.features import DEFAULT_FEATURE_DIR, load_features
from eegmr.linear import DEFAULT_ALPHAS, DecodingModel, EncodingModel
from eegmr.matching import Matcher, nested_fold_similarities, oof_predictions
from eegmr.submission import score_submission_file, write_submission
from predict import RESULT_FIELDS, preprocess

HEADS = ("dec", "decw", "enc", "enc_noise", "con")
N_CANDIDATES = 102  # test set size: 1 target + 101 distractors
STEPS = 10  # weights are multiples of 1 / STEPS
TIE = 5e-4  # a simpler strategy within this much of the best leave-one-fold-out score is preferred
Weights = tuple[int, ...]  # integer weights summing to STEPS


@dataclass(frozen=True)
class Member:
    head: str
    features: str
    frames: int
    config: str = ""  # con only

    @property
    def name(self) -> str:
        return f"{self.head}:{self.features}" + (f":{self.config}" if self.config else "")


def parse_member(spec: str, default_frames: int) -> Member:
    head, features, *rest = spec.split(":")
    assert head in HEADS, f"unknown head {head!r}, expected one of {HEADS}"
    config = rest.pop(0) if head == "con" else ""
    if head == "con":
        assert config in contrastive_configs(), f"unknown contrastive config {config!r}, expected {list(contrastive_configs())}"
    return Member(head, features, int(rest[0]) if rest else default_frames, config)


def row_z(sim: np.ndarray) -> np.ndarray:
    """Z-score every row over the candidates."""
    return (sim - sim.mean(axis=1, keepdims=True)) / sim.std(axis=1, keepdims=True)


def blend(sims: list[np.ndarray], weights: Weights) -> np.ndarray:
    """``sum_i (weights_i / STEPS) * row_z(sims_i)``."""
    return sum(w / STEPS * row_z(s) for w, s in zip(weights, sims) if w)


def lattice(n: int, total: int = STEPS) -> list[Weights]:
    """All integer weight vectors of length ``n`` summing to ``total``, first coordinate ascending."""
    if n == 1:
        return [(total,)]
    return [(k, *rest) for k in range(total + 1) for rest in lattice(n - 1, total - k)]


# ---------------------------------------------------------------- members

def encoder_alpha(features: np.ndarray, eeg: np.ndarray, window: slice) -> float:
    return EncodingModel(alpha_selection="retrieval", window=window).fit(features, eeg).alpha


def decoder_alpha(features: np.ndarray, eeg: np.ndarray, enc_alpha: float, window: slice, head: str) -> float:
    """Decoder alpha by training-only CV: nested (whiteners re-estimated per fold) for ``decw``, plain for ``dec``."""
    if head == "dec":
        return DecodingModel(window=window).fit(features, eeg).alpha
    curve = nested_cv_scores(features, eeg, DEFAULT_ALPHAS, enc_alpha, powers=(), window=window,
                             inputs=("whitened",))["whitened"]
    return float(DEFAULT_ALPHAS[curve.argmin()])


def contrastive_configs() -> dict[str, ContrastiveConfig]:
    """The CV-best config per variant on CLIP B-32 (M12), by short name."""
    from train_contrastive import grid  # scripts/ is on sys.path when run as a script
    warm, linear, two_tower = grid("selected")
    return {"warm": warm, "linear": linear, "two_tower_d256": two_tower}


def contrastive_similarity(cfg: ContrastiveConfig, dec_alpha: float, enc_alpha: float, F_trn: np.ndarray,
                           eeg_trn: np.ndarray, F_new: np.ndarray, eeg_new: np.ndarray, window: slice,
                           seeds: int) -> np.ndarray:
    """Mean over seeds of the row-z similarity ``(n_eeg_new, n_videos_new)`` of a head trained on the training part."""
    dev = gpu.default_device()
    W = noise_whiteners(F_trn, eeg_trn, enc_alpha, window, device=dev)
    X_trn, X_new, V_trn, V_new = prepare(eeg_trn, eeg_new, F_trn, F_new, W, window, dev)
    z = 0
    for seed in range(seeds):
        z = z + row_z(train_head(X_trn, V_trn, cfg, dec_alpha, seed).similarity(X_new, V_new).double().cpu().numpy())
    return z / seeds


def encoder_matcher(head: str, window: slice) -> Matcher:
    return Matcher(window=window, **(dict(kind="noise", noise_cov="per_time", noise_similarity="euclid")
                                     if head == "enc_noise" else {}))


def member_alphas(m: Member, features: np.ndarray, eeg: np.ndarray, window: slice) -> tuple[float | None, float | None]:
    """``(decoder alpha, encoder alpha)`` for this member (None where unused), chosen on the full training set."""
    enc = encoder_alpha(features, eeg, window) if m.head != "dec" else None
    dec = decoder_alpha(features, eeg, enc, window, m.head) if m.head in ("dec", "decw", "con") else None
    return dec, enc


def member_fold_similarities(m: Member, alphas: tuple, features: np.ndarray, eeg: np.ndarray, window: slice,
                             folds: list[tuple[np.ndarray, np.ndarray]], con_seeds: int = 3) -> list[np.ndarray]:
    """Held-out similarity (val x val) per outer fold; everything estimated from the fold's training part only."""
    dec_alpha, enc_alpha = alphas
    if m.head in ("enc", "enc_noise"):
        sims = nested_fold_similarities(features, eeg, enc_alpha, encoder_matcher(m.head, window))
        return [sim for _, sim in sims]
    if m.head == "con":
        cfg = contrastive_configs()[m.config]
        return [contrastive_similarity(cfg, dec_alpha, enc_alpha, features[trn], eeg[trn], features[val], eeg[val],
                                       window, con_seeds) for trn, val in folds]
    dec = DecodingVariant(dec_alpha, whiten_input=m.head == "decw", enc_alpha=enc_alpha, window=window)
    return [dec.fit(features[trn], eeg[trn]).similarity(eeg[val], features[val]) for trn, val in folds]


def member_test_similarity(m: Member, alphas: tuple, F_trn: np.ndarray, F_tst: np.ndarray, eeg: np.ndarray,
                           measured: np.ndarray, window: slice, con_seeds: int = 3) -> np.ndarray:
    dec_alpha, enc_alpha = alphas
    if m.head in ("enc", "enc_noise"):
        model = EncodingModel(alpha_selection="retrieval", window=window).fit(F_trn, eeg)
        matcher = encoder_matcher(m.head, window)
        oof = oof_predictions(F_trn, eeg, model.alpha) if matcher.kind != "corr" else None
        return matcher.fit(eeg, oof).similarity(measured, model.predict(F_tst))
    if m.head == "con":
        return contrastive_similarity(contrastive_configs()[m.config], dec_alpha, enc_alpha, F_trn, eeg, F_tst, measured,
                                      window, con_seeds)
    dec = DecodingVariant(dec_alpha, whiten_input=m.head == "decw", enc_alpha=enc_alpha, window=window)
    return dec.fit(F_trn, eeg).similarity(measured, F_tst)


# ---------------------------------------------------------------- criteria and weight search

class Evaluator:
    """Memoized fold criteria ``[sum percentile rank, sum E[top-1], sum E[top-5]]`` of a weight vector."""

    def __init__(self, z_sims: dict[int, list[list[np.ndarray]]]):
        self.sims = z_sims  # subject -> fold -> member -> (val x val) row-z-scored similarity
        self.memo: dict[tuple, np.ndarray] = {}

    def sums(self, subject: int, fold: int, w: Weights) -> np.ndarray:
        key = (subject, fold, w)
        if key not in self.memo:
            s = sum(wi / STEPS * z for wi, z in zip(w, self.sims[subject][fold]) if wi)
            n_distractors = len(s) - 1
            better = (s >= np.diag(s)[:, None]).sum(axis=1) - 1  # held-out distractors scoring >= the true video
            self.memo[key] = np.array([(better / n_distractors).sum(),
                                       hypergeom.cdf(0, n_distractors, better, N_CANDIDATES - 1).sum(),
                                       hypergeom.cdf(4, n_distractors, better, N_CANDIDATES - 1).sum()])
        return self.memo[key]

    def criteria(self, subject: int, folds: list[int], w: Weights) -> np.ndarray:
        """Per-trial means ``[pct rank, E[top-1], E[top-5]]`` over the given folds."""
        n = sum(len(self.sims[subject][f][0]) for f in folds)
        return sum(self.sums(subject, f, w) for f in folds) / n


def avg_criterion(c: np.ndarray) -> float:
    return float((c[1] + c[2]) / 2)


def search(ev: Evaluator, subjects: list[int], folds: list[int], n: int) -> Weights:
    """Weights maximizing the mean over ``subjects`` of (E[top-1] + E[top-5]) / 2 on ``folds``."""
    def objective(w: Weights) -> float:
        return float(np.mean([avg_criterion(ev.criteria(s, folds, w)) for s in subjects]))

    if n <= 3:
        cands = lattice(n)
        return cands[int(np.argmax([objective(w) for w in cands]))]
    singles = [tuple(STEPS if j == i else 0 for j in range(n)) for i in range(n)]
    equal = tuple(STEPS // n + (1 if i < STEPS % n else 0) for i in range(n))
    w = max(singles + [equal], key=objective)
    while True:  # steepest ascent over transfers of one step from member j to member i
        moves = [tuple(w[k] + (k == i) - (k == j) for k in range(n))
                 for i, j in itertools.permutations(range(n), 2) if w[j] > 0]
        best = max(moves, key=objective)
        if objective(best) <= objective(w) + 1e-12:
            return w
        w = best


def best_single(ev: Evaluator, subjects: list[int], folds: list[int], n: int) -> int:
    ws = [tuple(STEPS if j == i else 0 for j in range(n)) for i in range(n)]
    return int(np.argmax([np.mean([avg_criterion(ev.criteria(s, folds, w)) for s in subjects]) for w in ws]))


def fmt(w: Weights) -> str:
    return "/".join(f"{x / STEPS:.1f}" for x in w)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--con-seeds", type=int, default=3, help="contrastive members: average row-z similarities over this many seeds")
    p.add_argument("--member", action="append", required=True, metavar="HEAD:FEATURES[:FRAMES]",
                   help=f"ensemble member, repeatable; HEAD in {HEADS}")
    p.add_argument("--frames", type=int, default=8, help="default frames of the feature caches")
    p.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    p.add_argument("--window", type=int, nargs=2, default=[5, 60])
    p.add_argument("--max-seconds", type=float, default=None)
    p.add_argument("--select", choices=["auto", "single", "equal", "global", "subject"], default="auto",
                   help="which weights to use as the CV-chosen variant; auto: simplest within tolerance of the best "
                        "leave-one-fold-out score")
    p.add_argument("--cv-only", action="store_true", help="stop after the training-only CV (touches no test data)")
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    p.add_argument("--cv-out", type=Path, default=None, help="default: results/<experiment>_cv.csv")
    p.add_argument("--results", type=Path, default=Path("results/results.csv"))
    p.add_argument("--experiment", default="ens", help="prefix of the logged experiments")
    args = p.parse_args()
    members = [parse_member(spec, args.frames) for spec in args.member]
    n, window = len(members), slice(*args.window)
    args.cv_out = args.cv_out or Path("results") / f"{args.experiment}_cv.csv"
    names = [m.name for m in members]
    print("members:", names)

    # 1. training-only CV
    z_sims: dict[int, list[list[np.ndarray]]] = {}
    alphas: dict[tuple[int, int], tuple] = {}
    for s in args.subjects:
        X, ids = load_train(s, args.root)
        eeg = preprocess(X, args.max_seconds)
        folds = list(KFold(5, shuffle=True, random_state=0).split(ids))
        per_member = []
        for i, m in enumerate(members):
            feats = load_features(m.features, m.frames, ids, args.feature_dir)
            alphas[s, i] = member_alphas(m, feats, eeg, window)
            per_member.append(member_fold_similarities(m, alphas[s, i], feats, eeg, window, folds,
                                                          args.con_seeds))
        z_sims[s] = [[row_z(per_member[i][f]) for i in range(n)] for f in range(5)]
        print(f"sub-{s:02d} alphas (dec, enc):", {m.name: alphas[s, i] for i, m in enumerate(members)}, flush=True)
    ev = Evaluator(z_sims)
    all_folds, subjects = list(range(5)), args.subjects

    single_vecs = [tuple(STEPS if j == i else 0 for j in range(n)) for i in range(n)]
    equal_vec = tuple(STEPS // n + (1 if i < STEPS % n else 0) for i in range(n))
    print("\nper-subject CV, mean percentile rank / E[top1] / E[top5] of 102:")
    for s in subjects:
        crit = [ev.criteria(s, all_folds, w) for w in single_vecs]
        print(f"sub-{s:02d}  " + "   ".join(f"{names[i]}: {c[0]:.4f}/{c[1]:.3f}/{c[2]:.3f}" for i, c in enumerate(crit)),
              flush=True)

    best1 = best_single(ev, subjects, all_folds, n)
    w_global = search(ev, subjects, all_folds, n)
    w_subject = {s: search(ev, [s], all_folds, n) for s in subjects}
    print(f"\nbest single member (global CV): {names[best1]}")
    print("global weights:", dict(zip(names, (x / STEPS for x in w_global))))
    for s in subjects:
        print(f"sub-{s:02d} weights:", dict(zip(names, (x / STEPS for x in w_subject[s]))))

    # leave-one-fold-out: pick on 4 folds, score on the 5th
    loo = {k: [] for k in ("best single", "equal", "global", "per-subject")}
    for f in all_folds:
        keep = [k for k in all_folds if k != f]
        b, w_g = best_single(ev, subjects, keep, n), search(ev, subjects, keep, n)
        choice = {"best single": dict.fromkeys(subjects, single_vecs[b]), "equal": dict.fromkeys(subjects, equal_vec),
                  "global": dict.fromkeys(subjects, w_g), "per-subject": {s: search(ev, [s], keep, n) for s in subjects}}
        for k, ws in choice.items():
            loo[k].append(np.mean([avg_criterion(ev.criteria(s, [f], ws[s])) for s in subjects]))
    loo = {k: float(np.mean(v)) for k, v in loo.items()}
    singles_cv = {names[i]: float(np.mean([avg_criterion(ev.criteria(s, all_folds, w)) for s in subjects]))
                  for i, w in enumerate(single_vecs)}
    print("\nleave-one-fold-out mean of (E[top1] + E[top5]) / 2:", {k: round(v, 5) for k, v in loo.items()})
    print("single members, same criterion on all folds:", {k: round(v, 4) for k, v in singles_cv.items()})
    strategy = {"single": "best single", "subject": "per-subject"}.get(args.select, args.select)
    if strategy == "auto":
        top = max(loo.values())
        strategy = next(k for k in ("best single", "equal", "global", "per-subject") if loo[k] >= top - TIE)
    print("CV-chosen strategy:", strategy)

    args.cv_out.parent.mkdir(parents=True, exist_ok=True)
    cands = lattice(n) if n <= 3 else sorted({*single_vecs, equal_vec, w_global, *w_subject.values()})
    with args.cv_out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["subject", "weights", "pct_rank", "exp_top1", "exp_top5", *[f"alphas_{m.name}" for m in members]])
        for w in cands:
            for s in subjects:
                writer.writerow([s, fmt(w), *(f"{v:.6f}" for v in ev.criteria(s, all_folds, w)),
                                 *[alphas[s, i] for i in range(n)]])
            mean = np.mean([ev.criteria(s, all_folds, w) for s in subjects], axis=0)
            writer.writerow(["global", fmt(w), *(f"{v:.6f}" for v in mean), *[""] * n])
    if args.cv_only:
        return

    # 2. test
    variants: dict[str, dict[int, Weights]] = {f"m{i}": dict.fromkeys(subjects, w) for i, w in enumerate(single_vecs)}
    variants["equal"] = dict.fromkeys(subjects, equal_vec)
    variants["global"] = dict.fromkeys(subjects, w_global)
    variants["subject"] = w_subject
    variants["cv"] = {"best single": variants[f"m{best1}"], "equal": variants["equal"], "global": variants["global"],
                      "per-subject": variants["subject"]}[strategy]
    rows: dict[str, list[dict]] = {name: [] for name in variants}
    sim_dir = Path("results/similarity")
    for s in subjects:
        X, ids = load_train(s, args.root)
        eeg = preprocess(X, args.max_seconds)
        measured = preprocess(load_test(s, args.root), args.max_seconds)
        test_sims = [member_test_similarity(m, alphas[s, i], load_features(m.features, m.frames, ids, args.feature_dir),
                                            load_features(m.features, m.frames, TEST_IDS, args.feature_dir), eeg,
                                            measured, window, args.con_seeds) for i, m in enumerate(members)]
        true_ids = load_test_labels(s, args.root)
        for name, ws in variants.items():
            sim = blend(test_sims, ws[s])
            exp = f"{args.experiment}_{name}"
            (sim_dir / exp).mkdir(parents=True, exist_ok=True)
            np.save(sim_dir / exp / f"sub-{s:02d}.npy", sim)
            path = write_submission(sim, s, Path("submissions") / exp)
            if true_ids is None:
                continue
            scores = score_submission_file(path, true_ids)
            assert np.isclose(scores["top1"], evaluate(sim, true_ids)["top1"])
            rows[name].append({"subject": s, **scores, "alpha": 0.0, "n_pcs": 0})
        print(f"sub-{s:02d} done", flush=True)

    dev = [s for s in subjects if load_test_labels(s, args.root) is not None]
    if dev:
        print(f"\nDev (sub-{','.join(f'{s:02d}' for s in dev)}), paired Wilcoxon vs the best single member "
              f"({names[best1]} = {args.experiment}_m{best1}):")
        others = [f"{args.experiment}_{k}" for k in variants if k != f"m{best1}"]
        print(compare_table(f"{args.experiment}_m{best1}", others, dev, args.root, sim_dir))
        print("\nm<i> = member i alone:", dict(enumerate(names)), "| cv =", strategy)
    sel = ", ".join(f"sub-{s:02d}: {fmt(w)}" for s, w in variants["cv"].items())
    print(f"\nCV-chosen variant for sub-05/06: {args.experiment}_cv  weights [{' / '.join(names)}]  ({sel})")
    now = dt.datetime.now().isoformat(timespec="seconds")
    new_file = not args.results.exists()
    with args.results.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_FIELDS)
        if new_file:
            writer.writeheader()
        for name, rs in rows.items():
            for r in rs:
                writer.writerow({"time": now, "experiment": f"{args.experiment}_{name}",
                                 "features": "+".join(dict.fromkeys(m.features for m in members)),
                                 "frames": members[0].frames, "window": f"{args.window[0]}:{args.window[1]}", **r})


if __name__ == "__main__":
    main()
