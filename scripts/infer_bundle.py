"""Standalone inference for the headline ensemble (final_csls) from an exported bundle. Needs only numpy and scipy.

Pipeline per subject (identical to scripts/ensemble.py at track-matching 7297a59, experiment fin5_ctr_m5):
    test EEG (102, 17, 1500) @ 500 Hz
      -> crop to 0-1.5 s -> FIR zero-phase decimation x5 (100 Hz)       [eegmr.data.decimate]
      -> center-test: replace the mean over the 102 recordings by the training mean   [ensemble.center_like]
      -> per member: window 5:60 -> per-timepoint whitening (Sigma_t^-1/2) -> flatten (channel-major)
                     -> standardize with training stats
           decw: ridge  pred = X @ B + y_mean  -> cosine vs candidates standardized with training-video stats
           con : per seed  cos(X @ W_e, V @ W_v), row-z, mean over 3 seeds
      -> CSLS k=5 per member -> row-z -> weighted sum (0 / 0.4 / 0.3 / 0.2 / 0.1)
      -> rank: candidates 1001..1102 sorted by descending score, rows in the original recording order

Example:
    python scripts/infer_bundle.py --bundle bundle/krabiki_final_csls_bundle --root eeg-mind-reading-challenge \\
        --subjects 5 6 --out out_submissions
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.signal import decimate

TEST_IDS = np.arange(1001, 1103)


def preprocess(X: np.ndarray, max_seconds: float, sfreq: int, factor: int) -> np.ndarray:
    """Crop the raw EEG to ``max_seconds`` (before filtering), then decimate (as ``eegmr.data.decimate``)."""
    X = X[..., : int(round(max_seconds * sfreq))]
    return decimate(X, factor, ftype="fir", zero_phase=True, axis=-1).astype(np.float32)


def row_z(a: np.ndarray) -> np.ndarray:
    return (a - a.mean(axis=1, keepdims=True)) / a.std(axis=1, keepdims=True)


def normalize(a: np.ndarray) -> np.ndarray:
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


def csls(sim: np.ndarray, k: int) -> np.ndarray:
    """Cross-domain similarity local scaling, as ``eegmr.retrieval.csls``."""
    k_rec, k_cand = min(k, sim.shape[1]), min(k, sim.shape[0])
    r_rec = np.sort(sim, axis=1)[:, -k_rec:].mean(axis=1)
    r_cand = np.sort(sim, axis=0)[-k_cand:, :].mean(axis=0)
    return 2 * sim - r_rec[:, None] - r_cand[None, :]


def member_input(eeg: np.ndarray, w: dict, name: str, window: slice) -> np.ndarray:
    """Whitened (per timepoint) and standardized EEG window, flattened channel-major: ``(n, C * T_w)``."""
    x = eeg[..., window].astype(np.float64)
    x = np.einsum("tcd,ndt->nct", w[f"{name}/whiteners"].astype(np.float64), x).reshape(len(x), -1)
    return (x - w[f"{name}/x_mean"]) / w[f"{name}/x_std"]


def member_similarity(eeg: np.ndarray, w: dict, member: dict, candidates: np.ndarray, window: slice,
                      seeds: list[int]) -> np.ndarray:
    name = member["name"]
    X = member_input(eeg, w, name, window)
    cand = (candidates.astype(np.float64) - w[f"{name}/f_mean"]) / w[f"{name}/f_std"]
    if member["head"] == "decw":
        pred = X @ w[f"{name}/B"].astype(np.float64) + w[f"{name}/y_mean"]
        return normalize(pred) @ normalize(cand).T
    z = 0.0
    for seed in seeds:
        e = normalize(X @ w[f"{name}/seed{seed}/W_e"].astype(np.float64))
        v = normalize(cand @ w[f"{name}/seed{seed}/W_v"].astype(np.float64))
        z = z + row_z(e @ v.T)
    return z / len(seeds)


def rank_candidates(score: np.ndarray) -> np.ndarray:
    ranked = TEST_IDS[np.argsort(-score, axis=1, kind="stable")].astype(np.int64)
    assert ranked.shape == (102, 102) and all(np.array_equal(np.sort(r), TEST_IDS) for r in ranked)
    return ranked


def infer_subject(subject: int, bundle: Path, root: Path, meta: dict) -> np.ndarray:
    with np.load(bundle / "weights" / f"sub-{subject:02d}.npz") as f:
        w = {k: f[k] for k in f.files}
    window = slice(*meta["window"])
    eeg = preprocess(np.load(root / "tst" / f"sub-{subject:02d}_tst.npy").astype(np.float32), meta["max_seconds"],
                     meta["sfreq"], meta["decimate_factor"])
    if meta["center_test"]:
        eeg = (eeg - eeg.mean(axis=0) + w["train_eeg_mean"]).astype(np.float32)
    total = 0.0
    for m in meta["members"]:
        if m["weight"] == 0:
            continue
        sim = member_similarity(eeg, w, m, np.load(bundle / "candidates" / f"{m['name']}.npy"), window, meta["con_seeds"])
        total = total + m["weight"] * row_z(csls(sim, meta["csls_k"]))
    return rank_candidates(total)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True, help="challenge data root (tst/sub-XX_tst.npy)")
    p.add_argument("--subjects", type=int, nargs="+", default=[5, 6])
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    meta = json.loads((a.bundle / "bundle_meta.json").read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    for s in a.subjects:
        t = time.time()
        ranked = infer_subject(s, a.bundle, a.root, meta)
        path = a.out / f"sub-{s:02d}_ranked_candidate_list.npy"
        np.save(path, ranked)
        msg = f"sub-{s:02d}: wrote {path} ({time.time() - t:.1f} s)"
        labels = a.root / "tst_stim" / f"sub-{s:02d}_tst_stim.npy"
        if labels.exists():  # dev subjects only: score the written file
            y = np.load(labels).astype(int)
            pos = np.argmax(ranked == y[:, None], axis=1)
            msg += f"  top1={np.mean(pos < 1):.3f} top5={np.mean(pos < 5):.3f} median_rank={np.median(pos + 1):.1f}"
        print(msg, flush=True)


if __name__ == "__main__":
    main()
