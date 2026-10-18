"""Motion gate: a stand-alone pipeline step that decides, per candidate video,
whether to trust the image models (e.g. CLIP / DINOv2) or the video models
(e.g. MViT / V-JEPA) more. It is not a model of the EEG; it only produces
weights. Dependencies: numpy and opencv-python. Import nothing from any track.

How it works
1. compute: measure how much each video moves (mean absolute frame-to-frame
   change of a 64x64 grayscale version, over the whole 3 s). Only the video
   files are used: no EEG, no labels.
2. weights: for a candidate with motion z-score m (log scale, z-scored over the
   candidates being ranked), the members get these multipliers:
       image members:  exp(-beta * m)
       video members:  exp(+beta * m)
       other members:  1
   beta > 0 means "moving video -> trust video models more, static video ->
   trust image models more". beta = 0 means equal weights (no gating).
3. fit_beta: choose beta on TRAINING-ONLY cross-validation score matrices
   (rows = held-out training recordings, columns = their videos, true match on
   the diagonal). No test data or test labels are used, so beta is fixed before
   anything is scored.
4. apply: blend the members' test score matrices with the gate weights.

The blend is the usual equal-weight row-z-score sum, with each member scaled
per candidate column. Because the multipliers differ by candidate, every
column is then z-scored across the recordings (unsupervised normalization),
so no video is pushed up for every recording just by its weight. With
beta = 0 and col_z=False the result equals the plain equal-weight blend.

Command line
    python motion_gate.py compute --data-root eeg-mind-reading-challenge --out motion_scores.npz
    python motion_gate.py decide  --motion motion_scores.npz --beta 0.5 --out gate_decisions.csv
    python motion_gate.py selftest        # synthetic check, takes seconds

Python
    from motion_gate import MotionScores, fit_beta, apply_gate
    ms = MotionScores.load("motion_scores.npz")
    beta, table = fit_beta(cv_sims, cv_ids, ms, image=["clip_b32", "dinov2_b"], video=["vjepa2_l"])
    combined = apply_gate(test_sims, np.arange(1001, 1103), ms, beta,
                          image=["clip_b32", "dinov2_b"], video=["vjepa2_l"])
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

BETA_GRID = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)
ALL_IDS = np.arange(1, 1103)
FRAME_SIZE = 64

# ---------------------------------------------------------------------------- 1. motion per video
def video_motion(path: str | Path) -> float:
    """Mean absolute frame-to-frame change (grayscale, 64x64, values 0..1) over the whole video."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    prev, diffs = None, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE), interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        if prev is not None:
            diffs.append(float(np.abs(g - prev).mean()))
        prev = g
    cap.release()
    if not diffs:
        raise IOError(f"fewer than 2 frames in {path}")
    return float(np.mean(diffs))

@dataclass
class MotionScores:
    ids: np.ndarray       # video IDs, sorted
    motion: np.ndarray    # raw motion per video

    @classmethod
    def compute(cls, data_root: str | Path, ids: np.ndarray = ALL_IDS, progress: bool = True) -> "MotionScores":
        root = Path(data_root)
        values = []
        for k, vid in enumerate(ids):
            folder = "stimuli_train" if vid <= 1000 else "stimuli_test"
            values.append(video_motion(root / folder / f"{int(vid):04d}.mp4"))
            if progress and (k + 1) % 100 == 0:
                print(f"  {k + 1}/{len(ids)} videos")
        return cls(np.asarray(ids, dtype=int), np.asarray(values, dtype=np.float64))

    def save(self, path: str | Path) -> None:
        np.savez(path, ids=self.ids, motion=self.motion)

    @classmethod
    def load(cls, path: str | Path) -> "MotionScores":
        with np.load(path) as z:
            return cls(z["ids"].astype(int), z["motion"].astype(np.float64))

    def zscores(self, video_ids: Sequence[int]) -> np.ndarray:
        """Log motion of `video_ids`, z-scored across exactly these videos."""
        video_ids = np.asarray(video_ids, dtype=int)
        rows = np.searchsorted(self.ids, video_ids)
        if not np.array_equal(self.ids[np.clip(rows, 0, len(self.ids) - 1)], video_ids):
            raise KeyError("some video IDs have no motion score")
        x = np.log(self.motion[rows] + 1e-4)
        return (x - x.mean()) / (x.std() + 1e-12)

# ---------------------------------------------------------------------------- 2. gate weights
def gate_weights(m: np.ndarray, beta: float, members: Sequence[str],
                 image: Sequence[str], video: Sequence[str]) -> dict[str, np.ndarray]:
    """Per-candidate multiplier for every member, shape (n_candidates,)."""
    out = {}
    for name in members:
        sign = 1.0 if name in video else -1.0 if name in image else 0.0
        out[name] = np.exp(sign * beta * m)
    return out

def _row_z(a: np.ndarray) -> np.ndarray:
    return (a - a.mean(axis=1, keepdims=True)) / (a.std(axis=1, keepdims=True) + 1e-12)

def _col_z(a: np.ndarray) -> np.ndarray:
    return (a - a.mean(axis=0, keepdims=True)) / (a.std(axis=0, keepdims=True) + 1e-12)

def apply_gate(sims: Mapping[str, np.ndarray], candidate_ids: Sequence[int], motion: MotionScores,
               beta: float, image: Sequence[str], video: Sequence[str], col_z: bool = True) -> np.ndarray:
    """Blend member score matrices (n_recordings, n_candidates); columns = candidate_ids in order."""
    _check_groups(sims, image, video)
    m = motion.zscores(candidate_ids)
    w = gate_weights(m, beta, list(sims), image, video)
    total = sum(_row_z(np.asarray(s, dtype=np.float64)) * w[name][None, :] for name, s in sims.items())
    return _col_z(total) if col_z else total

# ---------------------------------------------------------------------------- 3. choose beta (training CV only)
def _diag_ranks(sim: np.ndarray) -> np.ndarray:
    """Rank (1 = best) of the diagonal entry in each row; ties count against."""
    return (sim >= np.diag(sim)[:, None]).sum(axis=1)

def _pct_rank(sim: np.ndarray) -> float:
    return float(((_diag_ranks(sim) - 1) / (sim.shape[1] - 1)).mean())

def fit_beta(cv_sims: Mapping[int, Mapping[str, Sequence[np.ndarray]]],
             cv_ids: Mapping[int, Sequence[np.ndarray]],
             motion: MotionScores, image: Sequence[str], video: Sequence[str],
             grid: Sequence[float] = BETA_GRID, col_z: bool = True) -> tuple[float, list[dict]]:
    """Choose beta by mean percentile rank of the true video on training-only CV folds (lower is better).

    cv_sims[subject][member][fold] = (n_val, n_val) scores, rows = held-out training recordings,
                                     columns = the same recordings' videos, same order (true on the diagonal)
    cv_ids[subject][fold]          = video IDs of those columns
    Returns (beta, table). The table has one row per beta: overall score and the per-subject scores, so you can
    see whether the subjects agree.
    """
    subjects = list(cv_sims)
    table = []
    for beta in grid:
        per_sub = {}
        for s in subjects:
            members = cv_sims[s]
            _check_groups(members, image, video)
            n_folds = len(next(iter(members.values())))
            per_sub[s] = float(np.mean([
                _pct_rank(apply_gate({n: members[n][k] for n in members}, cv_ids[s][k], motion,
                                     beta, image, video, col_z))
                for k in range(n_folds)]))
        table.append({"beta": float(beta), "pct_rank": float(np.mean(list(per_sub.values()))), **per_sub})
    best = min(table, key=lambda r: (r["pct_rank"], abs(r["beta"])))   # ties -> smaller |beta|
    return best["beta"], table

def print_fit_table(table: list[dict]) -> None:
    subs = [k for k in table[0] if k not in ("beta", "pct_rank")]
    print(f"{'beta':>6}{'mean pct rank':>15}" + "".join(f"{'sub-' + str(s).zfill(2):>9}" for s in subs))
    best = min(r["pct_rank"] for r in table)
    for r in table:
        mark = "  <- best" if r["pct_rank"] == best else ""
        print(f"{r['beta']:>6.2f}{r['pct_rank']:>15.4f}" + "".join(f"{r[s]:>9.4f}" for s in subs) + mark)
    print("(percentile rank of the true video on training CV folds: 0 = perfect, 0.5 = chance, lower is better)")

def _check_groups(members: Mapping, image: Sequence[str], video: Sequence[str]) -> None:
    unknown = [n for n in list(image) + list(video) if n not in members]
    if unknown:
        raise KeyError(f"members {unknown} are not in the score matrices ({list(members)})")
    if set(image) & set(video):
        raise ValueError("a member cannot be both 'image' and 'video'")

# ---------------------------------------------------------------------------- 4. human-readable decision
def decision_table(motion: MotionScores, candidate_ids: Sequence[int], beta: float) -> list[dict]:
    m = motion.zscores(candidate_ids)
    w_img, w_vid = np.exp(-beta * m), np.exp(beta * m)
    rows = []
    for vid, mz, a, b in zip(candidate_ids, m, w_img, w_vid):
        favor = "video models" if b > a * 1.05 else "image models" if a > b * 1.05 else "equal"
        rows.append({"video_id": int(vid), "motion_z": round(float(mz), 3),
                     "image_weight": round(float(a), 3), "video_weight": round(float(b), 3), "favor": favor})
    return rows

# ---------------------------------------------------------------------------- self-test on synthetic data
def selftest() -> None:
    """Two synthetic cases: (a) image models are right on static videos and video models on moving ones,
    (b) no such pattern. fit_beta should pick beta > 0 in (a) and stay near 0 in (b)."""
    rng = np.random.default_rng(0)
    ms = MotionScores(ALL_IDS, np.exp(rng.standard_normal(len(ALL_IDS))))
    for effect in (True, False):
        cv_sims, cv_ids = {}, {}
        for s in range(1, 7):
            ids = np.sort(rng.choice(np.arange(1, 1001), 900, replace=False))
            folds = np.array_split(rng.permutation(ids), 5)
            cv_ids[s] = [np.asarray(f) for f in folds]
            cv_sims[s] = {"clip": [], "vjepa": []}
            for f in folds:
                n = len(f)
                mz = ms.zscores(f)
                t = np.tanh(mz) if effect else np.zeros(n)
                eye = np.eye(n)
                cv_sims[s]["clip"].append(rng.standard_normal((n, n)) + 1.5 * (1 - t)[:, None] * eye)
                cv_sims[s]["vjepa"].append(rng.standard_normal((n, n)) + 1.5 * (1 + t)[:, None] * eye)
        beta, table = fit_beta(cv_sims, cv_ids, ms, image=["clip"], video=["vjepa"])
        print()
        print(f"synthetic case: {'video models better on moving videos' if effect else 'no motion effect'}")
        print_fit_table(table)
        print(f"chosen beta = {beta:+.2f}")
        if effect:
            assert beta > 0, "should favor video models on moving videos"
        else:
            assert abs(beta) <= 0.25, "should stay close to no gating"
        # beta = 0 without column z equals the plain equal-weight blend
        test = {"clip": rng.standard_normal((102, 102)), "vjepa": rng.standard_normal((102, 102))}
        plain = _row_z(test["clip"]) + _row_z(test["vjepa"])
        assert np.allclose(apply_gate(test, np.arange(1001, 1103), ms, 0.0, ["clip"], ["vjepa"], col_z=False), plain)
    print()
    print("selftest passed")

# ---------------------------------------------------------------------------- command line
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compute", help="measure motion of all 1,102 videos")
    c.add_argument("--data-root", default="eeg-mind-reading-challenge")
    c.add_argument("--out", default="motion_scores.npz")
    d = sub.add_parser("decide", help="print / save the per-video decision for the 102 test videos")
    d.add_argument("--motion", default="motion_scores.npz")
    d.add_argument("--beta", type=float, required=True)
    d.add_argument("--out", default="gate_decisions.csv")
    sub.add_parser("selftest", help="synthetic check")
    args = p.parse_args()

    if args.cmd == "compute":
        ms = MotionScores.compute(args.data_root)
        ms.save(args.out)
        print(f"saved {args.out}: {len(ms.ids)} videos, motion min {ms.motion.min():.4f}, "
              f"median {np.median(ms.motion):.4f}, max {ms.motion.max():.4f}")
    elif args.cmd == "decide":
        rows = decision_table(MotionScores.load(args.motion), np.arange(1001, 1103), args.beta)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        counts = {k: sum(r["favor"] == k for r in rows) for k in ("image models", "equal", "video models")}
        print(f"beta {args.beta:+.2f}: " + ", ".join(f"{v} videos favor {k}" for k, v in counts.items()))
        print(f"saved {args.out}")
    else:
        selftest()

if __name__ == "__main__":
    main()
