# Krabiki: motion gate (pre-pipeline weighting step)

This branch currently provides one reusable pipeline step: **`track-timestaged/motion_gate.py`**. It decides, for every
candidate video, how much to trust the **image models** (e.g. CLIP, DINOv2) versus the **video models** (e.g. MViT,
V-JEPA) when their scores are combined. It is not a model of the EEG and trains nothing on EEG: it only produces weights.

The rest of `track-timestaged/` is the earlier time-staged experts track (DINOv2 + MViT-v2 + hand-made features,
encoding per EEG time window; 9.1% top-1 honest, see the git history for its documentation). It is not needed for the
motion gate.

## Why

An ensemble with fixed weights treats every candidate the same. But image models describe a single scene well, while
video models describe movement. The gate lets **static candidates lean on image models and moving candidates on video
models**. Whether this helps on our data is an empirical question; the gate is designed so it can only be switched on
if training data supports it.

## How it works

1. **Measure motion (`compute`).** For each of the 1,102 videos: shrink every frame to 64 × 64 grayscale and average
   the absolute change between consecutive frames over the whole 3 s. Only the video files are used, no EEG and no
   labels. Result on our videos: min 0.0005, median 0.017, max 0.10, so there is a wide spread from static to moving.
2. **Turn motion into weights.** For the candidates being ranked, motion is log-scaled and z-scored (m). Each member's
   score for that candidate is multiplied by:
   - image members: `exp(-beta * m)`
   - video members: `exp(+beta * m)`
   - other members: `1`

   With beta = 0.5 and one image plus one video model, a very static candidate is judged about 73% by the image model and
   a very moving one about 73% by the video model. **beta = 0 means no gating** (the plain equal-weight blend).
3. **Choose beta on training data only (`fit_beta`).** beta is picked from a grid (−1 … +1) by the mean rank of the
   true video on **training-only cross-validation** score matrices. No test data or test labels are used, so beta is
   fixed before anything is scored. The printed table shows the score per subject, so one can see whether the subjects
   agree. If they don't, or beta = 0 wins, the gate should stay off.
4. **Apply (`apply_gate`).** Each member's test score matrix is row-z-scored, scaled per candidate column with its gate
   weight, summed, and finally every column is z-scored across the recordings. That last step is unsupervised and stops a
   candidate from being pushed up for every recording just because of its weight. With beta = 0 and no column step,
   the output equals the plain blend exactly.

All models still run on every test recording. The gate only changes how their scores are added up, per candidate. It
does not pick a model per recording (we can't: which video was watched is exactly what we don't know).

## Usage

Needs only `numpy` and `opencv-python`; it imports nothing from either track.

~~~bash
cd track-timestaged
python motion_gate.py selftest                                   # synthetic check, seconds
python motion_gate.py compute --data-root ../eeg-mind-reading-challenge --out motion_scores.npz   # ~1-2 min
python motion_gate.py decide --motion motion_scores.npz --beta 0.5 --out gate_decisions.csv     # readable per-video table
~~~

`motion_scores.npz` (18 KB, one number per video, no frames) is committed, so `compute` can be skipped.

In a pipeline (score matrices as numpy arrays, columns = candidate video IDs in order):

~~~python
from motion_gate import MotionScores, fit_beta, apply_gate, print_fit_table

ms = MotionScores.load("motion_scores.npz")
IMAGE, VIDEO = ["clip_b32", "dinov2_b_patch_all"], ["vjepa2_l_all"]

# cv_sims[subject][member][fold]: (n_val, n_val) scores of held-out TRAINING recordings vs their own videos
#                                 (true match on the diagonal); cv_ids[subject][fold]: those videos' IDs
beta, table = fit_beta(cv_sims, cv_ids, ms, IMAGE, VIDEO)
print_fit_table(table)

combined = apply_gate(test_sims, candidate_ids, ms, beta, IMAGE, VIDEO)   # test_sims: member -> (102, 102)
~~~

The main pipeline's `scripts/zero_cost.py` already stores training-only CV matrices per fold and the test matrix per
member (`member(...)["cv"]` and `["test"]`, folds = `KFold(5, shuffle=True, random_state=0)` over the training trials),
so it can feed `fit_beta` directly.

## Status

- **Tested:** the self-test on synthetic scores. When video models are really better on moving videos it picks
  beta = +1.0 (all 6 subjects agree); when there is no such pattern it picks beta = 0. `compute` and `decide` ran on all
  1,102 real videos.
- **Not yet tested on real EEG scores.** The next step is running `fit_beta` on the main pipeline's training-CV
  matrices, then comparing beta = 0 with the chosen beta on the sub-01..04 test sets. Adopt only if both agree.

## Rules

Uses only the video files and training data. No test labels, no training on test data, the same procedure for every
subject. The column normalization is unsupervised normalization of test scores (rule 5).
