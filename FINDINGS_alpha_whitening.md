# Finding: the post-PCA StandardScaler makes ridge alpha a no-op

From another Claude session (krabiki-e1), 2026-10-18. Diagnostic only: nothing in `src/`, `scripts/`, `results/` or
`submissions/` was changed. Please verify, then decide whether to adopt it.

## TL;DR
- Retrieval-CV alpha selection in `src/eegmr/linear.py` is **implemented correctly**.
- PROGRESS.md's "alpha barely matters" is real, but the reason is structural. The final `StandardScaler` in
  `_feature_transform` whitens the PCA output, so **every singular value of Z is identical** (s² = n). Ridge then
  shrinks all directions by the same factor. That only rescales the predicted deviation from the mean ERP, and
  correlation matching is nearly blind to that.
- **Dropping the final StandardScaler** (keeping scaler → PCA 95% → ridge, alpha by retrieval CV on training folds)
  improved the dev results:

| Pipeline (CLIP B-32, 8 frames, 0–3 s, window 5:60, alpha by retrieval CV) | top-1 | top-5 | median rank |
|---|---|---|---|
| current: scaler → PCA → **scaler** → ridge | 4.2% | 16.9% | 25.5 |
| proposed: scaler → PCA → ridge | **5.4%** | **21.1%** | **17.5** |

Per subject (current → proposed):

| sub | alpha | top-1 | top-5 | median rank | CV score range over alphas |
|---|---|---|---|---|---|
| 01 | 100 → 1e4 | 3.9 → 7.8% | 19.6 → 24.5% | 23 → 15 | 0.3373–0.3409 → 0.2783–0.3373 |
| 02 | 3162 → 1e4 | 2.9 → 4.9% | 20.6 → 24.5% | 21 → 13.5 | 0.3585–0.3619 → 0.3155–0.3587 |
| 03 | 3162 → 1e4 | 5.9 → 5.9% | 15.7 → 23.5% | 27 → 18 | 0.3665–0.3705 → 0.3071–0.3672 |
| 04 | 100 → 1e4 | 3.9 → 2.9% | 11.8 → 11.8% | 31 → 23.5 | 0.3978–0.4019 → 0.3469–0.3977 |

- Median rank improves on all 4 subjects. Top-5 gains about 4 pts (≈2 SE). The top-1 gain (+1.2 pts) is within noise
  (SE ≈ 1.1).
- The CV curve now has a real minimum, and retrieval CV picks alpha = 1e4 for every subject, an interior grid point.
  Test labels were used only for scoring, never for picking alpha.
- The "whitened" row reproduces PROGRESS.md's canonical `enc_clip_b32_f8_t1.5s_w5-60_retrieval` numbers exactly
  (4.2 / 16.9 / 25.5).

## Why (the math)
After `StandardScaler → PCA → StandardScaler`, the columns of Z are uncorrelated (PCA) and have unit variance (scaler),
so ZᵀZ = n·I. Measured on sub-01: all 233 values of s² equal 968.00. The ridge solution in SVD form is

    pred = Z_val V diag(s / (s² + α)) Uᵀ (Y − ȳ) + ȳ

With every s equal, `s/(s²+α)` is a single scalar. α no longer weighs strong against weak feature directions. It only
scales the video-specific deviation relative to the mean response ȳ, which is the same for every candidate. Pearson
correlation is scale-invariant, so the ranking barely moves, hence the flat CV curve. Without the second scaler the
PCA components keep their variance ordering. Ridge then shrinks the low-variance (noisier) components more, which is
the regularization we want.

## Checks done on the existing code (all passed)
1. The closed-form SVD ridge in `retrieval_cv_curve` matches `sklearn.linear_model.Ridge` to ~1e-7 (float32 noise)
   for α ∈ {1, 1e3, 1e6}.
2. `retrieval_cv_curve` scores only `window` (5:60). Fitting ridge on window-only targets gives the same window
   predictions as fitting on all 17×300 (max diff 1e-7), so the training crop cannot affect retrieval at fixed α.
3. Video IDs are unique per subject, so KFold over trials is KFold over videos and nothing leaks between folds.
4. `EncodingModel(alpha_selection="retrieval")` uses `argmin(cv_curve)`.
5. Diagnostic only: on the dev test sets, the CV-chosen α was within ≈0.003–0.007 mean-percentile-rank of the
   oracle-best α for every subject in the whitened pipeline. The selection method was never the problem.

## Suggested change
In `src/eegmr/linear.py`, `_feature_transform` (line 33–34):

```python
def _feature_transform(pca_variance: float, whiten: bool = False) -> Pipeline:
    steps = [StandardScaler(), PCA(n_components=pca_variance, svd_solver="full")]
    return make_pipeline(*steps, *([StandardScaler()] if whiten else []))
```

Then expose it via `EncodingModel` and `--whiten` in `scripts/predict.py`, so the old behaviour stays reproducible.

`retrieval_cv_curve` already handles non-centred Z correctly as long as ridge's intercept is used. Note its comment
"Z_trn is column-centred by the final scaler" stops being true without the scaler. PCA output is still centred on the
data it was fitted on, so the closed form stays valid, but please re-check that, or centre explicitly:
`zm = Z_trn.mean(0); Z_trn -= zm; Z_val -= zm`.

Rerun to log it properly:

```bash
python scripts/predict.py --features clip_b32 --frames 8 --alpha-selection retrieval --subjects 1 2 3 4 5 6
```

## Follow-ups this opens
- The alpha grid now matters. 1e4 was picked everywhere, so a finer grid around 1e3–1e5 is worth a look.
- PCA at 95% variance was tuned for the whitened setup. With real ridge shrinkage, try no PCA at all (ridge on all
  standardized features; n ≈ 900, so solve it in kernel/dual form) or a higher variance threshold.
- Re-test the earlier "alpha / crop doesn't matter" conclusions and the z-score / CSLS variants on the unwhitened
  pipeline.
- The same issue will hit any future ridge that uses scaler → PCA → scaler (decoding direction, DINOv2, audio).

## Reproduction scripts
Session scratchpad (may not survive). The logic is fully described above:
- `/tmp/claude-1901341344/-home-kparpie-Krabiki/13ce423f-42ee-4524-a625-dc390d6adbe6/scratchpad/validate_alpha.py`: checks 1–5
- `/tmp/claude-1901341344/-home-kparpie-Krabiki/13ce423f-42ee-4524-a625-dc390d6adbe6/scratchpad/no_sc2.py`: whitened vs unwhitened comparison

Run from the repo root with `.venv/bin/python <script>`.
