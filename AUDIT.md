# General audit: progress, assumptions, leakage, validation, failure points

Date: 2026-10-18. Scope: the headline submission `submissions/final_csls/` (track-matching 7297a59, experiment
`fin5_ctr_m5`) and everything it rests on. Authors: krabiki-e1 (sections 1, 3, 5, 6, 7) and krabiki-2f (section 4,
code-level checks). Numbers: dev = sub-01..04 test sets, top-1 / top-5 (chance 1.0 / 4.9). Details per model are in
`MODEL_BACKLOG.md`; history in `PROGRESS.md`.

> **Update (2026-10-18 evening):** the headline is now `final_v4_csls` (v4, dev 33.3 / 68.9). It has a quick audit: v3 rescore identical and v3 label-shuffle at chance (sub-01 1.0 / 3.9, sub-03 1.0 / 5.9); v4 rescore identical, sub-05/06 files == saved matrices, and the pooling code loads only other subjects' training data with zero-shot CV. The full controls below (§4) were run on `final_csls`. §5's projections refer to `final_csls`; for v3 they are sub-05 ~26 / ~58, sub-06 ~18 / ~48 (optimistic).

## 1. Progress, in one table

| Stage | Model | Dev top-1 / top-5 | Selected by |
|---|---|---|---|
| Baseline | M0 AlexNet encoding (organizers' recipe, re-implemented) | 2.0 / 8.6 | none |
| Fix | M2 CLIP encoding, alpha bug removed | 5.4 / 21.1 | train-CV (alpha) |
| Direction | M3 decoding EEG → CLIP | 14.0 / 36.0 | train-CV |
| Noise | M8a whitened decoder input | 14.7 / 42.2 | nested train-CV |
| Ensemble | M11 → M13 → M17 (DINOv2-patch, contrastive head, V-JEPA 2) | 15.2 → 17.4 → 20.8 / 45.3 → 46.8 → 48.0 | train-CV weights + LOFO |
| **Final** | **5 members + test centering + CSLS k=5 per member** | **24.0 / 57.1** | exhaustive train-CV weights + LOFO |
| Pending (v2) | final + AlexNet conv2 (pooled 6×6) member | 29.4 / 63.0 | train-CV + LOFO (krabiki-51, not yet final) |

## 2. Loose assumptions (and how much each matters)

| # | Assumption | Evidence for / against | Risk |
|---|---|---|---|
| A1 | sub-05/06 test recordings are processed like sub-01..04 (same averaging, same shift) | Same test/train std ratio (0.68, 0.63 vs dev 0.69–0.78) and the same anti-correlated mean ERP (−0.78, −0.73 vs dev −0.68 to −0.94) | low |
| A2 | Each test video appears exactly once per subject (102 × 102) | Organizers' format; dev `tst_stim` confirms it for sub-01..04. CSLS benefits from this structure. | low for this challenge; **high if the hidden scoring set differs** (see F4) |
| A3 | The 50–600 ms window is right for all subjects | Originally the organizers' notebook choice; later confirmed by train-CV (2f's 60-config sweep, decoder CV on 6 subjects, cc's window tests) | low |
| A4 | Ridge alphas chosen on the full training set (not nested inside the ensemble CV) don't bias the weight choice | Alphas are interior grid points and similar across subjects; the bias applies equally to all configurations | low (inflates LOFO slightly, uniformly) |
| A5 | Train-CV ranks configurations like the real test does | Spearman(LOFO, dev) = **0.82** over the 30 final configs (0.60 within CSLS-on); encoding sweep 0.68. Disagreements seen: M8b reliability cosine, M12 linear head, MLP. | medium for small differences (< ~0.003 LOFO) |
| A6 | Dev scores generalize to sub-05/06 | **No, on absolute level:** CV says sub-05/06 are weaker subjects (see F1) | **high for expectations** |
| A7 | Pretrained feature models haven't seen our clips | Moments in Time is excluded explicitly (V-JEPA 2 checked). Web-scale data (CLIP LAION-2B, V-JEPA YouTube) can't be ruled out. | low (affects stimulus features only, not EEG labels) |
| A8 | The organizers' channel order / preprocessing statement is correct | THINGS-EEG2 stores channels in the same order; spatial-correlation vs distance in our data matches it (Spearman −0.81..−0.86) | low |
| A9 | CSLS and test centering are allowed | CSLS confirmed by the organizers (2026-10-18). Centering is rule 5 (unsupervised normalization of test recordings). | resolved |

## 3. Validation: what our numbers do and don't mean
- **Two validation layers.** Train-CV (5-fold on training videos, KFold seed 0, all 6 subjects) makes every choice. Dev
  (sub-01..04 test sets) only confirms, with a paired Wilcoxon on per-trial ranks. Dev labels are allowed for
  validation and model selection (rule 4); we used them for confirmation and reporting.
- **CV underestimates absolute performance.** CV trains on 80% of trials and ranks ~170–195 held-out single trials;
  test ranks 102 cleaner (likely repeat-averaged) recordings. On sub-01..04 the final model's dev / CV ratio is ×2.0
  (top-1) and ×1.6 (top-5). Use CV for *relative* comparisons and "expected top-k of 102" for scale.
- **Selection bias at the last step is small.** Among the 21 CSLS-on final configurations, dev top-1 spans 22.3–24.5 and top-5
  53.9–57.8. The CV-chosen headline (24.0 / 57.1) is 0.6 pts (avg of top-1 / top-5) below the best-on-dev config, so it
  wasn't picked by dev.
- **Researcher degrees of freedom are the larger bias.** Over the session, dozens of variants were *looked at* on dev
  before being dropped or adopted (member candidates, window ideas, normalizations). Each adoption also had to pass
  train-CV, but the overall dev numbers are still somewhat optimistic. A fair guess is 1–3 pts on top-1 / top-5.
- **Noise level:** with 408 dev trials the SE is ~±2 pts top-1 / ±2.5 pts top-5 at our levels, and with 102 trials per subject it's
  ~±4 / ±5 pts. Per-subject dev differences under ~5 pts are noise.

## 4. Leakage and code-level checks
Evidence: `results/audit/` on track-proven (edfea8d). The controls ran on a detached checkout of track-matching 7297a59
with the audited code unchanged.

| Check | Result | Notes |
|---|---|---|
| **Data flow: test EEG** | **PASS** | Enters only at `ensemble.py:456`. Then crop / decimate, test centering (mean of the 102 *unlabeled* test recordings, rule 5), whitening and standardization with **training** statistics, and per-member CSLS on the unlabeled 102 × 102 matrix (organizer-approved). No fitted statistic (scaler, PCA, whitener, ridge, alpha, ensemble weights, early stopping) touches test EEG. |
| **Data flow: dev labels** | **PASS** | Used only after `write_submission`, for scoring / reporting (`ensemble.py:469/486`, `compare_table`). The CV block only calls `load_train`. No sub-01..04 test data is used for sub-05/06. |
| **Data flow: test-video features** | **PASS** | Used as candidates only, standardized with training-video statistics, extracted strictly per video. The AlexNet-p6 test rows don't change when the training rows are zeroed. |
| **Label-shuffle control, full final pipeline** (e1) | **PASS** | Using track-matching's own `ensemble.py` functions (all 5 members, alphas re-chosen on shuffled data, centering, CSLS k=5, headline weights 0/4/3/2/1). The reconstruction on real data reproduces the committed headline ranking exactly (sub-01). With shuffled training EEG ↔ video pairs: sub-01..04 top-1 0.0 / 0.0 / 2.0 / 2.0, top-5 3.9 / 4.9 / 3.9 / 5.9, **mean 1.0 % / 4.7 %, median rank ≈ 50 = chance** (1.0 / 4.9 / 51.5). Earlier, the same control gave chance for M3, M8a and M12. |
| **Row-permutation test** | **PASS** | Permuted test rows (sub-01, sub-05) → output rows equal baseline[perm], bit-identical 102/102. Centering and CSLS are permutation-equivariant (also checked separately by e1: max diff 0.0). Nothing depends on file order. |
| **Determinism** | **PASS** (same machine) | All 5 member test matrices, including the contrastive head over 3 SGD seeds, are bit-identical on a from-scratch recompute (sub-01, sub-05), as are 6 CV-fold files. Cross-hardware not tested. |
| **Re-score vs committed files** | **PASS** | A fresh sub-05 equals the committed `final_csls/sub-05`; a fresh sub-01 equals the saved dev matrix (top-1 29.4 % both). |
| **Selection audit** | **CONCERN** (legitimate under rule 4) | Weights, alphas and strategy are training-only. The *candidate set* was screened on dev about 10 times: window 5:60 (first picked on dev, later CV-confirmed), test centering (adopted mainly on dev evidence, later LOFO +0.0008), member pool (L-14 dropped on dev, members dev-confirmed before the search), M8b rejected on dev, the s3 exclusion rule includes a dev check, and the CSLS k = 5 vs 10 tie-break. So dev 24.0 / 57.1 is somewhat optimistic for sub-05/06 (see §3, §5). |
| **Environment** | **PASS / minor concerns** | All 13 third-party imports are pinned and match the installed versions (numpy 2.4.6, torch 2.11.0+cu128, scikit-learn 1.9.1, open_clip 3.3.0, timm 1.0.30, transformers 5.18.0, …). The V-JEPA HF revision is pinned; **CLIP B-32 (open_clip tag) and DINOv2 (timm) are not revision-pinned** (cached snapshots 1a25a44 / 4685c99 recorded). All-frames caches store fp16 embeddings. fp16-vs-fp32 re-extraction: not run. |

Caveat (not a leak): ridge alphas are chosen on the full training set and reused inside the CV folds (non-nested), so
the CV / LOFO numbers are slightly optimistic, uniformly across configurations.

## 5. Expected performance on the scored subjects (sub-05/06)
Per-subject train-CV of the final model (expected top-k of 102), scaled by the dev / CV ratio measured on sub-01..04:

| Subject | CV E[top-1] / E[top-5] | Dev (if labelled) | Projected test |
|---|---|---|---|
| sub-01 | 13.7 / 40.1 | 29.4 / 58.8 | – |
| sub-02 | 13.7 / 39.3 | 19.6 / 61.8 | – |
| sub-03 | 9.4 / 27.1 | 20.6 / 49.0 | – |
| sub-04 | 11.2 / 35.5 | 26.5 / 58.8 | – |
| **sub-05** | 9.2 / 28.1 | – | **~18 / ~45** (sub-03 has nearly the same CV and got 20.6 / 49.0) |
| **sub-06** | 6.1 / 21.5 | – | **~12 / ~35** (weakest subject; only 850 training trials) |

**Realistic scored average: ~15% top-1 / ~40% top-5,** not the 24 / 57 dev average. The reason is that the two scored
subjects are the two weakest by CV. Label-free behaviour on sub-05/06 is normal (≈60 distinct top-1 picks out of 102,
max 5–6 picks per candidate, score margins and agreement with M8a in the dev range), so there's no sign of a pipeline
failure there.

## 6. Possible failure points on the actual test set

| # | Failure point | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| F1 | sub-05/06 score well below the dev average (weaker subjects) | high | expectation only | Report the projected ~15 / 40, not 24 / 57 |
| F2 | sub-06 specifically: fewest trials (850), lowest CV | high | lower score | Nothing cheap. Subject pooling (paused) would help the weakest subjects most. |
| F3 | Row-order / ID errors in the submission file | low | fatal (unscorable) | Writer asserts shape / dtype / permutation. Dev files re-scored from disk match. Row-permutation test (2f) pending. |
| F4 | The hidden scoring differs in structure from dev (e.g. a different candidate set or repeats) | low | CSLS and centering assume one batch of 102 distinct videos per subject. If violated, the gain could shrink. | `final_nocsls/` backup exists; the organizers confirmed CSLS |
| F5 | Non-determinism: contrastive seeds and GPU float differences | medium | small rank flips; a rerun won't be bit-identical | Committed final files are the submission. Seed robustness shown (weights ≤ one 0.1 step). Ship weights / similarities. |
| F6 | Reproduction environment (feature extraction on another machine: video decoder, fp16, library versions) | medium | small feature differences → a few rank flips | Pin versions (requirements.txt), fixed frame indices, fp16 ≈ fp32 (cosine ≥ 0.9999). Consolidation should ship cached features or weights. |
| F7 | The final pipeline is spread across branches (track-matching ensemble.py, track-proven decoders, track-vjepa / dinov2 extractors) | high until consolidated | judges can't easily rerun it | Consolidation (G1–G4) |
| F8 | A v2 with AlexNet replaces the headline too fast | medium | unverified pipeline in the submission | Adopt v2 only after committed extraction code, CV selection, an independent rescore and the same audit controls |
| F9 | Dev numbers are optimistic (researcher degrees of freedom) | high | report credibility | State CV-first selection, the controls and the projected sub-05/06 numbers in the presentation |

## 7. Recommendations
0. **Verdict:** no leakage found. Every fitted statistic is training-only, the full pipeline falls to chance under label
   shuffling, outputs don't depend on row order, and the committed files reproduce bit-for-bit on this machine. The
   main honest caveats are dev optimism from candidate screening, and that the two scored subjects are weaker than the
   dev average.
1. Keep `final_csls/` as the submitted set until v2 (AlexNet) passes the same checks: committed code, CV selection,
   independent rescore, and the same controls (shuffle, row permutation, determinism).
1b. Pin the CLIP B-32 and DINOv2 weights by revision (open_clip / timm snapshot hashes 1a25a44 / 4685c99) during consolidation.
2. In the report, quote dev **and** the projected sub-05/06 range, plus the CV-first protocol and the controls.
3. Consolidate before the deadline (one branch, one command from saved weights, README, tests). It removes F7 and most of
   F5 / F6.
4. If time allows, the biggest remaining lever for the weak scored subjects is more data per subject (pooling or
   pretraining). That's a user decision (pooling is paused).
