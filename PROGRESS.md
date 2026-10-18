# Progress: track-proven

## Current best (mean over sub-01..04 test sets, 102 candidates)
| Experiment | top-1 | top-5 | median rank |
|---|---|---|---|
| **decw_clip_b32_f8_t1.5s_w5-60** (decoding with noise-whitened EEG input; `--head decoding --dec-whiten`) | **14.7%** | **42.2%** | **7.5** |
| dec_clip_b32_f8_t1.5s_w5-60 (decoding: EEG 50–600 ms → standardized CLIP B-32 f8 features, ridge, cosine; alpha by retrieval CV) | 14.0% | 36.0% | 8.2 |
| enc_clip_b32_f8_t1.5s_w5-60_retrieval (encoding: CLIP → EEG, scaler → PCA 95% → ridge, correlation) | 5.4% | 21.1% | 17.5 |

Decoding direction found by session krabiki-e1 and reproduced in-package (`linear.DecodingModel`, `predict.py --head
decoding`). Train-CV agrees without test labels: 0.207–0.315 per subject vs about 0.33–0.44 for encoding.
Noise-whitened decoder input (M8 in MODEL_BACKLOG.md): paired vs plain decoder 187 better / 141 worse, p=0.002; train-CV better on all 6 subjects. Reliability-weighted cosine was tried and not adopted (CV and dev disagree).
Submission files: `submissions/decw_clip_b32_f8_t1.5s_w5-60/sub-0{5,6}_ranked_candidate_list.npy` (best); plain decoder in `submissions/dec_clip_b32_f8_t1.5s_w5-60/` (encoding files are
still in `submissions/enc_clip_b32_f8_t1.5s_w5-60_retrieval/`).
CSLS is dropped for now (2026-10-18). k=10 on top gave 6.1% / 22.3% / 17.1, not significant (p=0.36). `--csls-k` stays in the code, off by default.

> **2026-10-18: all results below the "Whitening fix" section were run with the whitened transform** (scaler → PCA →
> scaler, now `--whiten`). Their "alpha / crop / CSLS doesn't matter" conclusions are superseded where noted.

## Done
- **Stage 0:** `eegmr.evaluate`, `eegmr.submission` (row order preserved, values are video IDs, asserts). Self-test:
  oracle similarity → 100% from the reloaded file, random → chance (0.98% / 5.0%), file score == in-memory score.
- **Stage 1 (partial):** cached `alexnet` (torchvision IMAGENET1K_V1, `features.3`) and `clip_b32` features, 1 and 8 frames.
- **Stage 2 (first step):** encoding model (decimate to 100 Hz → scaler → PCA 95% → [scaler, removed 2026-10-18] → RidgeCV → EEG 17×300,
  correlation over samples 5:60). Re-implemented from the brief, no organizer code. Only the features vary:

| Features | top-1 | top-5 | median rank |
|---|---|---|---|
| AlexNet, 8 frames (baseline recipe) | 2.0% | 8.6% | 35.1 |
| AlexNet, 1 frame | 1.7% | 7.1% | 40.8 |
| CLIP B-32, 1 frame | 3.4% | 15.0% | 29.1 |
| CLIP B-32, 8 frames | 5.1% | 16.9% | 25.9 |

Chance: 1.0% / 4.9% / 51.5. With 4×102 trials, the standard error is about ±1.1 pts on top-1 and ±1.9 pts on top-5.

### EEG crop / matching window (CLIP B-32, 8 frames)
The raw 500 Hz EEG is cropped before filtering, so nothing after the cut leaks in.

| Trained on | Matched on | top-1 | top-5 | median rank |
|---|---|---|---|---|
| 0–3 s | 50–600 ms | 5.1% | 16.9% | 25.9 |
| 0–1.5 s | 50–600 ms | 4.4% | 15.9% | 25.8 |
| 0–1.5 s | 50–1000 ms | 2.9% | 13.7% | 26.5 |
| 0–1.5 s | 0–1500 ms | 2.9% | 13.2% | 27.8 |
| 0–3 s | 0–3000 ms | 2.9% | 12.5% | 33.1 |

The 3 s vs 1.5 s gap (5.1% vs 4.4% top-1) comes **only from the ridge alpha** that RidgeCV picks: 10,000 vs 3,162 on
sub-02..04. With the alpha forced to the 3 s value, the 1.5 s similarity matrices are bit-identical (max diff 0), because
ridge predicts each time point independently. On a per-trial basis the two are not significantly different (Wilcoxon p=0.14).
Widening the matching window hurts: the useful information sits in roughly the first 600 ms.

### Alpha selection (CLIP B-32, 8 frames, matched on 50–600 ms, new grid 1..1e9 in quarter decades)
| Trained on | Alpha chosen by | top-1 | top-5 | median rank |
|---|---|---|---|---|
| 0–3 s | GCV (old coarse grid, from above) | 5.1% | 16.9% | 25.9 |
| 0–3 s | GCV | 4.7% | 16.7% | 26.4 |
| 0–1.5 s | GCV | 4.4% | 15.9% | 26.0 |
| 0–1.5 s | retrieval CV (5-fold on training videos) | 4.2% | 16.9% | 25.5 |
| 0–3 s | retrieval CV | 4.2% | 16.9% | 25.5 (identical, as expected) |

**Superseded:** alpha only "barely matters" because of whitening (see the Whitening fix section). The retrieval CV curve (mean percentile rank of the true video on held-out training folds)
is flat to the third decimal from alpha 1 to 10^4 for every subject, so the choice within that range is essentially
arbitrary and the test differences above are noise (a few near-tied trials flipping). The CV score also estimates
per-subject quality without test labels (0.5 = chance):
sub-01 0.337, sub-02 0.359, sub-03 0.367, sub-05 0.388, sub-04 0.398, **sub-06 0.439** (weakest, fewest trials: 850).

### Z-scoring and CSLS (on the canonical pipeline above)
| Variant | top-1 | top-5 | median rank | paired vs baseline (408 trials) |
|---|---|---|---|---|
| baseline | 4.2% | 16.9% | 25.5 | – |
| z-score channels | 4.2% | 16.7% | 26.0 | p=0.96 |
| CSLS k=10 | 4.7% | 16.4% | 24.8 | better 180 / worse 153, p=0.14 |
| z + CSLS k=5 | 4.4% | 17.2% | 24.2 | p=0.22 |
| z + CSLS k=10 | 4.2% | 16.2% | 24.4 | p=0.28 |
| z + CSLS k=20 | 4.2% | 16.4% | 25.0 | p=0.12 |

- **Z-scoring is a no-op**: the organizers' EEG is already normalized. Per-channel std is about 0.47 for every channel
  and subject (max/min ratio 1.0–1.1), and the means are about 0.
- **CSLS works as intended on hubs** (the most popular video appears in 17 → 12 of 102 top-5 lists; never-ranked videos
  6.5 → 1.5), and median rank improves slightly in every variant, but no change is significant.

### Whitening fix (found by session krabiki-e1, verified here; details in `FINDINGS_alpha_whitening.md`)
The second StandardScaler made every PCA component unit-variance, so Z^T Z = n·I (sub-01: all 233 s² = 968.0; without it,
s² ranges from 304 to 31,575). Ridge then shrank every direction equally, alpha only rescaled the predictions, and
correlation matching ignores scale. With the scaler removed, ridge shrinks the weak, noisy PCA directions more.

| CLIP B-32, 8 frames, window 5:60 | top-1 | top-5 | median rank | paired (408 trials) |
|---|---|---|---|---|
| whitened (old, `--whiten`), retrieval CV | 4.2% | 16.9% | 25.5 | reproduces old numbers exactly |
| **unwhitened, retrieval CV** (0–1.5 s and 0–3 s identical) | **5.4%** | **21.1%** | **17.5** | vs whitened: better 252 / worse 130, **p=2e-13** |
| unwhitened, GCV | 5.4% | 20.8% | 17.6 | |
| unwhitened, retrieval CV + CSLS k=10 | 6.1% | 22.3% | 17.1 | vs unwhitened: p=0.36 |

Retrieval CV now picks alpha = 1e4 for every subject (interior grid point); GCV picks 5.6e3–1.8e4. Median rank
improves on all four dev subjects (23/21/27/31 → 15/14/18/24).

### Encoding sweep: PCA threshold × matching window (train-CV over 6 subjects, 73-alpha grid; `results/sweep_clip_b32_f8.csv`)
CV spans only 0.328–0.351 over 60 configs. The best are pca none / 0.99 / 0.95 with window 5:60 (0.3282 / 0.3283 / 0.3290).
Window ranking by CV: 5:60 < 5:80 < 0:60 < 0:80 < 5:50 < ... < 10:40. PCA: none ≈ 0.99 ≈ 0.95 < 0.9 < 0.8.
Spearman(CV, dev median rank) = 0.68. **Encoding is on a plateau, and the canonical config stands.**

### Frames and onset pooling (CLIP B-32 from the all-frames cache, 0–1.5 s, window 5:60; paired vs PIL f8 canonical)
| Pooling | dec CV | dec top-1 / top-5 / medR | enc CV | enc top-1 / top-5 / medR |
|---|---|---|---|---|
| f8 mean | 0.2547 | 13.5 / 36.5 / 8.6 | 0.3291 | 5.6 / 21.6 / 17.4 |
| f16 mean | 0.2550 | 13.7 / 36.5 / 9.4 | 0.3292 | 5.9 / 21.8 / 17.5 |
| f32 mean | 0.2549 | 14.0 / 36.3 / 9.6 | 0.3287 | 5.9 / 22.5 / 17.4 |
| f32 first 0.5 s | 0.2531 | 11.8 / 38.0 / 9.0 | **0.3260** | 5.9 / 22.8 / 17.4 |
| f32 first 1.0 s | **0.2524** | 11.8 / 36.8 / 9.0 | 0.3268 | 6.1 / 23.3 / 17.1 |
| f32 4 segments concat | 0.2546 | 13.7 / 36.8 / 9.8 (worse, p=0.04) | 0.3275 | 5.9 / 21.6 / 17.4 |
| f8 concat | 0.2540 | 12.5 / 37.0 / 8.6 | 0.3282 | 5.4 / 21.6 / 17.5 |

No variant is significantly better on dev (all p ≥ 0.06). **More frames do nothing.** CV slightly prefers onset-only
pooling for both heads (the largest CV gain seen for encoding), but the test effect isn't significant.
Why: CLIP barely changes within a clip. First vs last frame of the same clip has median cosine 0.90, while two
different clips have 0.45, so each 3 s clip is essentially one scene to CLIP.

Infrastructure: `eegmr.gpu` (shared torch module: multi-alpha ridge via one SVD, correlation/cosine similarity,
encoding and decoding retrieval CV; 16 parity tests in `tests/test_gpu.py`). All-frames feature cache
(`features.extract_all_frames`, 5 min for all 1,102 videos; frames per video vary from 45 to 92 because fps varies).

## Matching experiments (track-matching)
Same encoding model throughout (CLIP B-32, 8 frames, scaler → PCA 95% → ridge, EEG 0–1.5 s, alpha by retrieval CV, window 5:60);
only the similarity between measured and predicted EEG changes. Code: `src/eegmr/matching.py`, flags in `scripts/predict.py`
(`--erp-subtract`, `--match {corr,weighted,noise}`, `--match-power`, `--noise-cov`, `--noise-similarity`). All statistics come from the
subject's training trials; the weighted / noise statistics use 5-fold OOF predictions (same folds as `retrieval_cv_curve`, alpha fixed
to the chosen one, transform refit per fold). With all options off the output is bit-identical (`match_base` = 5.4 / 21.1 / 17.5).

**Variant choice by training-only CV** (`scripts/select_matching.py` → `results/matching_cv.csv`): mean percentile rank of the true video
on held-out folds, 0.5 = chance, lower is better. Matcher parameters are estimated inside each outer fold's training part from an inner
5-fold OOF, so nothing scored is used to fit the matcher. Mean over sub-01..04 (the picks are the same with all six subjects):

| Variant | no ERP subtraction | + ERP subtraction |
|---|---|---|
| corr (baseline) | 0.3120 | 0.3127 |
| weighted p=0.5 / 1 / 2 | **0.3145** / 0.3253 / 0.3410 | **0.3150** / 0.3256 / 0.3419 |
| noise, per-timepoint cov, corr | 0.3017 | 0.3018 |
| noise, per-timepoint cov, Euclid | **0.2924** | **0.2924** |
| noise, pooled cov, corr | 0.3060 | 0.3065 |
| noise, pooled cov, Euclid | 0.2972 | 0.2972 |

Picked: 2a = p 0.5, 2b = per-timepoint covariance + negative squared Euclidean distance (both with and without #1).

**Dev results** (sub-01..04 test sets, 102 candidates; paired Wilcoxon on the 408 per-trial ranks vs `match_base`, `scripts/compare_matching.py`):

| Experiment | sub-01 top1/top5/med | sub-02 | sub-03 | sub-04 | mean top-1 | mean top-5 | mean med. rank | better/worse | p |
|---|---|---|---|---|---|---|---|---|---|
| match_base | 7.8 / 24.5 / 15.0 | 4.9 / 24.5 / 13.5 | 5.9 / 23.5 / 18.0 | 2.9 / 11.8 / 23.5 | 5.4% | 21.1% | 17.5 | – | – |
| + #1 `match_erp` | 7.8 / 24.5 / 13.0 | 2.9 / 23.5 / 13.0 | 5.9 / 23.5 / 17.5 | 2.9 / 10.8 / 23.5 | 4.9% | 20.6% | 16.8 | 107 / 108 | 0.87 |
| + 2a `match_weighted_p0.5` | 4.9 / 27.5 / 14.0 | 4.9 / 25.5 / 17.5 | 5.9 / 22.5 / 16.0 | 2.9 / 11.8 / 19.0 | 4.7% | 21.8% | 16.6 | 189 / 174 | 0.81 |
| + 2b `match_noise_pertime_euclid` | 5.9 / 32.4 / 11.5 | 8.8 / 30.4 / 14.0 | 13.7 / 33.3 / 9.0 | 6.9 / 30.4 / 12.0 | **8.8%** | **31.6%** | **11.6** | 246 / 143 | **2.9e-7** |
| + #1 + 2a `match_erp_weighted_p0.5` | 4.9 / 24.5 / 13.0 | 6.9 / 24.5 / 16.0 | 5.9 / 22.5 / 17.0 | 1.0 / 12.7 / 19.0 | 4.7% | 21.1% | 16.2 | 185 / 177 | 0.87 |
| + #1 + 2b `match_erp_noise_pertime_euclid` | identical to 2b | | | | 8.8% | 31.6% | 11.6 | 246 / 143 | 2.9e-7 |

- **#1 ERP subtraction does nothing** (p=0.87). Under Euclidean matching it cancels exactly (the distance only sees m − p), so row 6 equals row 4 bit for bit.
- **2a reliability weighting does not help**: CV gets monotonically worse with p, and the best p=0.5 is still worse than plain correlation; dev is flat (p=0.81).
- **2b noise whitening + Euclid is the one that works**: 5.4 → 8.8% top-1, 21.1 → 31.6% top-5, median rank 17.5 → 11.6 (top-5 improves on every subject; median rank on three of four, sub-02 is 13.5 → 14.0). Per-subject CV agrees on sub-02..06 but not on sub-01 (CV prefers plain corr there, 0.278 vs 0.294), where dev top-1 fell (7.8 → 5.9) while top-5 and median rank improved.
- The correlation versions of 2b (CV 0.3017 per-time, 0.3060 pooled) are better than plain corr (0.3120) but worse than Euclid; they were not run on the dev test sets. I did not separate "whitening" from "Euclid" on dev.
- Submission for sub-05/06 of the best variant: `submissions/match_noise_pertime_euclid/` (gitignored, local only). Reproduce:
  `PYTHONPATH=src python scripts/predict.py --features clip_b32 --frames 8 --max-seconds 1.5 --alpha-selection retrieval --subjects 1 2 3 4 5 6 --match noise --noise-cov per_time --noise-similarity euclid --experiment match_noise_pertime_euclid`
- Caveats: alpha is still chosen by plain-correlation retrieval CV, but Euclid distance depends on prediction amplitude, which alpha shrinks, so a re-tuned alpha under Euclid may do better. The CV score uses the alpha chosen on the full training set (training data only, but not nested). The OOF-based matcher is fitted on predictions from 80%-data models and applied to predictions from the full-data model.
- Torch (CUDA, float64 transform / float32 ridge) is used for the OOF predictions and matcher statistics; it matches the earlier sklearn/numpy CV scores to ≤6e-6 and `LedoitWolf` to 2e-14.

## Encoder alpha under Euclid, and CV-weighted decoder + encoder ensemble (track-matching)
### (a) Alpha re-tuning under noise-whitened Euclid matching (`scripts/sweep_alpha_euclid.py` → `results/alpha_euclid_cv.csv`)
Nested train-CV (matcher re-estimated per fold, alpha fixed per run, half-decade grid 1..1e9), mean percentile rank:

| Subject | best alpha, corr | best alpha, Euclid | Euclid CV at the corr-chosen alpha | Euclid CV at its own best | gain |
|---|---|---|---|---|---|
| 01 | 1e4 | 3.2e4 | 0.2941 | 0.2936 | 0.0004 |
| 02 | 1e4 | 1e4 | 0.2973 | 0.2973 | 0 |
| 03 | 1e4 | 1e4 | 0.2800 | 0.2800 | 0 |
| 04 | 1e4 | 1e4 | 0.2984 | 0.2984 | 0 |
| 05 | 3.2e4 | 1e4 | 0.3015 | 0.2968 | 0.0047 |
| 06 | 3.2e4 | 3.2e4 | 0.3435 | 0.3435 | 0 |

**Null result.** The Euclid curve has the same interior optimum as the correlation curve (mean over subjects: 1e4), so Euclid's scale sensitivity does not
move the best alpha. The encoder alpha stays as chosen by plain-correlation retrieval CV.

### (b) Ensemble S = w·z(dec) + (1−w)·z(enc) (`scripts/ensemble.py`, CV in `results/ensemble_cv_decw.csv`, raw decoder in `results/ensemble_cv_dec.csv`)
z = per-row z-score over the candidates. Decoder member = the noise-whitened decoder (`eegmr.decoding.DecodingVariant`, whiteners and alpha as in
`predict.py --head decoding --dec-whiten`, re-estimated inside each outer fold); encoder member = CLIP B-32 8 frames, per-timepoint noise whitening + Euclid
(matcher re-estimated inside each outer fold). Same folds (KFold 5, seed 0) as every other train-CV; alphas fixed to the full-training choice.
Criteria per w: mean percentile rank, expected top-1 and top-5 of 102 (hypergeometric on fold ranks). Same-fold references of krabiki-e1 reproduced
(dec 0.207 / 0.230 / 0.272 / 0.244 / 0.265 / 0.315 raw, equal-weight 0.216 / 0.234 / 0.251 / 0.241 / 0.257 / 0.309, expected top-k).

Global CV (mean over sub-01..06), whitened decoder:

| w (weight of the decoder) | 0 (enc only) | 0.5 | 0.8 | 0.9 | 1.0 (dec only) |
|---|---|---|---|---|---|
| mean percentile rank | 0.3025 | 0.2487 | 0.2339 | 0.2322 | **0.2319** |
| expected top-1 of 102 | 4.5% | 6.2% | 6.7% | **6.8%** | 6.8% |
| expected top-5 of 102 | 15.9% | 22.0% | 24.0% | 24.1% | **24.2%** |
| (top-1 + top-5) / 2 | 10.2% | 14.1% | 15.4% | 15.5% | **15.5%** |

Picks (global): avg → **w = 1.0**, percentile rank → 1.0, top-1 → 0.9, top-5 → 1.0. Per subject (avg criterion): 0.8 / 1.0 / 1.0 / 1.0 / 0.7 / 1.0 for sub-01..06
(CV gain over the decoder alone: sub-01 +0.003, sub-05 +0.003, all others 0).
Leave-one-fold-out check of (E[top-1] + E[top-5]) / 2: dec only 0.1550 | enc only 0.1021 | equal 0.1411 | global w 0.1547 | per-subject w 0.1553.
With the raw decoder the picks were global w = 0.7 (per subject 0.8 / 0.7 / 0.5 / 0.8 / 0.6 / 1.0), LOFO dec 0.1337 | equal 0.1371 | global 0.1400 | per-subject 0.1391.

**Once the decoder input is noise-whitened, the encoder adds nothing.** The CV optimum is the decoder alone. The earlier ensemble gain (M7, raw decoder) is consistent with the
whitened encoder supplying the noise handling that the raw decoder lacked; the whitened decoder now has it itself.

Dev (sub-01..04 test sets), paired Wilcoxon vs the whitened decoder alone (`scripts/compare_matching.py`; "better/worse" counts the trials whose rank changed):

| Experiment (whitened decoder) | top-1 | top-5 | median rank | better / worse | p |
|---|---|---|---|---|---|
| dec alone (`ens_decw_encnoise_dec`) | 14.7% | 42.2% | 7.5 | – | – |
| encoder alone (noise Euclid) | 8.8% | 31.6% | 11.6 | 146 / 233 | 4e-6 |
| equal weight | 13.2% | 41.7% | 7.8 | 167 / 174 | 0.46 |
| CV global w = 1.0 | = dec alone | | | | |
| CV per-subject w (0.8 / 1 / 1 / 1) | 13.7% | 42.6% | 7.4 | 38 / 28 | 0.013 |

The per-subject blend differs from the decoder only on sub-01 (w = 0.8: top-1 17.6 → 13.7%, top-5 46.1 → 48.0%); the other dev subjects are unchanged. Not a clear win.
Secondary, raw decoder (14.0 / 36.0 / 8.25): equal weight 12.0 / 41.7 / 8.0 (p = 0.002); CV global w = 0.7: 13.0 / 40.7 / 8.0 (better 192 / worse 121, p = 7e-8);
per-subject: 13.7 / 40.7 / 8.5 (p = 4e-8).

Submissions: the CV-best rule (leave-one-fold-out, auto) chose the per-subject weights, so `submissions/ens_decw_encnoise_subject/sub-05,06` were written (sub-05 w = 0.7,
sub-06 w = 1.0, which is identical to the `decw` decoder submission). Format validated (102 × 102 int64, each row a permutation of the test IDs).
**Recommendation:** the CV gain over the decoder alone is within noise (LOFO +0.0003), so the plain whitened decoder (`submissions/decw_clip_b32_f8_t1.5s_w5-60/`)
remains the headline; the ensemble file is the CV-best alternative.
Reproduce: `PYTHONPATH=src python scripts/ensemble.py --features clip_b32 --frames 8 --max-seconds 1.5 --decoder whitened --experiment ens_decw_encnoise --cv-out results/ensemble_cv_decw.csv`

## N-member CV-weighted ensemble: CLIP + DINOv2 decoders (track-matching)
`scripts/ensemble.py` now takes any number of members, `--member HEAD:FEATURES[:FRAMES]` with HEAD in `dec | decw | enc | enc_noise`; weights on the simplex (step 0.1),
exhaustive for N ≤ 3, steepest-ascent pair transfers for N > 3. Same train-CV (KFold 5, seed 0; whiteners and matchers re-estimated per outer fold; alphas from the full
training set), same criterion (mean of E[top-1] and E[top-5] of 102), strategies compared by leave-one-fold-out (best single member / equal / global w / per-subject w,
simplest within 0.0005 of the best wins). Regression check: two members `decw:clip_b32` + `enc_noise:clip_b32` reproduce the previous result exactly
(LOFO 0.1550 / 0.1411 / 0.1547 / 0.1553, same weights).

Members (whitened decoders, per-subject train-CV mean percentile rank, sub-01..06; weights are the CV-chosen global ones):

| Member | CV sub-01..06 | CV (E1+E5)/2 | dev top-1 / top-5 / median |
|---|---|---|---|
| decw:clip_b32 | 0.192 / 0.206 / 0.251 / 0.219 / 0.247 / 0.277 | 0.1550 | 14.7 / 42.2 / 7.5 |
| decw:dinov2_b_patch_all | 0.180 / 0.204 / 0.251 / 0.203 / 0.237 / 0.267 | 0.1609 | 13.0 / 41.7 / 7.5 |
| decw:dinov2_b_cls_all | 0.222 / 0.238 / 0.281 / 0.248 / 0.278 / 0.310 | 0.1280 | 10.8 / 34.3 / 11.9 |
| enc_noise:clip_b32 | 0.294 / 0.297 / 0.280 / 0.298 / 0.302 / 0.344 | 0.1021 | 8.8 / 31.6 / 11.6 |

**Result: DINOv2-patch adds to CLIP.** The CV-chosen weights are 0.5 / 0.5 (CLIP / patch) in every configuration; the encoder and the CLS decoder get weight 0.
Leave-one-fold-out (E1+E5)/2: CLIP 0.155 → patch alone 0.161 → **CLIP + patch 0.172** (equal and global agree; per-subject 0.171, i.e. no gain from per-subject weights).

| Members | CV-chosen weights (global) | LOFO: best single / equal / global / per-subject | dev (sub-01..04) top-1 / top-5 / median | vs CLIP decw alone (better / worse, p) |
|---|---|---|---|---|
| CLIP + patch | 0.5 / 0.5 | 0.161 / 0.172 / 0.172 / 0.171 | **15.2 / 45.3 / 7.2** | 179 / 140, 0.0024 |
| + enc_noise | 0.5 / 0.5 / 0 | 0.161 / 0.164 / 0.172 / 0.169 | same as above | same |
| + CLS + enc_noise (coordinate ascent) | 0.5 / 0.5 / 0 / 0 | 0.161 / 0.168 / 0.172 / 0.169 | same as above | same |

Versus the best single member by CV (patch, dev 13.0 / 41.7 / 7.5): better 192 / worse 132, p = 0.0007. Versus CLIP alone, top-1 +0.5 pts, top-5 +3.1 pts, median rank 7.5 → 7.2.
Dev, per subject (CLIP + patch, top-1 / top-5 / median): sub-01 20.6 / 49.0 / 6.0, sub-02 13.7 / 45.1 / 7.0, sub-03 12.7 / 41.2 / 9.5, sub-04 13.7 / 46.1 / 6.5.
Per-subject weights (CLIP / patch): 0.5/0.5, 0.5/0.5, 0.4/0.6, 0.3/0.7, 0.5/0.5, 0.4/0.6; their dev result is 15.7 / 44.1 / 7.4 (p = 0.016 vs CLIP) and LOFO says it does not beat the global weights.

Diagnostic only, not a selection: the equal-weight 3-member blend (0.4 / 0.3 / 0.3 incl. the encoder) reaches 17.9 / 44.6 / 7.2 on dev (better 200 / worse 136 vs CLIP, p = 8e-6),
but the CV prefers the 2-member blend by a clear margin (LOFO 0.164 vs 0.172), so it is not used. The dev set is 4 subjects × 102 trials (±1.1 pts on top-1).

Submissions (identical for all three configurations): `submissions/ens_clip_patch_cv/` (sub-05/06; format validated: 102 × 102 int64, rows are permutations of the test IDs;
top-1 agrees with the CLIP decw submission on 65% / 57% of rows). Local only (gitignored).
Reproduce: `PYTHONPATH=src python scripts/ensemble.py --member decw:clip_b32 --member decw:dinov2_b_patch_all --max-seconds 1.5 --experiment ens_clip_patch`
(CV tables in `results/ens_clip_patch_cv.csv`, `results/ens_clip_patch_enc_cv.csv`, `results/ens_4way_cv.csv`).

## Contrastive members in the ensemble (track-matching)
`scripts/ensemble.py` gains a member type `con:FEATURES:CONFIG` (CONFIG in `warm | linear | two_tower_d256`, the `selected` grid of `scripts/train_contrastive.py`; M12 heads in
`eegmr.contrastive`). A `con` member uses the decw member's per-fold noise whiteners (re-estimated on each outer fold's training part), `prepare` + `train_head` with the decoder
alpha chosen for those features, and its similarity is the **mean of row-z similarities over 3 SGD seeds (0, 1, 2)**, in the CV folds and in the final fit. Seed-0 single-run CV
reproduces M12 exactly (sub-01 two_tower_d256: 0.2200 / 0.0846 / 0.2783). Selection rule unchanged: weights on the 0.1 simplex by (E[top-1] + E[top-5]) / 2, strategy by
leave-one-fold-out (simplest within 0.0005 of the best).

Members, CV (E1+E5)/2 over 6 subjects (3-seed averages for `con`), and dev top-1 / top-5 / median:

| Member | CV | dev |
|---|---|---|
| decw:clip_b32 | 0.1550 | 14.7 / 42.2 / 7.5 |
| decw:dinov2_b_patch_all | 0.1609 | 13.0 / 41.7 / 7.5 |
| con:clip_b32:two_tower_d256 | 0.1477 | 16.7 / 45.8 / 7.0 |
| con:dinov2_b_patch_all:linear | 0.1628 | 15.2 / 45.1 / 7.1 |

Ensembles (CV-chosen global weights; LOFO = leave-one-fold-out of (E1+E5)/2):

| Run | Members (order as listed) | CV-chosen weights | LOFO: best single / equal / global / per-subject | strategy | dev top-1 / top-5 / median | vs decw clip alone (better/worse, p) | vs M11 (better/worse, p) |
|---|---|---|---|---|---|---|---|
| M11 reference | decw clip + decw patch | 0.5 / 0.5 | 0.161 / 0.172 / 0.172 / 0.171 | equal | 15.2 / 45.3 / 7.2 | 179 / 140, 0.0024 | – |
| a | decw clip + con clip two-tower | 0.5 / 0.5 | 0.1550 / 0.1635 / 0.1632 / 0.1623 | equal | 17.9 / 46.3 / 7.0 | 197 / 121, 5e-6 | 185 / 140, 0.049 |
| **b** | decw clip + decw patch + con clip two-tower | **0.1 / 0.6 / 0.3** | 0.1609 / 0.1750 / **0.1807** / 0.1781 | global | **17.4 / 46.8 / 6.0** | 219 / 95, 1e-12 | 185 / 103, 9e-7 |
| c | b + con patch linear | 0.1 / 0.4 / 0.3 / 0.2 | 0.1628 / 0.1797 / 0.1812 / 0.1807 | global | 18.6 / 47.8 / 6.1 | 211 / 125, 2e-6 | 194 / 105, 2e-6 |

Run b per subject, dev top-1 / top-5 / median (CV-chosen global weights): sub-01 17.6 / 52.9 / 4.0, sub-02 18.6 / 45.1 / 6.0, sub-03 13.7 / 43.1 / 8.0, sub-04 19.6 / 46.1 / 6.0.
Per-subject weights give no gain by LOFO (b: 0.1781 vs 0.1807 global; c: 0.1807 vs 0.1812).

**Reading.** The contrastive members add to the decoders: LOFO rises from 0.172 (M11) to 0.181 (b), and the improvement is already there in run a (decw clip alone 0.155 → 0.1635).
b and c are a tie on CV (0.1807 vs 0.1812); applying the same "simplest within 0.0005" rule across runs, **b is the CV-best combo** (3 members, 0.1 / 0.6 / 0.3). Run c's
4-member blend is higher on dev (18.6 / 47.8) but that is not what selects it. Note the weights: DINOv2-patch gets most of the mass, the CLIP decoder only 0.1, and the
two-tower head 0.3. A single con head (3-seed) beats its decw counterpart on dev (CLIP 16.7 vs 14.7 top-1) but loses on CV (0.1477 vs 0.1550); the CV is the better guide (dev ±1.1 pts).
Submissions (format validated: 102 × 102 int64, rows permutations of the test IDs; local, gitignored): `submissions/ens_con_b_cv/` (CV-best), also written `ens_con_c_cv/` and
`ens_con_a_cv/`. Top-1 agreement with the M11 blend on sub-05 / 06: b 70% / 77%, c 67% / 74%, a 52% / 47%.
Reproduce b: `PYTHONPATH=src python scripts/ensemble.py --member decw:clip_b32 --member decw:dinov2_b_patch_all --member con:clip_b32:two_tower_d256 --max-seconds 1.5 --experiment ens_con_b`
(CV tables `results/ens_con_{a,b,c}_cv.csv`).

### Seed robustness of run b (con member averaged over seeds 3, 4, 5 instead of 0, 1, 2; `results/ens_con_b_s345_cv.csv`)
The CV-chosen global weights move by one 0.1 transfer, **0.1 / 0.6 / 0.3 → 0.0 / 0.6 / 0.4** (CLIP decoder → two-tower), and the optimum is a flat ridge: with seeds 3-5 the top
global candidates are 0.0/0.6/0.4 (0.18070), 0.1/0.5/0.4 (0.18052), 0.0/0.5/0.5 (0.18007), 0.2/0.5/0.3 (0.18000), and the old weights 0.1/0.6/0.3 score 0.17924 (0.0015 lower);
with seeds 0-2 the new weights score 0.18023 against 0.18073 for the old ones (0.0005 lower). Leave-one-fold-out (E1+E5)/2 with seeds 3-5: best single 0.16091 / equal 0.17529 /
global **0.18036** / per-subject 0.17926 (seeds 0-2: 0.1807 global). The con member's CV (0.1477 for seeds 0-2) is essentially unchanged by the seeds.

Dev (sub-01..04), top-1 / top-5 / median, with seeds 3-5: new weights **17.6 / 47.3 / 6.8** (vs decw clip alone 14.7 / 42.2 / 7.5: better 198 / worse 129, p = 2e-6; vs M11 15.2 / 45.3 / 7.2: 195 / 108, p = 1e-6),
old weights 0.1/0.6/0.3 with the new seeds 16.4 / 46.1 / 6.5 (vs the seeds 0-2 run b, 17.4 / 46.8 / 6.0: 127 / 123, p = 0.67; new vs seeds 0-2 run b: 137 / 125, p = 0.41).
So the weights are stable to within one step and the result does not depend on the seed set beyond the dev noise (about ±1 pt top-1); the old submission `ens_con_b_cv` stays valid.
(`ensemble.py` gained `--con-seed-start` and `--ref-weights`.)

## THINGS-EEG2 Step 0: zero-shot transfer from image-RSVP EEG (track-matching)
**Setup.** External data: THINGS-EEG2 preprocessed 17-channel release (Gifford, Dwivedi, Roig, Cichy; NeuroImage 2022; OSF 3jk45 component anp5v, CC BY 4.0, `sub-01.zip`, `sub-02.zip`,
uploaded 2022-08-12; images OSF y63gw, THINGS license: research use only), sha256 of every file verified against OSF. Files converted once in isolated mode (`python -I src/eegmr/external.py convert`,
whitelisting unpickler) to plain float32 `.npz`. The release's stored channel order is exactly our organizer-given order (`Pz P3 P7 O1 Oz O2 P4 P8 P1 P5 PO7 PO3 POz PO4 PO8 P6 P2`).
THINGS images embedded with the same `open_clip ViT-B-32 / laion2b_s34b_b79k` as our videos. Ridge (EEG 17 ch x 50-600 ms = 935 values, 40 Hz low-pass, z-scored) -> z-scored CLIP vector, fitted on THINGS sub-01+02 (132,320
single trials), alpha by the held-out THINGS images. Sanity check on the 200 held-out THINGS images (200-way): single-trial top-1 3.4% / top-5 13% (chance 0.5% / 2.5%); 80-repetition average 8% / 29%.

**Zero-shot on our EEG (no fitting on our data).** Pooled over all 6 subjects' training recordings (~5,400 trials; candidates = the subject's training videos; expected top-k of 102 by the hypergeometric rule), chance: percentile rank 0.5, top-1 0.98%, top-5 4.9%:

| Variant (our side) | pct rank | E[top-1] | E[top-5] | dev (sub-01..04) top-1 / top-5 / median |
|---|---|---|---|---|
| raw, no whitening | 0.451 | 1.6% | 7.4% | 0.7-1.0% / 5.6-6.1% / 41.5-42.1 |
| noise-whitened (our decw whiteners) | 0.431 | 2.1% | 8.4% | 1.2% / 7.8% / 37.3 |
| common-average reference, no whitening (**used for the member**) | **0.429** | 2.0% | 8.0% | 1.5% / 6.6% / 36.8 |
| our own per-subject decw (for scale) | 0.232 | 6.8% | 24.2% | 14.7% / 42.2% / 7.5 |

A weak but real shared signal (percentile rank 0.43 vs 0.5, about 17 standard errors on the pooled trials), but top-1 is only ~2x chance, and on the 408 dev trials top-1 is 0.7-2.2% (standard error 0.5 pts at chance),
i.e. within noise of chance under the stated kill criterion (< ~2.5%). Dev top-5 (6-9%) and median rank (36-42 vs 51.5 at chance) show the same weak signal.

**Use as an ensemble member** (`ensemble.py --member decw:clip_b32 --member zs:clip_b32`, nested-fold standardization, `results/ens_zs_cv.csv`): CV picks w = 0.8 / 0.2 (decw / zero-shot), LOFO (E1+E5)/2: decw alone 0.1550, equal 0.1219, global 0.1603, per-subject 0.1591.
Dev: **14.5 / 40.7 / 7.9** vs decw alone 14.7 / 42.2 / 7.5 (better 150 / worse 152, p = 0.84). The CV gain (+0.005) does not show on dev.

**Use as a shrinkage prior** (`things_step0.py prior`, ridge shrunk toward the THINGS solution, 5-fold training CV, `results/things_prior_cv.csv`; CAR input, no whitening, per-subject best alpha): mean CV percentile rank 0.2405 (no prior) -> 0.2393 (0.5 x W_THINGS) -> **0.2389** (1 x) -> 0.2411 (2 x);
E[top-1] 5.95% -> 6.19%. Better on all 6 subjects at scale 1 (by 0.0004-0.0033) but tiny; the whitened decoder alone is at 0.232.

**Verdict: NO-GO for neural pretraining on THINGS-EEG2 (Steps 1-2).** The linear transfer is real but too weak to matter (the shared signal is mostly early visual; the stimulus domain differs: 100 ms object images in a 5 Hz RSVP stream vs 3-s dynamic scenes with sound), and a stronger encoder would raise within-THINGS accuracy, not close the domain gap.
Possible leftover, not run: restrict both sides to the first ~250 ms; more THINGS subjects (the linear fit is already data-saturated at 132k trials for 935 weights).

## V-JEPA 2 as an ensemble member (track-matching)
Cached features `features/vjepa2_l_all_f16.npz` (V-JEPA 2 ViT-L, from track-vjepa); single `decw:vjepa2_l_all:16`: train-CV (E1+E5)/2 0.1518, dev 13.0 / 39.5 / 7.2. Same CV-weighted tool and rules as before (5 folds seed 0, nested whiteners and matchers,
con member = mean of row-z similarities over seeds 0-2, strategy by 5-decimal leave-one-fold-out).

| Run | Members (order as listed) | CV-chosen weights | LOFO: best single / equal / global / per-subject | strategy | dev top-1 / top-5 / median | vs M13 (better/worse, p) | vs M11 (better/worse, p) |
|---|---|---|---|---|---|---|---|
| M11 | decw clip + decw patch | 0.5 / 0.5 | 0.161 / 0.172 / 0.172 / 0.171 | equal | 15.2 / 45.3 / 7.2 | – | – |
| M13 (b) | decw clip + decw patch + con clip two-tower | 0.1 / 0.6 / 0.3 | 0.1609 / 0.1750 / 0.1807 / 0.1781 | global | 17.4 / 46.8 / 6.0 | – | 185 / 103, 9e-7 |
| a | decw clip + decw patch + decw V-JEPA | 0.4 / 0.3 / 0.3 | 0.16091 / **0.18512** / 0.18402 / 0.18385 | equal | 16.2 / 44.4 / 6.9 | 153 / 161, 0.34 | 178 / 126, 1e-4 |
| **b** | decw clip + decw patch + decw V-JEPA + con clip two-tower | **0.2 / 0.3 / 0.3 / 0.2** | 0.16091 / 0.18925 / **0.19243** / 0.19153 | global | **20.8 / 48.0 / 6.2** | 166 / 127, 0.028 | 207 / 99, 2e-10 |

Run b per subject, dev top-1 / top-5 / median: sub-01 23.5 / 54.9 / 5.0, sub-02 20.6 / 48.0 / 6.0, sub-03 15.7 / 42.2 / 8.0, sub-04 23.5 / 47.1 / 6.0. Vs run a: 184 / 81, p = 9e-10; vs decw clip alone (14.7 / 42.2 / 7.5): 224 / 111, p = 7e-13.
Per-subject weights (run b): dev 20.6 / 49.8 / 5.6, but LOFO 0.19153 < global, so the global weights stand. Single members in run b (CV E1+E5)/2: clip 0.1550, patch 0.1609, V-JEPA 0.1518, con two-tower 0.1477.

**Reading.** V-JEPA adds as a third *decoder* member: LOFO 0.172 (M11) -> 0.185 (a), and with the contrastive head 0.1807 (M13) -> 0.1924 (b); the weights spread over all four members (the CLIP decoder keeps 0.2). The CV
prefers b over M13 by 0.012 (LOFO), and dev agrees in direction (+3.4 pts top-1, p = 0.028 per trial vs M13), so b is the new CV-best combo. Run a alone is a tie with M13 on dev (p = 0.34).
Submissions (validated: 102 x 102 int64, rows are permutations of the test IDs; local): `submissions/ens_vj_b_cv/` (CV-best), also `ens_vj_a_cv/`.
Reproduce b: `PYTHONPATH=src:scripts python scripts/ensemble.py --member decw:clip_b32 --member decw:dinov2_b_patch_all --member decw:vjepa2_l_all:16 --member con:clip_b32:two_tower_d256 --max-seconds 1.5 --experiment ens_vj_b`
(CV tables `results/ens_vj_{a,b}_cv.csv`).

## STATUS (paused for relocation): CSLS / center-test ensemble variants on M17 run b (track-matching)
- `ensemble.py` now has `--csls K --csls-stage {blend,member}` (off by default), `--center-test` (off by default), `--sim-cache DIR`, and exhaustive weight search for up to 5 members (the earlier greedy search for 4 members could stop in a local optimum: with seeds 0-2, 0.0/0.4/0.3/0.3 scores 0.19330 vs 0.19269 for the reported 0.2/0.3/0.3/0.2).
- **Finished** (CSVs `results/ens_vj_b_{re,c5blend,c5member,c10blend,c10member,s345,ctr}_cv.csv` were made with the OLD greedy search; `ens_vj_b_ctr_c5member_cv.csv` with the new exhaustive search): cached rerun of run b (exact reproduction), CSLS k=5/10 x blend/member (LOFO of chosen strategy 0.2043 / 0.2074 / 0.2059 / 0.2069, so member-stage k=5 is CV-best; dev 23.0/53.4, 24.5/56.6, 23.5/55.6, 23.3/55.4), con seeds 3-5 (weights 0.0/0.4/0.3/0.3, LOFO 0.19498, dev 18.4/48.5/5.8), center-test (greedy: 0.0/0.4/0.3/0.3, LOFO 0.19406), center-test + CSLS5 member (exhaustive: 0.1/0.4/0.3/0.2, LOFO 0.20855).
- **Not done / cut off:** the 5-member runs with `decw:clip_b32_mid8_all:8` (`ens_vj5`, `ens_vj5_ctr_c5member`; CV done for the latter, test phase cut), the final consolidation with the exhaustive search for every variant, sub-05/06 files `ens_vj_b_csls_cv/`, MODEL_BACKLOG M17 addendum.
- **Resume:** the member-similarity cache is local and not committed (`results/similarity/_cache`, about 120 files). Re-running any command below reuses it. Final plan: `git checkout results/results.csv`, then for each variant run
  `PYTHONPATH=src:scripts OMP_NUM_THREADS=8 python scripts/ensemble.py --member decw:clip_b32 --member decw:dinov2_b_patch_all --member decw:vjepa2_l_all:16 --member con:clip_b32:two_tower_d256 --max-seconds 1.5 --sim-cache results/similarity/_cache --experiment <name> [--csls 5 --csls-stage member] [--center-test] [--con-seed-start 3]`
  and optionally add `--member decw:clip_b32_mid8_all:8`; copy the CV-best CSLS variant's `submissions/<name>_cv` to `submissions/ens_vj_b_csls_cv`.

## Next
- Re-test on the unwhitened pipeline: z-scoring (still expected to be a no-op), finer alpha grid around 1e4,
  PCA variance threshold or no PCA (kernel ridge), and the per-subject CV quality estimates.
  (CSLS: dropped for now.)
- Bigger levers: features (DINOv2, audio, more frames).
- Sliding 100 ms window scan to find where the signal is (also report figure §8.1); then tune the window inside 0–600 ms.
- Decoding direction (EEG → CLIP space, cosine); DINOv2 and audio features; pooled subjects.
