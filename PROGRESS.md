# Progress: Krabiki (all tracks)

Overview of all research as of 2026-10-18 (work paused for machine relocation). Numbers are dev = mean over the sub-01..04
test sets (4 × 102 recordings, 102 candidates). Chance is 1.0% top-1 / 4.9% top-5 / median rank 51.5; the standard error
is about ±1.1 pts top-1 and ±1.9 pts top-5. Every model choice is made by **train-CV** (5-fold on training videos,
KFold seed 0, all 6 subjects). Dev only confirms, with a paired Wilcoxon on per-trial ranks. Per-model details:
`MODEL_BACKLOG.md` (M0–M19). Orientation for agents: `CLAUDE.md`.

## 1. Current best

| Model | top-1 | top-5 | Status |
|---|---|---|---|
| **M17**: CV-weighted ensemble of whitened decoders on CLIP B-32 + DINOv2-B patch + V-JEPA 2-L (16 f) + contrastive two-tower head | **20.8%** | **48.0%** | Best without CSLS. Per subject top-1 23.5 / 20.6 / 15.7 / 23.5. sub-05/06 files in `../Krabiki-matching/submissions/ens_vj_b_cv/` (local, gitignored). |
| M17 + CSLS (k=5, per member, CV-chosen) | **24.5%** | **56.6%** | **Pending the organizers' OK**: CSLS is label-free but transductive (uses all 102 test rows jointly). Must stay off by default until approved. |
| M8a: single whitened CLIP decoder | 14.7% | 42.2% | Simplest strong model. Committed submission: `submissions/decw_clip_b32_f8_t1.5s_w5-60/`. |

Note (krabiki-51): M17's 4-member weights came from a greedy search that stopped in a local optimum. The exhaustive search
prefers 0.0 / 0.4 / 0.3 / 0.3 (CV +0.0006). `ensemble.py` now searches exhaustively for N ≤ 5, and the final runs must be redone.

## 2. How we got here (dev top-1 / top-5)

| Step | Model | top-1 | top-5 | Key idea |
|---|---|---|---|---|
| M0 | AlexNet encoding (organizers' recipe) | 2.0 | 8.6 | baseline |
| M1 | CLIP B-32 encoding, whitened features | 4.2 | 16.9 | better features |
| M2 | CLIP encoding, second scaler removed | 5.4 | 21.1 | **bug fix**: post-PCA StandardScaler made ZᵀZ = n·I, so ridge alpha was a no-op |
| M5 | + noise-whitened Euclidean matching | 8.8 | 31.6 | per-timepoint Ledoit-Wolf noise covariance from OOF residuals |
| M3 | **decoding** EEG → CLIP, cosine | 14.0 | 36.0 | flipped direction: compare in a semantic space, not noisy EEG space |
| M8a | + noise-whitened decoder input | 14.7 | 42.2 | M5's whitening applied to the decoder input |
| M11 | + DINOv2-B patch decoder (0.5 / 0.5) | 15.2 | 45.3 | a second, different feature space |
| M13 | + contrastive two-tower head | 17.4 | 46.8 | InfoNCE head is weak alone but complementary |
| M17 | + V-JEPA 2 decoder | 20.8 | 48.0 | video model; the most different view (RSA 0.48–0.50 vs CLIP / DINOv2) |
| M17 + CSLS | hubness correction | 24.5 | 56.6 | pending the rules question |

Pipeline: EEG 50–600 ms (17 ch × 55 samples at 100 Hz, 1.5 s crop, FIR decimation) → per-timepoint noise whitening →
standardize → ridge (alpha by retrieval CV) → predicted feature vector → cosine vs 102 candidates → row-z → CV-weighted
sum over members → (CSLS) → ranking.

## 3. Findings that held up
- **Decoding beats encoding** (M3 vs M2: p = 9e-17). The shuffled-label control gives chance.
- **Noise whitening** of the EEG helps both directions (M5, M8a). Every statistic comes from training data only.
- **Feature diversity drives the ensemble**: CLIP, DINOv2-patch and V-JEPA decoders all get real weight; the contrastive
  head adds on top (verified with a shuffled-pair control).
- **Information timing**: peaks ~150–250 ms, weak after ~600 ms. 50–600 ms is the best window for both directions
  (2f's 60-config sweep, decoder CV, cc's window tests).
- **Train/test shift (e1)**: test EEG std is 0.63–0.78× training on every subject (likely repeat-averaged), and the test-set
  mean evoked response is *anti*-correlated with the training mean (−0.68 to −0.94) and ~2× larger. Centering test EEG
  on its own mean (rule 5) is a small, consistent gain (M18). CSLS gains are large partly because of this.
- **CSLS on decoders** is a large label-free gain (M18, krabiki-cc; verified by e1): decw CLIP 14.7 / 42.2 → 17.4 / 48.5;
  M17 → 24.5 / 56.6 with CV-chosen settings. It was not significant on the old encoder.
- **Stage 3 (paused, krabiki-2f)**: EEGNet is weak alone (CV 0.293 vs ridge 0.232), but as a 0.25 member next to M8a it
  improves CV on 6 of 6 subjects (0.2240). Its gap to ridge shrinks with more data. Dev confirmation pending (see the end
  of this file).

## 4. Tried and closed (negative or null)
Frame count / onset pooling (M6: CLIP frames within a clip are near-identical); PCA threshold (M2 sweep); per-channel z-score
(already normalized); subtracting the TRAINING mean ERP (M5 #1; the train/test shift explains it); reliability-weighted
matching / cosine (M5 #2a, M8b: first CV/dev disagreement, an example of CV overfitting); alpha re-tuning under Euclid
(M9); the encoder as a member next to whitened decoders (M9, weight 0); DINOv2 CLS token (M10); contrastive heads alone
and ridge warm start (M12); CLIP ViT-L-14 (worse than B-32), time jitter, extra windows 150–350 / 600–1200 ms,
candidate centering, std rescaling (M18); THINGS-EEG2 zero-shot transfer and prior (M15: ~2× chance only, stimulus-domain
gap; pretraining Steps 1–2 dropped); fusion with Azamat's time-staged experts (M19: his 9.1 / 25.7 reproduced exactly,
weight 0 next to M17).

## 5. Data and rules decisions
- **Channel names** (from the organizers): Pz P3 P7 O1 Oz O2 P4 P8 P1 P5 PO7 PO3 POz PO4 PO8 P6 P2. Preprocessing:
  0.1–40 Hz, ICA blinks removed, 500 Hz. These are the same 17 channels as THINGS-EEG2's preprocessed release.
- **Excluded**: the EEG Moments Dataset (OpenNeuro ds008257, likely contains this challenge's EEG), BOLD Moments /
  Algonauts 2021 data and annotations, anything built on the 1,102 Moments-in-Time clips, and models trained on them
  (e.g. TRIBE v2). The user is informing the organizers.
- **External data survey** (`DATA_SURVEY.md`): THINGS-EEG2 tested (M15, no-go). Video-EEG candidates: only SEED-DV
  fits (gated application, weak signal); CineBrain is a poor fit. Recommendation: no further external EEG for now.
- **Open questions for the organizers**: is CSLS (transductive, label-free) allowed? Are gated datasets "public"?
  One-to-one assignment is not used.
- **User decisions**: subject pooling paused; audio deferred; captioning models held.

## 6. Where the code is
| Branch (worktree) | Contents |
|---|---|
| `track-proven` (`/home/kparpie/Krabiki`) | main package `src/eegmr`, decoders (`decoding.py`), contrastive (M12), GPU helpers, Stage 3 WIP (`a8ca716`), CLAUDE.md, audits |
| `track-matching` (`../Krabiki-matching`) | matching (M5), `scripts/ensemble.py` (N-member CV weighting, `con:` / `zs:` members, CSLS / center-test WIP `f1c0314`), THINGS-EEG2 Step 0 |
| `track-dinov2` (`../Krabiki-dinov2`) | DINOv2-B extractor, `--all-frames` CLI |
| `track-vjepa` (`../Krabiki-vjepa`) | V-JEPA 2 extractor (transformers 5.18.0 pinned), `scripts/zero_cost.py` (M18), CLIP ViT-L features |
| `track-viz` (`../Krabiki-viz`) | 8 presentation figures + `presentation/make_figures.py`, `FIGURES.md` |

All five branches are on GitHub (Gundan-Dante/Krabiki; commits use the GitHub noreply email). Remote `main` is Azamat's
separate track, left untouched. The two WIP commits are local.

## 7. Resume plan
1. **krabiki-51**: final ensemble runs with the exhaustive weight search: M17 members ± center-test ± CSLS (k = 5 / 10,
   per member vs after blending) ± the CLIP block-8 member; write both submission sets (with / without CSLS). Resume
   notes: track-matching PROGRESS.md "STATUS (paused for relocation)". The member similarity cache
   (`results/similarity/_cache`) is local.
2. **krabiki-2f**: Stage 3 dev confirmation (command at the end of this file), then EEGNet as an ensemble member via
   ensemble.py.
3. Ensemble robustness: regularized stacking and rank fusion vs grid weights (one-time check).
4. **Consolidation** (G1–G4 in IMPLEMENTATION_AUDIT.md): merge all branches into one; one command that reproduces the
   final submission from saved per-subject weights (minutes) with a documented full retrain; README; tests. Decide CSLS
   on / off from the organizers' answer.

---

# Detailed logs (track-proven)

> Sections below are the original track-proven log. Results before the "Whitening fix" section used the whitened
> transform (now `--whiten`), and their "alpha / crop / CSLS doesn't matter" conclusions are superseded where noted.

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

## Next
Superseded by §7 "Resume plan" at the top of this file.

## Stage 3 feasibility (krabiki-2f), PAUSED 2026-10-18 for machine relocation; to be filed as M14
- Train-CV (CLIP B-32, 6 subjects; ridge/M8a = 0.2319): MLP heads alone 0.243 ≈ linear 0.245; EEGNet alone 0.293; ShallowFBCSPNet ≈ chance (0.41–0.45: band-power features drop the evoked waveform); augmentation hurts slightly. **As a 0.25-weight ensemble member with M8a, EEGNet 0–800 ms gives 0.2240, better on 6/6 subjects** (MLP 0.2283, M12 two-tower 0.2290). A window control (linear/MLP on 0–800 ms: 0.2305/0.2291) shows the gain is from the conv architecture, not the window.
- Learning curves (50/75/100% of training trials): ridge 0.275/0.249/0.232, EEGNet 0.356/0.320/0.292 (gap to ridge 0.081 → 0.071 → 0.061, so data-limited), EEGNet ensemble gain vs ridge −0.001/+0.0045/+0.008 (grows with data). Results: `results/stage3_{cv_rung1,cv_rung2,cv_conv,cv_windowctrl,curve}_clip_b32.csv`.
- Not done: dev confirmation (3 seeds) of EEGNet / MLP as members, the DINOv2 repeat, the M14 write-up and verdict. Resume with:
  `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONPATH=src .venv/bin/python scripts/train_stage3.py --mode dev --configs eegnet_do0.25_lr0.001_wd0.01_w0-80 mlp512_do0.5_lr0.001_wd0.01_w5-60 --seeds 0 1 2`
