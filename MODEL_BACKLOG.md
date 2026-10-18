# Model & research backlog

A catalogue of every model and experiment tried so far, including retired ones, so they can be picked up again later
(e.g. as ensemble members, report baselines or ablations). `PROGRESS.md` tracks the *current* state. This file is the
long-term record. Maintained by session krabiki-e1. Other sessions should append, not rewrite.

**Metric convention:** dev = test sets of sub-01..04 (4 × 102 recordings), reported as top-1 / top-5 / median rank.
Chance is 1.0% / 4.9% / 51.5. The standard error is about ±1.1 pts on top-1 and ±1.9 pts on top-5. "Train-CV" = mean
percentile rank of the true video on held-out folds of the *training* videos (0 = perfect, 0.5 = chance, lower is
better). It uses no test labels and is the score used for model selection.

**Shared pipeline pieces** (used by every entry unless noted): EEG cropped to 0–1.5 s at 500 Hz, then FIR zero-phase
decimation to 100 Hz. Window 5:60 = 50–600 ms. Evaluation and submission writer in `eegmr.evaluate` /
`eegmr.submission`. GPU ridge/CV helper in `eegmr.gpu`.

---

## Status overview

| ID | Model | Dev top-1 / top-5 / median | Status |
|---|---|---|---|
| M0 | AlexNet encoding (organizers' recipe, re-implemented) | 2.0 / 8.6 / 35.1 | Retired. Report baseline |
| M1 | CLIP B-32 encoding, whitened features | 4.2 / 16.9 / 25.5 | Retired. Reproducible via `--whiten` |
| M2 | CLIP B-32 encoding, unwhitened | 5.4 / 21.1 / 17.5 | **Kept**: ensemble member, at its ceiling |
| M3 | **CLIP B-32 decoding (EEG → CLIP, cosine)** | **14.0 / 36.0 / 9** | **Current best** |
| M4 | Ensemble M2 + M3, equal weights | 10.5 / 37.0 / 9 | Needs CV-chosen weights |
| M5 | Encoding + noise-whitened Euclidean matching (#2b) | 8.8 / 31.6 / 11.6 | **Kept**: best encoder, ensemble member (verified independently) |
| M6 | Frame count / pooling (8 / 16 / 32 / all frames, onset-weighted, concat) | no significant change | Closed: frames are not a lever with CLIP |
| M7 | Ensemble M5 + M3, equal weights (diagnostic) | 12.0 / 41.7 / 8.0 | Promising. Needs CV-chosen weights |

---

## M0: AlexNet encoding (baseline recipe)
- **What:** 8 evenly spaced frames → torchvision AlexNet (IMAGENET1K_V1) `features.3` → mean over frames →
  StandardScaler → PCA 95% → StandardScaler → RidgeCV (GCV) → predicts the full EEG (17 × 300). Retrieval = Pearson
  correlation between measured and predicted EEG over 50–600 ms.
- **Result:** 2.0 / 8.6 / 35.1 (1 frame: 1.7 / 7.1 / 40.8).
- **Notes:** reimplemented from the brief, no organizer code. It includes the whitening flaw (see M1).
- **Revisit:** only as the "organizers' baseline" row in the report.

## M1: CLIP B-32 encoding, whitened
- **What:** M0 with open_clip ViT-B-32 / laion2b_s34b_b79k features (8 frames, mean) in place of AlexNet.
- **Result:** 4.2 / 16.9 / 25.5 with alpha by retrieval CV. GCV gives 4.4–5.1 / 15.9–16.9 / ~26.
- **Why retired:** the post-PCA StandardScaler makes ZᵀZ = n·I, so all singular values are equal and ridge alpha only
  rescales predictions (a no-op for correlation matching). Details in `FINDINGS_alpha_whitening.md`.
- **Experiments run on it (conclusions superseded where whitening mattered):** crop 1.5 s vs 3 s; matching windows
  50–600 ms, 50–1000 ms, 0–1.5 s and 0–3 s (50–600 ms best); GCV vs retrieval CV; per-channel z-scoring (no-op: the EEG
  is already normalized, std ≈ 0.47 on every channel); CSLS k = 5/10/20 (fewer hubs, not significant).
- **Reproduce:** `scripts/predict.py --features clip_b32 --frames 8 --max-seconds 1.5 --alpha-selection retrieval --whiten`

## M2: CLIP B-32 encoding, unwhitened
- **What:** M1 without the second StandardScaler (scaler → PCA 95% → ridge). Alpha = 1e4 for all subjects by retrieval CV.
- **Result:** 5.4 / 21.1 / 17.5. Paired vs M1: 252 better / 130 worse, p = 2e-13.
- **Variants:** GCV 5.4 / 20.8 / 17.6. CSLS k=10: 6.1 / 22.3 / 17.1 (p = 0.36 vs M2, not adopted).
- **Sweep (krabiki-2f):** 60 configs (PCA none/.99/.95/.9/.8 × windows), 73-alpha grid, train-CV 0.328–0.351. Best by CV:
  PCA none 5:60 (5.1 / 20.8 / 16.9). PCA ≥ .95 makes no difference, and 5:60 is the best window. Spearman(train-CV, dev
  median rank) = 0.68, which validates CV-based selection. **Encoding is at its ceiling.**
- **Retrieve-then-rerank check:** reranking the top-K with the same scores is a no-op, since pairwise correlations
  don't depend on the candidate set. Recall@K for M2: K=1 5.4%, 5 21.1%, 10 37.0%, 20 56.1%, 30 68.4%.
- **Revisit:** as an ensemble member with M3 (it makes different errors), and as the target of M5.
- **Reproduce:** `scripts/predict.py --features clip_b32 --frames 8 --max-seconds 1.5 --alpha-selection retrieval`
- **Submission:** `submissions/enc_clip_b32_f8_t1.5s_w5-60_retrieval/`

## M3: CLIP B-32 decoding (current best)
- **What:** reverse direction. Input = EEG window 5:60 flattened (17 × 55 = 935 values), standardized with training
  stats. Ridge (one SVD, many alphas) → standardized CLIP B-32 f8 feature (512-d). Retrieval = cosine between the
  predicted CLIP vector and the 102 test videos' CLIP vectors (standardized with training-video stats). Window
  ∈ {5:60, 0:80, 0:150} and alpha chosen by train-CV: 5:60 for every subject, alpha 56–562.
- **Result:** 14.0 / 36.0 / 9 (pooled median; the mean of per-subject medians is 8.2). Per subject, top-1: 17.6 / 12.7 /
  13.7 / 11.8%. Paired vs M2: 268 better / 123 worse, p = 9e-17. Train-CV 0.207–0.315 (sub-06 weakest).
- **Control:** shuffling the training EEG ↔ video pairing gives chance (1.0 / 4.4 / 51), so there is no leakage.
- **Why it beats encoding:** encoding compares in EEG space, where most of the 935 values are noise. Decoding learns
  which channels and timepoints carry information and compares in a semantic space.
- **Matching ideas transfer:** standardizing CLIP with training stats already is "#1 subtract the mean response".
  A decoding analogue of #2 would weight CLIP dimensions by how well they are decoded (per-dim train-CV R²).
- **Reproduce:** `scripts/predict.py --head decoding --features clip_b32 --frames 8 --max-seconds 1.5` (krabiki-2f,
  uncommitted as of 2026-10-18 02:30). Prototype: krabiki-e1 scratchpad `decoding_gpu.py`.
- **Submission:** `submissions/dec_clip_b32_f8_t1.5s_w5-60/` (sub-05/06 validated: 102 × 102, each row is a permutation of 1001..1102).

## M4: Ensemble, encoding + decoding
- **What:** z-score each row of the M2 and M3 similarity matrices, then average them.
- **Result (equal weights):** 10.5 / 37.0 / 9. Better top-5 than M3 alone, worse top-1. Paired vs M2: p = 1e-34.
- **Next:** choose the weight by train-CV (out-of-fold similarity matrices for both heads). Add more members as they appear
  (DINOv2 decoder, frame variants).

## M5: Encoding + improved matching (krabiki-51, branch `track-matching`)
- **What:** M2 with a different matcher. Out-of-fold (OOF) predictions of the training trials (5-fold, alpha fixed)
  give residuals. For each timepoint in 5:60, a Ledoit-Wolf 17 × 17 spatial noise covariance is estimated from those
  residuals. Measured and predicted EEG are whitened with Σ^-1/2 per timepoint and compared by negative squared
  Euclidean distance. All statistics come from training data only.
- **Result (2b, per-timepoint covariance, Euclidean):** **8.8 / 31.6 / 11.6** (mean of per-subject medians). Paired vs M2:
  246 better / 143 worse, p = 2.9e-7. Per subject: 5.9/32.4/11.5, 8.8/30.4/14.0, 13.7/33.3/9.0, 6.9/30.4/12.0. Reproduced
  exactly by krabiki-e1 with an independent sklearn/scipy implementation.
- **Variants that did not help:** #1 subtract the mean response: 4.9 / 20.6 / 16.8, p = 0.87 (a no-op; it cancels exactly
  under Euclidean matching). #2a reliability weights w = max(r, 0)^p: best p = 0.5 gives 4.7 / 21.8 / 16.6, p = 0.81, and
  train-CV gets worse as p grows.
- **Nested train-CV (mean sub-01..04):** corr 0.312 | weighted p0.5 0.315 | noise per-time corr 0.302 | **per-time Euclid
  0.292** | pooled corr 0.306 | pooled Euclid 0.297.
- **Caveats:** sub-01 top-1 drops (7.8 → 5.9) and CV prefers plain corr there. Alpha was chosen under plain correlation,
  and Euclidean matching is scale-sensitive, so re-tuning alpha under Euclid is open.
- **Reproduce:** `cd /home/kparpie/Krabiki-matching && PYTHONPATH=src python scripts/predict.py --features clip_b32
  --frames 8 --max-seconds 1.5 --alpha-selection retrieval --match noise --noise-cov per_time --noise-similarity euclid`
- **Code:** `src/eegmr/matching.py`, `scripts/select_matching.py`, `scripts/compare_matching.py` (uncommitted).
  Submission: `submissions/match_noise_pertime_euclid/` in that worktree.

## M6: Frames & pooling (krabiki-2f): closed, not a lever
- **What:** CLIP B-32 on *every* frame (videos have 45–92 frames, mixed fps, all 3 s; cache `features/clip_b32_all.npz`,
  5 min on GPU). Derived f8/f16/f32. Pooling: mean, onset-only (`first:0.5s`, `first:1.0s`), 4 segments, concat.
- **Decoding (train-CV, dev):** f8 0.2547, 13.5/36.5/8.6 | f16 0.2550, 13.7/36.5/9.4 | f32 0.2549, 14.0/36.3/9.6 |
  first0.5s 0.2531, 11.8/38.0/9.0 | first1.0s 0.2524 (CV best), 11.8/36.8/9.0 | segments4 0.2546, 13.7/36.8/9.8 (p = 0.04,
  worse) | f8 concat 0.2540, 12.5/37.0/8.6.
- **Encoding:** f8 5.6/21.6/17.4 | f32 5.9/22.5/17.4 | first0.5s (CV best) 5.9/22.8/17.4 | first1.0s 6.1/23.3/17.1.
- **Conclusion:** nothing is significant (all p ≥ 0.06). Within a clip, CLIP frames are nearly identical (first vs last
  frame median cosine 0.90, between clips 0.45), so each clip is essentially one scene. Temporal or motion information
  would need other features (motion energy, video models).

## M7: Ensemble, noise-matched encoder (M5) + decoder (M3), diagnostic
- **What:** equal-weight sum of the row-z-scored similarity matrices.
- **Result:** 12.0 / 41.7 / 8.0 vs M3 alone 14.0 / 36.0 / 8.25. Per-trial ranks: better 184 / worse 157 vs M3, p = 0.002.
  Top-5 +5.7 pts, top-1 −2 pts.
- **Next:** choose the weight by train-CV using OOF similarity matrices for both heads. Weights were *not* tuned on dev.

---

## M8: Decoding with noise-whitened input / reliability-weighted cosine (krabiki-2f): (a) adopted, (b) not
- **Code:** `src/eegmr/decoding.py` (`nested_cv_scores`, `DecodingVariant`), `scripts/select_decoding.py` →
  `results/decoding_variants.csv`. Canonical run: `predict.py --head decoding --dec-whiten` →
  `submissions/decw_clip_b32_f8_t1.5s_w5-60/`.
- **(a) Noise-whitened input:** each timepoint's channel vector is multiplied by Σ_t^-1/2 (Ledoit-Wolf on the encoding
  model's OOF residuals; encoding alpha from encoding retrieval CV) before the EEG → CLIP ridge. Decoder alpha is
  re-chosen by nested train-CV, with whiteners re-estimated inside each outer fold. The chosen alpha rises from 56–562
  to 1000–3162.
- **(b) Reliability-weighted cosine:** each CLIP dimension is scaled by max(r_d, 0)^p, where r_d = corr(OOF-decoded, true)
  on training trials (weights estimated inside each outer fold, at every alpha), p ∈ {0.5, 1, 2}.
- **Train-CV (mean over 6 subjects):** raw 0.2553 | raw p0.5 / p1 / p2: 0.2461 / 0.2441 / 0.2545 | **whitened 0.2319**
  (better on all 6 subjects) | whitened p0.5 / p1 / p2: 0.2225 / **0.2216** / 0.2323.
- **Dev (sub-01..04), paired Wilcoxon vs M3 (raw):**

| Variant | top-1 | top-5 | median rank | better / worse vs raw | p |
|---|---|---|---|---|---|
| raw (M3) | 14.0% | 36.0% | 8.25 | – | – |
| **whitened** | **14.7%** | **42.2%** | **7.5** | 187 / 141 | **0.002** |
| whitened p0.5 | 15.0% | 39.7% | 8.1 | 182 / 144 | 0.002 |
| whitened p1 (CV pick) | 14.0% | 39.7% | 8.25 | 175 / 173 | 0.20 |
| raw p0.5 / p1 / p2 | 12.7 / 14.2 / 13.2% | 39.2 / 39.2 / 35.5% | 9.4 / 9.0 / 10.6 | | 0.22 / 0.87 / 0.05 |

- **Decision:** adopt (a). Top-5 +6.2 pts, improving on every dev subject (42.2 → 46.1 / 42.2 / 38.2 / 42.2 per subject).
  Do not adopt (b): CV prefers p0.5–1 on top of whitening for all 6 subjects, but dev doesn't confirm it.
  Against whitened alone: p0.5 is 139 better / 145 worse (p = 0.71), and p1 is 144 / 179 (p = 0.09, trending worse).
  This is the first case where train-CV and dev disagree. A possible cause is that the weights fit the training
  videos' feature distribution, which differs from the 102 test videos.

---

## Ideas not yet implemented (backlog)

| Idea | Expected value | Notes |
|---|---|---|
| **DINOv2 features** (decoding target / ensemble member) | medium–high, complementary | Next feature step. See the overview in the chat (2026-10-18) |
| CLIP ViT-L-14 | medium | Bigger CLIP, fits on 8 GB GPU |
| CV-weighted ensemble of heads and feature spaces | high, cheap | Needs out-of-fold similarity matrices |
| Subject pooling (hierarchical: shrink each subject toward the group solution) | medium–high | Also enables a group-model fallback for unseen subjects. Test leave-one-subject-out |
| Decoding with per-dimension reliability weights | done → M8 (not adopted) | Decoding analogue of M5 #2 |
| Reduced-rank regression / CCA | medium | Shared low-dim subspace. Still linear and fast |
| Linear contrastive head (InfoNCE on frozen features) | done → M12 (ensemble member, not a replacement) | Optimizes ranking directly. Smallest step toward Stage 3 |
| **EEGNet / ATM-style contrastive EEG encoder (Stage 3)**: *perspective plan, after the feature upgrades* | high ceiling, costly, uncertain | Small CNN on raw EEG (17 × T), symmetric InfoNCE against frozen video features (CLIP / DINOv2), learnable temperature. **Data is the bottleneck:** ~970 trials per subject is little for a network, so it likely needs subject pooling (~5,500 trials, currently paused by the user) to beat ridge decoding. Plan when picked up: per-subject normalization plus a subject embedding, heavy regularization (dropout, ±20 ms shift, channel dropout, noise), early stopping on 10% held-out training videos. Compare against M3 at equal features. Possible ensemble member even if it doesn't win alone. |
| Kernel ridge (RBF) | low–medium | Cheap test of whether nonlinearity helps |
| CSLS on decoding similarities | low–medium | Was not significant on encoding |
| One-to-one assignment (Hungarian / Sinkhorn) | high on top-1, rule risk | Exploits the test structure. Ask the organizers. Flag off by default |
| Audio (CLAP) | medium, novelty | Deferred by the user |
| Object-detector outputs (e.g. RF-DETR class counts) | low–medium, interpretable | RF-DETR is a detector with a DINOv2 backbone, not a diffusion model. Use its outputs, not its features |
| Diffusion-model features (e.g. Stable Diffusion UNet) | low | Used for reconstruction in the literature, rarely better than CLIP for retrieval |
| Time-resolved decoding (sliding 100 ms windows), channel importance | report figures (§8) | Cheap with the GPU helper |

## M9: Encoder alpha under Euclid; CV-weighted ensemble of the whitened decoder (M8a) and the noise-Euclid encoder (M5) (krabiki-51, branch `track-matching`)
- **Code:** `scripts/sweep_alpha_euclid.py` → `results/alpha_euclid_cv.csv`; `scripts/ensemble.py` →
  `results/ensemble_cv_decw.csv` (whitened decoder, headline) and `results/ensemble_cv_dec.csv` (raw decoder, secondary);
  `matching.nested_fold_similarities`.
- **(a) Alpha re-tuning under Euclid: null.** Nested train-CV over a half-decade grid: Euclid's optimum is the same interior point as
  the correlation optimum (1e4 for sub-01..04, one grid step off for sub-05/06). Re-tuning gains ≤ 0.0047 CV (sub-05), 0 on four subjects.
  The encoder alpha stays as chosen by plain-correlation retrieval CV.
- **(b) Ensemble** S = w·z(dec) + (1−w)·z(enc), w on a 0…1 grid (step 0.1), chosen by train-CV with out-of-fold similarity matrices
  (same KFold seed 0; whiteners and matcher re-estimated per outer fold). Criteria: mean percentile rank, expected top-1 and top-5 of 102
  (hypergeometric), picked by their mean. Same-fold numbers of krabiki-e1 reproduced.
- **Whitened decoder: the CV picks w = 1.0 (decoder alone)** (global avg criterion 15.5% at w = 1.0 and 0.9 vs 14.1% at 0.5; percentile
  rank 0.2319 at w = 1 vs 0.2487 at w = 0.5). Per-subject w: 0.8 / 1 / 1 / 1 / 0.7 / 1. Leave-one-fold-out: dec only 0.1550, global 0.1547,
  per-subject 0.1553, equal 0.1411. Equal weight on dev: 13.2 / 41.7 / 7.8 vs decoder 14.7 / 42.2 / 7.5 (p = 0.46). Per-subject blend:
  13.7 / 42.6 / 7.4 (p = 0.013, changes only sub-01: top-1 17.6 → 13.7, top-5 46.1 → 48.0).
- **Raw decoder (secondary):** CV picks global w = 0.7; dev 13.0 / 40.7 / 8.0 vs 14.0 / 36.0 / 8.25 (better 192 / worse 121, p = 7e-8).
- **Verdict:** the ensemble gain (M7) disappears once the decoder input is noise-whitened (M8a). The encoder is no longer a useful ensemble
  member for CLIP B-32 features; keep the whitened decoder alone as the headline. Submission files exist for the CV-best blend
  (`submissions/ens_decw_encnoise_subject/`, sub-06 identical to `decw`, sub-05 w = 0.7) but the evidence for preferring them is within noise.
- **Next:** an ensemble may help again with a different feature space (DINOv2, CLIP L-14) as the second member; reuse `ensemble.py` with two decoders.

## M10: DINOv2-B decoders (krabiki-e1, branch `track-dinov2`, commit 3da7561)
- **Features:** timm `vit_base_patch14_dinov2.lvd142m` (DINOv2 ViT-B/14, 2023) at 224 px on every frame (24 min on GPU),
  mean over 8 frames. Two descriptors per frame: CLS token and mean of the 256 patch tokens (768-d each).
  Representational similarity with CLIP B-32 (Spearman over video pairs): CLS 0.57, patch 0.72.
- **Whitened decoder (M8a recipe), dev top-1 / top-5 / mean-of-medians, train-CV mean over 6 subjects:**
  CLIP B-32 14.7 / 42.2 / 7.5 (CV 0.232) | **DINOv2 patch 13.0 / 41.7 / 7.5 (CV 0.224; better than CLIP on 5 of 6
  subjects, tie on sub-03)** | DINOv2 CLS 10.8 / 34.3 / 11.9 (CV 0.263) | CLS + patch concat 12.0 / 39.5 / 8.4.
  Patch vs CLIP paired: 169 / 191, p = 0.86 (equivalent).
- **Diagnostic equal-weight ensemble (row-z), not tuned:** CLIP + patch **15.2 / 45.3 / 7.25**, vs CLIP alone better 179 /
  worse 140, p = 0.0024. CLIP + CLS 12.3 / 39.7 (no gain).
- **Status:** CLIP + DINOv2-patch is the lead candidate for the final pipeline. Weights are to be chosen by train-CV with
  krabiki-51's N-member ensemble tool.
- **Reproduce:** `cd /home/kparpie/Krabiki-dinov2 && PYTHONPATH=src python scripts/extract_features.py --model dinov2_b
  --all-frames --frames 8`, then `scripts/predict.py --head decoding --dec-whiten --features dinov2_b_patch_all --frames 8
  --max-seconds 1.5 --alpha-selection retrieval`.

## M11: N-member CV-weighted ensemble, CLIP + DINOv2 decoders (krabiki-51, branch `track-matching`)
- **Code:** `scripts/ensemble.py` (generalised to N members: `--member HEAD:FEATURES[:FRAMES]`, HEAD in dec|decw|enc|enc_noise; simplex weights step 0.1,
  exhaustive N ≤ 3, coordinate ascent N > 3; LOFO rule for best-single / equal / global / per-subject), `compare_matching.compare_table`.
  CV tables: `results/ens_clip_patch_cv.csv`, `ens_clip_patch_enc_cv.csv`, `ens_4way_cv.csv`.
- **Result:** DINOv2-patch **adds** to CLIP. CV-chosen weights 0.5 / 0.5 (CLIP decw / DINOv2-patch decw); the DINOv2-CLS decoder and the CLIP encoder get weight 0
  in the 3- and 4-member searches. LOFO (E1+E5)/2: CLIP 0.155, patch 0.161, CLIP + patch 0.172 (per-subject weights 0.171, no gain).
- **Dev:** 15.2 / 45.3 / 7.2 vs CLIP decw alone 14.7 / 42.2 / 7.5 (better 179 / worse 140, p = 0.0024); vs patch alone 13.0 / 41.7 / 7.5 (p = 0.0007).
- **Caveat:** equal-weight CLIP + patch + encoder gets 17.9 / 44.6 on dev (diagnostic) but the CV rejects it (LOFO 0.164 vs 0.172); not used.
- **Submissions:** `submissions/ens_clip_patch_cv/` (local, sub-05/06), the CV-best combo. Candidate headline pending krabiki-e1's decision.
- **Next:** more diverse members (CLIP L-14, other layers/frames, DINOv2 with different pooling) now have a cheap test: add `--member` and read the CV weights.

---

## M12: Linear contrastive heads (InfoNCE on frozen features) (krabiki-2f): no gain alone, a useful ensemble member
- **Code:** `src/eegmr/contrastive.py`, `scripts/train_contrastive.py` (`--mode cv|dev`, `--grid main|selected|strong_wd`,
  `--ensemble-weights`, `--ens-weight`, `--device`), `tests/test_contrastive.py`. Perspective and prior:
  `CONTRASTIVE_AUDIT.md`. Results: `results/contrastive_cv_*.csv`, `results/contrastive_dev_*.csv`.
- **Setup:** inputs identical to M8a (EEG 5:60, 1.5 s crop, per-timepoint noise-whitened, standardized, 935-d; frozen
  standardized video features). Heads: **A** linear W (935 → D) + cosine; **B** two towers (935 → d, D → d), d ∈ {64,
  128, 256}; **C** A warm-started at the M8a ridge solution, with decay toward ridge or toward 0. Loss is symmetric
  InfoNCE with learnable τ (init 0.07, clamped to [0.01, 1]), full-batch AdamW. Training is two-phase: early stopping on
  an inner 15% of training videos (step 0 counts as a candidate; the warm start's ridge is fitted on the 85% only),
  then retraining on all training videos for the chosen number of steps. Selection uses the shared outer 5-fold
  train-CV (seed 0) with whiteners re-estimated per fold; the ridge reference on the same folds equals M8a's CV exactly.
- **Train-CV, CLIP B-32 (mean of 6 subjects; ridge = M8a 0.2319):** C warm 0.2330–0.2349 (early stopping keeps the
  ridge solution: mean 0.3–4 steps) | A linear 0.2479 at best (lr 1e-3; worse on 6/6 subjects; stops after 16–65
  steps) | B two-tower 0.2716 / 0.2832 / 0.3091 for d = 256 / 128 / 64. Weight decay in {1e-4, 1e-3, 1e-2} has no
  effect (training stops before decoupled decay acts), and strong decay {0.1, 1, 10} doesn't help either (0.248–0.249).
- **Dev, single head vs M8a CLIP decw (14.7 / 42.2 / 7.5), 3 seeds:** C is identical to M8a (0 steps, bit-identical
  ranks). A gives 15.6 ± 1.5 / 45.0 ± 1.1 / 7.6, not significant (p = 0.35–0.87); CV said worse, so this is a second
  CV/dev disagreement, this time with CV the conservative side. B gives 14.3 ± 2.3 / 40.8 ± 1.5 / 8.2 (p 0.13–0.17).
- **As ensemble member** (w·rowz(M8a) + (1−w)·rowz(head), same features). **Train-CV picks w = 0.75** for both heads
  (CLIP: two-tower 0.2290, better on 5/6 subjects; linear 0.2306, 4/6; vs 0.2319). Dev at w = 0.75, 3 seeds:

| Ensemble (w_M8a = 0.75) | top-1 | top-5 | paired vs M8a (per seed) |
|---|---|---|---|
| CLIP decw + CLIP two-tower d256 | **15.9 ± 1.1** | **45.5 ± 0.5** | p = 1.6e-6 / 3.4e-7 / 2.5e-7 |
| CLIP decw + CLIP linear | 15.2 ± 0.6 | 43.4 ± 0.9 | p = 3.2e-4 / 1.9e-4 / 1.5e-4 |
| DINOv2 decw (13.0 / 41.7) + DINOv2 linear | 13.4 ± 0.7 | 43.6 ± 0.9 | p = 2.1e-3 / 1.2e-4 / 7.9e-3 |
| DINOv2 decw + DINOv2 two-tower d256 | 13.6 ± 0.9 | 43.0 ± 1.5 | p = 1.2e-2 / 1.3e-4 / 7.1e-4 |

  DINOv2 train-CV: ridge 0.2239, ensemble w = 0.75 linear 0.2228 / two-tower 0.2230 (both better on 5/6 subjects).
- **Verdict:** the contrastive head is **not a replacement** for ridge decoding at this data scale. Warm start converges
  to ridge, and from scratch it overfits within tens of steps. It is a **cheap ensemble member**: weak alone, but it
  errs differently from ridge. CLIP decw + CLIP two-tower reaches about the level of M11 (CLIP + DINOv2 decw, 15.2 / 45.3)
  using one feature space. The CV gain is small (about 0.003, versus 0.023 for noise whitening) while dev p-values are
  strong, so weights should come from 51's N-member CV tool before adoption. SGD variance is real (single-head top-1
  std up to 2.3 pts), so ensemble members need 3 seeds or seed-averaging.
- **Next:** add a `con:<features>:<config>` member type to krabiki-51's `ensemble.py` (OOF similarities per outer fold
  are already produced inside `contrastive.cv_scores`). Test CLIP-decw + DINOv2-decw + CLIP two-tower (+ DINOv2 linear)
  by CV, and consider averaging 3 seeds per contrastive member to cut SGD variance. Runtime about 10 s per subject per
  head on GPU (two-phase training).

## M13: Contrastive members in the CV-weighted ensemble (krabiki-51, branch `track-matching`)
- **Code:** `scripts/ensemble.py` member type `con:FEATURES:CONFIG` (`warm | linear | two_tower_d256`, `eegmr.contrastive` M12 heads on the decw input with the
  per-fold whiteners and the decoder alpha of those features); similarity = mean of row-z similarities over 3 seeds, in the CV folds and in the final fit.
  CV tables `results/ens_con_{a,b,c}_cv.csv`; seed-0 single-head CV reproduces M12 exactly.
- **Result:** the contrastive heads ADD to the decoders. CV-best combo **b = decw clip 0.1 + decw DINOv2-patch 0.6 + con clip two-tower-d256 0.3** (global weights, chosen by
  LOFO of (E1+E5)/2): LOFO 0.1807 vs 0.172 for M11 (CLIP + patch 0.5/0.5) and 0.155 for decw clip alone. Dev 17.4 / 46.8 / 6.0 (M11 15.2 / 45.3 / 7.2: better 185 / worse 103,
  p = 9e-7; decw clip alone 14.7 / 42.2 / 7.5: 219 / 95, p = 1e-12).
- **Other runs:** (a) decw clip + con two-tower: 0.5/0.5, LOFO 0.1635, dev 17.9 / 46.3 / 7.0. (c) b + con DINOv2-patch linear: weights 0.1/0.4/0.3/0.2, LOFO 0.1812, dev 18.6 / 47.8 / 6.1.
  b vs c is a CV tie (0.0005), so the simpler b is taken. Per-subject weights never beat global ones by LOFO.
- **Single heads:** con two-tower (3 seeds) CV 0.1477 vs decw 0.1550 but dev 16.7 / 45.8 vs 14.7 / 42.2 (CV is the guide, dev is ±1.1); con DINOv2-patch linear CV 0.1628 (best single member).
- **Submissions:** `submissions/ens_con_b_cv/` (CV-best), `ens_con_c_cv/`, `ens_con_a_cv/` in the track-matching worktree (sub-05/06, local, format validated).
- **Next:** more con members/seeds (variance), contrastive heads on other feature spaces, and a decision on 4 vs 3 members once more data (e.g. a third seed set) is available.
- **M13 seed robustness (krabiki-51, track-matching 7bbe9c5):** with the con member averaged over seeds {3,4,5}
  instead of {0,1,2}, the CV-chosen weights move by one grid step (0.1/0.6/0.3 → 0.0/0.6/0.4). The optimum is flat (top
  candidates within 0.0015). Dev with the new seeds and weights: 17.6 / 47.3 / 6.8. New vs old run: 137 / 125, p = 0.41,
  indistinguishable. Seed noise is about ±1 pt top-1. **Recommendation for the final pipeline:** average the con member
  over 6 seeds (0–5) and keep CV-chosen weights (expected near 0.05 / 0.6 / 0.35).

## M16: V-JEPA 2 ViT-L video features, whitened decoder (krabiki-viz session, branch `track-vjepa`, commit 4910887 + this entry)
- **Features:** `facebook/vjepa2-vitl-fpc64-256` (MIT, HF revision b3c1679b7c34d3255ef3547f27c7b226aefab26f, via transformers 5.18.0). A
  clip-level encoder: one forward pass per video on **16 frames** (read_frames segment-centre indices), fp16, the model's own preprocessing
  (bilinear resize to 292, centre crop 256, ImageNet mean/std; the crop drops about 12% of the border). Pooled feature = mean over all final-layer tokens
  (`get_vision_features`), 1024-d; a 4-segment temporal pooling is cached too (`frames`, unused so far). Pretraining data (paper Table 1):
  SSv2, Kinetics 400/600/700, HowTo100M, YT-Temporal-1B, ImageNet. Moments in Time is not a named source. fp16 vs fp32 cosine ≥ 0.99999.
  Extraction: 16 frames = 2.6 min, 64 frames fp16 = 17 min (on a shared GPU). **The 64-frame run was not done (on hold).**
- **Single whitened decoder (M8a recipe), dev top-1 / top-5 / mean-of-medians:** **13.0 / 39.5 / 7.25**; per-subject top-1 10.8 / 14.7 / 12.7 / 13.7,
  top-5 45.1 / 43.1 / 34.3 / 35.3. Train-CV per subject 0.2033 / 0.2070 / 0.2443 / 0.2159 / 0.2471 / 0.2647 (mean **0.2304**).
  Reference: CLIP decw 14.7 / 42.2 / 7.5, CV mean 0.2320 (0.192 / 0.206 / 0.251 / 0.219 / 0.247 / 0.277); DINOv2 patch 13.0 / 41.7 / 7.5,
  CV mean 0.2237 (0.180 / 0.204 / 0.251 / 0.203 / 0.237 / 0.267). V-JEPA vs CLIP paired: 167 better / 199 worse, p = 0.39 (equivalent). CV
  puts V-JEPA between CLIP and the DINOv2 patch features on average (vs CLIP: better on sub-03, 04, 06, tie on 05, worse on 01, 02; vs DINOv2 patch: better only on sub-03 and 06).
  Decoder alphas 1000-3162, as for the other features.
- **Equal-weight row-z blends, diagnostic only, weights not tuned, paired Wilcoxon on 408 per-trial ranks vs M11 (CLIP + patch, 15.2 / 45.3 / 7.25):**

| Blend | top-1 | top-5 | mean median | better / worse vs M11 | p |
|---|---|---|---|---|---|
| (a) CLIP + V-JEPA | 16.7 | 43.1 | 7.38 | 169 / 167 | 0.25 |
| patch + V-JEPA | 16.2 | 44.9 | 6.88 | 171 / 156 | 0.26 |
| **(b) CLIP + patch + V-JEPA** | 16.2 | 44.4 | **6.88** | **176 / 122** | **1e-4** |

  (b) improves the per-trial rank over M11 (median 7.25 → 6.88, p = 1e-4) while top-5 is 0.9 pt lower and top-1 1.0 pt higher, so it is
  the same ballpark on the headline metrics but better on the whole rank distribution. (a) is as good as M11 with CLIP as the only other member.
- **Verdict:** V-JEPA 2 is a CLIP-sized single decoder with different information (blending with either CLIP or DINOv2 patch matches M11, and
  all three together improve the ranks). Worth adding as a third feature member for the CV-weighted ensemble (krabiki-51's N-member tool). Whether 64 frames
  adds anything is untested; given that CLIP gained nothing from more frames, the expected gain is small.
- **Reproduce:** `python scripts/extract_features.py --model vjepa2_l --frames 16`, then `python scripts/predict.py --head decoding --dec-whiten
  --features vjepa2_l_all --frames 16 --max-seconds 1.5 --alpha-selection retrieval`. Similarity matrices: `/home/kparpie/Krabiki-vjepa/results/similarity/decw_vjepa2_l_all_f16_t1.5s_w5-60/`.
  Sub-05/06 submission files (local, gitignored): `submissions/decw_vjepa2_l_all_f16_t1.5s_w5-60/`.

## M15: THINGS-EEG2 Step 0, zero-shot transfer from image-RSVP EEG (krabiki-51, branch `track-matching`): weak signal, no-go
- **Data (rule 2 record):** THINGS-EEG2 preprocessed 17-channel release, Gifford, Dwivedi, Roig, Cichy, NeuroImage 2022; OSF project 3jk45, component anp5v (`sub-01.zip`, `sub-02.zip`, uploaded 2022-08-12, CC BY 4.0) and
  component y63gw (`training_images.zip`, `test_images.zip`, `image_metadata.npy`; THINGS images: research use only). sha256 verified against the OSF API. Stored in `/home/kparpie/ext_data/things_eeg2/` (outside all repos);
  pickled dicts loaded once in isolated mode with a whitelisting unpickler and re-saved as plain float32 npz. Channel order of the release = the organizers' order for our data.
- **Code:** `src/eegmr/external.py` (safe conversion, alignment to our window), `scripts/things_step0.py` (`embed`, `fit`, `ours`, `prior`), ensemble member type `zs` in `scripts/ensemble.py`.
- **Result:** ridge THINGS EEG (17 ch, 50-600 ms) -> CLIP B-32 (same open_clip laion2b model as our videos). Within THINGS (200-way) single-trial top-1 3.4%. Zero-shot on our EEG (no fitting): pooled train-EEG percentile rank 0.43 (chance 0.5), E[top-1] ~2%,
  E[top-5] ~8%; dev top-1 0.7-2.2% (noise of chance), top-5 6-9%, median rank 36-42. As an ensemble member (decw + zs): CV picks 0.8 / 0.2, LOFO 0.1603 vs 0.1550, dev 14.5 / 40.7 / 7.9 vs 14.7 / 42.2 / 7.5 (p = 0.84). As a ridge shrinkage prior: CV percentile rank 0.2405 -> 0.2389, better on 6/6 subjects but negligible.
- **Verdict:** a weak but real shared early-visual signal; not worth neural pretraining (domain gap: static objects at 100 ms/5 Hz RSVP vs 3-s dynamic scenes with sound). Steps 1-2 of the plan are dropped.

## M17: V-JEPA 2 in the CV-weighted ensemble (krabiki-51, branch `track-matching`): new CV-best combo
- **Code/data:** `scripts/ensemble.py` (no change), features `features/vjepa2_l_all_f16.npz` from M16 (`decw:vjepa2_l_all:16`); CV tables `results/ens_vj_{a,b}_cv.csv`.
- **Result:** V-JEPA adds as a third decoder member. (a) decw clip + patch + V-JEPA: CV weights 0.4 / 0.3 / 0.3, LOFO 0.18512 (equal), dev 16.2 / 44.4 / 6.9 (vs M11 p = 1e-4, vs M13 p = 0.34).
  **(b) + con clip two-tower (3 seeds): CV-chosen global weights 0.2 / 0.3 / 0.3 / 0.2 (clip / patch / V-JEPA / con), LOFO 0.19243 (M13: 0.1807; M11: 0.172), dev 20.8 / 48.0 / 6.2.** *(Correction 2026-10-18: these four-member weights came from a greedy coordinate ascent that stopped in a local optimum; the exhaustive lattice search now used for up to 6 members gives 0.0 / 0.3 / 0.4 / 0.3, LOFO 0.19307, dev 20.1 / 50.2 / 5.5. See M17-final below.)*
- **Dev, run b:** 20.8% top-1 / 48.0% top-5 / median 6.2 (M13 17.4 / 46.8 / 6.0: better 166 / worse 127, p = 0.028; M11 15.2 / 45.3 / 7.2: 207 / 99, p = 2e-10; run a: 184 / 81, p = 9e-10; decw clip alone: 224 / 111, p = 7e-13).
  Per-subject dev top-1: 23.5 / 20.6 / 15.7 / 23.5. Per-subject weights do not beat the global ones by LOFO (0.19153 vs 0.19243).
- **Submissions:** `submissions/ens_vj_b_cv/` (CV-best, sub-05/06, validated, local), `ens_vj_a_cv/`.
- **Next:** more diverse decoder members (other V-JEPA layers/pooling, a second con head on V-JEPA/patch), or stop adding members: the LOFO gains are +0.012 (V-JEPA added to M13) and +0.007 (con added to run a).

## M19: Fusion with Azamat's time-staged experts (krabiki-e1, diagnostic)
- **What:** ran the teammate's `track-timestaged` pipeline (remote `main` b1a5343) unchanged except for the GPU device
  (throwaway worktree `../Krabiki-azamat`). Experts: hand-made low-level features, DINOv2-S blocks 4/8/12, MViT-v2-S
  (Kinetics-400), as encoding + correlation in early / middle / late EEG windows. Reproduced his honest LOSO result exactly:
  9.1 / 25.7 / 15.2.
- **Fusion with M17** (imported via his `08_import_scores.py`, which reproduced our numbers): his combiner gives all his
  experts weight 0 next to M17. A finer blend z(M17) + w·z(his ensemble), w ∈ {0, 0.1, 0.2, 0.3, 0.5}, with w chosen by
  LOSO over dev subjects: 20.3 / 47.8 vs M17 20.8 / 48.0, 120 better / 110 worse, p = 0.39. **No gain. Not adopted.**
- **Takeaway:** his features are diverse, but encoding + correlation scores are too weak next to the whitened decoders.
  If revisited, try his feature sets (MViT block 14, DINOv2 block 8, low-level) as **decw decoder targets** instead.

## M18: Zero-complexity gains on the whitened decoder: test-time centering, CSLS, windows, jitter, bigger CLIP (krabiki-cc, branch `track-vjepa`)
- **Code:** `scripts/zero_cost.py` (`run` / `report` / `blend`), `scripts/enc_center.py`, GPU encoders `clip_l14` and `clip_b32_mid8` in `features.py` (+ `extract_features.py --subset-only`). Full tables:
  `results/zero_cost_summary.txt`. The baseline reproduces the saved decw similarity matrix exactly and the CV 0.2319 (CLIP). CV = mean percentile rank over the 6 subjects (KFold 5, seed 0, whiteners re-estimated per
  outer fold, alpha chosen by CV; E[top-1] / E[top-5] are hypergeometric for a 102-pool); dev = sub-01..04 (14.7 / 42.2 / 7.5 for CLIP), paired Wilcoxon on 408 per-trial ranks vs the plain decoder of the same features.
  Never tuned on dev: variants are chosen by CV, dev confirms. Raw per-fold similarity matrices (3.9 GB) are in `results/zero_cost/` (gitignored, regenerable).
- **Results per idea (CLIP B-32 decw; deltas vs plain decw; dCV < 0 = better):**

| Idea | dCV | dev top-1 / top-5 / med | better / worse, p | Re-check DINOv2-patch / V-JEPA | Verdict |
|---|---|---|---|---|---|
| **5. CSLS k=5 / 10 / 20** (fold-internal in CV, at the alpha chosen without CSLS) | -0.0058 / -0.0059 / -0.0060 (E[top-1] +0.6, E[top-5] +1.1 to 1.3 pts) | **17.4 / 49.3 / 6.0** (k=5), 17.4 / 48.5 / 6.4, 17.2 / 48.0 / 6.25 | 205 / 92, **2e-12**; k=10 6e-13; k=20 1e-13 | patch: 16.7-17.4 / 46-47, p ~1e-13 (dCV -0.0070); V-JEPA: 15.4-16.2 / 46.8, p ~1e-15 (dCV -0.0066) | **ADOPT** (k = 10; k is insensitive). Transductive over the 102 test items, label-free: needs a rule-5 ruling |
| **1a. Test EEG centered** (own mean over the recordings replaced by the training mean) | -0.0002 | 15.0 / 41.9 / 7.75 | 149 / 103, 4e-4 | patch 13.0 / 42.4 / 7.12 (p 0.004, dCV -0.0006); V-JEPA 13.2 / 42.2 / 7.5 (p 0.008, dCV -0.0008) | **ADOPT** (small, consistent, free; CV understates it since folds don't carry the train/test shift) |
| 1c. Candidate features centered on their own mean | 0.0000 | 14.7 / 42.4 / 7.88 | 39 / 54, 0.08 | (not re-run) | drop (neutral); (a)+(c) = 15.2 / 41.9 / 7.75, p 4e-4, no better than (a) |
| 1b. Own mean AND std (per feature) of the test inputs; also candidates | 0.0000 | eeg 14.5 / 43.9 / 7.75 (p 0.01); eeg + cand 14.0 / 43.4 / 7.5 (p 7e-4) | 155 / 121; 155 / 117 | patch 12.5 / 42.4 (p 0.04); V-JEPA 12.5 / 41.4 (p 0.5) | drop (inconsistent) |
| 1. Centering for the ENCODER (noise-Euclid and corr matching) | – | noise-Euclid 8.8 / 31.6 / 11.6 -> 8.6 / 33.6 / 11.9; corr 5.4 / 21.1 -> 5.4 / 19.9 | p 0.25-0.55 | – | drop |
| 4. Time-jitter augmentation (training copies shifted by ±1, or ±1 and ±2 samples) | -0.0029 (±1), -0.0017 (±2) | 14.2 / 41.4 / 7.88 (±1); 13.7 / 40.9 / 8.0 | 125 / 119, 0.43; 136 / 138, 0.90 | patch dCV -0.0026, dev p 0.29; V-JEPA dCV -0.0028, p 0.85 | drop (CV gain of ~0.003 on all three, dev doesn't confirm) |
| 2. Extra windows 150-350 ms / 600-1200 ms | single decoders far worse: CV 0.2675 / 0.3950 (vs 0.2319) | alone 10.0 / 32.1 / 11.6 and 2.9 / 14.0 / 28.3; equal-weight blends with 5:60: 12.3 / 39.2 (+150-350), 9.1 / 34.8 (+600-1200), 11.3 / 36.0 (both) | blends worse, p 0.003 / 2e-10 / 5e-4 | – | **drop** (the useful signal is in 50-600 ms) |
| 3a. CLIP ViT-L-14 `laion2b_s32b_b82k` (HF laion/CLIP-ViT-L-14-laion2B-s32B-b82K rev 1627032197142fbe2a7cfec626f4ced3ae60d07a, 2022-09-14), 8 frames, fp16 | +0.0094 | 14.5 / 35.8 / 9.6 | 143 / 191, 0.003 (worse than B-32) | – | drop; as a 4th member in the 3-blend: CV +0.0032, dev p 0.14 (no gain) |
| 3b. CLIP B-32 block-8 CLS (residual stream after 8 of 12 blocks) | +0.0415 alone | 12.3 / 31.4 / 12.5 | worse, 1e-8 | – | alone drop. **4th member of the 3-blend: CV better** (see below): ensemble-member candidate |

- **Blends (equal-weight row-z, diagnostic, members' own CV alphas; reference M11 = CLIP + DINOv2-patch 15.2 / 45.3 / 7.25, CV 0.2155):**

| Blend | CV pct | dev top-1 / top-5 / med | vs M11 (better / worse, p) |
|---|---|---|---|
| CLIP + patch + V-JEPA (3-blend) | 0.2018 | 16.2 / 44.4 / 6.88 | 176 / 122, 1e-4 |
| 3-blend + CSLS 10 | 0.1950 | **21.3 / 52.7 / 5.0** | 236 / 85, 2e-16 |
| 3-blend, test EEG centered, + CSLS 10 | 0.1944 | **23.0 / 53.9 / 4.88** | 235 / 83, 6e-18 |
| CLIP + patch (M11) + CSLS 10 | 0.2099 | 22.5 / 47.5 / 6.0 | 218 / 71, 9e-19 |
| 3-blend + B-32 block-8 (4 members), centered, + CSLS 10 | 0.1901 | 23.3 / 54.7 / 4.75 | 233 / 101, 2e-17 |
| 3-blend + L-14 (4 members) + CSLS 10 | 0.1982 | 23.3 / 52.5 / 5.25 | 234 / 83, 2e-17 (CV worse than 3-blend) |

- **Hub check (sub-01..06, 3-blend, plain vs centered + CSLS 10):** the number of distinct top-1 candidates rises from 39-48 to 53-60 of 102 and the most-picked candidate drops from 6-12 to 5 picks, on all six subjects including sub-05/06; top-1
  picks agree between the two for 65-78% of recordings.
- **Caveats:** (1) CSLS and test centering use only unlabeled test items but are transductive across the 102 recordings/candidates. CSLS also exploits the (true) one-video-per-recording structure of the test design, as the CV folds
  do, though their pools are ~200 and not 102. The rule-5 reading is krabiki-e1's / the organizers' call. (2) CSLS was applied at the alpha chosen without it. (3) The 4-member CV gains need krabiki-51's CV-weighted tool before they count.
  (4) Dev top-5 of the centered variants is flat; the gain is in the per-trial rank distribution and top-1.
- **Recommendation:** adopt CSLS k=10 and test-EEG centering (a) in the final pipeline for every decw member and for the blend; add V-JEPA (M16) as third member; test the B-32 block-8 member in the CV-weighted ensemble;
  drop extra windows, L-14, jitter, per-feature rescaling and encoder centering.
- **Reproduce:** `python scripts/zero_cost.py run --features clip_b32 --frames 8 --specs base eeg_center ... && python scripts/zero_cost.py report --specs base eeg_center base@csls10 eeg_center@csls10`;
  `blend --ref clip_b32:8:base dinov2_b_patch_all:8:base --members clip_b32:8:eeg_center dinov2_b_patch_all:8:eeg_center vjepa2_l_all:16:eeg_center --csls 10`. Features: `extract_features.py --model clip_l14 --subset-only --frames 8` (and `clip_b32_mid8`).

---

## M14: Stage 3 feasibility, nonlinear EEG encoders with InfoNCE, per subject (krabiki-2f): viable as an ensemble member only, data-limited
- **Code:** `src/eegmr/stage3.py` (linear / MLP / residual MLP / braindecode EEGNet / ShallowFBCSPNet encoders, symmetric
  InfoNCE with learnable τ, optional augmentation, two-phase training as in M12, train-CV with per-fold whiteners and a
  `train_fraction` option), `scripts/train_stage3.py` (`--mode cv|dev|curve`), `tests/test_stage3.py`, and an **`s3:` member
  type in `scripts/ensemble.py`** (config's own window, per-fold whiteners, inner-split early stopping, 3-seed row-z mean,
  honors `--center-test`). Results: `results/stage3_{cv_rung1,cv_rung2,cv_conv,cv_windowctrl,curve,dev}_clip_b32.csv`,
  `results/ens_vj_b_s3_cv.csv`. Inputs are those of M8a/M12 (whitened, standardized EEG; frozen standardized CLIP B-32 f8).
- **Ladder (train-CV, mean of 6 subjects; ridge = M8a 0.2319; ens = 0.75·rowz(M8a) + 0.25·rowz(model)):**

| Rung | Model | alone | as 0.25 member | n subjects better than ridge (member) |
|---|---|---|---|---|
| 0 | linear InfoNCE (reference) | 0.2449 | 0.2302 | 5 |
| 1 | MLP 935→512→D, GELU, dropout 0.5 | 0.2429 | 0.2283 | 6 |
| 1 | residual MLP | 0.2434 | 0.2295 | 5 |
| 2 | **EEGNet** (18k params, 0–800 ms) | 0.2926 | **0.2240** | **6** |
| 2 | EEGNet 0–600 ms | 0.3064 | 0.2272 | 6 |
| 2 | ShallowFBCSPNet | 0.4532 (≈ chance) | 0.2356 | 0 |
| 3 | EEGNet + augmentation (±20 ms shift, channel dropout 0.1, noise 0.1) | 0.2981 | 0.2249 | 6 |
| ctrl | linear / MLP on the same 0–800 ms window | 0.2611 / 0.2555 | 0.2305 / 0.2291 | 4 / 6 |

  Nonlinearity on the M12 input adds nothing (MLP ≈ linear). Shallow's square-log-pool power features discard the evoked
  waveform. Augmentation slightly hurts. The window control shows that EEGNet's member value comes from the conv
  architecture, not the longer window. Rung 4 (ATM-style) was skipped because rung 2 is far from ridge alone.
- **Dev, 3 seeds, vs M8a CLIP decw (14.7 / 42.2):** EEGNet alone 10.0 ± 0.4 / 30.2 ± 1.6 (worse, p ≤ 8e-7). **0.75 M8a +
  0.25 EEGNet: 15.5 ± 0.2 / 45.2 ± 1.4** (p = 3e-8 / 1e-5 / 5e-4). With the M12 two-tower (0.6 / 0.2 / 0.2): 16.4 ± 0.9 /
  45.8 ± 0.7. MLP512 alone 15.7 ± 1.1 / 44.6 ± 0.9 (p = 0.01 / 0.09 / 0.12; CV said worse, the same CV/dev split as M12's
  linear head).
- **In the full CV-weighted ensemble (M17b members + s3 EEGNet, exhaustive search, no CSLS / centering):** LOFO
  per-subject 0.19578, global 0.1936, equal 0.19319 (M17b: global 0.19243). EEGNet gets weight 0.1–0.4 per subject (global
  0.2; highest on sub-06, the weakest subject, where it has the best E[top-1] of all members). CV-chosen (per-subject)
  dev **18.1 / 50.2 / 5.4** vs M17b 20.8 / 48.0 / 6.2: better on 143 trials / worse on 96, **p = 3e-5**. Top-5 +2.2, top-1
  −2.7 (11 trials). Global weights 19.1 / 47.1 (p = 0.16); equal 19.1 / 50.5 (p = 0.015). Sub-05/06 files:
  `submissions/ens_vj_b_s3_cv/` (local, validated).
- **Learning curves (train-CV on 50 / 75 / 100% of each outer training part):** ridge 0.275 / 0.249 / 0.232; EEGNet alone
  0.356 / 0.320 / 0.292 (gap to ridge 0.081 → 0.071 → 0.061, about −0.02 per doubling of data); EEGNet member gain over ridge
  −0.001 / +0.0045 / +0.008 (grows with data); linear / MLP gap to ridge 0.022 → 0.015 → 0.013 / 0.020 → 0.016 → 0.011.
- **Verdict:** **viable as an ensemble member, not as a standalone model at this data scale.** A tiny EEGNet adds
  complementary information (temporal filtering of the evoked response) that the linear decoders miss, and the
  effect is consistent across subjects, seeds and the CV/dev split. Standalone it is far behind ridge and **clearly
  data-limited**. A naive extrapolation of the gap (−0.02 per doubling) needs roughly 8× the per-subject trials to reach
  ridge, which only subject pooling (about 5.6k trials) or external pretraining could approach. **Pooling is the
  missing ingredient**, and it's paused by the user. Not done: the DINOv2 repeat, and `s3` combined with `--center-test`
  in the final grid (krabiki-51 will rerun the selection with s3 in the member set).


## M17-final: exhaustive weight search, CSLS, test-time centering, block-8 and s3 members (krabiki-51, branch `track-matching`)
- **CSLS is allowed by the organizers (2026-10-18).** `ensemble.py` flags: `--csls K --csls-stage {blend,member}`, `--center-test` (replace the scored set's own mean EEG by the training mean before whitening; label-free test normalization), `--sim-cache DIR`,
  exhaustive simplex search for up to 6 members (3003 points). Summary of the whole grid: `results/ens_final_summary.md`; CV tables `results/fin*_cv.csv`; logs/commands in PROGRESS.md.
- **Rule for choosing:** LOFO (leave-one-fold-out of (E1+E5)/2, strategy picked inside) decides; a member (block-8 CLIP CLS `decw:clip_b32_mid8_all:8`, s3 EEGNet) is kept only if it adds more than ~0.002 LOFO like for like (same weight strategy) and dev shows no top-1/top-5 trade.
- **Per-member CSLS beats CSLS after the blend** in every comparison (by 0.003-0.004 LOFO); k = 5 and k = 10 are tied (<= 0.0005). Test-time centering: LOFO +0.0008 with CSLS (members 5), tie without.
- **Final sets (sub-05/06 validated, local `submissions/`):**
  - **`final_csls/` (headline):** decw CLIP, decw DINOv2-patch, decw V-JEPA, con CLIP two-tower (3 seeds), decw CLIP block-8; weights 0.0 / 0.4 / 0.3 / 0.2 / 0.1 (global), test-time centering, CSLS k=5 per member. LOFO 0.21235, dev **24.0 / 57.1 / 4.2** (M17 reference 20.8 / 48.0 / 6.2: better 209 / worse 82, p = 3e-13).
  - `final_csls_s3/` (alternative): + s3 EEGNet, per-subject weights; LOFO 0.21445 (global-weights 0.21275, i.e. +0.0004 like for like), dev 24.5 / 57.8 / 4.2; vs the headline 122 / 100, p = 0.24. Not taken (gain only through per-subject weights, not significant, extra model).
  - **`final_nocsls/` (backup):** + s3, no centering, global weights 0.0 / 0.3 / 0.2 / 0.1 / 0.2 / 0.2 (clip / patch / V-JEPA / con / block-8 / s3); LOFO 0.19892, dev 19.1 / 50.0 / 5.5 (M17 reference: 163 / 124, p = 2e-4). `final_nocsls_simple/`: no s3, LOFO 0.19522, dev 19.4 / 49.0 / 5.8.
- **Other rows (LOFO / dev):** 4 members plain 0.19307 / 20.1, 50.2; 4 members + CSLS per member k=5 0.20735 / 24.5, 56.6; 5 members plain 0.19522; 5 members centered 0.19525 / 20.3, 50.5; 4 members centered + CSLS 0.20855.
- **Seed robustness (con seeds 3-5, exhaustive search):** the weights move by at most one 0.1 transfer (4 members 0.0/0.3/0.4/0.3 -> 0.0/0.4/0.3/0.3; 5 members plain 0.0/0.4/0.3/0.2/0.1 -> 0.0/0.3/0.3/0.3/0.1; the headline set keeps exactly 0.0/0.4/0.3/0.2/0.1, dev 25.5 / 55.9 / 4.0 vs 24.0 / 57.1 / 4.2, p = 0.37).
- **Caveats:** alphas are chosen on the full training set (not nested); the CSLS in the CV is applied to each held-out fold (~200 recordings) while the test set has 102; dev (4 x 102 trials) was never used for selection.
