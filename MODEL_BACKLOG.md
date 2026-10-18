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
| Linear contrastive head (InfoNCE on frozen features) | medium–high | Optimizes ranking directly. Smallest step toward Stage 3 |
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
