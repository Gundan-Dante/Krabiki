# Progress: track-proven

## Current best (mean over sub-01..04 test sets, 102 candidates)
| Experiment | top-1 | top-5 | median rank |
|---|---|---|---|
| **enc_clip_b32_f8_t1.5s_w5-60_retrieval** (CLIP ViT-B-32 / laion2b_s34b_b79k, 8 frames, scaler → PCA 95% → ridge, EEG 0–1.5 s, alpha by retrieval CV, no whitening) | **5.4%** | **21.1%** | **17.5** |

Submission files: `submissions/enc_clip_b32_f8_t1.5s_w5-60_retrieval/sub-0{5,6}_ranked_candidate_list.npy`.
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

## Next
- Re-test on the unwhitened pipeline: z-scoring (still expected to be a no-op), finer alpha grid around 1e4,
  PCA variance threshold or no PCA (kernel ridge), and the per-subject CV quality estimates.
  (CSLS: dropped for now.)
- Bigger levers: features (DINOv2, audio, more frames).
- Sliding 100 ms window scan to find where the signal is (also report figure §8.1); then tune the window inside 0–600 ms.
- Decoding direction (EEG → CLIP space, cosine); DINOv2 and audio features; pooled subjects.
