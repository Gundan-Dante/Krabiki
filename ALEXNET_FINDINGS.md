# AlexNet low-level member: preliminary findings (branch `track-alexnet`)

**Status: PRELIMINARY.** This branch is a snapshot of krabiki-51's work in progress (2026-10-18), taken from
track-matching 7297a59 plus its uncommitted files at that moment. Not yet audited like `final_csls` (AUDIT.md §4). The
submitted set stays `submissions/final_csls/` until this passes the same checks.

## Finding
Adding a **low-level visual member** (AlexNet conv2) to the headline ensemble gives the largest single gain since
decoding itself:

| Model | Dev top-1 | Dev top-5 | Median rank | LOFO (train-CV) |
|---|---|---|---|---|
| Headline `final_csls` (fin5_ctr_m5) | 24.0% | 57.1% | 4.25 | 0.21235 |
| **+ AlexNet conv2, pooled 6×6 (fin6x_alexp6_ctr_m5)** | **29.4%** | **63.0%** | **3.5** | **0.23619** |

- Per subject (top-1 / top-5 / median): sub-01 33.3 / 66.7 / 3, sub-02 36.3 / 68.6 / 3, sub-03 20.6 / 53.9 / 5,
  sub-04 27.5 / 62.7 / 3. Paired vs headline: better 199 / worse 93 recordings, Wilcoxon p = 2e-10.
- CV-chosen global weights: CLIP 0.0 / DINOv2-patch 0.3 / V-JEPA 0.2 / contrastive 0.2 / CLIP block-8 0.0 /
  **AlexNet 0.3**. Settings as in the headline: test centering, CSLS k=5 per member, exhaustive weight search. Dev was not
  used for any choice.
- **Verified by krabiki-e1:** rescored from the saved similarity matrices (29.4 / 63.0, medians 3 / 3 / 5 / 3, same
  Wilcoxon). **Leak check by krabiki-2f:** the AlexNet-p6 features are bit-identical to re-pooling the cached
  `alexnet_f8`, and the test-video rows don't change when the training rows are zeroed.

## What the member is
- Features: torchvision AlexNet `IMAGENET1K_V1` (public since 2014), `features.3` (conv2, pre-ReLU) on 8 frames per video,
  averaged over frames (`features/alexnet_f8.npz`, extracted early in the project with our own code, not organizer code).
  51 average-pools the 192 × 27 × 27 maps to 192 × 6 × 6 = 6,912 dims (`features/alexnet_p6_f8.npz`).
- Used as a whitened decoder (`decw:alexnet_p6:8`): EEG → noise whitening → ridge → predicted conv2 vector → cosine.
- **Why it helps:** our 17 electrodes are all posterior (visual cortex), which responds strongly to low-level structure
  (edges, contrast, texture, layout). CLIP, DINOv2 and V-JEPA encode semantics and wash this out. AlexNet alone is
  modest (13.0 / 33.6 with CSLS), but it covers variance the semantic members miss. This matches the literature (the ICLR
  2026 "dynamic mixture of experts" paper on this dataset) and Azamat's time-staged track (low-level features match
  early EEG).

## Code in this snapshot (work in progress)
- `src/eegmr/lowlevel.py`: low-level descriptors (AlexNet taps c1–c5, pooling, colour / frequency / edge / motion groups).
- `scripts/lowlevel_features.py`: CLI, e.g. `python scripts/lowlevel_features.py alexnet-pool --pool 3 6 9` (from the
  cached conv2), `alexnet-layers --layers c1 c3 c4 c5 --pool 6` (GPU), `descriptors` (CPU).
- `src/eegmr/features.py`: extra extractors (CLIP B-32 intermediate blocks 4 / 6 / 8 / 10).
- `tests/test_lowlevel.py`.
- Reproduce the verified result (after building `alexnet_p6`): `PYTHONPATH=src:scripts python scripts/ensemble.py
  --member decw:clip_b32 --member decw:dinov2_b_patch_all --member decw:vjepa2_l_all:16 --member con:clip_b32:two_tower_d256
  --member decw:clip_b32_mid8_all:8 --member decw:alexnet_p6:8 --max-seconds 1.5 --center-test --csls 5 --csls-stage member`

## In progress (CSVs included, not yet interpreted; don't quote)
`results/sel1_*_cv.csv` (screening of AlexNet layer / pooling variants c1 / c3 / c4, pool 3 / 9, and CLIP intermediate blocks)
and `results/fin6x_mid{4,6,10}_ctr_m5_cv.csv` (CLIP blocks as the 6th member). The low-level descriptor sweep (colour, spatial
frequency, edges, motion energy) is still running. Final v2 sets (`final_v2_csls/`, `final_v2_nocsls/`) will follow on
track-matching.

## Before adopting v2 as the submission
Committed extraction code (pooling included), CV selection, an independent rescore, and the AUDIT.md controls (label shuffle,
row permutation, determinism).
