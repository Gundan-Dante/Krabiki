# Audio + low-level audio-visual members: findings (branch `track-audio`)

**Bottom line:** adding two cheap members, sound (CLAP) and per-100-ms low-level audio-visual features, to the team
ensemble gives the best result so far. On dev (mean over sub-01..04) it reaches **33.1% top-1 / 67.9% top-5 / median
rank 2.9**. That beats the AlexNet v2 ensemble, at 29.4 / 63.0, on paired tests (p = 5e-7). Details and reproduce commands
are in `MODEL_BACKLOG.md` M20 / M21 / M22.

## Results (dev = sub-01..04 test sets, 102 candidates; chance 1.0 / 4.9 / 51.5)

| Ensemble | Settings | top-1 | top-5 | median rank |
|---|---|---|---|---|
| Team headline `final_csls` (this morning) | centering + CSLS 5 | 24.0% | 57.1% | 4.25 |
| M21: M17 + CLAP + low-level AV | CSLS 5 | 29.4% | 64.5% | 3.2 |
| v2: headline + AlexNet p6 (`ALEXNET_FINDINGS.md`), reproduced exactly | centering + CSLS 5 | 29.4% | 63.0% | 3.5 |
| **M22: v2 + CLAP + low-level AV** | **centering + CSLS 5** | **33.1%** | **67.9%** | **2.9** |
| M22 backup | no centering, no CSLS | 26.2% | 57.6% | 4.0 |

- **M22 vs v2:** 183 trials better / 90 worse, Wilcoxon p = 5e-7. Per-subject top-1 is 40.2 / 36.3 / 25.5 / 30.4 for M22
  and 33.3 / 36.3 / 20.6 / 27.5 for v2. Leave-one-fold-out CV is 0.264 vs 0.236.
- **CV weights** (global): DINOv2-patch 0.2, V-JEPA 0.2, contrastive 0.1, **AlexNet 0.2, low-level AV 0.2, CLAP 0.1**.
  CLIP and CLIP block-8 get 0.0. Spatial low-level features (AlexNet) and temporal/audio ones (ours) are complementary.
- **Submissions** (local and gitignored, validated as 102 x 102 int64 with each row a permutation of 1001..1102):
  headline candidate `submissions/merged_all_ctr_csls5_cv/`, backup `submissions/merged_all_plain_cv/`.

## The two new members
| Member | What it is | Alone (dev) | Why it helps |
|---|---|---|---|
| `decw:clap_all:1` | LAION-CLAP `clap-htsat-unfused` (pinned revision) embedding of each clip's audio, 512-d | 5.4 / 20.3 | Sound content. RSA vs CLIP is only 0.18 (V-JEPA's was 0.48), so it's the most different view we have. |
| `decw:lowav_all:1` | No model. Per 100 ms bin over 3 s: luminance, contrast, edges, motion, loudness (RMS), onset strength, 8 log-mel bands (background-noise texture). 30 x 14 = 420-d | 4.9 / 22.3 | Timing of brightness, motion and sound changes, which the posterior EEG follows. Both halves matter: separately, audio got weight 0.2 and visual 0.3. |

- Both use the standard whitened decoder, EEG window 50-600 ms. CV picked that window over 0-300 ms, 0-1.5 s and 0-3 s on 6/6
  subjects.
- **Controls:** decoders trained on shuffled training pairs score at chance (low-level 1.2 / 4.2, CLAP 0.5 / 3.9).
- **Rules:** CLAP is Apache-2.0 from 2023, and its training data (LAION-Audio-630K, at worst plus AudioSet) has no video
  data and no Moments in Time.
- **Data:** 43 clips have no usable audio, 3 of them test videos (1015, 1043, 1102). CLAP gives them the mean training
  embedding. The low-level audio features use "silence" values for them.

## Sub-05 / sub-06 (no labels: expected scores, not measured ones)
The test labels are withheld, so no accuracy can be measured. Train-CV is available for every subject. On dev, test top-1
is about 2.1x and top-5 about 1.6x the CV-expected values (the test EEG is cleaner, probably repeat-averaged). That ratio
gives a rough estimate:

| Subject | CV E[top-1] / E[top-5] (M22) | Estimated test top-1 / top-5 | CV gain over v2 (E[top-1]) |
|---|---|---|---|
| sub-01..04 (dev, measured) | 0.112-0.195 / 0.347-0.492 | 25.5-40.2% / 57.8-72.5% (real) | |
| **sub-05** | 0.126 / 0.363 | **~26% / ~58%** | 0.109 -> 0.126 (+16%) |
| **sub-06** | 0.090 / 0.303 | **~19% / ~48%** | 0.067 -> 0.090 (+35%) |

- These estimates are rough: they rest on 4 calibration subjects, each with about ±4 points of noise. sub-06 is the
  weakest subject (fewest training trials: 850).
- The audio / low-level members help the weak subjects most. For sub-06, the low-level member gets the largest weight
  (0.4).
- Sanity checks on the files: 58 / 56 distinct top-1 videos (no collapse onto a few hubs). The most any single video
  appears as top-1 is 6 / 4 times.

## Before adopting M22 as the submission
Run the same checks `AUDIT.md` lists for `final_csls`: an independent rescore from the saved similarities, label-shuffle
and row-permutation controls on the full 8-member ensemble, and determinism.

## Reproduce (one GPU, ~1 h on an RTX 3050 4 GB)
```bash
PYTHONPATH=src python scripts/extract_audio.py --model clap
PYTHONPATH=src python scripts/extract_audio.py --model lowlevel
PYTHONPATH=src python scripts/extract_features.py --model clip_b32 --frames 8
PYTHONPATH=src python scripts/extract_features.py --model dinov2_b --all-frames --frames 8
PYTHONPATH=src python scripts/extract_features.py --model vjepa2_l --frames 16
PYTHONPATH=src python scripts/extract_features.py --model clip_b32_mid8 --subset-only --frames 8
PYTHONPATH=src python scripts/extract_features.py --model alexnet --frames 8
PYTHONPATH=src python scripts/lowlevel_features.py alexnet-pool --pool 6
PYTHONPATH=src:scripts python scripts/ensemble.py --member decw:clip_b32 --member decw:dinov2_b_patch_all \
  --member decw:vjepa2_l_all:16 --member con:clip_b32:two_tower_d256 --member decw:clip_b32_mid8_all:8 \
  --member decw:alexnet_p6:8 --member decw:clap_all:1 --member decw:lowav_all:1 \
  --max-seconds 1.5 --center-test --csls 5 --csls-stage member --experiment merged_all_ctr_csls5
```
New code: `src/eegmr/audio.py`, `scripts/extract_audio.py`; `requirements.txt` pins `av==18.1.0`.
