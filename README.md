# Krabiki: EEG mind reading (team submission)

For each test EEG recording (17 posterior channels, 500 Hz, 3 s, recorded while a person watched a 3 s video with sound)
we rank the 102 test videos (IDs 1001–1102) from most to least likely. The method is zero-shot retrieval: linear decoders
map the EEG into the feature spaces of frozen public models (image, video, audio, low-level), and an ensemble weighted
by cross-validation on training data ranks the candidates.

| | |
|---|---|
| **Report** | [`REPORT.md`](REPORT.md): the final model, its algorithms and how it was validated |
| **Predictions** | [`submissions/final_v4b_csls/`](submissions/final_v4b_csls/): `sub-05_ranked_candidate_list.npy`, `sub-06_ranked_candidate_list.npy` (int64, 102 × 102, rows in the original test order, values = video IDs 1001–1102) |
| **Dev score** (test sets of sub-01..04, labels used only for scoring) | **33.8% top-1 / 68.1% top-5**, median rank 2.6 (chance 1.0% / 4.9% / 51.5). Per subject: 35.3/71.6, 36.3/68.6, 26.5/65.7, 37.3/66.7 |
| **Reproduce the predictions** (CPU, numpy + scipy, seconds) | [`inference/`](inference/): `python inference/predict.py --data-root <eeg-mind-reading-challenge> --out out/` |
| **Retrain everything** (one GPU, Python 3.11) | [`training/`](training/): `bash training/run_pipeline.sh` |

## Repository layout

| Folder | What it contains |
|---|---|
| `submissions/final_v4b_csls/` | The submitted sub-05 / sub-06 ranked candidate lists. |
| `inference/` | Standalone predictor (numpy + scipy only, no GPU, no deep-learning libraries) and **our trained weights** (`inference/model/`): per subject and member the noise whiteners, standardization statistics, ridge maps / contrastive projections, plus the test-video features. It reproduces the submitted files bit-for-bit. See [`inference/README.md`](inference/README.md). |
| `training/` | The full training pipeline: package `eegmr` (`training/src/eegmr`), the scripts on the final path (`training/scripts`), tests, pinned `requirements.txt`, and `run_pipeline.sh` (features → decoders → subject pooling → CV-weighted ensemble → ranked lists → export to `inference/model`). See [`training/README.md`](training/README.md). |
| `docs/` | How the method was developed and checked: [`IMPROVEMENTS.md`](docs/IMPROVEMENTS.md) (every step from 2% to 34%), [`ARCHITECTURE.md`](docs/ARCHITECTURE.md), [`AUDIT.md`](docs/AUDIT.md) (leakage and validation checks), [`MODEL_BACKLOG.md`](docs/MODEL_BACKLOG.md) (all experiments M0–M28 with numbers), [`PRESENTATION.md`](docs/PRESENTATION.md). |
| `research/` | Experiment and audit scripts, and earlier submission sets, that are not needed for the final model. Kept for transparency. |
| `results/` | Logs of the experiments (`results.csv`, cross-validation tables). |

The data folder `eeg-mind-reading-challenge/` and the feature caches `features/` are not in the repository (videos and
frames must not be committed). Put the data folder at the repository root, or pass its path explicitly.

## The final model (v4b)

```
test EEG (102 × 17 × 1500) → crop 0–1.5 s → FIR decimation to 100 Hz → test-set centering (unsupervised)
  → window 50–600 ms → per-timepoint noise whitening (Ledoit-Wolf, from training residuals) → standardize
  → 7 members, each EEG → feature space, cosine to the 102 candidates → CSLS k=5 → row z-score → weighted sum → ranking
```

| Member | Feature model (exact version) | Weight |
|---|---|---|
| Ridge decoder, subject-pooled | DINOv2 ViT-B/14, timm `vit_base_patch14_dinov2.lvd142m`, HF rev `4685c99dabffe5affac90bd99dbffd25801ae58d`, mean of patch tokens over 8 frames | 0.2 |
| Ridge decoder | V-JEPA 2 ViT-L, `facebook/vjepa2-vitl-fpc64-256`, HF rev `b3c1679b7c34d3255ef3547f27c7b226aefab26f`, 16 frames | 0.1 |
| Contrastive two-tower head (InfoNCE, 3 seeds) | open_clip ViT-B-32 / `laion2b_s34b_b79k`, HF `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` rev `1a25a446712ba5ee05982a381eed697ef9b435cf`, 8 frames | 0.1 |
| Ridge decoder, subject-pooled | torchvision AlexNet `IMAGENET1K_V1` (`alexnet-owt-7be5be79.pth`), conv5 pooled to 6 × 6, 8 frames | 0.2 |
| Ridge decoder | hand-crafted descriptors (colour, spatial frequency, edges, motion); no model | 0.1 |
| Ridge decoder | LAION-CLAP `laion/clap-htsat-unfused`, HF rev `8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a` (audio track) | 0.1 |
| Ridge decoder, subject-pooled | low-level audio-visual features per 100 ms (luminance, contrast, edges, motion, loudness, onsets, 8 mel bands); no model | 0.2 |

"Subject-pooled": the decoder of a subject is trained on its own training recordings plus the other five subjects'
**training** recordings at a small weight λ (0.05–0.1, chosen per subject by cross-validation). Every model choice
(alphas, λ, members, weights) was made by 5-fold cross-validation on training videos. The dev test sets only confirmed
results. Details: [`docs/IMPROVEMENTS.md`](docs/IMPROVEMENTS.md), [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Rules compliance

- Python 3.11. All external models are public, released before 2026-10-16, and loaded at the pinned versions above.
- No test data of any subject is used for training. Every fitted quantity comes from training recordings. The test
  recordings enter only through unsupervised normalization (centering on the test set's own mean) and CSLS hub
  correction on the unlabeled 102 × 102 similarity matrix (CSLS was approved by the organizers on 2026-10-18).
- Dev labels (`tst_stim`, sub-01..04) are read only after the ranked lists are written, for scoring. They are never used
  as templates for sub-05 / sub-06.
- Controls ([`docs/AUDIT.md`](docs/AUDIT.md)): with training EEG ↔ video pairs shuffled, the pipeline falls to chance.
  Outputs don't depend on the order of the test recordings, and reruns are bit-identical.
- We found that a public dataset (EEG Moments, OpenNeuro ds008257) likely contains this challenge's recordings, and
  deliberately never used it, nor BOLD Moments / Algonauts 2021 data or annotations of these clips.
