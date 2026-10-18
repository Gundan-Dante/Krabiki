# Starting point for audio features (branch `track-audio`)

This branch merges `track-proven` (decoders, contrastive head, CLAUDE.md, PROGRESS.md), `track-matching` (ensemble tool)
and `track-vjepa` (clip-level extractor pattern, `transformers` pinned). Read `CLAUDE.md` (rules, data, conventions) and
`PROGRESS.md` §8 (method notes) first. This page is the short version for audio.

## What we built that you can reuse
- **Decoder (our best single head, "decw"):** EEG 50–600 ms (17 ch × 55 samples at 100 Hz, from a 1.5 s crop) →
  per-timepoint **noise whitening** (Ledoit-Wolf noise covariance from out-of-fold training residuals) → standardize →
  **ridge** → predicted feature vector → **cosine** vs the 102 candidate videos' feature vectors → ranking.
  Alpha is chosen by **retrieval CV** on training videos (5-fold, KFold seed 0), never on dev.
  Code: `src/eegmr/decoding.py`; CLI: `scripts/predict.py --head decoding --dec-whiten`.
- **Contrastive head** (InfoNCE on the same whitened EEG inputs → frozen features): weak alone, useful as an ensemble
  member. Code: `src/eegmr/contrastive.py`, `scripts/train_contrastive.py`.
- **Ensemble tool:** `scripts/ensemble.py --member decw:<features>[:<frames>] --member con:<features>:<config> ...`
  Weights come from an exhaustive search on train-CV, strategy from leave-one-fold-out. It writes validated sub-05/06 files.
- **Reference numbers** (dev = sub-01..04, top-1 / top-5): CLIP B-32 decw 14.7 / 42.2; DINOv2-patch decw 13.0 / 41.7;
  V-JEPA 2 decw 13.0 / 39.5; best ensemble M17 20.8 / 48.0. Chance is 1.0 / 4.9.
  Per-subject train-CV (mean percentile rank, lower is better) for CLIP decw: 0.192 / 0.206 / 0.251 / 0.219 / 0.247 / 0.277.

## Step 1: audio alone (raw comparison)
1. **Rules check:** an audio model that is public, released before 2026-10-16, with its license recorded, and not trained
   on Moments in Time (the source of our clips). Candidate: LAION-CLAP via `transformers` (`ClapModel`); pin the exact
   HF revision. Don't use the BOLD Moments / EEG Moments datasets or their annotations.
2. **Extract one embedding per clip:** decode the audio from each mp4 (1,102 clips; **34 have no audio stream**, so give
   them a zero vector or the mean vector, and record which ones). Resample to the model's rate and embed the 3 s.
   Follow the `extract_clip_features` / `_vjepa2_l` pattern in `src/eegmr/features.py`, and cache as
   `features/<name>_all_f1.npz` with `ids` and `mean`, so `load_features("<name>_all", 1, ids)` works. Use the GPU.
3. **Run the same decoder on audio:**
   `PYTHONPATH=src python scripts/predict.py --head decoding --dec-whiten --features <name>_all --frames 1
   --max-seconds 1.5 --alpha-selection retrieval --subjects 1 2 3 4 5 6`
   Compare per-subject train-CV and dev top-1/top-5 with CLIP decw above. A paired Wilcoxon on per-trial ranks is the test.
4. **Optional quick diagnostics:** decode in the first 0–300 ms only vs 50–600 ms. Auditory responses may peak at a different
   time than visual ones.

## Step 2: add it to the visual models, gradually
`PYTHONPATH=src:scripts python scripts/ensemble.py --member decw:clip_b32 --member decw:<name>_all:1 --max-seconds 1.5
--experiment ens_clip_audio`, then add `decw:dinov2_b_patch_all`, `decw:vjepa2_l_all:16`, `con:clip_b32:two_tower_d256`.
Keep audio only if the CV tool gives it weight > 0 and leave-one-fold-out improves. Dev must agree in direction.

## Conventions
- `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8`. Shared GPU: keep jobs small.
- The venv is shared: dry-run any `pip install`, and never change torch (CUDA build) / numpy / scipy / sklearn.
- Data (`eeg-mind-reading-challenge/`) and `features/` are not in git. Re-extract the visual features with
  `scripts/extract_features.py` if you're on another machine.
- Never select on dev. Record results in `MODEL_BACKLOG.md` (append a new M-number).
- CSLS / test-set centering exist as options but are pending an organizer decision (CSLS). Report audio without them first.
