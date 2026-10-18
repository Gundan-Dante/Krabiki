# Krabiki: EEG mind reading (orientation for agents)

**Task:** EEG (17 channels, 500 Hz, 3 s) was recorded while people watched 3 s videos with sound. For each test recording,
rank the 102 test videos (IDs 1001–1102). Scored on top-1 / top-5 for sub-05 and sub-06 (chance 1.0% / 4.9%). Dev
subjects sub-01..04 have test labels (`tst_stim`) for validation only.

## Rules (hard)
- Python 3.11. External models and data must be public and released before 2026-10-16. Record exact names and versions.
- Never train on any subject's test data. Dev test labels are for validation and model selection only. The same code path
  runs for all subjects. Unsupervised normalization of test data is allowed.
- **Never use** the EEG Moments Dataset (OpenNeuro ds008257), BOLD Moments / Algonauts 2021 data, or annotations of
  our 1,102 clips, or models trained on them (e.g. TRIBE v2). They contain or describe this challenge's data.
- Submission: `sub-0X_ranked_candidate_list.npy`, int (102, 102), rows in the ORIGINAL test order, values are video IDs.
  Use `eegmr.submission.write_submission`.

## Data
`eeg-mind-reading-challenge/` (not in git): `trn/sub-XX_trn.npz` (key = video ID without leading zeros → (17, 1500)),
`tst/sub-XX_tst.npy` (102, 17, 1500), `tst_stim/` (dev labels), `stimuli_{train,test}/*.mp4`.
Channels, in order: Pz P3 P7 O1 Oz O2 P4 P8 P1 P5 PO7 PO3 POz PO4 PO8 P6 P2. Already band-passed 0.1–40 Hz, blinks
removed. These are the same 17 channels as the THINGS-EEG2 preprocessed release.

## Code (branch `submission`: the organizer-facing layout)
- `training/src/eegmr/`: `data` (loading, crop + decimation), `features` / `lowlevel` / `audio` (extractors, pinned
  revisions, cache in `features/`), `decoding` (whitened decoder), `matching` (noise whitening), `contrastive`, `gpu`
  (torch ridge / CV helpers), `retrieval` (CSLS), `evaluate`, `submission`.
- `training/scripts/`: `ensemble.py` (members, CV weights, test), `pool_member.py` (subject pooling), extractors,
  `export_bundle.py`. `bash training/run_pipeline.sh` runs the whole final pipeline (`JOBS=3` for parallel subjects).
- `inference/`: numpy + scipy predictor and the v4b weights (`inference/model/`); reproduces `submissions/final_v4b_csls`.
- `research/scripts/`: experiments and audit tooling not on the final path (e.g. single-model `predict.py`).

## Where things are
- **Start with `docs/HANDOFF.md`** (current state, final model, branches, open items).
- `docs/MODEL_BACKLOG.md`: every model and experiment with its numbers (M0…). The source of truth. Append, don't rewrite.
- `docs/`: `PROGRESS.md`, `ARCHITECTURE.md`, `AUDIT.md`, `IMPROVEMENTS.md`, `DATA_SURVEY.md`, earlier audits.
- Parallel work lives in git worktrees `../Krabiki-{matching,dinov2,viz,vjepa}` on branches `track-*`.

## Conventions
- Select by train-CV (5-fold on training videos, KFold seed 0). Confirm on dev with a paired Wilcoxon. Never tune on dev.
- Launch with `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8`. Use the GPU (torch, CUDA, 8 GB, shared).
- The venv is shared (`.venv`, torch CUDA build). Dry-run any `pip install` and never change torch / numpy / scipy / sklearn.
- Downloaded data goes outside the repo (e.g. `~/ext_data/`) and is loaded without pickle.
