# HANDOFF: state of the Krabiki EEG project (2026-10-18, end of day)

For teammates and their agents picking this up. Read this first, then `CLAUDE.md` (rules and conventions),
`PROGRESS.md` (detailed log), and `MODEL_BACKLOG.md` (every experiment M0–M28, the source of truth for numbers).

## 1. Where we stand
- **Final submission:** `submissions/final_v4b_csls/sub-0{5,6}_ranked_candidate_list.npy` on branch **`track-proven`**.
  Validated (102×102 int, rows in original test order, each row a permutation of 1001..1102).
- **Model v4b (M28):** 7 members, each a whitened ridge decoder EEG → feature space, except one contrastive head:
  pooled DINOv2-B patch (.2), V-JEPA 2-L 16 f (.1), contrastive two-tower on CLIP B-32 (.1), pooled AlexNet conv5 6×6 (.2),
  hand-crafted low-level descriptors (.1), CLAP audio (.1), pooled "lowav" low-level AV per 100 ms (.2). Global weights chosen by
  training CV, test-set centering, CSLS k=5 per member ("pooled" = subject pooling, see §3).
- **Dev (sub-01..04 test sets): 33.8% top-1 / 68.1% top-5 / median rank 2.6** (chance 1.0 / 4.9 / 51.5). The organizers' recipe
  gave 2.0 / 8.6. LOFO train-CV 0.2872.
- **Expected on the scored subjects** (estimate, probably optimistic): sub-05 ~26% / ~60%, sub-06 ~18% / ~49%.
- **Backups** (all in `submissions/`, see `submissions/README.md`): `final_v4_csls` (v4a, 33.3 / 68.9), `final_v3_csls`
  (32.4 / 66.7), `final_v3_nocsls` (no CSLS, 25.2 / 58.1), `final_csls` (24.0 / 57.1, fully audited), `decw_clip_b32_*` (14.7 / 42.2).

## 2. Documents (all on `track-proven`)
| File | What |
|---|---|
| `HANDOFF.md` | this file |
| `CLAUDE.md` | task, hard rules, data layout, channel names, conventions |
| `PRESENTATION.md` | 14-slide script with numbers, figures, Q&A, credits |
| `IMPROVEMENTS.md` | every major step from 2% to 34%, explained in plain language |
| `ARCHITECTURE.md` | pipeline (Mermaid diagram), member families, selection protocol, code map |
| `AUDIT.md` | leakage / validation audit (full controls on final_csls; quick audits on v3 / v4) |
| `EEGNET.md` | Stage 3 (EEGNet) attempt, why it fell short, references |
| `MODEL_BACKLOG.md` | all experiments M0–M28 with numbers (append-only) |
| `PROGRESS.md` | current best (§1) + the chronological log |
| `DATA_SURVEY.md` | external datasets surveyed; excluded ones; THINGS-EEG2 test |
| `IMPLEMENTATION_AUDIT.md`, `CONTRASTIVE_AUDIT.md` | earlier audits |
| `submissions/README.md` | every submission set, its numbers and the reproduce command |

## 3. The path (one line each; details in `IMPROVEMENTS.md`)
1. Organizers' recipe (AlexNet encoding): 2.0 / 8.6.
2. CLIP features: 4.2 / 16.9. Removing a bug (a post-PCA scaler made ridge alpha a no-op): 5.4 / 21.1.
3. **Decoding** (EEG 50–600 ms → ridge → feature vector, cosine vs candidates): **14.0 / 36.0**.
4. **EEG noise whitening** (per-timepoint Ledoit-Wolf noise covariance from training residuals): 14.7 / 42.2.
5. Ensemble of different feature models (DINOv2, V-JEPA, contrastive head), CV weights: 20.8 / 48.0.
6. CSLS hub correction (approved by the organizers) + test-set centering: 24.0 / 57.1.
7. **Low-level vision + audio** (AlexNet conv5, descriptors: krabiki-51; lowav + CLAP: Alikhan): **32.4 / 66.7**.
8. **Subject pooling** (other subjects' training trials at weight λ 0.05–0.2 per subject, chosen by zero-shot CV): 33.8 / 68.1.

Not adopted (with reasons in `MODEL_BACKLOG.md`): more frames, CLIP ViT-L, extra windows, THINGS-EEG2 transfer (M15), EEGNet in
the final (M14), motion gate (M23), pooling all six decoders (v4c, +0.0003), dropping the contrastive head (M27).

## 4. Code and branches (GitHub `Gundan-Dante/Krabiki`)
| Branch | Local worktree | Contents |
|---|---|---|
| `track-proven` | `/home/kparpie/Krabiki` | main package `src/eegmr`, final submissions, all docs, `scripts/export_bundle.py` + `infer_bundle.py` (numpy-only inference), audit tooling (`audit_controls.py`, `audit_compare.py`, `audit_runbook.sh`), Stage 3 |
| `track-matching` | `../Krabiki-matching` | **`scripts/ensemble.py`** (N-member CV weighting, exhaustive ≤7, CSLS, center-test, member types decw / dec / con / s3 / zs / enc / pdecw), low-level features (`src/eegmr/lowlevel.py`), final runs v3 / v4 (commit efb37e0) |
| `track-pool` | `../Krabiki-pool` | `scripts/pool_member.py` (subject-pooled decoders, `pdecw`) |
| `track-audio` | `../Krabiki-audio` | Alikhan: `src/eegmr/audio.py`, `scripts/extract_audio.py` (CLAP, lowav / lowvis / lowaud), M20–M22 |
| `track-viz` | `../Krabiki-viz` | presentation figures `presentation/figures/fig1..9`, `make_figures.py`, `FIGURES.md` (commit 187934c) |
| `track-dinov2`, `track-vjepa`, `track-alexnet` | | feature extractors / earlier snapshots (merged into the above) |
| `main` | | Azamat's separate track (time-staged experts, motion gate). Never force-push it. |

**The branches are NOT merged into one yet.** The final model needs `track-matching` (ensemble.py) + `track-pool` (pool_member.py)
+ the extractors; `track-matching` efb37e0 already contains the merges needed to reproduce v4b.

## 5. Reproduce
- **Fast (inference from saved weights, CPU, seconds):** `bundle/krabiki_final_v4b_bundle.zip` (287 MB, local only, gitignored; ask
  Kanat for the file). `python <bundle>/infer_bundle.py --bundle <bundle> --root <eeg-mind-reading-challenge> --subjects 5 6 --out out`
  reproduces `final_v4b_csls` bit-for-bit (verified twice). numpy + scipy only.
- **Full rerun (GPU):** extract features (`scripts/extract_features.py` for clip_b32 / dinov2_b `--all-frames` / vjepa2_l / alexnet;
  `scripts/lowlevel_features.py` for AlexNet conv5 pooling + descriptors; `scripts/extract_audio.py` for clap / lowlevel AV, needs
  `av==18.1.0`), build the pooled members with `scripts/pool_member.py` (track-pool), then on track-matching:
  `PYTHONPATH=src:scripts python scripts/ensemble.py --member pdecw:dinov2_b_patch_all --member decw:vjepa2_l_all:16 --member
  con:clip_b32:two_tower_d256 --member pdecw:alexnet_c5p6:8 --member decw:lowlevel_all:1 --member decw:clap_all:1 --member
  pdecw:lowav_all:1 --max-seconds 1.5 --sim-cache results/similarity/_cache --center-test --csls 5 --csls-stage member --experiment v4b_ctr_m5`
  (the pdecw entries come from pool_member.py caches; see PROGRESS.md "FINAL v4" on track-matching).

## 6. Rules and decisions you must keep
- Never train on test data. Dev labels (sub-01..04 `tst_stim`) are for validation only. Select by train-CV (KFold seed 0); confirm on
  dev with a paired Wilcoxon. Never tune on dev.
- **Never use** the EEG Moments Dataset (OpenNeuro ds008257), BOLD Moments / Algonauts 2021 data or annotations, anything built on
  the 1,102 Moments-in-Time clips, or models trained on them (e.g. TRIBE v2). They contain or describe this challenge's data.
- Organizer rulings: **CSLS is allowed** (2026-10-18). One-to-one assignment is not used. Whether gated datasets are "public" is open.
- User decisions: audio was later allowed (CLAP is in the final); subject pooling is now used (weighted, per-subject λ).
- Git: commit with the GitHub noreply email (`168792404+Kasha228@users.noreply.github.com`, already set in the repo config), not the
  corporate one. Plain pushes only; never force `main`.
- Compute: `OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8`; one shared 8 GB GPU. The venv is shared: dry-run any
  `pip install` and never change torch (CUDA build) / numpy / scipy / sklearn.

## 7. Open items / next steps (in priority order)
1. **Shuffle control on v4b** (only a rescore was done before the stop): `scripts/audit_runbook.sh` with the v4b variables (see the
   last report in PROGRESS / the runbook header). v3 passed it; v4b swaps three members for pooled versions.
2. **Consolidation:** merge `track-matching` + `track-pool` (+ `track-audio` extract code) into one branch, keep only the final path
   (move experiment scripts to `research/`), one command from saved weights, README, tests, pin the CLIP / DINOv2 revisions
   (G1–G4 in `IMPLEMENTATION_AUDIT.md`).
3. **Pooled EEGNet** (Stage 3 is data-limited; pooling is the natural test, ~15 min CV-only on sub-05/06). Not run.
4. Better cross-subject alignment before pooling (hyperalignment / Procrustes) could raise the useful λ and help sub-06 most.
5. Optional: a v4b inference bundle export for teammates already exists locally; the older `krabiki_final_csls_bundle.zip` is for
   the previous headline.

## 8. Who did what (sessions)
- **krabiki-e1 / krabiki-75** (coordinator): direction, the whitening-bug fix, decoding prototype, reviews and independent
  verifications, audits, docs, final selection.
- **krabiki-2f:** main package, GPU helpers, whitened decoder, contrastive head (M12), Stage 3 (M14), audits, pooling
  implementation (`pdecw`), inference bundles, ARCHITECTURE.md.
- **krabiki-51:** matching / noise whitening (M5), ensemble tool, THINGS-EEG2 test (M15), low-level features (M24), final
  selections v3 / v4 (M25, M28), data survey.
- **krabiki-cc / krabiki-c5:** presentation figures; V-JEPA features (M16), zero-cost ideas incl. CSLS (M18).
- **Alikhan:** CLAP audio + lowav features, super-ensemble (M20–M22), cross-hardware reproduction.
- **Azamat:** time-staged experts track, motion gate (M23 tests), research summary (`RESEARCH_ALTERNATIVES.md` on `main`).
