# Implementation audit: what is submission-ready, and what is missing

Audit of every model and finding in `MODEL_BACKLOG.md` (M0–M10 plus matching variants), and of the repo against the
brief (`ClaudeREADME.md`), as of `track-proven` @ `0df453f` (2026-10-18). Written by session krabiki-2f.
Read-only: no code was changed for this audit.

> **Update (2026-10-18, after krabiki-e1's review):** M9 (krabiki-51) and M10 (krabiki-e1) are folded in. The CLIP
> encoder (M5) is **no longer a useful ensemble member**, since CV picks the whitened decoder alone. The lead
> candidate is now an **ensemble of whitened decoders on CLIP B-32 and DINOv2-B patch-mean**. The final run (G2) must
> therefore support N decoder members with CV-chosen weights. Implementation of G1–G4 is **on hold** pending the
> user's approval.

Metric convention as in MODEL_BACKLOG: dev = test sets of sub-01..04, top-1 / top-5 / median rank (mean of
per-subject medians). Train-CV = mean percentile rank on held-out training folds (lower is better, 0.5 = chance).

---

## TL;DR

- **Final pipeline today:** M8a, the decoder with noise-whitened input. Dev **14.7 / 42.2 / 7.5**. Over 3 CV-fold
  seeds: 14.4 ± 0.4 / 41.7 ± 0.4 / 7.5 ± 0.0. Train-CV is better than the plain decoder on all 6 subjects, and the
  result was reproduced independently by krabiki-e1. It runs in about 20 s per subject. Its sub-05/06 files are
  committed and validated.
- **Lead candidate (M10):** whitened decoders on CLIP B-32 **+ DINOv2-B patch-mean**, equal weights:
  **15.2 / 45.3 / 7.25** on dev vs CLIP alone 14.7 / 42.2 (179 better / 140 worse, p = 0.0024). Weights are being
  chosen by train-CV with krabiki-51's N-member ensemble tool (pending). This is on branch `track-dinov2`, not yet
  on track-proven.
- **Not an addition anymore (M9):** the CLIP encoder M5. With the whitened decoder, CV picks w = 1.0 (decoder alone).
  Equal weight on dev gives 13.2 / 41.7 vs 14.7 / 42.2 (p = 0.46).
- **No single command exists** for features → train → predict. Today it takes 2 commands plus a manual install
  (shown below), and `predict.py`'s defaults are *not* the final pipeline.
- **Rule compliance is clean** for every adopted component: training-only statistics, labels read only after the file
  is written, one code path for all six subjects. CSLS is borderline-transductive, but it's off and dropped. Hungarian
  isn't implemented.
- **Biggest gaps, by value per effort:** (1) README (empty), (2) one-command final run with the right defaults,
  (3) **saving our model weights**, which rule 6 requires and which we don't do at all, (4) tests for the submission
  writer and the decoding / matching code paths. Details in §5.

---

## 1. Component audit

Legend for **State**: ✅ in code, CLI flag, one command | ⚠️ in code but not exposed / not one command | ❌ not in code.
**Tests** refers to `tests/test_gpu.py` (20 parity tests). Runtimes were measured on this machine with the GPU and
an 8-thread cap.

| ID | Component | Dev top-1 / top-5 / medR | State and exact command | Tests | Robustness | Rules | Cost | Verdict |
|---|---|---|---|---|---|---|---|---|
| M0 | AlexNet encoding (organizers' recipe) | 2.0 / 8.6 / 35.1 | ⚠️ `extract_features.py --model alexnet --frames 8`, then `predict.py --features alexnet --frames 8 --whiten`. **The numbers aren't reproducible with current code**: they used the old 17-alpha grid (0.1–1e7) and 3 s crop, and `DEFAULT_ALPHAS` is now 1–1e9 | – | n/a | ✅ | 2 × 617 MB caches; PCA on 140k dims (CPU, minutes) | **Report only** (rerun once with current code for the table) |
| M1 | CLIP encoding, whitened (bug) | 4.2 / 16.9 / 25.5 | ✅ `predict.py --features clip_b32 --frames 8 --max-seconds 1.5 --alpha-selection retrieval --whiten` (regression-checked exactly) | whiten path covered | n/a | ✅ | ~5 s/subj | **Report only** (the "whitening flaw" story) |
| M2 | CLIP encoding, unwhitened | 5.4 / 21.1 / 17.5 | ✅ `predict.py --features clip_b32 --frames 8 --max-seconds 1.5 --alpha-selection retrieval` | CV + ridge parity | Spearman(CV, dev medR) = 0.68 over 60 configs; at its ceiling | ✅ | ~5 s/subj | **Report only**; superseded by M5 as the encoder |
| M3 | CLIP decoding (plain) | 14.0 / 36.0 / 8.25; seeds 13.8 ± 0.3 / 35.9 ± 0.1 | ✅ `predict.py --head decoding --features clip_b32 --frames 8 --max-seconds 1.5` | gpu.decoding_cv_scores **not** parity-tested; DecodingModel untested | Better than M2 on all 4 subjects (p = 9e-17); CV agrees; shuffle control = chance | ✅ | 9 s/subj | **Fallback / report baseline** (superseded by M8a) |
| M4 | Ensemble M2 + M3, equal weights | 10.5 / 37.0 / 9 | ❌ dev diagnostic only, no code in repo | – | Weights not chosen by CV | ✅ if weights come from train-OOF | – | **Drop** (M7 supersedes) |
| M5 | Encoding + noise-whitened Euclidean matching | 8.8 / 31.6 / 11.6 | ✅ `predict.py --features clip_b32 --frames 8 --max-seconds 1.5 --alpha-selection retrieval --match noise --noise-cov per_time --noise-similarity euclid`. sub-05/06 files **not committed** on track-proven | fit_predict parity only; `Matcher` / `ledoit_wolf_covariance` not tested in repo (e1 checked them independently against sklearn LedoitWolf) | p = 2.9e-7 vs M2; top-5 up on all 4 subjects. **sub-01 top-1 drops (7.8 → 5.9) and CV prefers plain corr on sub-01**. Re-tuning alpha under Euclidean matching is a null result (M9a) | ✅ training residuals only | 7.5 s/subj | **Report only** (best encoder; no longer an ensemble member, see M9) |
| M6 | Frames / pooling (8/16/32, onset, concat) | no significant change | ⚠️ `--pooling` flag ✅; the all-frames cache has no CLI on track-proven, but **a CLI exists on `track-dinov2` (`3da7561`)**: `extract_features.py --model <m> --all-frames --frames 8` | – | All p ≥ 0.06; CLIP is nearly constant within a clip | ✅ | 5 min extraction | **Report only**; keep `--pooling mean` (default) |
| M7 | Ensemble M5 + M3, equal weights | 12.0 / 41.7 / 8.0 | ❌ dev diagnostic only | – | Top-5 +5.7 but top-1 −2 vs M3. Weights not chosen by CV | ✅ if weights come from train-OOF | – | **Superseded by M9** |
| M9 | CV-weighted ensemble M8a + M5 (krabiki-51) | CV picks w = 1.0 (decoder alone); equal weight 13.2 / 41.7 / 7.8 | ⚠️ `scripts/ensemble.py` on `track-matching` (not merged) | – | CV and dev agree: the encoder adds nothing once the decoder input is whitened (p = 0.46). The per-subject blend gives p = 0.013 but changes only sub-01, within noise | ✅ OOF similarity matrices, training only | small | **Drop the encoder member**; the ensemble tool is reused for M10 |
| M10 | Whitened decoders on DINOv2-B (krabiki-e1) | patch-mean alone 13.0 / 41.7 / 7.5 (CV 0.224, better than CLIP on 5 of 6 subjects); **CLIP + patch, equal weights: 15.2 / 45.3 / 7.25** | ⚠️ on `track-dinov2` @ `3da7561` (not merged): `extract_features.py --model dinov2_b --all-frames --frames 8`, then `predict.py --head decoding --dec-whiten --features dinov2_b_patch_all --frames 8 --max-seconds 1.5`. Ensemble weights: pending (krabiki-51 N-member tool) | ? (not audited on that branch) | Ensemble vs CLIP decw: 179 / 140, p = 0.0024. Weights not yet chosen by CV. Patch-alone vs CLIP is equivalent (p = 0.86). CLS token is weaker (10.8 / 34.3) | ✅ public model (timm `vit_base_patch14_dinov2.lvd142m`, 2023) | **24 min** all-frames extraction on GPU, plus `timm`; about 20 s/subject per decoder | **Final pipeline candidate** (pending CV weights and a paired dev check) |
| M8a | **Decoding, noise-whitened input** | **14.7 / 42.2 / 7.5**; seeds **14.4 ± 0.4 / 41.7 ± 0.4 / 7.5 ± 0.0** | ✅ `predict.py --head decoding --dec-whiten --features clip_b32 --frames 8 --max-seconds 1.5` → `submissions/decw_clip_b32_f8_t1.5s_w5-60/` (committed, validated) | **None** for `decoding.py` | Train-CV better on **all 6** subjects (0.255 → 0.232); dev top-5 up on all 4 subjects; top-1 down on sub-04 (11.8 → 10.8); p = 0.002 vs M3; reproduced independently by e1 (sklearn/scipy), shuffle control = chance | ✅ whiteners from training OOF residuals; nested inside CV folds | 20 s/subj | **Final pipeline (primary head)** |
| M8b | Decoding, reliability-weighted cosine | 14.0 / 39.7 / 8.25 (CV pick p=1) | ⚠️ `DecodingVariant(reliability_power=…)` and `select_decoding.py`; no `predict.py` flag | – | **CV/dev disagree**: CV prefers it on all 6 subjects, but vs M8a alone dev gives p = 0.71 (p0.5) and 0.09 trending worse (p1) | ✅ | +inner CV | **Drop**; report as the CV-overfitting example |
| #1 | ERP subtraction (matching) | 4.9 / 20.6 / 16.8 | ✅ `--erp-subtract` | erp CV path tested | p = 0.87; cancels exactly under Euclidean matching | ✅ | – | **Drop** (keep the flag as an ablation) |
| #2a | Reliability-weighted correlation (matching) | 4.7 / 21.8 / 16.6 | ✅ `--match weighted --match-power p` | – | CV gets worse with p; dev flat (p = 0.81) | ✅ | – | **Drop** |
| – | Noise matching with corr / pooled cov | CV 0.302 / 0.306 / 0.297 | ✅ `--noise-similarity corr`, `--noise-cov pooled` | – | Worse than per-timepoint Euclidean by CV | ✅ | – | Report only |
| – | Per-channel z-scoring | no-op | ✅ `--zscore` | – | EEG is already normalized (std ≈ 0.47 on every channel) | ✅ training stats | – | **Drop** |
| – | CSLS | 6.1 / 22.3 / 17.1 on M2 | ✅ `--csls-k k` (off) | – | Fewer hubs, not significant (p = 0.36) | ⚠️ **transductive**: uses column statistics of the 102 × 102 test similarity matrix, i.e. all test recordings jointly | – | **Drop** (dropped by the user) |
| – | Hungarian / Sinkhorn assignment | – | ❌ | – | – | ⚠️ exploits the one-to-one test structure; needs organizer approval | – | Don't implement unless approved |
| – | Matching window 50–600 ms, crop 1.5 s | – | ✅ defaults `--window 5 60`; `--max-seconds` default is **None (3 s)** | – | Window chosen by CV (encoding sweep, decoder CV) | ✅ | – | Final pipeline. Note that the crop is irrelevant for every retrieval-CV head (everything is window-restricted) |
| – | Alpha by retrieval CV | – | ✅ for decoding always; for encoding `--alpha-selection retrieval` (**default is still `gcv`**) | ✅ parity | GCV ≈ retrieval CV once unwhitened | ✅ | – | Final pipeline |

### Notes behind the table

- **Fold-seed robustness (new, for brief §7).** The only randomness in the linear pipeline is the KFold split, which
  drives alpha selection and, for M8a, the residual folds of the whiteners. I re-ran M3 and M8a with seeds 0/1/2 on
  sub-01..04. M3 gives 13.8 ± 0.3 / 35.9 ± 0.1 / 8.46 ± 0.36 (alphas 56–562). M8a gives 14.4 ± 0.4 / 41.7 ± 0.4 /
  7.50 ± 0.00 (alphas 1000–3162). The M8a gain is far larger than the seed noise. This came from an ad-hoc script
  and isn't in the repo yet (see gap G6).
- **GPU vs CPU determinism.** GPU float32 results are reproducible run to run (regression checks matched exactly). The
  CPU fallback agrees within float32 tolerance in the tests, but it could flip near-tied ranks, so the final numbers
  should be produced on the GPU or the device recorded.
- **Median-rank convention.** MODEL_BACKLOG M3 shows "9" (pooled median), while PROGRESS / predict.py report 8.2
  (mean of per-subject medians). Pick one for the report.

---

## 2. Rule compliance (ClaudeREADME §3)

| Rule | Status | Evidence |
|---|---|---|
| 1. Python 3.11 | ✅ | `.venv` = CPython 3.11.17; `pyproject.toml` requires `>=3.11,<3.12` |
| 2. Public pretrained models released before 2026-10-16, name and version recorded | ✅ / ⚠️ | open_clip `ViT-B-32 / laion2b_s34b_b79k` (2022), torchvision AlexNet `IMAGENET1K_V1`. Recorded in the `features.py` docstring and MODEL_BACKLOG, **not in a README**. DINOv2 (2023) will need the same record |
| 3. Never train on test data | ✅ | Every fit (scalers, PCA, ridge, whiteners, reliabilities) uses `load_train` data only. Test EEG enters only `similarity()` / `predict()` |
| 4. Dev test labels only for validation / model selection | ✅ | `load_test_labels` is called **after** `write_submission` in `predict.py` and is used only for scoring. Design choices (window, head, rejecting M8b) used dev results as model selection, which is allowed. Nothing is used as a template |
| 5. Unsupervised normalization of test recordings / video features | ✅ | We're stricter than allowed: test CLIP features are standardized with *training-video* statistics, and test EEG with training statistics |
| 6. Submit code, pipeline, **our own model weights**, predictions | ❌ **weights missing** | Models are refit on every run and nothing is saved. Ridge weights are tiny (decoder 935 × 512 + 55 whiteners of 17 × 17 per subject), so this is easy to fix (G3) |
| Same code path for sub-01..06 | ✅ | `run_subject` is identical for all subjects. The only branch is scoring after the file is written |
| Transductive / borderline | ⚠️ CSLS (off), Hungarian (absent) | No adopted component uses statistics of the test set |

---

## 3. Repo vs brief code-quality requirements (§6, §7, §9)

| Requirement | Status | Detail |
|---|---|---|
| §6 deliver value after every stage (working `predict.py`, valid files) | ✅ | Validated sub-05/06 files are committed for M2, M3 and M8a. `write_submission` asserts shape, dtype and permutation, and dev files are reloaded and re-scored |
| §6 Stage 0 self-test (write sub-01, reload, score) | ⚠️ | Done ad hoc and on every dev run inside `predict.py`, but there's **no test file** for `submission.py` / `evaluate.py` |
| §7 seeds fixed, mean ± std over 3 seeds | ⚠️ | KFold seed 0 is fixed. Fold-seed variance was measured for M3 and M8a (above) but isn't scripted, and there's no `--seed` flag |
| §7 results table with config per row | ⚠️ | `results/results.csv` has one row per subject and run, but the config (head, crop, match, whiten) lives only inside the `experiment` name. Many analyses (paired Wilcoxon, hubness, intra-clip similarity, seed check) came from ad-hoc scripts outside the repo |
| §9 `README.md`: install, train, predict, reproduce | ❌ | `README.md` is `# krabiki` only |
| §9 `requirements.txt` pinned | ✅ / ⚠️ | 154 pins from `pip freeze`, plus the cu128 extra index. It includes unused packages (braindecode, mne, aiohttp, ...). `pip install -e .` is required but undocumented. The global pip mirror on this laptop is unreachable (local issue; `.venv/pip.conf` works around it) |
| §9 `configs/` (YAML per experiment) | ❌ | Configuration is CLI flags only (fully specified, but there's no config file for the final run) |
| §9 layout `src/eegmr/{data,features,linear,contrastive,retrieval,evaluate,submission}` | ✅ / ⚠️ | All present except `contrastive.py` (Stage 3 not started). Extra modules: `gpu`, `matching`, `decoding` |
| §9 `scripts/{extract_features,train,predict,analyze}.py` | ⚠️ | `extract_features` ✅, `predict` ✅ (trains and predicts in one go). No `train.py` / `analyze.py`. Extra scripts: `sweep_linear`, `select_matching`, `compare_matching`, `select_decoding` |
| §9 `weights/` committed | ❌ | See rule 6 |
| §9 `submissions/` committed | ✅ | `.gitignore` allow-lists three canonical folders |
| §9 every path configurable, data root default | ✅ / ⚠️ | `--root`, `--feature-dir`, `--out`, `--results` exist. `results/similarity/` is hard-coded in `predict.py` |
| §9 one command each for extract / train / predict | ⚠️ | Extract ✅ and predict ✅ (2 commands). There's no single "from scratch" entry point, and `predict.py` defaults (`--head encoding`, `--alpha-selection gcv`, `--max-seconds` None) aren't the final pipeline |
| §9 type hints, docstrings, no notebook-only logic | ✅ / minor | Every function has return hints. A few public functions lack docstrings (`DecodingVariant.fit/similarity`, `EncodingModel.predict`, `gpu.as_tensor`, script `main`s). The `predict.py` module docstring still says "linear encoding model". No notebooks |
| Tests | ⚠️ | 20 parity tests (gpu ridge / CV / correlation, erp CV, matching.fit_predict) on CPU and CUDA. **Untested:** submission writer, evaluate, `decoding.py` (M8a, the final head), `Matcher` noise path, `ledoit_wolf_covariance` vs sklearn, and an end-to-end smoke run |
| Hygiene | minor | `testfile.txt`, `KrabikiFilestructure.txt` (276 KB) and the root `FINDINGS_*.md` could move to `docs/` or go |

### Is there ONE command from scratch to the final sub-05/06 files?

**No.** Today, after installing the environment by hand (uv/venv with Python 3.11, `pip install -r requirements.txt`,
`pip install -e .`), it takes two commands, about 5 minutes on this GPU:

```bash
python scripts/extract_features.py --model clip_b32 --frames 8                                       # ~3 min (GPU)
python scripts/predict.py --head decoding --dec-whiten --features clip_b32 --frames 8 --max-seconds 1.5   # ~2 min, 6 subjects
# -> submissions/decw_clip_b32_f8_t1.5s_w5-60/sub-0{5,6}_ranked_candidate_list.npy (validated; dev scored)
```

An ensemble final can't be produced by any command yet.

---

## 4. Recommended final pipeline

**Features (per decoder member):** open_clip ViT-B-32 / laion2b_s34b_b79k, 8 frames, mean (`clip_b32`), and, pending
CV weights, DINOv2-B ViT-B/14 (timm `vit_base_patch14_dinov2.lvd142m`), mean of the patch tokens over 8 frames
(`dinov2_b_patch_all`, from the all-frames cache).

**EEG:** crop 0–1.5 s at 500 Hz → FIR zero-phase decimation to 100 Hz → window 50–600 ms (samples 5:60). No z-scoring.

**Heads:** N whitened decoders (M8a recipe), one per feature space. Each member: EEG window × per-timepoint
Σ_t^-1/2 (Ledoit-Wolf on that feature space's encoding OOF residuals, encoding alpha by retrieval CV) → standardize →
ridge → standardized features, cosine to the candidates. Decoder alpha by nested train-CV. Members are combined as a
weighted sum of row-z-scored similarity matrices, with weights chosen by train-CV on out-of-fold similarities
(krabiki-51's N-member tool, **pending**). Adopt the ensemble only if the paired dev test against CLIP-decw alone
doesn't contradict CV (the M8b lesson). Otherwise the fallback is CLIP-decw alone (M8a).

**Off:** the CLIP encoder member (M9), CSLS, Hungarian, reliability weights (decoding and matching), ERP subtraction,
z-scoring, frame pooling variants.

**Flags for today's final (M8a alone):**
`--head decoding --dec-whiten --features clip_b32 --frames 8 --max-seconds 1.5 --window 5 60`

**Flags for the DINOv2 member (from `track-dinov2`):**
`--head decoding --dec-whiten --features dinov2_b_patch_all --frames 8 --max-seconds 1.5`. The ensemble command and
its weights come from krabiki-51's tool and aren't fixed yet.

---

## 5. Gaps ranked by value per effort, and TODO

| # | Gap | Value | Effort | Why |
|---|---|---|---|---|
| G1 | **README**: install (uv + Python 3.11, `pip install -r` + `-e .`), the 2 commands, expected outputs and runtime, model names and versions, how dev numbers are scored | high | low | 10% of the grade is code quality and reproducibility, and the README is currently empty |
| G2 | **One-command final run** for **N decoder members plus weights** (for example `scripts/run_final.py --config configs/final.yaml` listing members, as feature name, frames, head flags and weight). It should extract any missing features (including all-frames DINOv2), fit each member per subject, save each member's similarity matrices, combine them with the CV-chosen weights, validate and copy sub-05/06 to `submissions/final/`, and print dev scores per member and for the ensemble. It degrades to a single member (CLIP-decw) | high | low–medium | Removes flag-mistake risk at submission time. The single-head design is no longer enough (M10) |
| G3 | **Save model weights** per subject (`weights/<experiment>/sub-XX.npz`: ridge coefficients, scaler / PCA, whiteners, alpha) and an option to predict from them | high (rule 6) | low–medium | Required by the brief, and small |
| G4 | **Tests**: Stage 0 self-test (write → reload → score; oracle = 100%, random ≈ chance), `decoding.py` parity (whitened decoder vs a sklearn/scipy reference, as e1 did), `Matcher` noise path vs sklearn LedoitWolf, one-subject end-to-end smoke test | medium–high | low–medium | The final head currently has zero tests |
| G5 | ~~Ensemble M5 + M8a~~ **resolved (M9): CV picks the decoder alone.** The open part is the N-decoder ensemble (CLIP + DINOv2-patch) with CV weights, wired into G2 | high (top-5 +3.1, p = 0.0024 at equal weights) | medium | **Pending (krabiki-51's N-member tool)**, not duplicated here |
| G5b | Merge `track-dinov2` (DINOv2 extractor, all-frames CLI, `timm` pin) and `track-matching`'s `ensemble.py` into track-proven, then add a DINOv2 cache to the parity and smoke tests | high | low–medium | Prerequisite for a one-command final with DINOv2 |
| G6 | `--seed` flag plus a scripted 3-seed report (mean ± std) for the final heads | medium (§7) | low | Numbers already exist (§1 notes) and only need scripting |
| G7 | `scripts/analyze.py` for report figures (§8: time-resolved decoding, channel importance, feature-space comparison, failure cases, subject variability) and for the ad-hoc paired tests | medium (presentation and novelty) | medium | Today several published numbers can't be regenerated from the repo |
| G8 | `configs/final.yaml` (and a `--config` loader) | low–medium (§9) | medium | Flags already specify everything; mainly for compliance with the layout |
| G9 | Rerun M0 / M1 with current code for the report's baseline rows | low–medium | low | The current M0 numbers come from an older alpha grid |
| G10 | Hygiene: prune `requirements.txt` to what's used, fix the `predict.py` docstring, missing docstrings, move or remove `testfile.txt` / `KrabikiFilestructure.txt` / root findings files, add config columns (head, match, crop, device) to `results.csv` | low | low | Polish |

### Ordered TODO to make it submission-ready

**On hold (2026-10-18):** don't start G1–G4 until the user approves (krabiki-e1 has asked).

1. **G5b:** merge `track-dinov2` and `ensemble.py` into track-proven.
2. **G2 + G1:** one-command final run for N decoder members plus weights (`configs/final.yaml` covers G8 too), and a
   README describing it. It starts as CLIP-decw alone, a valid submission at every point, and switches to the
   CLIP + DINOv2 ensemble once the CV weights (G5) and the paired dev check are in.
3. **G3:** save and load per-subject weights for every member, and commit `weights/`.
4. **G4:** tests for the submission writer, `decoding.py`, the ensemble combiner and the `Matcher` noise path, plus an
   end-to-end smoke test.
5. **G6, G7:** seed report (now per member and for the ensemble) and the analysis/figures script.
6. **G9, G10:** baseline reruns, hygiene.
