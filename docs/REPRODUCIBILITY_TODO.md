# Reproducibility: verified and still to test (status 2026-10-18, 22:50)

## Verified (branch `submission`)
- `inference/predict.py` with `inference/model/` reproduces `submissions/final_v4b_csls` sub-05/06 byte-for-byte; dev 33.8 / 68.1.
- `ensemble` stage from the v4b member caches: same weights, LOFO 0.28716, sub-05/06 bit-identical.
- `decw` stage rebuilt from scratch: all caches bit-identical to the v4b ones (sub-01..06).
- Pooled DINOv2 member, sub-01, rebuilt from committed code: bit-identical.
- Weight search on the GPU (`Evaluator.precompute`) gives exactly the same values as the old one.
- **sub-05 (scored) rebuilt end to end from the feature caches:** all 7 members (CV folds + test) rebuilt from scratch;
  13 of 14 caches bit-identical, pooled lowav within 6e-9 (BLAS thread count). The final blend with them gives the same
  weights / LOFO 0.28716 and `sub-05_ranked_candidate_list.npy` identical to `final_v4b_csls`.
- A fresh `git clone` of `submission` from GitHub runs `inference/predict.py` and reproduces sub-05/06 byte-for-byte.
- Features rebuilt from scratch (krabiki-74): `clip_b32_f8`, `vjepa2_l_all_f16`, `lowlevel_all_f1` bit-identical.

## Still to test
1. **`pool` stage:** rebuild all 3 pooled members for sub-01..04 and sub-06 (sub-05 done) (`run_pipeline.sh pool`) and compare to the v4b caches.
2. **Unpooled members (sub-05 done):** V-JEPA, contrastive (3 seeds), descriptors, CLAP for the other subjects. Run the `ensemble` stage from an empty cache and compare the member caches.
3. **End-to-end:** `bash training/run_pipeline.sh` (all stages) → `verify` must give sub-05/06 identical to `final_v4b_csls`.
4. **Features not yet rebuilt:** `alexnet_c5p6_f8`, `clap_all_f1`, and `dinov2_b_patch_all_f8` via the original all-frames path.
5. **`JOBS=3` parallel mode:** must give caches identical to `JOBS=1`. Known issue: parallel ensemble runs write the same `--cv-out` / `--results` log files, which harms only those logs, not the caches.
6. **`submission-fast` branch (faster feature extraction, not merged):**
   - DINOv2 (8-frame batches) and lowav are not bit-identical: max diffs 4.9e-4 and 9.5e-7, cosine 1.0. Swapping those test features into inference changed only 2 entries of sub-01.
   - Decide: either accept it and re-export the model, or keep the original path.
   - Time each extractor uncontended.
7. **Export:** rerun `run_pipeline.sh export` and check that `inference/model` is unchanged (`meta.json` hashes). Add `--source "v4b_ctr_m5 (final_v4b_csls)"` to the export stage.
8. **Fresh machine:** clone, `pip install -r training/requirements.txt` (Python 3.11, CUDA 12.8), run the pipeline, compare.
9. **Label-shuffle control on v4b:** `research/scripts/audit_runbook.sh` + `audit_controls.py --mode shuffle_pool`.
   - Untested in the new layout.
   - `pool_member.py`'s λ=0 assert against the real decw cache will fail under shuffled labels; skip it in that mode.
   - Use a fresh cache dir, never the production one.
10. **macOS / CPU:** `training/requirements-cpu.txt` was generated from the pinned file (CUDA packages dropped, `+cu128` removed) but not resolved on macOS; if a pin has no macOS wheel, install that package unpinned and record the version. Inference needs only `inference/requirements.txt`.
