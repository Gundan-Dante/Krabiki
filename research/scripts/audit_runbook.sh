#!/usr/bin/env bash
# Audit + model-export runbook for a CV-weighted ensemble produced by training/scripts/ensemble.py (here or in another worktree).
# Fill in the variables (from the producing session's report), then run:  bash research/scripts/audit_runbook.sh <step>
# Steps: setup | rescore | rerun | permute | compare | shuffle | bundle | infer | package   (or "all")
#
# Layout assumed (the `submission` branch): this script and audit_controls.py / audit_compare.py live in research/scripts/,
# the pipeline in training/{src,scripts}, the numpy-only predictor in inference/predict.py. The audited COMMIT may use the
# old layout (src/ and scripts/ at its root) or the new one (training/); both are found.
set -euo pipefail

: "${COMMIT:?commit that produced the submission}"            # e.g. efb37e0
: "${SRC_TREE:?worktree holding its results/ and submissions/}" # e.g. /home/kparpie/Krabiki-matching
: "${EXP:?ensemble --experiment name}"                         # e.g. v4b_ctr_m5
: "${FINAL_DIR:?submission dir with the final sub-05/06}"      # e.g. final_v4b_csls
: "${MEMBERS:?member specs, space separated}"                  # e.g. "pdecw:dinov2_b_patch_all decw:vjepa2_l_all:16 ..."
: "${WEIGHTS:?final weights, integers summing to 10}"          # e.g. "2 1 1 2 1 1 2"
FLAGS=${FLAGS:-"--max-seconds 1.5 --center-test --csls 5 --csls-stage member"}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)            # research/scripts
REPO=${REPO:-$(dirname "$(dirname "$HERE")")}                 # repository root (holds training/, inference/, submissions/)
DATA_ROOT=${DATA_ROOT:-$REPO/eeg-mind-reading-challenge}
FEATURES=${FEATURES:-$REPO/features}
AUD=${AUD:-$REPO/../Krabiki-audit-$COMMIT}                    # detached checkout of the audited commit
PY=${PY:-python}
OUT=${OUT:-$REPO/results/audit_$EXP}
BUNDLE=${BUNDLE:-$REPO/bundle/krabiki_${FINAL_DIR}_bundle}    # <BUNDLE>/model + predict.py (the zip of step_package)
DATA="--root $DATA_ROOT --feature-dir $FEATURES"
MEM_ARGS=$(for m in $MEMBERS; do printf -- "--member %s " "$m"; done)
CACHE=$SRC_TREE/results/similarity/_cache
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
mkdir -p "$OUT"

code_dir() { [ -d "$AUD/training/src" ] && echo "$AUD/training" || echo "$AUD"; }   # folder with src/ and scripts/
compare() { PYTHONPATH=$REPO/training/src $PY "$HERE/audit_compare.py" "$@"; }

step_setup() {
  [ -d "$AUD" ] || git -C "$REPO" worktree add --detach "$AUD" "$COMMIT"
  mkdir -p "$AUD/audit_out"
}
step_rescore() {  # saved blended matrices -> ranked lists == written submissions; dev scores
  compare --out "$OUT/rescore.json" rescore \
    --sim-dir "$SRC_TREE/results/similarity/${EXP}_cv" --sub-dir "$SRC_TREE/submissions/${EXP}_cv"
}
prep_cache() {  # copy the producer's cache; drop the test similarities of sub-01 / sub-05 so they are recomputed
  rm -rf "$AUD/audit_out/$1" && mkdir -p "$AUD/audit_out/$1" && cp "$CACHE"/*.npz "$AUD/audit_out/$1/"
  rm -f "$AUD/audit_out/$1"/sub-01_*_test.npz "$AUD/audit_out/$1"/sub-05_*_test.npz
}
step_rerun() {
  prep_cache cache_rerun
  (cd "$AUD" && $PY "$HERE/audit_controls.py" --code-dir "$(code_dir)" --mode rerun -- $MEM_ARGS $FLAGS $DATA \
     --sim-cache audit_out/cache_rerun --experiment audit_rerun --results audit_out/r.csv --cv-out audit_out/r_cv.csv) \
     > "$OUT/rerun.log" 2>&1
}
step_permute() {
  prep_cache cache_perm
  (cd "$AUD" && $PY "$HERE/audit_controls.py" --code-dir "$(code_dir)" --mode rerun --permute --subjects-recompute 1 5 \
     --perm-out audit_out/perms.json -- $MEM_ARGS $FLAGS $DATA --sim-cache audit_out/cache_perm \
     --experiment audit_perm --results audit_out/p.csv --cv-out audit_out/p_cv.csv) > "$OUT/perm.log" 2>&1
  cp "$AUD/audit_out/perms.json" "$OUT/"
}
step_compare() {
  compare --out "$OUT/determinism.json" determinism \
    --new-cache "$AUD/audit_out/cache_rerun" --ref-cache "$CACHE" --subjects 1 5 \
    --new-sub "$AUD/submissions/audit_rerun_cv" --ref-sub "$SRC_TREE/submissions/$FINAL_DIR"
  compare --out "$OUT/permutation.json" permutation \
    --perms "$AUD/audit_out/perms.json" --base-sub "$AUD/submissions/audit_rerun_cv" --perm-sub "$AUD/submissions/audit_perm_cv"
}
step_shuffle() {  # training pairs permuted, alphas re-chosen, the FINAL weights fixed (variant "ref"); dev subjects only
  # one process per dev subject in parallel (4 threads each); the fixed --ref-weights make per-subject runs equivalent
  for s in 1 2 3 4; do
    (cd "$AUD" && OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
       $PY "$HERE/audit_controls.py" --code-dir "$(code_dir)" --mode shuffle -- $MEM_ARGS $FLAGS $DATA \
       --ref-weights $WEIGHTS --subjects $s --experiment audit_shuffle_s$s --results audit_out/s$s.csv \
       --cv-out audit_out/s${s}_cv.csv) > "$OUT/shuffle_s$s.log" 2>&1 &
  done
  wait
  for s in 1 2 3 4; do
    compare --out "$OUT/shuffle_s$s.json" score --sub-dir "$AUD/submissions/audit_shuffle_s${s}_ref" --subjects $s
  done
}
step_bundle() {  # SUBJECT_WEIGHTS=<json {"1": [...], ...}> overrides WEIGHTS for per-subject strategies
  W_ARGS=${SUBJECT_WEIGHTS:+--subject-weights $SUBJECT_WEIGHTS}
  # export with the audited code (--code-dir) into <BUNDLE>/model; predict.py + requirements make the folder self-contained
  $PY "$REPO/training/scripts/export_bundle.py" --code-dir "$(code_dir)" --cache "$CACHE" --out "$BUNDLE/model" $MEM_ARGS \
    ${W_ARGS:---weights $WEIGHTS} $FLAGS --source "$COMMIT, $EXP ($FINAL_DIR)" $DATA > "$OUT/export.log" 2>&1
  cp "$REPO/inference/predict.py" "$REPO/inference/requirements.txt" "$BUNDLE/"
}
step_package() {  # model card + docs, then zip (run after step_infer has verified the bundle)
  for f in ${DOCS:-ARCHITECTURE.md IMPROVEMENTS.md EEGNET.md AUDIT.md PRESENTATION.md}; do
    [ -f "$REPO/docs/$f" ] && cp "$REPO/docs/$f" "$BUNDLE/"; done
  if [ -n "${VIZ:-}" ]; then  # optional presentation figures (a folder with figures/ and FIGURES.md)
    mkdir -p "$BUNDLE/schematics" && cp "$VIZ/FIGURES.md" "$BUNDLE/schematics/"
    for f in ${FIGS:-fig1_progress fig2_encoding_vs_decoding fig4_ensemble_anatomy fig5_expected_score fig7_recall_at_k}; do
      cp "$VIZ/figures/$f.png" "$BUNDLE/schematics/"; done
  fi
  $PY "$REPO/training/scripts/bundle_readme.py" --model "$BUNDLE/model" --title "$FINAL_DIR" --dev "${DEV_LINE:-}"
  (cd "$(dirname "$BUNDLE")" && rm -f "$(basename "$BUNDLE").zip" && zip -qr -9 "$(basename "$BUNDLE").zip" "$(basename "$BUNDLE")")
  ls -la "$BUNDLE.zip"
}
step_infer() {  # CPU only; must equal the committed final sub-05/06
  CUDA_VISIBLE_DEVICES="" $PY -I "$BUNDLE/predict.py" --model "$BUNDLE/model" --data-root "$DATA_ROOT" \
    --subjects 1 2 3 4 5 6 --out "$OUT/bundle_out" | tee "$OUT/infer.log"
  compare --out "$OUT/bundle_vs_final.json" determinism \
    --new-cache /nonexistent --ref-cache /nonexistent --subjects 5 6 --new-sub "$OUT/bundle_out" \
    --ref-sub "$SRC_TREE/submissions/$FINAL_DIR"
}

case "${1:-all}" in
  all) step_setup; step_rescore; step_rerun; step_permute; step_compare; step_shuffle; step_bundle; step_infer; step_package ;;
  *) "step_$1" ;;
esac
