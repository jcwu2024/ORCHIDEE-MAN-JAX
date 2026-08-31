#!/bin/bash
# Prepare the fresh structured-GPP panel on a free CPU test node.

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:-$ROOT}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD}
HEAD_SHORT=${EXPECTED_GIT_HEAD:0:7}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/runtime/outputs/training/structured-gpp-fresh-validation-${HEAD_SHORT}}
PYTHON=$ROOT/.venvs/orcjax_cpu/bin/python

case "$(hostname)" in
  test01|test02|test03|test04|ibtest01|ibtest02|ibtest03|ibtest04) ;;
  *) echo "fresh validation preparation requires a free CPU test node" >&2; exit 2 ;;
esac
case "$WORKTREE" in "$ROOT"|"$ROOT/runtime/worktrees/"*) ;; *) exit 2 ;; esac
case "$OUTPUT_DIR" in "$ROOT/runtime/outputs/"*) ;; *) exit 2 ;; esac
test -x "$PYTHON"
test "$(cd "$WORKTREE" && git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
test -z "$(cd "$WORKTREE" && git status --porcelain --untracked-files=all)"

DATASET=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6/dataset_manifest.json
ACCEPTANCE=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/acceptance_report.json
BASE_STATISTICS=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/training_statistics.json
PROCESS_STATISTICS=$ROOT/runtime/outputs/training/first-neural-process-head-v1-statistics/process_training_statistics.json
FAST_STATISTICS=$ROOT/runtime/outputs/training/successor-fast-owner-statistics-c9f35c0/successor_fast_owner_statistics.json
TRAINING_CONTRACT=$WORKTREE/manifests/coarse_graining/target_oriented_bounded_training_v1.json
SCREEN_CONTRACT=$WORKTREE/manifests/coarse_graining/target_oriented_structured_gpp_fresh_screen_v1.json
CONSUMED_PANEL=$ROOT/runtime/outputs/training/target-oriented-validation-9b82b5a/validation_panel.json
for path in "$DATASET" "$ACCEPTANCE" "$BASE_STATISTICS" \
  "$PROCESS_STATISTICS" "$FAST_STATISTICS" "$TRAINING_CONTRACT" \
  "$SCREEN_CONTRACT" "$CONSUMED_PANEL"; do
  test -f "$path"
done

mkdir -p "$OUTPUT_DIR" "$ROOT/runtime/cache/xdg"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONHASHSEED=0
export XDG_CACHE_HOME=$ROOT/runtime/cache/xdg
export ORCHIDEE_REPO_ROOT=$WORKTREE ORCHIDEE_RUNTIME_ROOT=$ROOT/runtime
export ORCHIDEE_DATA_ROOT=$ROOT/runtime/data ORCHIDEE_REFERENCE_ROOT=$ROOT/runtime/assets
export ORCHIDEE_OUTPUT_ROOT=$ROOT/runtime/outputs PYTHONPATH=$WORKTREE
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$WORKTREE"

/usr/bin/time -v "$PYTHON" -m scripts.hpc.prepare_target_oriented_structured_gpp_fresh_validation \
  --dataset-manifest "$DATASET" \
  --acceptance-report "$ACCEPTANCE" \
  --base-statistics "$BASE_STATISTICS" \
  --process-statistics "$PROCESS_STATISTICS" \
  --fast-statistics "$FAST_STATISTICS" \
  --training-contract "$TRAINING_CONTRACT" \
  --screen-contract "$SCREEN_CONTRACT" \
  --consumed-validation-panel "$CONSUMED_PANEL" \
  --output-dir "$OUTPUT_DIR"

test -f "$OUTPUT_DIR/prepared_manifest.json"
sha256sum "$OUTPUT_DIR/prepared_manifest.json" "$OUTPUT_DIR/validation_panel.json"
