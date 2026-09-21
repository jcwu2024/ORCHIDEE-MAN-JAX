#!/bin/bash
# Freeze or prepare the 48-window panel for the current fresh method.

set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi
if (( $# != 0 )); then
  echo "usage: $0 [--dry-run]" >&2
  exit 2
fi

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:-$ROOT}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/runtime/outputs/training/fresh-method-validation-20260921}
CONSUMED_ROOT=${CONSUMED_ROOT:-$ROOT/runtime/outputs/training}
SELECTION_ONLY=${SELECTION_ONLY:-0}
PYTHON=$ROOT/.venvs/orcjax_cpu/bin/python

case "$WORKTREE" in "$ROOT"|"$ROOT/runtime/worktrees/"*|"$ROOT/runtime/checkouts/"*) ;; *) exit 2 ;; esac
case "$OUTPUT_DIR" in "$ROOT/runtime/outputs/"*) ;; *) exit 2 ;; esac
case "$CONSUMED_ROOT" in "$ROOT/runtime/outputs/"*) ;; *) exit 2 ;; esac
test -x "$PYTHON"

DATASET=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6/dataset_manifest.json
ACCEPTANCE=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/acceptance_report.json
BASE_STATISTICS=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/training_statistics.json
PROCESS_STATISTICS=$ROOT/runtime/outputs/training/first-neural-process-head-v1-statistics/process_training_statistics.json
FAST_STATISTICS=$ROOT/runtime/outputs/training/successor-fast-owner-statistics-c9f35c0/successor_fast_owner_statistics.json
TRAINING_CONTRACT=$WORKTREE/manifests/coarse_graining/target_oriented_bounded_training_v1.json

for path in "$DATASET" "$CONSUMED_ROOT/target-oriented-validation-9b82b5a/validation_panel.json" \
  "$CONSUMED_ROOT/structured-gpp-fresh-validation-c3ea51a/validation_panel.json" \
  "$CONSUMED_ROOT/successor-staged-validation-a2c2638/validation_panel.json" \
  "$CONSUMED_ROOT/conditioned-independent-validation-20260909/validation_panel.json"; do
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

args=(
  --dataset-manifest "$DATASET"
  --consumed-panel-root "$CONSUMED_ROOT"
  --output-dir "$OUTPUT_DIR"
)
if [[ "$SELECTION_ONLY" == 1 ]]; then
  args+=(--selection-only)
else
  args+=(
    --acceptance-report "$ACCEPTANCE"
    --base-statistics "$BASE_STATISTICS"
    --process-statistics "$PROCESS_STATISTICS"
    --fast-statistics "$FAST_STATISTICS"
    --training-contract "$TRAINING_CONTRACT"
  )
fi

if (( DRY_RUN )); then
  printf 'hostname=%s\nworktree=%s\n' "$(hostname -s)" "$WORKTREE"
  printf 'python=%q -m scripts.hpc.prepare_fresh_method_validation' "$PYTHON"
  printf ' %q' "${args[@]}"
  printf '\n'
  exit 0
fi

case "$(hostname -s)" in
  test01|test02|test03|test04|ibtest01|ibtest02|ibtest03|ibtest04) ;;
  *) echo "fresh-method validation preparation requires a free CPU test node" >&2; exit 2 ;;
esac

exec timeout --signal=TERM 60m "$PYTHON" -m scripts.hpc.prepare_fresh_method_validation "${args[@]}"
