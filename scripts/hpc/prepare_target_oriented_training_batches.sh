#!/bin/bash
# Prepare target-oriented training batches on a free CPU test node.

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:-$ROOT}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD}
UPDATES=${UPDATES:-8}
START_UPDATE=${START_UPDATE:-1}
END_UPDATE=${END_UPDATE:-$UPDATES}
HEAD_SHORT=${EXPECTED_GIT_HEAD:0:7}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/runtime/outputs/training/target-oriented-prepared-${UPDATES}-${HEAD_SHORT}/chunks/${START_UPDATE}-${END_UPDATE}}
REUSE_RETAINED_INPUTS_MANIFEST=${REUSE_RETAINED_INPUTS_MANIFEST:-}
LOG_PATH=${LOG_PATH:-}
BACKGROUND=${BACKGROUND:-0}
BACKGROUND_CHILD=${BACKGROUND_CHILD:-0}
PYTHON=$ROOT/.venvs/orcjax_cpu/bin/python

if [ "$BACKGROUND" = 1 ] && [ "$BACKGROUND_CHILD" = 0 ]; then
  BACKGROUND_CHILD=1 nohup bash "$0" </dev/null >/dev/null 2>&1 &
  echo "target-oriented preparation background_pid=$!"
  exit 0
fi

if [ -n "$LOG_PATH" ]; then
  case "$LOG_PATH" in "$ROOT/runtime/logs/"*) ;; *) exit 2 ;; esac
  exec >>"$LOG_PATH" 2>&1
fi

case "$(hostname)" in
  test01|test02|test03|test04|ibtest01|ibtest02|ibtest03|ibtest04) ;;
  *) echo "target-oriented preparation requires a free CPU test node" >&2; exit 2 ;;
esac
case "$WORKTREE" in "$ROOT"|"$ROOT/runtime/worktrees/"*) ;; *) exit 2 ;; esac
case "$OUTPUT_DIR" in "$ROOT/runtime/outputs/"*) ;; *) exit 2 ;; esac
case "$UPDATES" in 8|250|1000|3000) ;; *) echo "unsupported target-oriented preparation milestone" >&2; exit 2 ;; esac
test -x "$PYTHON"
test "$(cd "$WORKTREE" && git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
test -z "$(cd "$WORKTREE" && git status --porcelain --untracked-files=all)"

DATASET=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6/dataset_manifest.json
ACCEPTANCE=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/acceptance_report.json
BASE_STATISTICS=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/training_statistics.json
PROCESS_STATISTICS=$ROOT/runtime/outputs/training/first-neural-process-head-v1-statistics/process_training_statistics.json
FAST_STATISTICS=$ROOT/runtime/outputs/training/successor-fast-owner-statistics-c9f35c0/successor_fast_owner_statistics.json
TRAINING_CONTRACT=$WORKTREE/manifests/coarse_graining/target_oriented_bounded_training_v1.json
for path in "$DATASET" "$ACCEPTANCE" "$BASE_STATISTICS" \
  "$PROCESS_STATISTICS" "$FAST_STATISTICS" "$TRAINING_CONTRACT"; do
  test -f "$path"
done

PREPARE_ARGUMENTS=(
  --dataset-manifest "$DATASET"
  --acceptance-report "$ACCEPTANCE"
  --base-statistics "$BASE_STATISTICS"
  --process-statistics "$PROCESS_STATISTICS"
  --fast-statistics "$FAST_STATISTICS"
  --training-contract "$TRAINING_CONTRACT"
  --updates "$UPDATES"
  --start-update "$START_UPDATE"
  --end-update "$END_UPDATE"
)
if [ -n "$REUSE_RETAINED_INPUTS_MANIFEST" ]; then
  case "$REUSE_RETAINED_INPUTS_MANIFEST" in "$ROOT/runtime/outputs/"*) ;; *) exit 2 ;; esac
  test -f "$REUSE_RETAINED_INPUTS_MANIFEST"
  PREPARE_ARGUMENTS+=(--reuse-retained-inputs-manifest "$REUSE_RETAINED_INPUTS_MANIFEST")
fi
PREPARE_ARGUMENTS+=(--output-dir "$OUTPUT_DIR")

mkdir -p "$OUTPUT_DIR" "$ROOT/runtime/cache/xdg"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONHASHSEED=0
export XDG_CACHE_HOME=$ROOT/runtime/cache/xdg
export ORCHIDEE_REPO_ROOT=$WORKTREE ORCHIDEE_RUNTIME_ROOT=$ROOT/runtime
export ORCHIDEE_DATA_ROOT=$ROOT/runtime/data ORCHIDEE_REFERENCE_ROOT=$ROOT/runtime/assets
export ORCHIDEE_OUTPUT_ROOT=$ROOT/runtime/outputs PYTHONPATH=$WORKTREE
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$WORKTREE"

/usr/bin/time -v "$PYTHON" -m scripts.hpc.prepare_target_oriented_training_batches \
  "${PREPARE_ARGUMENTS[@]}"

test -f "$OUTPUT_DIR/prepared_manifest.json"
sha256sum "$OUTPUT_DIR/prepared_manifest.json"
