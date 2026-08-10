#!/bin/bash
# Preassemble a bounded successor schedule on a CPU test or compute node.

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:?set WORKTREE to the approved clean research worktree}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD to the approved commit}
UPDATES=${UPDATES:?set UPDATES between 2 and 500}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR under the project runtime root}
WORKER_INDEX=${WORKER_INDEX:-}
WORKER_COUNT=${WORKER_COUNT:-}
FINALIZE=${FINALIZE:-false}
PYTHON=$ROOT/.venvs/orcjax_cpu/bin/python

DATASET=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6/dataset_manifest.json
ACCEPTANCE=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/acceptance_report.json
BASE_STATISTICS=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/training_statistics.json
PROCESS_STATISTICS=$ROOT/runtime/outputs/training/first-neural-process-head-v1-statistics/process_training_statistics.json
FAST_STATISTICS=$ROOT/runtime/outputs/training/successor-fast-owner-statistics-c9f35c0/successor_fast_owner_statistics.json
RUNNER_CONTRACT=$WORKTREE/manifests/coarse_graining/successor_small_fit_runner_v1.json

case "$WORKTREE" in
  "$ROOT"|"$ROOT/runtime/worktrees/"*) ;;
  *) echo "WORKTREE escaped the project-owned root" >&2; exit 2 ;;
esac
case "$OUTPUT_DIR" in
  "$ROOT/runtime/"*) ;;
  *) echo "OUTPUT_DIR escaped the project runtime root" >&2; exit 2 ;;
esac
case "$UPDATES" in
  ''|*[!0-9]*) echo "UPDATES must be an integer" >&2; exit 2 ;;
esac
if [ "$UPDATES" -lt 2 ] || [ "$UPDATES" -gt 500 ]; then
  echo "UPDATES must lie in [2, 500]" >&2
  exit 2
fi
for path in "$DATASET" "$ACCEPTANCE" "$BASE_STATISTICS" \
  "$PROCESS_STATISTICS" "$FAST_STATISTICS" "$RUNNER_CONTRACT"; do
  test -f "$path"
done
test -x "$PYTHON"
test "$(cd "$WORKTREE" && git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
test -z "$(cd "$WORKTREE" && git status --porcelain --untracked-files=all)"

mkdir -p "$OUTPUT_DIR" "$ROOT/runtime/cache/xdg"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=true
export PYTHONHASHSEED=0
export XDG_CACHE_HOME=$ROOT/runtime/cache/xdg
export ORCHIDEE_REPO_ROOT=$WORKTREE
export ORCHIDEE_RUNTIME_ROOT=$ROOT/runtime
export ORCHIDEE_DATA_ROOT=$ROOT/runtime/data
export ORCHIDEE_REFERENCE_ROOT=$ROOT/runtime/assets
export ORCHIDEE_OUTPUT_ROOT=$ROOT/runtime/outputs
export PYTHONPATH=$WORKTREE
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
cd "$WORKTREE"

EXTRA_ARGS=()
if [ "$FINALIZE" = true ]; then
  test -n "$WORKER_COUNT"
  test -z "$WORKER_INDEX"
  EXTRA_ARGS+=(--finalize --worker-count "$WORKER_COUNT")
elif [ -n "$WORKER_INDEX" ] || [ -n "$WORKER_COUNT" ]; then
  test -n "$WORKER_INDEX"
  test -n "$WORKER_COUNT"
  EXTRA_ARGS+=(--worker-index "$WORKER_INDEX" --worker-count "$WORKER_COUNT")
fi

/usr/bin/time -v "$PYTHON" -m scripts.hpc.prepare_successor_small_fit_batches \
  --dataset-manifest "$DATASET" \
  --acceptance-report "$ACCEPTANCE" \
  --base-statistics "$BASE_STATISTICS" \
  --process-statistics "$PROCESS_STATISTICS" \
  --fast-statistics "$FAST_STATISTICS" \
  --runner-contract "$RUNNER_CONTRACT" \
  --updates "$UPDATES" \
  --output-dir "$OUTPUT_DIR" \
  "${EXTRA_ARGS[@]}"

if [ "$FINALIZE" = true ] || [ -z "$WORKER_INDEX" ]; then
  test -f "$OUTPUT_DIR/prepared_manifest.json"
  sha256sum "$OUTPUT_DIR/prepared_manifest.json"
else
  PARTIAL=$OUTPUT_DIR/partials/worker-$(printf '%04d' "$WORKER_INDEX").json
  test -f "$PARTIAL"
  sha256sum "$PARTIAL"
fi
