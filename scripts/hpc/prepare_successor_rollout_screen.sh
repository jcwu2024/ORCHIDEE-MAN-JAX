#!/bin/bash
# Prepare frozen successor validation windows on test01-test04 (no GPU, no Slurm).

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:?set WORKTREE to the approved clean research worktree}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD to the approved commit}
MODE=${MODE:?set MODE to free_smoke or formal_bounded_screen}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR under the project runtime root}
PYTHON=$ROOT/.venvs/orcjax_cpu/bin/python

DATASET=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6/dataset_manifest.json
ACCEPTANCE=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/acceptance_report.json
BASE_STATISTICS=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/training_statistics.json
PROCESS_STATISTICS=$ROOT/runtime/outputs/training/first-neural-process-head-v1-statistics/process_training_statistics.json
FAST_STATISTICS=$ROOT/runtime/outputs/training/successor-fast-owner-statistics-c9f35c0/successor_fast_owner_statistics.json
SCREEN_CONTRACT=$WORKTREE/manifests/coarse_graining/successor_7_30_rollout_screen_v1.json

case "$(hostname -s)" in
  test01|test02|test03|test04|ibtest01|ibtest02|ibtest03|ibtest04) ;;
  *) echo "rollout-screen preparation must run on test01-test04" >&2; exit 2 ;;
esac
case "$WORKTREE" in
  "$ROOT"|"$ROOT/runtime/worktrees/"*) ;;
  *) echo "WORKTREE escaped the project-owned root" >&2; exit 2 ;;
esac
case "$OUTPUT_DIR" in
  "$ROOT/runtime/"*) ;;
  *) echo "OUTPUT_DIR escaped the project runtime root" >&2; exit 2 ;;
esac
case "$MODE" in
  free_smoke|formal_bounded_screen) ;;
  *) echo "invalid rollout-screen MODE" >&2; exit 2 ;;
esac
for path in "$DATASET" "$ACCEPTANCE" "$BASE_STATISTICS" \
  "$PROCESS_STATISTICS" "$FAST_STATISTICS" "$SCREEN_CONTRACT"; do
  test -f "$path"
done
test -x "$PYTHON"
test "$(cd "$WORKTREE" && git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
test -z "$(cd "$WORKTREE" && git status --porcelain --untracked-files=all)"
test ! -e "$OUTPUT_DIR/prepared_manifest.json"
test ! -e "$OUTPUT_DIR/window_plan.json"
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

/usr/bin/time -v "$PYTHON" -m scripts.hpc.prepare_successor_rollout_screen \
  --dataset-manifest "$DATASET" \
  --acceptance-report "$ACCEPTANCE" \
  --base-statistics "$BASE_STATISTICS" \
  --process-statistics "$PROCESS_STATISTICS" \
  --fast-statistics "$FAST_STATISTICS" \
  --screen-contract "$SCREEN_CONTRACT" \
  --mode "$MODE" \
  --output-dir "$OUTPUT_DIR"

test -f "$OUTPUT_DIR/prepared_manifest.json"
test -f "$OUTPUT_DIR/window_plan.json"
sha256sum "$OUTPUT_DIR/prepared_manifest.json" "$OUTPUT_DIR/window_plan.json"
