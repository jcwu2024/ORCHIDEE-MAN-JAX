#!/bin/bash
# Prepare the fixed 48-window milestone-validation panel on a free CPU test node.

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:?set WORKTREE to the clean staged-training commit}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD}
VALIDATION_PANEL=${VALIDATION_PANEL:?set VALIDATION_PANEL to frozen v3 panel JSON}
OUTPUT_DIR=${OUTPUT_DIR:?set OUTPUT_DIR below the project runtime root}
PYTHON=$ROOT/.venvs/orcjax_cpu/bin/python

case "$(hostname)" in
  test01|test02|test03|test04|ibtest01|ibtest02|ibtest03|ibtest04) ;;
  *) echo "staged validation preparation requires a free CPU test node" >&2; exit 2 ;;
esac
case "$OUTPUT_DIR" in "$ROOT/runtime/"*) ;; *) exit 2 ;; esac
test "$(cd "$WORKTREE" && git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
test -z "$(cd "$WORKTREE" && git status --porcelain --untracked-files=all)"
test ! -e "$OUTPUT_DIR"

export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONHASHSEED=0
export XDG_CACHE_HOME=$ROOT/runtime/cache/xdg
export ORCHIDEE_REPO_ROOT=$WORKTREE ORCHIDEE_RUNTIME_ROOT=$ROOT/runtime
export ORCHIDEE_DATA_ROOT=$ROOT/runtime/data ORCHIDEE_REFERENCE_ROOT=$ROOT/runtime/assets
export ORCHIDEE_OUTPUT_ROOT=$ROOT/runtime/outputs PYTHONPATH=$WORKTREE
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$WORKTREE"

"$PYTHON" -m scripts.hpc.prepare_successor_staged_validation \
  --dataset-manifest "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6/dataset_manifest.json" \
  --acceptance-report "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/acceptance_report.json" \
  --base-statistics "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/training_statistics.json" \
  --process-statistics "$ROOT/runtime/outputs/training/first-neural-process-head-v1-statistics/process_training_statistics.json" \
  --fast-statistics "$ROOT/runtime/outputs/training/successor-fast-owner-statistics-c9f35c0/successor_fast_owner_statistics.json" \
  --runner-contract "$WORKTREE/manifests/coarse_graining/successor_training_staged_v3.json" \
  --validation-panel "$VALIDATION_PANEL" \
  --output-dir "$OUTPUT_DIR"
