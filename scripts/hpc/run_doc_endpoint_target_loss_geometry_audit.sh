#!/bin/bash
# Audit frozen DOC endpoint target/loss geometry on a free CPU test node.

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:-$ROOT}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD}
OUTPUT=${OUTPUT:?set OUTPUT below the project runtime root}
PYTHON=$ROOT/.venvs/orcjax_cpu/bin/python

case "$(hostname -s)" in
  test01|test02|test03|test04|ibtest01|ibtest02|ibtest03|ibtest04) ;;
  *) echo "DOC geometry audit requires a free CPU test node" >&2; exit 2 ;;
esac
case "$OUTPUT" in
  "$ROOT/runtime/outputs/"*) ;;
  *) echo "OUTPUT escaped the project runtime root" >&2; exit 2 ;;
esac
test -x "$PYTHON"
test "$(cd "$WORKTREE" && git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
test ! -e "$OUTPUT"
mkdir -p "$(dirname "$OUTPUT")"

export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONHASHSEED=0
export ORCHIDEE_REPO_ROOT=$WORKTREE
export ORCHIDEE_RUNTIME_ROOT=$ROOT/runtime
export ORCHIDEE_DATA_ROOT=$ROOT/runtime/data
export ORCHIDEE_REFERENCE_ROOT=$ROOT/runtime/assets
export ORCHIDEE_OUTPUT_ROOT=$ROOT/runtime/outputs
export PYTHONPATH=$WORKTREE

cd "$WORKTREE"
exec "$PYTHON" -m scripts.hpc.audit_doc_endpoint_target_loss_geometry \
  --dataset-manifest "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6/dataset_manifest.json" \
  --acceptance-report "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/acceptance_report.json" \
  --base-statistics "$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/training_statistics.json" \
  --process-statistics "$ROOT/runtime/outputs/training/first-neural-process-head-v1-statistics/process_training_statistics.json" \
  --fast-statistics "$ROOT/runtime/outputs/training/successor-fast-owner-statistics-c9f35c0/successor_fast_owner_statistics.json" \
  --output "$OUTPUT"
