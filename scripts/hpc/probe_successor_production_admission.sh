#!/bin/bash
# Verify all exact production-fit identities on a free CPU test node.

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:?set WORKTREE to the approved clean research worktree}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD to the approved commit}
OUTPUT=${OUTPUT:?set a new output path under the project runtime root}
PYTHON=$ROOT/.venvs/orcjax_cpu/bin/python

DATASET=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6/dataset_manifest.json
ACCEPTANCE=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/acceptance_report.json
BASE_STATISTICS=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/training_statistics.json
PROCESS_STATISTICS=$ROOT/runtime/outputs/training/first-neural-process-head-v1-statistics/process_training_statistics.json
FAST_STATISTICS=$ROOT/runtime/outputs/training/successor-fast-owner-statistics-c9f35c0/successor_fast_owner_statistics.json
PREPARED_MANIFEST=$ROOT/runtime/outputs/training/successor-small-fit-prepared-500u-d493699/prepared_manifest.json
COEFFICIENT_SOURCE_REPORT=$ROOT/runtime/outputs/training/successor-small-fit-hashseed0-cold-2u-1c63494/runner_report.json
RUNNER_CONTRACT=$WORKTREE/manifests/coarse_graining/successor_small_fit_runner_v1.json
CHECKPOINT_HANDOFF=$WORKTREE/manifests/coarse_graining/successor_small_fit_checkpoint_handoff_v1.json
COEFFICIENT_CONTRACT=$WORKTREE/manifests/coarse_graining/successor_small_fit_frozen_coefficients_v1.json

case "$(hostname -s)" in
  test01|test02|test03|test04) ;;
  *) echo "production admission must run on test01-test04" >&2; exit 2 ;;
esac
case "$WORKTREE" in
  "$ROOT"|"$ROOT/runtime/worktrees/"*) ;;
  *) echo "WORKTREE escaped the project-owned root" >&2; exit 2 ;;
esac
case "$OUTPUT" in
  "$ROOT/runtime/"*) ;;
  *) echo "OUTPUT escaped the project runtime root" >&2; exit 2 ;;
esac
for path in "$DATASET" "$ACCEPTANCE" "$BASE_STATISTICS" \
  "$PROCESS_STATISTICS" "$FAST_STATISTICS" "$PREPARED_MANIFEST" \
  "$COEFFICIENT_SOURCE_REPORT" "$RUNNER_CONTRACT" \
  "$CHECKPOINT_HANDOFF" "$COEFFICIENT_CONTRACT"; do
  test -f "$path"
done
test -x "$PYTHON"
test "$(cd "$WORKTREE" && git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
test -z "$(cd "$WORKTREE" && git status --porcelain --untracked-files=all)"
test ! -e "$OUTPUT"
mkdir -p "$(dirname "$OUTPUT")" "$ROOT/runtime/cache/xdg"

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

/usr/bin/time -v "$PYTHON" -m scripts.hpc.probe_successor_production_admission \
  --dataset-manifest "$DATASET" \
  --acceptance-report "$ACCEPTANCE" \
  --base-statistics "$BASE_STATISTICS" \
  --process-statistics "$PROCESS_STATISTICS" \
  --fast-statistics "$FAST_STATISTICS" \
  --runner-contract "$RUNNER_CONTRACT" \
  --checkpoint-handoff "$CHECKPOINT_HANDOFF" \
  --prepared-manifest "$PREPARED_MANIFEST" \
  --coefficient-contract "$COEFFICIENT_CONTRACT" \
  --coefficient-source-report "$COEFFICIENT_SOURCE_REPORT" \
  --output "$OUTPUT"

test -f "$OUTPUT"
sha256sum "$OUTPUT"
