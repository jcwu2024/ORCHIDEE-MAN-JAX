#!/bin/bash
# Diagnostic-only replay of a Teacher-era scientific source snapshot.
#SBATCH --job-name=orcjax-okleak-historical
#SBATCH --partition=cnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --exclude=ibc11b04n04
#SBATCH --time=01:00:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/ok_leak_historical_%j.txt
#SBATCH --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/ok_leak_historical_%j.txt
#SBATCH --no-requeue

set -euo pipefail

CANONICAL_ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CHECKOUT_ROOT=${CHECKOUT_ROOT:?set the isolated historical checkout}
PYTHON=$CANONICAL_ROOT/.venvs/orcjax_cpu/bin/python
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set the historical diagnostic commit}
OUTPUT=${OUTPUT:?set an output JSON under canonical runtime/outputs}
DATASET_ROOT=$CANONICAL_ROOT/runtime/outputs/training/pft14-daily-teacher-9point-1961-2010-v5-4c0f886
DATASET_MANIFEST=$DATASET_ROOT/dataset_manifest.json
TEACHER_PLAN=$CANONICAL_ROOT/runtime/plans/teacher_initial_10point_v4_1f19ed7.json

case "$CHECKOUT_ROOT" in
  "$CANONICAL_ROOT/runtime/worktrees/"*) ;;
  *) echo "CHECKOUT_ROOT must stay under canonical runtime/worktrees" >&2; exit 2 ;;
esac
case "$OUTPUT" in
  "$CANONICAL_ROOT/runtime/outputs/"*.json) ;;
  *) echo "OUTPUT must be a JSON under canonical runtime/outputs" >&2; exit 2 ;;
esac

test -x "$PYTHON"
test -f "$DATASET_MANIFEST"
test -f "$TEACHER_PLAN"
test "$(cd "$CHECKOUT_ROOT" && git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
test -z "$(cd "$CHECKOUT_ROOT" && git status --porcelain --untracked-files=no)"
mkdir -p "$(dirname "$OUTPUT")" "$CANONICAL_ROOT/runtime/logs" \
  "$CANONICAL_ROOT/runtime/cache/xdg"

export PYTHONPATH=$CHECKOUT_ROOT
export ORCHIDEE_REPO_ROOT=$CHECKOUT_ROOT
export ORCHIDEE_RUNTIME_ROOT=$CANONICAL_ROOT/runtime
export ORCHIDEE_DATA_ROOT=$CANONICAL_ROOT/runtime/data
export ORCHIDEE_REFERENCE_ROOT=$CANONICAL_ROOT/runtime/assets
export ORCHIDEE_OUTPUT_ROOT=$CANONICAL_ROOT/runtime/outputs
export XDG_CACHE_HOME=$CANONICAL_ROOT/runtime/cache/xdg
export JAX_COMPILATION_CACHE_DIR=$CANONICAL_ROOT/runtime/cache/xla/ok_leak_historical_outer_block_probe
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=True
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2

cd "$CHECKOUT_ROOT"
"$PYTHON" -m scripts.dev.probe_ok_leak_outer_block_reentry \
  --dataset-manifest "$DATASET_MANIFEST" \
  --plan "$TEACHER_PLAN" \
  --landpoint-id 069.0-119.0 \
  --year 1963 \
  --day-index 184 \
  --block-days 7 \
  --doc-sqrt-mode production \
  --output "$OUTPUT"

test -f "$OUTPUT"
