#!/bin/bash
#SBATCH --job-name=orcjax-okleak-outer
#SBATCH --partition=cnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --exclude=ibc11b04n04
#SBATCH --time=01:00:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/ok_leak_outer_%j.txt
#SBATCH --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/ok_leak_outer_%j.txt
#SBATCH --no-requeue

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
PYTHON=$ROOT/.venvs/orcjax_cpu/bin/python
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set the approved Git commit}
OUTPUT=${OUTPUT:?set an output JSON under runtime/outputs}
DATASET_ROOT=$ROOT/runtime/outputs/training/pft14-daily-teacher-9point-1961-2010-v5-4c0f886
DATASET_MANIFEST=$DATASET_ROOT/dataset_manifest.json
TEACHER_PLAN=$ROOT/runtime/plans/teacher_initial_10point_v4_1f19ed7.json

case "$OUTPUT" in
  "$ROOT/runtime/outputs/"*.json) ;;
  *) echo "OUTPUT must be a JSON under $ROOT/runtime/outputs" >&2; exit 2 ;;
esac

test -x "$PYTHON"
test -f "$DATASET_MANIFEST"
test -f "$TEACHER_PLAN"
test "$(cd "$ROOT" && git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
test -z "$(cd "$ROOT" && git status --porcelain --untracked-files=no)"
mkdir -p "$(dirname "$OUTPUT")" "$ROOT/runtime/logs" "$ROOT/runtime/cache/xdg"

export PYTHONPATH=$ROOT
export ORCHIDEE_REPO_ROOT=$ROOT
export ORCHIDEE_RUNTIME_ROOT=$ROOT/runtime
export ORCHIDEE_DATA_ROOT=$ROOT/runtime/data
export ORCHIDEE_REFERENCE_ROOT=$ROOT/runtime/assets
export ORCHIDEE_OUTPUT_ROOT=$ROOT/runtime/outputs
export XDG_CACHE_HOME=$ROOT/runtime/cache/xdg
export JAX_COMPILATION_CACHE_DIR=$ROOT/runtime/cache/xla/ok_leak_outer_block_probe
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=True
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2

cd "$ROOT"
"$PYTHON" -m scripts.dev.probe_ok_leak_outer_block_reentry \
  --dataset-manifest "$DATASET_MANIFEST" \
  --plan "$TEACHER_PLAN" \
  --landpoint-id 069.0-119.0 \
  --year 1963 \
  --day-index 184 \
  --output "$OUTPUT"

test -f "$OUTPUT"
