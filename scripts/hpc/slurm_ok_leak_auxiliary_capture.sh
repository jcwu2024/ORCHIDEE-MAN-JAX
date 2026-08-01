#!/bin/bash
# CAPTURE_LIMIT=1 is the required first smoke; unset it only after smoke acceptance.
#SBATCH --job-name=orcjax-okleak-capture
#SBATCH --partition=cnall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --exclude=ibc11b04n04
#SBATCH --time=00:30:00
#SBATCH --output=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/ok_leak_capture_%j.txt
#SBATCH --error=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/ok_leak_capture_%j.txt
#SBATCH --no-requeue

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
PYTHON=$ROOT/.venvs/orcjax_cpu/bin/python
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set the approved Git commit}
OUTPUT_DIR=${OUTPUT_DIR:?set an isolated output under runtime/outputs}
CAPTURE_PLAN=$ROOT/manifests/coarse_graining/ok_leak_auxiliary_capture_96day_v1.json
DATASET_ROOT=$ROOT/runtime/outputs/training/pft14-daily-teacher-9point-1961-2010-v5-4c0f886
DATASET_MANIFEST=$DATASET_ROOT/dataset_manifest.json
TEACHER_PLAN=$ROOT/runtime/plans/teacher_initial_10point_v4_1f19ed7.json
CACHE=$ROOT/runtime/cache/xla/ok_leak_auxiliary_capture

case "$OUTPUT_DIR" in
  "$ROOT/runtime/outputs/"*) ;;
  *) echo "OUTPUT_DIR must stay under $ROOT/runtime/outputs" >&2; exit 2 ;;
esac

test -x "$PYTHON"
test -f "$CAPTURE_PLAN"
test -f "$DATASET_MANIFEST"
test -f "$TEACHER_PLAN"
test "$(cd "$ROOT" && git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
test -z "$(cd "$ROOT" && git status --porcelain --untracked-files=no)"
mkdir -p "$OUTPUT_DIR" "$CACHE" "$ROOT/runtime/logs" "$ROOT/runtime/cache/xdg"

export PYTHONPATH=$ROOT
export ORCHIDEE_REPO_ROOT=$ROOT
export ORCHIDEE_RUNTIME_ROOT=$ROOT/runtime
export ORCHIDEE_DATA_ROOT=$ROOT/runtime/data
export ORCHIDEE_REFERENCE_ROOT=$ROOT/runtime/assets
export ORCHIDEE_OUTPUT_ROOT=$ROOT/runtime/outputs
export XDG_CACHE_HOME=$ROOT/runtime/cache/xdg
export JAX_COMPILATION_CACHE_DIR=$CACHE
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=True
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2

ARGS=(
  --capture-plan "$CAPTURE_PLAN"
  --dataset-manifest "$DATASET_MANIFEST"
  --dataset-root "$DATASET_ROOT"
  --teacher-plan "$TEACHER_PLAN"
  --output "$OUTPUT_DIR"
)
if [[ -n "${CAPTURE_LIMIT:-}" ]]; then
  ARGS+=(--limit "$CAPTURE_LIMIT")
fi

cd "$ROOT"
"$PYTHON" -m scripts.dev.capture_ok_leak_driver_batch "${ARGS[@]}"

test -f "$OUTPUT_DIR/capture_manifest.json"
test -f "$OUTPUT_DIR/plan_verification.json"
