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

CANONICAL_ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CHECKOUT_ROOT=${CHECKOUT_ROOT:-$CANONICAL_ROOT}
PYTHON=$CANONICAL_ROOT/.venvs/orcjax_cpu/bin/python
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set the approved Git commit}
OUTPUT_DIR=${OUTPUT_DIR:?set an isolated output under runtime/outputs}
CAPTURE_PLAN=$CHECKOUT_ROOT/manifests/coarse_graining/ok_leak_auxiliary_capture_96day_v1.json
DATASET_ROOT=$CANONICAL_ROOT/runtime/outputs/training/pft14-daily-teacher-9point-1961-2010-v5-4c0f886
DATASET_MANIFEST=$DATASET_ROOT/dataset_manifest.json
TEACHER_PLAN=$CANONICAL_ROOT/runtime/plans/teacher_initial_10point_v4_1f19ed7.json
CACHE_NAMESPACE=${CACHE_NAMESPACE:-ok_leak_auxiliary_capture}
CACHE=$CANONICAL_ROOT/runtime/cache/xla/$CACHE_NAMESPACE

case "$CHECKOUT_ROOT" in
  "$CANONICAL_ROOT"|"$CANONICAL_ROOT/runtime/worktrees/"*) ;;
  *) echo "CHECKOUT_ROOT must be the canonical checkout or its runtime/worktrees child" >&2; exit 2 ;;
esac
case "$OUTPUT_DIR" in
  "$CANONICAL_ROOT/runtime/outputs/"*) ;;
  *) echo "OUTPUT_DIR must stay under $CANONICAL_ROOT/runtime/outputs" >&2; exit 2 ;;
esac

test -x "$PYTHON"
test -f "$CAPTURE_PLAN"
test -f "$DATASET_MANIFEST"
test -f "$TEACHER_PLAN"
test "$(cd "$CHECKOUT_ROOT" && git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
test -z "$(cd "$CHECKOUT_ROOT" && git status --porcelain --untracked-files=no)"
mkdir -p "$OUTPUT_DIR" "$CACHE" "$CANONICAL_ROOT/runtime/logs" \
  "$CANONICAL_ROOT/runtime/cache/xdg"

export PYTHONPATH=$CHECKOUT_ROOT
export ORCHIDEE_REPO_ROOT=$CHECKOUT_ROOT
export ORCHIDEE_RUNTIME_ROOT=$CANONICAL_ROOT/runtime
export ORCHIDEE_DATA_ROOT=$CANONICAL_ROOT/runtime/data
export ORCHIDEE_REFERENCE_ROOT=$CANONICAL_ROOT/runtime/assets
export ORCHIDEE_OUTPUT_ROOT=$CANONICAL_ROOT/runtime/outputs
export XDG_CACHE_HOME=$CANONICAL_ROOT/runtime/cache/xdg
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

cd "$CHECKOUT_ROOT"
"$PYTHON" -m scripts.dev.capture_ok_leak_driver_batch "${ARGS[@]}"

test -f "$OUTPUT_DIR/capture_manifest.json"
test -f "$OUTPUT_DIR/plan_verification.json"
