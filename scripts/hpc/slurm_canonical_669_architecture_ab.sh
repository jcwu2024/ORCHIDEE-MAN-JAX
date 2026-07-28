#!/bin/bash
# Matched two-GPU architecture screen on the admitted 669-point dataset.
# No job is submitted by this file.
#SBATCH -J orcjax-669-arch-ab
#SBATCH -p gnall
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH -o /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/canonical_669_architecture_ab_%j.txt
#SBATCH -e /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/logs/canonical_669_architecture_ab_%j.txt
#SBATCH --no-requeue

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:?set WORKTREE to the approved research worktree}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD}
DATASET=${DATASET:?set DATASET to the admitted dataset manifest}
STATISTICS=${STATISTICS:?set STATISTICS to the admitted statistics JSON}
ACCEPTANCE=${ACCEPTANCE:?set ACCEPTANCE to the admitted acceptance report}
OUTPUT_ROOT=${OUTPUT_ROOT:?set OUTPUT_ROOT to a new A/B output directory}
PROTOCOL=$WORKTREE/manifests/coarse_graining/daily_teacher_669_training_protocol.json
EXPERIMENT=$WORKTREE/manifests/coarse_graining/canonical_669_axis_process_architecture_ab.json
PYTHON=$ROOT/.venvs/orcjax_gpu/bin/python
IMAGE=/apps/soft/sif/foundationpose

case "$WORKTREE" in
  "$ROOT/runtime/worktrees/"*) ;;
  *) echo "WORKTREE must stay under $ROOT/runtime/worktrees" >&2; exit 2 ;;
esac
for path in "$DATASET" "$STATISTICS" "$ACCEPTANCE" "$OUTPUT_ROOT"; do
  case "$path" in
    "$ROOT/runtime/"*) ;;
    *) echo "runtime inputs and outputs must stay under $ROOT/runtime" >&2; exit 2 ;;
  esac
done

test -f "$DATASET"
test -f "$STATISTICS"
test -f "$ACCEPTANCE"
test -f "$PROTOCOL"
test -f "$EXPERIMENT"
test -x "$PYTHON"
test "$(cd "$WORKTREE" && git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
test -z "$(cd "$WORKTREE" && git status --porcelain --untracked-files=all)"
if [[ -e "$OUTPUT_ROOT" && ! -d "$OUTPUT_ROOT" ]]; then
  echo "OUTPUT_ROOT exists and is not a directory: $OUTPUT_ROOT" >&2
  exit 2
fi
mkdir -p "$OUTPUT_ROOT" "$ROOT/runtime/logs" "$ROOT/runtime/cache/jax/orcjax_gpu"

CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
test -f "$CUDA_DRIVER"
test -f "$NVML_DRIVER"
GPU_BINDS="/WORK:/WORK,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"

run_python() {
  local visible_device=$1
  local cache_tag=$2
  shift 2
  mkdir -p "$ROOT/runtime/cache/jax/orcjax_gpu/$cache_tag"
  env \
    SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs \
    SINGULARITYENV_CUDA_VISIBLE_DEVICES="$visible_device" \
    SINGULARITYENV_JAX_ENABLE_X64=true \
    SINGULARITYENV_JAX_COMPILATION_CACHE_DIR="$ROOT/runtime/cache/jax/orcjax_gpu/$cache_tag" \
    SINGULARITYENV_ORCHIDEE_REPO_ROOT="$ROOT" \
    SINGULARITYENV_ORCHIDEE_RUNTIME_ROOT="$ROOT/runtime" \
    SINGULARITYENV_ORCHIDEE_DATA_ROOT="$ROOT/runtime/data" \
    SINGULARITYENV_ORCHIDEE_REFERENCE_ROOT="$ROOT/runtime/assets" \
    SINGULARITYENV_ORCHIDEE_OUTPUT_ROOT="$ROOT/runtime/outputs" \
    singularity exec \
      --nv \
      --bind "$GPU_BINDS" \
      --pwd "$WORKTREE" \
      "$IMAGE" \
      "$@"
}

COMMON_ARGS=(
  --experiment "$EXPERIMENT"
  --dataset "$DATASET"
  --statistics "$STATISTICS"
  --acceptance "$ACCEPTANCE"
  --protocol "$PROTOCOL"
  --output-root "$OUTPUT_ROOT"
)
PREFLIGHT="$OUTPUT_ROOT/architecture_ab_preflight.json"

run_python 0 preflight \
  "$PYTHON" -m research.daily_coarse_graining.canonical_architecture_ab_run \
  --phase prepare "${COMMON_ARGS[@]}"
test -f "$PREFLIGHT"

RUN_STARTED=$(date -u +%Y-%m-%dT%H:%M:%SZ)
printf '\n=== architecture arm attempt %s ===\n' "$RUN_STARTED" \
  >>"$OUTPUT_ROOT/flat_worker.log"
printf '\n=== architecture arm attempt %s ===\n' "$RUN_STARTED" \
  >>"$OUTPUT_ROOT/axis_process_worker.log"

run_python 0 flat \
  "$PYTHON" -m scripts.hpc.run_canonical_architecture_ab_gpu \
  --phase arm --arm flat --preflight "$PREFLIGHT" "${COMMON_ARGS[@]}" \
  >>"$OUTPUT_ROOT/flat_worker.log" 2>&1 &
FLAT_PID=$!
run_python 1 axis_process \
  "$PYTHON" -m scripts.hpc.run_canonical_architecture_ab_gpu \
  --phase arm --arm axis_process --preflight "$PREFLIGHT" "${COMMON_ARGS[@]}" \
  >>"$OUTPUT_ROOT/axis_process_worker.log" 2>&1 &
AXIS_PID=$!

set +e
wait "$FLAT_PID"
FLAT_STATUS=$?
wait "$AXIS_PID"
AXIS_STATUS=$?
set -e
if [[ "$FLAT_STATUS" -ne 0 || "$AXIS_STATUS" -ne 0 ]]; then
  echo "architecture workers failed: flat=$FLAT_STATUS axis_process=$AXIS_STATUS" >&2
  exit 1
fi

run_python 0 finalize \
  "$PYTHON" -m research.daily_coarse_graining.canonical_architecture_ab_run \
  --phase finalize --preflight "$PREFLIGHT" "${COMMON_ARGS[@]}"

test -f "$OUTPUT_ROOT/flat/training_report.json"
test -f "$OUTPUT_ROOT/flat/best_checkpoint.pkl"
test -f "$OUTPUT_ROOT/axis_process/training_report.json"
test -f "$OUTPUT_ROOT/axis_process/best_checkpoint.pkl"
test -f "$OUTPUT_ROOT/architecture_ab_report.json"
