#!/bin/bash
# Run or exactly resume the frozen matched Experiment B screen.

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:?set WORKTREE to the approved research worktree}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD}
DATASET=${DATASET:?set DATASET to the admitted dataset manifest}
STATISTICS=${STATISTICS:?set STATISTICS to the admitted statistics JSON}
OUTPUT_ROOT=${OUTPUT_ROOT:?set OUTPUT_ROOT to the immutable experiment root}
GPU_DEVICE=${GPU_DEVICE:-0}
PYTHON=$ROOT/.venvs/orcjax_gpu/bin/python
IMAGE=/apps/soft/sif/foundationpose
PROTOCOL=$WORKTREE/manifests/coarse_graining/canonical_669_rollout_stability_protocol.json
PREFLIGHT=$OUTPUT_ROOT/rollout_stability_preflight.json
CALIBRATION=$OUTPUT_ROOT/coefficient_calibration.json
EXECUTION=$OUTPUT_ROOT/rollout_stability_execution.json

case "$WORKTREE" in
  "$ROOT/runtime/worktrees/"*) ;;
  *) echo "WORKTREE must stay under $ROOT/runtime/worktrees" >&2; exit 2 ;;
esac
for path in "$DATASET" "$STATISTICS" "$OUTPUT_ROOT"; do
  case "$path" in
    "$ROOT/runtime/"*) ;;
    *) echo "runtime inputs and outputs must stay under $ROOT/runtime" >&2; exit 2 ;;
  esac
done

test -x "$PYTHON"
test -f "$DATASET"
test -f "$STATISTICS"
test -f "$PROTOCOL"
test -f "$PREFLIGHT"
test "$(cd "$WORKTREE" && git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
test -z "$(cd "$WORKTREE" && git status --porcelain --untracked-files=all)"
mkdir -p "$OUTPUT_ROOT" "$ROOT/runtime/cache/jax/orcjax_gpu/rollout-stability-experiment"

CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
test -f "$CUDA_DRIVER"
test -f "$NVML_DRIVER"
GPU_BINDS="/WORK:/WORK,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"

run_gpu() {
  env \
    SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs \
    SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU_DEVICE" \
    SINGULARITYENV_JAX_ENABLE_X64=true \
    SINGULARITYENV_JAX_PLATFORMS=cuda \
    SINGULARITYENV_JAX_COMPILATION_CACHE_DIR="$ROOT/runtime/cache/jax/orcjax_gpu/rollout-stability-experiment" \
    SINGULARITYENV_ORCHIDEE_REPO_ROOT="$ROOT" \
    SINGULARITYENV_ORCHIDEE_RUNTIME_ROOT="$ROOT/runtime" \
    SINGULARITYENV_ORCHIDEE_DATA_ROOT="$ROOT/runtime/data" \
    SINGULARITYENV_ORCHIDEE_REFERENCE_ROOT="$ROOT/runtime/assets" \
    SINGULARITYENV_ORCHIDEE_OUTPUT_ROOT="$ROOT/runtime/outputs" \
    SINGULARITYENV_PYTHONPATH="$WORKTREE" \
    singularity exec \
      --nv \
      --bind "$GPU_BINDS" \
      --pwd "$WORKTREE" \
      "$IMAGE" \
      "$PYTHON" -m scripts.hpc.run_rollout_stability_gpu "$@"
}

if [[ ! -f "$CALIBRATION" ]]; then
  run_gpu \
    --phase calibrate \
    --protocol "$PROTOCOL" \
    --dataset "$DATASET" \
    --statistics "$STATISTICS" \
    --output-root "$OUTPUT_ROOT" \
    --preflight "$PREFLIGHT"
fi

if [[ ! -f "$EXECUTION" ]]; then
  run_gpu \
    --phase manifest \
    --protocol "$PROTOCOL" \
    --dataset "$DATASET" \
    --statistics "$STATISTICS" \
    --output-root "$OUTPUT_ROOT" \
    --preflight "$PREFLIGHT" \
    --calibration "$CALIBRATION"
fi

for arm in one_step_continuation_control mixed_horizon_stability_v1; do
  run_gpu \
    --phase arm \
    --protocol "$PROTOCOL" \
    --dataset "$DATASET" \
    --statistics "$STATISTICS" \
    --output-root "$OUTPUT_ROOT" \
    --execution "$EXECUTION" \
    --arm "$arm"
done
