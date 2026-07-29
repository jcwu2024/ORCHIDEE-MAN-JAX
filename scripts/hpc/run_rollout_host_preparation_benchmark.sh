#!/bin/bash
# Run a bounded multi-landpoint host-preparation benchmark on free gln01.

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:?set WORKTREE to the approved research worktree}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD}
DATASET=${DATASET:?set DATASET to the admitted dataset manifest}
STATISTICS=${STATISTICS:?set STATISTICS to the admitted statistics JSON}
OUTPUT=${OUTPUT:?set OUTPUT to a new benchmark report}
GPU_DEVICE=${GPU_DEVICE:-0}
LANDPOINTS=${LANDPOINTS:-16}
HORIZON=${HORIZON:-7}
ANCHOR_BATCH_SIZE=${ANCHOR_BATCH_SIZE:-256}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-32}
SEED=${SEED:-20260728}
PYTHON=$ROOT/.venvs/orcjax_gpu/bin/python
IMAGE=/apps/soft/sif/foundationpose

case "$WORKTREE" in
  "$ROOT/runtime/worktrees/"*) ;;
  *) echo "WORKTREE must stay under $ROOT/runtime/worktrees" >&2; exit 2 ;;
esac
for path in "$DATASET" "$STATISTICS" "$OUTPUT"; do
  case "$path" in
    "$ROOT/runtime/"*) ;;
    *) echo "runtime inputs and outputs must stay under $ROOT/runtime" >&2; exit 2 ;;
  esac
done

test -f "$DATASET"
test -f "$STATISTICS"
test -x "$PYTHON"
test "$(cd "$WORKTREE" && git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
test -z "$(cd "$WORKTREE" && git status --porcelain --untracked-files=all)"
test ! -e "$OUTPUT"
mkdir -p "$(dirname "$OUTPUT")" "$ROOT/runtime/cache/jax/orcjax_gpu/rollout-host-benchmark"

CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
test -f "$CUDA_DRIVER"
test -f "$NVML_DRIVER"
GPU_BINDS="/WORK:/WORK,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"

env \
  SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs \
  SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU_DEVICE" \
  SINGULARITYENV_JAX_ENABLE_X64=true \
  SINGULARITYENV_JAX_COMPILATION_CACHE_DIR="$ROOT/runtime/cache/jax/orcjax_gpu/rollout-host-benchmark" \
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
    "$PYTHON" -m scripts.hpc.benchmark_rollout_host_preparation_gpu \
      --dataset "$DATASET" \
      --statistics "$STATISTICS" \
      --output "$OUTPUT" \
      --landpoints "$LANDPOINTS" \
      --horizon "$HORIZON" \
      --anchor-batch-size "$ANCHOR_BATCH_SIZE" \
      --rollout-batch-size "$ROLLOUT_BATCH_SIZE" \
      --seed "$SEED"
