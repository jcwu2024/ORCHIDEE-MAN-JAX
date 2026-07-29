#!/bin/bash
# Run the accepted real-shard rollout-stability smoke shapes on free gln01.

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:?set WORKTREE to the approved research worktree}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD}
DATASET=${DATASET:?set DATASET to the admitted dataset manifest}
STATISTICS=${STATISTICS:?set STATISTICS to the admitted statistics JSON}
PARENT_BEST_CHECKPOINT=${PARENT_BEST_CHECKPOINT:?set PARENT_BEST_CHECKPOINT}
PARENT_OPTIMIZER_CHECKPOINT=${PARENT_OPTIMIZER_CHECKPOINT:?set PARENT_OPTIMIZER_CHECKPOINT}
OUTPUT_ROOT=${OUTPUT_ROOT:?set OUTPUT_ROOT to a new smoke output directory}
GPU_DEVICE=${GPU_DEVICE:-0}
HORIZONS=${HORIZONS:-"1 3 7"}
PYTHON=$ROOT/.venvs/orcjax_gpu/bin/python
IMAGE=/apps/soft/sif/foundationpose

case "$WORKTREE" in
  "$ROOT/runtime/worktrees/"*) ;;
  *) echo "WORKTREE must stay under $ROOT/runtime/worktrees" >&2; exit 2 ;;
esac
for path in \
  "$DATASET" \
  "$STATISTICS" \
  "$PARENT_BEST_CHECKPOINT" \
  "$PARENT_OPTIMIZER_CHECKPOINT" \
  "$OUTPUT_ROOT"; do
  case "$path" in
    "$ROOT/runtime/"*) ;;
    *) echo "runtime inputs and outputs must stay under $ROOT/runtime" >&2; exit 2 ;;
  esac
done

test -f "$DATASET"
test -f "$STATISTICS"
test -f "$PARENT_BEST_CHECKPOINT"
test -f "$PARENT_OPTIMIZER_CHECKPOINT"
test -x "$PYTHON"
test "$(cd "$WORKTREE" && git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
test -z "$(cd "$WORKTREE" && git status --porcelain --untracked-files=all)"
mkdir -p "$OUTPUT_ROOT" "$ROOT/runtime/cache/jax/orcjax_gpu"

CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
test -f "$CUDA_DRIVER"
test -f "$NVML_DRIVER"
GPU_BINDS="/WORK:/WORK,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"

batch_size() {
  case "$1" in
    1) echo 256 ;;
    3) echo 64 ;;
    7) echo 32 ;;
    30) echo 8 ;;
    *) echo "unsupported rollout horizon: $1" >&2; return 2 ;;
  esac
}

for horizon in $HORIZONS; do
  rollout_batch_size=$(batch_size "$horizon")
  report="$OUTPUT_ROOT/horizon${horizon}-protocol-shape.json"
  test ! -e "$report"
  cache="$ROOT/runtime/cache/jax/orcjax_gpu/rollout-stability-h${horizon}"
  mkdir -p "$cache"
  env \
    SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs \
    SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU_DEVICE" \
    SINGULARITYENV_JAX_ENABLE_X64=true \
    SINGULARITYENV_JAX_COMPILATION_CACHE_DIR="$cache" \
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
      "$PYTHON" -m scripts.hpc.run_rollout_stability_smoke_gpu \
      --dataset "$DATASET" \
      --statistics "$STATISTICS" \
      --parent-best-checkpoint "$PARENT_BEST_CHECKPOINT" \
      --parent-optimizer-checkpoint "$PARENT_OPTIMIZER_CHECKPOINT" \
      --output "$report" \
      --horizon "$horizon" \
      --anchor-batch-size 256 \
      --rollout-batch-size "$rollout_batch_size"
done
