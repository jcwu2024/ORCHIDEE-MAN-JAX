#!/bin/bash
# Run or resume the frozen Experiment C post-training screening.

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:?set WORKTREE to the approved screening checkout}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD}
EXPERIMENT_ROOT=${EXPERIMENT_ROOT:?set EXPERIMENT_ROOT to completed training}
OUTPUT_ROOT=${OUTPUT_ROOT:?set OUTPUT_ROOT to a new screening directory}
GPU_DEVICE=${GPU_DEVICE:-0}
MAX_SHARDS_PER_SLICE=${MAX_SHARDS_PER_SLICE:-}
BATCH_SIZE_1=${BATCH_SIZE_1:-512}
BATCH_SIZE_7=${BATCH_SIZE_7:-512}
BATCH_SIZE_30=${BATCH_SIZE_30:-256}
CANDIDATE_DISABLED_GROUPS=${CANDIDATE_DISABLED_GROUPS:-}
PYTHON=$ROOT/.venvs/orcjax_gpu/bin/python
IMAGE=/apps/soft/sif/foundationpose
CACHE_ROOT=$ROOT/runtime/cache/jax/orcjax_gpu/causal-carbon-adapter-screening

case "$WORKTREE" in
  "$ROOT/runtime/worktrees/"*) ;;
  *) echo "WORKTREE must stay under $ROOT/runtime/worktrees" >&2; exit 2 ;;
esac
for path in "$EXPERIMENT_ROOT" "$OUTPUT_ROOT" "$CACHE_ROOT"; do
  case "$path" in
    "$ROOT/runtime/"*) ;;
    *) echo "screening path escaped the project runtime root: $path" >&2; exit 2 ;;
  esac
done

test -x "$PYTHON"
test -f "$EXPERIMENT_ROOT/training_execution.json"
for arm in \
  causal_interface_one_step_control \
  causal_interface_rollout_candidate; do
  test -f "$EXPERIMENT_ROOT/$arm/training_report.json"
  test -f "$EXPERIMENT_ROOT/$arm/checkpoint.pkl"
done
test "$(cd "$WORKTREE" && git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
test -z "$(cd "$WORKTREE" && git status --porcelain --untracked-files=all)"
mkdir -p "$OUTPUT_ROOT" "$CACHE_ROOT"

CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
test -f "$CUDA_DRIVER"
test -f "$NVML_DRIVER"
GPU_BINDS="/WORK:/WORK,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"

args=(
  --experiment-root "$EXPERIMENT_ROOT"
  --output-root "$OUTPUT_ROOT"
  --batch-size-1 "$BATCH_SIZE_1"
  --batch-size-7 "$BATCH_SIZE_7"
  --batch-size-30 "$BATCH_SIZE_30"
  --skip-dataset-content-hash-verification
)
if [[ -n "$MAX_SHARDS_PER_SLICE" ]]; then
  args+=(--max-shards-per-slice "$MAX_SHARDS_PER_SLICE")
fi
if [[ -n "$CANDIDATE_DISABLED_GROUPS" ]]; then
  IFS=',' read -r -a disabled_groups <<< "$CANDIDATE_DISABLED_GROUPS"
  for group_id in "${disabled_groups[@]}"; do
    args+=(--candidate-disabled-group "$group_id")
  done
fi

env \
  SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs \
  SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU_DEVICE" \
  SINGULARITYENV_JAX_ENABLE_X64=true \
  SINGULARITYENV_JAX_PLATFORMS=cuda \
  SINGULARITYENV_JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT" \
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
    "$PYTHON" -m scripts.hpc.run_causal_carbon_adapter_screening_gpu \
    "${args[@]}"
