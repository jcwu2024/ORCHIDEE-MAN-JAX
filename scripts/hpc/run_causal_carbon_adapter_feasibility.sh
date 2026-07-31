#!/bin/bash
# Run the bounded Experiment C gate directly on the free gln01 test host.

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:-$ROOT}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD}
GPU_DEVICE=${GPU_DEVICE:-0}
PYTHON=$ROOT/.venvs/orcjax_gpu/bin/python
IMAGE=/apps/soft/sif/foundationpose

DATASET=${DATASET:-$ROOT/runtime/outputs/training/pft14-daily-teacher-669-1961-2010-v5-7397d1e-w100/dataset_manifest.json}
STATISTICS=${STATISTICS:-$ROOT/runtime/outputs/acceptance/pft14-daily-teacher-669-1961-2010-v5-7397d1e-w100-final-20260728-r2/training_statistics.json}
ACCEPTANCE=${ACCEPTANCE:-$ROOT/runtime/outputs/acceptance/pft14-daily-teacher-669-1961-2010-v5-7397d1e-w100-final-20260728-r2/acceptance_report.json}
PARENT_ROOT=${PARENT_ROOT:-$ROOT/runtime/outputs/training/canonical-669-rollout-stability-v2-d6e43cc/one_step_continuation_control}
PARENT_CHECKPOINT=${PARENT_CHECKPOINT:-$PARENT_ROOT/checkpoint.pkl}
PARENT_REPORT=${PARENT_REPORT:-$PARENT_ROOT/training_report.json}
PLAN=${PLAN:-$ROOT/runtime/plans/teacher_669_1961_2010_v5_7397d1e_w100.json}
PROTOCOL=$WORKTREE/manifests/coarse_graining/canonical_669_causal_carbon_adapter_experiment.json
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/runtime/outputs/training/causal-carbon-adapter-feasibility-${EXPECTED_GIT_HEAD:0:7}}
CACHE_ROOT=$ROOT/runtime/cache/jax/orcjax_gpu/causal-carbon-adapter

test "$(hostname)" = gln01
test -x "$PYTHON"
for path in \
  "$DATASET" \
  "$STATISTICS" \
  "$ACCEPTANCE" \
  "$PARENT_CHECKPOINT" \
  "$PARENT_REPORT" \
  "$PLAN" \
  "$PROTOCOL"; do
  test -f "$path"
done
for path in "$DATASET" "$STATISTICS" "$ACCEPTANCE" "$PARENT_CHECKPOINT" \
  "$PARENT_REPORT" "$PLAN" "$OUTPUT_ROOT" "$CACHE_ROOT"; do
  case "$path" in
    "$ROOT/runtime/"*) ;;
    *) echo "runtime path escaped the project root: $path" >&2; exit 2 ;;
  esac
done
test "$(cd "$WORKTREE" && git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
test -z "$(cd "$WORKTREE" && git status --porcelain --untracked-files=all)"
mkdir -p "$OUTPUT_ROOT" "$CACHE_ROOT"

CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
test -f "$CUDA_DRIVER"
test -f "$NVML_DRIVER"
GPU_BINDS="/WORK:/WORK,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"

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
    "$PYTHON" -m scripts.hpc.run_causal_carbon_adapter_gpu \
      --protocol "$PROTOCOL" \
      --dataset "$DATASET" \
      --statistics "$STATISTICS" \
      --acceptance "$ACCEPTANCE" \
      --parent-checkpoint "$PARENT_CHECKPOINT" \
      --parent-report "$PARENT_REPORT" \
      --plan "$PLAN" \
      --output-root "$OUTPUT_ROOT"
