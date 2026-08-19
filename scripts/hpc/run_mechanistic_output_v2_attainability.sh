#!/bin/bash
# Run the free-gln01, zero-training mechanistic-output v2 attainability audit.

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:?set WORKTREE to the isolated mechanistic source tree}
EXPECTED_BASE_HEAD=${EXPECTED_BASE_HEAD:?set EXPECTED_BASE_HEAD}
GPU_DEVICE=${GPU_DEVICE:?set GPU_DEVICE after checking nvidia-smi}
SOURCE_TAG=${SOURCE_TAG:-mechanistic-output-v2-attainability}
PYTHON=$ROOT/.venvs/orcjax_gpu/bin/python
IMAGE=/apps/soft/sif/foundationpose

DATASET=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6/dataset_manifest.json
ACCEPTANCE=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/acceptance_report.json
BASE_STATISTICS=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/training_statistics.json
PROCESS_STATISTICS=$ROOT/runtime/outputs/training/first-neural-process-head-v1-statistics/process_training_statistics.json
FAST_STATISTICS=$ROOT/runtime/outputs/training/successor-fast-owner-statistics-c9f35c0/successor_fast_owner_statistics.json
OUTPUT=${OUTPUT:-$ROOT/runtime/outputs/training/$SOURCE_TAG/report.json}
CACHE_ROOT=${CACHE_ROOT:-$ROOT/runtime/cache/jax/orcjax_gpu/$SOURCE_TAG}

test -x "$PYTHON"
test "$(cd "$WORKTREE" && git rev-parse HEAD)" = "$EXPECTED_BASE_HEAD"
test ! -e "$OUTPUT"
mkdir -p "$(dirname "$OUTPUT")" "$CACHE_ROOT"

CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
GPU_BINDS="/WORK:/WORK,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"

echo "host=$(hostname) base_head=$EXPECTED_BASE_HEAD source_tag=$SOURCE_TAG gpu_device=$GPU_DEVICE output=$OUTPUT"
nvidia-smi -i "$GPU_DEVICE" --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader

env -u LD_LIBRARY_PATH \
  SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs \
  SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU_DEVICE" \
  SINGULARITYENV_JAX_ENABLE_X64=true \
  SINGULARITYENV_JAX_PLATFORMS=cuda \
  SINGULARITYENV_JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT" \
  SINGULARITYENV_ORCHIDEE_REPO_ROOT="$WORKTREE" \
  SINGULARITYENV_ORCHIDEE_RUNTIME_ROOT="$ROOT/runtime" \
  SINGULARITYENV_ORCHIDEE_DATA_ROOT="$ROOT/runtime/data" \
  SINGULARITYENV_ORCHIDEE_REFERENCE_ROOT="$ROOT/runtime/assets" \
  SINGULARITYENV_ORCHIDEE_OUTPUT_ROOT="$ROOT/runtime/outputs" \
  SINGULARITYENV_PYTHONPATH="$WORKTREE" \
  singularity exec --nv --bind "$GPU_BINDS" --pwd "$WORKTREE" "$IMAGE" \
    "$PYTHON" -m scripts.hpc.run_mechanistic_output_v2_attainability \
      --dataset-manifest "$DATASET" \
      --acceptance-report "$ACCEPTANCE" \
      --base-statistics "$BASE_STATISTICS" \
      --process-statistics "$PROCESS_STATISTICS" \
      --fast-statistics "$FAST_STATISTICS" \
      --batch-size 8 \
      --doc-fixed-point-iterations 8 \
      --output "$OUTPUT"

sha256sum "$OUTPUT"
