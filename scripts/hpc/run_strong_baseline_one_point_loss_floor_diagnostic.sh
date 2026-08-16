#!/bin/bash
# Run the bounded one-point loss-floor diagnostic on free gln01.

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:-$ROOT}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD to the approved commit}
GPU_DEVICE=${GPU_DEVICE:?set GPU_DEVICE after checking nvidia-smi}
PYTHON=$ROOT/.venvs/orcjax_gpu/bin/python
IMAGE=/apps/soft/sif/foundationpose

DATASET=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6/dataset_manifest.json
ACCEPTANCE=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/acceptance_report.json
BASE_STATISTICS=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/training_statistics.json
PROCESS_STATISTICS=$ROOT/runtime/outputs/training/first-neural-process-head-v1-statistics/process_training_statistics.json
FAST_STATISTICS=$ROOT/runtime/outputs/training/successor-fast-owner-statistics-c9f35c0/successor_fast_owner_statistics.json
TENDENCY_SCALE=$ROOT/runtime/outputs/training/successor-named-biomass-tendency-scales-5bd55c7/scale_asset.json
TENDENCY_ACCEPTANCE=$WORKTREE/manifests/coarse_graining/successor_named_biomass_tendency_scale_acceptance_v1.json
HEAD_SHORT=${EXPECTED_GIT_HEAD:0:7}
MAXIMUM_EPOCHS=${MAXIMUM_EPOCHS:-100}
EVALUATION_INTERVAL=${EVALUATION_INTERVAL:-5}
TAIL_UPDATES=${TAIL_UPDATES:-25}
TAIL_LEARNING_RATE=${TAIL_LEARNING_RATE:-1e-5}
RUN_ROOT=${RUN_ROOT:-$ROOT/runtime/outputs/training/strong-baseline-one-point-loss-floor-$HEAD_SHORT}
OUTPUT=${OUTPUT:-$RUN_ROOT/report.json}
DIAGNOSTIC_STATE=${DIAGNOSTIC_STATE:-$RUN_ROOT/diagnostic_state.pkl}
CACHE_ROOT=${CACHE_ROOT:-$ROOT/runtime/cache/jax/orcjax_gpu/strong-baseline-one-point-loss-floor-$HEAD_SHORT}

test -x "$PYTHON"
test "$(cd "$WORKTREE" && git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
test ! -e "$OUTPUT"
test ! -e "$DIAGNOSTIC_STATE"
mkdir -p "$RUN_ROOT" "$CACHE_ROOT"

case "$(hostname -s)" in
  gln01|ibgln01|ibc13b02n04) ;;
  *) echo "one-point loss-floor diagnostic requires free gln01" >&2; exit 2 ;;
esac

CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
GPU_BINDS="/WORK:/WORK,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"

echo "host=$(hostname) git_head=$EXPECTED_GIT_HEAD gpu_device=$GPU_DEVICE output=$OUTPUT"
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
    "$PYTHON" -m scripts.hpc.run_strong_baseline_one_point_qualification \
      --dataset-manifest "$DATASET" \
      --acceptance-report "$ACCEPTANCE" \
      --base-statistics "$BASE_STATISTICS" \
      --process-statistics "$PROCESS_STATISTICS" \
      --fast-statistics "$FAST_STATISTICS" \
      --batch-size 8 \
      --maximum-epochs "$MAXIMUM_EPOCHS" \
      --evaluation-interval "$EVALUATION_INTERVAL" \
      --diagnostic-tail-updates "$TAIL_UPDATES" \
      --diagnostic-tail-learning-rate "$TAIL_LEARNING_RATE" \
      --named-biomass-tendency-scale-asset "$TENDENCY_SCALE" \
      --named-biomass-tendency-scale-acceptance "$TENDENCY_ACCEPTANCE" \
      --diagnostic-state-output "$DIAGNOSTIC_STATE" \
      --output "$OUTPUT"

sha256sum "$OUTPUT" "$DIAGNOSTIC_STATE"
