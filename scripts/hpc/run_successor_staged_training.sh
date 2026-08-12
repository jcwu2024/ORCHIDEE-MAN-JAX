#!/bin/bash
# Run one staged-v3 matched training milestone and fixed-panel validation.

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:?set WORKTREE to the approved clean research worktree}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD to the approved commit}
GPU_DEVICE=${GPU_DEVICE:?set GPU_DEVICE from the Slurm allocation}
MILESTONE=${MILESTONE:?set MILESTONE to a staged contract milestone}
PREPARED_MANIFEST=${PREPARED_MANIFEST:?set the accepted stage manifest}
VALIDATION_PREPARED_MANIFEST=${VALIDATION_PREPARED_MANIFEST:?set fixed validation prepared manifest}
OUTPUT=${OUTPUT:?set a new stage report below the project runtime root}
RECOVERY_PATH=${RECOVERY_PATH:?set the staged recovery path}
BEST_CHECKPOINT=${BEST_CHECKPOINT:?set the staged best-checkpoint path}
PYTHON=$ROOT/.venvs/orcjax_gpu/bin/python
IMAGE=/apps/soft/sif/foundationpose

DATASET=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6/dataset_manifest.json
ACCEPTANCE=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/acceptance_report.json
BASE_STATISTICS=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/training_statistics.json
PROCESS_STATISTICS=$ROOT/runtime/outputs/training/first-neural-process-head-v1-statistics/process_training_statistics.json
FAST_STATISTICS=$ROOT/runtime/outputs/training/successor-fast-owner-statistics-c9f35c0/successor_fast_owner_statistics.json
RUNNER_CONTRACT=$WORKTREE/manifests/coarse_graining/successor_training_staged_v3.json
ASSETS=$ROOT/runtime/outputs/training/successor-staged-assets-b800db9
COEFFICIENT_SOURCE=$ROOT/runtime/outputs/training/successor-multibatch-coefficients-d3200df/probe_report.json
COEFFICIENT_PREPARED=$ROOT/runtime/outputs/training/successor-sufficiency-coeff-prepared-16-c7c6af6/prepared_manifest.json

case "$WORKTREE" in
  "$ROOT"|"$ROOT/runtime/worktrees/"*) ;;
  *) echo "WORKTREE escaped the project-owned root" >&2; exit 2 ;;
esac
case "$MILESTONE" in
  500|1000|2000|5000|10000) ;;
  *) echo "MILESTONE is not declared by staged-v3: $MILESTONE" >&2; exit 2 ;;
esac
for path in "$PREPARED_MANIFEST" "$VALIDATION_PREPARED_MANIFEST" \
  "$DATASET" "$ACCEPTANCE" "$BASE_STATISTICS" "$PROCESS_STATISTICS" \
  "$FAST_STATISTICS" "$ASSETS/frozen_coefficients.json" \
  "$ASSETS/validation_panel.json" "$COEFFICIENT_SOURCE" \
  "$COEFFICIENT_PREPARED"; do
  case "$path" in
    "$ROOT/runtime/"*) ;;
    *) echo "runtime input escaped the project root: $path" >&2; exit 2 ;;
  esac
  test -f "$path"
done
for path in "$OUTPUT" "$RECOVERY_PATH" "$BEST_CHECKPOINT"; do
  case "$path" in
    "$ROOT/runtime/"*) ;;
    *) echo "runtime output escaped the project root: $path" >&2; exit 2 ;;
  esac
done
test -f "$RUNNER_CONTRACT"
test -x "$PYTHON"
test "$(cd "$WORKTREE" && git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
test -z "$(cd "$WORKTREE" && git status --porcelain --untracked-files=all)"
test ! -e "$OUTPUT"
command -v nvidia-smi >/dev/null
nvidia-smi -L >/dev/null
mkdir -p "$(dirname "$OUTPUT")"

CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
test -f "$CUDA_DRIVER"
test -f "$NVML_DRIVER"
GPU_BINDS="/WORK:/WORK,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"

echo "host=$(hostname) git_head=$EXPECTED_GIT_HEAD gpu_device=$GPU_DEVICE milestone=$MILESTONE"
echo "prepared_manifest=$PREPARED_MANIFEST output=$OUTPUT"
nvidia-smi -i "$GPU_DEVICE" \
  --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader

env -u LD_LIBRARY_PATH \
  SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs \
  SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU_DEVICE" \
  SINGULARITYENV_JAX_ENABLE_X64=true \
  SINGULARITYENV_JAX_PLATFORMS=cuda,cpu \
  SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false \
  SINGULARITYENV_PYTHONHASHSEED=0 \
  SINGULARITYENV_ORCHIDEE_REPO_ROOT="$WORKTREE" \
  SINGULARITYENV_ORCHIDEE_RUNTIME_ROOT="$ROOT/runtime" \
  SINGULARITYENV_ORCHIDEE_DATA_ROOT="$ROOT/runtime/data" \
  SINGULARITYENV_ORCHIDEE_REFERENCE_ROOT="$ROOT/runtime/assets" \
  SINGULARITYENV_ORCHIDEE_OUTPUT_ROOT="$ROOT/runtime/outputs" \
  SINGULARITYENV_PYTHONPATH="$WORKTREE" \
  singularity exec --nv --bind "$GPU_BINDS" --pwd "$WORKTREE" "$IMAGE" \
    "$PYTHON" -m scripts.hpc.run_successor_staged_training \
      --dataset-manifest "$DATASET" \
      --acceptance-report "$ACCEPTANCE" \
      --base-statistics "$BASE_STATISTICS" \
      --process-statistics "$PROCESS_STATISTICS" \
      --fast-statistics "$FAST_STATISTICS" \
      --runner-contract "$RUNNER_CONTRACT" \
      --coefficient-contract "$ASSETS/frozen_coefficients.json" \
      --coefficient-report "$COEFFICIENT_SOURCE" \
      --coefficient-prepared-manifest "$COEFFICIENT_PREPARED" \
      --validation-panel "$ASSETS/validation_panel.json" \
      --validation-panel-sha256 1ac7490a2c7067bf3f66faea75836d80a9e6fdc62ebcdc8673caa3f445c70350 \
      --validation-prepared-manifest "$VALIDATION_PREPARED_MANIFEST" \
      --prepared-manifest "$PREPARED_MANIFEST" \
      --milestone "$MILESTONE" \
      --recovery-path "$RECOVERY_PATH" \
      --best-checkpoint "$BEST_CHECKPOINT" \
      --recovery-interval 25 \
      --output "$OUTPUT"

test -f "$OUTPUT"
test -f "$RECOVERY_PATH"
test -f "$BEST_CHECKPOINT"
sha256sum "$OUTPUT" "$RECOVERY_PATH" "$BEST_CHECKPOINT"
