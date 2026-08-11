#!/bin/bash
# Run a checkpoint-bound successor 7/30-day screen on gln01 or an allocated GPU.

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:?set WORKTREE to the approved clean research worktree}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD to the approved commit}
GPU_DEVICE=${GPU_DEVICE:?set GPU_DEVICE after checking nvidia-smi or from CUDA_VISIBLE_DEVICES}
PREPARED_MANIFEST=${PREPARED_MANIFEST:?set the accepted prepared manifest}
PREPARED_MANIFEST_SHA256=${PREPARED_MANIFEST_SHA256:?set its exact SHA256}
WINDOW_PLAN=${WINDOW_PLAN:?set the frozen window plan}
WINDOW_PLAN_FILE_SHA256=${WINDOW_PLAN_FILE_SHA256:?set its exact file SHA256}
FIT_REPORT=${FIT_REPORT:?set the successful 500-update fit report}
FIT_REPORT_SHA256=${FIT_REPORT_SHA256:?set its exact SHA256}
CHECKPOINT_DIR=${CHECKPOINT_DIR:?set the atomic dual-checkpoint directory}
CHECKPOINT_MANIFEST_SHA256=${CHECKPOINT_MANIFEST_SHA256:?set checkpoint_manifest.json SHA256}
OUTPUT=${OUTPUT:?set a new report path under the project runtime root}
PYTHON=$ROOT/.venvs/orcjax_gpu/bin/python
IMAGE=/apps/soft/sif/foundationpose

DATASET=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6/dataset_manifest.json
ACCEPTANCE=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/acceptance_report.json
BASE_STATISTICS=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/training_statistics.json
PROCESS_STATISTICS=$ROOT/runtime/outputs/training/first-neural-process-head-v1-statistics/process_training_statistics.json
FAST_STATISTICS=$ROOT/runtime/outputs/training/successor-fast-owner-statistics-c9f35c0/successor_fast_owner_statistics.json
SCREEN_CONTRACT=$WORKTREE/manifests/coarse_graining/successor_7_30_rollout_screen_v1.json
HEAD_SHORT=${EXPECTED_GIT_HEAD:0:7}
CACHE_ROOT=${CACHE_ROOT:-$ROOT/runtime/cache/jax/orcjax_gpu/successor-rollout-screen-$HEAD_SHORT}

case "$WORKTREE" in
  "$ROOT"|"$ROOT/runtime/worktrees/"*) ;;
  *) echo "WORKTREE escaped the project-owned root" >&2; exit 2 ;;
esac
for path in "$PREPARED_MANIFEST" "$WINDOW_PLAN" "$FIT_REPORT"; do
  case "$path" in
    "$ROOT/runtime/"*) ;;
    *) echo "runtime input escaped the project root: $path" >&2; exit 2 ;;
  esac
  test -f "$path"
done
for path in "$CHECKPOINT_DIR" "$OUTPUT" "$CACHE_ROOT"; do
  case "$path" in
    "$ROOT/runtime/"*) ;;
    *) echo "runtime path escaped the project root: $path" >&2; exit 2 ;;
  esac
done
for path in "$DATASET" "$ACCEPTANCE" "$BASE_STATISTICS" \
  "$PROCESS_STATISTICS" "$FAST_STATISTICS" "$SCREEN_CONTRACT" \
  "$CHECKPOINT_DIR/checkpoint_manifest.json"; do
  test -f "$path"
done
test -x "$PYTHON"
test "$(cd "$WORKTREE" && git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
test -z "$(cd "$WORKTREE" && git status --porcelain --untracked-files=all)"
test ! -e "$OUTPUT"
command -v nvidia-smi >/dev/null
nvidia-smi -L >/dev/null
mkdir -p "$(dirname "$OUTPUT")" "$CACHE_ROOT"

CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
test -f "$CUDA_DRIVER"
test -f "$NVML_DRIVER"
GPU_BINDS="/WORK:/WORK,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"

echo "host=$(hostname) git_head=$EXPECTED_GIT_HEAD gpu_device=$GPU_DEVICE output=$OUTPUT"
nvidia-smi -i "$GPU_DEVICE" \
  --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader

env -u LD_LIBRARY_PATH \
  SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs \
  SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU_DEVICE" \
  SINGULARITYENV_JAX_ENABLE_X64=true \
  SINGULARITYENV_JAX_PLATFORMS=cuda,cpu \
  SINGULARITYENV_JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT" \
  SINGULARITYENV_PYTHONHASHSEED=0 \
  SINGULARITYENV_ORCHIDEE_REPO_ROOT="$WORKTREE" \
  SINGULARITYENV_ORCHIDEE_RUNTIME_ROOT="$ROOT/runtime" \
  SINGULARITYENV_ORCHIDEE_DATA_ROOT="$ROOT/runtime/data" \
  SINGULARITYENV_ORCHIDEE_REFERENCE_ROOT="$ROOT/runtime/assets" \
  SINGULARITYENV_ORCHIDEE_OUTPUT_ROOT="$ROOT/runtime/outputs" \
  SINGULARITYENV_PYTHONPATH="$WORKTREE" \
  singularity exec --nv --bind "$GPU_BINDS" --pwd "$WORKTREE" "$IMAGE" \
    "$PYTHON" -m scripts.hpc.run_successor_rollout_screen \
      --dataset-manifest "$DATASET" \
      --acceptance-report "$ACCEPTANCE" \
      --base-statistics "$BASE_STATISTICS" \
      --process-statistics "$PROCESS_STATISTICS" \
      --fast-statistics "$FAST_STATISTICS" \
      --screen-contract "$SCREEN_CONTRACT" \
      --prepared-manifest "$PREPARED_MANIFEST" \
      --prepared-manifest-sha256 "$PREPARED_MANIFEST_SHA256" \
      --window-plan "$WINDOW_PLAN" \
      --window-plan-file-sha256 "$WINDOW_PLAN_FILE_SHA256" \
      --fit-report "$FIT_REPORT" \
      --fit-report-sha256 "$FIT_REPORT_SHA256" \
      --checkpoint-dir "$CHECKPOINT_DIR" \
      --checkpoint-manifest-sha256 "$CHECKPOINT_MANIFEST_SHA256" \
      --output "$OUTPUT"

test -f "$OUTPUT"
sha256sum "$OUTPUT"

