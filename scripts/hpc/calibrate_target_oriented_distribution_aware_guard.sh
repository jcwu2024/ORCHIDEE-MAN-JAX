#!/bin/bash
# Calibrate the distribution-aware Teacher guard on free gln01.

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:-$ROOT}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD}
GPU_DEVICE=${GPU_DEVICE:?set GPU_DEVICE after checking nvidia-smi}
OUTPUT=${OUTPUT:?set OUTPUT inside the project runtime output tree}
RUN_TIMEOUT=${RUN_TIMEOUT:-120m}
CACHE_ROOT=${CACHE_ROOT:-$ROOT/runtime/cache/jax/orcjax_gpu/target-oriented-distribution-aware-maintenance}
PYTHON=$ROOT/.venvs/orcjax_gpu/bin/python
IMAGE=/apps/soft/sif/foundationpose

case "$(hostname -s)" in gln01|ibgln01|ibc13b02n04) ;; *) exit 2 ;; esac
case "$WORKTREE" in "$ROOT"|"$ROOT/runtime/worktrees/"*) ;; *) exit 2 ;; esac
case "$OUTPUT" in "$ROOT/runtime/outputs/"*) ;; *) exit 2 ;; esac
case "$CACHE_ROOT" in "$ROOT/runtime/cache/"*) ;; *) exit 2 ;; esac
test -x "$PYTHON"
test "$(cd "$WORKTREE" && git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
test -z "$(cd "$WORKTREE" && git status --porcelain --untracked-files=all)"
test ! -e "$OUTPUT"

DATASET=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6/dataset_manifest.json
ACCEPTANCE=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/acceptance_report.json
BASE_STATISTICS=$ROOT/runtime/outputs/training/pft14-daily-teacher-v6-acceptance-v2/training_statistics.json
PROCESS_STATISTICS=$ROOT/runtime/outputs/training/first-neural-process-head-v1-statistics/process_training_statistics.json
FAST_STATISTICS=$ROOT/runtime/outputs/training/successor-fast-owner-statistics-c9f35c0/successor_fast_owner_statistics.json
TRAINING_CONTRACT=$WORKTREE/manifests/coarse_graining/target_oriented_bounded_training_v1.json
FLUX_BASIS_AMENDMENT=$WORKTREE/manifests/coarse_graining/target_oriented_flux_basis_amendment_v1.json
CANDIDATE_CONTRACT=$WORKTREE/manifests/coarse_graining/target_oriented_distribution_aware_maintenance_v1.json
CALIBRATION_CONTRACT=$WORKTREE/manifests/coarse_graining/target_oriented_distribution_aware_guard_calibration_v1.json
GPU_ADMISSION_REPORT=$ROOT/runtime/outputs/training/target-oriented-distribution-aware-maintenance-admission-54ada54/report.json
PREPARED=$ROOT/runtime/outputs/training/target-oriented-prepared-1000-a243029/prepared_manifest.json
VALIDATION_PREPARED=$ROOT/runtime/outputs/training/target-oriented-validation-9b82b5a/prepared_manifest.json
STAGE_CHECKPOINT=$ROOT/runtime/outputs/training/target-oriented-training-formal-u1000-433d286/stage-update-1000.pkl
STAGE_CHECKPOINT_SHA256=a566f9d95726984dc03cbf1f1a183dd0d1eae9e3479ce5055c64e571ef6a16c7

for path in "$DATASET" "$ACCEPTANCE" "$BASE_STATISTICS" \
  "$PROCESS_STATISTICS" "$FAST_STATISTICS" "$TRAINING_CONTRACT" \
  "$FLUX_BASIS_AMENDMENT" "$CANDIDATE_CONTRACT" "$CALIBRATION_CONTRACT" \
  "$GPU_ADMISSION_REPORT" "$PREPARED" "$VALIDATION_PREPARED" "$STAGE_CHECKPOINT"; do
  test -f "$path"
done
mkdir -p "$(dirname "$OUTPUT")" "$CACHE_ROOT"

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
  timeout --signal=TERM "$RUN_TIMEOUT" \
  singularity exec --nv --bind "$GPU_BINDS" --pwd "$WORKTREE" "$IMAGE" \
    "$PYTHON" -m scripts.hpc.calibrate_target_oriented_distribution_aware_guard \
      --dataset-manifest "$DATASET" \
      --acceptance-report "$ACCEPTANCE" \
      --base-statistics "$BASE_STATISTICS" \
      --process-statistics "$PROCESS_STATISTICS" \
      --fast-statistics "$FAST_STATISTICS" \
      --training-contract "$TRAINING_CONTRACT" \
      --flux-basis-amendment "$FLUX_BASIS_AMENDMENT" \
      --candidate-contract "$CANDIDATE_CONTRACT" \
      --calibration-contract "$CALIBRATION_CONTRACT" \
      --gpu-admission-report "$GPU_ADMISSION_REPORT" \
      --prepared-manifest "$PREPARED" \
      --prepared-updates 1000 \
      --validation-prepared-manifest "$VALIDATION_PREPARED" \
      --stage-checkpoint "$STAGE_CHECKPOINT" \
      --stage-checkpoint-sha256 "$STAGE_CHECKPOINT_SHA256" \
      --output "$OUTPUT"

test -f "$OUTPUT"
sha256sum "$OUTPUT"
