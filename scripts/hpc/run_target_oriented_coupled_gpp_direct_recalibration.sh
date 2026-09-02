#!/bin/bash
# Run the free-gln01, zero-retention direct common-direction smoke.

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:-$ROOT}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD}
GPU_DEVICE=${GPU_DEVICE:?set GPU_DEVICE after checking nvidia-smi}
OUTPUT=${OUTPUT:?set OUTPUT inside the project runtime diagnostics tree}
RUN_TIMEOUT=${RUN_TIMEOUT:-90m}
CACHE_ROOT=${CACHE_ROOT:-$ROOT/runtime/cache/jax/orcjax_gpu/target-oriented-coupled-gpp-objective-calibration}
PARENT_DIR=$ROOT/runtime/outputs/training/target-oriented-coupled-gpp-bounded-14903328
PYTHON=$ROOT/.venvs/orcjax_gpu/bin/python
IMAGE=/apps/soft/sif/foundationpose

case "$(hostname -s)" in
  gln01|ibgln01|ibc13b02n04) ;;
  *) exit 2 ;;
esac
case "$WORKTREE" in "$ROOT"|"$ROOT/runtime/worktrees/"*) ;; *) exit 2 ;; esac
case "$OUTPUT" in "$ROOT/runtime/outputs/diagnostics/"*.json) ;; *) exit 2 ;; esac
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
PREPARED=$ROOT/runtime/outputs/training/target-oriented-prepared-1000-a243029/prepared_manifest.json
VALIDATION_PREPARED=$ROOT/runtime/outputs/training/target-oriented-validation-9b82b5a/prepared_manifest.json
STAGE_CHECKPOINT=$ROOT/runtime/outputs/training/target-oriented-training-formal-u1000-433d286/stage-update-1000.pkl
STAGE_CHECKPOINT_SHA256=a566f9d95726984dc03cbf1f1a183dd0d1eae9e3479ce5055c64e571ef6a16c7
DISTRIBUTION_CONTRACT=$WORKTREE/manifests/coarse_graining/target_oriented_distribution_aware_maintenance_v1.json
DISTRIBUTION_DIR=$ROOT/runtime/outputs/training/target-oriented-distribution-aware-u250-14856147
DISTRIBUTION_REPORT=$DISTRIBUTION_DIR/report.json
DISTRIBUTION_CHECKPOINT=$DISTRIBUTION_DIR/stage-update-0250.pkl
DISTRIBUTION_CHECKPOINT_SHA256=baae0c13aa7f22f04e56bec43b915e7e2dab8c2ca775e91d260f26136d55c7da
STRUCTURED_CONTRACT=$WORKTREE/manifests/coarse_graining/target_oriented_structured_gpp_v1.json
BROAD_DIR=$ROOT/runtime/outputs/training/target-oriented-structured-gpp-broad-14883081
BROAD_REPORT=$BROAD_DIR/report.json
SELECTED_CHECKPOINT=$BROAD_DIR/stage-update-1000.pkl
COUPLED_CONTRACT=$WORKTREE/manifests/coarse_graining/target_oriented_coupled_gpp_carbon_feedback_v1.json
REACHABILITY_CONTRACT=$WORKTREE/manifests/coarse_graining/target_oriented_gpp_state_reachability_v1.json
GPU_ADMISSION_REPORT=$ROOT/runtime/outputs/diagnostics/target-oriented-coupled-gpp-carbon-feedback-real-v6-14897461.json
REACHABILITY_REPORT=$ROOT/runtime/outputs/diagnostics/target-oriented-gpp-state-reachability-14899147.json
OBJECTIVE_CONTRACT=$WORKTREE/manifests/coarse_graining/target_oriented_coupled_gpp_objective_calibration_v1.json
OBJECTIVE_REPORT=$ROOT/runtime/outputs/diagnostics/target-oriented-coupled-gpp-objective-calibration-14899511.json
BACKOFF_REPORT=$ROOT/runtime/outputs/diagnostics/target-oriented-coupled-gpp-optimizer-backoff-14902878.json
BOUNDED_CONTRACT=$WORKTREE/manifests/coarse_graining/target_oriented_coupled_gpp_bounded_training_v1.json
FRESH_MOMENT_CONTRACT=$WORKTREE/manifests/coarse_graining/target_oriented_coupled_gpp_fresh_moment_admission_v1.json
AUDIT_CONTRACT=$WORKTREE/manifests/coarse_graining/target_oriented_coupled_gpp_update3_common_direction_audit_v1.json
RECALIBRATION_CONTRACT=$WORKTREE/manifests/coarse_graining/target_oriented_coupled_gpp_direct_recalibration_smoke_v1.json
AUDIT_RESULT=$ROOT/runtime/outputs/diagnostics/target-oriented-coupled-gpp-update3-common-direction-e704749.json
PARENT_RESULT=$PARENT_DIR/report.json
PARENT_RECOVERY=$PARENT_DIR/recovery.pkl

for path in "$DATASET" "$ACCEPTANCE" "$BASE_STATISTICS" \
  "$PROCESS_STATISTICS" "$FAST_STATISTICS" "$TRAINING_CONTRACT" \
  "$FLUX_BASIS_AMENDMENT" "$PREPARED" "$VALIDATION_PREPARED" \
  "$STAGE_CHECKPOINT" "$DISTRIBUTION_CONTRACT" "$DISTRIBUTION_REPORT" \
  "$DISTRIBUTION_CHECKPOINT" "$STRUCTURED_CONTRACT" "$BROAD_REPORT" \
  "$SELECTED_CHECKPOINT" "$COUPLED_CONTRACT" "$REACHABILITY_CONTRACT" \
  "$GPU_ADMISSION_REPORT" "$REACHABILITY_REPORT" "$OBJECTIVE_CONTRACT" \
  "$OBJECTIVE_REPORT" "$BACKOFF_REPORT" "$BOUNDED_CONTRACT" \
  "$FRESH_MOMENT_CONTRACT" "$AUDIT_CONTRACT" "$RECALIBRATION_CONTRACT" \
  "$AUDIT_RESULT" "$PARENT_RESULT" "$PARENT_RECOVERY"; do
  test -f "$path"
done
mkdir -p "$(dirname "$OUTPUT")" "$CACHE_ROOT"

CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
GPU_BINDS="/WORK:/WORK,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"

echo "host=$(hostname) git_head=$EXPECTED_GIT_HEAD gpu_device=$GPU_DEVICE output=$OUTPUT parent=$PARENT_DIR"
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
    "$PYTHON" -m scripts.hpc.run_target_oriented_coupled_gpp_direct_recalibration \
      --dataset-manifest "$DATASET" \
      --acceptance-report "$ACCEPTANCE" \
      --base-statistics "$BASE_STATISTICS" \
      --process-statistics "$PROCESS_STATISTICS" \
      --fast-statistics "$FAST_STATISTICS" \
      --training-contract "$TRAINING_CONTRACT" \
      --flux-basis-amendment "$FLUX_BASIS_AMENDMENT" \
      --prepared-manifest "$PREPARED" \
      --prepared-updates 1000 \
      --validation-prepared-manifest "$VALIDATION_PREPARED" \
      --stage-checkpoint "$STAGE_CHECKPOINT" \
      --stage-checkpoint-sha256 "$STAGE_CHECKPOINT_SHA256" \
      --distribution-contract "$DISTRIBUTION_CONTRACT" \
      --distribution-training-report "$DISTRIBUTION_REPORT" \
      --distribution-checkpoint "$DISTRIBUTION_CHECKPOINT" \
      --distribution-checkpoint-sha256 "$DISTRIBUTION_CHECKPOINT_SHA256" \
      --structured-contract "$STRUCTURED_CONTRACT" \
      --broad-training-report "$BROAD_REPORT" \
      --selected-checkpoint "$SELECTED_CHECKPOINT" \
      --coupled-contract "$COUPLED_CONTRACT" \
      --reachability-contract "$REACHABILITY_CONTRACT" \
      --gpu-admission-report "$GPU_ADMISSION_REPORT" \
      --reachability-report "$REACHABILITY_REPORT" \
      --objective-calibration-contract "$OBJECTIVE_CONTRACT" \
      --objective-calibration-report "$OBJECTIVE_REPORT" \
      --optimizer-backoff-report "$BACKOFF_REPORT" \
      --bounded-training-contract "$BOUNDED_CONTRACT" \
      --fresh-moment-contract "$FRESH_MOMENT_CONTRACT" \
      --audit-contract "$AUDIT_CONTRACT" \
      --audit-result "$AUDIT_RESULT" \
      --recalibration-contract "$RECALIBRATION_CONTRACT" \
      --parent-result-report "$PARENT_RESULT" \
      --parent-recovery "$PARENT_RECOVERY" \
      --output "$OUTPUT"

test -f "$OUTPUT"
sha256sum "$OUTPUT"
