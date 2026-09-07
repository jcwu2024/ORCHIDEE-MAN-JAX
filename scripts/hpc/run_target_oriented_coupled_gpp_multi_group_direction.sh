#!/bin/bash
# Shared free-gln01 boundary for artifact-aware coupled-GPP audits and training.

set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
WORKTREE=${WORKTREE:-$ROOT}
EXPECTED_GIT_HEAD=${EXPECTED_GIT_HEAD:?set EXPECTED_GIT_HEAD}
GPU_DEVICE=${GPU_DEVICE:?set GPU_DEVICE after checking nvidia-smi}
OUTPUT=${OUTPUT:?set OUTPUT inside the project runtime outputs tree}
RUN_TIMEOUT=${RUN_TIMEOUT:-120m}
CACHE_ROOT=${CACHE_ROOT:-$ROOT/runtime/cache/jax/orcjax_gpu/target-oriented-coupled-gpp-objective-calibration}
PYTHON_MODULE=${PYTHON_MODULE:-scripts.hpc.audit_target_oriented_coupled_gpp_multi_group_direction}
ARTIFACT_AWARE_CONTRACT=${ARTIFACT_AWARE_CONTRACT:-}
MULTI_GROUP_RESULT=${MULTI_GROUP_RESULT:-}
CONTINUITY_CONTRACT=${CONTINUITY_CONTRACT:-}
ARTIFACT_AWARE_RESULT=${ARTIFACT_AWARE_RESULT:-}
ARTIFACT_AWARE_PERSISTENT_CONTRACT=${ARTIFACT_AWARE_PERSISTENT_CONTRACT:-}
CONTINUITY_RESULT=${CONTINUITY_RESULT:-}
PERSISTENT_TRAINING_REPORT=${PERSISTENT_TRAINING_REPORT:-}
PERSISTENT_REJECTED_PROPOSAL=${PERSISTENT_REJECTED_PROPOSAL:-}
PERSISTENT_ARTIFACT_AWARE_RECOVERY=${PERSISTENT_ARTIFACT_AWARE_RECOVERY:-}
LOCALIZED_JUMP_RESULT=${LOCALIZED_JUMP_RESULT:-}
SNOWHEAT_CAUSAL_RESULT=${SNOWHEAT_CAUSAL_RESULT:-}
CROSSED_ATTRIBUTION_RESULT=${CROSSED_ATTRIBUTION_RESULT:-}
STRUCTURED_PRECISION_RESULT=${STRUCTURED_PRECISION_RESULT:-}
POTENTIAL_INTERNAL_RESULT=${POTENTIAL_INTERNAL_RESULT:-}
CONDITIONING_CONTRACT=${CONDITIONING_CONTRACT:-}
HEAD_TOKEN_RESULT=${HEAD_TOKEN_RESULT:-}
CONDITIONED_ADMISSION_RESULT=${CONDITIONED_ADMISSION_RESULT:-}
CONDITIONED_DISCARDED_UPDATE_CONTRACT=${CONDITIONED_DISCARDED_UPDATE_CONTRACT:-}
CONDITIONED_DISCARDED_UPDATE_RESULT=${CONDITIONED_DISCARDED_UPDATE_RESULT:-}
CONDITIONED_PERSISTENT_CONTRACT=${CONDITIONED_PERSISTENT_CONTRACT:-}
PARENT_DIR=$ROOT/runtime/outputs/training/target-oriented-coupled-gpp-bounded-14903328
PYTHON=$ROOT/.venvs/orcjax_gpu/bin/python
IMAGE=/apps/soft/sif/foundationpose

case "$(hostname -s)" in gln01|ibgln01|ibc13b02n04) ;; *) exit 2 ;; esac
case "$WORKTREE" in "$ROOT"|"$ROOT/runtime/worktrees/"*) ;; *) exit 2 ;; esac
case "$OUTPUT" in "$ROOT/runtime/outputs/"*.json|"$ROOT/runtime/outputs/"*/*.json) ;; *) exit 2 ;; esac
case "$CACHE_ROOT" in "$ROOT/runtime/cache/"*) ;; *) exit 2 ;; esac
test -x "$PYTHON"
test "$(cd "$WORKTREE" && git rev-parse HEAD)" = "$EXPECTED_GIT_HEAD"
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
SCALE_AUDIT_CONTRACT=$WORKTREE/manifests/coarse_graining/target_oriented_coupled_gpp_update5_scale_audit_v1.json
SCALE_SELECTING_CONTRACT=$WORKTREE/manifests/coarse_graining/target_oriented_coupled_gpp_scale_selecting_recalibration_smoke_v1.json
TRANSFER_CONTRACT=$WORKTREE/manifests/coarse_graining/target_oriented_coupled_gpp_scale_selected_transfer_audit_v1.json
PERSISTENT_CONTRACT=$WORKTREE/manifests/coarse_graining/target_oriented_coupled_gpp_persistent_scale_training_v1.json
GENERALIZATION_CONTRACT=$WORKTREE/manifests/coarse_graining/target_oriented_coupled_gpp_direct_generalization_audit_v1.json
STEP_TRANSFER_CONTRACT=$WORKTREE/manifests/coarse_graining/target_oriented_coupled_gpp_direct_step_transfer_audit_v1.json
MULTI_GROUP_CONTRACT=$WORKTREE/manifests/coarse_graining/target_oriented_coupled_gpp_multi_group_direction_audit_v1.json
AUDIT_RESULT=$ROOT/runtime/outputs/diagnostics/target-oriented-coupled-gpp-update3-common-direction-e704749.json
SMOKE_RESULT=$ROOT/runtime/outputs/diagnostics/target-oriented-coupled-gpp-direct-recalibration-a32db38.json
SCALE_AUDIT_RESULT=$ROOT/runtime/outputs/diagnostics/target-oriented-coupled-gpp-update5-scale-audit-25ec5bc.json
SCALE_SELECTING_RESULT=$ROOT/runtime/outputs/diagnostics/target-oriented-coupled-gpp-scale-selecting-recalibration-aaeea61.json
TRANSFER_RESULT=$ROOT/runtime/outputs/diagnostics/target-oriented-coupled-gpp-scale-selected-transfer-67a4d40.json
PERSISTENT_RESULT=$ROOT/runtime/outputs/training/target-oriented-coupled-gpp-persistent-14911228/report.json
STEP_TRANSFER_RESULT=$ROOT/runtime/outputs/diagnostics/target-oriented-coupled-gpp-direct-step-transfer-48eb1ae.json
PARENT_RESULT=$PARENT_DIR/report.json
PARENT_RECOVERY=$PARENT_DIR/recovery.pkl

for path in "$DATASET" "$ACCEPTANCE" "$BASE_STATISTICS" "$PROCESS_STATISTICS" \
  "$FAST_STATISTICS" "$TRAINING_CONTRACT" "$FLUX_BASIS_AMENDMENT" "$PREPARED" \
  "$VALIDATION_PREPARED" "$STAGE_CHECKPOINT" "$DISTRIBUTION_CONTRACT" \
  "$DISTRIBUTION_REPORT" "$DISTRIBUTION_CHECKPOINT" "$STRUCTURED_CONTRACT" \
  "$BROAD_REPORT" "$SELECTED_CHECKPOINT" "$COUPLED_CONTRACT" \
  "$REACHABILITY_CONTRACT" "$GPU_ADMISSION_REPORT" "$REACHABILITY_REPORT" \
  "$OBJECTIVE_CONTRACT" "$OBJECTIVE_REPORT" "$BACKOFF_REPORT" "$BOUNDED_CONTRACT" \
  "$FRESH_MOMENT_CONTRACT" "$AUDIT_CONTRACT" "$RECALIBRATION_CONTRACT" \
  "$SCALE_AUDIT_CONTRACT" "$SCALE_SELECTING_CONTRACT" "$TRANSFER_CONTRACT" \
  "$PERSISTENT_CONTRACT" "$GENERALIZATION_CONTRACT" "$STEP_TRANSFER_CONTRACT" \
  "$MULTI_GROUP_CONTRACT" "$AUDIT_RESULT" "$SMOKE_RESULT" "$SCALE_AUDIT_RESULT" \
  "$SCALE_SELECTING_RESULT" "$TRANSFER_RESULT" "$PERSISTENT_RESULT" \
  "$STEP_TRANSFER_RESULT" "$PARENT_RESULT" "$PARENT_RECOVERY"; do
  test -f "$path"
done

EXTRA_AUDIT_ARGS=()
if [[ "$PYTHON_MODULE" == "scripts.hpc.audit_target_oriented_coupled_gpp_conditioned_attention_discarded_update" ]]; then
  for path in "$ARTIFACT_AWARE_CONTRACT" "$MULTI_GROUP_RESULT" "$CONTINUITY_CONTRACT" \
    "$ARTIFACT_AWARE_RESULT" "$ARTIFACT_AWARE_PERSISTENT_CONTRACT" "$CONTINUITY_RESULT" \
    "$PERSISTENT_TRAINING_REPORT" "$PERSISTENT_ARTIFACT_AWARE_RECOVERY" \
    "$CONDITIONING_CONTRACT" "$CONDITIONED_ADMISSION_RESULT" \
    "$CONDITIONED_DISCARDED_UPDATE_CONTRACT"; do
    test -n "$path"
    test -f "$path"
  done
  EXTRA_AUDIT_ARGS=(
    --artifact-aware-contract "$ARTIFACT_AWARE_CONTRACT"
    --multi-group-result "$MULTI_GROUP_RESULT"
    --continuity-contract "$CONTINUITY_CONTRACT"
    --artifact-aware-result "$ARTIFACT_AWARE_RESULT"
    --artifact-aware-persistent-contract "$ARTIFACT_AWARE_PERSISTENT_CONTRACT"
    --continuity-result "$CONTINUITY_RESULT"
    --parent-persistent-training-report "$PERSISTENT_TRAINING_REPORT"
    --parent-artifact-aware-recovery "$PERSISTENT_ARTIFACT_AWARE_RECOVERY"
    --conditioning-contract "$CONDITIONING_CONTRACT"
    --conditioned-admission-report "$CONDITIONED_ADMISSION_RESULT"
    --discarded-update-contract "$CONDITIONED_DISCARDED_UPDATE_CONTRACT"
  )
elif [[ "$PYTHON_MODULE" == "scripts.hpc.run_target_oriented_coupled_gpp_conditioned_attention_persistent_training" || \
  "$PYTHON_MODULE" == "scripts.hpc.audit_conditioned_attention_update4_derivative" || \
  "$PYTHON_MODULE" == "scripts.hpc.audit_conditioned_attention_window442" || \
  "$PYTHON_MODULE" == "scripts.hpc.audit_conditioned_attention_derivative_modes" ]]; then
  for path in "$ARTIFACT_AWARE_CONTRACT" "$MULTI_GROUP_RESULT" "$CONTINUITY_CONTRACT" \
    "$ARTIFACT_AWARE_RESULT" "$ARTIFACT_AWARE_PERSISTENT_CONTRACT" "$CONTINUITY_RESULT" \
    "$PERSISTENT_TRAINING_REPORT" "$PERSISTENT_ARTIFACT_AWARE_RECOVERY" \
    "$CONDITIONING_CONTRACT" "$CONDITIONED_ADMISSION_RESULT" \
    "$CONDITIONED_DISCARDED_UPDATE_CONTRACT" "$CONDITIONED_DISCARDED_UPDATE_RESULT" \
    "$CONDITIONED_PERSISTENT_CONTRACT"; do
    test -n "$path"
    test -f "$path"
  done
  EXTRA_AUDIT_ARGS=(
    --artifact-aware-contract "$ARTIFACT_AWARE_CONTRACT"
    --multi-group-result "$MULTI_GROUP_RESULT"
    --continuity-contract "$CONTINUITY_CONTRACT"
    --artifact-aware-result "$ARTIFACT_AWARE_RESULT"
    --artifact-aware-persistent-contract "$ARTIFACT_AWARE_PERSISTENT_CONTRACT"
    --continuity-result "$CONTINUITY_RESULT"
    --parent-persistent-training-report "$PERSISTENT_TRAINING_REPORT"
    --parent-artifact-aware-recovery "$PERSISTENT_ARTIFACT_AWARE_RECOVERY"
    --conditioning-contract "$CONDITIONING_CONTRACT"
    --conditioned-admission-report "$CONDITIONED_ADMISSION_RESULT"
    --discarded-update-contract "$CONDITIONED_DISCARDED_UPDATE_CONTRACT"
    --conditioned-discarded-update-result "$CONDITIONED_DISCARDED_UPDATE_RESULT"
    --conditioned-persistent-contract "$CONDITIONED_PERSISTENT_CONTRACT"
  )
elif [[ "$PYTHON_MODULE" == "scripts.hpc.run_target_oriented_coupled_gpp_artifact_aware_persistent_training" || \
  "$PYTHON_MODULE" == "scripts.hpc.audit_target_oriented_coupled_gpp_zero_displacement" || \
  "$PYTHON_MODULE" == "scripts.hpc.audit_target_oriented_coupled_gpp_localized_jump" || \
  "$PYTHON_MODULE" == "scripts.hpc.audit_target_oriented_coupled_gpp_snowheat_causal" || \
  "$PYTHON_MODULE" == "scripts.hpc.audit_target_oriented_coupled_gpp_day5_crossed_attribution" || \
  "$PYTHON_MODULE" == "scripts.hpc.audit_target_oriented_coupled_gpp_structured_precision_attribution" || \
  "$PYTHON_MODULE" == "scripts.hpc.audit_target_oriented_coupled_gpp_potential_internal_attribution" || \
  "$PYTHON_MODULE" == "scripts.hpc.audit_target_oriented_coupled_gpp_attention_head_token_attribution" || \
  "$PYTHON_MODULE" == "scripts.hpc.audit_target_oriented_coupled_gpp_conditioned_attention" ]]; then
  test -n "$ARTIFACT_AWARE_CONTRACT"
  test -n "$MULTI_GROUP_RESULT"
  test -n "$CONTINUITY_CONTRACT"
  test -n "$ARTIFACT_AWARE_RESULT"
  test -n "$ARTIFACT_AWARE_PERSISTENT_CONTRACT"
  test -n "$CONTINUITY_RESULT"
  test -f "$ARTIFACT_AWARE_CONTRACT"
  test -f "$MULTI_GROUP_RESULT"
  test -f "$CONTINUITY_CONTRACT"
  test -f "$ARTIFACT_AWARE_RESULT"
  test -f "$ARTIFACT_AWARE_PERSISTENT_CONTRACT"
  test -f "$CONTINUITY_RESULT"
  EXTRA_AUDIT_ARGS=(
    --artifact-aware-contract "$ARTIFACT_AWARE_CONTRACT"
    --multi-group-result "$MULTI_GROUP_RESULT"
    --continuity-contract "$CONTINUITY_CONTRACT"
    --artifact-aware-result "$ARTIFACT_AWARE_RESULT"
    --artifact-aware-persistent-contract "$ARTIFACT_AWARE_PERSISTENT_CONTRACT"
    --continuity-result "$CONTINUITY_RESULT"
  )
  if [[ "$PYTHON_MODULE" == "scripts.hpc.audit_target_oriented_coupled_gpp_zero_displacement" || \
    "$PYTHON_MODULE" == "scripts.hpc.audit_target_oriented_coupled_gpp_localized_jump" || \
    "$PYTHON_MODULE" == "scripts.hpc.audit_target_oriented_coupled_gpp_snowheat_causal" || \
    "$PYTHON_MODULE" == "scripts.hpc.audit_target_oriented_coupled_gpp_day5_crossed_attribution" || \
    "$PYTHON_MODULE" == "scripts.hpc.audit_target_oriented_coupled_gpp_structured_precision_attribution" || \
    "$PYTHON_MODULE" == "scripts.hpc.audit_target_oriented_coupled_gpp_potential_internal_attribution" || \
    "$PYTHON_MODULE" == "scripts.hpc.audit_target_oriented_coupled_gpp_attention_head_token_attribution" || \
    "$PYTHON_MODULE" == "scripts.hpc.audit_target_oriented_coupled_gpp_conditioned_attention" ]]; then
    test -n "$PERSISTENT_TRAINING_REPORT"
    test -n "$PERSISTENT_REJECTED_PROPOSAL"
    test -n "$PERSISTENT_ARTIFACT_AWARE_RECOVERY"
    test -f "$PERSISTENT_TRAINING_REPORT"
    test -f "$PERSISTENT_REJECTED_PROPOSAL"
    test -f "$PERSISTENT_ARTIFACT_AWARE_RECOVERY"
    EXTRA_AUDIT_ARGS+=(
      --parent-persistent-training-report "$PERSISTENT_TRAINING_REPORT"
      --parent-rejected-proposal "$PERSISTENT_REJECTED_PROPOSAL"
      --parent-artifact-aware-recovery "$PERSISTENT_ARTIFACT_AWARE_RECOVERY"
    )
    if [[ "$PYTHON_MODULE" == "scripts.hpc.audit_target_oriented_coupled_gpp_conditioned_attention" ]]; then
      test -n "$CONDITIONING_CONTRACT"
      test -n "$HEAD_TOKEN_RESULT"
      test -f "$CONDITIONING_CONTRACT"
      test -f "$HEAD_TOKEN_RESULT"
      EXTRA_AUDIT_ARGS+=(
        --conditioning-contract "$CONDITIONING_CONTRACT"
        --head-token-report "$HEAD_TOKEN_RESULT"
      )
    fi
    if [[ "$PYTHON_MODULE" == "scripts.hpc.audit_target_oriented_coupled_gpp_snowheat_causal" || \
      "$PYTHON_MODULE" == "scripts.hpc.audit_target_oriented_coupled_gpp_day5_crossed_attribution" || \
      "$PYTHON_MODULE" == "scripts.hpc.audit_target_oriented_coupled_gpp_structured_precision_attribution" || \
      "$PYTHON_MODULE" == "scripts.hpc.audit_target_oriented_coupled_gpp_potential_internal_attribution" || \
      "$PYTHON_MODULE" == "scripts.hpc.audit_target_oriented_coupled_gpp_attention_head_token_attribution" ]]; then
      test -n "$LOCALIZED_JUMP_RESULT"
      test -f "$LOCALIZED_JUMP_RESULT"
      EXTRA_AUDIT_ARGS+=(--localized-jump-report "$LOCALIZED_JUMP_RESULT")
    fi
    if [[ "$PYTHON_MODULE" == "scripts.hpc.audit_target_oriented_coupled_gpp_day5_crossed_attribution" || \
      "$PYTHON_MODULE" == "scripts.hpc.audit_target_oriented_coupled_gpp_structured_precision_attribution" || \
      "$PYTHON_MODULE" == "scripts.hpc.audit_target_oriented_coupled_gpp_potential_internal_attribution" || \
      "$PYTHON_MODULE" == "scripts.hpc.audit_target_oriented_coupled_gpp_attention_head_token_attribution" ]]; then
      test -n "$SNOWHEAT_CAUSAL_RESULT"
      test -f "$SNOWHEAT_CAUSAL_RESULT"
      EXTRA_AUDIT_ARGS+=(--snowheat-report "$SNOWHEAT_CAUSAL_RESULT")
    fi
    if [[ "$PYTHON_MODULE" == "scripts.hpc.audit_target_oriented_coupled_gpp_structured_precision_attribution" || \
      "$PYTHON_MODULE" == "scripts.hpc.audit_target_oriented_coupled_gpp_potential_internal_attribution" || \
      "$PYTHON_MODULE" == "scripts.hpc.audit_target_oriented_coupled_gpp_attention_head_token_attribution" ]]; then
      test -n "$CROSSED_ATTRIBUTION_RESULT"
      test -f "$CROSSED_ATTRIBUTION_RESULT"
      EXTRA_AUDIT_ARGS+=(--crossed-attribution-report "$CROSSED_ATTRIBUTION_RESULT")
    fi
    if [[ "$PYTHON_MODULE" == "scripts.hpc.audit_target_oriented_coupled_gpp_potential_internal_attribution" || \
      "$PYTHON_MODULE" == "scripts.hpc.audit_target_oriented_coupled_gpp_attention_head_token_attribution" ]]; then
      test -n "$STRUCTURED_PRECISION_RESULT"
      test -f "$STRUCTURED_PRECISION_RESULT"
      EXTRA_AUDIT_ARGS+=(--structured-precision-report "$STRUCTURED_PRECISION_RESULT")
    fi
    if [[ "$PYTHON_MODULE" == "scripts.hpc.audit_target_oriented_coupled_gpp_attention_head_token_attribution" ]]; then
      test -n "$POTENTIAL_INTERNAL_RESULT"
      test -f "$POTENTIAL_INTERNAL_RESULT"
      EXTRA_AUDIT_ARGS+=(--potential-internal-report "$POTENTIAL_INTERNAL_RESULT")
    fi
  fi
elif [[ "$PYTHON_MODULE" == "scripts.hpc.audit_target_oriented_coupled_gpp_artifact_aware_continuity" ]]; then
  test -n "$ARTIFACT_AWARE_CONTRACT"
  test -n "$MULTI_GROUP_RESULT"
  test -n "$CONTINUITY_CONTRACT"
  test -n "$ARTIFACT_AWARE_RESULT"
  test -f "$ARTIFACT_AWARE_CONTRACT"
  test -f "$MULTI_GROUP_RESULT"
  test -f "$CONTINUITY_CONTRACT"
  test -f "$ARTIFACT_AWARE_RESULT"
  EXTRA_AUDIT_ARGS=(
    --artifact-aware-contract "$ARTIFACT_AWARE_CONTRACT"
    --multi-group-result "$MULTI_GROUP_RESULT"
    --continuity-contract "$CONTINUITY_CONTRACT"
    --artifact-aware-result "$ARTIFACT_AWARE_RESULT"
  )
elif [[ "$PYTHON_MODULE" == "scripts.hpc.audit_target_oriented_coupled_gpp_artifact_aware_direction" ]]; then
  test -n "$ARTIFACT_AWARE_CONTRACT"
  test -n "$MULTI_GROUP_RESULT"
  test -f "$ARTIFACT_AWARE_CONTRACT"
  test -f "$MULTI_GROUP_RESULT"
  EXTRA_AUDIT_ARGS=(
    --artifact-aware-contract "$ARTIFACT_AWARE_CONTRACT"
    --multi-group-result "$MULTI_GROUP_RESULT"
  )
elif [[ -n "$ARTIFACT_AWARE_CONTRACT" || -n "$MULTI_GROUP_RESULT" || \
  -n "$CONTINUITY_CONTRACT" || -n "$ARTIFACT_AWARE_RESULT" || \
  -n "$ARTIFACT_AWARE_PERSISTENT_CONTRACT" || -n "$CONTINUITY_RESULT" || \
  -n "$PERSISTENT_TRAINING_REPORT" || -n "$PERSISTENT_REJECTED_PROPOSAL" || \
  -n "$PERSISTENT_ARTIFACT_AWARE_RECOVERY" || -n "$LOCALIZED_JUMP_RESULT" || \
  -n "$SNOWHEAT_CAUSAL_RESULT" || -n "$CROSSED_ATTRIBUTION_RESULT" || \
  -n "$STRUCTURED_PRECISION_RESULT" || -n "$POTENTIAL_INTERNAL_RESULT" || \
  -n "$CONDITIONING_CONTRACT" || -n "$HEAD_TOKEN_RESULT" || \
  -n "$CONDITIONED_ADMISSION_RESULT" || -n "$CONDITIONED_DISCARDED_UPDATE_CONTRACT" || \
  -n "$CONDITIONED_DISCARDED_UPDATE_RESULT" || -n "$CONDITIONED_PERSISTENT_CONTRACT" ]]; then
  exit 2
fi

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
    "$PYTHON" -m "$PYTHON_MODULE" \
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
      --scale-audit-contract "$SCALE_AUDIT_CONTRACT" \
      --scale-selecting-contract "$SCALE_SELECTING_CONTRACT" \
      --smoke-result "$SMOKE_RESULT" \
      --scale-audit-result "$SCALE_AUDIT_RESULT" \
      --parent-result-report "$PARENT_RESULT" \
      --parent-recovery "$PARENT_RECOVERY" \
      --transfer-contract "$TRANSFER_CONTRACT" \
      --scale-selecting-result "$SCALE_SELECTING_RESULT" \
      --persistent-contract "$PERSISTENT_CONTRACT" \
      --transfer-result "$TRANSFER_RESULT" \
      --generalization-contract "$GENERALIZATION_CONTRACT" \
      --persistent-result "$PERSISTENT_RESULT" \
      --step-transfer-contract "$STEP_TRANSFER_CONTRACT" \
      --direct-step-transfer-result "$STEP_TRANSFER_RESULT" \
      --multi-group-contract "$MULTI_GROUP_CONTRACT" \
      "${EXTRA_AUDIT_ARGS[@]}" \
      --output-dir "$(dirname "$OUTPUT")" \
      --output "$OUTPUT"

test -f "$OUTPUT"
sha256sum "$OUTPUT"
