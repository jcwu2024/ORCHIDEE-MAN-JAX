#!/bin/bash
# Two-GPU diagnostic on free or approved Slurm nodes; weights are a timing asset.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT=${CODE_ROOT:?declare reviewed checkout}
MODE=${1:?runtime or profile}
shift
DIAG=$ROOT/runtime/outputs/diagnostics
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  GPU_DEVICE=${CUDA_VISIBLE_DEVICES:?Slurm must declare allocated GPU devices}
else
  GPU_DEVICE=${GPU_DEVICE:-0,1}
fi
PRINT_ONLY=false
if [[ "${1:-}" == --dry-run ]]; then PRINT_ONLY=true; shift; fi
run_command() {
  if "$PRINT_ONLY"; then printf '%q ' "$@"; printf '\n'; else "$@"; fi
}
case "$MODE" in
  runtime)
    run_command env CODE_ROOT="$CODE_ROOT" GPU_DEVICE="$GPU_DEVICE" \
      PYTHON_MODULE=scripts.hpc.check_daily_training_runtime RUN_TIMEOUT=3m \
      OUTPUT="$DIAG/mixed-learning-dual-runtime-v1.json" \
      bash "$CODE_ROOT/scripts/hpc/run_fresh_start_one_day.sh" --backend gpu
    run_command env CODE_ROOT="$CODE_ROOT" GPU_DEVICE="$GPU_DEVICE" \
      PYTHON_MODULE=scripts.hpc.probe_dual_device_runtime RUN_TIMEOUT=3m \
      OUTPUT="$DIAG/mixed-learning-dual-placement-v1.json" \
      bash "$CODE_ROOT/scripts/hpc/run_fresh_start_one_day.sh" --backend gpu ;;
  profile)
    run_command env CODE_ROOT="$CODE_ROOT" GPU_DEVICE="$GPU_DEVICE" GPU_MEMORY_FRACTION=0.85 \
      PYTHON_MODULE=scripts.hpc.run_mixed_learning RUN_TIMEOUT="${RUN_TIMEOUT:-90m}" \
      OUTPUT="${OUTPUT:-$DIAG/mixed-learning-dual-profile-v1.json}" \
      bash "$CODE_ROOT/scripts/hpc/run_fresh_start_one_day.sh" \
      --protocol "$CODE_ROOT/manifests/coarse_graining/mixed_learning_model_v1.json" \
      --mixed-protocol "$CODE_ROOT/manifests/coarse_graining/mixed_learning_v1.json" \
      --catalogue-report "$ROOT/runtime/outputs/training/one-day-expansion-15061225/merge_report.json" \
      --recurrent-schedule "$DIAG/mixed-learning-200-recurrent-schedule-v1.json" \
      --selection "$DIAG/trained-forward-selection-v1.json" \
      --execution-policy "$CODE_ROOT/manifests/coarse_graining/mixed_learning_execution_fp32_checked_v1.json" \
      --profile-checkpoint "$ROOT/runtime/outputs/training/mixed-learning-v1/report.update-000200.checkpoint.pkl" \
      --profile-training-report "$ROOT/runtime/outputs/training/mixed-learning-v1/report.json" \
      --dual-device-profile "$@" ;;
  *) exit 2 ;;
esac
