#!/bin/bash
# Independent single-GPU budget check first; then unchanged dual replay.
# Free diagnostics only. No retained training updates or paid submission.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT=${CODE_ROOT:?declare reviewed budget-profile checkout}
DUAL_CODE_ROOT=$ROOT/runtime/checkouts/mixed-learning-parallel-routes-v1
DIAG=$ROOT/runtime/outputs/diagnostics
case "$(hostname -s)" in gln01|ibgln01|ibc13b02n04) ;; *) exit 2 ;; esac
DRY=false
if [[ "${1:-}" == --dry-run ]]; then DRY=true; fi

if "$DRY"; then
  CODE_ROOT="$CODE_ROOT" GPU_DEVICE=0 \
    bash "$CODE_ROOT/scripts/hpc/run_mixed_learning_budget_profile.sh" --dry-run
  CODE_ROOT="$DUAL_CODE_ROOT" RUN_TIMEOUT=150m \
    OUTPUT="$DIAG/mixed-learning-dual-profile-v2.json" \
    bash "$DUAL_CODE_ROOT/scripts/hpc/run_mixed_learning_dual_device.sh" profile --dry-run
  exit 0
fi

ACTIVE_GPU_PROCESSES=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)
if [[ -n "$ACTIVE_GPU_PROCESSES" ]]; then
  echo "budget_not_started=GPU0_process_present time=$(date -Is)"
  exit 0
fi
echo "budget_start=$(date -Is)"
set +e
CODE_ROOT="$CODE_ROOT" GPU_DEVICE=0 RUN_TIMEOUT=90m \
  OUTPUT="$DIAG/mixed-learning-budget-profile-v1.json" \
  bash "$CODE_ROOT/scripts/hpc/run_mixed_learning_budget_profile.sh" \
  > "$DIAG/mixed-learning-budget-profile-v1.log" 2>&1
BUDGET_STATUS=$?
set -e
echo "budget_exit_status=$BUDGET_STATUS time=$(date -Is)"

# The two studies are independent. A failed budget diagnostic is recorded but
# does not erase the useful dual retry; never compete with another GPU process.
ACTIVE_GPU_PROCESSES=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [[ -n "$ACTIVE_GPU_PROCESSES" ]]; then
  echo "dual_not_started=GPU_process_present time=$(date -Is)"
  exit "$BUDGET_STATUS"
fi
echo "dual_start=$(date -Is)"
set +e
CODE_ROOT="$DUAL_CODE_ROOT" RUN_TIMEOUT=150m \
  SINGULARITYENV_JAX_DEBUG_LOG_MODULES=jax._src.compiler \
  OUTPUT="$DIAG/mixed-learning-dual-profile-v2.json" \
  bash "$DUAL_CODE_ROOT/scripts/hpc/run_mixed_learning_dual_device.sh" profile \
  > "$DIAG/mixed-learning-dual-profile-v2.log" 2>&1
DUAL_STATUS=$?
set -e
echo "dual_exit_status=$DUAL_STATUS time=$(date -Is)"
if [[ "$BUDGET_STATUS" != 0 ]]; then exit "$BUDGET_STATUS"; fi
exit "$DUAL_STATUS"
