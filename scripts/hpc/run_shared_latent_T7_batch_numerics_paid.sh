#!/usr/bin/env bash
# Paid allocation admission belongs to the paid Python supervisor. Reuse the
# accepted deadlines and runtime child; do not initialize a GPU in this entry.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/run_shared_latent_T7_batch_numerics.sh"
RUNNER=$CODE/scripts/hpc/run_shared_latent_T7_batch_numerics_paid.py
paid_startup_shell_evidence() {
  # Explicit public startup fields only: never dump inherited environment.
  printf 'PAID_STARTUP_SHELL stage=%q exit_code=%q SLURM_JOB_ID=%q SLURM_JOB_GPUS=%q SLURM_STEP_GPUS=%q CUDA_VISIBLE_DEVICES=%q ORCJAX_SLURM_VISIBLE_DEVICES=%q expected_UUID=%q epochs_reply=%q allocation_start=%q allocation_end=%q now=%q shell_started_epoch=%q phase_epoch=%q leader_pid=%q leader_pgid=%q entry=%q child_python=%q cuda_driver=%q nvml_driver=%q toolchain=%q\n' \
    "$1" "${2:-}" "${SLURM_JOB_ID-}" "${SLURM_JOB_GPUS-}" "${SLURM_STEP_GPUS-}" \
    "${CUDA_VISIBLE_DEVICES-}" "${ORCJAX_SLURM_VISIBLE_DEVICES-}" "${ORCJAX_AFFINE_EXPECTED_GPU_UUID-}" \
    "${EPOCHS-}" "${ALLOCATION_START-}" "${ALLOCATION_END-}" "${NOW-${PHASE_NOW-}}" \
    "${SHELL_STARTED_EPOCH-}" "${PHASE_EPOCH-}" "${phase_leader-}" "${PHASE_LEADER_PGID-}" \
    "${ENTRY-${PAID_CHILD_ENTRY-}}" "${CHILD_PYTHON-${PAID_CHILD_PYTHON-}}" "${CUDA_DRIVER-}" "${NVML_DRIVER-}" "${TOOLCHAIN-}" >&2
}
paid_startup_shell_exit() {
  local code=$?
  if ((code != 0)); then paid_startup_shell_evidence "${PAID_STARTUP_STAGE:-entry}_rejected" "$code"; fi
}
PAID_STARTUP_STAGE=entry
if [[ ${1:-} == --runtime-child ]]; then
  PAID_STARTUP_STAGE=runtime_child
  PAID_CHILD_PYTHON=${2-}
  PAID_CHILD_ENTRY=${3-}
fi
trap paid_startup_shell_exit EXIT
paid_startup_shell_evidence "$PAID_STARTUP_STAGE"
if [[ ${1:-} == --runtime-child ]]; then
  [[ ${ORCJAX_AFFINE_GPU_EXECUTION_CONTRACT-} == T7_affine_allocated_single_gpu_v1 && -n ${ORCJAX_AFFINE_EXPECTED_GPU_UUID-} ]] || exit 2
  # The paid supervisor supplies this only after exact allocation mapping.
  export SINGULARITYENV_ORCJAX_AFFINE_EXPECTED_GPU_UUID=$ORCJAX_AFFINE_EXPECTED_GPU_UUID
  export SINGULARITYENV_ORCJAX_AFFINE_GPU_EXECUTION_CONTRACT=$ORCJAX_AFFINE_GPU_EXECUTION_CONTRACT
fi
if [[ ${1:-} != --runtime-child ]]; then
  # Preserve scheduler-visible tokens before the CPU-only main clears CUDA.
  export ORCJAX_SLURM_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}
  DRY=false
  for ARG in "$@"; do [[ $ARG != --dry-run ]] || DRY=true; done
  if [[ $DRY == false ]]; then
    PAID_STARTUP_STAGE=scheduler_allocation_clock
    # CPU-only scheduler metadata resolves the true allocation clock before
    # outer deadlines. 22.05.7 does not promise JOB_START/END_TIME env tokens.
    # Original tokens are crosschecked when present, never manufactured.
    [[ ${SLURM_JOB_ID:-} =~ ^[0-9]+$ ]] || exit 2
    [[ -z ${ORCJAX_NUMERICS_LAUNCHER_PYTHON:-} ]] || exit 2
    EPOCHS=$(timeout 10s "$CPU_PYTHON" -B "$RUNNER" --scheduler-allocation-epochs) || exit 2
    [[ $EPOCHS =~ ^[0-9]+[[:blank:]][0-9]+$ ]] || exit 2
    read -r ALLOCATION_START ALLOCATION_END EXTRA <<< "$EPOCHS"
    [[ $ALLOCATION_START =~ ^[0-9]+$ && $ALLOCATION_END =~ ^[0-9]+$ && -z $EXTRA ]] || exit 2
    NOW=$(date +%s)
    ((ALLOCATION_START <= NOW && NOW < ALLOCATION_END && ALLOCATION_END-ALLOCATION_START == 1200)) || exit 2
    SHELL_STARTED_EPOCH=$ALLOCATION_START
    paid_startup_shell_evidence scheduler_clock_admitted
  fi
fi
main "$@"
