#!/usr/bin/env bash
# Paid allocation admission belongs to the paid Python supervisor. Reuse the
# accepted deadlines and runtime child; do not initialize a GPU in this entry.
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/run_shared_latent_T7_batch_numerics.sh"
RUNNER=$CODE/scripts/hpc/run_shared_latent_T7_batch_numerics_paid.py
if [[ ${1:-} == --runtime-child && ${ORCJAX_AFFINE_EXPECTED_GPU_UUID+x} == x ]]; then
  # The paid supervisor supplies this only after exact allocation mapping.
  export SINGULARITYENV_ORCJAX_AFFINE_EXPECTED_GPU_UUID=$ORCJAX_AFFINE_EXPECTED_GPU_UUID
fi
if [[ ${1:-} != --runtime-child ]]; then
  # Preserve scheduler-visible tokens before the CPU-only main clears CUDA.
  export ORCJAX_SLURM_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}
  DRY=false
  for ARG in "$@"; do [[ $ARG != --dry-run ]] || DRY=true; done
  if [[ $DRY == false ]]; then
    # Allocation occupancy starts before this shell. The Python supervisor
    # cross-checks these exact epochs with scontrol before any GPU admission.
    [[ -n ${SLURM_JOB_ID:-} && ${SLURM_JOB_START_TIME:-} =~ ^[0-9]+$ && ${SLURM_JOB_END_TIME:-} =~ ^[0-9]+$ ]] || exit 2
    NOW=$(date +%s)
    ((SLURM_JOB_START_TIME <= NOW && NOW < SLURM_JOB_END_TIME && SLURM_JOB_END_TIME > SLURM_JOB_START_TIME && SLURM_JOB_END_TIME-SLURM_JOB_START_TIME <= 1200)) || exit 2
    SHELL_STARTED_EPOCH=$SLURM_JOB_START_TIME
  fi
fi
main "$@"
