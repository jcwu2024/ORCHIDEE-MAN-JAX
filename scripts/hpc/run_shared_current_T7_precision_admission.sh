#!/usr/bin/env bash
# CPU supervisor and accepted container bootstrap; no submission or scope creation.
set -euo pipefail
SELF=$(readlink -f -- "${BASH_SOURCE[0]}")
CURRENT_CODE=$(cd -- "$(dirname -- "$SELF")/../.." && pwd -P)
source "$CURRENT_CODE/scripts/hpc/run_shared_latent_T7_batch_numerics.sh"
CODE=$CURRENT_CODE
RUNNER=$CODE/scripts/hpc/run_shared_current_T7_precision_admission.py

if [[ ${1:-} == --runtime-child ]]; then
  shift
  [[ $# -ge 2 && $1 == "$GPU_PYTHON" ]] || exit 2
  CHILD_PYTHON=$1
  ENTRY=$2
  shift 2
  case "$ENTRY" in
    "$CODE/scripts/hpc/shared_affine_cuda_identity.py"|"$CODE/scripts/hpc/check_daily_training_runtime.py"|"$CODE/scripts/hpc/probe_shared_current_T7_precision_admission.py") ;;
    *) exit 2 ;;
  esac
  [[ -f $ENTRY ]] || exit 2
  if [[ $ENTRY == "$CODE/scripts/hpc/probe_shared_current_T7_precision_admission.py" ]]; then
    [[ ${ORCJAX_CURRENT_T7_SUPERVISED:-} == 1 ]] || exit 2
  fi
  START=${CURRENT_T7_SHELL_STARTED_EPOCH:-}
  [[ $START =~ ^[0-9]+$ ]] || exit 2
  NOW=$(date +%s)
  ((START <= NOW + 1 && NOW < START + 5220)) || exit 2
  # The CPU supervisor owns this session and its phase-specific deadline.
  # This independent fallback leaves the final 180 seconds for release.
  start_phase_deadman "$$" "$((START+5220))" "$((START+5380))" || exit 2
  [[ -n ${CUDA_VISIBLE_DEVICES-} && $CUDA_VISIBLE_DEVICES != *,* ]] || exit 2
  [[ -n ${ORCJAX_SLURM_VISIBLE_DEVICES-} && $ORCJAX_SLURM_VISIBLE_DEVICES != *,* ]] || exit 2
  if [[ $ENTRY == "$CODE/scripts/hpc/shared_affine_cuda_identity.py" ]]; then
    [[ $CUDA_VISIBLE_DEVICES == "$ORCJAX_SLURM_VISIBLE_DEVICES" && -z ${ORCJAX_AFFINE_EXPECTED_GPU_UUID-} ]] || exit 2
  else
    [[ ${ORCJAX_AFFINE_EXPECTED_GPU_UUID-} == GPU-* && $CUDA_VISIBLE_DEVICES == "$ORCJAX_AFFINE_EXPECTED_GPU_UUID" ]] || exit 2
  fi
  CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
  NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
  [[ -f $CUDA_DRIVER && -f $NVML_DRIVER && -d $TOOLCHAIN ]] || exit 2
  if [[ -v CUDA_DEVICE_ORDER ]]; then
    export SINGULARITYENV_CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER
  else
    unset SINGULARITYENV_CUDA_DEVICE_ORDER
  fi
  export SINGULARITYENV_ORCJAX_SLURM_VISIBLE_DEVICES=$ORCJAX_SLURM_VISIBLE_DEVICES
  export SINGULARITYENV_SLURM_JOB_GPUS=${SLURM_JOB_GPUS-}
  export SINGULARITYENV_SLURM_JOB_ID=${SLURM_JOB_ID-}
  export SINGULARITYENV_SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK-}
  export SINGULARITYENV_CURRENT_T7_SHELL_STARTED_EPOCH=$START
  export SINGULARITYENV_ORCJAX_CURRENT_T7_SUPERVISED=${ORCJAX_CURRENT_T7_SUPERVISED-}
  export SINGULARITYENV_ORCJAX_AFFINE_GPU_EXECUTION_CONTRACT=${ORCJAX_AFFINE_GPU_EXECUTION_CONTRACT-}
  if [[ -n ${ORCJAX_AFFINE_EXPECTED_GPU_UUID-} ]]; then
    export SINGULARITYENV_ORCJAX_AFFINE_EXPECTED_GPU_UUID=$ORCJAX_AFFINE_EXPECTED_GPU_UUID
  else
    unset SINGULARITYENV_ORCJAX_AFFINE_EXPECTED_GPU_UUID
  fi
  exec env -u LD_LIBRARY_PATH \
    SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs \
    SINGULARITYENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    SINGULARITYENV_JAX_PLATFORMS=cuda,cpu \
    SINGULARITYENV_JAX_ENABLE_X64=true \
    SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false \
    SINGULARITYENV_ORCHIDEE_REPO_ROOT="$CODE" \
    SINGULARITYENV_PYTHONPATH="$CODE:$CODE/scripts/hpc" \
    SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1 \
    SINGULARITYENV_OMP_NUM_THREADS=4 \
    SINGULARITYENV_OPENBLAS_NUM_THREADS=1 \
    SINGULARITYENV_PREPEND_PATH="$TOOLCHAIN/bin" \
    singularity exec --nv \
      --bind "/WORK:/WORK,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1,$TOOLCHAIN:$TOOLCHAIN:ro" \
      --pwd "$CODE" /apps/soft/sif/foundationpose \
      "$CHILD_PYTHON" -B "$ENTRY" "$@"
fi

ARGS=()
DRY=false
for ARG in "$@"; do
  if [[ $ARG == --dry-run ]]; then DRY=true; else ARGS+=("$ARG"); fi
done
if [[ $DRY == true ]]; then
  printf '%q ' "$CPU_PYTHON" -B "$RUNNER" "${ARGS[@]}"
  printf '\n'
  exit 0
fi
[[ -f $RUNNER && -x $CPU_PYTHON ]] || exit 2
[[ -z ${CURRENT_T7_LAUNCHER_PYTHON-} ]] || exit 2
export ORCJAX_SLURM_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES-}
export CURRENT_T7_SHELL_STARTED_EPOCH=$(date +%s)
# The CPU owner supplies the original scheduler mask only to its GPU children.
exec env CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu PYTHONDONTWRITEBYTECODE=1 \
  ORCHIDEE_REPO_ROOT="$CODE" PYTHONPATH="$CODE:$CODE/scripts/hpc" \
  "$CPU_PYTHON" -B "$RUNNER" "${ARGS[@]}"
