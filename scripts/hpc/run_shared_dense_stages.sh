#!/bin/bash
# Bounded actual-decoder component probe; no Teacher reads or training updates.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT=${CODE_ROOT:?set immutable production-helper checkout}
PROBE_SCRIPT=${PROBE_SCRIPT:-$CODE_ROOT/scripts/hpc/probe_shared_dense_stages.py}
DENSE_MODULE_FILE=${DENSE_MODULE_FILE:-$CODE_ROOT/research/daily_coarse_graining/daily_operator/shared_decoder_dense.py}
DRY_RUN=0
if [[ ${1:-} == --dry-run ]]; then DRY_RUN=1; shift; fi
if [[ -n ${SLURM_JOB_ID:-} ]]; then
  [[ ${SLURM_JOB_PARTITION:-} == gnall ]]
  DEVICE=${CUDA_VISIBLE_DEVICES:?Slurm must expose the allocated GPU}
else
  case "$(hostname -s)" in gln01|ibgln01|ibc13b02n04) ;; *) [[ $DRY_RUN == 1 ]] || exit 2 ;; esac
  DEVICE=${GPU_DEVICE:?select free GPU for a free-node test}
fi
CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
GPU_BINDS="/WORK:/WORK,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
CACHE_ROOT=$ROOT/runtime/cache/jax/orcjax_gpu/shared-dense-stages-v1
COMMAND=(timeout --signal=TERM "${RUN_TIMEOUT:-10m}" env -u LD_LIBRARY_PATH
  SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs
  SINGULARITYENV_CUDA_VISIBLE_DEVICES="$DEVICE"
  SINGULARITYENV_JAX_ENABLE_X64=true SINGULARITYENV_JAX_PLATFORMS=cuda,cpu
  SINGULARITYENV_XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
  SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
  SINGULARITYENV_JAX_COMPILATION_CACHE_DIR="$CACHE_ROOT"
  SINGULARITYENV_PYTHONPATH="$CODE_ROOT" SINGULARITYENV_ORCHIDEE_REPO_ROOT="$CODE_ROOT"
  SINGULARITYENV_OPENBLAS_NUM_THREADS=1 SINGULARITYENV_OMP_NUM_THREADS=4
  singularity exec --nv --bind "$GPU_BINDS" --pwd "$CODE_ROOT" /apps/soft/sif/foundationpose
  "$ROOT/.venvs/orcjax_gpu/bin/python" "$PROBE_SCRIPT"
  --decoder-module-file "$DENSE_MODULE_FILE")
if [[ $DRY_RUN == 1 ]]; then printf '%q ' "${COMMAND[@]}" "$@"; printf '\n'; exit 0; fi
cd "$CODE_ROOT"
mkdir -p "$CACHE_ROOT"
date
uptime
printf 'host=%s slurm_job=%s allocated_cpus=%s visible_devices=%s\n' \
  "$(hostname -s)" "${SLURM_JOB_ID:-free}" "${SLURM_CPUS_PER_TASK:-unset}" "$DEVICE"
taskset -pc $$
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory --format=csv,noheader
"${COMMAND[@]}" "$@"
