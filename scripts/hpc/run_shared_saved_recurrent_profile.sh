#!/bin/bash
# Free GPU profile of admitted recurrent binaries, with no gradient recompilation.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT="$ROOT/runtime/checkouts/shared-split-width-v1"
OVERLAY=${OVERLAY:-"$ROOT/runtime/transfer/shared-saved-recurrent-profile-20260926"}
GPU_INDEX=${GPU_INDEX:-1}
DRY_RUN=0
if [[ ${1:-} == --dry-run ]]; then DRY_RUN=1; shift; fi
case "$GPU_INDEX" in 0|1) ;; *) exit 2 ;; esac
case $(hostname -s) in gln01|ibgln01|ibc13b02n04) ;; *) [[ $DRY_RUN == 1 ]] || exit 2 ;; esac
CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
COMMAND=(timeout --signal=TERM --kill-after=30s 10m env -u LD_LIBRARY_PATH
  SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs
  SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU_INDEX"
  SINGULARITYENV_JAX_PLATFORMS=cuda,cpu SINGULARITYENV_JAX_ENABLE_X64=true
  SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
  SINGULARITYENV_ORCHIDEE_REPO_ROOT="$CODE_ROOT" SINGULARITYENV_PYTHONPATH="$CODE_ROOT"
  SINGULARITYENV_OPENBLAS_NUM_THREADS=1 SINGULARITYENV_OMP_NUM_THREADS=4
  singularity exec --nv --bind "/WORK:/WORK,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
  --pwd "$CODE_ROOT" /apps/soft/sif/foundationpose "$ROOT/.venvs/orcjax_gpu/bin/python"
  "$OVERLAY/profile_shared_saved_recurrent.py"
  --admission-report "$ROOT/runtime/outputs/diagnostics/shared-combined-recurrent-T7-20260926.json"
  --input-bundle "$ROOT/runtime/outputs/diagnostics/shared-split-dual-inputs-c599236.pkl"
  --prepared-context "$ROOT/runtime/outputs/diagnostics/shared-expert-real-daily-context-c599236.pkl"
  --output "$ROOT/runtime/outputs/diagnostics/shared-saved-recurrent-profile-T7-20260926.json" "$@")
if [[ $DRY_RUN == 1 ]]; then printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0; fi
[[ -f "$OVERLAY/profile_shared_saved_recurrent.py" ]] || { echo "staged profiler missing from $OVERLAY" >&2; exit 2; }
[[ -f "$OVERLAY/probe_shared_expert_real_daily.py" ]] || { echo "stage probe_shared_expert_real_daily.py beside the profiler in $OVERLAY" >&2; exit 2; }
trap 'STATUS=$?; echo "run_exit_status=$STATUS"' EXIT
date --iso-8601=seconds
uptime
grep Cpus_allowed_list /proc/self/status
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
"${COMMAND[@]}"
