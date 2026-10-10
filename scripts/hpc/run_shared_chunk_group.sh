#!/bin/bash
# Free gln01 GPU0, native G2 diagnostic only; never submits a scheduler job.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT="$ROOT/runtime/checkouts/shared-split-width-v1"
OVERLAY="$ROOT/runtime/transfer/shared-chunk-group-20261001"
DIAG="$ROOT/runtime/outputs/diagnostics"
RUN_ID=${RUN_ID:-20261001-g2}
[[ $RUN_ID =~ ^[a-zA-Z0-9-]+$ ]] || exit 2
DRY_RUN=0
if [[ ${1:-} == --dry-run ]]; then DRY_RUN=1; shift; fi
OUTPUT="$DIAG/shared-chunk-group-component-$RUN_ID.json"
LAUNCH_STARTED=$(date +%s.%N)
GPU_STARTED=false
finish_record() {
  STATUS=$?
  [[ $DRY_RUN == 0 ]] || return 0
  LAUNCH_FINISHED=$(date +%s.%N)
  {
    printf 'gpu_started=%s\npaid_jobs=0\ngpu_count=%s\nexit_status=%s\n' "$GPU_STARTED" "$([[ $GPU_STARTED == true ]] && echo 1 || echo 0)" "$STATUS"
    awk -v start="$LAUNCH_STARTED" -v finish="$LAUNCH_FINISHED" -v started="$GPU_STARTED" 'BEGIN {printf "launcher_wall_seconds=%.9f\nlauncher_card_hours=%.12f\n", finish-start, (started=="true" ? (finish-start)/3600 : 0)}'
  } > "$OUTPUT.launcher.txt"
}
trap finish_record EXIT
[[ $(git --git-dir="$CODE_ROOT/.git" --work-tree="$CODE_ROOT" rev-parse HEAD) == c599236bcf57b745e9bafed678298010f4c3b58f ]] || exit 2
if [[ $DRY_RUN == 0 ]]; then
  [[ $(hostname -s) == ibc13b02n04 && -z ${SLURM_JOB_ID:-} && ! -e $OUTPUT && ! -e $OUTPUT.launcher.txt ]] || exit 2
  LINE=$(nvidia-smi -i 0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  read -r MEMORY UTIL <<<"${LINE//,/ }"
  [[ $MEMORY =~ ^[0-9]+$ && $UTIL =~ ^[0-9]+$ && $MEMORY -le 2000 && $UTIL -le 5 ]] || exit 2
  [[ -z $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits) ]] || exit 2
fi
CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
COMMAND=(timeout --signal=TERM --kill-after=10s 900s env -u LD_LIBRARY_PATH -u LD_PRELOAD
  SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs SINGULARITYENV_CUDA_VISIBLE_DEVICES=0
  SINGULARITYENV_JAX_PLATFORMS=cuda,cpu SINGULARITYENV_JAX_ENABLE_X64=true
  SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false SINGULARITYENV_ORCHIDEE_REPO_ROOT="$CODE_ROOT"
  SINGULARITYENV_PYTHONPATH="$CODE_ROOT:$OVERLAY" SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
  SINGULARITYENV_PYTHONFAULTHANDLER=1 SINGULARITYENV_OPENBLAS_NUM_THREADS=1 SINGULARITYENV_OMP_NUM_THREADS=4
  SINGULARITYENV_XLA_FLAGS="--xla_gpu_enable_llvm_module_compilation_parallelism=false --xla_gpu_enable_command_buffer=FUSION,CUBLAS,CUSTOM_CALL,CUDNN"
  singularity exec --nv --bind "/WORK:/WORK,$ROOT/runtime/data:$ROOT/data:ro,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
  --pwd "$CODE_ROOT" /apps/soft/sif/foundationpose "$ROOT/.venvs/orcjax_gpu/bin/python"
  "$OVERLAY/probe_shared_chunk_group_component.py"
  --payload "$DIAG/shared-normalized-keys-payload-20261001.pkl"
  --default-report "$DIAG/shared-command-buffer-loops-threshold-component-default-20261001.json"
  --output "$OUTPUT")
if [[ $DRY_RUN == 1 ]]; then printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0; fi
date --iso-8601=seconds
hostname
uptime
grep Cpus_allowed_list /proc/self/status
nvidia-smi --query-gpu=index,name,driver_version,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
GPU_STARTED=true
"${COMMAND[@]}"
