#!/bin/bash
set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT="$ROOT/runtime/checkouts/shared-split-width-v1"
OVERLAY="$ROOT/runtime/transfer/shared-chunk-vmap-daily-20261001"
OLD_OVERLAY="$ROOT/runtime/transfer/shared-command-buffer-loops-20261001"
DIAG="$ROOT/runtime/outputs/diagnostics"
RUN_ID=${RUN_ID:-20261001-v1}
GPU_INDEX=${GPU_INDEX:-1}
DRY_RUN=0
if [[ ${1:-} == --dry-run ]]; then DRY_RUN=1; shift; fi
[[ ${1:-} == expert5-micro4 && $# == 1 ]] || exit 2
[[ $RUN_ID =~ ^[a-zA-Z0-9-]+$ && $GPU_INDEX == 1 ]] || exit 2

OUTPUT="$DIAG/shared-chunk-vmap-daily-$1-$RUN_ID.json"
LAUNCHER_RECORD="$OUTPUT.launcher.txt"
[[ $(git --git-dir="$CODE_ROOT/.git" --work-tree="$CODE_ROOT" rev-parse HEAD) == c599236bcf57b745e9bafed678298010f4c3b58f ]] || exit 2
[[ -f $OVERLAY/shared_chunk_vmap_execution.py && -f $OVERLAY/probe_shared_chunk_vmap_daily.py ]] || exit 2
[[ -f $OLD_OVERLAY/probe_shared_normalized_keys_daily.py ]] || exit 2
[[ -f $DIAG/shared-command-buffer-loops-runtime-preflight-default-20261001-vmap.public-preflight.json ]] || exit 2
[[ -f $DIAG/shared-split-profile-c599236.json && -f $DIAG/shared-split-dual-inputs-c599236.pkl ]] || exit 2
[[ -f $DIAG/shared-expert-real-daily-context-c599236.pkl && -f $DIAG/shared-combined-real-daily-20260925.json ]] || exit 2

finish_record() {
  local status=$?
  local finish
  finish=$(date +%s.%N)
  if [[ $DRY_RUN == 1 ]]; then return; fi
  {
    printf 'variant=expert5_whole_chunk_axis_native_vmap\ngpu_index=%s\npaid_jobs=0\nphase=%s\nstart_epoch=%s\nfinish_epoch=%s\nexit_status=%s\n' "$GPU_INDEX" "$PHASE" "$START" "$finish" "$status"
    printf 'observed_gpu_memory_util=%s\n' "${GPU_LINE:-not_checked}"
    awk -v a="$START" -v b="$finish" -v started="$GPU_STARTED" 'BEGIN {printf "launcher_wall_seconds=%.9f\ngpu_count=%d\ncard_hours=%.12f\n", b-a, (started=="true"?1:0), (started=="true"?(b-a)/3600:0)}'
  } > "$LAUNCHER_RECORD"
  printf 'run_exit_status=%s\n' "$status"
}

START=$(date +%s.%N)
GPU_STARTED=false
PHASE=validating_node_and_gpu
trap finish_record EXIT
if [[ $DRY_RUN == 0 ]]; then
  [[ $(hostname -s) == ibc13b02n04 && -z ${SLURM_JOB_ID:-} && ! -e $OUTPUT && ! -e $LAUNCHER_RECORD ]] || exit 2
  GPU_LINE=$(nvidia-smi -i 1 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  read -r MEMORY UTIL <<<"${GPU_LINE//,/ }"
  [[ $MEMORY =~ ^[0-9]+$ && $UTIL =~ ^[0-9]+$ && $MEMORY -le 2000 && $UTIL -le 5 ]] || exit 2
  [[ -z $(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader,nounits) ]] || exit 2
fi

CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
[[ -f $CUDA_DRIVER && -f $NVML_DRIVER ]]
COMMAND=(timeout --signal=TERM --kill-after=10s 2110s env -u LD_LIBRARY_PATH -u LD_PRELOAD
  SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs
  SINGULARITYENV_CUDA_VISIBLE_DEVICES=1
  SINGULARITYENV_JAX_PLATFORMS=cuda,cpu SINGULARITYENV_JAX_ENABLE_X64=true
  SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
  SINGULARITYENV_ORCHIDEE_REPO_ROOT="$CODE_ROOT"
  SINGULARITYENV_PYTHONPATH="$CODE_ROOT:$OVERLAY"
  SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1 SINGULARITYENV_PYTHONFAULTHANDLER=1
  SINGULARITYENV_OPENBLAS_NUM_THREADS=1 SINGULARITYENV_OMP_NUM_THREADS=4
  SINGULARITYENV_XLA_FLAGS=--xla_gpu_enable_llvm_module_compilation_parallelism=false
  SINGULARITYENV_SHARED_DAILY_PROBE="$OLD_OVERLAY/probe_shared_normalized_keys_daily.py"
  singularity exec --nv --bind "/WORK:/WORK,$ROOT/runtime/data:$ROOT/data:ro,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
  --pwd "$CODE_ROOT" /apps/soft/sif/foundationpose "$ROOT/.venvs/orcjax_gpu/bin/python"
  "$OVERLAY/probe_shared_chunk_vmap_daily.py"
  --output "$OUTPUT" --lower-seconds 240 --backend-seconds 1500 --process-seconds 2100
  --patch-owner "$ROOT/runtime/transfer/shared-combined-real-daily-20260925/probe_shared_expert_real_daily.py"
  --gate-owner "$ROOT/runtime/transfer/shared-union-micro4-20260929/probe_shared_union_micro4.py"
  --source-profile "$DIAG/shared-split-profile-c599236.json"
  --input-bundle "$DIAG/shared-split-dual-inputs-c599236.pkl"
  --prepared-context "$DIAG/shared-expert-real-daily-context-c599236.pkl"
  --baseline-admission "$DIAG/shared-combined-real-daily-20260925.json"
  --runtime-preflight "$DIAG/shared-command-buffer-loops-runtime-preflight-default-20261001-vmap.public-preflight.json"
  --candidate-module-file "$ROOT/runtime/transfer/shared-combined-recurrent-20260926/shared_decoder_expert_execution.py"
  --backbone-module-file "$ROOT/runtime/transfer/shared-combined-recurrent-20260926/shared_backbone_execution.py"
  --execution-module-file "$OVERLAY/shared_chunk_vmap_execution.py")

if [[ $DRY_RUN == 1 ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi
PHASE=running_gpu_command
GPU_STARTED=true
date --iso-8601=seconds
hostname
uptime
grep Cpus_allowed_list /proc/self/status
nvidia-smi --query-gpu=index,name,driver_version,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
"${COMMAND[@]}"
