#!/bin/bash
# Bounded free-node diagnostics only; no formal training or paid submission.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT="$ROOT/runtime/checkouts/shared-split-width-v1"
OVERLAY="$ROOT/runtime/transfer/shared-normalized-keys-20261001"
DIAG="$ROOT/runtime/outputs/diagnostics"
COMBINED="$ROOT/runtime/transfer/shared-combined-recurrent-20260926"
GPU_INDEX=${GPU_INDEX:-0}
DRY_RUN=0
if [[ ${1:-} == --dry-run ]]; then DRY_RUN=1; shift; fi
MODE=${1:-capture}
MICRO=${2:-1}
case "$MODE" in preflight|capture|screen|replay|unroll2|headorder) ;; *) exit 2 ;; esac
case "$MICRO" in 1|4) ;; *) exit 2 ;; esac
case "$GPU_INDEX" in 0|1) ;; *) exit 2 ;; esac
[[ $(git --git-dir="$CODE_ROOT/.git" --work-tree="$CODE_ROOT" rev-parse HEAD) == c599236bcf57b745e9bafed678298010f4c3b58f ]] || exit 2
if [[ $DRY_RUN == 0 ]]; then
  [[ $(hostname -s) == ibc13b02n04 ]] || exit 2
  LINE=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  read -r MEMORY UTIL <<<"${LINE//,/ }"
  [[ $MEMORY =~ ^[0-9]+$ && $UTIL =~ ^[0-9]+$ && $MEMORY -le 2000 && $UTIL -le 5 ]] || exit 2
  [[ -z $(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader,nounits) ]] || exit 2
fi
CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
RUNTIME=(env -u LD_LIBRARY_PATH SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs
  SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU_INDEX" SINGULARITYENV_JAX_PLATFORMS=cuda,cpu
  SINGULARITYENV_JAX_ENABLE_X64=true SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
  SINGULARITYENV_ORCHIDEE_REPO_ROOT="$CODE_ROOT" SINGULARITYENV_PYTHONPATH="$CODE_ROOT:$OVERLAY"
  SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1 SINGULARITYENV_OPENBLAS_NUM_THREADS=1 SINGULARITYENV_OMP_NUM_THREADS=4
  singularity exec --nv --bind "/WORK:/WORK,$ROOT/runtime/data:$ROOT/data:ro,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
  --pwd "$CODE_ROOT" /apps/soft/sif/foundationpose "$ROOT/.venvs/orcjax_gpu/bin/python")
if [[ $MODE == replay ]]; then
  COMMAND=(timeout --signal=TERM --kill-after=10s 360s "${RUNTIME[@]}" "$OVERLAY/probe_shared_normalized_key_replay.py"
    --screen-report "$DIAG/shared-normalized-keys-screen-micro$MICRO-20261001.json"
    --payload "$DIAG/shared-normalized-keys-payload-20261001.pkl"
    --output "$DIAG/shared-normalized-keys-replay-micro$MICRO-20261001.json")
elif [[ $MODE == preflight ]]; then
  COMMAND=(timeout --signal=TERM --kill-after=10s 180s "${RUNTIME[@]}" "$CODE_ROOT/scripts/hpc/check_daily_training_runtime.py"
    --backend gpu --output "$DIAG/shared-normalized-keys-runtime-20261001.json")
else
  ENTRYPOINT=probe_shared_normalized_keys.py
  PROCESS_BUDGET=900s
  OUTPUT="$DIAG/shared-normalized-keys-$MODE-micro$MICRO-20261001.json"
  if [[ $MODE == headorder ]]; then
    [[ $MICRO == 1 ]] || exit 2
    ENTRYPOINT=probe_shared_normalized_key_head_order.py
    PROCESS_BUDGET=300s
    OUTPUT="$DIAG/shared-normalized-key-head-order-micro1-20261001.json"
  fi
  if [[ $MODE == unroll2 ]]; then
    ENTRYPOINT=probe_shared_decoder_chunk_unroll.py
    OUTPUT="$DIAG/shared-decoder-chunk-unroll2-micro$MICRO-20261001.json"
  fi
  COMMAND=(timeout --signal=TERM --kill-after=10s "$PROCESS_BUDGET" "${RUNTIME[@]}" "$OVERLAY/$ENTRYPOINT"
    --mode "$([[ $MODE == unroll2 || $MODE == headorder ]] && echo screen || echo "$MODE")" --microbatch "$MICRO" --repeats 7
    --output "$OUTPUT"
    --payload "$DIAG/shared-normalized-keys-payload-20261001.pkl"
    --patch-owner "$ROOT/runtime/transfer/shared-combined-real-daily-20260925/probe_shared_expert_real_daily.py")
  if [[ $MODE == capture ]]; then
    COMMAND+=(--source-profile "$DIAG/shared-split-profile-c599236.json"
      --prepared-context "$DIAG/shared-expert-real-daily-context-c599236.pkl"
      --candidate-module-file "$COMBINED/shared_decoder_expert_execution.py"
      --backbone-module-file "$COMBINED/shared_backbone_execution.py")
  fi
  if [[ $MODE == unroll2 ]]; then COMMAND+=(--variant unroll2 --routes expert2 expert5); fi
fi
if [[ $DRY_RUN == 1 ]]; then printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0; fi
trap 'STATUS=$?; printf "run_exit_status=%s\n" "$STATUS"' EXIT
date --iso-8601=seconds
hostname
uptime
grep Cpus_allowed_list /proc/self/status
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
"${COMMAND[@]}"
