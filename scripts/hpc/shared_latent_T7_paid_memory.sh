#!/bin/bash
# Prepared paid route only: read-only single allocated GPU observations within phase process group.
set -euo pipefail
GPU=${1:?single physical GPU required}
LOG=${2:?memory receipt required}
shift 2
[[ ${1:-} == -- ]] || exit 2
shift
[[ $GPU =~ ^[0-9]+$ || $GPU =~ ^GPU-[a-fA-F0-9-]+$ ]] || exit 2
case "$LOG" in /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/latent-reader-T7-20261005-paid-*.memory.csv) ;; *) exit 2 ;; esac
PROCESS_LOG="$LOG.processes.csv"
[[ ! -e $LOG && ! -e $PROCESS_LOG && $# -gt 0 ]] || exit 2
printf 'epoch_seconds,memory_used_mib,utilization_percent\n' > "$LOG"
printf 'epoch_seconds,pid,memory_used_mib,process_name\n' > "$PROCESS_LOG"
sample() {
  local line epoch processes
  epoch=$(date +%s)
  line=$(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  printf '%s,%s\n' "$epoch" "$line" >> "$LOG"
  processes=$(nvidia-smi -i "$GPU" --query-compute-apps=pid,used_gpu_memory,process_name --format=csv,noheader,nounits)
  if [[ -n $processes ]]; then
    while IFS= read -r line; do printf '%s,%s\n' "$epoch" "$line" >> "$PROCESS_LOG"; done <<< "$processes"
  else
    printf '%s,none,0,none\n' "$epoch" >> "$PROCESS_LOG"
  fi
}
sample
(while true; do sample; sleep 2; done) &
MONITOR_PID=$!
RUN_PID=
cleanup() {
  kill -TERM "$MONITOR_PID" 2>/dev/null || true
  if [[ -n $RUN_PID ]]; then kill -TERM "$RUN_PID" 2>/dev/null || true; fi
  wait "$MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
"$@" &
RUN_PID=$!
set +e
wait "$RUN_PID"
RESULT=$?
set -e
RUN_PID=
cleanup
sample
exit "$RESULT"
