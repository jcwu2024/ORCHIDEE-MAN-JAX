#!/bin/bash
# Original-only saved full-state schedule; fixed original funding route, free gln01.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT="$ROOT/runtime/checkouts/shared-split-width-v1"
OVERLAY="$ROOT/runtime/transfer/shared-latent-reader-20261004"
DIAG="$ROOT/runtime/outputs/diagnostics"
LEDGER="$DIAG/shared-latent-reader-20261004.gpu-budget.json"
DRY_RUN=${1:-}
case "$DRY_RUN" in ''|--dry-run) ;; *) exit 2 ;; esac
MANIFEST=${ORIGINAL_SCHEDULE_MANIFEST:?explicit original-only full-state manifest required}
OUTPUT=${ORIGINAL_SCHEDULE_OUTPUT:-$DIAG/shared-original-full-state-schedule-20261005.json}
SESSION_SECONDS=${ORIGINAL_SCHEDULE_SECONDS:-500}
UPDATES=${ORIGINAL_SCHEDULE_UPDATES:-4}
[[ $SESSION_SECONDS =~ ^[0-9]+$ && $SESSION_SECONDS -ge 60 && $SESSION_SECONDS -le 500 ]] || exit 2
[[ $UPDATES =~ ^[34]$ ]] || exit 2
WALL_SECONDS=$((SESSION_SECONDS - 20))
[[ -f "$MANIFEST" && -f "$OVERLAY/probe_shared_original_full_state_schedule.py" && -f "$OVERLAY/probe_shared_original_full_state_schedule_worker.py" ]] || exit 2
[[ $(cd "$CODE_ROOT" && git rev-parse HEAD) == c599236bcf57b745e9bafed678298010f4c3b58f ]] || exit 2
case "$OUTPUT" in "$DIAG"/shared-original-full-state-schedule-20261005*.json) ;; *) exit 2 ;; esac
[[ ! -e "$OUTPUT" ]] || exit 2
CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
APPROVAL_NOTE=$(/apps/soft/anaconda3/bin/python -c 'import json,sys; data=json.load(open(sys.argv[1])); rows=[r for r in data["budget_authorizations"] if r["limit_gpu_seconds"]==25200]; assert len(rows)==1; print(rows[0]["approval_note"])' "$LEDGER")
COMMAND=(env -u LD_LIBRARY_PATH
  SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs SINGULARITYENV_CUDA_VISIBLE_DEVICES=0
  SINGULARITYENV_JAX_PLATFORMS=cuda,cpu SINGULARITYENV_JAX_ENABLE_X64=true
  SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
  SINGULARITYENV_JAX_COMPILATION_CACHE_DIR="$ROOT/runtime/cache/jax/orcjax_gpu/shared-latent-reader-20261004"
  SINGULARITYENV_JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 SINGULARITYENV_JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
  SINGULARITYENV_ORCHIDEE_REPO_ROOT="$CODE_ROOT" SINGULARITYENV_PYTHONPATH="$CODE_ROOT:$OVERLAY"
  SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1 SINGULARITYENV_OPENBLAS_NUM_THREADS=1 SINGULARITYENV_OMP_NUM_THREADS=4
  singularity exec --nv --bind "/WORK:/WORK,$ROOT/runtime/data:$ROOT/data:ro,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
  --pwd "$CODE_ROOT" /apps/soft/sif/foundationpose "$ROOT/.venvs/orcjax_gpu/bin/python" -B
  "$OVERLAY/probe_shared_original_full_state_schedule.py" --manifest "$MANIFEST" --output "$OUTPUT"
  --parent-cuda-device 0 --worker-cuda-device 1 --updates "$UPDATES" --warmup-proposals 0
  --wall-seconds "$WALL_SECONDS" --save-reserve-seconds 30)
BUDGET=(/apps/soft/anaconda3/bin/python -B "$OVERLAY/shared_gpu_budget.py"
  --ledger "$LEDGER" --stage original-full-state-schedule --seconds "$SESSION_SECONDS" --cards 2
  --approved-limit-gpu-seconds 25200 --approval-note "$APPROVAL_NOTE"
  --funding-scope original_18000 -- "${COMMAND[@]}")
if [[ $DRY_RUN == --dry-run ]]; then
  printf '%q ' "${BUDGET[@]}"
  printf '\n'
  exit 0
fi
[[ $(hostname -s) == ibc13b02n04 && -z ${SLURM_JOB_ID:-} ]] || exit 2
for GPU in 0 1; do
  LINE=$(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  read -r MEMORY UTIL <<<"${LINE//,/ }"
  [[ $MEMORY =~ ^[0-9]+$ && $UTIL =~ ^[0-9]+$ && $MEMORY -le 2000 && $UTIL -le 5 ]] || exit 2
done
[[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits) ]] || exit 2
{
  date -Is
  hostname
  taskset -pc $$
  uptime
  nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
  nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory --format=csv
} | tee "$OUTPUT.launch.txt"
set +e
"${BUDGET[@]}" 2>&1 | tee "$OUTPUT.log"
STATUS=${PIPESTATUS[0]}
set -e
{
  date -Is
  nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory --format=csv
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
} | tee "$OUTPUT.release.txt"
exit "$STATUS"
