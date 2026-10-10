#!/bin/bash
# Free-node saved-binary diagnostics only; full mode requires admitted recurrent gates.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT="$ROOT/runtime/checkouts/shared-split-width-v1"
OVERLAY="$ROOT/runtime/transfer/shared-normalized-keys-20261001"
DIAG="$ROOT/runtime/outputs/diagnostics"
UPDATES="$ROOT/runtime/transfer/shared-update-changing-slice-20260930"
MODE=${1:-}
HORIZON=${2:-7}
DRY_RUN=${3:-}
case "$MODE" in replay|full) ;; *) exit 2 ;; esac
case "$HORIZON" in 7|30) ;; *) exit 2 ;; esac
case "$DRY_RUN" in ''|--dry-run) ;; *) exit 2 ;; esac
GPU_INDEX=${GPU_INDEX:-0}
case "$GPU_INDEX" in 0|1) ;; *) exit 2 ;; esac
REF7="$DIAG/shared-recurrent-checkpoint-policy-20260930-v2-T7"
REF30="$DIAG/shared-normalized-keys-recurrent-T30-20261001"
if [[ $MODE == full ]]; then
  OUTPUT="$DIAG/shared-normalized-keys-full-update-20261001.json"
  BUDGET=1080
  DRIVER=probe_shared_normalized_keys_full_update.py
  EXTRA=(--tree-owner-file "$UPDATES/probe_shared_update_tree_slice.py"
    --slice-helper-file "$UPDATES/shared_daily_microbatch.py" --learning-runner-file "$UPDATES/run_shared_learning.py"
    --normalized-daily-report "$DIAG/shared-normalized-keys-real-daily-micro4-20261001.json"
    --normalized-t7-report "$DIAG/shared-normalized-keys-recurrent-T7-20261001-candidate.json"
    --normalized-t30-report "$DIAG/shared-normalized-keys-recurrent-T30-20261001-candidate.json"
    --t7-reference-report "$REF7-reference.json" --t7-reference-executable "$REF7.reference.executable.pkl"
    --t30-reference-report "$REF30-reference.json" --t30-reference-executable "$REF30.reference.executable.pkl")
else
  OUTPUT="$DIAG/shared-normalized-keys-recurrent-T$HORIZON-20261001-replay.json"
  BUDGET=420
  DRIVER=probe_shared_normalized_keys_recurrent_replay.py
  REFERENCE="$REF30"
  if [[ $HORIZON == 7 ]]; then REFERENCE="$REF7"; fi
  EXTRA=(--horizon "$HORIZON" --repeats 7
    --recurrent-report "$DIAG/shared-normalized-keys-recurrent-T$HORIZON-20261001-candidate.json"
    --reference-executable "$REFERENCE.reference.executable.pkl"
    --patch-owner "$ROOT/runtime/transfer/shared-combined-real-daily-20260925/probe_shared_expert_real_daily.py"
    --gate-owner "$ROOT/runtime/transfer/shared-union-micro4-20260929/probe_shared_union_micro4.py"
    --forward-probe "$ROOT/runtime/transfer/shared-forward-reverse-20260930/probe_shared_forward_reverse.py"
    --prepared-context "$DIAG/shared-expert-real-daily-context-c599236.pkl"
    --runtime-preflight "$DIAG/shared-normalized-keys-runtime-20261001.json")
fi
[[ $(git --git-dir="$CODE_ROOT/.git" --work-tree="$CODE_ROOT" rev-parse HEAD) == c599236bcf57b745e9bafed678298010f4c3b58f ]] || exit 2
if [[ $DRY_RUN != --dry-run ]]; then
  [[ $(hostname -s) == ibc13b02n04 && -z ${SLURM_JOB_ID:-} && ! -e "$OUTPUT" ]] || exit 2
  LINE=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  read -r MEMORY UTIL <<<"${LINE//,/ }"
  [[ $MEMORY =~ ^[0-9]+$ && $UTIL =~ ^[0-9]+$ && $MEMORY -le 2000 && $UTIL -le 5 ]] || exit 2
  [[ -z $(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader,nounits) ]] || exit 2
fi
CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
COMMAND=(timeout --signal=TERM --kill-after=10s "$BUDGET" env -u LD_LIBRARY_PATH
  SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU_INDEX"
  SINGULARITYENV_JAX_PLATFORMS=cuda,cpu SINGULARITYENV_JAX_ENABLE_X64=true SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
  SINGULARITYENV_JAX_COMPILATION_CACHE_DIR="$ROOT/runtime/cache/jax/orcjax_gpu/shared-update-tree-slice-20260929"
  SINGULARITYENV_ORCHIDEE_REPO_ROOT="$CODE_ROOT" SINGULARITYENV_PYTHONPATH="$CODE_ROOT:$OVERLAY"
  SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1 SINGULARITYENV_OPENBLAS_NUM_THREADS=1 SINGULARITYENV_OMP_NUM_THREADS=4
  singularity exec --nv --bind "/WORK:/WORK,$ROOT/runtime/data:$ROOT/data:ro,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
  --pwd "$CODE_ROOT" /apps/soft/sif/foundationpose "$ROOT/.venvs/orcjax_gpu/bin/python" "$OVERLAY/$DRIVER"
  --source-profile "$DIAG/shared-split-profile-c599236.json" --input-bundle "$DIAG/shared-split-dual-inputs-c599236.pkl"
  --daily-admission-report "$DIAG/shared-combined-real-daily-20260925.json"
  --seven-admission-report "$DIAG/shared-combined-recurrent-T7-20260926.json"
  --thirty-admission-report "$DIAG/shared-combined-recurrent-T30-20260926.json"
  --support-owner-file "$ROOT/runtime/transfer/shared-combined-full-update-20260926/probe_shared_serialized_dual.py"
  --output "$OUTPUT" "${EXTRA[@]}")
if [[ $DRY_RUN == --dry-run ]]; then printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0; fi
EXIT_RECORD="${OUTPUT%.json}.exit"
STARTED_AT=$(date --iso-8601=seconds)
printf 'started_at=%s\nphase=model\n' "$STARTED_AT" > "$EXIT_RECORD"
trap 'STATUS=$?; printf "started_at=%s\nended_at=%s\nmodel_process_exit_status=%s\n" "$STARTED_AT" "$(date --iso-8601=seconds)" "$STATUS" > "$EXIT_RECORD.tmp"; mv -f "$EXIT_RECORD.tmp" "$EXIT_RECORD"; printf "run_exit_status=%s\n" "$STATUS"' EXIT
date --iso-8601=seconds
hostname
uptime
grep Cpus_allowed_list /proc/self/status
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
"${COMMAND[@]}"
