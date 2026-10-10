#!/bin/bash
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT="$ROOT/runtime/checkouts/shared-split-width-v1"
OVERLAY="$ROOT/runtime/transfer/shared-recurrent-checkpoint-policy-20260930-v5"
DIAG="$ROOT/runtime/outputs/diagnostics"
REFERENCE_BASE="$DIAG/shared-recurrent-checkpoint-policy-20260930-v2-T7"
CANDIDATE_BASE="$DIAG/shared-recurrent-checkpoint-policy-20260930-v3-T7"
OUTPUT="$DIAG/shared-recurrent-checkpoint-policy-20260930-v5-T7-replay.json"
DRY_RUN=${1:-}
case "$DRY_RUN" in ''|--dry-run) ;; *) exit 2 ;; esac
GPU_INDEX=${GPU_INDEX:-0}
case "$GPU_INDEX" in 0|1) ;; *) exit 2 ;; esac
[[ $(git --git-dir="$CODE_ROOT/.git" --work-tree="$CODE_ROOT" rev-parse HEAD) == c599236bcf57b745e9bafed678298010f4c3b58f ]] || exit 2
[[ -f "$OVERLAY/probe_shared_recurrent_checkpoint_replay.py" ]] || exit 2
for asset in "$REFERENCE_BASE-reference.json" "$REFERENCE_BASE.reference.pkl" \
  "$REFERENCE_BASE.reference.executable.pkl" "$CANDIDATE_BASE-candidate.json" \
  "$CANDIDATE_BASE-candidate.result.pkl" "$CANDIDATE_BASE-candidate.executable.pkl"; do
  [[ -f "$asset" ]] || exit 2
done
if [[ "$DRY_RUN" != --dry-run ]]; then
  [[ -z ${SLURM_JOB_ID:-} && $(hostname -s) == ibc13b02n04 ]] || exit 2
  GPU_LINE=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  read -r GPU_MEMORY GPU_UTIL <<<"${GPU_LINE//,/ }"
  [[ $GPU_MEMORY =~ ^[0-9]+$ && $GPU_UTIL =~ ^[0-9]+$ && $GPU_MEMORY -le 2000 && $GPU_UTIL -le 5 ]] || exit 2
  [[ -z $(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader,nounits) ]] || exit 2
fi
CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
RUNTIME=(env -u LD_LIBRARY_PATH
  SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs
  SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU_INDEX"
  SINGULARITYENV_JAX_PLATFORMS=cuda,cpu
  SINGULARITYENV_JAX_ENABLE_X64=true
  SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
  SINGULARITYENV_JAX_COMPILATION_CACHE_DIR="$ROOT/runtime/cache/jax/orcjax_gpu/shared-checkpoint-policy-20260930-v2"
  SINGULARITYENV_ORCHIDEE_REPO_ROOT="$CODE_ROOT"
  SINGULARITYENV_PYTHONPATH="$CODE_ROOT"
  SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
  SINGULARITYENV_OPENBLAS_NUM_THREADS=1
  SINGULARITYENV_OMP_NUM_THREADS=4
  singularity exec --nv --bind
  "/WORK:/WORK,$ROOT/runtime/data:$ROOT/data:ro,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
  --pwd "$CODE_ROOT" /apps/soft/sif/foundationpose
  "$ROOT/.venvs/orcjax_gpu/bin/python")
COMMAND=(timeout --signal=TERM --kill-after=10s 900s "${RUNTIME[@]}"
  "$OVERLAY/probe_shared_recurrent_checkpoint_replay.py"
  --horizon 7 --repeats 3
  --source-profile "$DIAG/shared-split-profile-c599236.json"
  --input-bundle "$DIAG/shared-split-dual-inputs-c599236.pkl"
  --checkpoint "$DIAG/shared-split-profile-c599236.updates.update-000000.checkpoint.pkl"
  --reference-report "$REFERENCE_BASE-reference.json"
  --reference-result "$REFERENCE_BASE.reference.pkl"
  --reference-executable "$REFERENCE_BASE.reference.executable.pkl"
  --candidate-report "$CANDIDATE_BASE-candidate.json"
  --candidate-result "$CANDIDATE_BASE-candidate.result.pkl"
  --candidate-executable "$CANDIDATE_BASE-candidate.executable.pkl"
  --output "$OUTPUT")
if [[ "$DRY_RUN" == --dry-run ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi
exec > >(tee "${OUTPUT%.json}.log") 2>&1
STARTED_AT=$(date --iso-8601=seconds)
record_exit() {
  local status=$?
  trap - EXIT
  printf 'started_at=%s\nended_at=%s\nlauncher_exit_status=%s\n' \
    "$STARTED_AT" "$(date --iso-8601=seconds)" "$status" > "${OUTPUT%.json}.exit.tmp"
  mv -f "${OUTPUT%.json}.exit.tmp" "${OUTPUT%.json}.exit"
  printf 'launcher_exit_status=%s\n' "$status"
}
trap record_exit EXIT
date --iso-8601=seconds
hostname
uptime
grep Cpus_allowed_list /proc/self/status
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
"${COMMAND[@]}"
