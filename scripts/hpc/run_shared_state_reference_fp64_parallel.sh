#!/bin/bash
# Separately proposed additional compile. Run only after its explicit approval.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT="$ROOT/runtime/checkouts/shared-split-width-v1"
OVERLAY="$ROOT/runtime/transfer/shared-state-reference-fp64-parallel-20261002"
DIAG="$ROOT/runtime/outputs/diagnostics/shared-state-reference-fp64-parallel-20261002"
PRIOR="$ROOT/runtime/outputs/diagnostics/shared-state-reference-fp64-20261002"
TOOLCHAIN="$ROOT/runtime/toolchains/cuda-nvlink-12.6.85"
DRY_RUN=0
if [[ ${1:-} == --dry-run ]]; then DRY_RUN=1; shift; fi
[[ $# -ge 1 && $# -le 2 && ${1:-} =~ ^(preflight|model)$ ]] || exit 2
STAGE=$1
ATTEMPT=${2:-}
[[ -z $ATTEMPT || $ATTEMPT =~ ^[a-z0-9-]+$ ]] || exit 2
OUTPUT="$DIAG/$STAGE${ATTEMPT:+-$ATTEMPT}.json"
START=$(date +%s.%N)
PHASE=validating
GPU_STARTED=false
finish() {
  STATUS=$?
  [[ $DRY_RUN == 0 ]] || return 0
  {
    printf 'exit_status=%s\nphase=%s\ngpu_started=%s\ngpu_count=1\npaid_jobs=0\ncumulative_cap_seconds=4500\nprior_gpu_seconds=%s\n' "$STATUS" "$PHASE" "$GPU_STARTED" "${PRIOR_GPU_SECONDS:-0}"
    awk -v a="$START" -v b="$(date +%s.%N)" -v started="$GPU_STARTED" 'BEGIN {printf "launcher_wall_seconds=%.9f\nlauncher_card_hours=%.12f\n",b-a,(started=="true"?(b-a)/3600:0)}'
  } > "$OUTPUT.actual.exit"
}
[[ ! -e $OUTPUT && ! -e $OUTPUT.actual.exit && ! -e $OUTPUT.source.py ]] || exit 2
if [[ $DRY_RUN == 0 ]]; then mkdir -p "$DIAG"; fi
trap finish EXIT
if [[ $DRY_RUN == 0 ]]; then
  exec 9>"$DIAG/.stage.lock"
  flock -n 9 || exit 2
fi
shopt -s nullglob
for LOG in "$DIAG"/*.json.log; do
  [[ -e ${LOG%.log}.actual.exit ]] || { echo 'Unaccounted prior GPU stage; stop for review.' >&2; exit 2; }
done
EXITS=("$DIAG"/*.actual.exit)
PRIOR_GPU_SECONDS=0
if (( ${#EXITS[@]} )); then
  PRIOR_GPU_SECONDS=$(awk -F= 'FNR==1 {started=0} $1=="gpu_started" {started=($2=="true")} $1=="launcher_wall_seconds" && started {sum+=$2} END {printf "%.9f",sum}' "${EXITS[@]}")
fi
if [[ $STAGE == model ]]; then
  REPORTS=("$DIAG"/model*.json)
  if (( ${#REPORTS[@]} )) && grep -Eq '"model_compile_attempted"[[:space:]]*:[[:space:]]*1' "${REPORTS[@]}"; then
    echo 'The one additional expensive compile was already attempted; stop.' >&2
    exit 2
  fi
fi
[[ $(git --git-dir="$CODE_ROOT/.git" --work-tree="$CODE_ROOT" rev-parse HEAD) == c599236bcf57b745e9bafed678298010f4c3b58f ]] || exit 2
[[ -x $TOOLCHAIN/bin/nvlink && -x $OVERLAY/nvlink-bin/nvlink ]] || exit 2
[[ -f $PRIOR/model-exact-constant.json && -f $PRIOR/model-exact-constant.stablehlo.txt ]] || exit 2
if [[ $DRY_RUN == 0 ]]; then
  [[ $(hostname -s) == ibc13b02n04 && -z ${SLURM_JOB_ID:-} ]] || exit 2
  LINE=$(nvidia-smi -i 1 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  read -r MEMORY UTIL <<<"${LINE//,/ }"
  [[ $MEMORY =~ ^[0-9]+$ && $UTIL =~ ^[0-9]+$ && $MEMORY -le 2000 && $UTIL -le 5 ]] || exit 2
  [[ -z $(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader,nounits) ]] || exit 2
fi
CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
ENV=(env -u LD_LIBRARY_PATH -u LD_PRELOAD
  SINGULARITYENV_PREPEND_PATH="$OVERLAY/nvlink-bin:$TOOLCHAIN/bin"
  SINGULARITYENV_ORCHIDEE_NVLINK_LOG="$OUTPUT.nvlink.log" SINGULARITYENV_ORCHIDEE_NVLINK_STAGE="$STAGE"
  SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs SINGULARITYENV_CUDA_VISIBLE_DEVICES=1
  SINGULARITYENV_JAX_PLATFORMS=cuda,cpu SINGULARITYENV_JAX_ENABLE_X64=true
  SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false SINGULARITYENV_ORCHIDEE_REPO_ROOT="$CODE_ROOT"
  SINGULARITYENV_PYTHONPATH="$CODE_ROOT:$OVERLAY:$ROOT/runtime/transfer/shared-forward-reverse-20260930:$ROOT/runtime/transfer/shared-command-buffer-loops-20261001"
  SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1 SINGULARITYENV_PYTHONFAULTHANDLER=1
  SINGULARITYENV_OPENBLAS_NUM_THREADS=1 SINGULARITYENV_OMP_NUM_THREADS=4
  SINGULARITYENV_XLA_FLAGS='--xla_gpu_enable_llvm_module_compilation_parallelism=true --xla_gpu_force_compilation_parallelism=4')
CONTAINER=(singularity exec --nv --bind "/WORK:/WORK,$ROOT/runtime/data:$ROOT/data:ro,$TOOLCHAIN:$TOOLCHAIN:ro,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
  --pwd "$CODE_ROOT" /apps/soft/sif/foundationpose "$ROOT/.venvs/orcjax_gpu/bin/python")
REMAINING=$(awk -v used="$PRIOR_GPU_SECONDS" -v start="$START" -v now="$(date +%s.%N)" 'BEGIN {printf "%d",4500-used-(now-start)-20}')
(( REMAINING > 20 )) || exit 2
if [[ $STAGE == preflight ]]; then
  LIMIT=$(( REMAINING < 180 ? REMAINING : 180 ))
  ENTRY="$CODE_ROOT/scripts/hpc/check_daily_training_runtime.py"
  COMMAND=(timeout --signal=TERM --kill-after=10s "${LIMIT}s" "${ENV[@]}" "${CONTAINER[@]}" "$ENTRY"
    --backend gpu --output "$OUTPUT")
else
  LIMIT=$REMAINING
  ENTRY="$OVERLAY/probe_shared_state_reference_fp64_parallel.py"
  COMMAND=(timeout --signal=TERM --kill-after=10s "${LIMIT}s" "${ENV[@]}" "${CONTAINER[@]}" "$ENTRY"
    --runtime-preflight "$DIAG/preflight.json" --output "$OUTPUT" --process-seconds "$(( LIMIT - 20 ))"
    --required-prior-report "$PRIOR/model-exact-constant.json" --required-graph-file "$PRIOR/model-exact-constant.stablehlo.txt")
fi
if [[ $DRY_RUN == 1 ]]; then printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0; fi
cp "$ENTRY" "$OUTPUT.source.py"
cp "$OVERLAY/run_shared_state_reference_fp64_parallel.sh" "$OUTPUT.launcher.sh"
cp "$OVERLAY/nvlink-bin/nvlink" "$OUTPUT.nvlink-wrapper.sh"
if [[ $STAGE == model ]]; then
  cp "$OVERLAY/shared_neural_fp64_diagnostic.py" "$OUTPUT.policy.py"
  cp "$OVERLAY/shared_fp64_parallel_admission.py" "$OUTPUT.graph-admission.py"
fi
PHASE=executing
GPU_STARTED=true
{
  date --iso-8601=seconds
  hostname
  uptime
  grep Cpus_allowed_list /proc/self/status
  nvidia-smi --query-gpu=index,name,driver_version,memory.used,utilization.gpu --format=csv,noheader
  nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
  "${COMMAND[@]}"
} 2>&1 | tee "$OUTPUT.log"
