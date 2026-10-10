#!/bin/bash
# Free GPU0; original four samples, isolated selected neural precision policy.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT="$ROOT/runtime/checkouts/shared-split-width-v1"
OVERLAY="$ROOT/runtime/transfer/shared-daily-field-precision-lowering-20261001"
PUBLIC="$ROOT/runtime/transfer/shared-chunk-vmap-daily-dual-20261001"
DIAG="$ROOT/runtime/outputs/diagnostics"
STEM=shared-daily-field-precision-lowering-20261001
REFERENCE=shared-chunk-vmap-daily-dual-v4-20261001
OUTPUT="$DIAG/$STEM.json"
DRY_RUN=0
if [[ ${1:-} == --dry-run ]]; then DRY_RUN=1; shift; fi
[[ ${1:-} =~ ^(preflight|candidate)$ && $# == 1 ]] || exit 2
STAGE=$1
if [[ $STAGE == preflight ]]; then OUTPUT="$DIAG/$STEM-preflight.json"; fi
START=$(date +%s.%N)
GPU_STARTED=false
finish_record() {
  STATUS=$?
  [[ $DRY_RUN == 0 ]] || return 0
  FINISH=$(date +%s.%N)
  {
    printf 'variant=original_field_four_sample_neural_precision\ngpu_started=%s\npaid_jobs=0\nexit_status=%s\n' "$GPU_STARTED" "$STATUS"
    awk -v start="$START" -v finish="$FINISH" -v started="$GPU_STARTED" 'BEGIN {printf "launcher_wall_seconds=%.9f\ngpu_count=%d\nlauncher_card_hours=%.12f\n", finish-start,(started=="true"?1:0),(started=="true"?(finish-start)/3600:0)}'
  } > "$OUTPUT.launcher.txt"
}
[[ $(git --git-dir="$CODE_ROOT/.git" --work-tree="$CODE_ROOT" rev-parse HEAD) == c599236bcf57b745e9bafed678298010f4c3b58f ]] || exit 2
[[ -f $DIAG/$REFERENCE-reference.json && -f $DIAG/$REFERENCE.reference.executable.pkl && -f $DIAG/$REFERENCE.reference.pkl ]] || exit 2
[[ ! -e $OUTPUT && ! -e $OUTPUT.launcher.txt ]] || exit 2
trap finish_record EXIT
if [[ $DRY_RUN == 0 ]]; then
  [[ $(hostname -s) == ibc13b02n04 && -z ${SLURM_JOB_ID:-} ]] || exit 2
  LINE=$(nvidia-smi -i 0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  read -r MEMORY UTIL <<<"${LINE//,/ }"
  [[ $MEMORY =~ ^[0-9]+$ && $UTIL =~ ^[0-9]+$ && $MEMORY -le 2000 && $UTIL -le 5 ]] || exit 2
  [[ -z $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits) ]] || exit 2
fi
CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
BACKBONE="$ROOT/runtime/transfer/shared-combined-recurrent-20260926"
COMMAND=(timeout --signal=TERM --kill-after=10s 4210s env -u LD_LIBRARY_PATH -u LD_PRELOAD
  SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs SINGULARITYENV_CUDA_VISIBLE_DEVICES=0
  SINGULARITYENV_JAX_PLATFORMS=cuda,cpu SINGULARITYENV_JAX_ENABLE_X64=true
  SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false SINGULARITYENV_ORCHIDEE_REPO_ROOT="$CODE_ROOT"
  SINGULARITYENV_PYTHONPATH="$CODE_ROOT:$OVERLAY" SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
  SINGULARITYENV_PYTHONFAULTHANDLER=1 SINGULARITYENV_OPENBLAS_NUM_THREADS=1 SINGULARITYENV_OMP_NUM_THREADS=4
  SINGULARITYENV_XLA_FLAGS=--xla_gpu_enable_llvm_module_compilation_parallelism=false
  singularity exec --nv --bind "/WORK:/WORK,$ROOT/runtime/data:$ROOT/data:ro,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
  --pwd "$CODE_ROOT" /apps/soft/sif/foundationpose "$ROOT/.venvs/orcjax_gpu/bin/python"
  "$OVERLAY/probe_shared_daily_field_precision.py"
  --stage candidate --lowering-only --output "$OUTPUT" --result-output "$DIAG/$STEM.candidate.pkl" --executable-output "$DIAG/$STEM.candidate.executable.pkl"
  --lower-seconds 240 --backend-seconds 1500 --process-seconds 4200
  --patch-owner "$ROOT/runtime/transfer/shared-combined-real-daily-20260925/probe_shared_expert_real_daily.py"
  --gate-owner "$ROOT/runtime/transfer/shared-union-micro4-20260929/probe_shared_union_micro4.py"
  --source-profile "$DIAG/shared-split-profile-c599236.json" --input-bundle "$DIAG/shared-split-dual-inputs-c599236.pkl"
  --prepared-context "$DIAG/shared-expert-real-daily-context-c599236.pkl" --baseline-admission "$DIAG/shared-combined-real-daily-20260925.json"
  --runtime-preflight "$DIAG/shared-daily-field-precision-20261001-preflight.json"
  --candidate-module-file "$BACKBONE/shared_decoder_expert_execution.py" --backbone-module-file "$BACKBONE/shared_backbone_execution.py"
  --execution-module-file "$OVERLAY/shared_daily_precision_matrix_policy.py"
  --parameter-report "$DIAG/shared-chunk-vmap-daily-expert5-micro4-20261001-v2.json"
  --parameter-result "$DIAG/shared-chunk-vmap-daily-expert5-micro4-20261001-v2.reference.pkl"
  --dual-reference-report "$DIAG/$REFERENCE-reference.json" --dual-reference-result "$DIAG/$REFERENCE.reference.pkl"
  --dual-reference-executable "$DIAG/$REFERENCE.reference.executable.pkl")
if [[ $STAGE == preflight ]]; then
  COMMAND=(timeout --signal=TERM --kill-after=10s 300s env -u LD_LIBRARY_PATH -u LD_PRELOAD
    SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs SINGULARITYENV_CUDA_VISIBLE_DEVICES=0
    SINGULARITYENV_JAX_PLATFORMS=cuda,cpu SINGULARITYENV_JAX_ENABLE_X64=true
    SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false SINGULARITYENV_ORCHIDEE_REPO_ROOT="$CODE_ROOT"
    SINGULARITYENV_PYTHONPATH="$CODE_ROOT:$OVERLAY" SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
    SINGULARITYENV_OPENBLAS_NUM_THREADS=1 SINGULARITYENV_OMP_NUM_THREADS=4
    SINGULARITYENV_XLA_FLAGS=--xla_gpu_enable_llvm_module_compilation_parallelism=false
    singularity exec --nv --bind "/WORK:/WORK,$ROOT/runtime/data:$ROOT/data:ro,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
    --pwd "$CODE_ROOT" /apps/soft/sif/foundationpose "$ROOT/.venvs/orcjax_gpu/bin/python"
    "$CODE_ROOT/scripts/hpc/check_daily_training_runtime.py" --backend gpu --output "$OUTPUT")
fi
if [[ $DRY_RUN == 1 ]]; then printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0; fi
date --iso-8601=seconds
hostname
uptime
grep Cpus_allowed_list /proc/self/status
nvidia-smi --query-gpu=index,name,driver_version,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
GPU_STARTED=true
"${COMMAND[@]}"
