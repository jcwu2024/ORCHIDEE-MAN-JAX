#!/bin/bash
# Explicit engineering diagnostic; unchanged strict updated-parameter/AdamW gate.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT="$ROOT/runtime/checkouts/shared-split-width-v1"
OVERLAY="$ROOT/runtime/transfer/shared-daily-ffn-precision-full-update-20261002"
DIAG="$ROOT/runtime/outputs/diagnostics"
UPDATES="$ROOT/runtime/transfer/shared-update-changing-slice-20260930"
CANDIDATE=shared-daily-ffn-precision-scale4096-20261001
REFERENCE=shared-chunk-vmap-daily-dual-v4-20261001
REF7=shared-recurrent-checkpoint-policy-20260930-v2-T7
REF30=shared-original-t30-dual-reference-20261001
OUTPUT="$DIAG/shared-daily-ffn-precision-full-update-20261002.json"
DRY_RUN=0
if [[ ${1:-} == --dry-run ]]; then DRY_RUN=1; shift; fi
[[ $# == 0 ]] || exit 2
START=$(date +%s.%N)
GPU_STARTED=false
finish_record() {
  STATUS=$?
  [[ $DRY_RUN == 0 ]] || return 0
  FINISH=$(date +%s.%N)
  {
    printf 'variant=FFN_only_scale4096_compatibility_original_recurrent_full_update_diagnostic\ngpu_started=%s\npaid_jobs=0\nexit_status=%s\n' "$GPU_STARTED" "$STATUS"
    awk -v start="$START" -v finish="$FINISH" -v started="$GPU_STARTED" 'BEGIN {printf "launcher_wall_seconds=%.9f\ngpu_count=%d\nlauncher_card_hours=%.12f\n", finish-start,(started=="true"?1:0),(started=="true"?(finish-start)/3600:0)}'
  } > "$OUTPUT.launcher.txt"
}
[[ $(git --git-dir="$CODE_ROOT/.git" --work-tree="$CODE_ROOT" rev-parse HEAD) == c599236bcf57b745e9bafed678298010f4c3b58f ]] || exit 2
[[ ! -e $OUTPUT && ! -e $OUTPUT.launcher.txt ]] || exit 2
trap finish_record EXIT
if [[ $DRY_RUN == 0 ]]; then
  [[ -f $DIAG/$CANDIDATE.json && -f $DIAG/$CANDIDATE.candidate.executable.pkl ]] || exit 2
  [[ $(hostname -s) == ibc13b02n04 && -z ${SLURM_JOB_ID:-} ]] || exit 2
  LINE=$(nvidia-smi -i 0 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  read -r MEMORY UTIL <<<"${LINE//,/ }"
  [[ $MEMORY =~ ^[0-9]+$ && $UTIL =~ ^[0-9]+$ && $MEMORY -le 2000 && $UTIL -le 5 ]] || exit 2
  [[ -z $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader,nounits) ]] || exit 2
fi
CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
COMMAND=(timeout --signal=TERM --kill-after=10s 1090s env -u LD_LIBRARY_PATH -u LD_PRELOAD
  SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs SINGULARITYENV_CUDA_VISIBLE_DEVICES=0
  SINGULARITYENV_JAX_PLATFORMS=cuda,cpu SINGULARITYENV_JAX_ENABLE_X64=true
  SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false SINGULARITYENV_ORCHIDEE_REPO_ROOT="$CODE_ROOT"
  SINGULARITYENV_PYTHONPATH="$CODE_ROOT:$OVERLAY" SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
  SINGULARITYENV_OPENBLAS_NUM_THREADS=1 SINGULARITYENV_OMP_NUM_THREADS=4
  "SINGULARITYENV_XLA_FLAGS=--xla_gpu_enable_llvm_module_compilation_parallelism=false --xla_disable_hlo_passes=operand_upcaster"
  singularity exec --nv --bind "/WORK:/WORK,$ROOT/runtime/data:$ROOT/data:ro,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
  --pwd "$CODE_ROOT" /apps/soft/sif/foundationpose "$ROOT/.venvs/orcjax_gpu/bin/python"
  "$OVERLAY/probe_shared_normalized_keys_full_update.py"
  --execution-variant daily-only-ffn-precision-diagnostic
  --tree-owner-file "$UPDATES/probe_shared_update_tree_slice.py"
  --slice-helper-file "$UPDATES/shared_daily_microbatch.py" --learning-runner-file "$UPDATES/run_shared_learning.py"
  --source-profile "$DIAG/shared-split-profile-c599236.json" --input-bundle "$DIAG/shared-split-dual-inputs-c599236.pkl"
  --daily-admission-report "$DIAG/shared-combined-real-daily-20260925.json"
  --seven-admission-report "$DIAG/shared-combined-recurrent-T7-20260926.json"
  --thirty-admission-report "$DIAG/shared-combined-recurrent-T30-20260926.json"
  --support-owner-file "$ROOT/runtime/transfer/shared-combined-full-update-20260926/probe_shared_serialized_dual.py"
  --candidate-daily-report "$DIAG/$CANDIDATE.json"
  --daily-dual-reference-report "$DIAG/$REFERENCE-reference.json"
  --daily-dual-reference-executable "$DIAG/$REFERENCE.reference.executable.pkl"
  --daily-dual-candidate-executable "$DIAG/$CANDIDATE.candidate.executable.pkl"
  --execution-module-file "$ROOT/runtime/transfer/shared-daily-ffn-precision-scale4096-20261001/shared_daily_ffn_precision_policy.py"
  --prepared-context "$DIAG/shared-expert-real-daily-context-c599236.pkl"
  --candidate-module-file "$ROOT/runtime/transfer/shared-combined-recurrent-20260926/shared_decoder_expert_execution.py"
  --t7-reference-report "$DIAG/$REF7-reference.json" --t7-reference-executable "$DIAG/$REF7.reference.executable.pkl"
  --t30-reference-report "$DIAG/$REF30.json" --t30-reference-executable "$DIAG/$REF30.reference.executable.pkl"
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
