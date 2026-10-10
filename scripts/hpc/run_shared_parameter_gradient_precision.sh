#!/bin/bash
# PROPOSAL ONLY: explicit science approval is required before any GPU execution.
# One science candidate; at most one exact compatibility repair within budget.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT="$ROOT/runtime/checkouts/shared-split-width-v1"
OVERLAY="$ROOT/runtime/transfer/shared-parameter-gradient-precision-20261002"
DIAG="$ROOT/runtime/outputs/diagnostics/shared-parameter-gradient-precision-20261002"
SAVED="$ROOT/runtime/outputs/diagnostics"
BACKBONE="$ROOT/runtime/transfer/shared-combined-recurrent-20260926"
REF=shared-chunk-vmap-daily-dual-v4-20261001
DRY_RUN=0
GPU=1
ROUTE=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --dry-run) DRY_RUN=1; shift ;;
    --gpu) [[ ${2:-} =~ ^[01]$ ]] || exit 2; GPU=$2; shift 2 ;;
    route) ROUTE=1; shift; [[ $# == 0 ]] || exit 2; break ;;
    *) exit 2 ;;
  esac
done
[[ $ROUTE == 1 ]] || exit 2
START=$(date +%s.%N)
PHASE=preparing
GPU_STARTED=false
CAP=3500
OUTCOME=not_started
REPAIR_ATTEMPTED=0
finish() {
  STATUS=$?
  [[ $DRY_RUN == 0 ]] || return 0
  {
    printf 'exit_status=%s\nphase=%s\noutcome=%s\ngpu_started=%s\nphysical_gpu=%s\npaid_jobs=0\ncumulative_cap_seconds=3500\npublic_candidate_compile_attempts=%s\ncompatibility_repair_attempts=%s\n' "$STATUS" "$PHASE" "$OUTCOME" "$GPU_STARTED" "$GPU" "${PUBLIC_ATTEMPTED:-0}" "$REPAIR_ATTEMPTED"
    awk -v start="$START" -v finish="$(date +%s.%N)" -v started="$GPU_STARTED" 'BEGIN {printf "launcher_wall_seconds=%.9f\nlauncher_card_hours=%.12f\n",finish-start,(started=="true"?(finish-start)/3600:0)}'
  } > "$DIAG/route.actual.exit"
}
[[ $(git --git-dir="$CODE_ROOT/.git" --work-tree="$CODE_ROOT" rev-parse HEAD) == c599236bcf57b745e9bafed678298010f4c3b58f ]] || exit 2
[[ ! -e $DIAG/route.actual.exit && ! -e $DIAG/public-compile-attempted ]] || exit 2
if [[ $DRY_RUN == 0 ]]; then
  mkdir -p "$DIAG"
  [[ ! -e $DIAG/preflight.json && ! -e $DIAG/interface/interface.json && ! -e $DIAG/daily.json ]] || exit 2
  trap finish EXIT
  exec 9>"$DIAG/.route.lock"
  flock -n 9 || exit 2
  [[ $(hostname -s) == ibc13b02n04 && -z ${SLURM_JOB_ID:-} ]] || exit 2
  LINE=$(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  read -r MEMORY UTIL <<<"${LINE//,/ }"
  [[ $MEMORY == 0 && $UTIL == 0 ]] || exit 2
  [[ -z $(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits) ]] || exit 2
  CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
  NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
else
  CUDA_DRIVER=/usr/lib64/libcuda.so.525.85.12
  NVML_DRIVER=/usr/lib64/libnvidia-ml.so.525.85.12
fi
ENV=(env -u LD_LIBRARY_PATH -u LD_PRELOAD
  SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU"
  SINGULARITYENV_JAX_PLATFORMS=cuda,cpu SINGULARITYENV_JAX_ENABLE_X64=true
  SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false SINGULARITYENV_ORCHIDEE_REPO_ROOT="$CODE_ROOT"
  SINGULARITYENV_PYTHONPATH="$CODE_ROOT:$OVERLAY" SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
  SINGULARITYENV_OPENBLAS_NUM_THREADS=1 SINGULARITYENV_OMP_NUM_THREADS=4)
ORIGINAL_XLA='--xla_gpu_enable_llvm_module_compilation_parallelism=false'
REPAIRED_XLA="$ORIGINAL_XLA --xla_disable_hlo_passes=operand_upcaster"
SELECTED_XLA=$ORIGINAL_XLA
INTERFACE_REPORT="$DIAG/interface/interface.json"
CONTAINER=(singularity exec --nv --bind "/WORK:/WORK,$ROOT/runtime/data:$ROOT/data:ro,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
  --pwd "$CODE_ROOT" /apps/soft/sif/foundationpose "$ROOT/.venvs/orcjax_gpu/bin/python")
PREFLIGHT=("${ENV[@]}" "SINGULARITYENV_XLA_FLAGS=$ORIGINAL_XLA" "${CONTAINER[@]}" "$CODE_ROOT/scripts/hpc/check_daily_training_runtime.py" --backend gpu --output "$DIAG/preflight.json")
INTERFACE=("${ENV[@]}" "SINGULARITYENV_XLA_FLAGS=$ORIGINAL_XLA" "${CONTAINER[@]}" "$OVERLAY/probe_shared_parameter_gradient_precision.py" --mode gpu-interface --output-dir "$DIAG/interface")
INTERFACE_REPAIRED=("${ENV[@]}" "SINGULARITYENV_XLA_FLAGS=$REPAIRED_XLA" "${CONTAINER[@]}" "$OVERLAY/probe_shared_parameter_gradient_precision.py" --mode gpu-interface --output-dir "$DIAG/interface-operand-upcaster")
DAILY_ARGS=("$OVERLAY/probe_shared_parameter_gradient_precision_daily.py"
  --stage candidate --lower-seconds 240 --backend-seconds 1500 --process-seconds 2380
  --patch-owner "$ROOT/runtime/transfer/shared-combined-real-daily-20260925/probe_shared_expert_real_daily.py"
  --gate-owner "$ROOT/runtime/transfer/shared-union-micro4-20260929/probe_shared_union_micro4.py"
  --source-profile "$SAVED/shared-split-profile-c599236.json" --input-bundle "$SAVED/shared-split-dual-inputs-c599236.pkl"
  --prepared-context "$SAVED/shared-expert-real-daily-context-c599236.pkl" --baseline-admission "$SAVED/shared-combined-real-daily-20260925.json"
  --runtime-preflight "$DIAG/preflight.json" --candidate-module-file "$BACKBONE/shared_decoder_expert_execution.py"
  --backbone-module-file "$BACKBONE/shared_backbone_execution.py" --execution-module-file "$OVERLAY/shared_parameter_gradient_precision.py"
  --parameter-report "$SAVED/shared-chunk-vmap-daily-expert5-micro4-20261001-v2.json"
  --parameter-result "$SAVED/shared-chunk-vmap-daily-expert5-micro4-20261001-v2.reference.pkl"
  --dual-reference-report "$SAVED/$REF-reference.json" --dual-reference-result "$SAVED/$REF.reference.pkl"
  --dual-reference-executable "$SAVED/$REF.reference.executable.pkl")
if [[ $DRY_RUN == 1 ]]; then
  printf 'preflight cap300: '; printf '%q ' "${PREFLIGHT[@]}"; printf '\n'
  printf 'interface cap600: '; printf '%q ' "${INTERFACE[@]}"; printf '\n'
  printf 'known-rejection-only interface retry within SAME cap600: '; printf '%q ' "${INTERFACE_REPAIRED[@]}"; printf '\n'
  printf 'conditional daily cap2400: '; printf '%q ' "${ENV[@]}" "SINGULARITYENV_XLA_FLAGS=$ORIGINAL_XLA" "${CONTAINER[@]}" "${DAILY_ARGS[@]}" --interface-report "$INTERFACE_REPORT" --output "$DIAG/daily.json" --result-output "$DIAG/daily.candidate.pkl" --executable-output "$DIAG/daily.candidate.executable.pkl"; printf '\n'
  printf 'conditional repaired daily within SAME cap2400: '; printf '%q ' "${ENV[@]}" "SINGULARITYENV_XLA_FLAGS=$REPAIRED_XLA" "${CONTAINER[@]}" "${DAILY_ARGS[@]}" --interface-report "$DIAG/interface-operand-upcaster/interface.json" --output "$DIAG/daily-operand-upcaster.json" --result-output "$DIAG/daily-operand-upcaster.candidate.pkl" --executable-output "$DIAG/daily-operand-upcaster.candidate.executable.pkl"; printf '\n'
  exit 0
fi
date --iso-8601=seconds
hostname
uptime
grep Cpus_allowed_list /proc/self/status
nvidia-smi --query-gpu=index,name,driver_version,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
run_stage() {
  local requested=$1
  shift
  local remaining
  remaining=$(awk -v start="$START" -v now="$(date +%s.%N)" 'BEGIN {printf "%d",3500-(now-start)-20}')
  (( remaining > 20 )) || exit 124
  local limit=$(( remaining < requested ? remaining : requested ))
  timeout --signal=TERM --kill-after=10s "${limit}s" "$@" 2>&1 | tee "$DIAG/$PHASE.log"
}
GPU_STARTED=true
PHASE=preflight
OUTCOME=runtime_preflight_incomplete
run_stage 300 "${PREFLIGHT[@]}"
PHASE=interface
OUTCOME=interface_incomplete
INTERFACE_START=$(date +%s.%N)
if run_stage 600 "${INTERFACE[@]}"; then
  OUTCOME=interface_passed
else
  # Exactly one known compiler rejection permits the named compatibility
  # repair. Numerical failures and unknown rejection/codegen failures stop.
  if ! grep -Fq 'UNIMPLEMENTED: Unsupported algorithm on the current device(s): ALG_DOT_F16_F16_F32' "$DIAG/interface.log"; then
    OUTCOME=interface_failed_or_compatibility_incomplete
    exit 1
  fi
  RETEST_SECONDS=$(awk -v start="$INTERFACE_START" -v now="$(date +%s.%N)" 'BEGIN {printf "%d",600-(now-start)-10}')
  (( RETEST_SECONDS > 20 )) || { OUTCOME=compatibility_budget_incomplete; exit 124; }
  REPAIR_ATTEMPTED=1
  SELECTED_XLA=$REPAIRED_XLA
  INTERFACE_REPORT="$DIAG/interface-operand-upcaster/interface.json"
  PHASE=interface-operand-upcaster
  OUTCOME=repaired_compatibility_incomplete
  run_stage "$RETEST_SECONDS" "${INTERFACE_REPAIRED[@]}"
  OUTCOME=repaired_interface_passed
fi
# Success means original FP32 primal/state/bias strict gates, parameter
# engineering gates, explicit native FP16 codegen and >=1.05x major-shape gain.
PHASE=daily
OUTCOME=public_F4S_incomplete
PUBLIC_ATTEMPTED=1
touch "$DIAG/public-compile-attempted"
PUBLIC_START=$(date +%s.%N)
if run_stage 2400 "${ENV[@]}" "SINGULARITYENV_XLA_FLAGS=$SELECTED_XLA" "${CONTAINER[@]}" "${DAILY_ARGS[@]}" --interface-report "$INTERFACE_REPORT" --output "$DIAG/daily.json" --result-output "$DIAG/daily.candidate.pkl" --executable-output "$DIAG/daily.candidate.executable.pkl"; then
  OUTCOME=public_F4S_screen_passed_no_adoption
else
  if [[ $REPAIR_ATTEMPTED != 0 ]] || ! grep -Fq 'UNIMPLEMENTED: Unsupported algorithm on the current device(s): ALG_DOT_F16_F16_F32' "$DIAG/daily.log"; then
    OUTCOME=public_F4S_failed_or_compatibility_incomplete
    exit 1
  fi
  PUBLIC_REMAINING=$(awk -v start="$PUBLIC_START" -v now="$(date +%s.%N)" 'BEGIN {printf "%d",2400-(now-start)-20}')
  (( PUBLIC_REMAINING > 40 )) || { OUTCOME=compatibility_budget_incomplete; exit 124; }
  REPAIR_ATTEMPTED=1
  SELECTED_XLA=$REPAIRED_XLA
  INTERFACE_REPORT="$DIAG/interface-operand-upcaster/interface.json"
  PHASE=public-compatibility-interface
  OUTCOME=repaired_compatibility_incomplete
  RETEST_SECONDS=$(( PUBLIC_REMAINING < 600 ? PUBLIC_REMAINING : 600 ))
  run_stage "$RETEST_SECONDS" "${INTERFACE_REPAIRED[@]}"
  PUBLIC_REMAINING=$(awk -v start="$PUBLIC_START" -v now="$(date +%s.%N)" 'BEGIN {printf "%d",2400-(now-start)-10}')
  (( PUBLIC_REMAINING > 20 )) || { OUTCOME=compatibility_budget_incomplete; exit 124; }
  PHASE=daily-operand-upcaster
  OUTCOME=repaired_public_F4S_incomplete
  PUBLIC_ATTEMPTED=2
  run_stage "$PUBLIC_REMAINING" "${ENV[@]}" "SINGULARITYENV_XLA_FLAGS=$SELECTED_XLA" "${CONTAINER[@]}" "${DAILY_ARGS[@]}" --interface-report "$INTERFACE_REPORT" --output "$DIAG/daily-operand-upcaster.json" --result-output "$DIAG/daily-operand-upcaster.candidate.pkl" --executable-output "$DIAG/daily-operand-upcaster.candidate.executable.pkl"
  OUTCOME=repaired_public_F4S_screen_passed_no_adoption
fi
PHASE=completed
