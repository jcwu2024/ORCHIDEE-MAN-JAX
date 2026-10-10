#!/bin/bash
# Free gln01 only. All occupied stages share one persistent card-second ledger.
set -euo pipefail
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT="$ROOT/runtime/checkouts/shared-split-width-v1"
OVERLAY="$ROOT/runtime/transfer/shared-latent-reader-20261004"
DIAG="$ROOT/runtime/outputs/diagnostics"
LEDGER="$DIAG/shared-latent-reader-20261004.gpu-budget.json"
TOOLCHAIN="$ROOT/runtime/toolchains/cuda-nvlink-12.6.85"
STAGE=${1:-}
DRY_RUN=${2:-}
GPU_INDEX=${GPU_INDEX:-0}
SESSION_SECONDS=${SESSION_SECONDS:-}
BACKEND_SECONDS=${LATENT_READER_BACKEND_SECONDS:-3400}
CARDS=1
COMPILER_DIAGNOSTICS=${LATENT_READER_COMPILER_DIAGNOSTICS:-0}
case "$COMPILER_DIAGNOSTICS" in 0|1) ;; *) exit 2 ;; esac
COMPILER_HLO_SNAPSHOTS=${LATENT_READER_COMPILER_HLO_SNAPSHOTS:-0}
case "$COMPILER_HLO_SNAPSHOTS" in 0|1) ;; *) exit 2 ;; esac
case "$STAGE" in
  preflight) SESSION_SECONDS=180 ;;
  nvlink-smoke) SESSION_SECONDS=180 ;;
  original-changing) SESSION_SECONDS=1800; CARDS=2 ;;
  T7|T30|daily) SESSION_SECONDS=${SESSION_SECONDS:-6000} ;;
  masks) SESSION_SECONDS=${SESSION_SECONDS:-900}; BACKEND_SECONDS=${LATENT_READER_BACKEND_SECONDS:-420} ;;
  full-dual) SESSION_SECONDS=${SESSION_SECONDS:-1800}; CARDS=2 ;;
  *) exit 2 ;;
esac
case "$DRY_RUN" in ''|--dry-run) ;; *) exit 2 ;; esac
case "$GPU_INDEX" in 0|1) ;; *) exit 2 ;; esac
[[ "$SESSION_SECONDS" =~ ^[0-9]+$ && $SESSION_SECONDS -le 7200 && $SESSION_SECONDS -ge 20 ]] || exit 2
case "$STAGE" in T7|T30|daily|masks)
  [[ "$BACKEND_SECONDS" =~ ^[0-9]+$ && $BACKEND_SECONDS -ge 10 && $BACKEND_SECONDS -le "$SESSION_SECONDS" ]] || exit 2 ;;
esac
[[ $(cd "$CODE_ROOT" && git rev-parse HEAD) == c599236bcf57b745e9bafed678298010f4c3b58f ]] || exit 2
RUN_SUFFIX=${LATENT_READER_RUN_SUFFIX:-}
[[ -z $RUN_SUFFIX || $RUN_SUFFIX =~ ^[A-Za-z0-9_-]{1,64}$ ]] || exit 2
OUTPUT="$DIAG/shared-latent-reader-20261004-$STAGE${RUN_SUFFIX:+-$RUN_SUFFIX}.json"
if [[ $DRY_RUN != --dry-run ]]; then
  [[ $(hostname -s) == ibc13b02n04 && -z ${SLURM_JOB_ID:-} && ! -e "$OUTPUT" ]] || exit 2
  LINE=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  read -r MEMORY UTIL <<<"${LINE//,/ }"
  [[ $MEMORY =~ ^[0-9]+$ && $UTIL =~ ^[0-9]+$ && $MEMORY -le 2000 && $UTIL -le 5 ]] || exit 2
  [[ -z $(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader,nounits) ]] || exit 2
  if [[ $CARDS == 2 ]]; then
    [[ -z $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits) ]] || exit 2
  fi
fi
CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
[[ -x $TOOLCHAIN/bin/nvlink && -x $OVERLAY/nvlink-bin/nvlink ]] || exit 2
COMMAND=(env -u LD_LIBRARY_PATH
  SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU_INDEX"
  SINGULARITYENV_JAX_PLATFORMS=cuda,cpu SINGULARITYENV_JAX_ENABLE_X64=true
  SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
  SINGULARITYENV_JAX_COMPILATION_CACHE_DIR="$ROOT/runtime/cache/jax/orcjax_gpu/shared-latent-reader-20261004"
  SINGULARITYENV_JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0 SINGULARITYENV_JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
  SINGULARITYENV_PREPEND_PATH="$OVERLAY/nvlink-bin:$TOOLCHAIN/bin"
  SINGULARITYENV_ORCHIDEE_NVLINK_LOG="$OUTPUT.nvlink.log" SINGULARITYENV_ORCHIDEE_NVLINK_STAGE="$STAGE"
  SINGULARITYENV_ORCHIDEE_REPO_ROOT="$CODE_ROOT" SINGULARITYENV_PYTHONPATH="$CODE_ROOT:$OVERLAY"
  SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1 SINGULARITYENV_OPENBLAS_NUM_THREADS=1 SINGULARITYENV_OMP_NUM_THREADS=4
  singularity exec --nv --bind "/WORK:/WORK,$ROOT/runtime/data:$ROOT/data:ro,$TOOLCHAIN:$TOOLCHAIN:ro,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
  --pwd "$CODE_ROOT" /apps/soft/sif/foundationpose "$ROOT/.venvs/orcjax_gpu/bin/python")
if [[ $STAGE == original-changing ]]; then
  COMMAND=(env ENTRY_FILE="$OVERLAY/probe_shared_serialized_dual.py" OUTPUT="$OUTPUT"
    COMPARISON_ONLY=1 SCHEDULE_CHANGING_UPDATES=2 GPU_DEVICE=0 WORKER_GPU_DEVICE=1
    INPUT_BUNDLE="$DIAG/shared-split-dual-inputs-c599236.pkl"
    bash "$OVERLAY/run_shared_serialized_dual.sh" --daily-counts 32
    --daily-admission-report "$DIAG/shared-combined-real-daily-20260925.json"
    --seven-admission-report "$DIAG/shared-combined-recurrent-T7-20260926.json"
    --thirty-admission-report "$DIAG/shared-combined-recurrent-T30-20260926.json")
elif [[ $STAGE == preflight ]]; then
  COMMAND+=("$CODE_ROOT/scripts/hpc/check_daily_training_runtime.py" --backend gpu --output "$OUTPUT")
elif [[ $STAGE == nvlink-smoke ]]; then
  COMMAND+=("$OVERLAY/probe_shared_cuda_nvlink.py" --output "$OUTPUT")
elif [[ $STAGE == full-dual ]]; then
  WORKER_INDEX=${LATENT_READER_WORKER_GPU_INDEX:-1}
  [[ $WORKER_INDEX =~ ^[01]$ && $WORKER_INDEX != "$GPU_INDEX" ]] || exit 2
  CHANGING_UPDATES=${LATENT_READER_CHANGING_UPDATES:-4}
  EQUIVALENCE_UPDATES=${LATENT_READER_EQUIVALENCE_UPDATES:-2}
  [[ $CHANGING_UPDATES =~ ^[0-9]+$ && $CHANGING_UPDATES -ge 2 && $CHANGING_UPDATES -le 16 ]] || exit 2
  [[ $EQUIVALENCE_UPDATES =~ ^[0-9]+$ && $EQUIVALENCE_UPDATES -ge 2 && $EQUIVALENCE_UPDATES -le 16 ]] || exit 2
  [[ $((2 + 3 + CHANGING_UPDATES + 2 * EQUIVALENCE_UPDATES)) -le 16 ]] || exit 2
  COMMAND+=("$OVERLAY/probe_shared_latent_dual.py"
    --manifest "${LATENT_READER_DUAL_MANIFEST:?full dual check requires explicit admitted component manifest}"
    --output "$OUTPUT" --worker-cuda-device "$WORKER_INDEX"
    --repeats 3 --changing-updates "$CHANGING_UPDATES" --equivalence-updates "$EQUIVALENCE_UPDATES")
else
  PROBE="$OVERLAY/probe_shared_latent_reader.py"
  if [[ $STAGE == masks ]]; then PROBE="$OVERLAY/probe_shared_latent_reader_masks.py"; fi
  COMMAND+=("$PROBE" --component "$STAGE" --output "$OUTPUT"
    --backend-seconds "$BACKEND_SECONDS"
    --patch-owner "$ROOT/runtime/transfer/shared-combined-real-daily-20260925/probe_shared_expert_real_daily.py"
    --gate-owner "$ROOT/runtime/transfer/shared-union-micro4-20260929/probe_shared_union_micro4.py"
    --forward-probe "$ROOT/runtime/transfer/shared-forward-reverse-20260930/probe_shared_forward_reverse.py"
    --source-profile "$DIAG/shared-split-profile-c599236.json" --input-bundle "$DIAG/shared-split-dual-inputs-c599236.pkl"
    --prepared-context "$DIAG/shared-expert-real-daily-context-c599236.pkl"
    --daily-admission-report "$DIAG/shared-combined-real-daily-20260925.json"
    --seven-admission-report "$DIAG/shared-combined-recurrent-T7-20260926.json"
    --thirty-admission-report "$DIAG/shared-combined-recurrent-T30-20260926.json"
    --support-owner-file "$ROOT/runtime/transfer/shared-combined-full-update-20260926/probe_shared_serialized_dual.py"
    --runtime-preflight "$DIAG/shared-latent-reader-20261004-preflight.json"
    --backbone-module-file "$ROOT/runtime/transfer/shared-combined-recurrent-20260926/shared_backbone_execution.py")
  if [[ $STAGE == masks ]]; then
    COMMAND+=(--parameters "${LATENT_READER_PARAMETERS_FILE:?mask check requires exact saved fresh parameters}"
      --candidate-admission "${LATENT_READER_CANDIDATE_ADMISSION:?mask check requires passed real T7 admission}")
  elif [[ $STAGE == T7 ]]; then
    COMMAND+=(--original-full-reference "$DIAG/shared-recurrent-checkpoint-policy-20260930-v2-T7.reference.executable.pkl")
    if [[ -n ${LATENT_READER_PARAMETERS_FILE:-} ]]; then
      COMMAND+=(--parameters "$LATENT_READER_PARAMETERS_FILE")
    fi
  else
    COMMAND+=(--parameters "${LATENT_READER_PARAMETERS_FILE:?daily/T30 require the exact admitted saved fresh parameters path}")
    if [[ $STAGE == T30 ]]; then
      COMMAND+=(--original-full-reference "$DIAG/shared-original-t30-dual-reference-20261001.reference.executable.pkl"
        --original-full-admission "$DIAG/shared-original-t30-dual-reference-20261001.json")
    else
      COMMAND+=(--original-full-reference "$DIAG/shared-chunk-vmap-daily-dual-v4-20261001.reference.executable.pkl"
        --original-full-admission "$DIAG/shared-chunk-vmap-daily-dual-v4-20261001-reference.json")
    fi
  fi
  if [[ -n ${LATENT_READER_SAVED_EXECUTABLE:-} ]]; then
    [[ $STAGE != masks ]] || exit 2
    [[ -n $RUN_SUFFIX ]] || exit 2
    [[ -n ${LATENT_READER_PARAMETERS_FILE:-} ]] || { printf '%s\n' 'saved executable recovery requires LATENT_READER_PARAMETERS_FILE' >&2; exit 2; }
    COMMAND+=(--saved-executable "$LATENT_READER_SAVED_EXECUTABLE")
  fi
  if [[ ${LATENT_READER_DIAGNOSTIC_REPLAY_FAILURES:-0} == 1 ]]; then
    [[ -n ${LATENT_READER_SAVED_EXECUTABLE:-} && $STAGE == daily ]] || exit 2
    COMMAND+=(--diagnostic-continue-replay-failures
      --diagnostic-reference-result "${LATENT_READER_DIAGNOSTIC_REFERENCE_RESULT:?preserved candidate first full result required}"
      --diagnostic-prior-failure-report "${LATENT_READER_DIAGNOSTIC_PRIOR_FAILURE_REPORT:?preserved numerical-failure report required}")
  elif [[ ${LATENT_READER_DIAGNOSTIC_REPLAY_FAILURES:-0} != 0 ]]; then
    exit 2
  fi
  if [[ ${LATENT_READER_SAVE_LOWERED_IR:-0} == 1 ]]; then
    [[ $STAGE != masks && -z ${LATENT_READER_SAVED_EXECUTABLE:-} ]] || exit 2
    COMMAND+=(--save-lowered-ir)
  elif [[ ${LATENT_READER_SAVE_LOWERED_IR:-0} != 0 ]]; then
    exit 2
  fi
  FORCE_COMPILATION_PARALLELISM=${LATENT_READER_FORCE_COMPILATION_PARALLELISM:-0}
  case "$FORCE_COMPILATION_PARALLELISM" in 0|4) ;; *) exit 2 ;; esac
  if [[ $FORCE_COMPILATION_PARALLELISM == 4 ]]; then
    [[ $STAGE != masks && -z ${LATENT_READER_SAVED_EXECUTABLE:-} ]] || exit 2
    COMMAND+=(--force-compilation-parallelism 4)
  fi
fi
if [[ $COMPILER_DIAGNOSTICS == 1 ]]; then
  COMMAND=(env SINGULARITYENV_TF_CPP_VMODULE=hlo_pass_pipeline=1,gpu_compiler=1,llvm_gpu_backend=1 "${COMMAND[@]}")
fi
if [[ $COMPILER_HLO_SNAPSHOTS == 1 ]]; then
  [[ $STAGE == daily || $STAGE == nvlink-smoke ]] || exit 2
  HLO_DUMP_DIRECTORY="$DIAG/shared-latent-reader-20261004-$STAGE${RUN_SUFFIX:+-$RUN_SUFFIX}-compiler-hlo"
  HLO_DUMP_FLAGS="--xla_dump_to=$HLO_DUMP_DIRECTORY --xla_dump_hlo_as_text=true --xla_dump_hlo_module_re=^jit_objective$"
  COMMAND=(env "SINGULARITYENV_XLA_FLAGS=$HLO_DUMP_FLAGS" "${COMMAND[@]}")
fi
WRAPPED=(/apps/soft/anaconda3/bin/python "$OVERLAY/shared_gpu_budget.py"
  --ledger "$LEDGER" --stage "$STAGE" --seconds "$SESSION_SECONDS" --cards "$CARDS")
if [[ -n ${LATENT_READER_APPROVED_LIMIT_GPU_SECONDS:-} ]]; then
  [[ $LATENT_READER_APPROVED_LIMIT_GPU_SECONDS =~ ^[0-9]+$ && $LATENT_READER_APPROVED_LIMIT_GPU_SECONDS -ge 18000 ]] || exit 2
  [[ -n ${LATENT_READER_BUDGET_APPROVAL_NOTE:-} ]] || exit 2
  WRAPPED+=(--approved-limit-gpu-seconds "$LATENT_READER_APPROVED_LIMIT_GPU_SECONDS"
    --approval-note "$LATENT_READER_BUDGET_APPROVAL_NOTE")
elif [[ -n ${LATENT_READER_BUDGET_APPROVAL_NOTE:-} ]]; then
  exit 2
fi
WRAPPED+=(-- "${COMMAND[@]}")
if [[ $DRY_RUN == --dry-run ]]; then printf '%q ' "${WRAPPED[@]}"; printf '\n'; exit 0; fi
date --iso-8601=seconds
hostname
uptime
grep Cpus_allowed_list /proc/self/status
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
"${WRAPPED[@]}"
