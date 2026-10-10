#!/bin/bash
# Free idle gln01 only. This launcher never submits a scheduler job.
set -euo pipefail
LAUNCH_STARTED=$(date +%s.%N)
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT="$ROOT/runtime/checkouts/shared-split-width-v1"
OVERLAY="$ROOT/runtime/transfer/shared-command-buffer-loops-20261001"
DIAG="$ROOT/runtime/outputs/diagnostics"
GPU_INDEX=${GPU_INDEX:-0}
DRY_RUN=0
if [[ ${1:-} == --dry-run ]]; then DRY_RUN=1; shift; fi
MODE=${1:-support}
ARM=${2:-default}
RUNTIME_PROFILE=${RUNTIME_PROFILE:-legacy}
case "$RUNTIME_PROFILE" in legacy|compat) ;; *) exit 2 ;; esac
case "$ARM" in default|while) ;; *) exit 2 ;; esac
RUN_ID=${RUN_ID:-20261001}
[[ $RUN_ID =~ ^[a-zA-Z0-9-]+$ ]] || exit 2
case "$MODE" in support|support-replay|threshold-support|threshold-component|runtime-support|runtime-preflight|runtime-component|runtime-replay|chunk-vmap|stage-screen|shape-preserved|shape-preserved-daily|shape-preserved-daily-hlo|shape-preserved-T7|shape-preserved-T7-replay|whole-vmap-T7|whole-vmap-T7-replay|whole-vmap-T30-reference|whole-vmap-T30|original-T30-reference|shape-preserved-T30) ;; *) exit 2 ;; esac
if [[ $MODE == chunk-vmap || $MODE == shape-preserved ]]; then [[ $ARM == default && $RUNTIME_PROFILE == legacy ]] || exit 2; fi
if [[ $MODE == threshold-* ]]; then [[ $ARM == default ]] || exit 2; fi
if [[ $MODE == runtime-preflight ]]; then [[ $ARM == default ]] || exit 2; fi
if [[ $RUNTIME_PROFILE == compat ]]; then [[ $MODE == runtime-* ]] || exit 2; fi
case "$GPU_INDEX" in 0|1) ;; *) exit 2 ;; esac
OUTPUT="$DIAG/shared-command-buffer-loops-$MODE-$ARM-$RUN_ID.json"
if [[ $MODE == shape-preserved-T7-replay ]]; then OUTPUT="$DIAG/shared-command-buffer-loops-shape-preserved-T7-$ARM-$RUN_ID.replay.json"; fi
if [[ $MODE == whole-vmap-T7-replay ]]; then OUTPUT="$DIAG/shared-command-buffer-loops-whole-vmap-T7-$ARM-$RUN_ID.replay.json"; fi
GPU_STARTED=false
PHASE=validating_checkout
LAUNCHER_RECORD="$OUTPUT.launcher.txt"
if [[ -e $OUTPUT || -e $LAUNCHER_RECORD ]]; then LAUNCHER_RECORD="$OUTPUT.attempt-$$.launcher.txt"; fi
finish_record() {
  STATUS=$?
  LAUNCH_FINISHED=$(date +%s.%N)
  if [[ $DRY_RUN == 1 ]]; then return; fi
  {
    printf 'gpu_started=%s\npaid_jobs=0\nruntime_profile=%s\nphase=%s\nstart_epoch=%s\nfinish_epoch=%s\nexit_status=%s\n' "$GPU_STARTED" "$RUNTIME_PROFILE" "$PHASE" "$LAUNCH_STARTED" "$LAUNCH_FINISHED" "$STATUS"
    printf 'observed_gpu_memory_util=%s\n' "${LINE:-not_checked}"
    awk -v start="$LAUNCH_STARTED" -v finish="$LAUNCH_FINISHED" -v started="$GPU_STARTED" 'BEGIN {printf "launcher_wall_seconds=%.9f\ngpu_count=%d\nlauncher_card_hours=%.12f\n", finish-start, (started=="true" ? 1 : 0), (started=="true" ? (finish-start)/3600 : 0)}'
  } > "$LAUNCHER_RECORD"
  printf 'run_exit_status=%s\n' "$STATUS"
}
trap finish_record EXIT
[[ $(git --git-dir="$CODE_ROOT/.git" --work-tree="$CODE_ROOT" rev-parse HEAD) == c599236bcf57b745e9bafed678298010f4c3b58f ]] || exit 2
TYPES=FUSION,CUBLAS,CUSTOM_CALL,CUDNN
if [[ $ARM == while ]]; then TYPES+=,WHILE; fi
PROCESS_BUDGET=180s
VLOG_LEVEL=0
if [[ $MODE == support && $ARM == while ]]; then VLOG_LEVEL=2; fi
if [[ $MODE == threshold-component || $MODE == runtime-component || $MODE == chunk-vmap || $MODE == stage-screen || $MODE == shape-preserved ]]; then PROCESS_BUDGET=900s; fi
if [[ $MODE == shape-preserved-daily ]]; then PROCESS_BUDGET=2110s; fi
if [[ $MODE == shape-preserved-T7 || $MODE == whole-vmap-T7 || $MODE == whole-vmap-T30-reference || $MODE == whole-vmap-T30 || $MODE == original-T30-reference || $MODE == shape-preserved-T30 ]]; then PROCESS_BUDGET=7210s; fi
if [[ $MODE == shape-preserved-T7-replay || $MODE == whole-vmap-T7-replay ]]; then PROCESS_BUDGET=910s; fi
CUDA_LIBRARY_ENV=(SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs)
RUNTIME_ARGS=(--runtime-profile "$RUNTIME_PROFILE")
if [[ $RUNTIME_PROFILE == compat ]]; then
  PHASE=validating_manifest_and_libraries
  TOOLCHAIN_ROOT="$ROOT/runtime/toolchains/cuda-runtime-compat-12.3.101-driver-545.23.08"
  COMPAT_DIR=${COMPAT_DIR:-$TOOLCHAIN_ROOT/usr/local/cuda-12.3/compat}
  CUDART_DIR=${CUDART_DIR:-$TOOLCHAIN_ROOT/usr/local/cuda-12.3/targets/x86_64-linux/lib}
  RUNTIME_MANIFEST=${RUNTIME_MANIFEST:-$TOOLCHAIN_ROOT/manifest.json}
  [[ $COMPAT_DIR == "$ROOT/runtime/toolchains/"* && $CUDART_DIR == "$ROOT/runtime/toolchains/"* && $RUNTIME_MANIFEST == "$ROOT/runtime/toolchains/"* ]] || exit 2
  if [[ $DRY_RUN == 0 ]]; then
    # The lead releases this prepared interface only after manifest acceptance.
    [[ ${CANDIDATE_RUNTIME_GPU_CONFIRMED:-0} == 1 && -f $RUNTIME_MANIFEST ]] || exit 2
    for LIBRARY in "$COMPAT_DIR/libcuda.so.1" "$CUDART_DIR/libcudart.so.12"; do
      RESOLVED_LIBRARY=$(readlink -f "$LIBRARY")
      [[ -f $RESOLVED_LIBRARY && $RESOLVED_LIBRARY == "$ROOT/runtime/toolchains/"* ]] || exit 2
    done
  fi
  CUDA_LIBRARY_ENV=(SINGULARITYENV_LD_LIBRARY_PATH="$COMPAT_DIR:$CUDART_DIR:/.singularity.d/libs"
    SINGULARITYENV_LD_PRELOAD="$COMPAT_DIR/libcuda.so.1:$CUDART_DIR/libcudart.so.12")
  RUNTIME_ARGS+=(--runtime-manifest "$RUNTIME_MANIFEST")
  if [[ $MODE != runtime-preflight ]]; then
    RUNTIME_ARGS+=(--runtime-preflight "$DIAG/shared-command-buffer-loops-runtime-preflight-default-$RUN_ID.json")
  fi
fi
if [[ $DRY_RUN == 0 ]]; then
  PHASE=validating_node_and_gpu_occupancy
  [[ $(hostname -s) == ibc13b02n04 && -z ${SLURM_JOB_ID:-} && ! -e $OUTPUT ]] || exit 2
  LINE=$(nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  read -r MEMORY UTIL <<<"${LINE//,/ }"
  [[ $MEMORY =~ ^[0-9]+$ && $UTIL =~ ^[0-9]+$ && $MEMORY -le 2000 && $UTIL -le 5 ]] || exit 2
  [[ -z $(nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader,nounits) ]] || exit 2
fi
CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
ENTRY_SCRIPT="$OVERLAY/probe_shared_command_buffer_loops.py"
ENTRY_ARGS=(--mode "$MODE" --arm "$ARM" --output "$OUTPUT" "${RUNTIME_ARGS[@]}")
if [[ $MODE == stage-screen ]]; then
  [[ $ARM == default && $RUNTIME_PROFILE == legacy ]] || exit 2
  ENTRY_SCRIPT="$OVERLAY/probe_shared_chunk_batch_stages.py"
  ENTRY_ARGS=(--output "$OUTPUT" --payload "$DIAG/shared-normalized-keys-payload-20261001.pkl")
fi
DIAGNOSTIC_XLA_FLAGS="--xla_gpu_enable_llvm_module_compilation_parallelism=false --xla_gpu_enable_command_buffer=$TYPES --xla_dump_to=$OVERLAY/dumps-$MODE-$ARM-$RUN_ID --xla_dump_hlo_pass_re=command-buffer-scheduling"
if [[ $MODE == shape-preserved-daily ]]; then
  [[ $ARM == default && $RUNTIME_PROFILE == legacy ]] || exit 2
  COMBINED="$ROOT/runtime/transfer/shared-combined-recurrent-20260926"
  ENTRY_SCRIPT="$OVERLAY/probe_shared_normalized_keys_daily.py"
  ENTRY_ARGS=(--output "$OUTPUT" --lower-seconds 240 --backend-seconds 1500 --process-seconds 2100
    --patch-owner "$ROOT/runtime/transfer/shared-combined-real-daily-20260925/probe_shared_expert_real_daily.py"
    --gate-owner "$ROOT/runtime/transfer/shared-union-micro4-20260929/probe_shared_union_micro4.py"
    --source-profile "$DIAG/shared-split-profile-c599236.json"
    --input-bundle "$DIAG/shared-split-dual-inputs-c599236.pkl"
    --prepared-context "$DIAG/shared-expert-real-daily-context-c599236.pkl"
    --baseline-admission "$DIAG/shared-combined-real-daily-20260925.json"
    --runtime-preflight "$DIAG/shared-command-buffer-loops-runtime-preflight-default-20261001-vmap.public-preflight.json"
    --candidate-module-file "$COMBINED/shared_decoder_expert_execution.py"
    --backbone-module-file "$COMBINED/shared_backbone_execution.py"
    --execution-module-file "$OVERLAY/shared_chunk_batch_execution.py")
  DIAGNOSTIC_XLA_FLAGS="--xla_gpu_enable_llvm_module_compilation_parallelism=false --xla_gpu_enable_command_buffer=$TYPES"
fi
if [[ $MODE == shape-preserved-daily-hlo ]]; then
  [[ $ARM == default && $RUNTIME_PROFILE == legacy ]] || exit 2
  ENTRY_SCRIPT="$OVERLAY/inspect_saved_chunk_batch_hlo.py"
  ENTRY_ARGS=(--executable "$DIAG/shared-command-buffer-loops-shape-preserved-daily-default-20261001-vmap.candidate.executable.pkl"
    --output "$OUTPUT" --compress-hlo)
  DIAGNOSTIC_XLA_FLAGS="--xla_gpu_enable_llvm_module_compilation_parallelism=false --xla_gpu_enable_command_buffer=$TYPES"
fi
if [[ $MODE == shape-preserved-T7 || $MODE == shape-preserved-T7-replay || $MODE == whole-vmap-T7 || $MODE == whole-vmap-T7-replay || $MODE == whole-vmap-T30-reference || $MODE == whole-vmap-T30 || $MODE == original-T30-reference || $MODE == shape-preserved-T30 ]]; then
  [[ $ARM == default && $RUNTIME_PROFILE == legacy ]] || exit 2
  COMBINED="$ROOT/runtime/transfer/shared-combined-recurrent-20260926"
  REF_BASE="$DIAG/shared-recurrent-checkpoint-policy-20260930-v2-T7"
  HORIZON=7
  RECURRENT_STAGE=candidate
  RECURRENT_PROCESS_BUDGET=7200
  if [[ $MODE == shape-preserved-T7-replay || $MODE == whole-vmap-T7-replay ]]; then RECURRENT_PROCESS_BUDGET=900; fi
  EXECUTION_MODULE="$OVERLAY/shared_chunk_batch_execution.py"
  EXECUTION_VARIANT=preserve128
  REQUIRED_DAILY="$DIAG/shared-command-buffer-loops-shape-preserved-daily-default-20261001-vmap.json"
  CANDIDATE_MODE=shape-preserved-T7
  if [[ $MODE == whole-vmap-* ]]; then
    EXECUTION_MODULE="$ROOT/runtime/transfer/shared-chunk-vmap-daily-20261001/shared_chunk_vmap_execution.py"
    EXECUTION_VARIANT=whole-chunk-vmap
    REQUIRED_DAILY="$DIAG/shared-chunk-vmap-daily-expert5-micro4-20261001-v2.json"
    CANDIDATE_MODE=whole-vmap-T7
  fi
  REF_REPORT="$REF_BASE-reference.json"
  if [[ $MODE == original-T30-reference || $MODE == shape-preserved-T30 || $MODE == whole-vmap-T30-reference || $MODE == whole-vmap-T30 ]]; then
    HORIZON=30
    REF_BASE="$DIAG/shared-command-buffer-loops-original-T30-reference-default-$RUN_ID"
    REF_REPORT="$REF_BASE.json"
    if [[ $MODE == original-T30-reference ]]; then RECURRENT_STAGE=reference; fi
    if [[ $MODE == whole-vmap-T30-reference || $MODE == whole-vmap-T30 ]]; then
      REF_BASE="$DIAG/shared-command-buffer-loops-whole-vmap-T30-reference-default-$RUN_ID"
      REF_REPORT="$REF_BASE.json"
      if [[ $MODE == whole-vmap-T30-reference ]]; then RECURRENT_STAGE=reference; fi
    fi
  fi
  ENTRY_SCRIPT="$OVERLAY/probe_shared_normalized_keys_recurrent.py"
  ENTRY_ARGS=(--horizon "$HORIZON" --stage "$RECURRENT_STAGE" --output "$OUTPUT"
    --lower-seconds 360 --backend-seconds 5400 --process-seconds "$RECURRENT_PROCESS_BUDGET"
    --patch-owner "$ROOT/runtime/transfer/shared-combined-real-daily-20260925/probe_shared_expert_real_daily.py"
    --gate-owner "$ROOT/runtime/transfer/shared-union-micro4-20260929/probe_shared_union_micro4.py"
    --forward-probe "$ROOT/runtime/transfer/shared-forward-reverse-20260930/probe_shared_forward_reverse.py"
    --source-profile "$DIAG/shared-split-profile-c599236.json" --input-bundle "$DIAG/shared-split-dual-inputs-c599236.pkl"
    --prepared-context "$DIAG/shared-expert-real-daily-context-c599236.pkl"
    --daily-admission-report "$DIAG/shared-combined-real-daily-20260925.json"
    --seven-admission-report "$DIAG/shared-combined-recurrent-T7-20260926.json"
    --thirty-admission-report "$DIAG/shared-combined-recurrent-T30-20260926.json"
    --support-owner-file "$ROOT/runtime/transfer/shared-combined-full-update-20260926/probe_shared_serialized_dual.py"
    --runtime-preflight "$DIAG/shared-command-buffer-loops-runtime-preflight-default-20261001-vmap.public-preflight.json"
    --candidate-module-file "$COMBINED/shared_decoder_expert_execution.py"
    --backbone-module-file "$COMBINED/shared_backbone_execution.py"
    --reference-report "$REF_REPORT" --reference-result "$REF_BASE.reference.pkl"
    --reference-executable "$REF_BASE.reference.executable.pkl"
    --execution-module-file "$EXECUTION_MODULE" --execution-variant "$EXECUTION_VARIANT"
    --required-daily-report "$REQUIRED_DAILY")
  if [[ $HORIZON == 30 ]]; then
    if [[ $EXECUTION_VARIANT == whole-chunk-vmap ]]; then
      ENTRY_ARGS+=(--required-seven-report "$DIAG/shared-command-buffer-loops-whole-vmap-T7-default-$RUN_ID.json"
        --required-seven-replay-report "$DIAG/shared-command-buffer-loops-whole-vmap-T7-default-$RUN_ID.replay.json")
    else
      ENTRY_ARGS+=(--required-seven-report "$DIAG/shared-command-buffer-loops-shape-preserved-T7-default-20261001-vmap.json"
        --required-seven-replay-report "$DIAG/shared-command-buffer-loops-shape-preserved-T7-default-20261001-vmap.replay.json")
    fi
  fi
  if [[ $MODE == shape-preserved-T7-replay || $MODE == whole-vmap-T7-replay ]]; then
    ENTRY_SCRIPT="$OVERLAY/probe_shared_chunk_batch_replay.py"
    ENTRY_ARGS+=(--candidate-report "$DIAG/shared-command-buffer-loops-$CANDIDATE_MODE-default-$RUN_ID.json")
  fi
  DIAGNOSTIC_XLA_FLAGS="--xla_gpu_enable_llvm_module_compilation_parallelism=false --xla_gpu_enable_command_buffer=$TYPES"
fi
COMMAND=(timeout --signal=TERM --kill-after=10s "$PROCESS_BUDGET" env -u LD_LIBRARY_PATH -u LD_PRELOAD
  "${CUDA_LIBRARY_ENV[@]}" SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU_INDEX"
  SINGULARITYENV_JAX_PLATFORMS=cuda,cpu SINGULARITYENV_JAX_ENABLE_X64=true
  SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false SINGULARITYENV_ORCHIDEE_REPO_ROOT="$CODE_ROOT"
  SINGULARITYENV_PYTHONPATH="$CODE_ROOT:$OVERLAY" SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
  SINGULARITYENV_PYTHONFAULTHANDLER=1
  SINGULARITYENV_TF_CPP_VMODULE=command_buffer_scheduling=2
  SINGULARITYENV_TF_CPP_MAX_VLOG_LEVEL="$VLOG_LEVEL" SINGULARITYENV_TF_CPP_MIN_LOG_LEVEL=0
  SINGULARITYENV_OPENBLAS_NUM_THREADS=1 SINGULARITYENV_OMP_NUM_THREADS=4
  SINGULARITYENV_XLA_FLAGS="$DIAGNOSTIC_XLA_FLAGS"
  singularity exec --nv --bind "/WORK:/WORK,$ROOT/runtime/data:$ROOT/data:ro,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
  --pwd "$CODE_ROOT" /apps/soft/sif/foundationpose "$ROOT/.venvs/orcjax_gpu/bin/python"
  "$ENTRY_SCRIPT" "${ENTRY_ARGS[@]}")
if [[ $MODE == support-replay ]]; then
  COMMAND+=(--default-report "$DIAG/shared-command-buffer-loops-support-default-20261001.json"
    --while-report "$DIAG/shared-command-buffer-loops-support-while-20261001.json")
fi
if [[ $MODE == threshold-component || $MODE == runtime-component || $MODE == runtime-replay || $MODE == chunk-vmap || $MODE == shape-preserved ]]; then
  COMMAND+=(--payload "$DIAG/shared-normalized-keys-payload-20261001.pkl" --microbatch 1 --route expert5)
fi
if [[ $MODE == chunk-vmap || $MODE == shape-preserved ]]; then
  COMMAND+=(--default-report "$DIAG/shared-command-buffer-loops-threshold-component-default-20261001.json")
fi
if [[ $MODE == runtime-replay ]]; then
  COMMAND+=(--default-report "$DIAG/shared-command-buffer-loops-runtime-component-default-$RUN_ID.json"
    --while-report "$DIAG/shared-command-buffer-loops-runtime-component-while-$RUN_ID.json")
fi
if [[ $DRY_RUN == 1 ]]; then printf '%q ' "${COMMAND[@]}"; printf '\n'; exit 0; fi
date --iso-8601=seconds
hostname
uptime
grep Cpus_allowed_list /proc/self/status
nvidia-smi --query-gpu=index,name,driver_version,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
PHASE=running_gpu_command
GPU_STARTED=true
"${COMMAND[@]}"
