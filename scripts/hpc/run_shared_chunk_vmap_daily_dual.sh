#!/bin/bash
set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT="$ROOT/runtime/checkouts/shared-split-width-v1"
OVERLAY="$ROOT/runtime/transfer/shared-chunk-vmap-daily-dual-20261001"
V4_OVERLAY="$ROOT/runtime/transfer/shared-chunk-vmap-daily-20261001"
DAILY_OVERLAY="$ROOT/runtime/transfer/shared-combined-real-daily-20260925"
BACKBONE_OVERLAY="$ROOT/runtime/transfer/shared-combined-recurrent-20260926"
GATE_OVERLAY="$ROOT/runtime/transfer/shared-union-micro4-20260929"
DIAG="$ROOT/runtime/outputs/diagnostics"
STEM=shared-chunk-vmap-daily-dual-v4-20261001
GPU_INDEX=${GPU_INDEX:-1}
DRY_RUN=0
if [[ ${1:-} == --dry-run ]]; then DRY_RUN=1; shift; fi
[[ ${1:-} =~ ^(preflight|reference|candidate)$ && $# == 1 ]] || exit 2
STAGE=$1
[[ $GPU_INDEX == 1 ]] || exit 2
PREFLIGHT="$DIAG/$STEM-preflight.public.json"
if [[ $STAGE == preflight ]]; then
  OUTPUT=$PREFLIGHT
else
  OUTPUT="$DIAG/$STEM-$STAGE.json"
fi
START=$(date +%s.%N)
GPU_STARTED=false
PHASE=validating_paths
RECORD_OWNED=false
if [[ ! -e $OUTPUT && ! -e $OUTPUT.launcher.txt ]]; then RECORD_OWNED=true; fi
finish_record() {
  local status=$?
  local finish
  finish=$(date +%s.%N)
  if [[ $DRY_RUN == 1 || $RECORD_OWNED != true ]]; then return; fi
  {
    printf 'variant=field_expert128+backbone_source_reuse+expert5_whole_chunk_axis_vmap_daily_dual\nstage=%s\ngpu_index=%s\npaid_jobs=0\nphase=%s\nstart_epoch=%s\nfinish_epoch=%s\nexit_status=%s\n' "$STAGE" "$GPU_INDEX" "$PHASE" "$START" "$finish" "$status"
    printf 'observed_gpu_memory_util=%s\n' "${GPU_LINE:-not_checked}"
    awk -v a="$START" -v b="$finish" -v started="$GPU_STARTED" 'BEGIN {printf "launcher_wall_seconds=%.9f\ngpu_count=%d\ncard_hours=%.12f\n", b-a, (started=="true"?1:0), (started=="true"?(b-a)/3600:0)}'
  } > "$OUTPUT.launcher.txt"
  printf 'run_exit_status=%s\n' "$status"
}
trap finish_record EXIT
[[ $(git --git-dir="$CODE_ROOT/.git" --work-tree="$CODE_ROOT" rev-parse HEAD) == c599236bcf57b745e9bafed678298010f4c3b58f ]] || exit 2
[[ -f $OVERLAY/shared_daily_dual_ad.py && -f $OVERLAY/probe_shared_chunk_vmap_daily_dual.py ]] || exit 2
[[ -f $V4_OVERLAY/shared_chunk_vmap_execution.py ]] || exit 2
[[ -f $DAILY_OVERLAY/probe_shared_expert_real_daily.py && -f $GATE_OVERLAY/probe_shared_union_micro4.py ]] || exit 2
[[ -f $BACKBONE_OVERLAY/shared_decoder_expert_execution.py && -f $BACKBONE_OVERLAY/shared_backbone_execution.py ]] || exit 2
[[ -f $DIAG/shared-split-profile-c599236.json && -f $DIAG/shared-split-dual-inputs-c599236.pkl ]] || exit 2
[[ -f $DIAG/shared-expert-real-daily-context-c599236.pkl && -f $DIAG/shared-combined-real-daily-20260925.json ]] || exit 2
[[ -f $DIAG/shared-chunk-vmap-daily-expert5-micro4-20261001-v2.json ]] || exit 2
[[ -f $DIAG/shared-chunk-vmap-daily-expert5-micro4-20261001-v2.reference.pkl ]] || exit 2

if [[ $STAGE != preflight ]]; then
  [[ -f $PREFLIGHT ]] || exit 2
  python3 - "$PREFLIGHT" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
if report.get("status") != "passed" or report.get("passed") is not True or report.get("actual_backend") != "gpu":
    raise SystemExit("daily runtime preflight did not pass on GPU")
PY
fi
[[ ! -e $OUTPUT && ! -e $OUTPUT.launcher.txt ]] || exit 2
if [[ $DRY_RUN == 0 ]]; then
  [[ $(hostname -s) == ibc13b02n04 && -z ${SLURM_JOB_ID:-} ]] || exit 2
  GPU_LINE=$(nvidia-smi -i 1 --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)
  read -r MEMORY UTIL <<<"${GPU_LINE//,/ }"
  [[ $MEMORY =~ ^[0-9]+$ && $UTIL =~ ^[0-9]+$ && $MEMORY -le 2000 && $UTIL -le 5 ]] || exit 2
  [[ -z $(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader,nounits) ]] || exit 2
fi

CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
[[ -f $CUDA_DRIVER && -f $NVML_DRIVER ]]
PHASE=building_command
COMMON_ENV=(env -u LD_LIBRARY_PATH -u LD_PRELOAD
  SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs
  SINGULARITYENV_CUDA_VISIBLE_DEVICES=1
  SINGULARITYENV_JAX_PLATFORMS=cuda,cpu SINGULARITYENV_JAX_ENABLE_X64=true
  SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
  SINGULARITYENV_ORCHIDEE_REPO_ROOT="$CODE_ROOT"
  SINGULARITYENV_PYTHONPATH="$CODE_ROOT:$OVERLAY"
  SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1 SINGULARITYENV_PYTHONFAULTHANDLER=1
  SINGULARITYENV_OPENBLAS_NUM_THREADS=1 SINGULARITYENV_OMP_NUM_THREADS=4
  SINGULARITYENV_XLA_FLAGS=--xla_gpu_enable_llvm_module_compilation_parallelism=false)
SINGULARITY=(singularity exec --nv --bind "/WORK:/WORK,$ROOT/runtime/data:$ROOT/data:ro,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
  --pwd "$CODE_ROOT" /apps/soft/sif/foundationpose "$ROOT/.venvs/orcjax_gpu/bin/python")

if [[ $STAGE == preflight ]]; then
  COMMAND=(timeout --signal=TERM --kill-after=10s 300s "${COMMON_ENV[@]}" "${SINGULARITY[@]}"
    "$CODE_ROOT/scripts/hpc/check_daily_training_runtime.py" --backend gpu --output "$OUTPUT")
else
  DUAL_ARGS=(--stage "$STAGE" --output "$OUTPUT"
    --result-output "$DIAG/$STEM.$STAGE.pkl"
    --executable-output "$DIAG/$STEM.$STAGE.executable.pkl"
    --lower-seconds 240 --backend-seconds 1500 --process-seconds 4200
    --patch-owner "$DAILY_OVERLAY/probe_shared_expert_real_daily.py"
    --gate-owner "$GATE_OVERLAY/probe_shared_union_micro4.py"
    --source-profile "$DIAG/shared-split-profile-c599236.json"
    --input-bundle "$DIAG/shared-split-dual-inputs-c599236.pkl"
    --prepared-context "$DIAG/shared-expert-real-daily-context-c599236.pkl"
    --baseline-admission "$DIAG/shared-combined-real-daily-20260925.json"
    --runtime-preflight "$PREFLIGHT"
    --candidate-module-file "$BACKBONE_OVERLAY/shared_decoder_expert_execution.py"
    --backbone-module-file "$BACKBONE_OVERLAY/shared_backbone_execution.py"
    --execution-module-file "$V4_OVERLAY/shared_chunk_vmap_execution.py"
    --parameter-report "$DIAG/shared-chunk-vmap-daily-expert5-micro4-20261001-v2.json"
    --parameter-result "$DIAG/shared-chunk-vmap-daily-expert5-micro4-20261001-v2.reference.pkl")
  if [[ $STAGE == candidate ]]; then
    DUAL_ARGS+=(--dual-reference-report "$DIAG/$STEM-reference.json"
      --dual-reference-result "$DIAG/$STEM.reference.pkl"
      --dual-reference-executable "$DIAG/$STEM.reference.executable.pkl")
  fi
  COMMAND=(timeout --signal=TERM --kill-after=10s 4210s "${COMMON_ENV[@]}" "${SINGULARITY[@]}"
    "$OVERLAY/probe_shared_chunk_vmap_daily_dual.py" "${DUAL_ARGS[@]}")
fi

if [[ $DRY_RUN == 1 ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi
PHASE=running_gpu_command
GPU_STARTED=true
date --iso-8601=seconds
hostname
uptime
grep Cpus_allowed_list /proc/self/status
nvidia-smi --query-gpu=index,name,driver_version,memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader
"${COMMAND[@]}"
