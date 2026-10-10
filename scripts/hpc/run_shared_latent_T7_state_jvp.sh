#!/usr/bin/env bash
set -euo pipefail

STAGE=${1:?cpu-readiness or execute required}
shift
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime
PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE=$ROOT/checkouts/shared-split-width-v1
OVERLAY=$ROOT/transfer/T7-state-jvp-20261009
COMMON=$ROOT/transfer/shared-latent-reader-20261004
DIAG=$ROOT/outputs/diagnostics
TOOLCHAIN=$ROOT/toolchains/cuda-nvlink-12.6.85
PREFIX=${T7_STATE_JVP_PREFIX:-T7-state-jvp-20261009}
GPU=${T7_STATE_JVP_GPU_INDEX:-1}
DRY=${1:-}
RECOVERY=false
COMPLETION=false
[[ $PREFIX == T7-state-jvp-20261009-B4-recovery ]] && RECOVERY=true
[[ $PREFIX == T7-B1-state-jvp-completion-20261009 ]] && COMPLETION=true
[[ $COMPLETION == true ]] && OVERLAY=$ROOT/transfer/T7-B1-state-jvp-completion-20261009
FUNDING_SCOPE=latent_T7_state_jvp_20261009
[[ $COMPLETION == true ]] && FUNDING_SCOPE=latent_T7_B1_state_jvp_completion_20261009

[[ $GPU == 0 || $GPU == 1 ]]
[[ -z $DRY || $DRY == --dry-run ]]

case "$STAGE" in
  cpu-readiness) PLATFORM=cpu ;;
  execute) PLATFORM=cuda,cpu ;;
  *) echo "invalid stage: $STAGE" >&2; exit 2 ;;
esac

HOST=$(hostname -s 2>/dev/null || hostname)

OUTPUT=$DIAG/shared-latent-reader-20261004/$PREFIX.result.json
APPROVAL=$DIAG/shared-latent-reader-20261004/T7-state-jvp-20261009.approved.json
[[ $STAGE == cpu-readiness ]] && APPROVAL=$DIAG/shared-latent-reader-20261004/$PREFIX.approval.json
LEDGER=$DIAG/shared-latent-reader-20261004/T7-state-jvp-20261009.occupation.json
MEMORY=$DIAG/latent-reader-T7-20261005-state-jvp-20261009.memory.csv
NVLINK_STAGE=T7-state-jvp
REPAIR_ARGS=()
RUNTIME_ARGS=()
SESSION_SECONDS=3600
if [[ $RECOVERY == true ]]; then
  REPAIR_RECORD=$DIAG/shared-latent-reader-20261004/$PREFIX.recovery-budget.json
  REPAIR_ARGS+=(--repair-record "$REPAIR_RECORD")
  RUNTIME_ARGS+=(--runtime-preflight "$DIAG/shared-latent-reader-20261004/T7-state-jvp-20261009.runtime-preflight.json" --reuse-runtime-preflight)
  MEMORY=$DIAG/latent-reader-T7-20261005-state-jvp-20261009-B4-recovery.memory.csv
  NVLINK_STAGE=T7-state-jvp-b4-recovery
  SESSION_SECONDS=2400
fi
if [[ $COMPLETION == true ]]; then
  APPROVAL=$DIAG/shared-latent-reader-20261004/$PREFIX.approved.json
  [[ $STAGE == cpu-readiness ]] && APPROVAL=$DIAG/shared-latent-reader-20261004/$PREFIX.approval.json
  LEDGER=$DIAG/shared-latent-reader-20261004/$PREFIX.occupation.json
  MEMORY=$DIAG/latent-reader-T7-20261005-$PREFIX.memory.csv
  NVLINK_STAGE=T7-state-jvp-b1-completion
  SESSION_SECONDS=3600
  RUNTIME_ARGS+=(--runtime-preflight "$DIAG/shared-latent-reader-20261004/T7-state-jvp-20261009.runtime-preflight.json" --reuse-runtime-preflight)
fi
[[ $STAGE == cpu-readiness ]] && OUTPUT=$DIAG/shared-latent-reader-20261004/$PREFIX.cpu-readiness.json

COMMAND=(env -u LD_LIBRARY_PATH SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs
  SINGULARITYENV_CUDA_VISIBLE_DEVICES="$GPU" SINGULARITYENV_JAX_PLATFORMS="$PLATFORM"
  SINGULARITYENV_JAX_ENABLE_X64=true SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
  SINGULARITYENV_ORCHIDEE_REPO_ROOT="$CODE" SINGULARITYENV_PYTHONPATH="$CODE:$OVERLAY:$COMMON"
  SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1 SINGULARITYENV_OMP_NUM_THREADS=4
  SINGULARITYENV_OPENBLAS_NUM_THREADS=1
  SINGULARITYENV_PREPEND_PATH="$COMMON/nvlink-bin:$TOOLCHAIN/bin"
  SINGULARITYENV_ORCHIDEE_NVLINK_LOG="$OUTPUT.nvlink.log"
  SINGULARITYENV_ORCHIDEE_NVLINK_STAGE="$NVLINK_STAGE")
SINGULARITY=(singularity exec)
BIND_ARGS=(--bind "/WORK:/WORK,$ROOT/data:/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/data:ro,$TOOLCHAIN:$TOOLCHAIN:ro")
if [[ $STAGE == execute ]]; then
  CUDA_DRIVER_LIB=$(readlink -f /usr/lib64/libcuda.so)
  NVML_DRIVER_LIB=$(readlink -f /usr/lib64/libnvidia-ml.so)
  [[ -f $CUDA_DRIVER_LIB && -f $NVML_DRIVER_LIB ]]
  SINGULARITY+=(--nv)
  BIND_ARGS+=(--bind "$CUDA_DRIVER_LIB:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER_LIB:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1")
fi
COMMAND+=("${SINGULARITY[@]}" "${BIND_ARGS[@]}"
  --pwd "$CODE" /apps/soft/sif/foundationpose "$PROJECT/.venvs/orcjax_gpu/bin/python" -B
  "$OVERLAY/run_shared_latent_T7_state_jvp.py" --stage "$STAGE" --output-prefix "$PREFIX"
  --approval-report "$APPROVAL")
if [[ $COMPLETION == true ]]; then
  COMMAND+=(--b1-completion)
fi
if [[ $RECOVERY == true ]]; then
  COMMAND+=("${RUNTIME_ARGS[@]}" "${REPAIR_ARGS[@]}")
elif [[ $COMPLETION == true ]]; then
  COMMAND+=("${RUNTIME_ARGS[@]}")
fi
[[ $DRY == --dry-run && $COMPLETION == true ]] && COMMAND+=(--plan-only)

BOOTSTRAP='import importlib.util,json,sys; from pathlib import Path; a=json.load(open(sys.argv.pop(1))); helper=sys.argv.pop(1); expected_scope=sys.argv.pop(1); assert a["approved"] is True and a["funding_scope"]==expected_scope and a["limit_gpu_seconds"]==3600 and a["gpu_count"]==1 and a["model_compile_cap_per_route"]==1 and a["optimizer_update_cap"]==0; s=importlib.util.spec_from_file_location("state_jvp_budget",helper); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m.LIMIT=3600; m.SESSION_LIMIT=3600; p=Path(sys.argv[sys.argv.index("--ledger")+1]);
with m.locked(p) as d:
  d.setdefault("independent_funding_scope",a["funding_scope"]); d.setdefault("approval_binding",{"scope":a["funding_scope"],"output":a["output_report"],"entry":a["entry_script"],"cap":a["limit_gpu_seconds"]}); assert d["independent_funding_scope"]==a["funding_scope"] and d["approval_binding"]=={"scope":a["funding_scope"],"output":a["output_report"],"entry":a["entry_script"],"cap":a["limit_gpu_seconds"]} and d["limit_gpu_seconds"]==3600 and d["session_limit_seconds"]==3600
m.main()'
WRAPPED=(/apps/soft/anaconda3/bin/python -B -c "$BOOTSTRAP" "$APPROVAL" "$COMMON/shared_gpu_budget.py" "$FUNDING_SCOPE"
  --ledger "$LEDGER" --stage T7-state-jvp --seconds "$SESSION_SECONDS" --cards 1 --
  bash "$COMMON/shared_latent_T7_memory.sh" "$GPU" "$MEMORY" -- "${COMMAND[@]}")

if [[ $DRY == --dry-run ]]; then
  if [[ $STAGE == cpu-readiness ]]; then printf '%q ' "${COMMAND[@]}";
  else printf '%q ' "${WRAPPED[@]}"; fi
  printf '\n'
  exit 0
fi
if [[ $STAGE == cpu-readiness ]]; then exec "${COMMAND[@]}"; fi
[[ $STAGE == cpu-readiness ]] || [[ -f $APPROVAL && ! -e $OUTPUT && ( $HOST == gln01 || $HOST == ibc13b02n04 ) ]]
if [[ $RECOVERY == true ]]; then [[ -f $REPAIR_RECORD ]]; fi
[[ -d $CODE && -d $OVERLAY ]]
[[ $(cd "$CODE" && git rev-parse HEAD) == c599236bcf57b745e9bafed678298010f4c3b58f ]]
if [[ $STAGE == execute ]]; then
  [[ -z ${SLURM_JOB_ID:-} ]]
  [[ -z $(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits) ]]
  IFS=, read -r MEMORY_USED GPU_UTIL <<< "$(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)"
  [[ ${MEMORY_USED// /} == 0 && ${GPU_UTIL// /} == 0 ]]
  printf 'node=%s gpu=%s memory=%s utilization=%s\n' "$(hostname -s)" "$GPU" "$MEMORY_USED" "$GPU_UTIL"
  date --iso-8601=seconds
  uptime
fi
exec "${WRAPPED[@]}"
