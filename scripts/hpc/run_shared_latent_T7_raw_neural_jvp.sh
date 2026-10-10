#!/usr/bin/env bash
# New independent unapproved scope; no previous occupation ledger is reused.
set -euo pipefail
STAGE=${1:?cpu-readiness or execute required}
shift
DRY=${1:-}
[[ $# -le 1 && ( -z $DRY || $DRY == --dry-run ) ]]
ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime
PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE=$ROOT/checkouts/shared-split-width-v1
PREFIX=T7-raw-neural-jvp-20261010
OVERLAY=$ROOT/transfer/$PREFIX
COMMON=$ROOT/transfer/shared-latent-reader-20261004
DIAG=$ROOT/outputs/diagnostics/shared-latent-reader-20261004
TOOLCHAIN=$ROOT/toolchains/cuda-nvlink-12.6.85
GPU=${T7_RAW_NEURAL_JVP_GPU_INDEX:-1}
[[ $GPU == 0 || $GPU == 1 ]]
HOST=$(hostname -s)
case "$STAGE" in
  cpu-readiness) PLATFORM=cpu; VISIBLE=; APPROVAL=$DIAG/$PREFIX.approval.json ;;
  execute) PLATFORM=cuda,cpu; VISIBLE=$GPU; APPROVAL=$DIAG/$PREFIX.approved.json ;;
  *) exit 2 ;;
esac
COMMAND=(env -u LD_LIBRARY_PATH SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs
  SINGULARITYENV_CUDA_VISIBLE_DEVICES="$VISIBLE" SINGULARITYENV_JAX_PLATFORMS="$PLATFORM"
  SINGULARITYENV_JAX_ENABLE_X64=true SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
  SINGULARITYENV_ORCHIDEE_REPO_ROOT="$CODE" SINGULARITYENV_PYTHONPATH="$CODE:$OVERLAY:$COMMON"
  SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1 SINGULARITYENV_OMP_NUM_THREADS=4
  SINGULARITYENV_OPENBLAS_NUM_THREADS=1 SINGULARITYENV_PREPEND_PATH="$COMMON/nvlink-bin:$TOOLCHAIN/bin"
  SINGULARITYENV_ORCHIDEE_NVLINK_LOG="$DIAG/$PREFIX.result.json.nvlink.log"
  SINGULARITYENV_ORCHIDEE_NVLINK_STAGE="$PREFIX"
  singularity exec)
if [[ $STAGE == execute ]]; then
  COMMAND+=(--nv)
  if [[ $DRY != --dry-run ]]; then
    CUDA_DRIVER_LIB=$(readlink -f /usr/lib64/libcuda.so)
    NVML_DRIVER_LIB=$(readlink -f /usr/lib64/libnvidia-ml.so)
    [[ -f $CUDA_DRIVER_LIB && -f $NVML_DRIVER_LIB ]]
    COMMAND+=(--bind "$CUDA_DRIVER_LIB:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER_LIB:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1")
  fi
fi
COMMAND+=(--bind "/WORK:/WORK,$ROOT/data:$PROJECT/data:ro,$TOOLCHAIN:$TOOLCHAIN:ro"
  --pwd "$CODE" /apps/soft/sif/foundationpose "$PROJECT/.venvs/orcjax_gpu/bin/python" -B
  "$OVERLAY/run_shared_latent_T7_raw_neural_jvp.py" --stage "$STAGE" --approval-report "$APPROVAL")
BOOTSTRAP='import importlib.util,json,sys; from pathlib import Path; a=json.load(open(sys.argv.pop(1))); helper=sys.argv.pop(1); assert a["approved"] is True and a["approval_origin"]=="explicit_human_user" and a["approval_note"].strip() and a["funding_scope"]=="latent_T7_raw_neural_jvp_20261010" and type(a["limit_gpu_seconds"]) is int and a["limit_gpu_seconds"]==5400 and a["gpu_count"]==1 and a["model_compile_cap_per_route"]==1 and a["optimizer_update_cap"]==0; s=importlib.util.spec_from_file_location("first_day_budget",helper); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m.LIMIT=5400; m.SESSION_LIMIT=5400; p=Path(sys.argv[sys.argv.index("--ledger")+1]);
with m.locked(p) as d:
  binding={"scope":a["funding_scope"],"output":a["output_report"],"entry":a["entry_script"],"cap":5400}; d.setdefault("independent_funding_scope",a["funding_scope"]); d.setdefault("approval_binding",binding); assert d["independent_funding_scope"]==a["funding_scope"] and d["approval_binding"]==binding and d["limit_gpu_seconds"]==5400 and d["session_limit_seconds"]==5400
m.main()'
WRAPPED=(/apps/soft/anaconda3/bin/python -B -c "$BOOTSTRAP" "$APPROVAL" "$COMMON/shared_gpu_budget.py"
  --ledger "$DIAG/$PREFIX.occupation.json" --stage "$PREFIX" --seconds 5400 --cards 1 --
  bash "$COMMON/shared_latent_T7_memory.sh" "$GPU"
  "$ROOT/outputs/diagnostics/latent-reader-T7-20261005-$PREFIX.memory.csv" -- "${COMMAND[@]}")
if [[ $DRY == --dry-run ]]; then
  if [[ $STAGE == cpu-readiness ]]; then printf '%q ' "${COMMAND[@]}"; else printf '%q ' "${WRAPPED[@]}"; fi
  printf '\n'
  exit 0
fi
[[ -d $CODE && -d $OVERLAY ]]
[[ $(cd "$CODE" && git rev-parse HEAD) == c599236bcf57b745e9bafed678298010f4c3b58f ]]
if [[ $STAGE == cpu-readiness ]]; then
  [[ $HOST == test0[1-4] || $HOST == ibtest0[1-4] ]]
  exec "${COMMAND[@]}"
fi
[[ $HOST == gln01 || $HOST == ibc13b02n04 ]]
[[ -z ${SLURM_JOB_ID:-} && -f $APPROVAL && ! -e $DIAG/$PREFIX.result.json ]]
[[ -z $(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits) ]]
IFS=, read -r MEMORY_USED GPU_UTIL <<< "$(nvidia-smi -i "$GPU" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)"
[[ ${MEMORY_USED// /} == 0 && ${GPU_UTIL// /} == 0 ]]
date --iso-8601=seconds
uptime
printf 'node=%s gpu=%s affinity=%s\n' "$HOST" "$GPU" "$(taskset -pc $$)"
exec "${WRAPPED[@]}"
