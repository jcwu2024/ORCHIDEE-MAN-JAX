#!/usr/bin/env bash
# One separately approved, single-occupation corrected daily compile/AD probe.
set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT="$ROOT/runtime/checkouts/shared-split-width-v1"
OVERLAY="$ROOT/runtime/transfer/shared-latent-reader-20261004"
DIAG="$ROOT/runtime/outputs/diagnostics"
SCOPE_DIAG="$DIAG/shared-latent-reader-20261004"
TOOLCHAIN="$ROOT/runtime/toolchains/cuda-nvlink-12.6.85"
PYTHON=/apps/soft/anaconda3/bin/python
CONTAINER=/apps/soft/sif/foundationpose
CODE_REVISION=c599236bcf57b745e9bafed678298010f4c3b58f
MANIFEST="$SCOPE_DIAG/corrected-full-update-manifest.json"
SOURCE_PROFILE="$DIAG/shared-split-profile-c599236.json"
PREPARED_CONTEXT="$DIAG/shared-expert-real-daily-context-c599236.pkl"
PARAMETERS="$DIAG/shared-latent-reader-20261004-T7-assembled-input.parameters.pkl"
FIELD_ADAPTER="$ROOT/runtime/transfer/shared-combined-recurrent-20260926/shared_decoder_expert_execution.py"
BACKBONE_ADAPTER="$ROOT/runtime/transfer/shared-combined-recurrent-20260926/shared_backbone_execution.py"
TRANSFORMER_MODULE="$OVERLAY/shared_transformer.py"
LATENT_MODULE="$OVERLAY/shared_latent_reader.py"
TURNOVER_OVERLAY="$OVERLAY/turnover_leaf_age_fall.guarded-source.py"
PROBE="$OVERLAY/probe_shared_latent_corrected_daily.py"
BUDGET="$OVERLAY/shared_latent_daily_budget.py"
MEMORY_HELPER="$OVERLAY/shared_latent_corrected_daily_memory.sh"
APPROVAL="$DIAG/latent-reader-daily-corrected-20261007-free-approval.json"
LEDGER="$DIAG/latent-reader-daily-corrected-20261007-free-budget.json"
OUTPUT="$DIAG/latent-reader-daily-20261006-corrected-candidate.json"
CANDIDATE_EXECUTABLE="${OUTPUT%.json}.executable.pkl"
COMPILE_RECEIPT="${OUTPUT%.json}.compile.json"
IDENTITY_SIDECAR="$CANDIDATE_EXECUTABLE.identity.json"
PREFLIGHT="$DIAG/latent-reader-daily-20261006-corrected-candidate.runtime-preflight.json"
MEMORY_CSV="$OUTPUT.memory.csv"
PROBE_WALL=6840
LOWER_SECONDS=600
BACKEND_SECONDS=4800
CALL_SECONDS=180
PAIRS=3
FINALIZATION_RESERVE_SECONDS=360

die() { printf 'error: %s\n' "$*" >&2; exit 2; }
print_command() { printf '%q ' "$@"; printf '\n'; }

if [[ ${1:-} != __run ]]; then
[[ ${1:-} == run ]] || die "usage: $0 run --gpu-index {0|1} [--occupation-id UUID] [--occupation-seconds 7200] [--dry-run]"
shift
GPU_INDEX=
CPU_AFFINITY=8-15
OCCUPATION_SECONDS=7200
RUN_ID=
DRY_RUN=0
while (($#)); do
  case $1 in
    --gpu-index) GPU_INDEX=${2:?}; shift 2 ;;
    --cpu-affinity) CPU_AFFINITY=${2:?}; shift 2 ;;
    --occupation-seconds) OCCUPATION_SECONDS=${2:?}; shift 2 ;;
    --occupation-id) RUN_ID=${2:?}; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) die "unknown option: $1" ;;
  esac
done
[[ $GPU_INDEX == 0 || $GPU_INDEX == 1 ]] || die "select one gln01 V100 by index"
[[ $CPU_AFFINITY == 8-15 ]] || die "daily scope pins CPUs 8-15"
[[ $OCCUPATION_SECONDS == 7200 ]] || die "fresh daily scope reserves exactly 7200 seconds"
if [[ -z $RUN_ID ]]; then
  if ((DRY_RUN)); then RUN_ID=00000000-0000-4000-8000-000000000000
  else RUN_ID=$(/apps/soft/anaconda3/bin/python -B -c 'import uuid; print(uuid.uuid4())')
  fi
fi
[[ $RUN_ID =~ ^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$ ]] || die "occupation id must be a UUID"
OCCUPATION_LEDGER="$DIAG/latent-reader-daily-corrected-20261007-free-$RUN_ID.occupation.json"
INTERNAL=(bash "$0" __run "$RUN_ID" "$GPU_INDEX" "$PROBE_WALL" "$OUTPUT" "$PREFLIGHT" "$MEMORY_CSV" "$APPROVAL" "$LEDGER" "$OCCUPATION_LEDGER" "$CPU_AFFINITY")
INTERNAL_COMMAND_SHA256=$("$PYTHON" -B "$BUDGET" fingerprint-command -- "${INTERNAL[@]}")
INTERNAL+=("$INTERNAL_COMMAND_SHA256")
WRAPPED=("$PYTHON" -B "$BUDGET" execute --approval-receipt "$APPROVAL"
  --aggregate-ledger "$LEDGER" --occupation-ledger "$OCCUPATION_LEDGER"
  --occupation-id "$RUN_ID" --seconds "$OCCUPATION_SECONDS" --gpu-index "$GPU_INDEX"
  --cpu-affinity "$CPU_AFFINITY" -- "${INTERNAL[@]}")

if ((DRY_RUN)); then
  printf 'scope=shared_latent_corrected_daily_free_7200_20261007 funding=free\n'
  printf 'approval_gate=fresh approved daily receipt required; T30 approval/ledger are rejected\n'
  printf 'gpu_index=%s cpu_affinity=%s omp_threads=4 openblas_threads=1\n' "$GPU_INDEX" "$CPU_AFFINITY"
  printf 'occupation_seconds=%s probe_wall_seconds=%s finalization_reserve_seconds=%s\n' "$OCCUPATION_SECONDS" "$PROBE_WALL" "$FINALIZATION_RESERVE_SECONDS"
  printf 'lower_seconds=%s backend_seconds=%s call_seconds=%s pairs=%s candidate_compiles=1 original_compiles=0 complete_ad_calls=8\n' "$LOWER_SECONDS" "$BACKEND_SECONDS" "$CALL_SECONDS" "$PAIRS"
  printf 'source_revision=%s\nmanifest=%s\noutput=%s\nexecutable=%s\n' "$CODE_REVISION" "$MANIFEST" "$OUTPUT" "$CANDIDATE_EXECUTABLE"
  printf 'approval=%s\naggregate_ledger=%s\noccupation_ledger=%s\nrun_id=%s\n' "$APPROVAL" "$LEDGER" "$OCCUPATION_LEDGER" "$RUN_ID"
  print_command "${WRAPPED[@]}"
  bash "$0" __run "$RUN_ID" "$GPU_INDEX" "$PROBE_WALL" "$OUTPUT" "$PREFLIGHT" "$MEMORY_CSV" "$APPROVAL" "$LEDGER" "$OCCUPATION_LEDGER" "$CPU_AFFINITY" "$INTERNAL_COMMAND_SHA256" --dry-run
  exit 0
fi

[[ -f $APPROVAL ]] || die "a fresh approved 7200-second daily receipt is required"
[[ -f $MANIFEST ]] || die "corrected full-update manifest is missing"
[[ ! -e $OUTPUT && ! -e $PREFLIGHT && ! -e $COMPILE_RECEIPT && ! -e $CANDIDATE_EXECUTABLE && ! -e $IDENTITY_SIDECAR && ! -e $MEMORY_CSV && ! -e "$MEMORY_CSV.processes.csv" ]] || die "preserve existing candidate evidence; this single-use grant cannot overwrite it"
exec "${WRAPPED[@]}"
fi

if [[ ${1:-} == __run ]]; then
  shift
  INTERNAL_RUN_ID=${1:?}
  INTERNAL_GPU=${2:?}
  INTERNAL_WALL=${3:?}
  INTERNAL_OUTPUT=${4:?}
  INTERNAL_PREFLIGHT=${5:?}
  INTERNAL_MEMORY=${6:?}
  INTERNAL_APPROVAL=${7:?}
  INTERNAL_LEDGER=${8:?}
  INTERNAL_OCCUPATION_LEDGER=${9:?}
  INTERNAL_CPU_AFFINITY=${10:?}
  INTERNAL_COMMAND_SHA256=${11:?}
  INTERNAL_DRY=${12:-}
  if [[ $INTERNAL_DRY != --dry-run ]]; then
    "$PYTHON" -B "$BUDGET" verify-running --approval-receipt "$INTERNAL_APPROVAL" \
      --aggregate-ledger "$INTERNAL_LEDGER" --occupation-ledger "$INTERNAL_OCCUPATION_LEDGER" \
      --occupation-id "$INTERNAL_RUN_ID" --gpu-index "$INTERNAL_GPU" \
      --command-sha256 "$INTERNAL_COMMAND_SHA256"
    [[ $(hostname -s) == gln01 || $(hostname -s) == ibc13b02n04 ]] || die "free daily occupation is restricted to gln01"
    [[ -z ${SLURM_JOB_ID:-} ]] || die "free daily occupation refuses a Slurm allocation"
    [[ $(cd "$CODE_ROOT" && git rev-parse HEAD) == "$CODE_REVISION" ]] || die "immutable source revision differs"
    [[ -f $MANIFEST && -f $PROBE && -f $BUDGET && -f $MEMORY_HELPER ]] || die "daily manifest/probe/guard/memory helper is missing"
    [[ ! -e $INTERNAL_OUTPUT ]] || die "preserve existing daily report"
    internal_executable="${INTERNAL_OUTPUT%.json}.executable.pkl"
    internal_compile_receipt="${INTERNAL_OUTPUT%.json}.compile.json"
    [[ ! -e $INTERNAL_PREFLIGHT && ! -e $internal_compile_receipt && ! -e $internal_executable && ! -e "$internal_executable.identity.json" ]] || die "preserve existing daily phase evidence"
  fi

  if [[ $INTERNAL_DRY == --dry-run ]]; then
    CUDA_DRIVER=/usr/lib64/libcuda.so.1
    NVML_DRIVER=/usr/lib64/libnvidia-ml.so.1
  else
    CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
    NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
    [[ -n $CUDA_DRIVER && -n $NVML_DRIVER ]] || die "CUDA/NVML driver libraries are unavailable"
  fi
  CONTAINER_PREFIX=(env -u LD_LIBRARY_PATH
    SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs
    SINGULARITYENV_CUDA_VISIBLE_DEVICES="$INTERNAL_GPU"
    SINGULARITYENV_JAX_PLATFORMS=cuda,cpu
    SINGULARITYENV_JAX_ENABLE_X64=true
    SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false
    SINGULARITYENV_JAX_COMPILATION_CACHE_DIR="$ROOT/runtime/cache/jax/orcjax_gpu/latent-reader-daily-corrected-20261007"
    SINGULARITYENV_JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
    SINGULARITYENV_JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
    SINGULARITYENV_PREPEND_PATH="$OVERLAY/nvlink-bin:$TOOLCHAIN/bin"
    SINGULARITYENV_ORCHIDEE_NVLINK_LOG="$INTERNAL_OUTPUT.nvlink.log"
    SINGULARITYENV_ORCHIDEE_NVLINK_STAGE=latent-daily-corrected-candidate
    SINGULARITYENV_ORCHIDEE_REPO_ROOT="$CODE_ROOT"
    SINGULARITYENV_PYTHONPATH="$CODE_ROOT:$OVERLAY"
    SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1
    SINGULARITYENV_OPENBLAS_NUM_THREADS=1
    SINGULARITYENV_OMP_NUM_THREADS=4)
  CONTAINER_ARGS=(--nv --bind "/WORK:/WORK,$ROOT/runtime/data:$ROOT/data:ro,$TOOLCHAIN:$TOOLCHAIN:ro,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1"
    --pwd "$CODE_ROOT" "$CONTAINER")
  PREFLIGHT_COMMAND=("${CONTAINER_PREFIX[@]}" singularity exec "${CONTAINER_ARGS[@]}"
    "$ROOT/.venvs/orcjax_gpu/bin/python" -B "$CODE_ROOT/scripts/hpc/check_daily_training_runtime.py"
    --backend gpu --output "$INTERNAL_PREFLIGHT")
  PROBE_COMMAND=("${CONTAINER_PREFIX[@]}" singularity exec "${CONTAINER_ARGS[@]}"
    "$ROOT/.venvs/orcjax_gpu/bin/python" -B "$PROBE"
    --manifest "$MANIFEST" --source-profile "$SOURCE_PROFILE"
    --prepared-context "$PREPARED_CONTEXT" --parameters "$PARAMETERS"
    --field-adapter "$FIELD_ADAPTER" --backbone-adapter "$BACKBONE_ADAPTER"
    --transformer-module "$TRANSFORMER_MODULE" --latent-module "$LATENT_MODULE"
    --turnover-overlay "$TURNOVER_OVERLAY" --runtime-preflight "$INTERNAL_PREFLIGHT"
    --output "$INTERNAL_OUTPUT" --wall-seconds "$INTERNAL_WALL"
    --lower-seconds "$LOWER_SECONDS" --backend-seconds "$BACKEND_SECONDS"
    --call-seconds "$CALL_SECONDS" --pairs "$PAIRS" --force-compilation-parallelism 4)
  PHASE_SEQUENCE='
set -e
output=$1
gpu=$2
count=$3
shift 3
preflight=("${@:1:count}")
shift "$count"
shift
probe=("$@")
timeout --kill-after=5s 120s "${preflight[@]}"
test -s "$output"
deadline=$((SECONDS + 20))
while true; do
  apps=$(timeout 3s nvidia-smi -i "$gpu" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits) || { printf "post-preflight GPU process query failed\\n" >&2; exit 125; }
  [[ -z $apps || $apps == *"No running processes"* ]] || { printf "post-preflight selected GPU has a compute process\\n" >&2; exit 125; }
  gpu_state=$(timeout 3s nvidia-smi -i "$gpu" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits) || { printf "post-preflight GPU state query failed\\n" >&2; exit 125; }
  IFS=, read -r memory utilization <<< "$gpu_state"
  memory=${memory//[[:space:]]/}
  utilization=${utilization//[[:space:]]/}
  [[ $memory =~ ^[0-9]+$ && $utilization =~ ^[0-9]+$ ]] || { printf "post-preflight selected GPU state is not numeric\\n" >&2; exit 125; }
  (( memory <= 100 )) || { printf "post-preflight selected GPU memory exceeds 100 MiB\\n" >&2; exit 125; }
  (( utilization == 0 )) && break
  (( SECONDS < deadline )) || { printf "post-preflight selected GPU did not become idle within 20 seconds\\n" >&2; exit 125; }
  sleep 1
done
"${probe[@]}"
'
  PHASE_COMMAND=(bash -c "$PHASE_SEQUENCE" _ "$INTERNAL_PREFLIGHT" "$INTERNAL_GPU" "${#PREFLIGHT_COMMAND[@]}" "${PREFLIGHT_COMMAND[@]}" -- "${PROBE_COMMAND[@]}")
  MEMORY_COMMAND=(bash "$MEMORY_HELPER" "$INTERNAL_GPU" "$INTERNAL_MEMORY" -- "${PHASE_COMMAND[@]}")
  if [[ $INTERNAL_DRY == --dry-run ]]; then
    bash -n <<< "$PHASE_SEQUENCE" || die "post-preflight daily command has invalid Bash syntax"
    print_command "${PREFLIGHT_COMMAND[@]}"
    print_command "${MEMORY_COMMAND[@]}"
    exit 0
  fi
  "${MEMORY_COMMAND[@]}"
  exit $?
fi

die "invalid internal daily runner mode"
