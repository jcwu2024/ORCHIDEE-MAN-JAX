#!/usr/bin/env bash
# Bounded free-node T30 compilation and saved-graph comparison; never submits Slurm.
set -euo pipefail

ROOT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_ROOT="$ROOT/runtime/checkouts/shared-split-width-v1"
OVERLAY="$ROOT/runtime/transfer/shared-latent-reader-20261004"
DIAG="$ROOT/runtime/outputs/diagnostics"
TOOLCHAIN="$ROOT/runtime/toolchains/cuda-nvlink-12.6.85"
PYTHON=/apps/soft/anaconda3/bin/python
CONTAINER=/apps/soft/sif/foundationpose
CODE_REVISION=c599236bcf57b745e9bafed678298010f4c3b58f
MANIFEST="$OVERLAY/latent-reader-T30-20261006.manifest.json"
APPROVAL="$DIAG/latent-reader-T30-20261006-free-approval.json"
AGGREGATE_LEDGER="$DIAG/latent-reader-T30-20261006-free-budget.json"

die() { printf 'error: %s\n' "$*" >&2; exit 2; }
print_command() { printf '%q ' "$@"; printf '\n'; }

PHASE=${1:-}
[[ -n $PHASE ]] || die "usage: $0 {compile-original|compile-candidate|matched} [options]"
if [[ $PHASE == __run ]]; then
  shift
  INTERNAL_PHASE=${1:?}
  INTERNAL_RUN_ID=${2:?}
  INTERNAL_GPU=${3:?}
  INTERNAL_WALL=${4:?}
  INTERNAL_OUTPUT=${5:?}
  INTERNAL_PREFLIGHT=${6:?}
  INTERNAL_ORIGINAL=${7:-}
  INTERNAL_CANDIDATE=${8:-}
  INTERNAL_BACKEND=${9:?}
  INTERNAL_APPROVAL=${10:?}
  INTERNAL_LEDGER=${11:?}
  INTERNAL_OCCUPATION_LEDGER=${12:?}
  INTERNAL_MEMORY=${13:?}
  INTERNAL_COMMAND_SHA256=${14:?}
  INTERNAL_DRY=${15:-}
  if [[ $INTERNAL_DRY != --dry-run ]]; then
    "$PYTHON" -B "$OVERLAY/shared_latent_recurrent_phase_budget.py" verify-running \
      --approval-receipt "$INTERNAL_APPROVAL" --aggregate-ledger "$INTERNAL_LEDGER" \
      --occupation-ledger "$INTERNAL_OCCUPATION_LEDGER" --phase "$INTERNAL_PHASE" \
      --occupation-id "$INTERNAL_RUN_ID" --gpu-index "$INTERNAL_GPU" \
      --command-sha256 "$INTERNAL_COMMAND_SHA256"
    [[ $(hostname -s) == gln01 || $(hostname -s) == ibc13b02n04 ]] || die "free phase must run on gln01/ibc13b02n04"
    [[ -z ${SLURM_JOB_ID:-} ]] || die "free phase refuses a Slurm allocation"
    [[ $(cd "$CODE_ROOT" && git rev-parse HEAD) == "$CODE_REVISION" ]] || die "immutable code revision differs"
    [[ ! -e $INTERNAL_OUTPUT && ! -e $INTERNAL_PREFLIGHT ]] || die "preserve existing T30 phase evidence"
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
    SINGULARITYENV_JAX_COMPILATION_CACHE_DIR="$ROOT/runtime/cache/jax/orcjax_gpu/latent-reader-T30-20261006"
    SINGULARITYENV_JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
    SINGULARITYENV_JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
    SINGULARITYENV_PREPEND_PATH="$OVERLAY/nvlink-bin:$TOOLCHAIN/bin"
    SINGULARITYENV_ORCHIDEE_NVLINK_LOG="$INTERNAL_OUTPUT.nvlink.log"
    SINGULARITYENV_ORCHIDEE_NVLINK_STAGE="latent-T30-$INTERNAL_PHASE"
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
    "$ROOT/.venvs/orcjax_gpu/bin/python" -B "$OVERLAY/probe_shared_latent_reader_T7_pair.py"
    --stage "$INTERNAL_PHASE" --horizon 30
    --manifest "$MANIFEST"
    --prepared-context "$DIAG/shared-expert-real-daily-context-c599236.pkl"
    --parameters "$DIAG/shared-latent-reader-20261004-T7-assembled-input.parameters.pkl"
    --field-adapter "$ROOT/runtime/transfer/shared-combined-recurrent-20260926/shared_decoder_expert_execution.py"
    --backbone-adapter "$ROOT/runtime/transfer/shared-combined-recurrent-20260926/shared_backbone_execution.py"
    --transformer-module "$OVERLAY/shared_transformer.py"
    --latent-module "$OVERLAY/shared_latent_reader.py"
    --turnover-overlay "$OVERLAY/turnover_leaf_age_fall.guarded-source.py"
    --runtime-preflight "$INTERNAL_PREFLIGHT"
    --wall-seconds "$INTERNAL_WALL" --lower-seconds 600 --backend-seconds "$INTERNAL_BACKEND"
    --call-seconds 180 --pairs 3 --force-compilation-parallelism 4 --output "$INTERNAL_OUTPUT")
  case "$INTERNAL_PHASE" in
    compile-original|compile-candidate) ;;
    matched)
      [[ -n $INTERNAL_ORIGINAL && -n $INTERNAL_CANDIDATE ]] || die "matched requires both corrected T30 executable paths"
      if [[ $INTERNAL_DRY != --dry-run ]]; then
        [[ -f $INTERNAL_ORIGINAL && -f $INTERNAL_CANDIDATE ]] || die "matched executables must already exist; compile fallback is forbidden"
      fi
      PROBE_COMMAND+=(--original-executable "$INTERNAL_ORIGINAL" --candidate-executable "$INTERNAL_CANDIDATE")
      ;;
    *) die "invalid internal phase" ;;
  esac
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
  [[ $memory =~ ^[0-9]+$ && $utilization =~ ^[0-9]+$ ]] || { printf "post-preflight GPU state is not numeric\\n" >&2; exit 125; }
  (( memory <= 100 )) || { printf "post-preflight selected GPU memory exceeds 100 MiB\\n" >&2; exit 125; }
  (( utilization == 0 )) && break
  (( SECONDS < deadline )) || { printf "post-preflight selected GPU did not become idle within 20 seconds\\n" >&2; exit 125; }
  sleep 1
done
"${probe[@]}"
'
  PHASE_COMMAND=(bash -c "$PHASE_SEQUENCE" _ "$INTERNAL_PREFLIGHT" "$INTERNAL_GPU" "${#PREFLIGHT_COMMAND[@]}" "${PREFLIGHT_COMMAND[@]}" -- "${PROBE_COMMAND[@]}")
  MEMORY_COMMAND=(bash "$OVERLAY/shared_latent_T7_memory.sh" "$INTERNAL_GPU" "$INTERNAL_MEMORY" -- "${PHASE_COMMAND[@]}")
  if [[ $INTERNAL_DRY == --dry-run ]]; then
    bash -n <<< "$PHASE_SEQUENCE" || die "post-preflight phase sequence has invalid Bash syntax"
    print_command "${PREFLIGHT_COMMAND[@]}"
    print_command "${MEMORY_COMMAND[@]}"
    exit 0
  fi
  "${MEMORY_COMMAND[@]}"
  exit $?
fi

shift
APPROVAL_FILE=$APPROVAL
LEDGER_FILE=$AGGREGATE_LEDGER
GPU_INDEX=
CPU_AFFINITY=8-15
OCCUPATION_SECONDS=
RUN_ID=
ORIGINAL_EXECUTABLE=
CANDIDATE_EXECUTABLE=
DRY_RUN=0
while (($#)); do
  case $1 in
    --approval-receipt) APPROVAL_FILE=${2:?}; shift 2 ;;
    --ledger) LEDGER_FILE=${2:?}; shift 2 ;;
    --gpu-index) GPU_INDEX=${2:?}; shift 2 ;;
    --cpu-affinity) CPU_AFFINITY=${2:?}; shift 2 ;;
    --occupation-seconds) OCCUPATION_SECONDS=${2:?}; shift 2 ;;
    --run-id) RUN_ID=${2:?}; shift 2 ;;
    --original-executable) ORIGINAL_EXECUTABLE=${2:?}; shift 2 ;;
    --candidate-executable) CANDIDATE_EXECUTABLE=${2:?}; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) die "unknown option: $1" ;;
  esac
done

case "$PHASE" in
  compile-original) DEFAULT_SECONDS=4800; INTERNAL_BACKEND=3600 ;;
  compile-candidate) DEFAULT_SECONDS=7200; INTERNAL_BACKEND=6000 ;;
  matched) DEFAULT_SECONDS=3600; INTERNAL_BACKEND=6000 ;;
  *) die "phase must be compile-original, compile-candidate or matched" ;;
esac
[[ $GPU_INDEX == 0 || $GPU_INDEX == 1 ]] || die "gln01 GPU index must be 0 or 1"
[[ $CPU_AFFINITY == 8-15 ]] || die "this approved scope pins exactly CPUs 8-15"
[[ -z $RUN_ID ]] && { if ((DRY_RUN)); then RUN_ID=00000000-0000-4000-8000-000000000000; else RUN_ID=$(/apps/soft/anaconda3/bin/python -c 'import uuid; print(uuid.uuid4())'); fi; }
[[ $RUN_ID =~ ^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$ ]] || die "run id must be a UUID"
[[ -z $OCCUPATION_SECONDS ]] && OCCUPATION_SECONDS=$DEFAULT_SECONDS
[[ $OCCUPATION_SECONDS =~ ^[0-9]+$ && $OCCUPATION_SECONDS -ge 120 && $OCCUPATION_SECONDS -le $DEFAULT_SECONDS ]] || die "occupation seconds exceed this phase's per-occupation cap"
NON_PROBE_RESERVE=360
PROBE_WALL=$((OCCUPATION_SECONDS-NON_PROBE_RESERVE))
[[ $PROBE_WALL -ge 120 ]] || die "occupation must leave at least 120 seconds for phase work"

OUTPUT="$DIAG/latent-reader-T30-20261006-$PHASE-$RUN_ID.json"
PREFLIGHT="$DIAG/latent-reader-T30-20261006-$PHASE-$RUN_ID.runtime-preflight.json"
OCCUPATION_LEDGER="$DIAG/latent-reader-T30-20261006-$PHASE-$RUN_ID.occupation.json"
MEMORY_CSV="$DIAG/latent-reader-T30-20261006-$PHASE-$RUN_ID.memory.csv"
if [[ $PHASE == matched ]]; then
  [[ -n $ORIGINAL_EXECUTABLE && -n $CANDIDATE_EXECUTABLE ]] || die "matched requires --original-executable and --candidate-executable"
  [[ $ORIGINAL_EXECUTABLE != "$CANDIDATE_EXECUTABLE" ]] || die "matched needs distinct original and candidate graphs"
  for executable in "$ORIGINAL_EXECUTABLE" "$CANDIDATE_EXECUTABLE"; do
    executable_name=${executable##*/}
    [[ $executable == "$DIAG"/* && $executable_name =~ ^latent-reader-T30-20261006-compile-(original|candidate)-[0-9a-fA-F-]{36}\.executable\.pkl$ ]] || die "matched executable path is outside the T30 compile artifact family"
    if (( ! DRY_RUN )); then [[ -f $executable && -f $executable.identity.json ]] || die "matched requires saved graph and identity sidecar"; fi
  done
  [[ ${ORIGINAL_EXECUTABLE##*/} == *-compile-original-* ]] || die "first executable must be the corrected original graph"
  [[ ${CANDIDATE_EXECUTABLE##*/} == *-compile-candidate-* ]] || die "second executable must be the corrected candidate graph"
else
  [[ -z $ORIGINAL_EXECUTABLE && -z $CANDIDATE_EXECUTABLE ]] || die "compile occupations do not accept existing executable arguments"
fi
if (( ! DRY_RUN )); then
  [[ -f $CODE_ROOT/.git/HEAD ]] || die "immutable source checkout is unavailable"
  [[ -f $OVERLAY/probe_shared_latent_reader_T7_pair.py && -f $OVERLAY/shared_latent_recurrent_phase_budget.py && -f $OVERLAY/shared_latent_T7_memory.sh ]] || die "deployed T30 runner/budget/memory helper is missing"
  [[ ! -e $OUTPUT && ! -e $PREFLIGHT && ! -e $MEMORY_CSV && ! -e $MEMORY_CSV.processes.csv ]] || die "preserve existing phase output; use a new UUID for a retest"
  if [[ $PHASE != matched ]]; then
    GRAPH_FAMILY="$DIAG/latent-reader-T30-20261006-$PHASE-*.executable.pkl"
    compgen -G "$GRAPH_FAMILY" >/dev/null && die "an executable already exists for this architecture; compilation is load-only"
    compgen -G "$GRAPH_FAMILY.identity.json" >/dev/null && die "an executable identity already exists for this architecture; compilation is load-only"
  fi
fi

INTERNAL=(bash "$0" __run "$PHASE" "$RUN_ID" "$GPU_INDEX" "$PROBE_WALL" "$OUTPUT" "$PREFLIGHT" "$ORIGINAL_EXECUTABLE" "$CANDIDATE_EXECUTABLE" "$INTERNAL_BACKEND" "$APPROVAL_FILE" "$LEDGER_FILE" "$OCCUPATION_LEDGER" "$MEMORY_CSV")
INTERNAL_COMMAND_SHA256=$("$PYTHON" -B "$OVERLAY/shared_latent_recurrent_phase_budget.py" fingerprint-command -- "${INTERNAL[@]}")
INTERNAL+=("$INTERNAL_COMMAND_SHA256")
WRAPPED=("$PYTHON" "$OVERLAY/shared_latent_recurrent_phase_budget.py" execute
  --approval-receipt "$APPROVAL_FILE" --aggregate-ledger "$LEDGER_FILE" --occupation-ledger "$OCCUPATION_LEDGER"
  --phase "$PHASE" --occupation-id "$RUN_ID" --seconds "$OCCUPATION_SECONDS" --gpu-index "$GPU_INDEX"
  --cpu-affinity "$CPU_AFFINITY" -- "${INTERNAL[@]}")
if ((DRY_RUN)); then
  printf 'phase=%s funding=free scope=latent_reader_T30_free_15600_20261006\n' "$PHASE"
  printf 'approval_gate=requires approved=true receipt; template is rejected\n'
  printf 'gpu_index=%s cpu_affinity=%s omp_threads=4 openblas_threads=1\n' "$GPU_INDEX" "$CPU_AFFINITY"
  printf 'host_memory_precheck_gib=64 rss_limit=none\n'
  printf 'aggregate_ledger=%s\noccupation_ledger=%s\nrun_id=%s\n' "$LEDGER_FILE" "$OCCUPATION_LEDGER" "$RUN_ID"
  printf 'occupation_cap_seconds=%s cumulative_cap_seconds=15600\n' "$OCCUPATION_SECONDS"
  printf 'finalization_charge_reserve_seconds=60\n'
  printf 'continuous_gpu_memory_and_process_logs=%s,%s.processes.csv\n' "$MEMORY_CSV" "$MEMORY_CSV"
  printf 'probe_wall_seconds=%s\n' "$PROBE_WALL"
  print_command "${WRAPPED[@]}"
  bash "$0" __run "$PHASE" "$RUN_ID" "$GPU_INDEX" "$PROBE_WALL" "$OUTPUT" "$PREFLIGHT" "$ORIGINAL_EXECUTABLE" "$CANDIDATE_EXECUTABLE" "$INTERNAL_BACKEND" "$APPROVAL_FILE" "$LEDGER_FILE" "$OCCUPATION_LEDGER" "$MEMORY_CSV" "$INTERNAL_COMMAND_SHA256" --dry-run
  exit 0
fi
[[ -f $APPROVAL_FILE ]] || die "approved T30 free-scope receipt is required"
exec "${WRAPPED[@]}"
