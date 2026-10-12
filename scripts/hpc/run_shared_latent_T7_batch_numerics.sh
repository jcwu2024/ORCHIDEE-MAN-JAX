#!/usr/bin/env bash
# Ordinary entry is CPU-only. The supervisor launches runtime-child under its
# already-started process-group deadline; driver/container startup is charged.
set -euo pipefail
SHELL_STARTED_EPOCH=$(date +%s.%N)
PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CPU_PYTHON=$PROJECT/.venvs/orcjax_cpu/bin/python
GPU_PYTHON=$PROJECT/.venvs/orcjax_gpu/bin/python
TOOLCHAIN=$PROJECT/runtime/toolchains/cuda-nvlink-12.6.85
CODE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
RUNNER=$CODE/scripts/hpc/run_shared_latent_T7_batch_numerics.py

# Called in an already-created phase session, before readlink/container work.
# The timer stays in that same group, so its PGID cannot be recycled while the
# supervisor is stalled. No Python cleanup is needed to enforce this deadline.
start_phase_deadman() {
  local phase_leader=$1 work_deadline=$2 hard_deadline=$3
  local PHASE_LEADER_PGID
  PHASE_LEADER_PGID=$(ps -o pgid= -p "$phase_leader" | tr -d ' ') || true
  if declare -F paid_startup_shell_evidence >/dev/null; then paid_startup_shell_evidence phase_group_binding; fi
  [[ $PHASE_LEADER_PGID == "$phase_leader" ]] || return 2
  (
    trap '' TERM INT
    local now term_sent=false
    while true; do
      # In this subshell BASHPID is the timer PID; $$ remains the phase leader.
      [[ $(ps -o pgid= -p "$BASHPID" | tr -d ' ') == "$phase_leader" ]] || exit 2
      now=$(date +%s)
      if ! kill -0 "$phase_leader" 2>/dev/null; then
        kill -KILL -- "-$phase_leader" 2>/dev/null || true
        exit 0
      fi
      if ((now >= hard_deadline)); then
        kill -KILL -- "-$phase_leader" 2>/dev/null || true
        exit 0
      fi
      if ((now >= work_deadline)) && [[ $term_sent == false ]]; then
        term_sent=true
        kill -TERM -- "-$phase_leader" 2>/dev/null || true
      fi
      sleep 1
    done
  ) >/dev/null 2>&1 &
  PHASE_TIMER_PID=$!
}

runtime_child() {
  local runtime_mode=$1
  shift
  [[ $# -ge 2 && $1 == "$GPU_PYTHON" ]] || exit 2
  CHILD_PYTHON=$1
  ENTRY=$2
  shift 2
  # Only the deadline-owning Python supervisor supplies this internal mode.
  [[ -z ${ORCJAX_NUMERICS_LAUNCHER_PYTHON:-} ]] || exit 2
  if [[ $runtime_mode == --runtime-discovery ]]; then
    [[ $ENTRY == "$CODE/scripts/hpc/shared_affine_cuda_identity.py" && ${ORCJAX_AFFINE_GPU_EXECUTION_CONTRACT-} == T7_affine_allocated_single_gpu_v2 && -z ${ORCJAX_AFFINE_EXPECTED_GPU_UUID-} ]] || exit 2
    [[ ${CUDA_VISIBLE_DEVICES-} == "${ORCJAX_SLURM_VISIBLE_DEVICES-}" && ( $CUDA_VISIBLE_DEVICES =~ ^[0-9]+$ || $CUDA_VISIBLE_DEVICES =~ ^GPU-[0-9a-fA-F-]+$ ) ]] || exit 2
  else
    [[ $ENTRY == "$CODE/scripts/hpc/check_daily_training_runtime.py" || $ENTRY == "$CODE/scripts/hpc/probe_shared_latent_T7_batch_numerics.py" ]] || exit 2
    [[ ${CUDA_VISIBLE_DEVICES:-} == GPU-* && $CUDA_VISIBLE_DEVICES != *,* ]] || exit 2
  fi
  [[ -f $ENTRY ]] || exit 2
  for ARG in "$@"; do [[ $ARG != --dry-run && $ARG != --dry-run=* ]] || exit 2; done
  PHASE_EPOCH=${ORCJAX_NUMERICS_SHELL_STARTED_EPOCH:-}
  [[ $PHASE_EPOCH =~ ^[0-9]+([.][0-9]+)?$ ]] || exit 2
  PHASE_START=${PHASE_EPOCH%.*}
  PHASE_NOW=$(date +%s)
  ((PHASE_START <= PHASE_NOW+1 && PHASE_NOW-PHASE_START < 1200)) || exit 2
  start_phase_deadman "$$" "$((PHASE_START+1010))" "$((PHASE_START+1180))" || exit 2
  CUDA_DRIVER=$(readlink -f /usr/lib64/libcuda.so.1)
  NVML_DRIVER=$(readlink -f /usr/lib64/libnvidia-ml.so.1)
  # The paid entry opts in; free callers gain no extra read or output.
  if declare -F paid_startup_shell_evidence >/dev/null; then paid_startup_shell_evidence runtime_driver_bindings; fi
  [[ -f $CUDA_DRIVER && -f $NVML_DRIVER && -d $TOOLCHAIN ]] || exit 2
  # Numeric scheduler masks must keep the original CUDA ordering in the same
  # runtime namespace. Never invent PCI_BUS_ID ordering to reinterpret them.
  if [[ ${ORCJAX_AFFINE_GPU_EXECUTION_CONTRACT-} == T7_affine_allocated_single_gpu_v2 ]]; then
    if [[ -v CUDA_DEVICE_ORDER ]]; then export SINGULARITYENV_CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER
    else unset SINGULARITYENV_CUDA_DEVICE_ORDER; fi
    export SINGULARITYENV_ORCJAX_SLURM_VISIBLE_DEVICES=${ORCJAX_SLURM_VISIBLE_DEVICES-}
    export SINGULARITYENV_SLURM_JOB_GPUS=${SLURM_JOB_GPUS-}
  fi
  exec env -u LD_LIBRARY_PATH \
    SINGULARITYENV_LD_LIBRARY_PATH=/.singularity.d/libs \
    SINGULARITYENV_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    SINGULARITYENV_JAX_PLATFORMS=cuda,cpu \
    SINGULARITYENV_JAX_ENABLE_X64=true \
    SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false \
    SINGULARITYENV_ORCHIDEE_REPO_ROOT="$CODE" \
    SINGULARITYENV_PYTHONPATH="$CODE:$CODE/scripts/hpc" \
    SINGULARITYENV_PYTHONDONTWRITEBYTECODE=1 \
    SINGULARITYENV_OMP_NUM_THREADS=4 \
    SINGULARITYENV_OPENBLAS_NUM_THREADS=1 \
    SINGULARITYENV_PREPEND_PATH="$TOOLCHAIN/bin" \
    singularity exec --nv \
      --bind "/WORK:/WORK,$CUDA_DRIVER:/usr/lib/x86_64-linux-gnu/libcuda.so.1,$NVML_DRIVER:/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1,$TOOLCHAIN:$TOOLCHAIN:ro" \
      --pwd "$CODE" /apps/soft/sif/foundationpose \
      "$CHILD_PYTHON" -B "$ENTRY" "$@"
}

# Injectable shell function for local clock/setsid/kill mocks; production main
# fixes the absolute deadlines and interpreter and exposes no test-cap flag.
guarded_cpu_launch() {
  local work_deadline=$1 hard_deadline=$2 marker=$3
  shift 3
  local cpu_pid timer_pid result now delay
  [[ ! -e $marker ]] || return 2
  command -v setsid >/dev/null || { printf '%s\n' 'setsid required for live launch' >&2; return 2; }
  setsid "$@" &
  cpu_pid=$!
  (
    delay_pid=
    trap '[[ -z $delay_pid ]] || kill -TERM "$delay_pid" 2>/dev/null || true' EXIT
    now=$(date +%s)
    delay=$((work_deadline-now)); ((delay > 0)) || delay=0
    sleep "$delay" & delay_pid=$!
    wait "$delay_pid"
    delay_pid=
    # Persistent one-shot closure even if CPU startup blocked before reserve.
    (set -o noclobber; printf '%s\n' '{"schema_version":"T7_batch_numerics_outer_timeout_v1","status":"deadline","scope_closed":true,"automatic_retry_allowed":false,"reservation_state":"charged_or_startup_unreserved_unknown","GPU_release_verified":false,"resource_state":"unverified","reason":"outer_work_deadline"}' > "$marker") || true
    # Negative signal target is exclusively the child-created CPU PG, never
    # this shell's or another task's group. Live setsid must establish this PG.
    if [[ $(ps -o pgid= -p "$cpu_pid" | tr -d ' ') == "$cpu_pid" ]]; then
      kill -TERM -- "-$cpu_pid" 2>/dev/null || true
    fi
    now=$(date +%s)
    delay=$((hard_deadline-now)); ((delay > 0)) || delay=0
    sleep "$delay" & delay_pid=$!
    wait "$delay_pid"
    delay_pid=
    if [[ $(ps -o pgid= -p "$cpu_pid" | tr -d ' ') == "$cpu_pid" ]]; then
      kill -KILL -- "-$cpu_pid" 2>/dev/null || true
    fi
  ) &
  timer_pid=$!
  # A caller signal must not disarm the hard-deadline protection while the
  # supervisor handles bounded GPU cleanup. Interrupted wait is resumed below.
  trap 'if [[ $(ps -o pgid= -p "$cpu_pid" | tr -d " ") == "$cpu_pid" ]]; then kill -TERM -- "-$cpu_pid" 2>/dev/null || true; fi' INT TERM
  set +e
  while true; do
    wait "$cpu_pid"
    result=$?
    kill -0 "$cpu_pid" 2>/dev/null || break
  done
  set -e
  kill -TERM "$timer_pid" 2>/dev/null || true
  wait "$timer_pid" 2>/dev/null || true
  trap - INT TERM
  [[ ! -e $marker ]] || return 124
  return "$result"
}

main() {
if [[ ${1:-} == --runtime-child || ${1:-} == --runtime-discovery ]]; then runtime_child "$@"; fi

DRY=false
for ARG in "$@"; do
  [[ $ARG != --shell-started-epoch && $ARG != --shell-started-epoch=* ]] || exit 2
  if [[ $ARG == --dry-run ]]; then DRY=true; fi
done
if [[ -n ${ORCJAX_NUMERICS_LAUNCHER_PYTHON:-} ]]; then
  [[ $DRY == true ]] || { printf '%s\n' 'Python override requires --dry-run' >&2; exit 2; }
  CPU_PYTHON=$ORCJAX_NUMERICS_LAUNCHER_PYTHON
fi
export PYTHONDONTWRITEBYTECODE=1
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=
export ORCHIDEE_REPO_ROOT=$CODE
export PYTHONPATH="$CODE:$CODE/scripts/hpc"
if [[ $DRY == true ]]; then
  exec "$CPU_PYTHON" -B "$RUNNER" --shell-started-epoch "$SHELL_STARTED_EPOCH" "$@"
fi
OUTPUT=
ARGS=("$@")
for ((I=0; I<${#ARGS[@]}; I++)); do
  if [[ ${ARGS[I]} == --output ]]; then OUTPUT=${ARGS[I+1]:-}; fi
  if [[ ${ARGS[I]} == --output=* ]]; then OUTPUT=${ARGS[I]#--output=}; fi
done
[[ $OUTPUT == "$PROJECT/runtime/outputs/"*.json && $OUTPUT != *'/../'* && $OUTPUT != *'/./'* ]] || exit 2
MARKER=${OUTPUT%.json}.outer-timeout.json
[[ -d $(dirname -- "$MARKER") && ! -e $MARKER ]] || exit 2
START=${SHELL_STARTED_EPOCH%.*}
guarded_cpu_launch "$((START+1010))" "$((START+1190))" "$MARKER" \
  "$CPU_PYTHON" -B "$RUNNER" --shell-started-epoch "$SHELL_STARTED_EPOCH" "$@"
}

if [[ ${BASH_SOURCE[0]} == "$0" ]]; then main "$@"; fi
