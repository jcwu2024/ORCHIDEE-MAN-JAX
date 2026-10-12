#!/usr/bin/env bash
set -euo pipefail

PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
CODE_DIR=$PROJECT/runtime/worktrees/global-throughput-cpu48-b64-v2
CHECK_DIR=$PROJECT/runtime/outputs/research/global_throughput/cpu48-b64-repair-v2-20261012/target-validation
CASE=${1:-valid-fallback}
case "$CASE" in
  valid-fallback) expected_launcher_status=1; scontrol_status=0; scontrol_stderr="" ;;
  command-failure) expected_launcher_status=2; scontrol_status=1; scontrol_stderr="slurm_load_jobs error: Invalid job id specified" ;;
  missing-fields|duplicate-fields|cpu-mismatch|tres-conflict) expected_launcher_status=2; scontrol_status=0; scontrol_stderr="" ;;
  *) echo "unknown target-validation case: $CASE" >&2; exit 2 ;;
esac
JOB_ID=v2-target-validation-20261012-$CASE
CASE_DIR=$CHECK_DIR/results/$CASE
RUN_ROOT=$PROJECT/runtime/outputs/research/global_throughput/cpu48-b64-execution-v2-20261012/target-validation/launcher-$CASE
FIXTURE=$CHECK_DIR/fixtures/$CASE.stdout
SRUN_LOG=$CASE_DIR/srun-stub-invocations.txt

export SLURM_JOB_ID=$JOB_ID
export SLURM_JOB_PARTITION=cnall
export SLURM_JOB_NODELIST=ibtest01
export SLURM_JOB_NUM_NODES=1
export SLURM_NNODES=1
export SLURM_NTASKS=1
export SLURM_CPUS_PER_TASK=48
unset SLURM_JOB_GPUS ORCHJAX_SLURM_ALLOC_TRES ORCHJAX_SLURM_ALLOC_TRES_SOURCE

export CODE_DIR
export EXPECTED_CODE_COMMIT=b1b5cf0835da298bddb388e61d754feeb6c14cf2
export SLURM_BIN=$CHECK_DIR/stubs
export RUN_ROOT
export EVIDENCE_DIR=$PROJECT/runtime/outputs/research/global_throughput/cpu48-b64-execution-v2-20261012
export ORCHJAX_TEST_SCONTROL_FIXTURE=$FIXTURE
export ORCHJAX_TEST_SCONTROL_EXITCODE=$scontrol_status
export ORCHJAX_TEST_SCONTROL_STDERR=$scontrol_stderr
export ORCHJAX_CONTROLLED_SRUN_LOG=$SRUN_LOG

[[ -f $FIXTURE && ! -e $RUN_ROOT && ! -e $CASE_DIR && ! -e $ORCHJAX_CONTROLLED_SRUN_LOG ]] || {
  echo "target validation outputs must be new" >&2
  exit 2
}
mkdir -p "$CASE_DIR"
export PYTHON=$PROJECT/.venvs/orcjax_cpu/bin/python
export ENTRY=$CODE_DIR/scripts/dev/benchmark_global_server_cpu.py
export COORDINATOR=$CODE_DIR/scripts/hpc/global_cpu48_b64_coordinator.py
export PREPARED_INPUT=$PROJECT/runtime/outputs/research/global_throughput/round4-inputs-remote-canonical/prepared_inputs_remote_64.pkl
export GPU_REFERENCE_DIR=$PROJECT/runtime/outputs/research/global_throughput/gpu-round4/model
export CONFIG=$CODE_DIR/configs/orchidee_man_250919.yaml

set +e
bash "$CODE_DIR/scripts/hpc/slurm_global_server_cpu_48core.sh" >"$CASE_DIR/launcher.stdout" 2>"$CASE_DIR/launcher.stderr"
launcher_status=$?
set -e
printf '%s\n' "$launcher_status" >"$CASE_DIR/launcher.exitcode"
printf 'case=%s launcher_exit_code=%s expected=%s\n' "$CASE" "$launcher_status" "$expected_launcher_status"
[[ $launcher_status -eq $expected_launcher_status ]]
