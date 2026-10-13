set -euo pipefail
STAGE=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/staging/cpu48-b64-v4-20261014/environment/exact-pin
PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
SLURM_BIN=/rmprog/slurm/v22.05.7/bin
LOGDIR=$STAGE/checks/job-$SLURM_JOB_ID
mkdir -p "$LOGDIR"
CODE_DIR=$PROJECT/runtime/worktrees/global-throughput-cpu48-b64-v4
PYTHON=$PROJECT/.venvs/orcjax_cpu/bin/python
ADMISSION=$STAGE/scalar-admission.json
OLD_SCALAR=$PROJECT/runtime/outputs/research/global_throughput/cpu48-b64-execution-v3-20261012/15820467/scalar
PIN=$STAGE/execution-code-pin.json
{
  printf 'job_id=%s\npartition=%s\nhost=%s\ncpus_per_task=%s\ntasks=%s\ncpus_on_node=%s\njob_cpus_per_node=%s\n' \
    "$SLURM_JOB_ID" "$SLURM_JOB_PARTITION" "$SLURM_CPUS_PER_TASK" \
    "$SLURM_NTASKS" "$SLURM_CPUS_ON_NODE" "$SLURM_JOB_CPUS_PER_NODE"
  printf 'started='; date -Is
  printf 'affinity='; taskset -pc $$
  grep '^Cpus_allowed_list:' /proc/self/status
  uptime
  "$SLURM_BIN/scontrol" show job -o "$SLURM_JOB_ID"
} >"$LOGDIR/allocation.txt" 2>&1
[[ "$SLURM_JOB_PARTITION" == cntest && "$SLURM_CPUS_PER_TASK" == 1 && "$(hostname)" == ibtest03 ]]
[[ -x "$PYTHON" && -f "$ADMISSION" && -f "$PIN" && -f "$STAGE/PROPOSED_SUBMISSION.sh" ]]
[[ ! -e "$CODE_DIR" ]]
[[ ! -e "$PROJECT/runtime/outputs/research/global_throughput/cpu48-b64-execution-v4-20261014/$SLURM_JOB_ID" ]]
if bash -n "$STAGE/PROPOSED_SUBMISSION.sh" >"$LOGDIR/proposal-bashn.stdout.txt" 2>"$LOGDIR/proposal-bashn.stderr.txt"; then echo 0 >"$LOGDIR/proposal-bashn.exitcode.txt"; else echo 1 >"$LOGDIR/proposal-bashn.exitcode.txt"; exit 1; fi
if bash -n "$STAGE/code/scripts/hpc/slurm_global_server_cpu_48core.sh" >"$LOGDIR/launcher-bashn.stdout.txt" 2>"$LOGDIR/launcher-bashn.stderr.txt"; then echo 0 >"$LOGDIR/launcher-bashn.exitcode.txt"; else echo 1 >"$LOGDIR/launcher-bashn.exitcode.txt"; exit 1; fi
for prefix in SBATCH_ ORCHJAX_; do
  while IFS= read -r variable; do unset "$variable"; done < <(compgen -v "$prefix")
done
unset SLURM_BIN CODE_DIR PYTHON ENTRY COORDINATOR PREPARED_INPUT GPU_REFERENCE_DIR CONFIG
unset CPU_REFERENCE_DIR CPU_REFERENCE_ADMISSION_RECORD EXPECTED_CPU_REFERENCE_ADMISSION_SHA256
unset RUN_ROOT EXPECTED_CODE_COMMIT EVIDENCE_DIR ORCHIDEE_REPO_ROOT ORCHIDEE_DATA_ROOT
[[ "$SLURM_CPUS_PER_TASK" == 1 ]]
unset SLURM_CPUS_PER_TASK
export CODE_DIR=$PROJECT/runtime/worktrees/global-throughput-cpu48-b64-v4
export PYTHON=$PROJECT/.venvs/orcjax_cpu/bin/python
export EXPECTED_CODE_COMMIT=60dccbddde10be4295aab7e7fe55fa7864d3c3f3
export CPU_REFERENCE_DIR=$OLD_SCALAR
export CPU_REFERENCE_ADMISSION_RECORD=$ADMISSION
export EXPECTED_CPU_REFERENCE_ADMISSION_SHA256=f71eb5db4e638615ef5e06be5783d193bbfc1bd7563dd30c2424e04b40c14a8d
export PYTHONDONTWRITEBYTECODE=1
if bash "$STAGE/code/scripts/hpc/slurm_global_server_cpu_48core.sh" --dry-run >"$LOGDIR/launcher-dry-run.stdout.txt" 2>"$LOGDIR/launcher-dry-run.stderr.txt"; then echo 0 >"$LOGDIR/launcher-dry-run.exitcode.txt"; else echo 1 >"$LOGDIR/launcher-dry-run.exitcode.txt"; exit 1; fi
"$PYTHON" -B - "$LOGDIR/launcher-dry-run.stdout.txt" "$STAGE" "$LOGDIR" "$CODE_DIR" "$PYTHON" "$SLURM_JOB_ID" <<'PYTHON_CHECK'
import json, os, shlex, subprocess, sys
from pathlib import Path
out, stage, logdir, code_dir, python, job_id = map(str, sys.argv[1:])
stage, logdir = Path(stage), Path(logdir)
lines = Path(out).read_text().splitlines()
line = next(line for line in lines if line.startswith("coordinator="))
original = shlex.split(line[len("coordinator="):])
assert original[0] == python, original
assert original[1] == code_dir + "/scripts/hpc/global_cpu48_b64_coordinator.py", original[1]
def option(name):
    return original[original.index(name) + 1]
assert option("--entry") == code_dir + "/scripts/dev/benchmark_global_server_cpu.py"
assert option("--output-dir").endswith("/" + job_id), option("--output-dir")
assert option("--code-dir") == code_dir
transformed = list(original)
transformed[1] = str(stage / "code/scripts/hpc/global_cpu48_b64_coordinator.py")
transformed[transformed.index("--entry") + 1] = str(stage / "code/scripts/dev/benchmark_global_server_cpu.py")
assert transformed[transformed.index("--code-dir") + 1] == code_dir
receipt = {"original_coordinator_argv": original, "transformed_coordinator_argv": transformed,
           "only_changed": ["coordinator script path", "--entry value"], "job_id": job_id}
(logdir / "coordinator-argv.json").write_text(json.dumps(receipt, indent=2) + "\n")
env = os.environ.copy()
env["PYTHONDONTWRITEBYTECODE"] = "1"
with (logdir / "coordinator-dry-run.stdout.txt").open("w") as stdout, \
     (logdir / "coordinator-dry-run.stderr.txt").open("w") as stderr:
    result = subprocess.run(transformed, cwd=stage / "code", env=env, stdout=stdout, stderr=stderr, check=False)
(logdir / "coordinator-dry-run.exitcode.txt").write_text(str(result.returncode) + "\n")
if result.returncode:
    raise SystemExit(result.returncode)
report = json.loads((logdir / "coordinator-dry-run.stdout.txt").read_text())
assert report["status"] == "dry_run"
assert report["model_execution"] is False and report["JAX_imported"] is False
(logdir / "coordinator-dry-run-assertions.txt").write_text(
    "status=" + report["status"] + "\nmodel_execution=false\nJAX_imported=false\n")
PYTHON_CHECK
"$PYTHON" -B - "$STAGE" "$ADMISSION" "$OLD_SCALAR" "$PIN" "$LOGDIR" <<'PYTHON_ADMISSION'
import json, sys
from pathlib import Path
stage, admission, old_scalar, pin_path, logdir = map(Path, sys.argv[1:])
sys.path.insert(0, str(stage / "code/scripts/dev"))
from global_server_reference import validate_cpu_reference_admission
pin = json.loads(pin_path.read_text())
source = pin["external_scalar_reference"]["candidate_identity"]
expected_sha = pin["external_scalar_reference"]["admission_record"]["sha256"]
assert source["job_id"] is None
result = validate_cpu_reference_admission(admission, expected_sha, old_scalar, expected_source=source)
(logdir / "scalar-admission-validation.stdout.txt").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
print("admission_validation=passed")
PYTHON_ADMISSION
echo 0 >"$LOGDIR/scalar-admission-validation.exitcode.txt"
[[ ! -e "$CODE_DIR" ]]
[[ ! -e "$PROJECT/runtime/outputs/research/global_throughput/cpu48-b64-execution-v4-20261014/$SLURM_JOB_ID" ]]
printf 'finished=' >"$LOGDIR/finished.txt"; date -Is >>"$LOGDIR/finished.txt"
printf 'checks_passed job_id=%s host=%s\n' "$SLURM_JOB_ID" "$(hostname)"
