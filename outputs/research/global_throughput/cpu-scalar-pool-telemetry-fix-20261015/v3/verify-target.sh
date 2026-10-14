#!/bin/bash
# Only run inside the authorized free one-CPU/300-second test allocation.
set -euo pipefail
PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
EVIDENCE=$PROJECT/runtime/outputs/research/global_throughput/cpu-scalar-pool-telemetry-fix-20261015/v3
source <(sed '/^submission=/,$d' "$EVIDENCE/PROPOSED_SUBMISSION.sh")
PYTHON=$PROJECT/.venvs/orcjax_cpu/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export ORCHIDEE_REPO_ROOT=$CODE_DIR ORCHIDEE_DATA_ROOT=$PROJECT/runtime/data
export TMPDIR=$EVIDENCE/scratch
[[ $SLURM_CPUS_PER_TASK == 1 && $SLURM_JOB_PARTITION == cntest ]]
[[ $(cd "$CODE_DIR" && git rev-parse HEAD) == "$EXPECTED_CODE_COMMIT" ]]
[[ -z $(cd "$CODE_DIR" && git status --porcelain --untracked-files=all) ]]
mkdir -p "$TMPDIR"
cd "$CODE_DIR"
date -u +%FT%TZ
hostname
uptime
taskset -pc $$
bash --version
/rmprog/slurm/v22.05.7/bin/scontrol show job --details "$SLURM_JOB_ID" > "$EVIDENCE/free-allocation.txt"
"$PYTHON" -B -c 'import jax_orchidee; print("actual_module_import=" + str(jax_orchidee.__file__))'
"$PYTHON" -B tests/unit/test_global_server_cpu_startup.py --scratch-root "$TMPDIR" --report-json "$EVIDENCE/target-startup-matrix.json"
bash -n scripts/hpc/slurm_global_server_cpu_4core.sh
bash -n "$EVIDENCE/PROPOSED_SUBMISSION.sh"
bash "$EVIDENCE/PROPOSED_SUBMISSION.sh" --dry-run > "$EVIDENCE/submission-target-dry-run.txt"
# The real free allocation is one CPU. This planned pool4 expansion is print-only.
SLURM_CPUS_PER_TASK=4 bash scripts/hpc/slurm_global_server_cpu_4core.sh --dry-run > "$EVIDENCE/launcher-target-dry-run.txt"
"$PYTHON" -B - "$EVIDENCE" "$CODE_DIR" <<'PY'
import datetime, hashlib, json, os, shlex, socket, subprocess, sys
from argparse import Namespace
from pathlib import Path
from scripts.dev.global_cpu_candidate_identity import candidate_identity

evidence, code = map(Path, sys.argv[1:])
pin = json.loads((evidence / 'execution-code-pin.json').read_bytes())
admission_bytes = (evidence / 'scalar-admission.json').read_bytes()
assert hashlib.sha256(admission_bytes).hexdigest() == pin['scalar_admission']['sha256']
admission = json.loads(admission_bytes)
identity = candidate_identity(Namespace(entry=code / 'scripts/dev/benchmark_global_server_cpu.py',
    prepared_input=Path(os.environ['PREPARED_INPUT']), gpu_reference_dir=Path(os.environ['GPU_REFERENCE_DIR'])), code)
runtime_job = identity['job_id']
identity['job_id'] = None
assert identity == admission['candidate_identity'], (identity, admission['candidate_identity'])
lines = (evidence / 'launcher-target-dry-run.txt').read_text().splitlines()
expanded = next(line[len('command='):] for line in lines if line.startswith('command='))
command = shlex.split(expanded)
env_index = next(i for i, arg in enumerate(command) if arg.endswith('/global_cpu_child_environment.py'))
parser_command = command[env_index - 1:] + ['--dry-run']
result = subprocess.run(parser_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
(evidence / 'parser-target-dry-run.stdout').write_bytes(result.stdout)
(evidence / 'parser-target-dry-run.stderr').write_bytes(result.stderr)
assert result.returncode == 0, result.stderr.decode(errors='replace')
parser = json.loads(result.stdout)
assert parser['model_execution'] is False and parser['JAX_imported'] is False
assert parser['mode'] == 'pool' and parser['CPU_budget'] == 4
matrix = json.loads((evidence / 'target-startup-matrix.json').read_bytes())
assert matrix['passed'] and matrix['failures'] == 0 and matrix['errors'] == 0
assert matrix['JAX_imported'] is False and matrix['model_execution'] is False
assert matrix['launcher_sha256'] == pin['changed_blob_sha256']['scripts/hpc/slurm_global_server_cpu_4core.sh']
assert matrix['test_sha256'] == pin['changed_blob_sha256']['tests/unit/test_global_server_cpu_startup.py']
assert 'jax' not in sys.modules
assert len(os.sched_getaffinity(0)) == 1
receipt = {
    'schema': 'global_scalar_pool_telemetry_fix_target_check_v1',
    'utc': datetime.datetime.utcnow().isoformat() + 'Z', 'node': socket.gethostname(),
    'job_id': runtime_job, 'partition': os.environ['SLURM_JOB_PARTITION'], 'allocated_CPUs': 1,
    'affinity': sorted(os.sched_getaffinity(0)), 'python': sys.executable,
    'bash_version': subprocess.check_output(['bash', '--version']).decode().splitlines()[0],
    'execution_commit': pin['execution_code_commit'], 'candidate_identity_matches': True,
    'candidate_identity': identity, 'admission_sha256': pin['scalar_admission']['sha256'],
    'startup_test_count': matrix['test_count'], 'startup_matrix_passed': True,
    'real_launcher_and_parser_dry_run': 'passed; planned four CPU metadata, print/parse only',
    'expanded_command': command, 'normalized_parser': parser,
    'reference_loading_evidence_reused_job': '15866448',
    'reference_arrays_deserialized_or_audited': False,
    'JAX_imported': False, 'physical_runner_created': False, 'physical_lower_compile_run': False,
    'paid_submitted': False, 'new_paid_submission_allowance': 0,
}
(evidence / 'target-verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({key: receipt[key] for key in ('job_id', 'node', 'allocated_CPUs', 'execution_commit',
    'candidate_identity_matches', 'startup_test_count', 'physical_lower_compile_run', 'paid_submitted')}))
PY
[[ -z $(git status --porcelain --untracked-files=all) ]]
date -u +%FT%TZ
