#!/bin/bash
set -euo pipefail
PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
E=$PROJECT/runtime/outputs/research/global_throughput/cpu-scalar-pool-pilot-v3-20261015
V=$PROJECT/runtime/outputs/research/global_throughput/cpu-scalar-pool-telemetry-fix-20261015/v3
C=$PROJECT/runtime/worktrees/global-throughput-cpu-scalar-pool-v3
R=$PROJECT/runtime/deployment/registry.json
[[ $(realpath -m "$E") == "$PROJECT/runtime/outputs/research/global_throughput/cpu-scalar-pool-pilot-v3-20261015" ]]
"$PROJECT/.venvs/orcjax_cpu/bin/python" -B - "$E" "$V" "$C" "$R" <<'PY'
import datetime, hashlib, json, os, re, subprocess, sys
e, v, c, registry_path = sys.argv[1:]
pin = 'bae03d9ef78726f00987737cdc58eac705a922ae'
proposal = v + '/PROPOSED_SUBMISSION.sh'
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=c).decode().strip() == pin
assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=c).strip()
assert hashlib.sha256(open(proposal, 'rb').read()).hexdigest() == 'db9a5a7bb4a382f373e7cf7cb4d340d185d32293a305a086c585dd6db57070f9'
assert hashlib.sha256(open(v + '/scalar-admission.json', 'rb').read()).hexdigest() == '259936bb09f5d0006895b1f5ae17fa298c0baada07edcb23b2355ed892ef796a'
before = open(registry_path, 'rb').read()
registry = json.loads(before)
key = 'global_cpu_scalar_pool_paid_pilot_v3_20261015'
assert key not in registry, 'do not repeat or overwrite paid pilot record'
attempt = {'schema': 'scalar_pool_paid_submission_attempt_v3',
    'utc': datetime.datetime.utcnow().isoformat() + 'Z',
    'authorization': '2026-10-15 human approval relayed by root: exact v3 single paid pilot',
    'command': ['bash', proposal], 'paid_sbatch_invocations': 1,
    'maximum_allowed_invocations': 1, 'automatic_retry_permitted': False,
    'execution_commit': pin, 'execution_ref': 'codex/global-cpu-scalar-pool-v3',
    'code_dir': c, 'evidence_dir': e,
    'proposal_sha256': 'db9a5a7bb4a382f373e7cf7cb4d340d185d32293a305a086c585dd6db57070f9',
    'admission_sha256': '259936bb09f5d0006895b1f5ae17fa298c0baada07edcb23b2355ed892ef796a',
    'resources': {'partition': 'cnall', 'nodes': 1, 'tasks': 1, 'CPUs': 4, 'GPUs': 0,
        'wall_seconds': 5400, 'no_requeue': True, 'exclusive': False,
        'nodelist': None, 'memory_reservation': None, 'workers': 4,
        'startup_concurrency': 1, 'startup_timeout_seconds_per_worker': 1200,
        'command_timeout_seconds': 300, 'RSS_abort_GiB': 64,
        'host_MemAvailable_guard_GiB': 64, 'RLIMIT_AS': None},
    'worst_case_allocated_CPU_hours': 6, 'worst_case_CNY': [0.30, 0.42],
    'rates_verified': False, 'paid_submissions_remaining': 0,
    'state': 'submission_attempt_recorded_before_sbatch'}
attempt_path = e + '/paid-submission-attempt.json'
with open(attempt_path, 'x') as stream:
    json.dump(attempt, stream, indent=2, sort_keys=True)
    stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
open(e + '/registry-before.json', 'xb').write(before)
def update_registry(expected):
    current = open(registry_path, 'rb').read()
    assert current == expected, 'concurrent registry edit; no overwrite'
    records = json.loads(current)
    records[key] = dict(attempt)
    temporary = registry_path + '.paid-v3.tmp'
    with open(temporary, 'x') as stream:
        json.dump(records, stream, indent=2, sort_keys=True); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    assert open(registry_path, 'rb').read() == current
    os.rename(temporary, registry_path)
    return open(registry_path, 'rb').read()
registered = update_registry(before)
result = subprocess.run(['bash', proposal], stdin=subprocess.DEVNULL,
    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
open(e + '/paid-sbatch.stdout', 'xb').write(result.stdout)
open(e + '/paid-sbatch.stderr', 'xb').write(result.stderr)
attempt.update(returncode=result.returncode, stdout=result.stdout.decode(errors='replace'),
    stderr=result.stderr.decode(errors='replace'))
if result.returncode == 0 and re.fullmatch(r'[0-9]+(?:;[^\s]+)?\s*', attempt['stdout']):
    attempt['job_id'] = attempt['stdout'].strip().split(';')[0]
    attempt['state'] = 'submitted_once'
    open(e + '/paid-job-id.txt', 'x').write(attempt['job_id'] + '\n')
else:
    attempt['state'] = 'failed_or_unknown_do_not_resubmit'
with open(attempt_path, 'w') as stream:
    json.dump(attempt, stream, indent=2, sort_keys=True); stream.write('\n')
update_registry(registered)
print(json.dumps(attempt, sort_keys=True), flush=True)
if attempt['state'] != 'submitted_once': raise SystemExit(1)
PY
JOB=$(cat "$E/paid-job-id.txt")
/rmprog/slurm/v22.05.7/bin/scontrol show job --details "$JOB" | tee "$E/paid-initial-scontrol.txt"
