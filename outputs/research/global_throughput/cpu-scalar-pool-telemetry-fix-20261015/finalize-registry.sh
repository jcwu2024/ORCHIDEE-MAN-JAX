#!/bin/bash
set -euo pipefail
PROJECT=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
RT=$PROJECT/runtime
EVIDENCE=$RT/outputs/research/global_throughput/cpu-scalar-pool-telemetry-fix-20261015
PYTHON=$PROJECT/.venvs/orcjax_cpu/bin/python
[[ $(realpath "$EVIDENCE") == "$EVIDENCE" ]]
"$PYTHON" -B - "$RT/deployment/registry.json" "$EVIDENCE" <<'PY'
import datetime, hashlib, json, os, pathlib, subprocess, sys
registry_path, evidence = sys.argv[1:]
base = pathlib.Path(evidence)
before = pathlib.Path(registry_path).read_bytes()
registry = json.loads(before)
initial = json.loads((base / 'registry-before.json').read_bytes())
keys = ['global_cpu_scalar_pool_telemetry_fix_v2_20261015',
        'global_cpu_scalar_pool_telemetry_fix_v3_20261015']
assert {k: v for k, v in registry.items() if k not in keys} == initial, 'unexpected changes to previous records'
failed = json.loads((base / 'target-startup-matrix.json').read_bytes())
target_bytes = (base / 'v3/target-verification.json').read_bytes()
target = json.loads(target_bytes)
assert failed['test_count'] == 14 and failed['failures'] == 3
assert target['startup_matrix_passed'] and target['candidate_identity_matches']
assert target['paid_submitted'] is False and target['physical_lower_compile_run'] is False
for key in keys:
    entry = registry[key]
    assert entry['paid_submission_count'] == 0 and entry['paid_submissions_remaining'] == 0
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=entry['code_dir']).decode().strip() == entry['execution_commit']
    assert subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=entry['code_dir']) == b''
registry[keys[0]].update(status='superseded_after_target_test_assertion_failure',
    free_verification_job='15866780', target_startup_tests={'passed': 11, 'failed': 3},
    failure='Bash 4.2 pipeline exit status differs; all three failures still blocked dispatch',
    superseded_by=target['execution_commit'])
registry[keys[1]].update(status='deployed_free_checks_passed_awaiting_supervisor_review',
    free_verification_job=target['job_id'], target_startup_tests={'passed': 14, 'failed': 0},
    target_verification_sha256=hashlib.sha256(target_bytes).hexdigest(),
    reference_loading_reused_job='15866448', physical_model_executed=False)
temporary = registry_path + '.scalar-pool-telemetry-final.tmp'
with open(temporary, 'x') as stream:
    json.dump(registry, stream, indent=2, sort_keys=True)
    stream.write('\n')
assert pathlib.Path(registry_path).read_bytes() == before, 'concurrent edit; preserve temporary file'
os.rename(temporary, registry_path)
receipt = {'schema': 'scalar_pool_telemetry_fix_registry_finalization_v1',
    'utc': datetime.datetime.utcnow().isoformat() + 'Z',
    'registry_before_sha256': hashlib.sha256(before).hexdigest(),
    'registry_after_sha256': hashlib.sha256(pathlib.Path(registry_path).read_bytes()).hexdigest(),
    'all_previous_records_preserved': True, 'only_modified_keys': keys,
    'entries': {key: registry[key] for key in keys}, 'new_paid_submission_allowance': 0}
(base / 'registry-finalization.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps({'all_previous_records_preserved': True, 'new_paid_submission_allowance': 0,
                  'execution_commit': target['execution_commit']}))
PY
/rmprog/slurm/v22.05.7/bin/sacct -j 15866780,15866807 -P --format=JobID,JobName,Partition,NodeList,AllocCPUS,AllocTRES,ElapsedRaw,State,ExitCode > "$EVIDENCE/free-check-accounting.psv"
cat "$EVIDENCE/free-check-accounting.psv"
