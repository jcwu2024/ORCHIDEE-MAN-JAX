"""Check sealed identities and retained receipts without physics/reference decoding."""
from pathlib import Path
import copy
import hashlib
import json
import re
import subprocess

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[3]
OLD = BASE.parent / 'cpu-scalar-pool-preparation-20261014'
PILOT = BASE.parent / 'cpu-scalar-pool-pilot-20261014/terminal-evidence'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def main():
    source_admission = json.loads((OLD / 'scalar-admission.json').read_bytes())
    checks = {}
    for name, directory in [('superseded_v2', BASE), ('final_v3', BASE / 'v3')]:
        pin = json.loads((directory / 'execution-code-pin.json').read_bytes())
        commit = pin['execution_code_commit']
        changed = git('diff-tree', '--no-commit-id', '--name-only', '-r', pin['base_execution_commit'], commit).decode().splitlines()
        assert sorted(changed) == sorted(pin['execution_changed_paths'])
        assert len(changed) == 2
        for scope, oid in pin['unchanged_tree_oids'].items():
            assert git('rev-parse', commit + ':' + scope).decode().strip() == oid
            assert git('rev-parse', pin['base_execution_commit'] + ':' + scope).decode().strip() == oid
        for path, expected in pin['changed_blob_sha256'].items():
            assert sha(git('show', commit + ':' + path)) == expected
        admission_bytes = (directory / 'scalar-admission.json').read_bytes()
        assert sha(admission_bytes) == pin['scalar_admission']['sha256']
        admission = json.loads(admission_bytes)
        restored = copy.deepcopy(admission)
        for key in ('execution_commit', 'execution_code_dir'):
            restored['candidate_identity'][key] = source_admission['candidate_identity'][key]
        assert restored == source_admission
        assert sha((directory / 'PROPOSED_SUBMISSION.sh').read_bytes()) == pin['proposed_submission']['sha256']
        assert sha((directory / pin['bundle']['name']).read_bytes()) == pin['bundle']['sha256']
        assert pin['pilot']['authorized'] is False and pin['paid_submitted'] is False
        assert pin['pilot']['allocated_CPUs'] == pin['pilot']['workers'] == 4
        assert pin['pilot']['wall_seconds'] == 5400
        deployment = json.loads((directory / 'deployment-receipt.json').read_bytes())
        assert deployment['pin_sha256'] == sha((directory / 'execution-code-pin.json').read_bytes())
        assert deployment['execution_commit'] == commit
        checks[name] = {'commit': commit, 'changed_paths': changed, 'admission_only_two_identity_fields_changed': True}
    target_bytes = (BASE / 'v3/target-verification.json').read_bytes()
    target = json.loads(target_bytes)
    matrix = json.loads((BASE / 'v3/target-startup-matrix.json').read_bytes())
    local = json.loads((BASE / 'local-startup-matrix-v3.json').read_bytes())
    assert matrix['passed'] and matrix['test_count'] == 14 and local['passed']
    assert matrix['launcher_sha256'] == local['launcher_sha256']
    assert matrix['test_sha256'] == local['test_sha256']
    assert target['candidate_identity_matches'] and target['startup_matrix_passed']
    assert target['allocated_CPUs'] == 1 and target['affinity'] == [0]
    assert not target['JAX_imported'] and not target['physical_lower_compile_run'] and not target['paid_submitted']
    registry = json.loads((BASE / 'registry-finalization.json').read_bytes())
    assert registry['all_previous_records_preserved'] and registry['new_paid_submission_allowance'] == 0
    entry = registry['entries']['global_cpu_scalar_pool_telemetry_fix_v3_20261015']
    assert entry['target_verification_sha256'] == sha(target_bytes)
    assert entry['paid_submission_count'] == entry['paid_submissions_remaining'] == 0
    preserved = {'terminal-result.json': '6e5ee70e0827a0d01636b21691f1c12a7ca4a6abe0f992fcf152ef4fe619e7c0',
                 'terminal-evidence-manifest.json': '7ef597517c4ad651ae802e9556952a79cb7507d2ba4a6fc2c7d665fe18060fa4'}
    for name, expected in preserved.items():
        assert sha((PILOT / name).read_bytes()) == expected
    docs = ['README.md', 'SERVER_BENCHMARK.md', 'SUPERVISOR_HANDOFF.md', 'NEW_PROJECT_PROMPT.md',
            'CPU_SCALAR_POOL_TELEMETRY_FIX.md']
    for name in docs:
        document = ROOT / 'docs/research/global_throughput' / name
        for link in re.findall(r'\]\(([^)]+)\)', document.read_text(encoding='utf-8')):
            if '://' not in link and not link.startswith('#'):
                assert (document.parent / link.split('#', 1)[0]).exists(), (name, link)
    record = {'schema': 'scalar_pool_telemetry_fix_integrity_v1', 'checks_passed': True,
        'candidates': checks, 'target_job': target['job_id'], 'target_test_count': matrix['test_count'],
        'unchanged_old_failure_receipts': preserved, 'admission_target_match': True,
        'previous_registry_records_preserved': True, 'entry_document_links_exist': True,
        'reference_arrays_decoded_or_audited': False, 'physical_model_executed': False,
        'paid_submitted': False, 'new_paid_submission_allowance': 0}
    (BASE / 'integrity-verification.json').write_bytes((json.dumps(record, indent=2) + '\n').encode())
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
