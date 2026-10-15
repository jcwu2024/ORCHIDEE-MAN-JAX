"""Read-only metadata/file identity checks for this terminal delivery, no model."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
PROJECT = Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest_path = HERE / 'delivery-manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    for record in manifest['files']:
        path = HERE / record['name']
        assert path.is_file() and path.stat().st_size == record['size_bytes'], record['name']
        assert sha(path) == record['sha256'], record['name']
    before = json.loads((HERE / 'registry-before.json').read_bytes())
    current_bytes = (PROJECT / 'runtime/deployment/registry.json').read_bytes()
    current = json.loads(current_bytes)
    assert all(current[key] == value for key, value in before.items())
    key = 'global_cpu_scalar_pool_paid_pilot_v3_20261015'
    assert current[key]['state'] == 'closed_failed_prejax_physical_core_admission'
    assert current[key]['paid_sbatch_invocations'] == 1
    assert current[key]['paid_submissions_remaining'] == 0
    assert current_bytes == (HERE / 'registry-after.json').read_bytes()
    code = PROJECT / 'runtime/worktrees/global-throughput-cpu-scalar-pool-v3'
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=code).decode().strip() == 'bae03d9ef78726f00987737cdc58eac705a922ae'
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=code).strip()
    original_log = PROJECT / 'runtime/logs/global_scalar_pool_15869035.txt'
    original_context = PROJECT / 'runtime/outputs/research/global_throughput/server-benchmark/cpu4-15869035/launch-context.txt'
    assert sha(original_log) == sha(HERE / 'job-15869035.raw.log')
    assert sha(original_context) == sha(HERE / 'launch-context.raw.txt')
    original_inventory = json.loads((HERE / 'terminal-inventory.json').read_bytes())
    for record in original_inventory['files']:
        assert sha(HERE / record['name']) == record['sha256']
    receipt = {'schema': 'global_cpu_scalar_pool_v3_delivery_integrity_v1',
        'job_id': '15869035', 'status': 'passed',
        'delivery_manifest_sha256': sha(manifest_path),
        'source_target_byte_match_count': len(manifest['files']),
        'files': manifest['files'],
        'all_49_prior_registry_records_unchanged': len(before) == 49,
        'registry_after_sha256': hashlib.sha256(current_bytes).hexdigest(),
        'paid_submission_count': 1, 'paid_submissions_remaining': 0,
        'immutable_candidate_clean_at_delivery': True,
        'original_job_log_and_context_match_recovered_copies': True,
        'all_21_original_raw_receipt_files_still_match': len(original_inventory['files']) == 21,
        'scientific_arrays_opened': False, 'JAX_imported': False,
        'model_execution': False, 'additional_scheduler_jobs_submitted': False,
        'verification_scope': 'metadata, small text/JSON and source byte identity only',
        'archive_transport_note': 'initial tar emitted directory-mtime warning; all21 archived raw files individually verified against remote SHA; no missing raw bytes'}
    (HERE / 'integrity-verification.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps({key: value for key, value in receipt.items() if key != 'files'}, sort_keys=True))


if __name__ == '__main__':
    main()
