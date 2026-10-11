"""Administrative registration only; no JAX/model/data loading."""
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

project = Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX')
runtime = project / 'runtime'
registry = runtime / 'deployment/registry.json'
pin = '3b4ae35e695f85480ecb72461e4ba93f4534bbd8'
code = runtime / 'worktrees/global-throughput-gpu-lifecycle-v1'
assert code.resolve() == code and code.is_relative_to(runtime)
original = registry.read_bytes()
assert hashlib.sha256(original).hexdigest() == 'be35fd8949f8b1df0581424641b70398e1ff0a382baa5b9e9ee1dcfd966ecb97'
record = json.loads(original)
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=code, text=True).strip() == pin
assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=code).strip()
assert not (code / '.git/objects/info/alternates').exists()
assert all(item['path'] != str(code) for item in record['retained_checkouts'])
now = datetime.now(timezone.utc).isoformat()
receipt = {
    'schema': 'global_lifecycle_exact_deployment_v1', 'utc': now,
    'execution_sha': pin, 'code_dir': str(code), 'branch': 'research/global-throughput',
    'bundle': str(runtime / 'transfer/gpu-lifecycle-20261011-3b4ae35/global-throughput-3b4ae35.bundle'),
    'bundle_sha256': 'c41277078f7c05efac832a7f25136e5cc64b037f8594774c64e2af86fbfa4ed5',
    'bundle_bytes': 4861356, 'registry_previous_sha256': hashlib.sha256(original).hexdigest(),
    'checkout_porcelain': '', 'source_overlay': None, 'alternate_dependencies': [],
    'old_checkouts_preserved': True, 'main_daily_refs_preserved': True,
    'authorization': 'User explicitly approved bounded lifecycle two-arm plan via supervisor 01a128d2-3ec2-77a1-8d51-1665c220d2ca on 2026-10-11.',
    'allocation': {'partition': 'gnall', 'nodes': 1, 'tasks': 1, 'V100_32GB_count': 1,
                   'Slurm_CPUs': 4, 'wall_seconds': 7200, 'requeue': False, 'exclusive': False,
                   'primary_worst_case_CNY': 4.80, 'conservative_worst_case_CNY': 4.96},
    'submitted': False,
}
transfer = runtime / 'transfer/gpu-lifecycle-20261011-3b4ae35'
(transfer / 'registry-before.json').write_bytes(original)
record['retained_checkouts'].append({'path': str(code), 'sha': pin, 'branch': 'DETACHED',
                                    'porcelain': '', 'alternate_dependencies': [], 'role': 'approved B1024 lifecycle diagnostic'})
for route in record['branch_routes']:
    if route['branch'] == 'research/global-throughput':
        route.update(mirror_sha=pin, execution_checkout=str(code), execution_sha=pin,
                     role='approved immutable B1024 lifecycle execution; historical checkouts retained')
record.update(updated_utc=now, latest_global_lifecycle_deployment=receipt,
              branch_sync_status='Global exact 3b4ae35 imported by non-forced verified bundle; main/daily refs preserved.')
temporary = registry.with_name('registry.gpu-lifecycle.tmp')
assert not temporary.exists()
temporary.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
assert registry.read_bytes() == original
os.replace(temporary, registry)
destination = runtime / 'deployment/gpu-lifecycle-20261011-deployment.json'
assert not destination.exists()
destination.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
print(json.dumps(receipt))
