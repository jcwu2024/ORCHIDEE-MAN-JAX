#!/bin/bash
set -euo pipefail
P=/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
E=$P/runtime/outputs/research/global_throughput/cpu-scalar-pool-pilot-v3-20261015
"$P/.venvs/orcjax_cpu/bin/python" -B - "$E" "$P/runtime/deployment/registry.json" <<'PY'
import datetime, hashlib, json, os, sys
from pathlib import Path
e, registry_path = Path(sys.argv[1]), Path(sys.argv[2])
original = json.loads((e / 'registry-before.json').read_bytes())
before = registry_path.read_bytes()
registry = json.loads(before)
key = 'global_cpu_scalar_pool_paid_pilot_v3_20261015'
assert all(registry[name] == record for name, record in original.items())
assert registry[key]['state'] == 'submitted_once'
assert registry[key]['job_id'] == '15869035'
assert registry[key]['paid_sbatch_invocations'] == 1
analysis = json.loads((e / 'result-analysis.json').read_bytes())
assert analysis['state'] == 'FAILED' and analysis['exit_code'] == '1:0'
entry = registry[key]
entry.update(state='closed_failed_prejax_physical_core_admission',
    terminal_utc=datetime.datetime.utcnow().isoformat() + 'Z',
    terminal_job_state=analysis['state'], terminal_exit_code=analysis['exit_code'],
    actual_node=analysis['node'], actual_allocated_CPUs=analysis['allocated_CPUs'],
    actual_parent_elapsed_seconds=analysis['parent_elapsed_seconds'],
    actual_affinity=analysis['prejax_affinity'],
    actual_unique_socket_core_keys=analysis['observed_unique_socket_core_keys'],
    physical_workers_started=0, JAX_imported=False, physical_model_executed=False,
    scientific_accepted=False, paid_submissions_remaining=0,
    automatic_retry_permitted=False, new_paid_approval_required=True,
    actual_accounting=analysis['accounting'],
    terminal_analysis_sha256=hashlib.sha256((e / 'result-analysis.json').read_bytes()).hexdigest(),
    original_job_log_sha256=hashlib.sha256((e / 'job-15869035.raw.log').read_bytes()).hexdigest(),
    all_prior_registry_records_unchanged=True)
temporary = Path(str(registry_path) + '.paid-v3-final.tmp')
with temporary.open('x') as stream:
    json.dump(registry, stream, indent=2, sort_keys=True); stream.write('\n')
    stream.flush(); os.fsync(stream.fileno())
assert registry_path.read_bytes() == before, 'concurrent registry edit; stop without overwrite'
os.rename(temporary, registry_path)
after = registry_path.read_bytes()
final = json.loads(after)
assert all(final[name] == record for name, record in original.items())
(e / 'registry-after.json').write_bytes(after)
receipt = {'schema': 'scalar_pool_v3_terminal_registry_finalization_v1',
    'registry_key': key, 'prior_record_count': len(original),
    'all_prior_records_unchanged': True, 'records_added_by_this_pilot': 1,
    'registry_before_submission_sha256': hashlib.sha256((e / 'registry-before.json').read_bytes()).hexdigest(),
    'registry_before_finalization_sha256': hashlib.sha256(before).hexdigest(),
    'registry_after_sha256': hashlib.sha256(after).hexdigest(),
    'terminal_entry': entry}
(e / 'registry-finalization.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
print(json.dumps(receipt, sort_keys=True))
PY
