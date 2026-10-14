"""Summarize preserved terminal evidence; never alter execution source or seals."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REMOTE = '/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/cpu48-b64-execution-v4-20261014'

def load(path):
    return json.loads((ROOT/path).read_bytes())

def save(path, value):
    with (ROOT/path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')

def digest(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()

collection = load('collector-result.json')
report = load('paid-run/batch/report.json')
orchestration = load('paid-run/orchestration.json')
receipt = load('terminal/collector-execution-receipt.json')
preflight = load('preflight/15853600/production-preflight.json')
assert collection['status'] == 'incomplete_run_failure_preserved'
assert report['status'] == 'failed' and report['phases'] == [] and report['timing'] == {}
assert report['gates'] == []
assert receipt['job_id'] == '15853855' and receipt['cpus_per_task'] == '1'
assert digest('collector-result.json') == 'd7faeb702d568c897aefdc0f3ca2b560684f326edee2e90a270fc681cd7db0ca'
for path, expected in {
    'paid-run/orchestration.json':'2e0a400f41aa598f07dcefdb43443c0f1be0fb841351a104c98b1ccc4422af9c',
    'paid-run/batch.log':'a3e67e45f944a7c0f0c92d73fe5f843de210db09eb7da0b8fc069b4157eb5fae',
    'paid-run/batch/report.json':'da686739ea670f3c7dfb1067540099a284f8f6f7314029341ef0f07bcc4a1169',
    'terminal/terminal-parent.sacct.raw':'ad106ba30c62b51f499f72c70f790b9056c7c01ca97a927c9e013fd0a2dc9285',
    'terminal/terminal-full.sacct.raw':'0cf8b8f1823c1877554ee40efb80c8ea9bd1d57123d0bce31d3d5dac7fe4af96',
    'terminal/parent-accounting.json':'182d2d9bd2755be940f135c45a95866576b92d1dd19137bb013f363259b772f7',
}.items():
    assert digest(path) == expected, path

resources = report['server_resources']
actual = resources['candidate_provenance']
admitted = resources['external_scalar_reference_admission']['candidate_identity']
differences = {key:{'admission':admitted.get(key),'actual_resource_candidate':actual.get(key)}
               for key in sorted(set(actual)|set(admitted)) if actual.get(key) != admitted.get(key)}
assert differences == {'job_id':{'admission':None,'actual_resource_candidate':'15853683'}}
save('identity-and-child-receipts.json', {
    'schema':'global_cpu48_v4_preserved_identity_and_child_receipts_v1',
    'source_report_sha256':digest('paid-run/batch/report.json'),
    'actual_candidate_provenance':actual,
    'saved_admission_candidate_identity':admitted,
    'saved_summary_field_differences':differences,
    'difference_limitation':'These saved resource summaries do not serialize the full current-execution identity compared by failing load_reference; the job_id difference is not a diagnosed root cause.',
    'child_log_receipts':[json.loads(line) for line in (ROOT/'paid-run/batch.log').read_text().splitlines()
                          if line.startswith('{"schema": "global_cpu_child_prejax_v1"')],
    'final_resource_receipt':resources['actual_child_prejax_receipt'],
    'scientific_acceptance':False,
})

parent = collection['whole_parent_allocation_cost']
old = collection['reused_scalar_source_parent_cost']
text = f'''# CPU48 B64 v4 current result

The sole approved paid attempt, job **15853683**, ended **FAILED, exit 1:0** on
`ibc11b09n36` at 2026-10-14 09:56:55 Asia/Shanghai, after 50 seconds.
No complete B64 measurement or CPU/GPU throughput ratio is available.
Scientific acceptance remains **false**. This attempt is closed: no source
repair, second paid submission, extension, cancellation, or scalar regeneration
was performed.

## Immutable deployment and preflight boundary

The fresh detached checkout is
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-cpu48-b64-v4`
at `60dccbddde10be4295aab7e7fe55fa7864d3c3f3`; the 104-file physical tree
`75d8b1e9ccc9ef55d25344c66ad88d3742b3b3ba` and science SHA
`11b6db89c467c25bee347528838844c5ca7e1c36dd1adb5bbdc082b5513fc2f5`
are unchanged. Bundle/pin/original proposal/admission hashes were verified;
the bundle prerequisite and all 15 actual harness blobs passed. The checkout
has no Git alternates and was still clean after the paid run and collector.

Free production-path preflight **15853600**, `cntest`, `ibtest01`, 1 CPU,
passed actual `candidate_identity` and `validate_cpu_reference_admission`
identity/seal checks with the real preflight job ID. The unchanged proposal
and launcher/coordinator/runner dry-run passed using production paths.
No JAX/model was imported/executed in that preflight. This evidence did not
exercise the real benchmark runner's post-import `load_reference` comparison.

The paid child subsequently logged `exec_boundary_ready` and
`resource_admission_passed` with the exact thread policy, 48 distinct physical
cores and parent `cpu=48` allocation before importing JAX. The real runner
then recorded JAX/jaxlib 0.4.38, one CPU device and x64 enabled. Its first
substantive fault was:

```text
ValueError: FULL64 CPU scalar admission candidate provenance differs from current execution identity
benchmark_global_server_cpu.py:411 -> global_server_reference.py:283
```

`batch/report.json` preserves the complete traceback. Its `phases`, `timing`
and `gates` are empty; no lower/compile/first-call/warm measurements or saved
candidate validation arrays/checkpoints were produced. The child ended after
43.821602583 s; coordinator scope was 43.900519609 s, both shorter than the
50-second scheduler parent allocation.

The saved `server_resources.candidate_provenance` and admission
`candidate_identity` summaries differ only in `job_id` (`15853683` versus
`null`). The same kind of job-ID difference passed the preflight and collector
admission validators. The failing `load_reference` full expected/current
identity pair was not independently serialized; these summaries do not
diagnose the cause of that failure. No broader diagnosis or repair was run.

## Independent failure collection and cost

Pinned `scripts/dev/collect_global_cpu48_b64.py`, SHA
`efcb32efa754bbdbb75936f358dc6cb5a286e6742948096bda3e3ba22d27a9e3`,
ran once using canonical CPU Python in free job **15853855**, `cntest`,
`ibtest01`, 1 CPU, affinity `[0]`. It exited successfully and produced
`incomplete_run_failure_preserved`, with claim scope
`failure accounting only; no throughput claim`. It revalidated the scalar
admission and bound the new and old parent rows separately; it did not run
the physical model or certify candidate array gates.

| Parent scope | Job | Seconds | Allocated core-hours | CNY at 0.05 | CNY at 0.07 |
| --- | --- | ---: | ---: | ---: | ---: |
| New paid allocation, charged once | 15853683 | 50 | {parent['allocated_parent_CPU_request_core_hours']:.9f} | {parent['primary_cost_cny_estimate']:.9f} | {parent['conservative_cost_cny_estimate']:.9f} |
| Prior scalar source, separate and charged once | 15820467 | 2197 | {old['allocated_CPU_request_core_hours']:.9f} | {old['primary_cost_cny_estimate']:.9f} | {old['conservative_cost_cny_estimate']:.9f} |

These are unverified billing scenarios, not invoices. The collector's generic
new-parent scope string mentions a scalar step; this batch-only run contains
no new scalar child. Whole-parent cost is `48 × 50 / 3600` core-hours, not the
shorter coordinator interval. Prior scalar reuse adds no new allocation
charge. Scheduler MaxRSS is blank/zero; `/usr/bin/time -v` reports 1,044,316 KiB
and runner Linux `ru_maxrss` 1,069,379,584 B for the failed process.
These failure-resource observations are not warm throughput telemetry.

## Preserved scientific boundaries and evidence

The unchanged workload contract is 64 true unique points, Days 2–8 and 9–15,
896 point-days, all 760 native leaves/masks/dtypes and the original tolerances
and gates. Candidate numerical/permutation/replacement/reset/nonfinite gates
were not reached. The reused scalar's prior three finite cross-GPU failing
points/four comparisons and sealed-proof-only reset limitation remain.

Existing GPU B64 comparator job 15579869 retains core median
5.632416466251016 s (159.07914575720022 point-days/s) and wrapper median
6.222332641482353 s (143.99744462818447 point-days/s). With no new CPU samples,
no matching-boundary ratio or warm cost comparison can be computed. GPU whole
parent cost spans other work and must not be divided by 896. B1024 remains a
different-workload reference. See `GPU_COMPARATOR_EVIDENCE.md`.

Evidence is locally under this directory and remotely at `{REMOTE}`:

- `preflight/15853600/production-preflight.json` and original preflight logs/argv.
- `paid-run/orchestration.json`, `paid-run/batch.log`, `paid-run/batch/report.json`
  mirror remote `15853683/`; originals retain stderr and complete failure details.
- `identity-and-child-receipts.json` is a derived compact extraction with source hash.
- `terminal/submission.*.raw`, `terminal/slurm-parent.log`, parent/step sacct rows,
  exact parent accounting, clean-checkout and collector execution receipts.
- `collector-result.json`, SHA `{digest('collector-result.json')}`.
- `operations/` preserves exact SSH/SCP argv, scripts, raw stdout/stderr and receipts.
- `runtime-evidence-manifest.json` maps local compact evidence hashes to remote originals.

The only original `sbatch` call returned `Submitted batch job 15853683`;
the one-attempt marker and raw receipt remain. All execution source/pins,
old data/checkouts and completed scalar reference outputs remain unchanged.
'''
with (ROOT/'CURRENT_RESULT.md').open('x', encoding='utf-8', newline='\n') as f:
    f.write(text)

registry = load('registry-deployed.json')
entry = registry['latest_global_cpu48_b64_v4_deployment']
entry.update({'submitted':True, 'sbatch_invocation_count':1, 'job_id':'15853683',
    'terminal_state':'FAILED','terminal_exit_code':'1:0','terminal_node':'ibc11b09n36',
    'terminal_elapsed_seconds':50,'closed_attempt':True,'repaired':False,'resubmitted':False,
    'collector_job_id':'15853855','collector_status':collection['status'],
    'collector_result_path':REMOTE+'/collector-result.json','collector_result_sha256':digest('collector-result.json'),
    'terminal_accounting_path':REMOTE+'/terminal/parent-accounting.json',
    'terminal_accounting_sha256':digest('terminal/parent-accounting.json'),
    'first_fault':report['message'],
    'actual_child_resource_admission_passed':resources['actual_child_prejax_receipt']['allocation_admitted'],
    'whole_parent_allocation_cost':parent,'reused_scalar_source_parent_cost':old,
    'current_result_path':REMOTE+'/CURRENT_RESULT.md',
    'current_result_sha256':digest('CURRENT_RESULT.md'),
    'postrun_checkout_state':'detached_clean','scientific_acceptance':False,
    'registry_before_terminal_sha256':digest('registry-deployed.json'),
    'terminal_update_utc':datetime.now(timezone.utc).isoformat()})
save('registry-terminal.json', registry)

paths = ['CURRENT_RESULT.md','identity-and-child-receipts.json','collector-result.json',
         'registry-deployed.json','registry-terminal.json']
paths += [p.relative_to(ROOT).as_posix() for p in (ROOT/'paid-run').rglob('*') if p.is_file()]
paths += [p.relative_to(ROOT).as_posix() for p in (ROOT/'terminal').rglob('*') if p.is_file()]
paths += [p.relative_to(ROOT).as_posix() for p in (ROOT/'preflight/15853600').rglob('*') if p.is_file()]
records=[]
for path in sorted(paths):
    remote_path=path
    if path.startswith('paid-run/'): remote_path='15853683/'+path[len('paid-run/'):]
    if path=='registry-terminal.json': remote_path='/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/deployment/registry.json'
    elif path=='registry-deployed.json': remote_path='/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/deployment/registry.before-cpu48-v4-terminal.json'
    else: remote_path=REMOTE+'/'+remote_path
    records.append({'local_path':path,'remote_path':remote_path,'sha256':digest(path),'bytes':(ROOT/path).stat().st_size})
save('runtime-evidence-manifest.json', {'schema':'global_cpu48_v4_terminal_evidence_v1',
    'job_id':'15853683','terminal_status':'FAILED','scientific_acceptance':False,'records':records,
    'scope':'byte hashes bind collected evidence; no model or independent candidate-array success claim',
    'derived_local_records':['CURRENT_RESULT.md','identity-and-child-receipts.json','registry-terminal.json'],
    'transport_operation_receipts':'operations/'})
print(json.dumps({'summary':str(ROOT/'CURRENT_RESULT.md'),'registry_terminal_sha256':digest('registry-terminal.json'),
                  'manifest_sha256':digest('runtime-evidence-manifest.json'),'record_count':len(records)}))
