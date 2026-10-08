import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
report = {}
for label, filename in [('before', 'canonical-sequential-lower.jsonl'), ('after', 'canonical-fixed-sequential-lower.jsonl')]:
    path = root / filename
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines()]
    lowered = [r for r in rows if r['phase'] == 'lowered']
    retained = next(r for r in rows if r['phase'] == 'previous_after_next')
    assert lowered[0]['metadata'] == retained['metadata']
    report[label] = {'file': filename, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                     'completed': rows[-1]['phase'] == 'complete',
                     'earlier_stage_metadata_unchanged': True,
                     'stages': [{'size': r['size'], 'lower_seconds': r['seconds'],
                                 'public_leaves': len(r['metadata']['public']),
                                 'internal_leaves': len(r['metadata']['internal']),
                                 'kept': len(r['metadata']['kept'])} for r in lowered],
                     'lifted_constants': [{'inputs':r['inputs'], 'count':r['lifted_constants'],
                                           'tracers':[v for v in r['lifted'] if v['tracer']],
                                           'source':r['debug']} for r in rows
                                          if r['phase']=='pjit_jaxpr' and r['lifted_constants']]}
assert report['before']['stages'][1]['internal_leaves'] == 673
assert all(r['public_leaves']==r['internal_leaves']==536 for r in report['after']['stages'])
report.update(real_model_compile=False, real_model_execution=False,
              canonical_jax='0.4.38', cpu_affinity_max=4,
              first_outer_wall_conservative_seconds=211,
              first_wall_method='JSONL creation09:43:34UTC to lastwrite09:47:04UTC, rounded up with one-second margin; no stopwatch was recorded',
              tiny_outer_wall_seconds=float((root/'tiny-wall-seconds.txt').read_text()),
              fixed_outer_wall_seconds=float((root/'fixed-wall-seconds.txt').read_text()),
              first_hook_cleanup_error='Capture wrapper lacked cache_clear; interpreter exit emitted ignored atexit AttributeError after complete. Process exit0; fixed capture preserves cache_clear/restores hook.',
              limitation='Post-fix real lowering used exact in-memory replacement of only the host mask helper in immutable v3 checkout; no files there changed. GPU compilation, invocation, full scientific gates and timing remain pending.')
report['total_outer_wall_conservative_seconds'] = report['first_outer_wall_conservative_seconds'] + report['tiny_outer_wall_seconds'] + report['fixed_outer_wall_seconds']
assert report['total_outer_wall_conservative_seconds'] <= 600
(root/'causal-evidence-summary.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'stages': {k:report[k]['stages'] for k in ['before','after']},'total_outer_wall':report['total_outer_wall_conservative_seconds']}))
