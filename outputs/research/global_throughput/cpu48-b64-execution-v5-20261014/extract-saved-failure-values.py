"""Read just the three reported failing leaf arrays, using NumPy; no model/JIT."""
import hashlib,json,pickle,re,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[3]))
def load(name,expected):
    raw=(ROOT/name).read_bytes()
    assert hashlib.sha256(raw).hexdigest()==expected
    return pickle.loads(raw)
actual=load('raw_B64_batch_before_parity.pkl','a135f9436afa868176bf8cc76f28134218e510c65b75909075ba24a8964810ce')
scalar=load('source-scalar.pkl','1cddeaf80c2fc36f173835a5ff1bd4d8d520b144d4c96289ef6fe86dde898dfa')
report=json.loads((ROOT/'model-report.json').read_bytes())
gate=next(g for g in report['gates'] if g['name']=='B64_batch_cross_mode_server_scalar')
assert actual['identity']['lanes']==scalar['identity']['lanes']
def descend(tree,indices):
    semantic=[]
    for i in indices:
        semantic.append(tree._fields[i] if hasattr(tree,'_fields') else str(i))
        tree=tree[i]
    return np.asarray(tree),semantic
details=[]
for item in gate['failures']:
    indices=list(map(int,re.findall(r'\[(\d+)\]',item['path'])))
    a,semantic=descend(actual['outputs'],indices)
    r,_=descend(scalar['value']['outputs'],indices)
    assert a.shape==r.shape and a.dtype==r.dtype
    assert np.isfinite(a).all() and np.isfinite(r).all()
    delta=np.abs(a-r); bound=1e-8+1e-10*np.abs(r); ratio=delta/bound
    bad=np.argwhere(delta>bound)
    coord=np.unravel_index(np.argmax(ratio),a.shape)
    assert np.isclose(float(ratio[coord]),item['max_bound_ratio'],rtol=1e-14,atol=0)
    lane=int(coord[0]); lane_identity=actual['identity']['lanes'][lane]
    details.append({'path':item['path'],'semantic_path':semantic,'array_shape':list(a.shape),
        'dtype':str(a.dtype),'block_days':[[2,8],[9,15]][indices[0]],
        'failing_element_count':len(bad),'failing_lane_indices':sorted(set(int(x[0]) for x in bad)),
        'worst_bound_ratio_coordinate':list(map(int,coord)),'worst_lane_index':lane,
        'worst_lane_identity':lane_identity,'candidate_value':float(a[coord]),
        'scalar_reference_value':float(r[coord]),'absolute_difference':float(delta[coord]),
        'allowed_absolute_bound':float(bound[coord]),'bound_ratio':float(ratio[coord]),
        'max_abs_over_entire_leaf':float(delta.max())})
result={'schema':'global_cpu48_v5_saved_failure_values_v1','status':'three_reported_cpu_internal_failure_leaves_recomputed',
    'raw_candidate_sha256':'a135f9436afa868176bf8cc76f28134218e510c65b75909075ba24a8964810ce',
    'scalar_source_sha256':'1cddeaf80c2fc36f173835a5ff1bd4d8d520b144d4c96289ef6fe86dde898dfa',
    'gate_name':gate['name'],'atol':1e-8,'rtol':1e-10,'details':details,
    'local_numpy':np.__version__,'model_execution':False,'JAX_imported':'jax' in sys.modules,
    'scope':'NumPy extraction from existing saved arrays only; no repair, new model call or tolerance change; not an accepted sealed batch stage or full scientific re-audit',
    'GPU_failure_causality':'not inferred from separate GPU tolerance failures'}
(ROOT/'saved-failure-values.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(result,indent=2))
