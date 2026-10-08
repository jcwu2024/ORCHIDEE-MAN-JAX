"""Bounded canonical CPU lower-only capture; never compile or execute model."""
import os
os.environ.update(JAX_ENABLE_X64='true', JAX_PLATFORMS='cpu', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', ORCHJAX_DISABLE_COMPILATION_CACHE='1')
import sys, json, time, gc, hashlib
from pathlib import Path
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
root=Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-sweetspot-v3')
os.environ['ORCHIDEE_REPO_ROOT']=str(root)
os.environ['ORCHIDEE_DATA_ROOT']='/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/data'
sys.path.insert(0,str(root))
import jax
from jax._src import pjit, core
from scripts.dev import benchmark_global_scale as scale
from scripts.dev import benchmark_global_gpu_capacity as capacity
if os.environ.get('REPAIR_MASK') == '1':
    import numpy as np
    from functools import lru_cache
    from jax_orchidee.sechiba import diffuco
    @lru_cache(maxsize=32)
    def _diffuco_pft_column_mask(nvm: int, pft_index: int):
        mask = np.arange(int(nvm)) == int(pft_index)
        mask.setflags(write=False)
        return mask
    diffuco._diffuco_pft_column_mask = _diffuco_pft_column_mask
jax.config.update('jax_enable_x64',True)
jax.config.update('jax_enable_compilation_cache',False)
def emit(**data): print(json.dumps(data,default=str),flush=True)
def sig(value):
    return {'shape':list(value.shape), 'dtype':str(value.dtype), 'weak_type':bool(getattr(value,'weak_type',False))}
def snapshot(lowered):
    args=lowered._lowering.compile_args
    info=args.get('all_args_info')
    return {'lowered_id':id(lowered),'lowering_id':id(lowered._lowering), 'all_args_info_id':id(info),
            'public':[sig(v) for v in jax.tree_util.tree_leaves(lowered.in_avals)],
            'internal':None if info is None else [sig(v) for v in info.in_avals],
            'kept':sorted(args['kept_var_idx']), 'global':[sig(v) for v in args['global_in_avals']]}
original=pjit._create_pjit_jaxpr
def capture(*args,**kwargs):
    result=original(*args,**kwargs)
    closed,consts,_,_=result
    if len(closed.in_avals)>=500 or consts:
        emit(phase='pjit_jaxpr',inputs=len(closed.in_avals),closed_constants=len(closed.consts),lifted_constants=len(consts),
             lifted=[{'class':type(c).__name__,'tracer':isinstance(c,core.Tracer),**sig(c)} for c in consts],
             debug=str(closed.jaxpr.debug_info))
    return result
pjit._create_pjit_jaxpr=capture
capture.cache_clear=original.cache_clear
emit(phase='admission',host=os.uname().nodename,jax=jax.__version__,affinity=sorted(os.sched_getaffinity(0)),model_compile=False,model_execution=False,
     sources={m:hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest() for m,mod in [('pjit',pjit),('core',core)]})
avail=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
assert avail>=64*1024**3
lanes,second=scale.read_prepared('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/round4-inputs-remote-canonical/prepared_inputs_remote_64.pkl',root/'configs/orchidee_man_250919.yaml')
expanded,others,_=capacity.capacity_layout(lanes,second)
previous=None
for size in [256,512]:
    scale.base.driver._COMPILED_SECHIBA_SCAN_CACHE.clear()
    jax.clear_caches(); gc.collect()
    arguments=jax.device_put(scale.base.batch.stack_lanes(expanded[:size]))
    pairs,tree=jax.tree_util.tree_flatten_with_path((tuple(arguments),{}))
    emit(phase='arguments',size=size,paths=[{'path':str(p),**sig(v)} for p,v in pairs],tree=str(tree))
    fn=scale.base.batch.make_batch_transition(expanded[:size])
    started=time.perf_counter()
    traced=jax.jit(fn).trace(*arguments)
    emit(phase='trace',size=size,fn_id=id(fn),traced_id=id(traced),jaxpr_id=id(traced.jaxpr),invars=len(traced.jaxpr.jaxpr.invars),
         constvars=len(traced.jaxpr.jaxpr.constvars),constants=[{'class':type(c).__name__,'tracer':isinstance(c,core.Tracer),**sig(c)} for c in traced.jaxpr.consts],
         attrs=sorted(traced.__dict__))
    lowered=traced.lower()
    emit(phase='lowered',size=size,seconds=time.perf_counter()-started,metadata=snapshot(lowered))
    if previous is not None: emit(phase='previous_after_next',size=256,metadata=snapshot(previous))
    previous=lowered
emit(phase='complete',model_compile=False,model_execution=False)
pjit._create_pjit_jaxpr=original
