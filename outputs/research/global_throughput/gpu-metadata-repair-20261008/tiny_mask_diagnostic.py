import os, sys, time, json
os.environ.update(JAX_ENABLE_X64='true', JAX_PLATFORMS='cpu', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
sys.path.insert(0,'/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-sweetspot-v3')
import jax, numpy as np
from functools import lru_cache
from jax_orchidee.sechiba import diffuco as module
started=time.perf_counter()
def emit(**d): print(json.dumps(d),flush=True)
old=module._diffuco_pft_column_mask
def repaired(nvm,pft_index):
    mask=np.arange(int(nvm))==int(pft_index)
    mask.setflags(write=False)
    return mask
for policy in ['original','immutable_host_mask']:
    module._diffuco_pft_column_mask=old if policy=='original' else lru_cache(maxsize=32)(repaired)
    module._diffuco_pft_column_mask.cache_clear()
    before=None
    for size in [2,3]:
        jax.clear_caches()
        backgrounds=[np.arange(size*14,dtype=np.float64).reshape(size,14)+k for k in range(5)]
        columns=[np.full((size,1),100+k,dtype=np.float64) for k in range(5)]
        args=backgrounds+columns
        fn=lambda *a:module._diffuco_insert_pft_output_columns(*a,pft_index=13)
        lowered=jax.jit(fn).lower(*args)
        info=lowered._lowering.compile_args['all_args_info']
        mask=module._diffuco_pft_column_mask(14,13)
        emit(policy=policy,size=size,public=len(jax.tree_util.tree_leaves(lowered.in_avals)),internal=len(info.in_avals),
             cached_class=type(mask).__name__,cached_dtype=str(mask.dtype),cached_shape=list(mask.shape),
             tracer_origin=str(getattr(mask,'_line_info',None)))
        if policy=='original' and size==3: continue
        actual=lowered.compile()(*args)
        for a,b,c in zip(actual,backgrounds,columns):
            expected=b.copy(); expected[:,13:14]=c
            np.testing.assert_array_equal(a,expected)
            assert a.dtype==expected.dtype
        emit(policy=policy,size=size,native_output_equality=True,outputs=5)
emit(phase='complete',jax=jax.__version__,seconds=time.perf_counter()-started)
