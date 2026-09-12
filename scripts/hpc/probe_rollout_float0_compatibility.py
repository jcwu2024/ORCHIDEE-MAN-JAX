"""Tiny exact-runtime reproduction of scalar float0 AOT argument handling."""
import argparse
from pathlib import Path

import numpy as np


def run(output):
    from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend
    from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json

    jax,devices = initialize_gpu_backend()
    import jax.numpy as jnp

    from research.daily_coarse_graining.daily_operator.rollout_loss import make_rollout_loss

    def step(p,s,d,day):
        following = p['gain']*s+day
        return following,d+1,{'scalar_mask':jnp.array(True),'scalar_count':d,
                              'defined':jnp.ones_like(s,dtype=bool)}

    def loss(p,s,obs,target,context):
        return jnp.mean((obs['state']-target)**2)

    arguments = ({'gain':jnp.array(.8)},jnp.array([.5]),jnp.array(0,dtype=jnp.int32),
                 jnp.array([.2,.3]),jnp.zeros((2,1)),{})
    old = make_rollout_loss(jax,jnp,step,loss)
    old._slice = lambda tree,index:jax.tree_util.tree_map(lambda v:v[index],tree)
    old_error = None
    try:
        old.streamed_value_and_grad(*arguments)
    except TypeError as error:
        old_error = str(error)
    fixed = make_rollout_loss(jax,jnp,step,loss)
    reference = jax.jit(jax.value_and_grad(fixed.loss,argnums=(0,1)))(*arguments)
    actual = fixed.streamed_value_and_grad(*arguments)
    passed = all(np.allclose(a,b,atol=1e-10,rtol=1e-10) for a,b in zip(
        jax.tree_util.tree_leaves(actual),jax.tree_util.tree_leaves(reference),strict=True))
    report = {'status':'passed' if passed else 'rejected','jax_version':jax.__version__,
              'device':str(devices[0]),'old_indexing_error':old_error,
              'old_numpy_void_reproduced':old_error is not None and 'numpy.void' in old_error,
              'fixed_streamed_matches_direct':passed,'updates':0}
    _atomic_json(output,report)
    print(report,flush=True)
    if not passed:
        raise RuntimeError('fixed scalar-cotangent reverse differs from direct AD')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args,_ = parser.parse_known_args()
    run(args.output)
