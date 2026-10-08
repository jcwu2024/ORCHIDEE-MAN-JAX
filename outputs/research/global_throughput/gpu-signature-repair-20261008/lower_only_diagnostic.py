import os, sys, json, time
from pathlib import Path
os.environ.update(JAX_ENABLE_X64='true', JAX_PLATFORMS='cpu', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', ORCHJAX_DISABLE_COMPILATION_CACHE='1')
root=Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/worktrees/global-throughput-gpu-sweetspot-v1')
os.environ['ORCHIDEE_REPO_ROOT']=str(root)
os.environ['ORCHIDEE_DATA_ROOT']='/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/data'
sys.path.insert(0,str(root))
import jax, numpy as np
from scripts.dev import benchmark_global_scale as scale
from scripts.dev import benchmark_global_gpu_capacity as capacity
jax.config.update('jax_enable_x64',True)
jax.config.update('jax_enable_compilation_cache',False)
def emit(**data): print(json.dumps(data,default=str),flush=True)
avail=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
emit(phase='admission',host=os.uname().nodename,available_bytes=avail,jax=jax.__version__,model_compile=False,model_execution=False)
assert avail>=64*1024**3
lanes, second=scale.read_prepared('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/research/global_throughput/round4-inputs-remote-canonical/prepared_inputs_remote_64.pkl', root/'configs/orchidee_man_250919.yaml')
expanded,others,_=capacity.capacity_layout(lanes,second)
arguments=jax.device_put(scale.base.batch.stack_lanes(expanded[:256]))
leaves,tree=jax.tree_util.tree_flatten((tuple(arguments),{}))
emit(phase='arguments',leaves=len(leaves),groups=[len(jax.tree_util.tree_leaves(v)) for v in arguments],tree=str(tree))
scale.base.driver._COMPILED_SECHIBA_SCAN_CACHE.clear()
fn=scale.base.batch.make_batch_transition(expanded[:256])
started=time.perf_counter()
lowered=jax.jit(fn).lower(*arguments)
args=lowered._lowering.compile_args
info=args.get('all_args_info')
emit(phase='lowered',seconds=time.perf_counter()-started,public_in_leaves=len(jax.tree_util.tree_leaves(lowered.in_avals)),public_tree_equal=lowered.in_tree==tree,global_input_avals=len(args['global_in_avals']),all_args_info_leaves=None if info is None else len(info.in_avals),kept_count=len(args['kept_var_idx']),kept_max=max(args['kept_var_idx']),compile_keys=sorted(args))
# No .compile() or transition invocation. Preserve exact leaf signatures.
emit(phase='signatures',caller=[(list(v.shape),str(v.dtype)) for v in leaves],lowered_all=None if info is None else [(list(v.shape),str(v.dtype)) for v in info.in_avals])
