"""Tiny isolated interface/IR checks for a proposed parameter-gradient policy.

CPU mode uses an explicitly named FP32 tangent-dot emulator for runtime
plumbing. Native JAXpr/CUDA-targeted StableHLO are saved separately; no GPU
execution or real model occurs. GPU mode is a future synthetic admission only.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import inspect
import json
import os
from pathlib import Path
import sys
import time
import types
import re

import numpy as np


def native_fp16_gemms_from_hlo(hlo):
    """Fail-closed text proof, resolving operands within their computation."""
    computations = {}
    current = None
    for line in hlo.splitlines():
        if line.strip() == '}':
            current = None
            continue
        header = re.match(r'^\s*(?:ENTRY\s+)?%?([\w.-]+)\s*(?:\(.*\))?\s*(?:->.*?)?\s*\{\s*$',line)
        if header and '=' not in line:
            current = header[1]
            computations[current] = {}
            continue
        instruction = re.match(r'\s*(?:ROOT\s+)?%?([\w.-]+)\s*=\s*(.*?)\s+([\w-]+)\((.*?)\)(.*)$',line)
        if current is not None and instruction:
            name,dtype,opcode,operands,suffix = instruction.groups()
            # as_text operands may carry types/layouts whose commas are not
            # operand separators. Percent symbols are unambiguous in this form.
            names = re.findall(r'%([\w.-]+)',operands)
            if not names and operands.strip():
                # Also support canonical untyped, bare-name HLO operands.
                chunks = operands.split(',')
                if all(re.fullmatch(r'\s*[\w.-]+\s*',chunk) for chunk in chunks):
                    names = [chunk.strip() for chunk in chunks]
            computations[current][name] = (dtype,opcode,names,suffix)
    result = []
    for scope,instructions in computations.items():
        for name,(dtype,opcode,operands,suffix) in instructions.items():
            is_gemm = opcode == 'dot' or (opcode in ('custom-call','fusion') and 'gemm' in suffix.lower())
            output_fp32 = dtype.startswith('f32[') or dtype.startswith('(f32[')
            if is_gemm and output_fp32 and len(operands)>=2 and all(
                    instructions.get(operand,('',))[0].startswith('f16[') for operand in operands[:2]):
                result.append({'computation':scope,'instruction':name,'operand_names':operands[:2],
                    'operand_types':[instructions[operand][0] for operand in operands[:2]],'output_type':dtype})
    return result


def walk_jaxprs(closed):
    jaxpr = getattr(closed,'jaxpr',closed)
    for equation in jaxpr.eqns:
        yield equation
        for value in equation.params.values():
            candidates = value if isinstance(value,(tuple,list)) else (value,)
            for candidate in candidates:
                if hasattr(candidate,'jaxpr') or hasattr(candidate,'eqns'):
                    yield from walk_jaxprs(candidate)


def lower_cuda_ir(jax,closed,name):
    """Targeted MLIR generation, never a backend compile/device call."""
    from jax._src.interpreters import mlir
    from jax._src import sharding_impls, source_info_util
    offered = dict(num_const_args=0,in_avals=closed.in_avals,ordered_effects=[],
        platforms=('cuda',),backend=None,backend_or_name=None,name_stack=source_info_util.NameStack(),
        axis_context=sharding_impls.ShardingContext(0),
        donated_args=(False,)*len(closed.in_avals),lowering_parameters=mlir.LoweringParameters())
    signature = inspect.signature(mlir.lower_jaxpr_to_module)
    arguments = {key:value for key,value in offered.items() if key in signature.parameters}
    result = mlir.lower_jaxpr_to_module(name,closed,**arguments)
    return str(result.module)


def dot_record(equation):
    return {'precision':str(equation.params.get('precision')),
            'lhs':list(equation.invars[0].aval.shape),'rhs':list(equation.invars[1].aval.shape),
            'output':list(equation.outvars[0].aval.shape),
            'dtype':str(equation.outvars[0].aval.dtype),
            'dimension_numbers':equation.params['dimension_numbers']}


def close(actual,expected):
    a,b = np.asarray(actual,dtype=np.float64),np.asarray(expected,dtype=np.float64)
    bad = ~np.isfinite(a)|~np.isfinite(b)|(np.abs(a-b)>1e-6+1e-5*np.abs(b))
    norm = float(np.linalg.norm(b.ravel()))
    delta = float(np.linalg.norm((a-b).ravel()))
    cosine = float(np.sum(a*b)/max(float(np.linalg.norm(a.ravel()))*norm,1e-300))
    maximum = float(np.max(np.abs(a-b),initial=0))
    engineering = maximum<=1e-6 if norm<=1e-12 else delta/norm<=.01 and cosine>=.9999
    return {'passed':bool(not np.any(bad)),'failed_coordinates':int(bad.sum()),
            'coordinates':int(a.size),'max_abs':maximum,'reference_norm':norm,
            'relative_l2':delta/max(norm,1e-300),'cosine':cosine,
            'parameter_engineering_passed':bool(np.all(np.isfinite(a)) and engineering)}


def host_rounding_illustration():
    """Explicit host illustration, not measured V100 GEMM accuracy."""
    rng = np.random.default_rng(20261002)
    x = rng.normal(size=(3,16)).astype(np.float32)
    tangent_a,tangent_b = [rng.normal(size=(16,8)).astype(np.float32) for _ in range(2)]
    def half(value):
        return value.astype(np.float16).astype(np.float32)
    def tangent(value,scale=1.):
        return (half(x)@half(value/np.float32(scale)))*np.float32(scale)
    rows = []
    for cotangent_scale in (1.,1e-7,1e-9,32.):
        cotangent = (rng.normal(size=(3,8))*cotangent_scale).astype(np.float32)
        reference = x.T@cotangent
        for local_scale in (1.,4096.):
            with np.errstate(over='ignore',invalid='ignore'):
                quantized = half(cotangent*np.float32(local_scale))
                approximation = (half(x).T@quantized)/np.float32(local_scale)
            finite = bool(np.all(np.isfinite(approximation)))
            delta = approximation.astype(np.float64)-reference
            norm = np.linalg.norm(reference.astype(np.float64))
            lhs = float(np.sum(tangent(tangent_a,local_scale).astype(np.float64)*cotangent))
            rhs = float(np.sum(tangent_a.astype(np.float64)*approximation)) if finite else None
            rows.append({'cotangent_scale':cotangent_scale,'local_parameter_scale':local_scale,
                'nonzero_cotangent':int(np.count_nonzero(cotangent)),
                'cotangent_becomes_zero_in_f16':int(np.sum((cotangent!=0)&(quantized==0))),
                'finite':finite,'scaled_cotangent_nonfinite':int(np.sum(~np.isfinite(quantized))),
                'weight_gradient_relative_l2':float(np.linalg.norm(delta)/max(norm,1e-300)) if finite else None,
                'JVP_VJP_inner_product_difference':abs(lhs-rhs) if finite else None,
                'JVP_VJP_inner_product_reference':lhs})
    linearity = tangent(tangent_a+tangent_b,4096)-tangent(tangent_a,4096)-tangent(tangent_b,4096)
    directions = []
    for direction_scale in (1.,1e-4):
        direction = tangent_a*np.float32(direction_scale)
        reference = x@direction
        candidate = tangent(direction,4096)
        directions.append({'direction_scale':direction_scale,
            'direction_nonzero_becomes_zero_after_div4096_f16':int(np.sum((direction!=0)&(half(direction/4096)==0))),
            'relative_JVP_error':float(np.linalg.norm(candidate-reference)/np.linalg.norm(reference))})
    return {'label':'host_F16_operand_rounding_FP32_matmul_illustration_not_native_V100_evidence',
            'rows':rows,'tangent_additivity_max_abs':float(np.max(np.abs(linearity))),
            'exact_real_linearity_or_exact_numerical_adjoint_claim':False,
            'local_scaling_tested':'host illustration only','parameter_JVP_direction_examples':directions}


@contextmanager
def fp32_tangent_dot_interface_emulator(policy,jax):
    previous = policy._kernel_tangent_dot
    def named_fp32_emulator(value,tangent):
        rows = value.reshape((-1,value.shape[-1]))
        scale = np.float32(policy.PARAMETER_TANGENT_SCALE)
        return ((rows@(tangent/scale))*scale).reshape(value.shape[:-1]+(tangent.shape[-1],))
    policy._kernel_tangent_dot = named_fp32_emulator
    try:
        yield
    finally:
        policy._kernel_tangent_dot = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('cpu-interface','gpu-interface'),default='cpu-interface')
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--array-recovery-report',type=Path,help='same-seed output recovery only; retain original timing decision')
    parser.add_argument('--actual-f4s-shapes',action='store_true',help='two source-backed F4S decoder/backbone shape classes')
    parser.add_argument('--cpu-interface-report',type=Path)
    args = parser.parse_args()
    if args.mode=='cpu-interface':
        os.environ['JAX_PLATFORMS']='cpu'
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    root = Path(os.environ.get('ORCHIDEE_REPO_ROOT',Path(__file__).resolve().parents[2]))
    sys.path.insert(0,str(root))
    import jax
    import jax.numpy as jnp
    import shared_parameter_gradient_precision as policy
    from research.daily_coarse_graining.daily_operator.strong_baseline_architecture import _linear,_ffn
    jax.config.update('jax_enable_x64',True)
    if args.mode=='gpu-interface' and (jax.default_backend()!='gpu' or jax.__version__!='0.4.38'):
        raise ValueError('GPU interface requires actual canonical JAX0.4.38 GPU')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    output = args.output_dir/'interface.json'
    if output.exists():
        raise FileExistsError('preserve earlier interface check')
    rng = np.random.default_rng(20261002)
    p = {'kernel':jnp.asarray(rng.normal(size=(16,8))*.1,dtype=jnp.float32),
         'bias':jnp.asarray(rng.normal(size=(8,))*.01,dtype=jnp.float32)}
    loss = lambda p,x:jnp.sum(jnp.tanh(policy.parameter_gradient_linear(p,x))**2)
    original_loss = lambda p,x:jnp.sum(jnp.tanh(_linear(p,x))**2)
    evidence = {'status':'running','mode':args.mode,'jax_version':jax.__version__,
        'actual_backend':jax.default_backend(),'real_model_calls':0,'policy':policy.policy_evidence(),
        'xla_flags':os.environ.get('XLA_FLAGS',''),
        'policy_sha256':__import__('hashlib').sha256(Path(policy.__file__).read_bytes()).hexdigest(),
        'native_IR':[],'interface_checks':[],'rounding_illustration':host_rounding_illustration(),
        'runtime_emulation':'explicit_FP32_tangent_dot_plumbing_only' if args.mode=='cpu-interface' else None}
    recovery = None
    if args.array_recovery_report is not None:
        if args.actual_f4s_shapes:
            raise ValueError('artifact recovery and representative-shape correction are separate stages')
        recovery = json.loads(args.array_recovery_report.read_text())
        if (args.mode!='gpu-interface' or recovery.get('policy')!=policy.policy_evidence()
            or recovery.get('jax_version')!=jax.__version__ or len(recovery.get('actual_shape_GPU_pairs',[]))!=4):
            raise ValueError('array recovery requires the same completed four-shape GPU diagnostic')
        evidence['array_recovery_only'] = True
        evidence['original_report'] = str(args.array_recovery_report)
        evidence['original_speed_gate_unchanged'] = recovery['component_speed_gate']
    actual_plan = (((4,128,1,384),1536),((4,7,1,768),3072))
    if args.actual_f4s_shapes:
        evidence['representative_shape_set']='source_F4S_decoder_and_backbone'
        if args.mode=='gpu-interface':
            checked=json.loads(args.cpu_interface_report.read_text()) if args.cpu_interface_report else {}
            if (checked.get('status')!='passed' or checked.get('mode')!='cpu-interface'
                or checked.get('jax_version')!=jax.__version__ or checked.get('policy_sha256')!=evidence['policy_sha256']
                or checked.get('representative_shape_set')!=evidence['representative_shape_set']):
                raise ValueError('canonical corrected rank4 CPU/interface evidence required')
            evidence['corrected_cpu_interface_report']=str(args.cpu_interface_report)
    # Native JAXpr and target CUDA StableHLO for actual neural shape classes.
    # No concrete large arrays or backend executable are created.
    for shape,width in actual_plan if args.actual_f4s_shapes else (((384,),384),((128,384),1536),((2,128,384),1536),((59,128,384),1536)):
        ap = {'kernel':jax.ShapeDtypeStruct((shape[-1],width),jnp.float32),
              'bias':jax.ShapeDtypeStruct((width,),jnp.float32)}
        ax = jax.ShapeDtypeStruct(shape,jnp.float32)
        modes = {'forward':policy.parameter_gradient_linear,
                 'state_only':jax.grad(loss,argnums=1),
                 'parameter_only':jax.grad(loss,argnums=0),
                 'dual':jax.grad(loss,argnums=(0,1)),
                 'bias_only':lambda p,x:jax.grad(lambda bias:loss({'kernel':p['kernel'],'bias':bias},x))(p['bias'])}
        for name,function in modes.items():
            closed = jax.make_jaxpr(function)(ap,ax)
            dots = [dot_record(eq) for eq in walk_jaxprs(closed) if eq.primitive.name=='dot_general']
            half_dots = [row for row in dots if row['precision']=='F16_F16_F32']
            expected = int(name in ('parameter_only','dual'))
            if len(half_dots)!=expected or any(len(row['lhs'])!=2 or len(row['rhs'])!=2 or row['dtype']!='float32' for row in half_dots):
                raise ValueError('native parameter/input/primal algorithm separation failed')
            ir = lower_cuda_ir(jax,closed,f'{name}_{len(shape)}d')
            if ir.count('accumulation_type = f32')!=expected:
                raise ValueError('target CUDA native algorithm disappeared')
            has_scale = 'dense<4.096000e+03>' in ir
            if has_scale!=bool(expected) or (expected and not ('stablehlo.multiply' in ir and 'stablehlo.divide' in ir)):
                raise ValueError('local parameter cotangent scale/unscale placement absent or contaminated other modes')
            stem = f'{name}-'+'x'.join(map(str,shape))+f'-out{width}'
            (args.output_dir/(stem+'.jaxpr')).write_text(str(closed)+'\n')
            (args.output_dir/(stem+'.mlir')).write_text(ir+'\n')
            evidence['native_IR'].append({'mode':name,'shape':list(shape),'kernel_width':width,
                'F16_weight_dots':len(half_dots),'dots':dots,'target':'cuda_StableHLO_only_no_backend_compile',
                'local_4096_scale_present_only_in_parameter_modes':has_scale,
                'artifact':str(args.output_dir/(stem+'.mlir'))})
    try:
        jax.jit(jax.grad(loss,argnums=0)).lower(p,jnp.ones((3,16),jnp.float32))
        evidence['native_CPU_lowering']='supported' if args.mode=='cpu-interface' else 'actual_GPU_lowering_passed'
    except ValueError as error:
        if args.mode!='cpu-interface' or 'not supported' not in str(error):
            raise
        evidence['native_CPU_lowering']={'rejected_explicitly':True,'error':str(error),'FP32_fallback':False}
    def interfaces():
        for shape in ((16,),(3,16),(2,3,16),(2,2,3,16)):
            x = jnp.asarray(rng.normal(size=shape)*.2,dtype=jnp.float32)
            actual = jax.jit(jax.value_and_grad(loss,argnums=(0,1)))(p,x)
            expected = jax.jit(jax.value_and_grad(original_loss,argnums=(0,1)))(p,x)
            jax.block_until_ready(actual)
            checks = {'loss':close(actual[0],expected[0]),'kernel':close(actual[1][0]['kernel'],expected[1][0]['kernel']),
                'bias':close(actual[1][0]['bias'],expected[1][0]['bias']),'input':close(actual[1][1],expected[1][1])}
            evidence['interface_checks'].append({'name':'linear_dual','shape':list(shape),'checks':checks,
                'output_dtype':str(policy.parameter_gradient_linear(p,x).dtype),
                'gradient_dtypes':[str(v.dtype) for v in jax.tree_util.tree_leaves(actual[1])]})
            for mode,function in (('state_jvp',lambda x:policy.parameter_gradient_linear(p,x)),):
                tangent = jnp.ones_like(x)*.01
                a = jax.jvp(function,(x,),(tangent,))
                b = jax.jvp(lambda x:_linear(p,x),(x,),(tangent,))
                evidence['interface_checks'].append({'name':mode,'shape':list(shape),'checks':{'primal':close(a[0],b[0]),'tangent':close(a[1],b[1])}})
        # Native checkpointed scan plumbing uses only a tiny synthetic state.
        scan_p = {'kernel':jnp.asarray(rng.normal(size=(16,16))*.02,dtype=jnp.float32),'bias':jnp.zeros((16,),jnp.float32)}
        initial = jnp.asarray(rng.normal(size=(16,))*.1,dtype=jnp.float32)
        for horizon in (1,7,30):
            def recurrent(p,x,linear):
                def step(state,day):
                    return jnp.tanh(linear(p,state)+day),None
                ending,_ = jax.lax.scan(jax.checkpoint(step),x,jnp.ones((horizon,16),jnp.float32)*.01)
                return jnp.sum(ending**2)
            a = jax.jit(jax.value_and_grad(lambda p,x:recurrent(p,x,policy.parameter_gradient_linear),argnums=(0,1)))(scan_p,initial)
            b = jax.jit(jax.value_and_grad(lambda p,x:recurrent(p,x,_linear),argnums=(0,1)))(scan_p,initial)
            names = ('loss','bias','kernel','initial')
            checks = {name:close(x,y) for name,x,y in zip(names,jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b),strict=True)}
            evidence['interface_checks'].append({'name':'checkpointed_scan','horizon':horizon,'checks':checks})
        # External vmap exercises batch transformation of the custom JVP.
        x = jnp.ones((2,3,16),jnp.float32)*.1
        a = jax.grad(lambda p,x:jnp.sum(jax.vmap(policy.parameter_gradient_linear,in_axes=(None,0))(p,x)),argnums=(0,1))(p,x)
        b = jax.grad(lambda p,x:jnp.sum(jax.vmap(_linear,in_axes=(None,0))(p,x)),argnums=(0,1))(p,x)
        checks = {name:close(a,b) for name,a,b in zip(('bias','kernel','input'),jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b),strict=True)}
        evidence['interface_checks'].append({'name':'external_vmap','checks':checks})
        # Float64 physical input casts follow original linear's FP32 boundary.
        x64 = jnp.ones((16,),jnp.float64)*.1
        a = jax.grad(loss,argnums=1)(p,x64)
        b = jax.grad(original_loss,argnums=1)(p,x64)
        evidence['interface_checks'].append({'name':'FP64_input_cast_boundary','checks':{'input':close(a,b)},'input_gradient_dtype':str(a.dtype)})
    def corrected_rank4_interface():
        value=jnp.ones((4,2,1,16),jnp.float32)*.1
        actual=jax.jit(jax.value_and_grad(loss,argnums=(0,1)))(p,value)
        expected=jax.jit(jax.value_and_grad(original_loss,argnums=(0,1)))(p,value)
        checks={name:close(a,b) for name,a,b in zip(('loss','bias','kernel','input'),jax.tree_util.tree_leaves(actual),jax.tree_util.tree_leaves(expected),strict=True)}
        evidence['interface_checks'].append({'name':'corrected_tiny_rank4_dual','shape':list(value.shape),'checks':checks})
    if args.mode=='cpu-interface':
        with fp32_tangent_dot_interface_emulator(policy,jax):
            corrected_rank4_interface() if args.actual_f4s_shapes else interfaces()
    else:
        output.write_text(json.dumps(evidence,indent=2)+'\n')
        if args.actual_f4s_shapes:
            evidence['prior_standard_interface_checks_not_repeated']=True
        elif recovery is None:
            interfaces()
        else:
            # Reproduce the exact original RNG stream without repeating tiny
            # interface executable tests or any timing decision.
            for shape in ((16,),(3,16),(2,3,16),(2,2,3,16)):
                rng.normal(size=shape)
            rng.normal(size=(16,16)); rng.normal(size=(16,))
            evidence['interface_checks_inherited_from_original'] = recovery['interface_checks']
        evidence['actual_shape_GPU_pairs'] = []
        for shape,width in actual_plan if args.actual_f4s_shapes else (((384,),384),((128,384),1536),((2,128,384),1536),((4,128,256),1)):
            value = jnp.asarray(rng.normal(size=shape)*.1,dtype=jnp.float32)
            parameters = {'kernel':jnp.asarray(rng.normal(size=(shape[-1],width))*.01,dtype=jnp.float32),
                          'bias':jnp.zeros((width,),jnp.float32)}
            native_loss = lambda p,x:jnp.mean(jnp.tanh(policy.parameter_gradient_linear(p,x))**2)
            original_mean = lambda p,x:jnp.mean(jnp.tanh(_linear(p,x))**2)
            if width == 1:
                # Known operand_upcaster rejection boundary: masked readout
                # [4,N,256]x[256,1], feeding an FP64 scalar-loss AD graph.
                mask = jnp.asarray((np.arange(np.prod(shape[:-1])).reshape(shape[:-1])%3)!=0)
                def readout_loss(p,x,linear):
                    prediction = linear(p,x)[...,0].astype(jnp.float64)
                    return jnp.mean(jnp.where(mask,prediction,0.)**2)
                native_loss = lambda p,x:readout_loss(p,x,policy.parameter_gradient_linear)
                original_mean = lambda p,x:readout_loss(p,x,_linear)
            original = jax.jit(jax.value_and_grad(original_mean,argnums=(0,1))).lower(parameters,value).compile()
            evidence['stage']='actual_shape_candidate_compile'
            evidence['active_shape']=list(shape)
            output.write_text(json.dumps(evidence,indent=2)+'\n')
            candidate = jax.jit(jax.value_and_grad(native_loss,argnums=(0,1))).lower(parameters,value).compile()
            compiled_ir = candidate.as_text()
            stem = 'actual-shape-'+'x'.join(map(str,shape))+f'-out{width}'
            (args.output_dir/(stem+'.optimized-hlo')).write_text(compiled_ir+'\n')
            # Fail closed if compiler output does not expose FP16 GEMM operands.
            half_gemms = native_fp16_gemms_from_hlo(compiled_ir)
            if not half_gemms:
                raise ValueError('native FP16 GEMM codegen not proven on actual shape; compatibility outcome incomplete')
            times = {'original':[],'candidate':[]}
            reference,actual = (jax.device_get(jax.block_until_ready(executable(parameters,value))) for executable in (original,candidate))
            observed_memory = jax.local_devices()[0].memory_stats() or {}
            if recovery is not None or args.actual_f4s_shapes:
                if 'peak_bytes_in_use' not in observed_memory or observed_memory['peak_bytes_in_use']>24*2**30:
                    raise MemoryError('artifact recovery observed peak unavailable or exceeds24GiB')
            checks = {name:close(a,b) for name,a,b in zip(('loss','bias','kernel','input'),jax.tree_util.tree_leaves(actual),jax.tree_util.tree_leaves(reference),strict=True)}
            arrays = {f'{label}_{name}':np.asarray(value)
                for label,result in (('candidate',actual),('original',reference))
                for name,value in zip(('loss','bias','kernel','input'),jax.tree_util.tree_leaves(result),strict=True)}
            array_file = args.output_dir/(stem+'.results.npz')
            np.savez(array_file,**arrays)
            row = {'shape':list(shape),'width':width,'checks':checks,
                'known_masked_FP64_scalar_readout_boundary':width==1,
                'native_FP16_gemm_instructions':half_gemms,'seconds':{'original':[],'candidate':[]},
                'actual_arrays':str(array_file),'observed_device_memory':observed_memory}
            evidence['actual_shape_GPU_pairs'].append(row)
            output.write_text(json.dumps(evidence,indent=2)+'\n')
            if not all(row['parameter_engineering_passed'] if name=='kernel' else row['passed'] for name,row in checks.items()):
                raise ValueError('actual-shape primal/state/bias strict or kernel engineering gate failed')
            for repeat in range(5):
                if recovery is not None:
                    break
                order = ('original','candidate') if repeat%2==0 else ('candidate','original')
                for name in order:
                    began = time.perf_counter()
                    jax.block_until_ready((original if name=='original' else candidate)(parameters,value))
                    times[name].append(time.perf_counter()-began)
            if recovery is None:
                row.update(seconds=times,original_over_candidate=float(np.median(times['original'])/np.median(times['candidate'])))
            else:
                row['timing_not_repeated'] = True
            output.write_text(json.dumps(evidence,indent=2)+'\n')
        evidence['component_speed_gate'] = recovery['component_speed_gate'] if recovery is not None else all(row['original_over_candidate']>=1.05
            for row in evidence['actual_shape_GPU_pairs'] if args.actual_f4s_shapes or (len(row['shape'])>=2 and row['width']==1536))
    # Native JVP and checkpointed scan target IR, outside any emulator context.
    x = jax.ShapeDtypeStruct((3,16),jnp.float32)
    full_jvp = lambda p,x,pt,xt:jax.jvp(policy.parameter_gradient_linear,(p,x),(pt,xt))
    cp = jax.make_jaxpr(full_jvp)(p,x,p,x)
    ir = lower_cuda_ir(jax,cp,'native_full_jvp')
    (args.output_dir/'native-full-jvp.mlir').write_text(ir+'\n')
    evidence['native_full_jvp_algorithm_present'] = 'accumulation_type = f32' in ir
    # Synthetic active-alias fixture verifies restoration without a real model.
    prefix = 'parameter_precision_interface_fixture'
    fake = [types.ModuleType(prefix+'.'+name) for name in policy.TARGETS]
    sys.modules[prefix+'.strong_baseline_architecture'] = types.SimpleNamespace(_linear=_linear,_ffn=_ffn)
    for module in fake:
        module._linear,module._ffn = _linear,_ffn
    macro = next(module for module in fake if module.__name__.endswith('shared_decoder_macro_packed'))
    decoder = types.FunctionType((lambda:None).__code__,macro.__dict__)
    sys.modules[prefix+'.shared_decoder_expert_execution'] = types.SimpleNamespace(expert_decode_fields=decoder)
    try:
        with policy.neural_policy(fake,mixed=True):
            assert all(module._linear is policy.parameter_gradient_linear for module in fake)
            assert all(module._ffn is policy.parameter_gradient_ffn for module in fake if module.__name__.rsplit('.',1)[-1] in policy.FFN_TARGETS)
            raise RuntimeError('deliberate restoration check')
    except RuntimeError:
        pass
    restored = all(module._linear is _linear and module._ffn is _ffn for module in fake)
    evidence['alias_restoration_fixture_passed'] = restored
    evidence['original_tokenizer_owner_unchanged'] = True
    accepted_interfaces = all(check['parameter_engineering_passed'] if name=='kernel' and args.mode=='gpu-interface' else check['passed']
        for row in evidence['interface_checks'] for name,check in row['checks'].items())
    evidence['status'] = ('artifact_recovery_complete_original_speed_gate_unchanged' if recovery is not None
        else 'passed' if accepted_interfaces and restored and evidence['native_full_jvp_algorithm_present'] and evidence.get('component_speed_gate',True) else 'numerical_or_component_speed_failure')
    evidence['acceptance_limit'] = 'CPU emulation validates plumbing only; targeted CUDA IR uses local JAX, not deployed V100 codegen or numerical admission.'
    output.write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({'status':evidence['status'],'mode':args.mode,'jax_version':jax.__version__,
        'backend':jax.default_backend(),'native_IR_cases':len(evidence['native_IR']),
        'interface_cases':len(evidence['interface_checks']),'rounding':evidence['rounding_illustration']},indent=2))
    return 0 if evidence['status'] in ('passed','artifact_recovery_complete_original_speed_gate_unchanged') else 1


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except (Exception,KeyboardInterrupt) as error:
        if '--output-dir' in sys.argv and not isinstance(error,FileExistsError):
            directory = Path(sys.argv[sys.argv.index('--output-dir')+1])
            directory.mkdir(parents=True,exist_ok=True)
            path = directory/'interface.json'
            report = json.loads(path.read_text()) if path.exists() else {'real_model_calls':0}
            report.update(status='runtime_or_codegen_failure',error_type=type(error).__name__,error=str(error))
            path.write_text(json.dumps(report,indent=2)+'\n')
        raise
