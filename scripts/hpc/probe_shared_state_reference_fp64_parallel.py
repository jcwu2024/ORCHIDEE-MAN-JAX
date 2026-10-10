"""Separately proposed original-public T7 FP64 parallel-compile diagnostic.

Requires separately authorized precision execution. No optimizer/parameter AD.
"""
from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import copy
import importlib
import dataclasses
import hashlib
import json
import re
from pathlib import Path
import pickle
from types import SimpleNamespace
import time
import numpy as np
from probe_shared_command_buffer_loops import comparison, deadline, write, forbid_component_compile
from shared_fp64_parallel_admission import graph_identity
from shared_neural_fp64_diagnostic import MODULES, PACKAGE, neural_fp64_functions, exact_physical_constant_widening


def scalar_metric(value, reference):
    value,reference=float(value),float(reference)
    error=abs(value-reference); allowance=1e-6+1e-5*abs(reference)
    return dict(value=value,reference=reference,absolute_error=error,
                strict_allowance=allowance,passed=bool(error<=allowance))


def exact_tree_bridge(jax, original, converted):
    left,structure=jax.tree_util.tree_flatten_with_path(original)
    right,other=jax.tree_util.tree_flatten_with_path(converted)
    if structure!=other:
        raise ValueError('prepared treedef/class identity changed')
    rows=[]
    for (path,a),(_,b) in zip(left,right,strict=True):
        a,b=np.asarray(a),np.asarray(b)
        expected=a.astype(np.float64) if a.dtype.kind=='f' else a
        # Byte comparison preserves undefined NaN placeholders too.
        if a.shape!=b.shape or expected.dtype!=b.dtype or expected.tobytes()!=b.tobytes():
            raise ValueError('prepared value differs beyond exact widening: '+str(path))
        rows.append(dict(path=jax.tree_util.keystr(path),shape=list(a.shape),
                         original_dtype=str(a.dtype),prepared_dtype=str(b.dtype),
                         exact_widen_or_unchanged=True,
                         original_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                         prepared_sha256=hashlib.sha256(b.tobytes()).hexdigest()))
    return dict(passed=True,leaf_count=len(rows),leaves=rows)


def mask_inventory(tree):
    rows={};arrays={}
    def walk(value,path,stack):
        if isinstance(value,(np.ndarray,np.generic)) or hasattr(value,'dtype') and hasattr(value,'shape'):
            a=np.asarray(value)
            if a.dtype.kind=='b':
                rows[path]=dict(shape=list(a.shape),true_count=int(np.count_nonzero(a)),
                                false_count=int(a.size-np.count_nonzero(a)),
                                sha256=hashlib.sha256(a.tobytes()).hexdigest())
                arrays[path]=a.copy()
            return
        if id(value) in stack:
            return
        stack=stack|{id(value)}
        if isinstance(value,dict):
            for key,child in value.items():walk(child,path+'.'+str(key),stack)
        elif dataclasses.is_dataclass(value):
            for field in dataclasses.fields(value):walk(getattr(value,field.name),path+'.'+field.name,stack)
        elif isinstance(value,(tuple,list)):
            for i,child in enumerate(value):
                name=value._fields[i] if hasattr(value,'_fields') else str(i)
                walk(child,path+'.'+name,stack)
    walk(tree,'source',set())
    return rows,arrays


def widen_context(tree):
    """Exactly widen calibration arrays without reconstructing serialized classes."""
    memo={};rows=[]
    def walk(value,path):
        if id(value) in memo:return memo[id(value)]
        if isinstance(value,(np.ndarray,np.generic)) or hasattr(value,'dtype') and hasattr(value,'shape'):
            a=np.asarray(value)
            if a.dtype==np.dtype('float32'):
                b=a.astype(np.float64)
                if not np.array_equal(b.astype(np.float32).view(np.uint32),a.view(np.uint32)):
                    raise ValueError('context widening changed source bits: '+path)
                rows.append(dict(path=path,shape=list(a.shape),original_dtype='float32',prepared_dtype='float64',
                    original_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                    prepared_sha256=hashlib.sha256(b.tobytes()).hexdigest(),exact_widen=True))
                memo[id(value)]=b
                return b
            return value
        if isinstance(value,dict):
            result=value.copy();memo[id(value)]=result
            for key,child in value.items():result[key]=walk(child,path+'.'+str(key))
            return result
        if dataclasses.is_dataclass(value) and not isinstance(value,type):
            result=copy.copy(value);memo[id(value)]=result
            for field in dataclasses.fields(value):
                object.__setattr__(result,field.name,walk(getattr(value,field.name),path+'.'+field.name))
            if type(result) is not type(value):raise ValueError('context class identity changed')
            return result
        if isinstance(value,(tuple,list)):
            children=[walk(child,path+'.'+str(i)) for i,child in enumerate(value)]
            result=type(value)(*children) if hasattr(value,'_fields') else type(value)(children)
            memo[id(value)]=result
            return result
        return value
    return walk(tree,'context'),rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-preflight',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--process-seconds',type=int,default=4430)
    parser.add_argument('--required-prior-report',type=Path,required=True)
    parser.add_argument('--required-graph-file',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve completed stages')
    started=time.perf_counter()
    report=dict(status='prepared_runtime_started',precision='full_neural_FP64_physical_FP64',
                parameter_gradient_admission=False,scientific_acceptance=False,training_updates=0,
                paid_jobs=0,gpu_count=1,rtol=1e-5,atol=1e-6,rows=[],
                model_compile_attempted=0,model_compiles=0,model_compile_completed=False,
                budgets_seconds=dict(lower=360,backend=3600,execution=600,process=args.process_seconds))
    def emit(stage,**values):
        report.update(stage=stage,elapsed_seconds=time.perf_counter()-started,**values)
        write(args.output,report)
        print(stage,flush=True)
    emit('initializing')
    try:
        with deadline(args.process_seconds,'process',emit):
            flags=os.environ.get('XLA_FLAGS','').split()
            expected=['--xla_gpu_enable_llvm_module_compilation_parallelism=true','--xla_gpu_force_compilation_parallelism=4']
            if flags!=expected:raise ValueError('exact proposed parallel4 flags required')
            nvlink=shutil.which('nvlink')
            expected_nvlink='/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/shared-state-reference-fp64-parallel-20261002/nvlink-bin/nvlink'
            if nvlink!=expected_nvlink:raise ValueError('transparent independent nvlink wrapper must resolve first')
            version=subprocess.run([nvlink,'--version'],check=True,text=True,capture_output=True)
            if 'V12.6.85' not in version.stdout:raise ValueError('independent nvlink version differs')
            prior=json.loads(args.required_prior_report.read_text())
            prior_root=Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-state-reference-fp64-20261002')
            if (args.required_prior_report!=prior_root/'model-exact-constant.json'
                    or args.required_graph_file!=prior_root/'model-exact-constant.stablehlo.txt'
                    or prior.get('stablehlo_f32_tensor_absent') is not True
                    or prior.get('status')!='timeout' or prior.get('timeout_scope')!='backend'):
                raise ValueError('completed prior dtype admission and backend-timeout evidence required')
            emit('compiler_environment_checked',xla_flags=flags,nvlink_wrapper=nvlink,nvlink_version=version.stdout.strip(),
                 compiler_options={'xla_gpu_enable_llvm_module_compilation_parallelism':True},
                 required_prior_report=str(args.required_prior_report),required_graph_file=str(args.required_graph_file))
            from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend
            jax,devices=initialize_gpu_backend()
            import jax.numpy as jnp
            from jax.experimental.serialize_executable import serialize
            import probe_shared_forward_reverse as forward
            from scripts.hpc.run_shared_learning import load_learning_checkpoint
            from research.daily_coarse_graining.daily_operator.shared_recurrent_training import make_shared_recurrent_objective
            from research.daily_coarse_graining.daily_operator import shared_process_head
            preflight=json.loads(args.runtime_preflight.read_text())
            if preflight.get('status')!='passed' or preflight.get('actual_backend')!='gpu' or jax.__version__!='0.4.38' or len(devices)!=1:
                raise ValueError('canonical actual GPU preflight/runtime required')
            root=Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX'); diag=root/'runtime/outputs/diagnostics'
            source=SimpleNamespace(source_profile=diag/'shared-split-profile-c599236.json',
                input_bundle=diag/'shared-split-dual-inputs-c599236.pkl',prepared_context=diag/'shared-expert-real-daily-context-c599236.pkl',
                daily_admission_report=diag/'shared-combined-real-daily-20260925.json',
                seven_admission_report=diag/'shared-combined-recurrent-T7-20260926.json',
                thirty_admission_report=diag/'shared-combined-recurrent-T30-20260926.json',
                support_owner_file=root/'runtime/transfer/shared-combined-full-update-20260926/probe_shared_serialized_dual.py')
            profile,ref,bundle,context,checkpoint_path,admissions=forward._assets(source)
            original_context=context
            context,context_bridge=widen_context(context)
            checkpoint=load_learning_checkpoint(checkpoint_path,expected_method=ref['method'])
            if checkpoint['completed_updates']!=0 or ref['source_git_head']!='c599236bcf57b745e9bafed678298010f4c3b58f':
                raise ValueError('original update0/source identity differs')
            weights=checkpoint['parameters']
            leaves=jax.tree_util.tree_leaves(weights)
            if len(leaves)!=300 or sum(np.size(v) for v in leaves)!=64884481:
                raise ValueError('parameter count/leaf scope differs')
            def widen(v):
                a=np.asarray(v)
                return a.astype(np.float64) if a.dtype.kind=='f' else a.copy()
            host_parameters=jax.tree_util.tree_map(widen,weights)
            weight_bridge=exact_tree_bridge(jax,weights,host_parameters)
            parameters=jax.device_put(host_parameters)
            case=bundle['payloads'][0]['cases'][7]
            if np.shape(case[0])!=(3854,) or np.asarray(case[0]).dtype!=np.float64 or np.shape(case.days[0].continuous_state)[0]!=7:
                raise ValueError('original T7/state ABI differs')
            if int(bundle['artifact_ids'][0])!=742:
                raise ValueError('artifact742 required')
            prepared_case=jax.tree_util.tree_map(widen,case)
            input_bridge=exact_tree_bridge(jax,case,prepared_case)
            arguments=(parameters,*jax.device_put(prepared_case))
            direction=np.eye(1,3854,33,dtype=np.float64)[0]
            owner=forward._load_module(root/'runtime/transfer/shared-combined-real-daily-20260925/probe_shared_expert_real_daily.py','scripts.hpc.probe_shared_expert_real_daily')
            staged=root/'runtime/transfer/shared-combined-recurrent-20260926'
            adapter=owner.load_staged_adapter(staged/'shared_decoder_expert_execution.py')
            backbone=owner.load_backbone_assembler(staged/'shared_backbone_execution.py')
            # This is the exact saved original identity used by the completed
            # native-FP32 JVP diagnosis; no old candidate executable is loaded.
            identity=dict(source_head=ref['source_git_head'],source_profile_sha256=hashlib.sha256(source.source_profile.read_bytes()).hexdigest(),
                method=ref['method'],artifact_id=742,horizon=7,checkpoint=str(checkpoint_path.resolve()),
                jax_version=jax.__version__,admitted_executable=admissions['T7']['executable'],
                candidate_decoder_sha256=hashlib.sha256((staged/'shared_decoder_expert_execution.py').read_bytes()).hexdigest(),
                candidate_backbone_sha256=hashlib.sha256((staged/'shared_backbone_execution.py').read_bytes()).hexdigest())
            fp32_dir=diag/'shared-original-state-jvp-20261002'
            with (fp32_dir/'model.original-dual-baseline.pkl').open('rb') as stream:
                fp32=pickle.load(stream)
            if fp32['identity']!=identity:
                raise ValueError('saved FP32 provenance/identity differs')
            fp32_jvp=json.loads((fp32_dir/'model.json').read_text())
            if fp32_jvp['identity']!=identity:
                raise ValueError('saved FP32 JVP identity differs')
            masks_tree=dict(case=prepared_case,layout=context['head'].layout,
                            normalization=context['head'].input_normalization,state_scales=context['state_scales'],
                            template=context['template'],contract=context['head'].contract)
            original_masks,_original_arrays=mask_inventory(dict(case=case,layout=original_context['head'].layout,
                normalization=original_context['head'].input_normalization,state_scales=original_context['state_scales'],
                template=original_context['template'],contract=original_context['head'].contract))
            prepared_masks,mask_arrays=mask_inventory(masks_tree)
            if not original_masks or original_masks!=prepared_masks:
                raise ValueError('declared/source/science/PFT masks changed during preparation')
            for key,value in dict(identity=identity,input_bridge=input_bridge,weight_bridge=weight_bridge,
                                  context_bridge=context_bridge,source_masks=prepared_masks).items():
                if prior.get(key)!=value:raise ValueError('prior admitted preparation differs: '+key)
            write(args.output.with_suffix('.input-identity.pkl'),dict(identity=identity,input_bridge=input_bridge,
                  weight_bridge=weight_bridge,context_bridge=context_bridge,masks=prepared_masks,mask_arrays=mask_arrays,
                  original_initial_state=np.asarray(case[0])))
            np.savez(args.output.with_suffix('.source-masks.npz'),**mask_arrays)
            emit('prepared_identity_checked',identity=identity,parameter_leaf_count=300,
                 input_bridge=input_bridge,weight_bridge=weight_bridge,context_bridge=context_bridge,source_masks=prepared_masks,
                 source_mask_identity_passed=True,fp32_original_saved_loss=float(fp32['loss_aux'][0]),
                 fp32_original_declared_mask_change=int(fp32['loss_aux'][1]['declared_mask_change']))
            modules=tuple(importlib.import_module(PACKAGE+name) for name in MODULES)
            with neural_fp64_functions(modules) as (replacement,manifest), exact_physical_constant_widening() as constant_manifest:
                adapter=replacement[id(adapter)];backbone=replacement[id(backbone)]
                _,objective=make_shared_recurrent_objective(context['head'],context['template'],context['state_scales'],context['science_scales'])
                def combined(p,state,discrete,days,targets,loss_context,tangent):
                    def loss(x):
                        return objective(p,x,discrete,days,targets,loss_context)
                    value,gradient=jax.value_and_grad(loss,has_aux=True)(state)
                    _,directional=jax.jvp(lambda x:loss(x)[0],(state,),(tangent,))
                    return value,gradient,directional
                emit('inputs_ready',parameter_coordinates=64884481,parameter_leaves=300,
                     parameter_bytes=64884481*8,neural_source_overlay=manifest,
                     exact_physical_constant_bridge=constant_manifest,artifact_id=742,horizon=7)
                began=time.perf_counter()
                with owner.candidate_process_head_patches(shared_process_head,adapter,backbone),deadline(360,'lower',emit):
                    lowered=jax.jit(combined).lower(*arguments,jax.device_put(direction))
                stable=str(lowered.compiler_ir(dialect='stablehlo'))
                args.output.with_suffix('.stablehlo.txt').write_text(stable)
                # Full public graph must not retain an f32 tensor. This fails
                # closed rather than claiming a complete reference prematurely.
                residual=re.findall(r'tensor<[^>]*(?:bf16|f16|f32)[^>]*>',stable)
                emit('lowering_saved',lower_seconds=time.perf_counter()-began,fp32_residual_tensor_count=len(residual),
                     fp32_residual_examples=sorted(set(residual))[:20])
                if residual:
                    raise ValueError('full public lowering retains FP32 tensor; inspect saved graph')
                same_graph=graph_identity(stable,args.required_graph_file.read_text())
                emit('graph_identity_checked',graph_identity=same_graph)
                if not same_graph['passed']:raise ValueError('public FP64 computational graph differs; stop before compile')
                if prior['neural_source_overlay']!=manifest or prior['exact_physical_constant_bridge']!=constant_manifest:
                    raise ValueError('audited neural/physical constant overlay identity differs')
                emit('lowered',lower_seconds=time.perf_counter()-began,stablehlo_f32_tensor_absent=True)
                began=time.perf_counter()
                emit('compiling',model_compile_attempted=1,model_compiles=0,model_compile_completed=False)
                with deadline(3600,'backend',emit):
                    binary=lowered.compile(compiler_options={'xla_gpu_enable_llvm_module_compilation_parallelism':True})
                emit('compiled',backend_seconds=time.perf_counter()-began,model_compiles=1,model_compile_completed=True)
                with deadline(600,'execution',emit):
                    m=binary.memory_analysis()
                    estimate=m.argument_size_in_bytes+m.output_size_in_bytes+m.temp_size_in_bytes-m.alias_size_in_bytes
                    if estimate>=24*2**30:
                        raise MemoryError('estimated live exceeds24GiB')
                    write(args.output.with_suffix('.executable.pkl'),serialize(binary))
                    emit('executable_saved',estimated_live_bytes=int(estimate))
                    first={};defined=np.isfinite(np.asarray(case[0]))&(np.abs(np.asarray(case[0]))<.5e20)
                    reference_value=None
                    for repeat in range(8):
                        for coordinate in ((33,199,428,200) if repeat%2==0 else (200,428,199,33)):
                            tangent=jax.device_put(np.eye(1,3854,coordinate,dtype=np.float64)[0])
                            began=time.perf_counter()
                            with forbid_component_compile(True):
                                value=jax.device_get(jax.block_until_ready(binary(*arguments,tangent)))
                            execution_seconds=time.perf_counter()-began
                            (loss,aux),gradient,jvp=value
                            # Preserve every complete result before evaluating gates.
                            write(args.output.with_suffix(f'.r{repeat}.c{coordinate}.pkl'),value)
                            if any(int(v) for v in aux['hard'].values()) or len(aux['hard'])!=14:
                                raise ValueError('physical hard counts failed')
                            if not all(np.isfinite(v).all() for v in jax.tree_util.tree_leaves(value)):
                                raise ValueError('nonfinite result')
                            if np.any(np.asarray(gradient)[~defined]!=0):
                                raise ValueError('undefined-state derivative not exactly zero')
                            if devices[0].memory_stats()['peak_bytes_in_use']>=24*2**30:
                                raise MemoryError('observed active peak exceeds24GiB')
                            previous=first.setdefault(coordinate,value)
                            if int(aux['declared_mask_change'])!=int(fp32['loss_aux'][1]['declared_mask_change']):
                                raise ValueError('declared mask count differs from original')
                            reference_value=value if reference_value is None else reference_value
                            invariance=comparison(jax,value[:2],reference_value[:2])
                            row=dict(repeat=repeat,coordinate=coordinate,loss=float(loss),jvp=float(jvp),vjp=float(gradient[coordinate]),
                                execution_seconds=execution_seconds,jvp_vjp=scalar_metric(jvp,gradient[coordinate]),
                                jvp_vjp_pass=scalar_metric(jvp,gradient[coordinate])['passed'],
                                self_comparison=comparison(jax,value,previous),direction_invariance=invariance,
                                fp32_primal_aux_comparison=comparison(jax,value[0],fp32['loss_aux']),
                                fp32_state_gradient_comparison=comparison(jax,gradient,fp32['state_gradient']),
                                fp32_selected_vjp=scalar_metric(gradient[coordinate],fp32['state_gradient'][coordinate]),
                                fp32_selected_jvp=scalar_metric(jvp,fp32_jvp['summary'][str(coordinate)]['jvp_range'][0]),
                                physics_hard_counts={k:int(v) for k,v in aux['hard'].items()},
                                declared_mask_change=int(aux['declared_mask_change']),undefined_state_count=int(np.count_nonzero(~defined)),
                                undefined_state_exact_zero=True,finite=True)
                            report['rows'].append(row)
                            emit('direction_saved',latest=row,device_memory=devices[0].memory_stats())
                        emit('repeat_saved',repeat=repeat,device_memory=devices[0].memory_stats())
                    passed=all(row['jvp_vjp_pass'] and row['self_comparison']['passed'] and row['direction_invariance']['passed'] for row in report['rows'])
                    emit('completed',status='diagnostic_consistent' if passed else 'diagnostic_discrepancy',model_compiles=1,
                         default_changed=False,parameter_gradient_admission=False,training_throughput_claim=False)
    except BaseException as error:
        emit('exception',status='error',error_type=type(error).__name__,error=str(error))
        raise


if __name__=='__main__':
    main()
