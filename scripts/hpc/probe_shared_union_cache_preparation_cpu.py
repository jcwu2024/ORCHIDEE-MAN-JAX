"""Bounded source construction and saved-IR evidence; no model compilation.

The six transformed stage jaxprs come from the existing native staged AD owner.
Only physical/loss units 1-4 are lowered, using JAX 0.4.38's native pjit lowering.
CPU IR equality is not GPU cache compatibility or numerical acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str) + '\n', encoding='utf-8')
    os.replace(temporary, path)


@contextmanager
def forbid_model_compilation(report):
    from jax._src import compiler
    originals = (compiler.compile_or_get_cached, compiler.backend_compile)
    report.update(model_compile_or_get_cached_calls=0, model_backend_compile_calls=0)
    def lookup(*arguments, **keywords):
        report['model_compile_or_get_cached_calls'] += 1
        raise RuntimeError('model compiler lookup is forbidden in CPU preparation')
    def backend(*arguments, **keywords):
        report['model_backend_compile_calls'] += 1
        raise RuntimeError('model backend compilation is forbidden in CPU preparation')
    compiler.compile_or_get_cached, compiler.backend_compile = lookup, backend
    try:
        yield
    finally:
        compiler.compile_or_get_cached, compiler.backend_compile = originals


def stage_equations(closed):
    names = ('neural_stage', 'transition_stage', 'loss_stage',
             'loss_stage', 'transition_stage', 'neural_stage')
    stages = [equation for equation in closed.jaxpr.eqns
              if equation.primitive.name == 'pjit' and equation.params.get('name') in set(names)]
    if tuple(equation.params['name'] for equation in stages) != names:
        raise ValueError('native staged AD jaxpr does not contain the expected six ordered pjit units')
    return stages


def lower_stage(equation):
    from jax._src import pjit
    from jax._src.interpreters import mlir
    return pjit._pjit_lower(**equation.params, lowering_platforms=('cpu',),
                           lowering_parameters=mlir.LoweringParameters(), pgle_profiler=None).stablehlo()


def describe_module(neural, module):
    evidence = neural.load_cache_evidence()
    return {'main_signature': neural.module_signature(module),
            'jax_canonical_computation_sha256': evidence.native_computation_digest(module),
            'source_text_sha256': hashlib.sha256(module.operation.get_asm().encode()).hexdigest()}


def ordered_type_differences(old, new):
    result = {}
    for field in ('argument_types', 'result_types'):
        left, right = old[field], new[field]
        differences = [{'index': index, 'saved': left[index] if index < len(left) else None,
                        'source': right[index] if index < len(right) else None}
                       for index in range(max(len(left), len(right)))
                       if (left[index] if index < len(left) else None) !=
                          (right[index] if index < len(right) else None)]
        result[field] = {'saved_count': len(left), 'source_count': len(right),
                         'different_positions': len(differences), 'first_differences': differences[:8]}
    return result


def tiny_capture_context_check(jax, neural, args):
    """Compare eager staged AD with whole-function abstract tracing on CPU."""
    import jax.numpy as jnp
    from jax._src.interpreters import mlir
    from jaxlib.mlir import ir
    def neural_stage(p, state):
        return {'tiny': p * state}, {'tiny': state > 0}
    def transition_stage(p, state, outputs):
        return {'used': outputs['tiny'] + state ** 2 + p,
                'unused': jnp.exp(p * state) + p}
    def loss_stage(p, state, transition):
        return jnp.sum(transition['used'] ** 2 + p + 0.1 * state), {
            'hard': {'nonfinite': jnp.sum(~jnp.isfinite(transition['used']))},
            'unused_auxiliary': transition['unused']}
    stages = (jax.jit(jax.checkpoint(neural_stage)), jax.jit(transition_stage), jax.jit(loss_stage))
    function = neural.native_staged_dispatcher(jax, stages)
    report = {'identity': {'scope': 'tiny_eager_vs_whole_abstract_capture'},
              'scope': 'same_tiny_CPU_function_not_actual_model_boundary_validation'}
    stage_args = SimpleNamespace(output=args.output.with_suffix('.capture-context.json'), mode='tiny',
                                 neural_remat=True, reuse_stage_report=None,
                                 required_cache_ordinals=(), memory_limit_gib=1,
                                 final_neural_compile_seconds=60)
    helper = SimpleNamespace(atomic_pickle=lambda path, value: _pickle(path, value))
    inputs = (np.array([0.3, 0.4], np.float32), np.array([0.5, 0.6], np.float32), np.float32(1))
    with neural.stage_compilation_evidence(stage_args, helper, lambda *a, **kw: None, report):
        actual = jax.device_get(jax.block_until_ready(function(*inputs)))
    if len(report['compile_units']) != 6:
        raise ValueError('tiny eager stage capture did not contain six units')
    for value in jax.tree_util.tree_leaves(actual):
        if not np.all(np.isfinite(value)):
            raise ValueError('tiny eager staged output or gradient is nonfinite')
    abstract = jax.tree_util.tree_map(lambda value: jax.ShapeDtypeStruct(np.shape(value), np.asarray(value).dtype), inputs)
    lower_report = {}
    comparisons = []
    with forbid_model_compilation(lower_report):
        equations = stage_equations(jax.make_jaxpr(function)(*abstract))
        for ordinal in (1, 2, 3, 4):
            source = describe_module(neural, lower_stage(equations[ordinal]))
            path = Path(report['compile_units'][ordinal]['stablehlo_artifact'])
            with path.open('rb') as handle:
                source_text = pickle.load(handle)
            with mlir.make_ir_context():
                saved = describe_module(neural, ir.Module.parse(source_text))
            comparisons.append({'ordinal': ordinal, 'actual_eager': saved, 'whole_abstract_lower': source,
                                'ordered_ABI_equal': saved['main_signature'] == source['main_signature'],
                                'JAX_canonical_IR_equal': saved['jax_canonical_computation_sha256']['value'] ==
                                                         source['jax_canonical_computation_sha256']['value'],
                                'ordered_type_differences': ordered_type_differences(saved['main_signature'], source['main_signature'])})
    report.update(whole_abstract_lower_guard=lower_report, physical_loss_comparison=comparisons,
                  whole_abstract_capture_can_change_tiny_interface=any(not unit['ordered_ABI_equal'] for unit in comparisons),
                  limitation='A demonstrated tiny capture-context difference does not identify the real GPU miss cause.')
    return report


def tiny_compatibility(jax, neural, args):
    import jax.numpy as jnp
    cache = args.output.with_suffix('.tiny-cache')
    if cache.exists():
        raise FileExistsError('preserve prior tiny cache; choose a fresh output stem')
    jax.config.update('jax_compilation_cache_dir', str(cache))
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 0)
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', 0)
    def transition_stage(value):
        return value * value + 0.25
    report = {'identity': {'scope': 'tiny_CPU_native_observer_compatibility'}}
    stage_args = SimpleNamespace(output=args.output.with_suffix('.tiny.json'), mode='tiny',
                                 neural_remat=False, reuse_stage_report=None,
                                 required_cache_ordinals=(), memory_limit_gib=1,
                                 final_neural_compile_seconds=60)
    helper = SimpleNamespace(atomic_pickle=lambda path, value: _pickle(path, value))
    with neural.stage_compilation_evidence(stage_args, helper, lambda *a, **kw: None, report):
        result = jax.jit(transition_stage)(np.array([1., 2.], np.float32))
        result = np.asarray(jax.block_until_ready(result))
    if not np.array_equal(result, np.array([1.25, 4.25], np.float32)):
        raise ValueError('tiny observer changed the result')
    def neural_stage(p, state):
        return {'tiny': p * state}, {'tiny': state > 0}
    def transition_stage(p, state, outputs):
        return outputs['tiny'] + state ** 2 + p
    def loss_stage(p, state, transition):
        return jnp.sum(transition ** 2 + p + 0.1 * state), {'hard': {'nonfinite': jnp.sum(~jnp.isfinite(transition))}}
    stages = (jax.jit(jax.checkpoint(neural_stage)), jax.jit(transition_stage), jax.jit(loss_stage))
    lower_report = {}
    with forbid_model_compilation(lower_report):
        abstract = jax.ShapeDtypeStruct((2,), np.float32)
        closed = jax.make_jaxpr(neural.native_staged_dispatcher(jax, stages))(
            abstract, abstract, jax.ShapeDtypeStruct((), np.float32))
        equations = stage_equations(closed)
        lower_report['stage_names'] = [equation.params['name'] for equation in equations]
        lower_report['lowered_physical_loss_units'] = [
            {'ordinal': ordinal, **describe_module(neural, lower_stage(equations[ordinal]))}
            for ordinal in (1, 2, 3, 4)]
    report['tiny_native_AD_lower_only'] = lower_report
    report['tiny_capture_context_comparison'] = tiny_capture_context_check(jax, neural, args)
    report.update(passed=True, scope='tiny_CPU_only_not_model_execution', result=result.tolist(), cache=str(cache))
    return report


def _pickle(path, value):
    with path.open('wb') as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)


def run(args):
    started = time.perf_counter()
    sys.path.insert(0, str(args.repo_root))
    import jax
    import jax.numpy as jnp
    import jaxlib
    from jax._src.interpreters import mlir
    from jaxlib.mlir import ir
    if jax.__version__ != '0.4.38' or jax.default_backend() != 'cpu':
        raise ValueError('preparation requires actual CPU JAX 0.4.38')
    neural = load_module('cache_preparation_neural_owner', args.neural_probe_file)
    report = {'status': 'running', 'scope': 'source_consistent_CPU_native_stage_lower_only_preparation',
              'jax_version': jax.__version__, 'jaxlib_version': jaxlib.__version__,
              'host': os.uname().nodename if hasattr(os, 'uname') else os.environ.get('COMPUTERNAME'),
              'GPU_execution': False, 'model_execution': False, 'training_updates': 0,
              'GPU_cache_compatibility': 'not_tested', 'scientific_acceptance': False,
              'process_limit_seconds': args.process_seconds}
    def emit(stage, **fields):
        report.update(stage=stage, elapsed_seconds=time.perf_counter()-started, **fields)
        write_json(args.output, report)
        print(stage, flush=True)
    try:
        with neural.deadline(args.process_seconds, 'CPU_preparation', emit):
            emit('tiny_observer_check')
            report['tiny_compatibility'] = tiny_compatibility(jax, neural, args)
            if args.tiny_only:
                emit('completed', status='tiny_compatibility_passed', model_compile_or_get_cached_calls=0,
                     model_backend_compile_calls=0)
                return report
            helper = neural.load_helper(args.real_daily_probe_file)
            from research.daily_coarse_graining.daily_operator import shared_process_head, shared_transformer
            from research.daily_coarse_graining.daily_operator.shared_recurrent_training import (
                make_shared_joint_daily_objective,
            )
            from scripts.hpc.run_shared_learning import load_learning_checkpoint
            _, reference = helper.trusted_profile(args)
            identity = helper.context_identity(reference, helper.initial_checkpoint(reference), 742)
            head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.repo_root, text=True).strip()
            if head != identity['source_git_head']:
                raise ValueError('immutable source checkout differs')
            with args.prepared_context.open('rb') as handle:
                context = pickle.load(handle)
            if context['identity'] != identity:
                raise ValueError('prepared context identity differs')
            parameters = load_learning_checkpoint(Path(identity['initial_checkpoint']),
                                                  expected_method=reference['method'])['parameters']
            if sum(np.size(value) for value in jax.tree_util.tree_leaves(parameters)) != 64_884_481:
                raise ValueError('parameter scope differs')
            with args.reference_result.open('rb') as handle:
                saved_reference = pickle.load(handle)
            prior = json.loads(args.reuse_stage_report.read_text(encoding='utf-8'))
            screen = load_module('cache_preparation_union_screen', args.neural_probe_file.with_name('probe_shared_union_neural_precision.py'))
            screen.validate_prior_report(prior, identity)
            host_sample = jax.tree_util.tree_map(lambda value: np.asarray(value)[0], context['daily4'])
            screen.validate_full_result(jax, saved_reference, identity, parameters,
                                        host_sample.inputs.continuous_state, reference=True)
            # Match both old source owners exactly before the model compiler guard.
            sample = jax.tree_util.tree_map(lambda value: jnp.asarray(value)[0], context['daily4'])
            jax.block_until_ready(sample)
            sample_leaves = jax.tree_util.tree_leaves(sample)
            host_leaves = jax.tree_util.tree_leaves(host_sample)
            if any(not np.array_equal(np.asarray(left), right, equal_nan=True)
                   or left.dtype != right.dtype for left, right in zip(sample_leaves, host_leaves, strict=True)):
                raise ValueError('source consistent and host validation samples differ')
            adapter = helper.load_staged_adapter(args.candidate_module_file)
            assembler = helper.load_backbone_assembler(args.backbone_module_file)
            package = 'research.daily_coarse_graining.daily_operator.'
            modules = [shared_transformer] + [sys.modules[package+name] for name in
                ('shared_backbone_execution', 'shared_backbone_projection_reuse', 'shared_decoder_macro_packed')]
            captured = {}
            def captured_raw(*arguments, **keywords):
                outputs, masks = adapter(*arguments, **keywords, dynamic_projection_execution='union')
                captured['raw_masks'] = masks
                return outputs, masks
            stages = neural.staged_daily_functions(jax, context['head'], sample, context,
                make_shared_joint_daily_objective, captured, type(context['head']).learned_outputs,
                neural_remat=True)
            function = neural.native_staged_dispatcher(jax, stages)
            abstract = jax.tree_util.tree_map(lambda value: jax.ShapeDtypeStruct(value.shape, value.dtype),
                                             (parameters, sample.inputs.continuous_state))
            emit('source_capture_ready', identity=identity, sample_capture={
                 'stage_source': 'jax.tree_util.tree_map(lambda value: jnp.asarray(value)[0], context[daily4])',
                 'host_validation_source': 'independent_numpy_row0', 'leaf_count': len(sample_leaves),
                 'values_shapes_dtypes_equal': True,
                 'construction_kernels': 'tiny_eager_import_and_sample_slices_allowed_before_model_guard'})
            with forbid_model_compilation(report), helper.candidate_process_head_patches(shared_process_head, captured_raw, assembler), neural.neural_policy(modules, mixed=True):
                emit('trace_native_staged_AD')
                closed = jax.make_jaxpr(function)(*abstract, jax.ShapeDtypeStruct((), np.float32))
                equations = stage_equations(closed)
                report['native_jaxpr_stage_names'] = [equation.params['name'] for equation in equations]
                units = report['physical_loss_units'] = []
                for ordinal in (1, 2, 3, 4):
                    emit('lower_physical_loss_unit', active_unit=ordinal)
                    module = lower_stage(equations[ordinal])
                    path = args.output.with_suffix(f'.stage-{ordinal:02d}.hlo.pkl')
                    _pickle(path, module.operation.get_asm())
                    current = describe_module(neural, module)
                    old_path = args.reuse_stage_report.with_suffix(f'.stage-{ordinal:02d}.hlo.pkl')
                    with old_path.open('rb') as handle:
                        old_source = pickle.load(handle)
                    with mlir.make_ir_context():
                        old_module = ir.Module.parse(old_source)
                        old = describe_module(neural, old_module)
                    signature_equal = current['main_signature'] == old['main_signature']
                    left, right = current['jax_canonical_computation_sha256'], old['jax_canonical_computation_sha256']
                    digest_equal = (left['status'] == right['status'] == 'observed' and left['value'] == right['value'])
                    units.append({'ordinal': ordinal, 'module': str(module.operation.attributes['sym_name']),
                                  'source_constructed': current, 'saved_GPU_IR': old,
                                  'ordered_ABI_equal': signature_equal, 'JAX_canonical_IR_equal': digest_equal,
                                  'ordered_type_differences': ordered_type_differences(old['main_signature'], current['main_signature']),
                                  'source_constructed_artifact': str(path), 'saved_artifact': str(old_path),
                                  'native_GPU_key_metadata': 'unavailable_in_prior_report_not_reconstructed',
                                  'CPU_native_cache_key': 'not_requested_lower_only'})
            emit('completed', status='CPU_preparation_completed',
                 all_physical_loss_ordered_ABI_equal=all(unit['ordered_ABI_equal'] for unit in units),
                 all_physical_loss_JAX_canonical_IR_equal=all(unit['JAX_canonical_IR_equal'] for unit in units),
                 limitations=['CPU lowering does not establish GPU cache compatibility.',
                              'Whole-dispatcher make_jaxpr adds an outer abstract trace and may change symbolic-zero/DCE interfaces relative to eager staged dispatch.',
                              'Original GPU options, flags, topology and native component digests remain unrecorded.',
                              'No physical IR substitution, cache rewrite, model execution or numerical screen occurred.'])
    except Exception as error:
        emit('exception', status='CPU_preparation_failed', error_type=type(error).__name__, error=str(error))
        raise
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ('repo-root', 'neural-probe-file', 'output'):
        result.add_argument('--'+name, type=Path, required=True)
    for name in ('real-daily-probe-file', 'source-profile', 'prepared-context', 'reference-result',
                 'reuse-stage-report', 'candidate-module-file', 'backbone-module-file'):
        result.add_argument('--'+name, type=Path)
    result.add_argument('--tiny-only', action='store_true')
    result.add_argument('--process-seconds', type=int, default=300)
    return result


if __name__ == '__main__':
    args = parser().parse_args()
    if not 0 < args.process_seconds <= 300:
        raise ValueError('CPU preparation must be bounded to at most 300 seconds')
    if args.output.exists():
        raise FileExistsError('preserve existing preparation report')
    if not args.tiny_only and any(getattr(args, name) is None for name in
        ('real_daily_probe_file', 'source_profile', 'prepared_context', 'reference_result',
         'reuse_stage_report', 'candidate_module_file', 'backbone_module_file')):
        raise ValueError('actual preparation requires all declared saved asset and owner paths')
    run(args)
