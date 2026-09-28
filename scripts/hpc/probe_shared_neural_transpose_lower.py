"""FP32 field/union native neural transpose shape lowering, never execution.

Reuses probe_shared_neural_precision::staged_daily_functions:159-212 with its
checkpointed neural stage and canonical learned outputs/masks. Native jax.vjp
is traced with ShapeDtypeStruct parameters/state/output cotangents. No physical
transition/loss is called; all executable lookups are forbidden during lowering.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import importlib.util
import math
import os
import pickle
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np

ROOT = Path(os.environ.get('ORCHIDEE_REPO_ROOT', Path(__file__).resolve().parents[2]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def deadline(seconds, label, emit):
    def expired():
        emit('timeout', status='timeout', timeout_scope=label, limit_seconds=seconds)
        os._exit(124)
    timer = threading.Timer(seconds, expired)
    timer.daemon = True
    timer.start()
    try:
        yield
    finally:
        timer.cancel()


def shape_tree(jax, value):
    return jax.tree_util.tree_map(lambda x: jax.ShapeDtypeStruct(np.shape(x), np.asarray(x).dtype), value)


def tree_signature(jax, tree):
    leaves = [{'path': jax.tree_util.keystr(path), 'shape': list(value.shape), 'dtype': str(value.dtype),
               'coordinates': math.prod(value.shape), 'bytes': math.prod(value.shape) * value.dtype.itemsize}
              for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]]
    return {'leaves': len(leaves), 'coordinates': sum(row['coordinates'] for row in leaves),
            'nominal_bytes': sum(row['bytes'] for row in leaves), 'leaf_shapes': leaves}


def native_transpose(jax, neural):
    def shape_only_neural_transpose(parameters, state, cotangent):
        _, pullback = jax.vjp(neural, parameters, state)
        return pullback(cotangent)
    return jax.jit(shape_only_neural_transpose)


def cotangent_shapes(jax, outputs):
    def tangent(value):
        dtype = value.dtype if np.issubdtype(value.dtype, np.inexact) else jax.dtypes.float0
        return jax.ShapeDtypeStruct(value.shape, dtype)
    return jax.tree_util.tree_map(tangent, outputs)


def ir_summary(module, signature):
    from jaxlib.mlir import ir
    def count(operation):
        counts = Counter([operation.name])
        for region in operation.regions:
            for block in region.blocks:
                for child in block.operations:
                    counts.update(count(child.operation))
        return counts
    total = count(module.operation)
    functions = []
    loops = []
    for operation in module.body.operations:
        operation = operation.operation
        if operation.name != 'func.func':
            continue
        name = str(operation.attributes['sym_name']).strip('"')
        counts = count(operation)
        callees = set()
        def relationships(child):
            if child.name == 'func.call':
                callees.add(ir.FlatSymbolRefAttr(child.attributes['callee']).value)
            if child.name == 'stablehlo.while':
                regions = []
                for region in child.regions:
                    region_counts = Counter()
                    for block in region.blocks:
                        for nested in block.operations:
                            region_counts.update(count(nested.operation))
                    regions.append({'operations': sum(region_counts.values()),
                                    'dot_general': region_counts['stablehlo.dot_general']})
                loops.append({'function': name, 'operand_type_samples': [str(value.type) for value in child.operands[:6]],
                              'regions': regions})
            for region in child.regions:
                for block in region.blocks:
                    for nested in block.operations:
                        relationships(nested.operation)
        relationships(operation)
        functions.append({'name': name, 'operations': sum(counts.values()),
                          'dot_general': counts['stablehlo.dot_general'], 'while': counts['stablehlo.while'],
                          'callees': sorted(callees)})
    by_name = {row['name']: row for row in functions}
    reachable, pending = set(), ['main']
    while pending:
        name = pending.pop()
        if name in reachable or name not in by_name:
            continue
        reachable.add(name)
        pending.extend(by_name[name]['callees'])
    reachable_counts = {key: sum(by_name[name][key] for name in reachable)
                        for key in ('operations', 'dot_general', 'while')}
    return {'main_signature': signature(module), 'operations': sum(total.values()),
            'dot_general': total['stablehlo.dot_general'], 'while': total['stablehlo.while'],
            'operation_counts': dict(total), 'function_count': len(functions),
            'main_reachable_unique_functions': len(reachable), 'reachable_helper_totals': reachable_counts,
            'reachable_count_note': 'Each reachable function definition counted once, not dynamic calls or FLOPs.',
            'while_regions': loops,
            'main': next(row for row in functions if row['name'] == 'main'),
            'largest_functions': sorted(functions, key=lambda row: row['operations'], reverse=True)[:20]}


@contextmanager
def compiler_guard(signature, state):
    """Track small initialization, forbid any cached/new execution when shaping."""
    from jax._src import compiler
    backend_name = 'backend_compile' if hasattr(compiler, 'backend_compile') else 'backend_compile_and_load'
    original_lookup, original_backend = compiler.compile_or_get_cached, getattr(compiler, backend_name)
    state['backend_entrypoint'] = backend_name
    active = None
    def lookup(*arguments, **keywords):
        nonlocal active
        module = arguments[1] if len(arguments) > 1 else keywords['computation']
        name = str(module.operation.attributes['sym_name']).strip('"')
        row = {'module': name, 'phase': state['phase'], 'signature': signature(module), 'backend_calls': 0}
        if state['phase'] != 'initialization' or any(word in name for word in ('neural_stage', 'transition_stage', 'loss_stage', 'neural_transpose')):
            state['forbidden_lookups'].append(row)
            raise RuntimeError('executable lookup forbidden in lower-only diagnostic: ' + name)
        if row['signature']['result_nominal_bytes'] > 2**20:
            state['forbidden_lookups'].append(row)
            raise RuntimeError('initialization executable exceeds 1MiB small-table bound: ' + name)
        state['initialization_lookups'].append(row)
        active = row
        began = time.perf_counter()
        try:
            return original_lookup(*arguments, **keywords)
        finally:
            row['seconds'] = time.perf_counter() - began
            active = None
    def backend(*arguments, **keywords):
        if state['phase'] != 'initialization':
            raise RuntimeError('backend compilation forbidden in lower-only diagnostic')
        if active is None:
            raise RuntimeError('unobserved initialization backend compilation forbidden')
        active['backend_calls'] += 1
        return original_backend(*arguments, **keywords)
    compiler.compile_or_get_cached = lookup
    setattr(compiler, backend_name, backend)
    try:
        yield
    finally:
        compiler.compile_or_get_cached = original_lookup
        setattr(compiler, backend_name, original_backend)


def run(args):
    os.environ['JAX_COMPILATION_CACHE_DIR'] = str(args.stage_cache)
    os.environ['JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS'] = '0'
    os.environ['JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES'] = '0'
    import jax
    jax.config.update('jax_compilation_cache_dir', str(args.stage_cache))
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 0)
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', 0)
    probe = load_script('scripts.hpc.probe_shared_neural_precision', args.neural_probe_file)
    helper = probe.load_helper(args.real_daily_probe_file)
    if args.output.exists():
        raise FileExistsError('refusing to overwrite lower-only report')
    guard = {'phase': 'initialization', 'initialization_lookups': [], 'forbidden_lookups': []}
    report = {'status': 'running', 'compiler_guard': guard, 'routes': {},
              'jax_version': jax.__version__, 'devices': [str(x) for x in jax.devices()],
              'training_updates': 0, 'production_adoption': False, 'numerical_acceptance': False,
              'scope': 'native_checkpointed_neural_transpose_shapes_and_IR_only_no_execution',
              'route_limit_seconds': args.route_seconds, 'process_limit_seconds': args.process_seconds}
    started = time.perf_counter()
    def emit(stage, **fields):
        report.update(stage=stage, elapsed_seconds=time.perf_counter() - started, **fields)
        helper.atomic_json(args.output, report)
        print(stage, flush=True)
    emit('initialization_start')
    with deadline(args.process_seconds, 'whole_process', emit), compiler_guard(probe.module_signature, guard):
        from research.daily_coarse_graining.daily_operator import shared_process_head
        from research.daily_coarse_graining.daily_operator.shared_recurrent_training import make_shared_joint_daily_objective
        from scripts.hpc.run_shared_learning import load_learning_checkpoint
        _, reference = helper.trusted_profile(args)
        if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() != reference['source_git_head']:
            raise ValueError('requires declared immutable source checkout')
        identity = helper.context_identity(reference, helper.initial_checkpoint(reference), 742)
        with args.prepared_context.open('rb') as stream:
            context = pickle.load(stream)
        if context['identity'] != identity:
            raise ValueError('prepared artifact742 identity mismatch')
        checkpoint = load_learning_checkpoint(Path(identity['initial_checkpoint']), expected_method=reference['method'])
        parameters = shape_tree(jax, checkpoint['parameters'])
        sample = jax.tree_util.tree_map(lambda x: np.asarray(x)[0], context['daily4'])
        state = shape_tree(jax, sample.inputs.continuous_state)
        parameter_signature = tree_signature(jax, parameters)
        if parameter_signature['coordinates'] != 64_884_481:
            raise ValueError('full-network parameter count differs')
        adapter = helper.load_staged_adapter(args.candidate_module_file)
        assembler = helper.load_backbone_assembler(args.backbone_module_file)
        learned = type(context['head']).learned_outputs
        emit('initialization_complete', identity=identity, parameters=parameter_signature,
             state=tree_signature(jax, state), neural_remat=True,
             lowering_boundary='jit(native_vjp(checkpointed_jitted_neural_stage))(abstract_parameters,state,output_cotangent)')
        guard['phase'] = 'shape_lowering'
        expected_cotangent_signature = expected_gradient_signature = None
        for route in ('field', 'union'):
            captured = {}
            def captured_raw(*arguments, **keywords):
                raw, masks = adapter(*arguments, **keywords, dynamic_projection_execution=route)
                captured['raw_masks'] = masks
                return raw, masks
            neural = probe.staged_daily_functions(jax, context['head'], sample, context,
                make_shared_joint_daily_objective, captured, learned, neural_remat=True)[0]
            emit('route_shape_start', active_route=route)
            began = time.perf_counter()
            with helper.candidate_process_head_patches(shared_process_head, captured_raw, assembler), deadline(args.route_seconds, route + '_trace_lower', emit):
                outputs = jax.eval_shape(neural, parameters, state)
                output_shape_seconds = time.perf_counter() - began
                cotangent = cotangent_shapes(jax, outputs)
                transpose = native_transpose(jax, neural)
                emit('route_lower_start', active_route=route, output_shape_seconds=output_shape_seconds)
                lower_began = time.perf_counter()
                lowered = transpose.lower(parameters, state, cotangent)
                lower_seconds = time.perf_counter() - lower_began
                gradients = jax.eval_shape(transpose, parameters, state, cotangent)
            guard['phase'] = 'IR_inspection'
            module = lowered.compiler_ir('stablehlo')
            text = module.operation.get_asm()
            artifact = args.output.with_suffix('.' + route + '.hlo.pkl')
            helper.atomic_pickle(artifact, text)
            summary = ir_summary(module, probe.module_signature)
            cotangent_signature = tree_signature(jax, cotangent)
            gradient_signature = tree_signature(jax, gradients)
            if expected_cotangent_signature is not None and (expected_cotangent_signature != cotangent_signature or expected_gradient_signature != gradient_signature):
                raise ValueError('paired field/union cotangent or complete-gradient shapes differ')
            expected_cotangent_signature, expected_gradient_signature = cotangent_signature, gradient_signature
            if tree_signature(jax, gradients[0]) != parameter_signature or tree_signature(jax, gradients[1]) != tree_signature(jax, state):
                raise ValueError('complete parameter/state gradient shapes differ from inputs')
            report['routes'][route] = {'output_shape_seconds': output_shape_seconds, 'lower_seconds': lower_seconds,
                'trace_lower_seconds': output_shape_seconds + lower_seconds,
                'trace_lower_save_inspect_seconds': time.perf_counter() - began,
                'neural_outputs': tree_signature(jax, outputs), 'cotangent': cotangent_signature,
                'gradients': gradient_signature, 'ir': summary, 'stablehlo_text_bytes': len(text.encode()),
                'stablehlo_artifact': str(artifact), 'executable_lookup_calls': 0, 'backend_compile_calls': 0}
            emit('route_complete', active_route=route)
            guard['phase'] = 'shape_lowering'
        guard['phase'] = 'IR_inspection'
        if args.historical_reverse_ir:
            from jax._src.interpreters import mlir
            from jaxlib.mlir import ir
            with args.historical_reverse_ir.open('rb') as stream:
                old_text = pickle.load(stream)
            with mlir.make_ir_context():
                old = ir_summary(ir.Module.parse(old_text), probe.module_signature)
            report['historical_reverse'] = {'artifact': str(args.historical_reverse_ir),
                'stablehlo_text_bytes': len(old_text.encode()), 'ir': old,
                'same_main_signature_as_paired_field': old['main_signature'] == report['routes']['field']['ir']['main_signature'],
                'comparability': 'historical_separate_pjit_transpose_vs_new_outer_native_vjp_wrapper_not_assumed_identical'}
        emit('completed', status='lowered_only', full_graph_executable_lookups=0, full_graph_backend_compiles=0,
             forward_executions=0, transpose_executions=0)
        return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-context', type=Path, required=True)
    parser.add_argument('--source-profile', type=Path, required=True)
    parser.add_argument('--real-daily-probe-file', type=Path, required=True)
    parser.add_argument('--neural-probe-file', type=Path, required=True)
    parser.add_argument('--candidate-module-file', type=Path, required=True)
    parser.add_argument('--backbone-module-file', type=Path, required=True)
    parser.add_argument('--historical-reverse-ir', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage-cache', type=Path, required=True)
    parser.add_argument('--route-seconds', type=int, default=180)
    parser.add_argument('--process-seconds', type=int, default=480)
    run(parser.parse_args())
