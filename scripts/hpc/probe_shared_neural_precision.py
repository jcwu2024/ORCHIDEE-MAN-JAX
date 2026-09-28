"""Bounded real daily loss/native-AD precision diagnostic, never training.

Uses the canonical shared joint-daily objective without inventing cotangents.
Only explicit shared-neural helper aliases change dot policy. Source formulas:
strong_baseline_architecture.py::_linear/_ffn:886-898; shared_transformer.py;
shared_backbone_projection_reuse.py; shared_decoder_macro_packed.py.
Physics, tokenizer/carbon encoders, attention scores and controls stay unchanged.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import pickle
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get('ORCHIDEE_REPO_ROOT', Path(__file__).resolve().parents[2]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONTRACT = {'exact_rtol': 1e-5, 'exact_atol': 1e-6, 'loss_relative_limit': 1e-3,
            'gradient_relative_l2_limit': 1e-2, 'gradient_cosine_minimum': 0.9999,
            'near_zero_reference_norm': 1e-12, 'near_zero_absolute_max_limit': 1e-6,
            'inactive_reference_gradient_coordinates_must_remain_zero': True,
            'output_normalized_discrepancies': 'descriptive_only_no_invented_gate',
            'scope': 'provisional_engineering_screen_not_science_or_production_admission'}


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


def algorithm_linear(parameters, value):
    import jax
    import jax.numpy as jnp
    x = jnp.asarray(value, jnp.float32)
    kernel = jnp.asarray(parameters['kernel'], jnp.float32)
    return jax.lax.dot_general(x, kernel, (((x.ndim - 1,), (0,)), ((), ())),
                              precision=jax.lax.DotAlgorithmPreset.F16_F16_F32,
                              preferred_element_type=jnp.float32) + parameters['bias']


def algorithm_ffn(parameters, value):
    import jax
    return algorithm_linear(parameters['second'], jax.nn.gelu(algorithm_linear(parameters['first'], value)))


@contextmanager
def neural_policy(modules, *, mixed):
    """Never modify the original strong-baseline helper or any physical module."""
    saved = []
    try:
        if mixed:
            for module in modules:
                for name, replacement in (('_linear', algorithm_linear), ('_ffn', algorithm_ffn)):
                    if hasattr(module, name):
                        saved.append((module, name, getattr(module, name)))
                        setattr(module, name, replacement)
        yield
    finally:
        for module, name, original in reversed(saved):
            setattr(module, name, original)


def vector_metrics(jax, actual, reference):
    """Groupwise FP64 host reductions; near-zero ratios are intentionally absent."""
    a_leaves, a_tree = jax.tree_util.tree_flatten(actual)
    b_leaves, b_tree = jax.tree_util.tree_flatten(reference)
    if a_tree != b_tree:
        raise ValueError('comparison pytree mismatch')
    count = failures = nonfinite = zeros_a = zeros_b = gained_zero = zero_changed = 0
    a2 = b2 = delta2 = dot = maximum = 0.0
    for a, b in zip(a_leaves, b_leaves, strict=True):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError('comparison shape/dtype mismatch')
        a, b = a.astype(np.float64), b.astype(np.float64)
        finite = np.isfinite(a) & np.isfinite(b)
        nonfinite += int(np.count_nonzero(~finite))
        count += a.size
        zeros_a += int(np.count_nonzero(a == 0))
        zeros_b += int(np.count_nonzero(b == 0))
        gained_zero += int(np.count_nonzero((a == 0) & (b != 0)))
        zero_changed += int(np.count_nonzero((b == 0) & (a != 0)))
        failures += int(np.count_nonzero(~(finite & np.isclose(a, b, rtol=1e-5, atol=1e-6))))
        av, bv = a[finite], b[finite]
        delta = av - bv
        a2 += float(np.sum(av * av)); b2 += float(np.sum(bv * bv))
        delta2 += float(np.sum(delta * delta)); dot += float(np.sum(av * bv))
        maximum = max(maximum, float(np.max(np.abs(delta), initial=0)))
    near_zero = np.sqrt(b2) <= CONTRACT['near_zero_reference_norm']
    relative = None if near_zero or nonfinite else float(np.sqrt(delta2 / b2))
    cosine = None if near_zero or not a2 or nonfinite else float(np.clip(dot / np.sqrt(a2 * b2), -1, 1))
    screening = not nonfinite and (maximum <= 1e-6 if near_zero else
                                  relative <= 1e-2 and cosine is not None and cosine >= 0.9999)
    return {'coordinates': count, 'nonfinite': nonfinite, 'exact_failed': failures,
            'exact_route_passed': failures == 0, 'max_absolute': maximum if not nonfinite else None,
            'reference_norm': float(np.sqrt(b2)), 'candidate_norm': float(np.sqrt(a2)),
            'delta_norm': float(np.sqrt(delta2)), 'relative_l2': relative, 'cosine': cosine,
            'near_zero_reference': bool(near_zero), 'screening_passed': bool(screening),
            'reference_zero_fraction': zeros_b / max(count, 1),
            'candidate_zero_fraction': zeros_a / max(count, 1),
            'new_zero_coordinates': gained_zero, 'reference_zero_became_nonzero': zero_changed}


def parameter_groups(parameters):
    result = {}
    for root, value in parameters.items():
        if root == 'shared_transformer':
            result.update({root + '.' + name: child for name, child in value.items()})
        else:
            result[root] = value
    return result


def state_groups(gradient, contract):
    leaves = [leaf for leaf in contract.state_leaves if not leaf.discrete]
    if sum(leaf.stop - leaf.start for leaf in leaves) != np.size(gradient):
        raise ValueError('semantic state slices do not cover the continuous gradient')
    return {leaf.key: np.asarray(gradient)[leaf.start:leaf.stop] for leaf in leaves}


def undefined_state_gradients_zero(jax, result, context):
    sample = jax.tree_util.tree_map(lambda value: value[0], context['daily4'])
    defined = np.asarray(context['head'].input_assembler.assemble(sample.inputs).state_defined.other)
    if defined.dtype != np.bool_ or defined.shape != np.shape(result['gradients'][1]):
        raise ValueError('continuous defined-mask dtype/shape mismatch')
    return bool(np.all(np.asarray(result['gradients'][1])[~defined] == 0))


def load_helper(path):
    name = 'scripts.hpc.probe_shared_expert_real_daily'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def staged_daily_functions(jax, head, sample, context, make_objective, captured, learned_method, *, neural_remat=False):
    """Factor existing shared owners, never reimplement loss or physical rules.

    Owners: model.py::bind_daily_operator_dynamic_retained_inputs:136-160;
    mechanistic_process_head.py::prediction_from_learned_outputs:225-249 and
    materialize_post_update:251-274; shared_training.py::make_shared_daily_objective.
    """
    from research.daily_coarse_graining.daily_operator import shared_training
    from research.daily_coarse_graining.daily_operator.model import bind_daily_operator_dynamic_retained_inputs

    def replaced(state):
        return sample._replace(inputs=sample.inputs._replace(continuous_state=state))

    def neural_stage(parameters, state):
        assembled = head.input_assembler.assemble(replaced(state).inputs)
        encoded = head.assemble_context(parameters, assembled)
        captured.clear()
        outputs = learned_method(head, parameters, encoded, assembled)
        return outputs, captured['raw_masks']

    class DecodedHead:
        def __init__(self, outputs):
            self.outputs = outputs

        def assemble_context(self, parameters, assembled):
            # The original post-update method explicitly discards this context.
            return ()

        def __call__(self, parameters, encoded, assembled):
            return head.prediction_from_learned_outputs(self.outputs, assembled)

        def materialize_post_update(self, *arguments):
            return head.materialize_post_update(*arguments)

    def transition_stage(parameters, state, outputs):
        row = replaced(state)
        transition = bind_daily_operator_dynamic_retained_inputs(
            input_assembler=head.input_assembler, process_head=DecodedHead(outputs),
            retained_tail_template=context['template'])
        return transition(parameters, row.inputs, row.day_inputs)

    def loss_stage(parameters, state, result):
        original = shared_training.bind_daily_operator_dynamic_retained_inputs
        try:
            shared_training.bind_daily_operator_dynamic_retained_inputs = lambda **_: lambda *args: result
            objective = make_objective(context['runtime'], head, context['template'],
                context['state_scales'], context['science_scales'], include_state_diagnostics=True)
        finally:
            shared_training.bind_daily_operator_dynamic_retained_inputs = original
        return objective(parameters, replaced(state))

    if neural_remat:
        neural_stage = jax.checkpoint(neural_stage)
    return tuple(jax.jit(function) for function in (neural_stage, transition_stage, loss_stage))


def native_staged_dispatcher(jax, stages):
    neural, transition, loss_function = stages

    def measured(parameters, state, multiplier):
        outputs, masks = neural(parameters, state)
        result = transition(parameters, state, outputs)
        loss, auxiliary = loss_function(parameters, state, result)
        return loss * multiplier, {'auxiliary': auxiliary, 'outputs': outputs, 'raw_masks': masks}

    return jax.value_and_grad(measured, argnums=(0, 1), has_aux=True)


def staged_runtime_check(jax, *, neural_remat=False):
    """Tiny deployed check includes neural, transition and direct loss paths."""
    import jax.numpy as jnp
    def neural(p, state):
        return {'tiny': jnp.exp(p * state)}, {'tiny': state > 0}
    def transition(p, state, output):
        return output['tiny'] + state ** 2 + p * state
    def loss(p, state, result):
        return jnp.sum(result ** 2 + 0.1 * state + p), {'hard': {'nonfinite': jnp.sum(~jnp.isfinite(result))}}
    staged = native_staged_dispatcher(jax, (jax.jit(jax.checkpoint(neural) if neural_remat else neural),
        jax.jit(transition), jax.jit(loss)))
    def whole(p, state, multiplier):
        output, mask = neural(p, state)
        scalar, auxiliary = loss(p, state, transition(p, state, output))
        return scalar * multiplier, {'auxiliary': auxiliary, 'outputs': output, 'raw_masks': mask}
    direct = jax.jit(jax.value_and_grad(whole, argnums=(0, 1), has_aux=True))
    inputs = (jnp.array([0.3, 0.4]), jnp.array([0.5, 0.6]), jnp.asarray(1.0, jnp.float32))
    actual, reference = jax.device_get(jax.block_until_ready((staged(*inputs), direct(*inputs))))
    for left, right in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(reference), strict=True):
        if not np.all(np.isfinite(left)) or not np.allclose(left, right, rtol=1e-5, atol=1e-6):
            raise ValueError('native staged deployed runtime gate failed')
    return {'passed': True, 'scope': 'tiny_staged_native_AD_interfaces_not_actual_model_parity',
            'native_result_leaves': [{'dtype': str(np.asarray(value).dtype), 'value': np.asarray(value).tolist()}
                for value in jax.tree_util.tree_leaves(actual)]}


def module_signature(module):
    from jaxlib.mlir import ir
    for function in module.body.operations:
        operation = function.operation
        if operation.name != 'func.func' or str(operation.attributes['sym_name']).strip('"') != 'main':
            continue
        signature = ir.FunctionType(operation.attributes['function_type'].value)
        def nominal_bytes(types):
            total = 0
            for value in types:
                tensor = ir.RankedTensorType(value)
                dtype = str(tensor.element_type)
                bits = int(dtype[1:]) if dtype[0] in ('f', 'i') and dtype[1:].isdigit() else 0
                total += math.prod(tensor.shape) * max(bits // 8, 1)
            return total
        return {'argument_leaves': len(signature.inputs), 'argument_nominal_bytes': nominal_bytes(signature.inputs),
                'result_leaves': len(signature.results), 'result_nominal_bytes': nominal_bytes(signature.results)}
    raise ValueError('compiled stage has no main signature')


@contextmanager
def stage_compilation_evidence(args, helper, emit, report):
    """Observe native compiler units and retain each completed persistent cache."""
    from jax._src import compiler
    original = compiler.compile_or_get_cached
    original_backend = compiler.backend_compile
    active = None
    events = []
    report['compile_units'] = events
    prior_units = []
    prior_mode = args.mode
    if args.reuse_stage_report is not None:
        prior = json.loads(args.reuse_stage_report.read_text())
        if prior.get('identity') != report.get('identity') or prior.get('neural_remat') != args.neural_remat:
            raise ValueError('partial native stage identity differs')
        prior_units = prior.get('compile_units', [])
        prior_mode = prior.get('mode')

    def backend(*arguments, **keywords):
        if active is not None and active.get('required_cache_hit'):
            active['status'] = 'completed_unit_cache_miss'
            emit('stopped_before_backend', status='completed_unit_cache_miss')
            raise RuntimeError('refusing to recompile completed unit: ' + active['module'])
        began = time.perf_counter()
        if active is not None:
            active['backend_compile_calls'] += 1
            active['backend_status'] = 'compiling'
            emit('backend_compile', active_compile_unit=active['ordinal'])
        result = original_backend(*arguments, **keywords)
        if active is not None:
            active['backend_status'] = 'completed'
            active['backend_compile_seconds'] += time.perf_counter() - began
        return result

    def lookup(*arguments, **keywords):
        nonlocal active
        module = arguments[1] if len(arguments) > 1 else keywords['computation']
        name = str(module.operation.attributes['sym_name']).strip('"')
        if not any(stage in name for stage in ('neural_stage', 'transition_stage', 'loss_stage')):
            return original(*arguments, **keywords)
        ordinal = len(events)
        artifact = args.output.with_suffix(f'.stage-{ordinal:02d}.hlo.pkl')
        helper.atomic_pickle(artifact, module.operation.get_asm())
        event = {'module': name, 'ordinal': ordinal, 'stablehlo_artifact': str(artifact), 'status': 'compiling'}
        event.update(backend_compile_calls=0, backend_compile_seconds=0.0)
        event['main_signature'] = module_signature(module)
        require_old = prior_mode == args.mode or 'neural_stage' not in name
        if require_old and ordinal < len(prior_units) and prior_units[ordinal].get('status') == 'completed':
            old = prior_units[ordinal]
            if (old['module'], old['main_signature']) != (name, event['main_signature']):
                raise ValueError('completed stage module/signature changed')
            event['required_cache_hit'] = True
        events.append(event)
        emit('compile_stage', active_compile_unit=ordinal)
        if args.neural_remat and 'neural_stage' in name and ordinal == 0:
            signature = event['main_signature']
            if signature['result_leaves'] >= 18_964 or signature['result_nominal_bytes'] >= 3_858_177_208:
                event['status'] = 'residual_interface_not_reduced'
                emit('stopped_before_backend', status='interface_not_reduced')
                raise ValueError('checkpoint did not reduce previous measured residual interface')
        began = time.perf_counter()
        active = event
        limit = args.final_neural_compile_seconds if ordinal == 5 and 'neural_stage' in name else 300
        event['compile_limit_seconds'] = limit
        try:
            with deadline(limit, 'compile_unit_' + name, emit):
                result = original(*arguments, **keywords)
        finally:
            active = None
        event.update(status='completed', seconds=time.perf_counter() - began)
        event['cache_hit'] = event['backend_compile_calls'] == 0
        stats = result.get_compiled_memory_stats()
        if stats is not None:
            sizes = {key: int(getattr(stats, key)) for key in
                ('argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes', 'alias_size_in_bytes')}
            sizes['estimated_live_bytes'] = sizes['argument_size_in_bytes'] + sizes['output_size_in_bytes'] + sizes['temp_size_in_bytes'] - sizes['alias_size_in_bytes']
            event['memory'] = sizes
            if sizes['estimated_live_bytes'] > args.memory_limit_gib * 2**30:
                raise ValueError('stage declared memory bound exceeded: ' + name)
        emit('stage_completed', active_compile_unit=None)
        return result

    compiler.compile_or_get_cached = lookup
    compiler.backend_compile = backend
    try:
        yield
    finally:
        compiler.compile_or_get_cached = original
        compiler.backend_compile = original_backend


def compare_results(jax, actual, reference, context):
    comparisons = {'loss': vector_metrics(jax, actual['loss'], reference['loss']),
                   'parameters': {}, 'continuous_state': {}, 'outputs': {}}
    for name, candidate in parameter_groups(actual['gradients'][0]).items():
        comparisons['parameters'][name] = vector_metrics(jax, candidate, parameter_groups(reference['gradients'][0])[name])
    expected_state = state_groups(reference['gradients'][1], context['head'].contract)
    for name, candidate in state_groups(actual['gradients'][1], context['head'].contract).items():
        comparisons['continuous_state'][name] = vector_metrics(jax, candidate, expected_state[name])
    head = context['head']
    masks_equal = True
    inactive_gradients_zero = True
    inactive_gradients_zero &= undefined_state_gradients_zero(jax, actual, context)
    for name, candidate in actual['diagnostics']['outputs'].items():
        expected = reference['diagnostics']['outputs'][name]
        metrics = vector_metrics(jax, candidate, expected)
        normalization = head.output_normalization
        if head.persistent_coordinates is not None and name in head.persistent_coordinates.fields:
            normalization = head.persistent_coordinates.normalization
        scale = normalization.scale.get(name)
        if scale is None:
            metrics['normalization'] = 'missing_existing_output_scale'
        else:
            scale = np.asarray(scale)
            if scale.size == np.size(candidate):
                scale = scale.reshape(np.shape(candidate))
            try:
                scale = np.broadcast_to(scale, np.shape(candidate))
            except ValueError:
                metrics['normalization'] = 'incompatible_existing_output_scale_shape'
                comparisons['outputs'][name] = metrics
                continue
            valid = np.isfinite(scale) & (scale > 0)
            if np.all(valid):
                discrepancy = (np.asarray(candidate) - np.asarray(expected)) / scale
                metrics.update(normalization='existing_TRAIN_output_scale',
                               normalized_rms=float(np.sqrt(np.mean(discrepancy ** 2))),
                               normalized_max=float(np.max(np.abs(discrepancy), initial=0)))
            else:
                metrics['normalization'] = 'invalid_or_zero_existing_output_scale'
        comparisons['outputs'][name] = metrics
    for name, mask in actual['diagnostics']['raw_masks'].items():
        expected = reference['diagnostics']['raw_masks'][name]
        masks_equal &= bool(np.asarray(mask).dtype == np.asarray(expected).dtype and np.array_equal(mask, expected))
    state_masks_equal = True
    for name in ('predicted_defined', 'expected_defined'):
        mask = actual['diagnostics']['auxiliary']['state_diagnostics'][name]
        expected = reference['diagnostics']['auxiliary']['state_diagnostics'][name]
        metrics = vector_metrics(jax, mask, expected)
        state_masks_equal &= metrics['exact_route_passed']
    physical = actual['diagnostics']['auxiliary']['hard']
    hard = {name: int(np.asarray(value)) for name, value in physical.items()}
    ref_loss = abs(float(np.asarray(reference['loss'])))
    loss_delta = abs(float(np.asarray(actual['loss']) - np.asarray(reference['loss'])))
    loss_relative = None if ref_loss <= 1e-12 else loss_delta / ref_loss
    loss_passed = loss_delta <= 1e-6 if loss_relative is None else loss_relative <= 1e-3
    gradient_rows = list(comparisons['parameters'].values()) + list(comparisons['continuous_state'].values())
    all_finite = comparisons['loss']['nonfinite'] == 0 and all(row['nonfinite'] == 0 for row in gradient_rows + list(comparisons['outputs'].values()))
    passed = loss_passed and all_finite and masks_equal and state_masks_equal and inactive_gradients_zero and not any(hard.values()) and all(row['screening_passed'] for row in gradient_rows)
    return {'comparisons': comparisons, 'loss_relative_difference': loss_relative,
            'loss_absolute_difference': loss_delta, 'raw_masks_exact': masks_equal,
            'state_diagnostic_masks_exact': bool(state_masks_equal),
            'undefined_input_state_gradients_exact_zero': inactive_gradients_zero,
            'candidate_hard_counts': hard, 'screening_passed': bool(passed),
            'exact_route_passed': comparisons['loss']['exact_route_passed'] and
                all(row['exact_route_passed'] for row in gradient_rows + list(comparisons['outputs'].values()))}


def configure_stage_cache(args):
    if args.execution != 'staged':
        return None
    cache = args.stage_cache or args.output.with_suffix('.stages-cache')
    # Environment first also covers any future helper-level JAX imports.
    os.environ['JAX_COMPILATION_CACHE_DIR'] = str(cache)
    os.environ['JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS'] = '0'
    os.environ['JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES'] = '0'
    import jax
    jax.config.update('jax_compilation_cache_dir', str(cache))
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 0)
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', 0)
    return cache


def run(args):
    cache = configure_stage_cache(args)
    helper = load_helper(args.real_daily_probe_file)
    report = {'status': 'running', 'stage': 'initializing', 'contract': CONTRACT,
              'training_updates': 0, 'precision_adopted': False, 'scientific_acceptance': False,
              'mode': args.mode, 'loss_multiplier': args.loss_multiplier,
              'process_limit_seconds': args.process_seconds,
              'final_neural_compile_limit_seconds': args.final_neural_compile_seconds}
    started = time.perf_counter()

    def emit(stage, **fields):
        report.update(stage=stage, elapsed_seconds=time.perf_counter() - started, **fields)
        helper.atomic_json(args.output, report)
        print(stage, flush=True)

    with deadline(args.process_seconds, 'whole_process', emit):
        import jax
        if args.cache_smoke_only:
            from jax._src import compiler
            original_backend = compiler.backend_compile
            calls = []
            def observed_backend(*arguments, **keywords):
                calls.append(1)
                return original_backend(*arguments, **keywords)
            compiler.backend_compile = observed_backend
            try:
                with deadline(60, 'tiny_crossprocess_cache', emit):
                    gate = staged_runtime_check(jax, neural_remat=args.neural_remat)
            finally:
                compiler.backend_compile = original_backend
            emit('completed', status='cache_smoke_completed', staged_runtime_gate=gate,
                 backend_compile_calls=len(calls), cache_directory=str(cache),
                 executable_cache_files=len(list(cache.glob('*-cache'))))
            return report
        import jax.numpy as jnp
        # Eager package initialization keeps physical constant tables out of tracing.
        from research.daily_coarse_graining.daily_operator import shared_process_head, shared_transformer
        from research.daily_coarse_graining.daily_operator.shared_recurrent_training import make_shared_joint_daily_objective
        from scripts.hpc.run_shared_learning import load_learning_checkpoint
        from scripts.hpc.run_shared_transformer_admission import memory_sizes
        from jax.experimental.serialize_executable import serialize

        if args.require_gpu and (len(jax.local_devices()) != 1 or jax.local_devices()[0].platform != 'gpu'):
            raise RuntimeError('one visible GPU required')
        if args.execution == 'staged':
            with deadline(60, 'tiny_staged_runtime', emit):
                emit('staged_runtime_passed', staged_runtime_gate=staged_runtime_check(jax, neural_remat=args.neural_remat))
        profile, reference = helper.trusted_profile(args)
        observed_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        if observed_head != reference['source_git_head']:
            raise ValueError('requires declared immutable source checkout')
        with args.prepared_context.open('rb') as stream:
            context = pickle.load(stream)
        identity = helper.context_identity(reference, helper.initial_checkpoint(reference), 742)
        if context['identity'] != identity:
            raise ValueError('prepared artifact742 identity mismatch')
        checkpoint = load_learning_checkpoint(Path(identity['initial_checkpoint']), expected_method=reference['method'])
        parameters = jax.device_put(checkpoint['parameters'])
        parameter_count = sum(np.size(x) for x in jax.tree_util.tree_leaves(parameters))
        if parameter_count != 64_884_481:
            raise ValueError('full network parameter count drift')
        sample = jax.tree_util.tree_map(lambda value: jnp.asarray(value)[0], context['daily4'])
        state = sample.inputs.continuous_state
        adapter = helper.load_staged_adapter(args.candidate_module_file)
        assembler = helper.load_backbone_assembler(args.backbone_module_file)
        package = 'research.daily_coarse_graining.daily_operator.'
        modules = [shared_transformer] + [sys.modules[package + name] for name in
                   ('shared_backbone_execution', 'shared_backbone_projection_reuse', 'shared_decoder_macro_packed')]
        objective = make_shared_joint_daily_objective(context['runtime'], context['head'], context['template'],
                    context['state_scales'], context['science_scales'], include_state_diagnostics=True)
        captured = {}
        original_learned = type(context['head']).learned_outputs

        def captured_raw(*a, **kw):
            raw, masks = adapter(*a, **kw)
            captured['raw_masks'] = masks
            return raw, masks

        def learned(head, *a, **kw):
            outputs = original_learned(head, *a, **kw)
            captured['outputs'] = outputs
            return outputs

        def measured(p, continuous_state, multiplier):
            captured.clear()
            replaced = sample._replace(inputs=sample.inputs._replace(continuous_state=continuous_state))
            loss, auxiliary = objective(p, replaced)
            return loss * multiplier, {'auxiliary': auxiliary, **captured}

        function = jax.jit(jax.value_and_grad(measured, argnums=(0, 1), has_aux=True))
        multiplier = jnp.asarray(args.loss_multiplier, jnp.float32)
        emit('before_lower', identity=identity, parameter_count=parameter_count, state_shape=list(state.shape),
             jax_version=jax.__version__, devices=[str(d) for d in jax.devices()],
             eligible_neural_modules=[m.__name__ for m in modules])
        original_class_method = type(context['head']).learned_outputs
        staged = args.execution == 'staged'
        if staged:
            function = native_staged_dispatcher(jax, staged_daily_functions(
                jax, context['head'], sample, context, make_shared_joint_daily_objective,
                captured, original_learned, neural_remat=args.neural_remat))
            emit('before_staged_native_ad', execution='staged', neural_remat=args.neural_remat,
                 persistent_stage_cache=str(cache))
        try:
            if not staged:
                type(context['head']).learned_outputs = learned
            with helper.candidate_process_head_patches(shared_process_head, captured_raw, assembler), neural_policy(modules, mixed=args.mode == 'candidate'):
                began = time.perf_counter()
                if staged:
                    with stage_compilation_evidence(args, helper, emit, report):
                        result = jax.block_until_ready(function(parameters, state, multiplier))
                    executable = function
                else:
                    with deadline(300, 'lower_and_backend_compile', emit):
                        lowered = function.lower(parameters, state, multiplier)
                        emit('before_backend_compile', lower_seconds=time.perf_counter() - began)
                        executable = lowered.compile()
                compile_seconds = time.perf_counter() - began
                if staged:
                    timings = []
                    for _ in range(args.repeats):
                        began = time.perf_counter()
                        result = jax.block_until_ready(executable(parameters, state, multiplier))
                        timings.append(time.perf_counter() - began)
        finally:
            type(context['head']).learned_outputs = original_class_method
        memory = None if staged else memory_sizes(executable)
        emit('compiled', compile_seconds=compile_seconds, memory=memory,
             cold_first_call_seconds=compile_seconds if staged else None,
             timing_scope='trace_compile_cache_persist_dispatch_and_first_execution' if staged else 'lower_and_backend_compile')
        if memory is not None and memory['estimated_live_bytes'] > args.memory_limit_gib * 2**30:
            raise ValueError('declared memory bound exceeded')
        if not staged:
            helper.atomic_pickle(args.output.with_suffix('.executable.pkl'), serialize(executable))
            result = jax.block_until_ready(executable(parameters, state, multiplier))
            timings = []
            for _ in range(args.repeats):
                began = time.perf_counter()
                result = jax.block_until_ready(executable(parameters, state, multiplier))
                timings.append(time.perf_counter() - began)
        (loss, diagnostics), gradients = jax.device_get(result)
        saved = {'identity': identity, 'contract': CONTRACT, 'loss': loss / args.loss_multiplier,
                 'diagnostics': diagnostics, 'gradients': jax.tree_util.tree_map(lambda x: x / args.loss_multiplier, gradients),
                 'warm_seconds': timings, 'mode': args.mode, 'loss_multiplier': args.loss_multiplier}
        saved['execution'] = args.execution
        saved['neural_remat'] = args.neural_remat
        helper.atomic_pickle(args.result, saved)
        emit('result_saved', result_artifact=str(args.result), warm_seconds=timings)
        hard = helper.hard_counts(diagnostics['auxiliary'])
        if any(hard.values()):
            raise ValueError('physical hard gate failed: ' + str(hard))
        finite = all(np.all(np.isfinite(x)) for x in jax.tree_util.tree_leaves((saved['loss'], saved['gradients'], diagnostics['outputs'])))
        if not finite:
            raise ValueError('nonfinite loss, output or native gradients')
        if not undefined_state_gradients_zero(jax, saved, context):
            raise ValueError('undefined continuous input gradient is not exact zero')
        if staged:
            device_memory = jax.local_devices()[0].memory_stats()
            emit('staged_result_validated', device_memory=device_memory)
            if device_memory is not None and device_memory.get('peak_bytes_in_use', 0) > args.memory_limit_gib * 2**30:
                raise ValueError('staged device peak exceeds declared memory bound')
        if args.mode == 'candidate':
            with args.reference_result.open('rb') as stream:
                saved_reference = pickle.load(stream)
            if saved_reference['identity'] != identity or saved_reference['mode'] != 'reference':
                raise ValueError('completed FP32 reference identity drift')
            if (saved_reference['execution'], saved_reference['neural_remat'], saved_reference['loss_multiplier']) != (args.execution, args.neural_remat, args.loss_multiplier):
                raise ValueError('reference execution/AD/scaling boundary differs')
            comparison = compare_results(jax, saved, saved_reference, context)
            emit('completed', status='screened', **comparison)
        else:
            emit('completed', status='reference_saved', hard_counts=hard,
                 undefined_input_state_gradients_exact_zero=True, all_measured_values_finite=True,
                 physical_output_dtypes={name: str(np.asarray(value).dtype) for name, value in diagnostics['outputs'].items()})
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'result', 'source-profile', 'prepared-context', 'real-daily-probe-file',
                 'candidate-module-file', 'backbone-module-file'):
        result.add_argument('--' + name, type=Path, required=True)
    result.add_argument('--reference-result', type=Path)
    result.add_argument('--mode', choices=('reference', 'candidate'), required=True)
    result.add_argument('--execution', choices=('monolithic', 'staged'), default='staged')
    result.add_argument('--stage-cache', type=Path)
    result.add_argument('--neural-remat', action='store_true')
    result.add_argument('--reuse-stage-report', type=Path)
    result.add_argument('--cache-smoke-only', action='store_true')
    result.add_argument('--process-seconds', type=int, default=600)
    result.add_argument('--final-neural-compile-seconds', type=int, default=300)
    result.add_argument('--require-gpu', action='store_true')
    result.add_argument('--repeats', type=int, default=3)
    result.add_argument('--memory-limit-gib', type=int, default=12)
    result.add_argument('--loss-multiplier', type=float, default=1.0)
    return result


if __name__ == '__main__':
    arguments = parser().parse_args()
    if arguments.loss_multiplier <= 0 or not np.isfinite(arguments.loss_multiplier):
        raise ValueError('positive finite loss multiplier required')
    if arguments.mode == 'candidate' and arguments.reference_result is None:
        raise ValueError('candidate requires completed FP32 reference')
    inputs = [arguments.source_profile, arguments.prepared_context, arguments.reference_result,
              arguments.real_daily_probe_file, arguments.candidate_module_file, arguments.backbone_module_file]
    outputs = [arguments.output, arguments.result, arguments.output.with_suffix('.executable.pkl')]
    if len({p.resolve() for p in outputs}) != len(outputs) or any(p.resolve() in {o.resolve() for o in outputs} for p in inputs if p is not None):
        raise ValueError('input/output collision')
    if any(p.exists() for p in outputs):
        raise FileExistsError('refusing to overwrite existing diagnostic output')
    run(arguments)
