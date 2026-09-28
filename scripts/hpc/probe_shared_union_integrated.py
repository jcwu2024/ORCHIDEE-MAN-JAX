"""Bounded FP32 union daily loss/full native AD with saved field-binary reference.

Owners: probe_shared_neural_precision::staged_daily_functions:159-212 and
native_staged_dispatcher:215-224; shared_recurrent_training.py:243-277;
shared_training.py:128-265; shared_daily_microbatch.py:78-102.
No independent cotangent, physical formula, optimizer or recurrent run.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import os
import pickle
from pathlib import Path
import subprocess
import sys
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


def repeat_row0(jax, daily4):
    def repeat(value):
        value = np.asarray(value)
        if not value.shape or value.shape[0] != 4:
            raise ValueError('every prepared sample leaf must have four leading rows')
        result = np.repeat(value[:1], 4, axis=0)
        if result.shape != value.shape or result.dtype != value.dtype:
            raise ValueError('repeated-row shape/dtype changed')
        return result
    repeated = jax.tree_util.tree_map(repeat, daily4)
    if jax.tree_util.tree_structure(repeated) != jax.tree_util.tree_structure(daily4):
        raise ValueError('complete sample pytree changed')
    return repeated


@contextmanager
def forbid_graph_lookup(jax):
    from jax._src import compiler
    original = compiler.compile_or_get_cached
    def forbidden(*arguments, **keywords):
        raise RuntimeError('saved field replay attempted a new graph executable lookup')
    compiler.compile_or_get_cached = forbidden
    try:
        yield
    finally:
        compiler.compile_or_get_cached = original


def tree_summary(jax, tree):
    leaves = jax.tree_util.tree_leaves(tree)
    coordinates = nonfinite = nonzero = 0
    norm_squared = maximum = 0.0
    for value in leaves:
        value = np.asarray(value)
        coordinates += value.size
        nonfinite += int(np.count_nonzero(~np.isfinite(value)))
        nonzero += int(np.count_nonzero(value))
        finite = value[np.isfinite(value)].astype(np.float64)
        norm_squared += float(np.sum(finite * finite))
        maximum = max(maximum, float(np.max(np.abs(finite), initial=0)))
    return {'leaves': len(leaves), 'coordinates': coordinates, 'nonfinite': nonfinite,
            'nonzero': nonzero, 'norm': float(np.sqrt(norm_squared)), 'max_absolute': maximum}


def mask_summary(jax, masks):
    result = []
    for path, value in jax.tree_util.tree_flatten_with_path(masks)[0]:
        value = np.asarray(value)
        if value.dtype != np.bool_:
            raise ValueError('diagnostic defined mask is not boolean')
        result.append({'path': jax.tree_util.keystr(path), 'shape': list(value.shape),
                       'defined': int(np.count_nonzero(value)), 'coordinates': value.size})
    return result


def run(args):
    started = time.perf_counter()
    # Configure cache before importing helpers, JAX or physical constant tables.
    os.environ['JAX_COMPILATION_CACHE_DIR'] = str(args.stage_cache)
    os.environ['JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS'] = '0'
    os.environ['JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES'] = '0'
    probe = load_script('union_native_stage_owners', args.neural_probe_file)
    args.execution = 'staged'
    probe.configure_stage_cache(args)
    helper = probe.load_helper(args.real_daily_probe_file)
    import jax
    import jax.numpy as jnp
    from jax.experimental.serialize_executable import deserialize_and_load
    from research.daily_coarse_graining.daily_operator import shared_process_head
    from research.daily_coarse_graining.daily_operator.shared_recurrent_training import make_shared_joint_daily_objective
    from scripts.hpc.run_shared_learning import load_learning_checkpoint

    report = {'status': 'running', 'scope': 'actual_artifact742_row0_union_canonical_daily_full_native_AD',
              'production_adoption': False, 'scientific_acceptance': False, 'training_updates': 0,
              'precision': 'unchanged_FP32_neural_FP64_physical', 'mode': 'union',
              'loss_multiplier': 1.0, 'neural_remat': True, 'state_gradient_reference': None,
              'strict_rtol': 1e-5, 'strict_atol': 1e-6, 'prior_component_failed_coordinates': 27,
              'process_limit_seconds': args.process_seconds, 'reference_is_timing_comparator': False}
    def emit(stage, **fields):
        report.update(stage=stage, elapsed_seconds=time.perf_counter() - started, **fields)
        helper.atomic_json(args.output, report)
        print(stage, flush=True)

    with probe.deadline(args.process_seconds, 'whole_process', emit):
        if len(jax.local_devices()) != 1 or jax.local_devices()[0].platform != 'gpu':
            raise ValueError('requires exactly one visible GPU')
        with probe.deadline(60, 'tiny_staged_runtime', emit):
            emit('tiny_staged_runtime_passed', runtime_gate=probe.staged_runtime_check(jax, neural_remat=True))
        profile, reference = helper.trusted_profile(args)
        observed = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        if observed != reference['source_git_head']:
            raise ValueError('immutable source checkout differs')
        with args.prepared_context.open('rb') as stream:
            context = pickle.load(stream)
        identity = helper.context_identity(reference, helper.initial_checkpoint(reference), 742)
        if context['identity'] != identity:
            raise ValueError('prepared input identity differs')
        expected_binary = Path(profile['executables']['daily']['executable']).resolve()
        if Path(context['daily_executable']).resolve() != expected_binary:
            raise ValueError('saved canonical binary identity differs')
        checkpoint = load_learning_checkpoint(Path(identity['initial_checkpoint']), expected_method=reference['method'])
        parameters = jax.device_put(checkpoint['parameters'])
        repeated = repeat_row0(jax, context['daily4'])
        reference_identity = identity | {'diagnostic_sample': 'row0_repeated_four_times_complete_sample_pytree',
                                         'serialized_field_binary': str(expected_binary)}
        emit('field_binary_reference_start', identity=identity, jax_version=jax.__version__,
             devices=[str(value) for value in jax.devices()], reference_identity=reference_identity)
        with probe.deadline(120, 'saved_field_binary_replay', emit), forbid_graph_lookup(jax):
            with expected_binary.open('rb') as stream:
                serialized = pickle.load(stream)
            supplied_tree = jax.tree_util.tree_structure(((parameters, repeated), {}))
            if serialized[1] != supplied_tree:
                raise ValueError('saved binary dynamic input treedef differs')
            began = time.perf_counter()
            field_binary = deserialize_and_load(*serialized)
            load_seconds = time.perf_counter() - began
            del serialized
            device_repeated = jax.device_put(repeated)
            began = time.perf_counter()
            baseline = jax.device_get(jax.block_until_ready(field_binary(parameters, device_repeated)))
            reference_execution_seconds = time.perf_counter() - began
            del field_binary, device_repeated
        hard = helper.hard_counts(baseline[0][1])
        if any(hard.values()) or tree_summary(jax, (baseline[0][0], baseline[1]))['nonfinite']:
            raise ValueError('saved field numerical/physical reference failed')
        if tree_summary(jax, baseline[1])['coordinates'] != 64_884_481:
            raise ValueError('saved field parameter gradient size differs')
        helper.atomic_pickle(args.output.with_suffix('.field-reference.pkl'),
            {'identity': reference_identity, 'result': baseline, 'new_model_graph_lookups': 0,
             'reference_state_gradients': None})
        emit('field_reference_saved', reference_loss=float(np.asarray(baseline[0][0])),
             reference_hard_counts=hard, reference_deserialize_seconds=load_seconds,
             reference_execution_seconds=reference_execution_seconds,
             repeated_sample_leaves=len(jax.tree_util.tree_leaves(repeated)),
             serialized_input_tree_matches=True, reference_new_model_graph_lookups=0)
        sample = jax.tree_util.tree_map(lambda value: jnp.asarray(value)[0], context['daily4'])
        state = sample.inputs.continuous_state
        if sum(np.size(value) for value in jax.tree_util.tree_leaves(parameters)) != 64_884_481 or state.shape != (3854,):
            raise ValueError('full parameter/state shape changed')
        adapter = helper.load_staged_adapter(args.candidate_module_file)
        assembler = helper.load_backbone_assembler(args.backbone_module_file)
        captured = {}
        def captured_raw(*arguments, **keywords):
            outputs, masks = adapter(*arguments, **keywords, dynamic_projection_execution='union')
            captured['raw_masks'] = masks
            return outputs, masks
        stages = probe.staged_daily_functions(jax, context['head'], sample, context,
            make_shared_joint_daily_objective, captured, type(context['head']).learned_outputs, neural_remat=True)
        function = probe.native_staged_dispatcher(jax, stages)
        args.mode = 'union'
        args.neural_remat = True
        args.loss_multiplier = 1.0
        began = time.perf_counter()
        emit('union_native_AD_start')
        with helper.candidate_process_head_patches(shared_process_head, captured_raw, assembler):
            with probe.stage_compilation_evidence(args, helper, emit, report):
                result = jax.block_until_ready(function(parameters, state, jnp.asarray(1.0, jnp.float32)))
            first_call_seconds = time.perf_counter() - began
            times = []
            for _ in range(args.repeats):
                began = time.perf_counter()
                result = jax.block_until_ready(function(parameters, state, jnp.asarray(1.0, jnp.float32)))
                times.append(time.perf_counter() - began)
        (loss, diagnostics), gradients = jax.device_get(result)
        saved = {'identity': identity, 'diagnostic_sample': 'prepared_micro4_row0', 'loss': loss,
                 'diagnostics': diagnostics, 'gradients': gradients, 'route': 'union',
                 'warm_seconds': times, 'native_ad': 'all_parameters_and_continuous_state'}
        helper.atomic_pickle(args.result, saved)
        emit('union_result_saved', result_artifact=str(args.result),
             cold_trace_compile_cache_persist_dispatch_execute_seconds=first_call_seconds, warm_seconds=times)
        summaries = {name: tree_summary(jax, tree) for name, tree in
                     (('parameters', gradients[0]), ('continuous_state', gradients[1]),
                      ('outputs', diagnostics['outputs']), ('loss', loss))}
        hard = helper.hard_counts(diagnostics['auxiliary'])
        undefined_zero = probe.undefined_state_gradients_zero(jax, saved, context)
        comparison = {'loss': helper.compare_trees(jax, loss, baseline[0][0]),
                      'all_parameter_gradients': helper.compare_trees(jax, gradients[0], baseline[1])}
        mean_auxiliary = {name: helper.compare_trees(jax, diagnostics['auxiliary'][name], baseline[0][1][name])
                          for name in ('process_day', 'process_labels', 'state_change_anchor',
                                       'flux', 'stock', 'joint_daily_contribution')}
        parameter_groups = {name: probe.vector_metrics(jax, value, probe.parameter_groups(baseline[1])[name])
                            for name, value in probe.parameter_groups(gradients[0]).items()}
        masks = {'raw': mask_summary(jax, diagnostics['raw_masks']),
                 'predicted_defined': mask_summary(jax, diagnostics['auxiliary']['state_diagnostics']['predicted_defined']),
                 'expected_defined': mask_summary(jax, diagnostics['auxiliary']['state_diagnostics']['expected_defined'])}
        shape_ok = summaries['parameters']['coordinates'] == 64_884_481 and summaries['continuous_state']['coordinates'] == 3854
        valid = shape_ok and not any(hard.values()) and undefined_zero and not any(row['nonfinite'] for row in summaries.values())
        memory = jax.local_devices()[0].memory_stats()
        if memory is not None and memory.get('peak_bytes_in_use', 0) > args.memory_limit_gib * 2**30:
            valid = False
        strict_pass = all(row['passed'] for row in comparison.values())
        emit('completed', status='bounded_validity_passed' if valid else 'bounded_validity_failed',
             measured_values=summaries, hard_counts=hard, masks=masks,
             undefined_input_state_gradients_exact_zero=undefined_zero, device_memory=memory,
             comparisons=comparison, parameter_group_comparisons=parameter_groups,
             mean_auxiliary_comparisons=mean_auxiliary,
             strict_loss_parameter_reference_passed=strict_pass, state_gradient_parity='unknown_no_reference',
             full_output_mask_parity='unknown_saved_binary_has_only_scalar_auxiliary',
             production_adoption=False, numerical_full_gradient_acceptance=False)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'result', 'stage-cache', 'source-profile', 'prepared-context',
                 'real-daily-probe-file', 'neural-probe-file', 'candidate-module-file',
                 'backbone-module-file', 'reuse-stage-report'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--process-seconds', type=int, default=1000)
    parser.add_argument('--final-neural-compile-seconds', type=int, default=600)
    parser.add_argument('--memory-limit-gib', type=int, default=24)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    outputs = (args.output, args.result, args.output.with_suffix('.field-reference.pkl'))
    if len({path.resolve() for path in outputs}) != len(outputs):
        raise ValueError('output collision')
    if any(path.exists() for path in outputs):
        raise FileExistsError('refusing to overwrite diagnostic outputs')
    run(args)
