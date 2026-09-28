"""Cache-only observed replay of the unchanged actual row0 native daily AD.

Existing owners: probe_shared_neural_precision::staged_daily_functions:159-212,
native_staged_dispatcher:215-224; probe_shared_union_integrated.py:174-238.
Instrument compiler and executable dispatch, never replace differentiated stages.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import cProfile
import importlib.util
import io
import os
import pickle
from pathlib import Path
import pstats
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


def operation_counts(module):
    counts = Counter()
    def visit(operation):
        counts[operation.name] += 1
        for region in operation.regions:
            for block in region.blocks:
                for child in block.operations:
                    visit(child.operation)
    visit(module.operation)
    return dict(counts)


def permitted_helper(name, counts, signature, *, initialization):
    if any(stage in name for stage in ('neural_stage', 'transition_stage', 'loss_stage')):
        return False
    if any(counts.get(operation, 0) for operation in
           ('stablehlo.dot_general', 'stablehlo.while', 'stablehlo.convolution', 'func.call')):
        return False
    # Permit only explicit scalar/elementwise AD glue, not a new model fragment.
    allowed = {'builtin.module', 'func.func', 'func.return', 'stablehlo.constant',
               'stablehlo.add', 'stablehlo.subtract', 'stablehlo.multiply', 'stablehlo.divide',
               'stablehlo.negate', 'stablehlo.broadcast_in_dim', 'stablehlo.reshape',
               'stablehlo.convert', 'stablehlo.iota', 'stablehlo.compare', 'stablehlo.select',
               'stablehlo.maximum', 'stablehlo.minimum', 'stablehlo.is_finite',
               'stablehlo.and', 'stablehlo.or', 'stablehlo.not', 'stablehlo.reduce',
               'stablehlo.return', 'stablehlo.abs', 'stablehlo.power', 'stablehlo.slice',
               'stablehlo.concatenate', 'stablehlo.dynamic_slice', 'stablehlo.scatter'}
    byte_limit = 2**20 if initialization else 32 * 2**20
    return (set(counts) <= allowed and sum(counts.values()) <= 128 and
            signature['result_nominal_bytes'] <= byte_limit and
            signature['argument_nominal_bytes'] <= 128 * 2**20)


@contextmanager
def observed_runtime(jax, probe, report, emit, args):
    from jax._src import compiler
    from jax._src.interpreters import pxla
    original_lookup = compiler.compile_or_get_cached
    backend_name = 'backend_compile' if hasattr(compiler, 'backend_compile') else 'backend_compile_and_load'
    original_backend = getattr(compiler, backend_name)
    original_execute = pxla.ExecuteReplicated.__call__
    active = None
    report.update(compiler_events=[], dispatch_events=[], backend_entrypoint=backend_name)

    def backend(*arguments, **keywords):
        if active is None or not active['permitted_tiny_helper']:
            emit('forbidden_backend', status='stopped_before_full_backend')
            raise RuntimeError('cache-only replay forbids model or unknown backend compilation')
        active['backend_compile_calls'] += 1
        began = time.perf_counter()
        try:
            return original_backend(*arguments, **keywords)
        finally:
            active['backend_seconds'] += time.perf_counter() - began

    def lookup(*arguments, **keywords):
        nonlocal active
        module = arguments[1] if len(arguments) > 1 else keywords['computation']
        name = str(module.operation.attributes['sym_name']).strip('"')
        signature = probe.module_signature(module)
        stage = any(part in name for part in ('neural_stage', 'transition_stage', 'loss_stage'))
        counts = {} if stage else operation_counts(module)
        event = {'ordinal': len(report['compiler_events']), 'phase': report['phase'], 'module': name,
                 'main_signature': signature, 'operation_counts': counts, 'is_model_stage': stage,
                 'permitted_tiny_helper': permitted_helper(name, counts, signature,
                                                          initialization=report['phase'] == 'initialization'),
                 'backend_compile_calls': 0, 'backend_seconds': 0.0, 'status': 'lookup_started'}
        report['compiler_events'].append(event)
        emit('executable_lookup')
        previous = active
        active = event
        began = time.perf_counter()
        try:
            with probe.deadline(args.cache_seconds if stage else 30, 'cache_lookup_' + name, emit):
                result = original_lookup(*arguments, **keywords)
            event.update(status='completed', cache_hit=event['backend_compile_calls'] == 0)
            return result
        except Exception as error:
            event.update(status='failed', error=str(error))
            raise
        finally:
            event['seconds'] = time.perf_counter() - began
            active = previous
            emit('executable_lookup_returned')

    def execute(self, *arguments):
        began = time.perf_counter()
        output = original_execute(self, *arguments)
        returned = time.perf_counter()
        synchronized = report.get('synchronize_dispatch', False)
        if synchronized:
            jax.block_until_ready(output)
        report['dispatch_events'].append({'phase': report['phase'], 'module': self.name,
            'submission_seconds': returned - began,
            'synchronized_seconds': time.perf_counter() - began if synchronized else None,
            'synchronization_changes_schedule': synchronized})
        return output

    compiler.compile_or_get_cached = lookup
    setattr(compiler, backend_name, backend)
    pxla.ExecuteReplicated.__call__ = execute
    try:
        yield
    finally:
        compiler.compile_or_get_cached = original_lookup
        setattr(compiler, backend_name, original_backend)
        pxla.ExecuteReplicated.__call__ = original_execute


def dispatch_totals(events, phase):
    grouped = {}
    for event in events:
        if event['phase'] != phase:
            continue
        row = grouped.setdefault(event['module'], {'calls': 0, 'submission_seconds': 0.0,
                                                  'synchronized_seconds': 0.0})
        row['calls'] += 1
        row['submission_seconds'] += event['submission_seconds']
        row['synchronized_seconds'] += event['synchronized_seconds'] or 0.0
    return grouped


def run(args):
    started = time.perf_counter()
    os.environ['JAX_COMPILATION_CACHE_DIR'] = str(args.stage_cache)
    os.environ['JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS'] = '0'
    os.environ['JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES'] = '0'
    probe = load_script('union_replay_stage_owners', args.neural_probe_file)
    args.execution = 'staged'
    probe.configure_stage_cache(args)
    helper = probe.load_helper(args.real_daily_probe_file)
    import jax
    import jax.numpy as jnp
    report = {'status': 'running', 'phase': 'initialization', 'scope': 'unchanged_row0_union_cache_replay',
              'training_updates': 0, 'production_adoption': False, 'calls': [],
              'process_limit_seconds': args.process_seconds, 'call_limit_seconds': args.call_seconds,
              'strict_rtol': 1e-5, 'strict_atol': 1e-6, 'new_model_backend_compiles': 0,
              'python_dispatch_hook_covers_cpp_fastpath': False}
    def emit(stage, **fields):
        report.update(stage=stage, elapsed_seconds=time.perf_counter() - started, **fields)
        helper.atomic_json(args.output, report)
        if stage not in ('executable_lookup', 'executable_lookup_returned'):
            print(stage, report['phase'], flush=True)

    with probe.deadline(args.process_seconds, 'whole_replay', emit), observed_runtime(jax, probe, report, emit, args):
        from research.daily_coarse_graining.daily_operator import shared_process_head
        from research.daily_coarse_graining.daily_operator.shared_recurrent_training import make_shared_joint_daily_objective
        from scripts.hpc.run_shared_learning import load_learning_checkpoint
        if len(jax.local_devices()) != 1 or jax.local_devices()[0].platform != 'gpu':
            raise ValueError('requires one visible GPU')
        _profile, reference = helper.trusted_profile(args)
        current = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        if current != reference['source_git_head']:
            raise ValueError('immutable source identity differs')
        with args.prepared_context.open('rb') as stream:
            context = pickle.load(stream)
        with args.saved_result.open('rb') as stream:
            expected = pickle.load(stream)
        identity = helper.context_identity(reference, helper.initial_checkpoint(reference), 742)
        if context['identity'] != identity or expected['identity'] != identity or expected['route'] != 'union':
            raise ValueError('saved reference or prepared sample differs')
        if expected['diagnostic_sample'] != 'prepared_micro4_row0':
            raise ValueError('saved sample is not row0')
        checkpoint = load_learning_checkpoint(Path(identity['initial_checkpoint']), expected_method=reference['method'])
        parameters = jax.device_put(checkpoint['parameters'])
        sample = jax.tree_util.tree_map(lambda value: jnp.asarray(value)[0], context['daily4'])
        state = sample.inputs.continuous_state
        multiplier = jnp.asarray(1.0, jnp.float32)
        adapter = helper.load_staged_adapter(args.candidate_module_file)
        assembler = helper.load_backbone_assembler(args.backbone_module_file)
        captured = {}
        def captured_raw(*arguments, **keywords):
            output, masks = adapter(*arguments, **keywords, dynamic_projection_execution='union')
            captured['raw_masks'] = masks
            return output, masks
        stages = probe.staged_daily_functions(jax, context['head'], sample, context,
            make_shared_joint_daily_objective, captured, type(context['head']).learned_outputs, neural_remat=True)
        function = probe.native_staged_dispatcher(jax, stages)
        emit('initialized', identity=identity, jax_version=jax.__version__,
             devices=[str(device) for device in jax.devices()], skipped_reference_binary_replay=True,
             skipped_unchanged_runtime_preflight=True)
        with helper.candidate_process_head_patches(shared_process_head, captured_raw, assembler):
            for phase in ('cold_cache', 'warm_1', 'warm_2', 'warm_3_profiled', 'synchronized_dispatch'):
                report['phase'] = phase
                report['synchronize_dispatch'] = phase == 'synchronized_dispatch'
                event_start = len(report['compiler_events'])
                profile = cProfile.Profile() if phase == 'warm_3_profiled' else None
                emit('call_start')
                trace_started = False
                if profile is not None:
                    trace_path = args.output.with_suffix('.jax-profile')
                    try:
                        jax.profiler.start_trace(str(trace_path), create_perfetto_trace=True)
                        trace_started = True
                        report['jax_profiler'] = {'status': 'started', 'path': str(trace_path)}
                    except Exception as error:
                        report['jax_profiler'] = {'status': 'unavailable', 'error': str(error)}
                with probe.deadline(args.call_seconds, phase, emit):
                    if profile is not None:
                        profile.enable()
                    began = time.perf_counter()
                    result = function(parameters, state, multiplier)
                    submitted = time.perf_counter()
                    jax.block_until_ready(result)
                    completed = time.perf_counter()
                    if profile is not None:
                        profile.disable()
                if trace_started:
                    jax.profiler.stop_trace()
                    report['jax_profiler'].update(status='saved', files=[str(path) for path in trace_path.rglob('*') if path.is_file()])
                began_copy = time.perf_counter()
                (loss, diagnostics), gradients = jax.device_get(result)
                host_copy_seconds = time.perf_counter() - began_copy
                comparisons = {name: helper.compare_trees(jax, actual, wanted) for name, actual, wanted in
                    (('loss', loss, expected['loss']), ('parameters', gradients[0], expected['gradients'][0]),
                     ('continuous_state', gradients[1], expected['gradients'][1]),
                     ('outputs', diagnostics['outputs'], expected['diagnostics']['outputs']),
                     ('raw_masks', diagnostics['raw_masks'], expected['diagnostics']['raw_masks']),
                     ('auxiliary', diagnostics['auxiliary'], expected['diagnostics']['auxiliary']))}
                # Boolean masks require exact equality in addition to numeric parity.
                exact_masks = all(np.array_equal(a, b) for a, b in zip(
                    jax.tree_util.tree_leaves(diagnostics['raw_masks']),
                    jax.tree_util.tree_leaves(expected['diagnostics']['raw_masks']), strict=True))
                saved = {'identity': identity, 'phase': phase, 'loss': loss, 'diagnostics': diagnostics,
                         'gradients': gradients, 'route': 'union', 'instrumentation_changes_AD': False}
                if phase == 'synchronized_dispatch':
                    helper.atomic_pickle(args.output.with_suffix('.result.pkl'), saved)
                row = {'phase': phase, 'call_wall_seconds': completed - began,
                       'python_call_return_seconds': submitted - began,
                       'final_ready_wait_seconds': completed - submitted, 'host_copy_seconds': host_copy_seconds,
                       'new_lookup_events': len(report['compiler_events']) - event_start,
                       'comparisons': comparisons, 'raw_masks_exact': exact_masks,
                       'hard_counts': helper.hard_counts(diagnostics['auxiliary']),
                       'dispatch_totals': dispatch_totals(report['dispatch_events'], phase),
                       'altered_synchronized_schedule': report['synchronize_dispatch'],
                       'synchronization_only_covers_python_intercepted_dispatch': report['synchronize_dispatch']}
                report['calls'].append(row)
                if profile is not None:
                    profile.dump_stats(str(args.output.with_suffix('.profile.pstats')))
                    text = io.StringIO()
                    pstats.Stats(profile, stream=text).strip_dirs().sort_stats('cumulative').print_stats(35)
                    report['cprofile_text'] = text.getvalue()
                    # Native profiler output is retained in JSON and binary; no extra text file.
                emit('call_completed')
                if not all(value['passed'] for value in comparisons.values()) or not exact_masks or any(row['hard_counts'].values()):
                    raise ValueError('instrumented replay numerical/mask/physical parity failed')
        report['phase'] = 'completed'
        model_events = [event for event in report['compiler_events'] if event['is_model_stage']]
        if len(model_events) != 6 or not all(event.get('cache_hit') for event in model_events):
            raise ValueError('requires exactly six saved model-unit cache hits')
        emit('completed', status='cache_replay_parity_passed', model_stage_cache_hits=len(model_events),
             tiny_helper_backend_compiles=sum(event['backend_compile_calls'] for event in report['compiler_events']),
             device_memory=jax.local_devices()[0].memory_stats())
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('output', 'stage-cache', 'source-profile', 'prepared-context', 'saved-result',
                 'real-daily-probe-file', 'neural-probe-file', 'candidate-module-file', 'backbone-module-file'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--process-seconds', type=int, default=480)
    parser.add_argument('--call-seconds', type=int, default=100)
    parser.add_argument('--cache-seconds', type=int, default=120)
    args = parser.parse_args()
    if any(path.exists() for path in (args.output, args.output.with_suffix('.result.pkl'), args.output.with_suffix('.profile.pstats'))):
        raise FileExistsError('refusing to overwrite replay evidence')
    run(args)
