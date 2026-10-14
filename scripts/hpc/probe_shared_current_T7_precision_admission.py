"""Current B1/T7 diagnostic: one native state VJP/JVP graph, scalar FD reuse.

Dry-run binds the real public objective and validates original host assets, but
never calls, traces, lowers or compiles that objective. Execute requires a fresh
external allocation approval; this module neither creates nor migrates scopes.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, nullcontext
import gzip
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import pickle
import re
import sys
import time
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
COORDINATES = (434, 2382, 2621, 200)
CAPS = dict(load=300, preflight=240, lower=300, compile=3000, calls=1380, reserve=180)
TOTAL_SECONDS = 5400
FP32_REPORT_SHA = '01f9dcc7e4ff2d61a76c8067514169a238cd081d613ae87a599bbf7ed981709c'


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('dry-run', 'execute'), required=True)
    p.add_argument('--check-approval-only', action='store_true')
    for name in ('parameters', 'prepared-context', 'training-prepared-manifest',
                 'dataset-manifest', 'saved-identity-receipt', 'state-input-metadata',
                 'turnover-overlay', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    for name in ('approval-report', 'runtime-preflight', 'fp32-reference'):
        p.add_argument('--' + name, type=Path)
    p.add_argument('--source-cache', type=Path)
    return p


def resolve_source_cache(output, configured=None):
    """Use an explicit shared cache or keep the historical output-local default."""
    return Path(configured) if configured is not None else Path(output).parent/'source-cache'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def json_value(value):
    """JSON metadata is strict; complete nonfinite values remain in raw pickles."""
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return {'nonfinite': str(value), 'finite_value': None}
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, float, int, bool)):
        return value
    return str(value)


def strict_journal_class(base):
    class StrictJournal(base):
        def emit(self, stage, **values):
            self.report.update(stage=stage, elapsed_seconds=time.monotonic()-self.started)
            self.report['stage_events'].append({'stage': stage, 'elapsed_seconds': self.report['elapsed_seconds'], **values})
            save_json(self.output, self.report)
            print(json.dumps(json_value({'stage': stage, **values}), allow_nan=False), flush=True)
    return StrictJournal


def save_raw(path, value):
    """Persist complete results before any admission or derivative gate."""
    path = Path(path)
    with path.open('xb') as f:
        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    return {'path': str(path), 'sha256': sha(path), 'size_bytes': path.stat().st_size}


def unit_direction(coordinate):
    if coordinate not in COORDINATES:
        raise ValueError('fixed current coordinate required')
    direction = np.zeros(3854, dtype=np.float64)
    direction[coordinate] = 1.0
    return direction


def compare_arrays(actual, reference):
    a, r = np.asarray(actual), np.asarray(reference)
    if a.shape != r.shape:
        return {'passed': False, 'reason': 'shape', 'actual_shape': list(a.shape), 'reference_shape': list(r.shape)}
    if a.dtype.kind in 'biu' or r.dtype.kind in 'biu':
        failed = a != r
        return {'passed': not np.any(failed), 'coordinates': a.size,
                'failed_indices': np.argwhere(failed).tolist(), 'comparison': 'exact_discrete_or_mask'}
    finite = np.isfinite(a) & np.isfinite(r)
    failed = ~finite | (np.abs(a-r) > 1e-6 + 1e-5*np.abs(r))
    return {'passed': not np.any(failed), 'coordinates': a.size,
            'failed_indices': np.argwhere(failed).tolist(), 'rtol': 1e-5, 'atol': 1e-6,
            'maximum_finite_absolute_difference': float(np.max(np.abs(a[finite]-r[finite]), initial=0.0))}


def compare_result_tree(jax, actual, reference):
    if jax.tree_util.tree_structure(actual) != jax.tree_util.tree_structure(reference):
        return {'passed': False, 'reason': 'tree_structure'}
    rows = [compare_arrays(a, r) for a, r in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(reference), strict=True)]
    return {'passed': all(r['passed'] for r in rows), 'leaves': rows}


def compiled_memory_gate(executable):
    analysis = executable.memory_analysis()
    if analysis is None:
        raise RuntimeError('compiled memory estimate unavailable; sampled runtime memory still required')
    rows = {key: int(getattr(analysis, key)) for key in ('argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes', 'alias_size_in_bytes')}
    total = rows['argument_size_in_bytes'] + rows['output_size_in_bytes'] + rows['temp_size_in_bytes'] - rows['alias_size_in_bytes']
    return {**rows, 'estimated_total_bytes': total, 'under_24_GiB': total <= 24*2**30,
            'estimate_is_not_sampled_runtime_peak': True}


def load_fp32_reference(path, *, full=False):
    """Reuse original saved evidence; no old executable or FP32 computation."""
    if path is None or sha(path) != FP32_REPORT_SHA:
        raise ValueError('exact original B1 independent-analysis JSON required')
    compact = json.loads(Path(path).read_text(encoding='utf-8'))
    if (compact['schema_version'] != 'T7_state_jvp_completion_independent_analysis_v1'
            or compact['coordinates'] != list(COORDINATES)
            or compact['tolerance'] != {'rtol': 1e-5, 'atol': 1e-6}):
        raise ValueError('original FP32 reference protocol differs')
    result = {'report': str(path), 'report_sha256': sha(path),
        'full_state_previous_validation': compact['endpoint_analysis']['B1']['state_gradient_native_self_repeat'],
        'native_jvp': {str(r['coordinate']): r['native']['actual'] for r in compact['JVP_analysis']['repeats_vs_own_first_cycle'] if r['route'] == 'B1' and r['cycle'] == 1},
        'native_vjp_projections': {str(r['coordinate']): r['reference'] for r in compact['endpoint_analysis']['B1']['four_selected_projection_native_self_repeat']},
        'full_state_loaded': False, 'full_state_comparison': 'not_performed'}
    if full:
        endpoint = Path(compact['endpoint_analysis']['B1']['paths']['begin'])
        endpoint.resolve().relative_to(Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-latent-reader-20261004').resolve())
        from scripts.hpc.shared_current_T7_precision_assets import HistoricalAssetUnpickler
        before = sha(endpoint)
        with endpoint.open('rb') as f:
            saved = HistoricalAssetUnpickler(f).load()
        if saved.get('route') != 'B1' or saved.get('phase') != 'begin':
            raise ValueError('saved original native B1 endpoint identity differs')
        gradient = np.asarray(saved['result'][1][1])
        if gradient.shape != (3854,) or gradient.dtype != np.float64 or not np.all(np.isfinite(gradient)):
            raise ValueError('original full3854 FP64 state VJP required')
        if any(float(gradient[c]) != result['native_vjp_projections'][str(c)] for c in COORDINATES):
            raise ValueError('original saved B1 endpoint disagrees with independent projection receipt')
        if sha(endpoint) != before:
            raise ValueError('read-only original endpoint changed')
        result.update(full_state_loaded=True, state_gradient=gradient.copy(),
            baseline_forward=baseline_forward_metadata(saved['result'][0]),
            endpoint={'path': str(endpoint), 'sha256': before},
            endpoint_hash_previously_pinned=False,
            endpoint_binding='canonical original report path, B1/begin ABI, full finite state and four independently saved exact native projections')
    return result


def baseline_forward_metadata(value):
    """Retain saved scalar/auxiliary structure and dtype/finite facts only."""
    def leaf(item):
        if isinstance(item, dict):
            return {str(key): leaf(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [leaf(child) for child in item]
        try:
            array = np.asarray(item)
        except Exception:
            return {'python_type': type(item).__name__, 'available': False}
        if array.dtype.kind not in 'biufc':
            if np.isscalar(item) and isinstance(item, (str, bytes)):
                return {'python_type': type(item).__name__, 'value': json_value(item)}
            return {'python_type': type(item).__name__, 'available': False}
        finite = np.isfinite(array)
        row = {'dtype': str(array.dtype), 'shape': list(array.shape),
               'finite': bool(np.all(finite)), 'finite_count': int(np.count_nonzero(finite)),
               'element_count': int(array.size)}
        if array.ndim == 0:
            row['value'] = json_value(array.item())
        return row

    if isinstance(value, (tuple, list)) and len(value) == 2:
        loss, auxiliary = value
        return {'available': True, 'loss': leaf(loss), 'auxiliary': leaf(auxiliary)}
    return {'available': False, 'saved_forward': leaf(value),
            'reason': 'saved result[0] is not a (loss, auxiliary) pair'}


def result_gates(jax, result):
    (loss, auxiliary), state_gradient, tangent = result
    bad = []
    for path, value in jax.tree_util.tree_flatten_with_path(result)[0]:
        a = np.asarray(value)
        if a.dtype.kind in 'fc' and not np.all(np.isfinite(a)):
            bad.append({'path': jax.tree_util.keystr(path), 'nonfinite_indices': np.argwhere(~np.isfinite(a)).tolist()})
    hard = {str(k): np.asarray(v).tolist() for k, v in auxiliary.get('hard', {}).items()}
    abi = np.shape(state_gradient) == (3854,) and np.asarray(state_gradient).dtype == np.float64
    primal_finite = all(np.all(np.isfinite(np.asarray(v))) for v in jax.tree_util.tree_leaves((loss, auxiliary)))
    forward_passed = bool(primal_finite and all(np.all(np.asarray(v) == 0) for v in hard.values()))
    return {'loss': float(loss), 'hard': hard, 'nonfinite': bad,
            'state_vjp_abi': abi, 'forward_passed': forward_passed,
            'passed': not bad and forward_passed and abi,
            'trajectory_state20_audited': False}


def check_approval(args, argv=None):
    if args.approval_report is None:
        raise PermissionError('fresh exact allocation approval required')
    approval = json.loads(args.approval_report.read_text(encoding='utf-8'))
    expected = {'schema_version': 'current_T7_precision_approval_v2', 'approved': True,
                'approval_origin': 'explicit_human_user', 'gpu_count': 1,
                'gpu_model': 'V100', 'slurm_cpus': 4, 'nonexclusive': True,
                'wall_seconds': TOTAL_SECONDS, 'memory_limit_gib': 24,
                'model_compile_cap': 1, 'retry': False, 'worst_case_cost_cny': 3.72,
                'working_directory': str(ROOT), 'entry_script': str(Path(__file__).resolve()),
                'entry_sha256': sha(__file__), 'output': str(args.output), 'stage_caps': CAPS}
    for key, value in expected.items():
        if type(approval.get(key)) is not type(value) or approval[key] != value:
            raise PermissionError('allocation approval differs: ' + key)
    command_args = [arg for arg in (sys.argv[1:] if argv is None else argv) if arg != '--check-approval-only']
    if approval.get('argv') != command_args or not approval.get('approval_note'):
        raise PermissionError('exact command and human approval note required')
    if os.environ.get('SLURM_CPUS_PER_TASK') != '4':
        raise RuntimeError('four actual Slurm CPUs required')
    approval['_observed_cpu_affinity'] = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None
    approval['_CPU_admission_limit'] = 'The owner validates actual allocated TRES=4 CPUs; wide OS affinity is recorded, not treated as additional allocated CPUs. LLVM parallelism is4; no affinity subset is invented.'
    return approval


def bind_real_objective(assets, args, stack):
    """Bind unchanged source-reuse, guarded physics and public loss owners."""
    from scripts.hpc import probe_shared_latent_reader_T7_pair as pair
    from scripts.hpc.shared_current_T7_precision_guard import CURRENT_MODULES, PACKAGE, current_neural_fp64_context
    from research.daily_coarse_graining.daily_operator import shared_process_head, shared_recurrent_training, shared_backbone_execution
    from jax_orchidee.stomate import carbon_kernels
    if sha(args.turnover_overlay) != pair.CORRECTED_SHA:
        raise ValueError('original guarded turnover bytes required')
    actual_turnover = hashlib.sha256(inspect.getsource(carbon_kernels.turnover_leaf_age_fall).replace('\r\n', '\n').encode()).hexdigest()
    if actual_turnover == pair.CORRECTED_SHA:
        physics = {'sha256': actual_turnover, 'path': str(args.turnover_overlay), 'mode': 'identical_corrected_function_already_loaded'}
    elif actual_turnover == pair.BASE_SHA:
        physics = stack.enter_context(pair.install_turnover_overlay(carbon_kernels, args.turnover_overlay, pair.BASE_SHA))
    else:
        raise ValueError('loaded turnover differs from both admitted base and corrected source')
    transformer = importlib.import_module(PACKAGE + 'shared_transformer')
    stack.enter_context(pair.candidate_backbone_patches(shared_process_head,
        transformer.shared_transformer_raw_outputs,
        shared_backbone_execution.assemble_shared_transformer_context_with_source_reuse))
    modules = tuple(importlib.import_module(PACKAGE + name) for name in CURRENT_MODULES)
    mathematical = assets['manifest_metadata']['saved_B1_mathematical_identity']
    source_rows = {}
    for key, module_name in (('source_reuse_assembler_sha256', 'shared_backbone_execution'),
                             ('source_reuse_projection_sha256', 'shared_backbone_projection_reuse')):
        module = importlib.import_module(PACKAGE + module_name)
        digest = sha(module.__file__)
        if digest != mathematical['parameter_provenance']['identity'][key]:
            raise ValueError('source-reuse execution source changed: ' + module_name)
        source_rows[module_name] = {'path': module.__file__, 'sha256': digest}
    for historical_path, digest in mathematical['sources']['siblings'].items():
        module_name = Path(historical_path).stem
        module = importlib.import_module(PACKAGE + module_name)
        actual = sha(module.__file__)
        if actual != digest:
            raise ValueError('historical sibling source changed: ' + module_name)
        source_rows[module_name] = {'path': module.__file__, 'sha256': actual}
    _, precision = stack.enter_context(current_neural_fp64_context(modules, enabled=True))
    functions = {}
    saved = assets['manifest_metadata']['saved_B1_mathematical_identity']['scientific_functions']
    for name in ('make_shared_recurrent_objective', 'make_canonical_step', 'make_rollout_loss'):
        f = getattr(shared_recurrent_training, name)
        digest = hashlib.sha256(inspect.getsource(f).replace('\r\n', '\n').encode()).hexdigest()
        if digest != saved[name]['sha256']:
            raise ValueError('public scientific function changed: ' + name)
        functions[name] = {'sha256': digest, 'source': inspect.getsourcefile(f)}
    context = assets['context']
    _, objective = shared_recurrent_training.make_shared_recurrent_objective(
        assets['candidate_head'], context['template'], context['state_scales'], context['science_scales'])
    from jax_orchidee.sechiba import diffuco
    source_rows['diffuco_host_mask_cache'] = {'path': diffuco.__file__, 'sha256': sha(diffuco.__file__),
        'function_sha256': hashlib.sha256(inspect.getsource(diffuco._diffuco_pft_column_mask).replace('\r\n', '\n').encode()).hexdigest(),
        'difference_from_historical_base': 'Function-preserving immutable NumPy Boolean PFT mask cache replaces cached JAX tracer. Whole-file historical SHA equality is not claimed.'}
    return objective, {'precision': precision, 'physics': physics, 'public_functions': functions, 'execution_sources': source_rows,
                       'loss': '(flux+stock)/9', 'native_B1_scale': 1.0}


def widened_inputs(jax, assets):
    from scripts.hpc.shared_current_T7_precision_guard import exact_array_widen
    def widen(value):
        return exact_array_widen(np.asarray(value))
    params = jax.tree_util.tree_map(widen, assets['parameters'])
    case = jax.tree_util.tree_map(widen, assets['case'])
    return (params, *case)


def run_calls(jax, executable, arguments, plans, output, journal):
    """Same executable for native repeats and all FD scalar-forward samples."""
    from scripts.hpc.shared_current_T7_fd_guard import analyze_fd
    def call(label, state, direction):
        journal.report['full_model_call_attempts'] = journal.report.get('full_model_call_attempts', 0) + 1
        raw = jax.device_get(executable(arguments[0], state, *arguments[2:], direction))
        journal.report['full_model_calls'] = journal.report.get('full_model_calls', 0) + 1
        receipt = save_raw(output.with_name(output.stem + '.' + label + '.raw.pkl'), raw)
        gates = result_gates(jax, raw)
        journal.report['results'].append({'label': label, 'raw': receipt, 'gates': gates})
        journal.emit('raw_saved', label=label)
        return raw, gates
    baseline, native, fd_results = None, {}, {}
    for cycle in range(4):
        for coordinate in COORDINATES:
            raw, gates = call(f'native.{cycle}.{coordinate}', arguments[1], unit_direction(coordinate))
            if baseline is None:
                baseline = raw
            first = native.get(coordinate, [{}])[0]
            own_vjp = float(raw[1][coordinate])
            native.setdefault(coordinate, []).append({'jvp': float(raw[2]), 'vjp': own_vjp,
                'native_jvp_vs_vjp': compare_arrays(raw[2], own_vjp),
                'native_repeat': compare_arrays(raw[2], first.get('jvp', float(raw[2]))),
                'complete_raw_repeat': compare_result_tree(jax, raw[:2], baseline[:2]), 'gates': gates})
    for coordinate in COORDINATES:
        plan = plans[str(coordinate)]
        samples = {}
        for i, h in enumerate(plan['steps']):
            samples[h] = {}
            for side in (('plus', 'minus') if plan['mode'] == 'central' else ('plus',)):
                state = np.asarray(arguments[1]).copy()
                state[coordinate] += h if side == 'plus' else -h
                try:
                    raw, gates = call(f'fd.{coordinate}.{i}.{side}', state, unit_direction(coordinate))
                    samples[h][side] = {'loss': float(raw[0][0]), 'branch': None,
                        'reported_mask_change': raw[0][1].get('declared_mask_change'), 'hard_passed': gates['forward_passed'],
                        'AD_passed': gates['passed']}
                except (TimeoutError, MemoryError):
                    raise
                except Exception as error:
                    samples[h][side] = {'loss': float('nan'), 'error': repr(error)}
                    journal.report.setdefault('call_exceptions', []).append({'coordinate': coordinate, 'step': h, 'side': side, 'error': repr(error)})
                    journal.emit('call_exception_saved')
        try:
            fd_results[str(coordinate)] = analyze_fd(coordinate, certificate=plan['certificate'],
                raw_value=plan['raw_value'], baseline_loss=float(baseline[0][0]), baseline_branch=None,
                steps=plan['steps'], samples=samples, mode=plan['mode'], ad_value=native[coordinate][0]['jvp'],
                state20_raw=float(arguments[1][20]), state20_defined=False)
        except ValueError as error:
            fd_results[str(coordinate)] = {'fd_status': 'inconclusive', 'reason': str(error),
                'target': coordinate, 'state20_status': 'strict_nonfinite_or_undefined_failure'}
        fd_results[str(coordinate)]['sample_records'] = samples
        fd_results[str(coordinate)]['baseline_forward_passed'] = result_gates(jax, baseline)['forward_passed']
        if not fd_results[str(coordinate)]['baseline_forward_passed']:
            fd_results[str(coordinate)].update(fd_status='inconclusive', reason='baseline public forward/physical gate failed')
        journal.report['fd'] = fd_results
        journal.emit('direction_complete', coordinate=coordinate)
    return baseline, native, fd_results


def main():
    args = parser().parse_args()
    if args.output.exists():
        raise FileExistsError('preserve prior diagnostic output')
    approval = check_approval(args) if args.stage == 'execute' else None
    if args.stage == 'dry-run':
        os.environ['JAX_PLATFORMS'] = 'cpu'
    if args.check_approval_only:
        if args.stage != 'execute':
            raise ValueError('approval-only requires execute stage')
        print(json.dumps({'approval_checked': True, 'GPU_executed': False}))
        return 0
    if args.stage == 'execute':
        if os.environ.get('ORCJAX_CURRENT_T7_SUPERVISED') != '1':
            raise PermissionError('execute requires the owned current-T7 supervisor before GPU bootstrap')
        from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend
        jax, _ = initialize_gpu_backend()
    else:
        import jax
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_enable_compilation_cache', False)
    from scripts.hpc.shared_current_T7_precision_assets import load_assets
    from scripts.hpc.shared_current_T7_fd_guard import make_combined
    from scripts.hpc.shared_current_T7_fd_guard import predeclared_fd_plan
    from scripts.hpc.probe_shared_leaf_full_state_cuda import Journal
    journal = strict_journal_class(Journal)(args.output, sum(CAPS.values()) - CAPS['reserve']) if approval else None
    try:
        with ExitStack() as stack:
            from scripts.hpc.shared_current_T7_precision_guard import historical_expert_source_context
            source_cache = resolve_source_cache(args.output, args.source_cache)
            historical_expert = stack.enter_context(historical_expert_source_context(ROOT, source_cache))
            with journal.deadline(CAPS['load'], 'asset_load') if journal else nullcontext():
                assets = load_assets(jax, parameters=args.parameters, prepared_context=args.prepared_context,
                    training_prepared_manifest=args.training_prepared_manifest, dataset_manifest=args.dataset_manifest,
                    saved_identity_receipt=args.saved_identity_receipt)
                fp32 = load_fp32_reference(args.fp32_reference, full=True) if approval else None
            metadata = json.loads(args.state_input_metadata.read_text(encoding='utf-8'))
            rows = {int(r['coordinate']): r for r in metadata['rows'] if r.get('sample') == 3 and r.get('artifact_id') == 299}
            plans = {str(c): predeclared_fd_plan(c, float(assets['case'][0][c]), float(rows[c]['continuous_scale'])) for c in COORDINATES}
            from scripts.hpc.shared_current_T7_precision_guard import widen_context
            prepared_context, context_widening = widen_context({'context': assets['context'], 'candidate_head': assets['candidate_head']})
            assets.update(prepared_context)
            objective, binding = bind_real_objective(assets, args, stack)
            arguments = widened_inputs(jax, assets)
            combined = make_combined(jax, objective)
            report = {'schema_version': 'current_T7_precision_preparation_v2', 'status': 'real_callable_prepared_unexecuted',
                'asset_identity': assets['manifest_metadata'], 'original_assets': assets['original_assets'],
                'binding': binding, 'historical_expert_source': historical_expert,
                'historical_expert_source_cache': str(source_cache.resolve()),
                'context_widening': context_widening, 'fd_plans': plans,
                'entry_sha256': sha(__file__), 'stage_caps': CAPS, 'total_seconds': TOTAL_SECONDS,
                'full_model_calls': 0, 'full_model_traces': 0, 'full_model_compiles': 0,
                'full_model_call_attempts': 0, 'full_model_trace_attempts': 0, 'full_model_compile_attempts': 0,
                'state20': {'raw': str(np.asarray(arguments[1])[20]), 'strict_failure': True, 'new_trajectory_audited': False},
                'full_FP32_state_comparison_performed': False, 'production_adoption': False}
            import jax_orchidee
            report['source_checkout'] = {'root': str(ROOT), 'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'jax_orchidee_import': jax_orchidee.__file__, 'diagnostic_entry': str(Path(__file__).resolve()),
                'current_checkout_distinct_from_historical_asset_source': True}
            if args.fp32_reference is not None:
                report['FP32_reference'] = load_fp32_reference(args.fp32_reference)
            if args.stage == 'dry-run':
                save_json(args.output, report)
                print(json.dumps({'status': report['status'], 'output': str(args.output)}))
                return 0
            journal.report.update(report, status='running', approval_sha256=sha(args.approval_report))
            if args.runtime_preflight is None:
                raise ValueError('actual deployed runtime preflight required')
            preflight = json.loads(args.runtime_preflight.read_text())
            if preflight.get('passed') is not True or preflight.get('actual_backend') != 'gpu' or preflight.get('jax_version') != '0.4.38' or len(jax.devices()) != 1 or jax.default_backend() != 'gpu':
                raise RuntimeError('single GPU/deployed runtime preflight not admitted')
            options = {'xla_gpu_enable_llvm_module_compilation_parallelism': True, 'xla_gpu_force_compilation_parallelism': 4}
            with journal.deadline(CAPS['lower'], 'lower'):
                journal.report['full_model_trace_attempts'] = 1
                journal.emit('sole_lower_started')
                lowered = jax.jit(combined).lower(*arguments, unit_direction(434))
                journal.report['full_model_traces'] = 1
                text = lowered.as_text()
                hlo = args.output.with_suffix('.lowered.mlir.gz')
                with gzip.open(hlo, 'xt', encoding='utf-8') as f:
                    f.write(text)
                journal.report['lowered'] = {'path': str(hlo), 'sha256': sha(hlo)}
                if re.search(r'(?:\bf(?:16|32)\b|\bbf16\b)', text):
                    raise RuntimeError('residual low precision in diagnostic lowering; no compile')
            with journal.deadline(CAPS['compile'], 'compile'):
                journal.report['full_model_compile_attempts'] = 1
                journal.emit('sole_compile_started')
                executable = lowered.compile(compiler_options=options)
                journal.report['full_model_compiles'] = 1
                from jax.experimental.serialize_executable import serialize
                serialized = serialize(executable)
                journal.report['executable'] = save_raw(args.output.with_suffix('.executable.pkl'), serialized)
                optimized = executable.as_text()
                optimized_path = args.output.with_suffix('.optimized-hlo.gz')
                with gzip.open(optimized_path, 'xt', encoding='utf-8') as f:
                    f.write(optimized)
                journal.report['optimized_hlo'] = {'path': str(optimized_path), 'sha256': sha(optimized_path)}
                journal.emit('executable_saved_before_admission')
                journal.report['compiled_memory'] = compiled_memory_gate(executable)
                journal.emit('compiled_memory_saved')
                if not journal.report['compiled_memory']['under_24_GiB']:
                    raise MemoryError('compiled memory estimate exceeds24GiB; no calls')
                if re.search(r'(?:\bf(?:16|32)\b|\bbf16\b)', optimized):
                    raise RuntimeError('residual low precision in saved optimized executable; no calls')
            with journal.deadline(CAPS['calls'], 'native_and_fd_calls'):
                baseline, native, fd = run_calls(jax, executable, arguments, plans, args.output, journal)
                full_comparison = compare_arrays(baseline[1], fp32.pop('state_gradient'))
                for c in COORDINATES:
                    direction_fd = fd[str(c)]
                    if direction_fd.get('fd_status') == 'converged':
                        value = direction_fd['fd_value']
                        if plans[str(c)]['mode'] == 'central':
                            direction_fd['FD_vs_native_FP64_VJP'] = compare_arrays(native[c][0]['vjp'], value)
                            direction_fd['FD_vs_saved_FP32_JVP'] = compare_arrays(fp32['native_jvp'][str(c)], value)
                            direction_fd['FD_vs_saved_FP32_VJP'] = compare_arrays(fp32['native_vjp_projections'][str(c)], value)
                        else:
                            direction_fd['one_sided_differences_only'] = {'FP64_VJP': float(native[c][0]['vjp']-value),
                                'saved_FP32_JVP': float(fp32['native_jvp'][str(c)]-value),
                                'saved_FP32_VJP': float(fp32['native_vjp_projections'][str(c)]-value),
                                'two_sided_equivalence_claimed': False}
                journal.report.update(FP32_reference=fp32, full_FP32_state_comparison_performed=True,
                    full_state_FP64_vs_saved_FP32=full_comparison,
                    selected_native_FP64_vs_FP32={str(c): {
                        'JVP': compare_arrays(native[c][0]['jvp'], fp32['native_jvp'][str(c)]),
                        'VJP': compare_arrays(native[c][0]['vjp'], fp32['native_vjp_projections'][str(c)])} for c in COORDINATES})
                journal.report.update(native=native, fd=fd, status='completed_with_separate_state20_strict_failure')
                journal.emit('diagnostic_complete')
            return 3
    except Exception as error:
        if journal:
            journal.report.update(status='execution_failed', error=repr(error))
            journal.emit('failure_saved')
        raise
    finally:
        if journal:
            journal.timer.cancel()


if __name__ == '__main__':
    raise SystemExit(main())
