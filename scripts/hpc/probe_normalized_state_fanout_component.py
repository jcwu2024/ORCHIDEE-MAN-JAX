"""Bounded tokenizer fan-out diagnostic; preparation never imports JAX.

The scalar uses declared synthetic summary cotangents, not public-loss
cotangents. Actual original/candidate execution requires a new GPU window.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time

ROOT = Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX')
SOURCE_CHECKOUT = ROOT / 'runtime/checkouts/shared-split-width-v1'
PREPARED_RELATIVE = Path('runtime/outputs/diagnostics/shared-bank-adjoint-proved-binary-T7-20261003/next-component-preparation')
COORDINATES = (33, 199, 428)
RTOL, ATOL = 1e-5, 1e-6
MEMORY_LIMIT = 24 * 2**30


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + '\n')
    os.replace(temporary, path)


def write_result(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


class Budget:
    def __init__(self, seconds, destination):
        self.started = time.perf_counter()
        self.seconds = seconds
        self.destination = destination

    @contextmanager
    def stage(self, cap, label, emit):
        remaining = self.seconds - (time.perf_counter() - self.started)
        allowed = min(cap, remaining)
        if allowed <= 0:
            raise TimeoutError('original component deadline exhausted')
        emit('stage_start', stage_label=label, effective_cap_seconds=allowed)
        # A separate stdlib process enforces native compiler/device-call limits.
        watchdog_code = (
            'import json,os,signal,sys,time;from pathlib import Path;'
            'time.sleep(float(sys.argv[1]));'
            'Path(sys.argv[3]).write_text(json.dumps({"stage":sys.argv[4],"exit":124}));'
            'os.kill(int(sys.argv[2]),signal.SIGTERM);'
            'time.sleep(10);os.kill(int(sys.argv[2]),signal.SIGKILL)'
        )
        watchdog = subprocess.Popen([sys.executable, '-B', '-c', watchdog_code,
            str(allowed), str(os.getpid()), str(self.destination / (label + '.timeout.json')), label])
        began = time.perf_counter()
        try:
            yield
        finally:
            watchdog.terminate()
            try:
                watchdog.wait(timeout=2)
            except subprocess.TimeoutExpired:
                watchdog.kill()
                watchdog.wait(timeout=2)
            emit('stage_end', stage_label=label, stage_seconds=time.perf_counter() - began)


def compare(np, actual, reference):
    if np.shape(actual) != np.shape(reference) or np.asarray(actual).dtype != np.asarray(reference).dtype:
        raise ValueError('component comparison ABI differs')
    actual, reference = np.asarray(actual), np.asarray(reference)
    if actual.dtype == np.bool_:
        failed = int(np.count_nonzero(actual != reference))
        maximum = 0.0
    else:
        error = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
        allowed = ATOL + RTOL * np.abs(reference.astype(np.float64))
        failed = int(np.count_nonzero(~(np.isfinite(actual) & np.isfinite(reference) & (error <= allowed))))
        maximum = float(np.max(error, initial=0))
    return dict(passed=failed == 0, failed_coordinates=failed, coordinates=actual.size, max_abs_error=maximum)


def comparisons(np, actual, reference):
    return {
        'loss': compare(np, actual[0], reference[0]),
        'summaries': [compare(np, a, b) for a, b in zip(actual[1], reference[1], strict=True)],
        'masks': [compare(np, a, b) for a, b in zip(actual[2], reference[2], strict=True)],
        'all_raw_state_gradients': compare(np, actual[3], reference[3]),
        'native_directional_JVPs': compare(np, actual[4], reference[4]),
    }


def all_passed(value):
    if isinstance(value, dict):
        if 'passed' in value:
            return value['passed']
        return all(all_passed(child) for child in value.values())
    return all(all_passed(child) for child in value)


def hard_admission(jax, np, result, defined, baseline, memory):
    if not isinstance(result, tuple) or len(result) != 5:
        raise ValueError('expected loss/summaries/masks/full-state-VJP/three-JVP ABI')
    if len(result[1]) != 8 or len(result[2]) != 8:
        raise ValueError('expected field summary and all seven axis summaries/masks')
    if np.shape(result[0]) != () or np.shape(result[3]) != (3854,) or np.asarray(result[3]).dtype != np.float64 or np.shape(result[4]) != (3,):
        raise ValueError('component loss/state derivative/JVP ABI differs')
    for index, (summary, mask) in enumerate(zip(result[1], result[2], strict=True)):
        rows = 257 if index == 0 else 140
        if np.shape(summary) != (rows, 4) or np.asarray(summary).dtype != np.float32 or np.shape(mask) != (rows,) or np.asarray(mask).dtype != np.bool_:
            raise ValueError('component summary/mask ABI differs')
    finite = all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree_util.tree_leaves(result))
    undefined_zero = bool(np.all(np.asarray(result[3])[~defined] == 0))
    masks_equal = baseline is None or all(np.array_equal(a, b) for a, b in zip(result[2], baseline[2], strict=True))
    memory_passed = 'peak_bytes_in_use' in memory and memory['peak_bytes_in_use'] <= MEMORY_LIMIT
    return dict(passed=bool(finite and undefined_zero and masks_equal and memory_passed),
        finite=bool(finite), undefined_gradient_exact_zero=undefined_zero,
        undefined_coordinates=int(np.count_nonzero(~defined)), summary_masks_unchanged=masks_equal,
        observed_memory_passed=memory_passed, observed_whole_process_memory=memory)


def directional_checks(np, result):
    rows = []
    for index, coordinate in enumerate(COORDINATES):
        vjp, jvp = float(result[3][coordinate]), float(result[4][index])
        error = abs(vjp - jvp)
        allowance = ATOL + RTOL * abs(jvp)
        rows.append(dict(coordinate=coordinate, VJP=vjp, native_JVP=jvp,
            absolute_error=error, allowance=allowance, passed=error <= allowance))
    return dict(passed=all(row['passed'] for row in rows), rows=rows,
        role='component synthetic cotangents only; no public-loss or reference waiver')


@contextmanager
def backend_guard(compiler, report, emit, arm=None):
    """Count native backend attempts and prohibit fallback/replay compilation."""
    saved = compiler.backend_compile

    def guarded(*args, **kwargs):
        if arm == 'setup':
            report['setup_backend_attempts'] += 1
            emit('setup_helper_backend_compile_attempt', setup_backend_attempts=report['setup_backend_attempts'])
            return saved(*args, **kwargs)
        if arm is None:
            report['forbidden_backend_attempts'] += 1
            emit('forbidden_backend_compile', forbidden_backend_attempts=report['forbidden_backend_attempts'])
            raise RuntimeError('component values must reuse the two compiled executables')
        report['backend_attempts'][arm] += 1
        emit('native_backend_compile_attempt', arm=arm, backend_attempts=report['backend_attempts'])
        if report['backend_attempts'][arm] > 1:
            raise RuntimeError('second component backend compile/fallback is forbidden')
        return saved(*args, **kwargs)

    compiler.backend_compile = guarded
    try:
        yield
    finally:
        compiler.backend_compile = saved


def capture_optimized_IR(executable, destination, label):
    """Preserve actual optimized modules without recompilation or truncation."""
    from jaxlib import xla_client
    options = xla_client._xla.HloPrintOptions.short_parsable()
    options.print_large_constants = False
    options.print_backend_config = False
    options.print_program_shape = False
    options.print_metadata = True
    options.print_operand_shape = False
    options.print_operand_names = True
    options.print_percent = True
    options.print_ids = True
    rows = []
    for index, module in enumerate(executable.runtime_executable().hlo_modules()):
        raw = module.to_string(options).encode()
        path = destination / f'{label}.compact-{index}.hlo'
        path.write_bytes(raw)
        rows.append(dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
            truncated=False, scope_metadata_printed=True, operand_names_printed=True))
    if not rows:
        raise ValueError('component executable returned no optimized HLO modules')
    manifest = dict(arm=label, modules=rows, mechanism_status='pending_CPU_actual_IR_review',
        large_constants_printed=False, backend_config_printed=False, new_compilations=0,
        claim_limit='actual optimized component IR capture only; no inferred F64 fan-in proof')
    write_json(destination / (label + '.actual-IR.json'), manifest)
    return manifest


def component_plan(policy, source_root, prepared_dir):
    plan = policy.prepare_plan(source_root, output_dir=prepared_dir)
    bridge = json.loads((prepared_dir / 'component-input-bridge.json').read_text())
    if any(bridge[key] is not True for key in ('actual500_and_original_JVP_and_existing_FP64_identity_equal',
            'raw3854_saved_original_fingerprint_equal', 'raw_defined_and_first_day_active_PFT_exact')):
        raise ValueError('prepared component identity/raw/source-mask bridge failed')
    if any(bridge['exact_saved_array_bridges'][key]['exact_array_equal'] is not True
           for key in ('field_pool', 'axis_pool', 'tail_mask')):
        raise ValueError('prepared field/axis/tail source-mask bridge failed')
    plan['actual_input_bridge'] = bridge
    return plan


def run(args, emit, report, budget):
    with budget.stage(90, 'component_inputs', emit):
        import normalized_state_fanout_component as policy
        plan = component_plan(policy, SOURCE_CHECKOUT, args.prepared_dir)
        report['component_plan'] = plan
        preflight = json.loads(args.runtime_preflight.read_text())
        if preflight.get('passed') is not True or preflight.get('actual_backend') != 'gpu':
            raise ValueError('new actual-runtime daily AD preflight did not pass')
        from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend
        jax, devices = initialize_gpu_backend()
        from jax._src import compiler
        import numpy as np
        if jax.__version__ != '0.4.38' or len(devices) != 1:
            raise ValueError('canonical single-GPU JAX runtime differs')
        with backend_guard(compiler, report, emit, 'setup'):
            inputs = policy.load_component_inputs(args.root, args.prepared_dir)
            raw, mean, scale = (np.asarray(inputs[key]) for key in ('raw', 'mean', 'scale'))
            defined = np.asarray(inputs['defined'])
            if raw.shape != (3854,) or raw.dtype != np.float64 or mean.shape != raw.shape or scale.shape != raw.shape or defined.shape != raw.shape or defined.dtype != np.bool_:
                raise ValueError('declared dynamic component input ABI differs')
            if np.count_nonzero(~defined) != 21:
                raise ValueError('source-defined undefined state scope differs')
            arguments = jax.device_put(tuple(inputs[key] for key in ('raw', 'mean', 'scale', 'tail_mask', 'defined', 'cotangents')))
            identity = inputs['identity']
            eager_arms = {label: policy.make_component(inputs['field_pool'], inputs['axis_pool'],
                inputs['threshold'], candidate=candidate) for label, candidate in (('original', False), ('candidate', True))}
            emit('inputs_ready', identity=identity, parameter_gradients=False,
                component_bundle_bound=True,
                cotangent_role='artificial sin(arange+1+consumer_index*0.125), computed F64/stored F32; not public-loss cotangents')
    compiled = {}
    first = {}
    timings = {'original': [], 'candidate': []}
    checks = []
    supplements = []
    hard_checks = []

    def evaluate(label, stage, baseline=None):
        with budget.stage(15, stage, emit), backend_guard(compiler, report, emit):
            began = time.perf_counter()
            result = jax.device_get(jax.block_until_ready(compiled[label](*arguments)))
            seconds = time.perf_counter() - began
        result_path = args.output_dir / (stage + '.pkl')
        write_result(result_path, dict(identity=identity, arm=label, result=result, seconds=seconds,
            hard_admission='pending'))
        try:
            hard = hard_admission(jax, np, result, defined, baseline, devices[0].memory_stats() or {})
        except ValueError as error:
            emit('completed', status='hard_admission_failed', arm=label, result_file=stage + '.pkl',
                hard_ABI_error=str(error), stop_reason='first ABI hard failure saved; no following arm or pair')
            return None
        if not hard['passed']:
            write_result(result_path, dict(identity=identity, arm=label, result=result, seconds=seconds,
                hard_admission=hard))
            emit('completed', status='hard_admission_failed', arm=label, result_file=stage + '.pkl',
                hard_admission=hard, stop_reason='first hard failure saved; no following arm or pair')
            return None
        supplement = directional_checks(np, result)
        hard_checks.append(dict(arm=label, result_file=stage + '.pkl', **hard))
        write_result(result_path, dict(identity=identity, arm=label,
            result=result, hard_admission=hard, native_directional_checks=supplement, seconds=seconds))
        emit('component_result_saved', arm=label, result_file=stage + '.pkl', hard_admission=hard,
            native_directional_checks=supplement, seconds=seconds)
        supplements.append(dict(arm=label, result_file=stage + '.pkl', **supplement))
        return result, seconds

    for label, candidate in (('original', False), ('candidate', True)):
        # Record before lower: an ordinary interruption must not silently retry.
        report['compile_attempts'][label] += 1
        emit('component_compile_attempt', arm=label, compile_attempts=report['compile_attempts'])
        with budget.stage(60, label + '_single_compile', emit), backend_guard(compiler, report, emit, label):
            compiled[label] = jax.jit(eager_arms[label]).lower(*arguments).compile()
        with budget.stage(15, label + '_optimized_IR_capture', emit), backend_guard(compiler, report, emit):
            captured = capture_optimized_IR(compiled[label], args.output_dir, label)
        emit('actual_optimized_IR_saved', arm=label, actual_IR=captured)
        evaluated = evaluate(label, label + '_first', first.get('original'))
        if evaluated is None:
            return
        first[label] = evaluated[0]
    initial = comparisons(np, first['candidate'], first['original'])
    emit('first_pair_saved', initial_comparisons=initial)
    # Strict numerical errors remain diagnostic and do not suppress three pairs.
    for repeat in range(3):
        order = ('original', 'candidate') if repeat % 2 == 0 else ('candidate', 'original')
        results = {}
        for label in order:
            evaluated = evaluate(label, f'{label}_pair_{repeat}', first['original'])
            if evaluated is None:
                return
            results[label], seconds = evaluated
            timings[label].append(seconds)
        comparison = comparisons(np, results['candidate'], results['original'])
        drift = {label: comparisons(np, results[label], first[label]) for label in order}
        checks.append(dict(repeat=repeat, order=order, comparisons=comparison, self_drift=drift))
        write_json(args.output_dir / f'pair-{repeat}.json', checks[-1])
        emit('matched_pair_saved', pairs=checks, matched_seconds=timings)
    strict = all_passed(initial) and all(all_passed(row['comparisons']) and all_passed(row['self_drift']) for row in checks)
    directional = all(row['passed'] for row in supplements)
    forward_checks = [{'loss': initial['loss'], 'summaries': initial['summaries']}] + [
        {'loss': row['comparisons']['loss'], 'summaries': row['comparisons']['summaries']} for row in checks]
    cross_vjp_checks = [initial['all_raw_state_gradients']] + [row['comparisons']['all_raw_state_gradients'] for row in checks]
    emit('completed', status='component_numerical_passed_IR_review_pending' if strict and directional else 'component_numerical_discrepancy',
        strict_numerical_passed=strict, native_directional_passed=directional,
        initial_comparisons=initial, matched_pairs=checks, matched_seconds=timings,
        classifications=dict(
            structure=dict(status='pending_CPU_actual_IR_review', mechanism_claim=False),
            forward=dict(passed=all(all_passed(row) for row in forward_checks), comparisons=forward_checks),
            finite_masks_undefined_zero=dict(passed=all(row['passed'] for row in hard_checks), checks=hard_checks),
            strict_crossarm_VJP=dict(passed=all(row['passed'] for row in cross_vjp_checks), comparisons=cross_vjp_checks,
                reference_waiver=False, reference_adoption=False),
            native_JVP_consistency=dict(passed=directional, checks=supplements,
                public_loss_cotangents=False, crossarm_strict_failure_cannot_be_waived=True),
            timing=dict(status='component_diagnostic_no_speed_threshold', matched_seconds=timings,
                full_update_speed_claim=False)),
        median_component_seconds={label: float(np.median(values)) for label, values in timings.items()},
        component_speed_claim=False, mechanism_status='pending_CPU_actual_IR_review', mechanism_claim=False,
        public_loss_verified=False, full_optimizer_updates=0,
        reference_adoption=False, production_adoption=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--prepared-dir', type=Path)
    parser.add_argument('--runtime-preflight', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--process-seconds', type=float, default=350)
    parser.add_argument('--cli-check-only', action='store_true')
    parser.add_argument('--prepare-ast-only', action='store_true')
    args = parser.parse_args(argv)
    if not 0 < args.process_seconds <= 350:
        parser.error('component process cap must be 1..350 seconds')
    args.prepared_dir = args.prepared_dir or args.root / (PREPARED_RELATIVE if args.root == ROOT else Path(*PREPARED_RELATIVE.parts[1:]))
    if args.cli_check_only:
        if any(name == 'jax' or name.startswith('jax.') for name in sys.modules):
            raise RuntimeError('pure CLI validation imported JAX')
        print(json.dumps(dict(status='component_CLI_checked', JAX_imports=0, GPU_calls=0,
            Teacher_reads=0, model_values=0, compilation_attempts=0, process_cap=args.process_seconds)))
        return 0
    if args.prepare_ast_only:
        import normalized_state_fanout_component as policy
        source_root = SOURCE_CHECKOUT if args.root == ROOT else args.root
        plan = component_plan(policy, source_root, args.prepared_dir)
        if any(name == 'jax' or name.startswith('jax.') for name in sys.modules):
            raise RuntimeError('pure interface/AST preparation imported JAX')
        args.prepared_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.prepared_dir / 'component-plan.json', plan)
        print(json.dumps(dict(status='component_AST_plan_checked', plan=plan, JAX_imports=0,
            GPU_calls=0, Teacher_reads=0, component_calls=0, compilation_attempts=0), indent=2))
        return 0
    if args.root != ROOT or args.output_dir is None or args.runtime_preflight is None:
        parser.error('actual component route requires canonical root, output-dir and runtime-preflight')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', stage_events=[], compile_attempts={'original': 0, 'candidate': 0},
        backend_attempts={'original': 0, 'candidate': 0}, forbidden_backend_attempts=0,
        setup_backend_attempts=0,
        mode='normalized_state_fanout_synthetic_component', coordinates=list(COORDINATES),
        rtol=RTOL, atol=ATOL, training_updates=0, reference_adoption=False, production_adoption=False)
    budget = Budget(args.process_seconds, args.output_dir)

    def emit(event, **fields):
        report['stage_events'].append(dict(event=event, elapsed_seconds=time.perf_counter() - budget.started,
            UTC_seconds=time.time(), **fields))
        report.update(fields)
        write_json(args.output_dir / 'model.json', report)
        print('component stage=' + event, flush=True)

    try:
        emit('starting')
        run(args, emit, report, budget)
    except BaseException as error:
        emit('exception', status='component_runtime_failed', error=repr(error))
        raise
    return 0 if report['status'] == 'component_numerical_passed_IR_review_pending' else 1


if __name__ == '__main__':
    raise SystemExit(main())
