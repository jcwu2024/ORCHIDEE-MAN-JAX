"""One native original-FP32 public T7 consumer-cotangent capture; no training."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import pickle
import sys
import threading
import time
from types import SimpleNamespace
import public_state_summary_cotangents_T7 as policy

ROOT = Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX')
MEMORY_LIMIT = 24 * 2**30


class Budget:
    def __init__(self, seconds):
        self.started = time.perf_counter()
        self.limit = seconds

    @contextmanager
    def stage(self, cap, label, emit):
        allowed = min(cap, self.limit - (time.perf_counter() - self.started))
        if allowed <= 0:
            raise TimeoutError('original dispatch allowance exhausted')
        emit('stage_start', stage_label=label, effective_timeout_seconds=allowed)
        def expire():
            try:
                emit('timeout', status='timeout', stage_label=label)
            finally:
                os._exit(124)
        timer = threading.Timer(allowed, expire)
        timer.daemon = True
        timer.start()
        try:
            yield
        finally:
            timer.cancel()


def synthetic_interface_check():
    """Tiny CPU native scan check: distinct days/consumers and recurrence AD."""
    import jax
    import jax.numpy as jnp
    from research.daily_coarse_graining.daily_operator.rollout_loss import make_rollout_loss
    class Seam:
        @contextmanager
        def bind_day(self, delta):
            self.delta = delta
            yield
    seam = Seam()
    weights = jnp.arange(1, 9, dtype=jnp.float32)
    def step(p, state, discrete, day):
        change = jnp.sum(seam.delta[:, 33] * weights)
        state = p * state + day + change
        return state, discrete, {'value': state}
    rollout = make_rollout_loss(jax, jnp, step, lambda *args: 0.)
    objective = rollout.bind_checkpointed_loss_with_aux(
        lambda p, initial, observations, targets, context: (jnp.sum(observations['value']), {'original': jnp.asarray(0.)}))
    capture = policy.bind_rollout(rollout, objective, seam)
    delta = jnp.zeros(policy.DELTA_SHAPE, jnp.float32)
    result = jax.jit(jax.value_and_grad(capture, argnums=(0, 1, 6), has_aux=True))(
        jnp.asarray(2.), jnp.asarray([1.]), {}, jnp.arange(7.)[:, None], {}, {}, delta)
    actual = result[1][2][:, :, 33]
    expected = jnp.asarray([127, 63, 31, 15, 7, 3, 1])[:, None] * weights[None, :]
    if not bool(jnp.all(actual == expected)) or not bool(jnp.all(result[1][2][:, :, 199] == 0)):
        raise ValueError('dynamic day/consumer native cotangent binding failed')
    return dict(status='passed', backend=jax.default_backend(), delta_shape=list(delta.shape),
                day_weights=[127,63,31,15,7,3,1], consumer_weights=list(range(1,9)),
                parameter_and_state_gradients=True, public_model_evaluations=0, Teacher_reads=0)


def run(args, emit, report):
    # Existing asset helpers and model/JAX imports are actual-mode only.
    with report['budget'].stage(300, 'setup_assets_original_binary_replay', emit):
        import hashlib
        import numpy as np
        from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend
        jax, devices = initialize_gpu_backend()
        from jax.experimental.serialize_executable import deserialize_and_load
        from scripts.hpc.run_shared_learning import load_learning_checkpoint
        from scripts.hpc.run_shared_transformer_admission import memory_sizes
        from research.daily_coarse_graining.daily_operator.shared_recurrent_training import make_shared_recurrent_objective
        from research.daily_coarse_graining.daily_operator import shared_process_head
        import probe_shared_forward_reverse as forward
        import probe_shared_bank_adjoint_T7 as assets
        from probe_shared_normalized_keys_recurrent import validate_inputs
        from probe_shared_command_buffer_loops import forbid_component_compile
        if jax.__version__ != '0.4.38' or len(devices) != 1:
            raise ValueError('requires one canonical JAX0.4.38 GPU')
        preflight = json.loads(args.runtime_preflight.read_text())
        if not preflight.get('passed') or preflight.get('status') != 'passed' or preflight.get('actual_backend') != 'gpu' or preflight.get('jax_version') != jax.__version__:
            raise ValueError('actual deployed preflight differs or failed')
        metadata = assets.inspect_assets(args.root)
        files = assets.paths(args.root)
        source = SimpleNamespace(**{key: files[key] for key in ('source_profile', 'input_bundle', 'prepared_context',
            'daily_admission_report', 'seven_admission_report', 'thirty_admission_report', 'support_owner_file')})
        _profile, ref, bundle, context, checkpoint_path, admissions = forward._assets(source)
        checkpoint = load_learning_checkpoint(checkpoint_path, expected_method=ref['method'])
        host_parameters = checkpoint['parameters']
        if checkpoint['completed_updates'] != 0 or len(jax.tree_util.tree_leaves(host_parameters)) != 300 or sum(np.size(x) for x in jax.tree_util.tree_leaves(host_parameters)) != 64884481:
            raise ValueError('original update0/full-network scope differs')
        case = bundle['payloads'][0]['cases'][7]
        state = np.asarray(case[0])
        if len(case) != 5 or state.shape != (3854,) or state.dtype != np.float64 or np.shape(case.days[0].continuous_state)[0] != 7:
            raise ValueError('original T7 ABI differs')
        if len(context['head'].layout.expert_names) != 7:
            raise ValueError('requires seven original expert-axis consumers')
        defined = np.isfinite(state) & (np.abs(state) < .5e20)
        mask_inventory = assets.load_mask_inventory(files['mask_inventory_source'])
        mask_evidence, input_masks = mask_inventory(dict(case=case, layout=context['head'].layout,
            normalization=context['head'].input_normalization, state_scales=context['state_scales'],
            template=context['template'], contract=context['head'].contract))
        if len(input_masks) != 112 or np.count_nonzero(~defined) != 21:
            raise ValueError('source mask scope differs')
        assets.write_pickle(args.output_dir / 'capture-input-snapshot.pkl', dict(case=case,
            normalization=context['head'].input_normalization, layout=context['head'].layout,
            zero_delta=np.zeros(policy.DELTA_SHAPE, dtype=np.float32), checkpoint=str(checkpoint_path),
            completed_updates=0, prepared_inputs_reused=True))
        assets.write_json(args.output_dir / 'asset-metadata.json', metadata)
        sources = args.output_dir / 'original-source-snapshot'
        sources.mkdir(exist_ok=False)
        for name, source_path in (('decoder.py', files['decoder']), ('backbone.py', files['backbone']),
            ('strong_baseline_architecture.py', Path(sys.modules['research.daily_coarse_graining.daily_operator.strong_baseline_architecture'].__file__))):
            (sources / name).write_bytes(source_path.read_bytes())
        parameters = jax.device_put(host_parameters)
        arguments = (parameters, *jax.device_put(case))
        identity = dict(source_head=ref['source_git_head'], source_profile_sha256=hashlib.sha256(files['source_profile'].read_bytes()).hexdigest(),
            method=ref['method'], artifact_id=int(bundle['artifact_ids'][0]), horizon=7, checkpoint=str(checkpoint_path.resolve()),
            jax_version=jax.__version__, admitted_executable=admissions['T7']['executable'],
            candidate_decoder_sha256=hashlib.sha256(files['decoder'].read_bytes()).hexdigest(),
            candidate_backbone_sha256=hashlib.sha256(files['backbone'].read_bytes()).hexdigest())
        if identity != metadata['reference_identity']:
            raise ValueError('original public full-AD identity differs')
        with files['reference_result'].open('rb') as stream:
            saved = pickle.load(stream)
        if saved['identity'] != identity:
            raise ValueError('original saved result identity differs')
        with forbid_component_compile(True):
            with files['reference_executable'].open('rb') as stream:
                serialized = pickle.load(stream)
            original = deserialize_and_load(*serialized)
            validate_inputs(jax, serialized, original, arguments)
            reference_memory = memory_sizes(original)
            if reference_memory['estimated_live_bytes'] > MEMORY_LIMIT:
                raise MemoryError('original estimated memory exceeds24GiB')
            baseline = jax.device_get(jax.block_until_ready(original(*arguments)))
        baseline_hard = assets.hard_admission(jax, baseline, defined, host_parameters)
        saved_bridge = assets.compare_full(jax, baseline, saved['result'])
        assets.write_pickle(args.output_dir / 'original-replay.pkl', dict(identity=identity, result=baseline,
            hard_admission=baseline_hard, saved_result_comparison=saved_bridge))
        assets.write_pickle(args.output_dir / 'source-masks.pkl', dict(identity=identity, defined=defined,
            input_boolean_masks=input_masks, mask_evidence=mask_evidence))
        if not baseline_hard['passed']:
            emit('completed', status='original_hard_admission_failed', hard_admission=baseline_hard)
            return
        owner = forward._load_module(args.root / 'runtime/transfer/shared-combined-real-daily-20260925/probe_shared_expert_real_daily.py',
                                     'scripts.hpc.probe_shared_expert_real_daily')
        adapter = owner.load_staged_adapter(files['decoder'])
        backbone = owner.load_backbone_assembler(files['backbone'])
        seam = policy.SummarySeam(backbone.__globals__['assemble_strong_baseline_context'])
        rollout, objective = make_shared_recurrent_objective(context['head'], context['template'], context['state_scales'], context['science_scales'])
        captured = policy.bind_rollout(rollout, objective, seam)
        delta = jax.device_put(np.zeros(policy.DELTA_SHAPE, dtype=np.float32))
        capture_arguments = (*arguments, delta)
        backward = jax.jit(jax.value_and_grad(captured, argnums=(0, 1, 6), has_aux=True))
        emit('reference_ready', identity=identity, original_saved_result_comparison=saved_bridge,
             mask_evidence=mask_evidence, seam=seam.evidence, reference_memory=reference_memory,
             auxiliary_leaves=len(jax.tree_util.tree_leaves(baseline[0][1])))
    with report['budget'].stage(3100, 'sole_capture_lower_and_backend', emit):
        emit('capture_compile_start', capture_compile_attempts=1)
        with owner.candidate_process_head_patches(shared_process_head, adapter, backbone), policy.install(backbone, seam):
            began = time.perf_counter()
            lowered = backward.lower(*capture_arguments)
            emit('capture_lowered', lower_seconds=time.perf_counter() - began)
            began = time.perf_counter()
            executable = lowered.compile()
            emit('capture_compiled', backend_seconds=time.perf_counter() - began)
    memory = memory_sizes(executable)
    if memory['estimated_live_bytes'] > MEMORY_LIMIT:
        raise MemoryError('capture estimated memory exceeds24GiB')
    with report['budget'].stage(90, 'capture_calls_and_raw_save', emit), forbid_component_compile(True):
        calls_started = time.perf_counter()
        emit('capture_call_start', capture_calls=1)
        began = time.perf_counter()
        result = jax.device_get(jax.block_until_ready(executable(*capture_arguments)))
        seconds = time.perf_counter() - began
        # Preserve the actual result before any hard, bridge, or interpretation gate.
        assets.write_pickle(args.output_dir / 'public-cotangents.result.pkl', dict(identity=identity, result=result,
            consumers=list(policy.CONSUMERS), delta_shape=list(policy.DELTA_SHAPE), delta_dtype='float32',
            zero_delta=True, original_parameter_state_AD_preserved=True, seam=seam.evidence))
        original_scope = (result[0], result[1][:2])
        hard = assets.hard_admission(jax, original_scope, defined, host_parameters)
        cotangents = np.asarray(result[1][2])
        cotangent_ABI = cotangents.shape == policy.DELTA_SHAPE and cotangents.dtype == np.float32
        cotangent_finite = bool(np.isfinite(cotangents).all())
        bridge = assets.compare_full(jax, original_scope, baseline)
        stats = devices[0].memory_stats() or {}
        observed_memory = 'peak_bytes_in_use' in stats and stats['peak_bytes_in_use'] <= MEMORY_LIMIT
        first_passed = hard['passed'] and cotangent_ABI and cotangent_finite and observed_memory and all(row['passed'] for row in bridge.values())
        repeats_passed = True
        emit('raw_capture_saved', strict_zero_delta_bridge=bridge, hard_admission=hard,
             cotangent_ABI_passed=cotangent_ABI, cotangent_finite=cotangent_finite,
             capture_call_seconds=seconds, capture_memory=memory, observed_memory=stats,
             observed_memory_passed=observed_memory)
        repeats = []
        if first_passed:
            for repeat in range(1, 3):
                remaining = min(90-(time.perf_counter()-calls_started),
                    report['budget'].limit-(time.perf_counter()-report['budget'].started))
                if remaining < max(10, 1.5*seconds+3):
                    emit('repeat_budget_insufficient', repeat_stability='unverified',
                         completed_repeat_calls=len(repeats), remaining_seconds=remaining)
                    break
                emit('repeat_call_start', capture_calls=repeat+1)
                extra = jax.device_get(jax.block_until_ready(executable(*capture_arguments)))
                # Compact repeat arrays are saved before stability gates. Full
                # parameter gradients are compared in memory, without duplication.
                repeat_path = args.output_dir / f'public-cotangents.repeat-{repeat}.pkl'
                compact = dict(identity=identity, loss_aux=extra[0], state_gradient=extra[1][1],
                    cotangents=extra[1][2], consumers=list(policy.CONSUMERS),
                    parameter_comparison=dict(passed=None, status='raw_saved_before_comparison'))
                assets.write_pickle(repeat_path, compact)
                extra_scope = (extra[0], extra[1][:2])
                comparison = assets.compare_full(jax, extra_scope, original_scope)
                compact['parameter_comparison'] = comparison['all_parameters']
                assets.write_pickle(repeat_path, compact)
                cotangent_comparison = assets.strict_comparison(jax, extra[1][2], result[1][2])
                extra_hard = assets.hard_admission(jax, extra_scope, defined, host_parameters)
                repeat_memory = devices[0].memory_stats() or {}
                repeat_memory_passed = 'peak_bytes_in_use' in repeat_memory and repeat_memory['peak_bytes_in_use'] <= MEMORY_LIMIT
                row = dict(repeat=repeat, full_scope_comparison=comparison,
                    cotangent_self_drift=cotangent_comparison, hard_admission=extra_hard,
                    observed_memory=repeat_memory, observed_memory_passed=repeat_memory_passed,
                    passed=extra_hard['passed'] and repeat_memory_passed and cotangent_comparison['passed'] and all(v['passed'] for v in comparison.values()))
                repeats.append(row)
                emit('repeat_saved', repeat_comparisons=repeats,
                     repeat_stability='passed' if len(repeats)==2 and all(r['passed'] for r in repeats) else 'pending_or_discrepant')
                del extra, extra_scope, compact
                if not row['passed']:
                    repeats_passed = False
                    break
    # Preserve the compiled executable only within remaining budget. No huge HLO
    # text export: this capture changes no original-FP32 precision boundary.
    if first_passed and report['budget'].limit - (time.perf_counter() - report['budget'].started) > 30:
        with report['budget'].stage(30, 'capture_binary_save', emit):
            from jax.experimental.serialize_executable import serialize
            assets.write_pickle(args.output_dir / 'capture.executable.pkl', serialize(executable))
            emit('capture_binary_saved', binary_bytes=(args.output_dir / 'capture.executable.pkl').stat().st_size)
    final_status = ('strict_zero_delta_or_hard_bridge_failed' if not first_passed else
                    'public_cotangents_captured_repeat_drift' if not repeats_passed else
                    'public_cotangents_captured_review_pending')
    emit('completed', status=final_status, first_bridge_passed=first_passed,
         repeat_stability='passed' if len(repeats)==2 and all(r['passed'] for r in repeats) else 'unverified_or_discrepant',
         scientific_interpretation=False, reference_adoption=False, training_updates=0, production_adoption=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--runtime-preflight', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--process-seconds', type=float, default=3590)
    parser.add_argument('--cli-check-only', action='store_true')
    parser.add_argument('--source-check', type=Path)
    parser.add_argument('--synthetic-interface-check', action='store_true')
    args = parser.parse_args(argv)
    if not 0 < args.process_seconds <= 3590:
        parser.error('process budget must be within3590s')
    if args.cli_check_only:
        print(json.dumps(dict(status='CLI_passed', JAX_imports=int('jax' in sys.modules), GPU_calls=0, Teacher_reads=0)))
        return 0
    if args.source_check:
        print(json.dumps(policy.source_check(args.source_check), indent=2))
        return 0
    if args.synthetic_interface_check:
        print(json.dumps(synthetic_interface_check(), indent=2))
        return 0
    if args.output_dir is None or args.runtime_preflight is None:
        parser.error('actual mode requires output-dir and runtime-preflight')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', capture_compile_attempts=0, stage_events=[], budget=Budget(args.process_seconds),
                  precision='original_FP32_no_precision_candidate', full_parameter_coordinates=64884481,
                  initial_state_coordinates=3854, native_public_T7=True, custom_VJP=False)
    lock = threading.RLock()
    def emit(event, **fields):
        with lock:
            report['stage_events'].append(dict(event=event, elapsed_seconds=time.perf_counter()-report['budget'].started, **fields))
            report.update(stage=event, **fields)
            temporary = args.output_dir / 'model.json.tmp'
            temporary.write_text(json.dumps({k:v for k,v in report.items() if k != 'budget'}, indent=2, default=str)+'\n')
            os.replace(temporary, args.output_dir / 'model.json')
            print(event, flush=True)
    def expire():
        # Parent timeout is authoritative even if a writer is busy.
        if lock.acquire(blocking=False):
            try:
                emit('process_timeout', status='timeout')
            finally:
                lock.release()
        os._exit(124)
    timer = threading.Timer(args.process_seconds, expire)
    timer.daemon = True
    timer.start()
    try:
        emit('starting')
        run(args, emit, report)
    except BaseException as error:
        emit('exception', status='resource_failed' if isinstance(error, MemoryError) else 'runtime_or_admission_failed', error=repr(error))
        raise
    finally:
        timer.cancel()
    return 0 if report['status'] == 'public_cotangents_captured_review_pending' else 1


if __name__ == '__main__':
    raise SystemExit(main())
