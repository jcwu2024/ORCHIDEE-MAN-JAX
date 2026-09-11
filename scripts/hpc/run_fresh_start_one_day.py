"""Fresh initialization and full-network one-day admission/learning on existing v6."""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def bootstrap(args, protocol):
    # This loader owns existing v6 schemas/statistics and small example rows;
    # it loads no pretrained checkpoints. Disable its unused legacy probe
    # initialization; every candidate weight comes from our one seed.
    from scripts.hpc.run_successor_fast_owner_gpu_probe import _load_extended_runtime
    from scripts.hpc.run_target_oriented_attention_gpu_admission import _head
    from scripts.hpc.run_target_oriented_full_step_preflight import _prepare_template
    backend = {}
    if args.shape_only:
        def cpu_backend():
            import jax
            return jax, jax.devices('cpu')
        backend['backend_initializer'] = cpu_backend
    runtime = _load_extended_runtime(args, initialize_probe_parameters=False, **backend)
    from research.daily_coarse_graining.daily_operator.fresh_start import build_fresh_head, initialize_fresh_parameters
    from research.daily_coarse_graining.daily_operator.mechanistic_output_contract import (
        build_mechanistic_learned_schema,
        load_mechanistic_output_contract,
    )
    from research.daily_coarse_graining.daily_operator.mechanistic_process_head import build_mechanistic_output_layout
    from research.daily_coarse_graining.daily_operator.strong_baseline_contract import load_strong_baseline_contract
    from research.daily_coarse_graining.daily_operator.target_oriented_contract import load_target_oriented_contract
    from research.daily_coarse_graining.daily_operator.target_oriented_training import (
        derive_target_oriented_scales,
        load_target_oriented_state_anchor_policy,
    )
    architecture, mechanism, target = load_strong_baseline_contract(), load_mechanistic_output_contract(), load_target_oriented_contract()
    schema = build_mechanistic_learned_schema(runtime['schema'], mechanism)
    layout = build_mechanistic_output_layout(adapter=runtime['adapter'], complete_schema=runtime['schema'],
        learned_schema=schema, owner_contract=runtime['owner_contract'], architecture_contract=architecture,
        mechanism_contract=mechanism, inputs=runtime['probe_assembled'])
    runtime.update(architecture_contract=architecture, mechanism_contract=mechanism,
                   learned_schema=schema, target_layout=layout)
    runtime.pop('parameters', None)
    manifest = json.loads(args.prepared_manifest.read_text())
    artifacts = []
    for identifier in protocol['data']['artifact_ids']:
        record = next(v for v in manifest['entries'] if int(v['update']) == identifier)
        path = (args.prepared_manifest.parent / record['path']).resolve()
        path.relative_to(args.prepared_manifest.parent.resolve())
        with gzip.open(path, 'rb') as handle:
            artifact = pickle.load(handle)
        if int(artifact.schedule_entry['update']) != identifier:
            raise ValueError('prepared sample identity differs')
        artifacts.append(artifact)
    template = _prepare_template(runtime, artifacts[0].schedule_entry)
    template = replace(template, static_owner=replace(template.static_owner, include_modelout_diagnostics=False))
    head = build_fresh_head(_head(runtime, target), template, threshold=protocol['input_conditioning']['threshold'])
    parameters = initialize_fresh_parameters(head, runtime['probe_assembled'], seed=protocol['seed'])
    scales = derive_target_oriented_scales(runtime['base_statistics'].arrays['next_state'].scale,
        runtime['process_statistics'].outputs['daily.fast_gpp'].scale, runtime['contract'],
        state_anchor_policy=load_target_oriented_state_anchor_policy(), flux_basis=tuple(protocol['fluxes']))
    jax = runtime['jax']
    index = protocol['data']['one_day_sample_index']
    samples = [jax.tree_util.tree_map(lambda v: v[index], a.batch.one_day) for a in artifacts]
    metadata = []
    for artifact, sample in zip(artifacts, samples, strict=True):
        rows = [(record, row) for record in artifact.schedule_entry['components']['one_day'] for row in record['start_rows']]
        record, row = rows[index]
        metadata.append(dict(record, selected_start_row=row, prepared_day_index=int(np.asarray(sample.inputs.day_index))))
    return runtime, head, template, scales, parameters, samples, metadata


def run(args):
    from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
    protocol = json.loads(args.protocol.read_text())
    if protocol['input_conditioning']['scope'] != 'slowproc_stomate_previous_step_state.DOC':
        raise ValueError('this prototype implements the declared DOC-only input policy')
    if protocol['trainable_scope'] != 'all_neural_parameters' or protocol['optimizer']['name'] != 'AdamW':
        raise ValueError('this prototype requires all-neural-parameter AdamW')
    started = time.perf_counter()
    report = {'status': 'running', 'protocol': protocol, 'completed_updates': 0,
              'source_git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'checkpoint_inherited': False, 'monitors': [], 'updates': []}

    def emit(stage, **values):
        report.update(values, stage=stage, elapsed_seconds=time.perf_counter() - started)
        _atomic_json(args.output, report)
        print(f'fresh_start stage={stage} elapsed={report["elapsed_seconds"]:.1f}s', flush=True)

    def hard_ok(aux):
        return all(int(v) == 0 for v in aux['hard'].values())

    try:
        emit('bootstrap')
        runtime, head, template, scales, parameters, samples, metadata = bootstrap(args, protocol)
        jax = runtime['jax']
        import jax.numpy as jnp

        from research.daily_coarse_graining.daily_operator.fresh_start_training import make_objective, make_update
        from research.daily_coarse_graining.daily_operator.target_oriented_training import (
            initialize_target_oriented_adamw,
        )
        from scripts.hpc.run_successor_fast_owner_gpu_probe import _tree_numeric_report
        count = sum(v.size for v in jax.tree_util.tree_leaves(parameters))
        if count != protocol['parameter_count']:
            raise ValueError(f'fresh parameter count {count} differs')
        report.update(parameter_count=count, trainable_parameter_count=count, sample_metadata=metadata)
        objective = make_objective(runtime, head, template, scales, protocol)
        if args.shape_only:
            emit('trace_complete_objective')
            shapes = [jax.eval_shape(objective, parameters, sample) for sample in samples]
            emit('completed', status='passed', classification='complete_objective_shapes_only',
                 loss_shapes=[list(value[0].shape) for value in shapes],
                 parameter_count=count, trainable_parameter_count=count)
            return report
        forward = jax.jit(objective)
        backward = jax.jit(jax.value_and_grad(objective, has_aux=True))
        emit('initial_forward')
        baseline = [forward(parameters, sample) for sample in samples]
        jax.block_until_ready(baseline)
        emit('initial_forward_completed', initial=baseline)
        if not all(np.isfinite(float(value)) and hard_ok(aux) for value, aux in baseline):
            emit('stopped', status='rejected', reason='initial_forward_physical')
            return report
        emit('compile_full_gradient')
        compiled = backward.lower(parameters, samples[0]).compile()
        memory = compiled.memory_analysis()
        sizes = {name: int(getattr(memory, name)) for name in
                 ('argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes', 'alias_size_in_bytes')}
        sizes['estimated_live_bytes'] = sizes['argument_size_in_bytes'] + sizes['output_size_in_bytes'] + sizes['temp_size_in_bytes'] - sizes['alias_size_in_bytes']
        emit('gradient_memory', compiled_gradient_memory=sizes)
        if sizes['estimated_live_bytes'] > protocol['admission']['memory_limit_gib'] * 2 ** 30:
            emit('stopped', status='rejected', reason='full_gradient_memory')
            return report
        (value, aux), gradient = compiled(parameters, samples[0])
        jax.block_until_ready(gradient)
        numeric = _tree_numeric_report(jax, gradient)
        groups = {'source_encoder': {k: v for k, v in gradient.items() if k != 'target_oriented'}, **gradient['target_oriented']}
        families = {k: _tree_numeric_report(jax, v) for k, v in groups.items()}
        doc_heads = [gradient['target_oriented']['state_heads'][i] for i, field in enumerate(head.learned_schema.fields)
                     if 'DOC' in field.name and gradient['target_oriented']['state_heads'][i]]
        participation = {
            'doc_state_heads': _tree_numeric_report(jax, doc_heads),
            'maintenance': families['distribution_aware_maintenance'],
            'gpp': families['structured_gpp'],
            'shared_encoder': families['source_encoder'],
        }
        emit('gradient_completed', gradient=numeric, gradient_families=families, participation=participation)
        if not numeric['all_finite'] or numeric['l2_norm'] == 0 or not hard_ok(aux):
            emit('stopped', status='rejected', reason='full_gradient')
            return report
        if not all(v['all_finite'] and v['l2_norm'] > 0 for v in participation.values()):
            emit('stopped', status='rejected', reason='module_gradient_participation')
            return report
        # True rounded parameter displacement, compared with AD along exactly
        # that displacement. Two epsilons diagnose curvature/rounding separately.
        direction = jax.tree_util.tree_map(lambda g: (g / numeric['l2_norm']).astype(g.dtype), gradient)
        tangent = jax.jit(lambda p, d, sample: jax.jvp(lambda v: objective(v, sample)[0], (p,), (d,))[1])
        derivative_rows = []
        for epsilon in protocol['admission']['finite_difference_epsilons']:
            plus = jax.tree_util.tree_map(lambda p, d: (p + epsilon*d).astype(p.dtype), parameters, direction)
            minus = jax.tree_util.tree_map(lambda p, d: (p - epsilon*d).astype(p.dtype), parameters, direction)
            effective = jax.tree_util.tree_map(lambda a, b, p: ((a.astype('float64') - b.astype('float64')) / (2*epsilon)).astype(p.dtype), plus, minus, parameters)
            left, right = forward(plus, samples[0]), forward(minus, samples[0])
            finite_difference = float((left[0] - right[0]) / (2*epsilon))
            reverse = float(sum(jnp.vdot(g.astype('float64'), d.astype('float64')) for g, d in zip(jax.tree_util.tree_leaves(gradient), jax.tree_util.tree_leaves(effective), strict=True)))
            jvp = float(tangent(parameters, effective, samples[0]))
            row = {'epsilon': epsilon, 'finite_difference': finite_difference, 'reverse_projection': reverse,
                   'jvp_projection': jvp,
                   'passed': bool(np.isfinite(finite_difference) and np.isclose(finite_difference, reverse,
                       atol=protocol['admission']['derivative_absolute_tolerance'], rtol=protocol['admission']['derivative_relative_tolerance'])
                       and np.isclose(jvp, reverse, atol=protocol['admission']['derivative_absolute_tolerance'],
                                      rtol=protocol['admission']['derivative_relative_tolerance'])
                       and hard_ok(left[1]) and hard_ok(right[1]))}
            derivative_rows.append(row)
            emit('derivative_probe', derivatives=derivative_rows)
        if not all(v['passed'] for v in derivative_rows):
            emit('stopped', status='rejected', reason='derivative_agreement')
            return report
        device_memory = runtime['devices'][0].memory_stats()
        emit('admission_passed', device_memory_stats=device_memory)
        if args.admission_only:
            emit('completed', status='passed', classification='one_day_full_parameter_admission_only')
            return report
        del gradient, groups, direction, plus, minus, effective
        state = initialize_target_oriented_adamw(parameters)
        update = make_update(protocol)
        report['monitors'].append({'update': 0, 'samples': baseline})
        for iteration in range(1, protocol['training']['updates'] + 1):
            began = time.perf_counter()
            accumulated = None
            for sample in samples:
                (loss, local), grad = backward(parameters, sample)
                if not np.isfinite(float(loss)) or not hard_ok(local):
                    emit('stopped', status='rejected', reason='training_forward')
                    return report
                accumulated = grad if accumulated is None else jax.tree_util.tree_map(jnp.add, accumulated, grad)
            accumulated = jax.tree_util.tree_map(lambda g: g / len(samples), accumulated)
            parameters, state, norm, clipped = update(parameters, accumulated, state)
            jax.block_until_ready(parameters)
            if not np.isfinite(float(norm)):
                emit('stopped', status='rejected', reason='training_gradient')
                return report
            report['updates'].append({'update': iteration, 'gradient_norm': float(norm), 'clip_scale': float(clipped),
                                      'seconds': time.perf_counter()-began})
            if iteration in protocol['training']['monitor_updates']:
                current = [forward(parameters, sample) for sample in samples]
                jax.block_until_ready(current)
                report['monitors'].append({'update': iteration, 'samples': current})
                if not all(np.isfinite(float(loss)) and hard_ok(aux) for loss, aux in current):
                    emit('stopped', status='rejected', reason='monitor_forward')
                    return report
            emit('update_completed', completed_updates=iteration)
        final = report['monitors'][-1]['samples']
        passed = (np.mean([float(v[0]) for v in final]) < np.mean([float(v[0]) for v in baseline])
                  and np.mean([float(v[1]['flux_loss']) for v in final]) < np.mean([float(v[1]['flux_loss']) for v in baseline]))
        checkpoint = args.output.with_suffix('.checkpoint.pkl')
        with checkpoint.open('wb') as handle:
            pickle.dump({'protocol': protocol, 'source_git_head': report['source_git_head'],
                         'parameters': jax.device_get(parameters), 'optimizer': jax.device_get(state),
                         'purpose': 'fresh_start_probe_not_final_model'}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with checkpoint.open('rb') as handle:
            restored = pickle.load(handle)
        restored_parameters = jax.device_put(restored['parameters'])
        restored_value = float(forward(restored_parameters, samples[0])[0])
        checkpoint_replay = bool(np.isclose(restored_value, float(final[0][0]), atol=1e-6, rtol=1e-5))
        emit('completed', status='completed', learning_screen_passed=bool(passed), checkpoint=str(checkpoint),
             checkpoint_forward_replay=checkpoint_replay, device_memory_stats=runtime['devices'][0].memory_stats())
    except Exception as exc:
        emit('failed', status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    return report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset-manifest', 'acceptance-report', 'base-statistics', 'process-statistics', 'fast-statistics', 'prepared-manifest', 'output'):
        result.add_argument('--'+name, type=Path, required=True)
    result.add_argument('--protocol', type=Path, default=ROOT/'manifests/coarse_graining/fresh_start_one_day_probe_v1.json')
    result.add_argument('--admission-only', action='store_true')
    result.add_argument('--shape-only', action='store_true', help='CPU trace of the complete loss on actual samples; no model execution or updates')
    return result


if __name__ == '__main__':
    run(parser().parse_args())
