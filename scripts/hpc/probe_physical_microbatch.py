"""Same-batch physical-v2 microbatch throughput probe; no retained model."""
from __future__ import annotations

import gzip
import itertools
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from scripts.hpc import probe_feedback_loss_calibration as calibration
from scripts.hpc import probe_feedback_loss_updates as admission
from scripts.hpc import run_fresh_start_one_day as prototype


def run(args):
    from research.daily_coarse_graining.daily_operator.fresh_start_training import make_objective, make_update
    from research.daily_coarse_graining.daily_operator.one_day_training_state import make_gradient_accumulator
    from research.daily_coarse_graining.daily_operator.prepared_one_day import PreparedOneDayPool
    from research.daily_coarse_graining.daily_operator.target_oriented_training import initialize_target_oriented_adamw
    from scripts.hpc.probe_one_day_batch import _compiled_memory, _device_memory, _resolved_reference
    from scripts.hpc.probe_public_rollout_staged import StageStore
    from scripts.hpc.run_fresh_start_one_day_pool import pool_plan
    from scripts.hpc.run_successor_fast_owner_gpu_probe import _tree_numeric_report
    from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json

    started = time.perf_counter()
    report = dict(schema_version='physical_microbatch_probe_v1', status='running',
                  microbatch_size=args.microbatch_size, parent_batch_size=64,
                  source_git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  checkpoint_written=False, checkpoint_inherited=False,
                  scientific_acceptance=False, updates=[], completed_updates=0)

    def emit(stage, **values):
        report.update(stage=stage, elapsed_seconds=time.perf_counter()-started, **values)
        _atomic_json(args.output, report)
        print(f'microbatch={args.microbatch_size} stage={stage} seconds={report["elapsed_seconds"]:.1f}', flush=True)

    try:
        configuration = json.loads(args.benchmark_protocol.read_text())
        if configuration['schema_version'] != 'physical_microbatch_benchmark_v1':
            raise ValueError('unknown benchmark protocol')
        if args.microbatch_size not in configuration['microbatch_sizes']:
            raise ValueError('microbatch is outside the declared benchmark')
        model = json.loads(args.protocol.read_text())
        pool_configuration = json.loads(args.pool_protocol.read_text())
        admission_configuration = json.loads(args.admission_config.read_text())
        admission._validate_inputs(pool_configuration, model, {}, shape_only=args.shape_only,
                                   admission_configuration=admission_configuration)
        coefficients = admission._coefficients(admission_configuration)
        merge = json.loads(args.catalogue_report.read_text())
        if merge['status'] != 'passed':
            raise ValueError('expanded catalogue has not passed merge')
        catalogue_path = _resolved_reference(merge['catalogue'], relative_to=args.catalogue_report.resolve())
        with gzip.open(catalogue_path, 'rt') as handle:
            catalogue = json.load(handle)
        sampler, _ = pool_plan(catalogue, pool_configuration)
        batches = tuple(itertools.islice(sampler.iter_epoch(0), 2))
        rows = tuple(row for batch in batches for row in batch.rows)
        if len(rows) != 64 or len({(r.artifact_id, r.batch_row) for r in rows}) != 64:
            raise ValueError('benchmark requires the first 64 unique TRAIN rows')
        identity = dict(model_protocol=model, benchmark_protocol=configuration,
                        rows=[r.as_metadata() for r in rows], coefficients=coefficients)
        # Lists survive JSON and pickle identically for comparison across processes.
        identity = json.loads(json.dumps(identity))
        report.update(identity=identity, same_batch_repeated=True,
                      timing_protocol='one_discarded_warmup_then_three_sequential_updates; '
                                      'synchronized_IO_gradient_accumulation_optimizer; host_finiteness_separate',
                      runtime_evidence=configuration['runtime_evidence'])
        emit('bootstrap')
        runtime, head, template, scales, parameters, _, _ = prototype.bootstrap(args, model)
        jax = runtime['jax']
        import jax.numpy as jnp

        report['parameter_count'] = sum(p.size for p in jax.tree_util.tree_leaves(parameters))
        graph = calibration._loss_protocol(model, canopy_source_pft_index=calibration._canopy_source_pft_index(runtime, head))
        objective = make_objective(runtime, head, template, scales, graph)
        backward = calibration._make_evaluator(jax, jnp, objective, calibration._base_weights(graph))
        update = make_update(model)
        add, finish = make_gradient_accumulator(jax, jnp)
        coefficient_array = jnp.asarray(coefficients, dtype=jnp.float64)
        source = _resolved_reference(catalogue['source_manifest'], relative_to=catalogue_path)
        pool = PreparedOneDayPool(source, cache_size=max(4, len({r.artifact_id for r in rows})))
        if pool.identity != catalogue['source_identity']:
            raise ValueError('prepared reader identity differs')
        groups = tuple(rows[i:i+args.microbatch_size] for i in range(0, 64, args.microbatch_size))

        def sample(group):
            return pool.batch([(r.artifact_id, r.batch_row) for r in group])

        first = sample(groups[0])
        if args.shape_only:
            (_, _), grad = jax.eval_shape(backward, parameters, first, coefficient_array)
            state = jax.eval_shape(initialize_target_oriented_adamw, parameters)
            jax.eval_shape(update, parameters, grad, state)
            emit('completed', status='passed', classification='gradient_optimizer_shapes_only')
            return report

        emit('compile_gradient')
        began = time.perf_counter()
        executable = backward.lower(parameters, first, coefficient_array).compile()
        memory = _compiled_memory(executable)
        emit('gradient_compiled', compile_seconds=time.perf_counter()-began, compiled_memory=memory)
        if memory['estimated_live_bytes'] > configuration['compiled_memory_limit_gib'] * 1024**3:
            emit('stopped', status='resource_limited', reason='compiled_gradient_memory')
            return report
        optimizer = initialize_target_oriented_adamw(parameters)
        jax.block_until_ready((parameters, optimizer))

        def complete_update(current, state):
            began = time.perf_counter()
            total = jax.tree_util.tree_map(jnp.zeros_like, current)
            total_loss = 0.0
            hard_totals = {}
            for group in groups:
                (loss_aux, gradient) = executable(current, sample(group), coefficient_array)
                loss, auxiliary = loss_aux
                total = add(total, gradient, len(group))
                jax.block_until_ready((loss, auxiliary, total))
                hard = calibration._hard_summary(jax.device_get(auxiliary['hard']), batch_size=len(group))
                for key, value in hard['totals'].items():
                    hard_totals[key] = hard_totals.get(key, 0) + value
                if not np.isfinite(float(loss)) or not hard['all_zero']:
                    raise ValueError('microbatch loss/physical gate failed')
                total_loss += float(loss) * len(group)
            gradient = finish(total, 64)
            proposed, proposed_state, norm, clip = update(current, gradient, state)
            jax.block_until_ready((proposed, proposed_state, norm, clip))
            compute_seconds = time.perf_counter()-began
            check_start = time.perf_counter()
            numeric = _tree_numeric_report(jax, gradient)
            finite = (numeric['all_finite'] and numeric['l2_norm'] > 0
                      and admission._host_tree_finite(jax, proposed)
                      and admission._host_tree_finite(jax, proposed_state)
                      and np.isfinite(float(norm)) and 0 < float(clip) <= 1)
            if not finite:
                raise ValueError('complete optimizer update nonfinite or gradient empty')
            checked_seconds = time.perf_counter()-began
            record = dict(loss=total_loss/64, compute_seconds=compute_seconds,
                          check_seconds=time.perf_counter()-check_start,
                          total_seconds=checked_seconds, samples_per_second=64/compute_seconds,
                          hard=hard_totals, gradient=numeric, parameters_optimizer_finite=True,
                          gradient_norm=float(norm), clip_scale=float(clip))
            return proposed, proposed_state, gradient, record

        emit('warmup_complete_update')
        warm_parameters, warm_optimizer, warm_gradient, warm_record = complete_update(parameters, optimizer)
        # Two diagnostic arrays in one container; not a model/optimizer checkpoint.
        host_start, host_proposed = jax.device_get((parameters, warm_parameters))
        delta = jax.tree_util.tree_map(lambda p, q: q-p, host_start, host_proposed)
        store = StageStore(args.output.with_suffix('.stages'), identity)
        store.write('first_update', dict(loss=warm_record['loss'], gradient=warm_gradient, parameter_delta=delta), jax)
        emit('warmup_saved', warmup=warm_record,
             comparison_artifact=str(args.output.with_suffix('.stages')/'first_update.pkl'))
        del warm_parameters, warm_optimizer, warm_gradient, host_start, host_proposed, delta
        for index in range(configuration['timed_updates']):
            parameters, optimizer, gradient, record = complete_update(parameters, optimizer)
            del gradient
            report['updates'].append(dict(update=index+1, **record))
            report['completed_updates'] = index+1
            emit('update_completed')
        seconds = [r['compute_seconds'] for r in report['updates']]
        peak = _device_memory(runtime['devices'][0])
        emit('completed', status='passed', median_compute_seconds=float(np.median(seconds)),
             median_samples_per_second=64/float(np.median(seconds)), memory=peak,
             reader=pool.statistics(), all_hard_counts_zero=True,
             warmup_updates_discarded=1, retained_model=False,
             all_updates_finite=True)
        return report
    except Exception as error:
        emit('failed', status='failed', error=f'{type(error).__name__}: {error}')
        raise


def parser():
    result = prototype.parser()
    for name in ('benchmark-protocol', 'pool-protocol', 'admission-config', 'catalogue-report'):
        result.add_argument('--'+name, type=Path, required=True)
    result.add_argument('--microbatch-size', type=int, choices=(8, 16), required=True)
    return result


if __name__ == '__main__':
    run(parser().parse_args())
