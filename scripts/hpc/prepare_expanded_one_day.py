"""Materialize a declared range of expanded one-day rows without Teacher runs."""
from __future__ import annotations

import argparse
import gzip
import json
import pickle
import subprocess
import time
from pathlib import Path

import numpy as np


def exact_tree(jax, left, right):
    a, ta = jax.tree_util.tree_flatten(left)
    b, tb = jax.tree_util.tree_flatten(right)
    return ta == tb and all(np.asarray(x).dtype == np.asarray(y).dtype and
                           np.array_equal(x, y, equal_nan=True)
                           for x, y in zip(a, b, strict=True))


def run(args):
    from research.daily_coarse_graining.daily_operator.expanded_one_day import prepare_one_day_rows
    from research.daily_coarse_graining.daily_operator.prepared_one_day import PreparedOneDayPool
    from research.daily_coarse_graining.daily_operator.target_oriented_training import (
        PreparedTargetOrientedTrainingBatch,
    )
    from scripts.hpc.prepare_target_oriented_training_batches import _initialize_cpu_backend
    from scripts.hpc.run_successor_fast_owner_gpu_probe import _load_extended_runtime
    from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
    from scripts.hpc.target_oriented_training_prepared import (
        PREPARED_ARTIFACT_SCHEMA,
        PreparedTargetOrientedUpdateArtifact,
    )

    started = time.perf_counter()
    plan = json.loads(args.plan.read_text())
    if plan['schema_version'] != 'one_day_expansion_plan_v1' or not all(plan['checks'].values()):
        raise ValueError('expansion plan is not valid')
    if str(args.dataset_manifest.resolve()) != plan['dataset_manifest']:
        raise ValueError('dataset differs from the declared expansion plan')
    entries = plan['entries']
    if args.entry_ids:
        ids = [int(value) for value in args.entry_ids.split(',')]
    else:
        ids = list(range(args.start_entry, (args.end_entry or len(entries))+1))
    if not ids or len(set(ids)) != len(ids) or min(ids) < 1 or max(ids) > len(entries):
        raise ValueError('invalid preparation entry range')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir/'prepared_manifest.json'
    if report_path.exists():
        raise FileExistsError('use a new output directory for this preparation range')
    report = {'schema_version': 'expanded_one_day_preparation_v1', 'status': 'running',
              'plan': str(args.plan.resolve()), 'entry_ids': ids, 'entries': [],
              'identity': {**plan['source_identity'], 'preparation_plan': str(args.plan.resolve()),
                           'expansion_seed': plan['seed'], 'rows_per_point_year': plan['rows_per_point_year']},
              'source_git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'teacher_regenerated': False, 'training_updates': 0, 'sealed_test_accessed': False,
              'complete_expansion': len(ids) == len(entries)}

    def emit(stage, **values):
        report.update(stage=stage, elapsed_seconds=time.perf_counter()-started, **values)
        _atomic_json(report_path, report)
        print(f'expanded_one_day stage={stage} completed={len(report["entries"])} seconds={report["elapsed_seconds"]:.1f}', flush=True)

    emit('bootstrap')
    try:
        runtime = _load_extended_runtime(args, backend_initializer=_initialize_cpu_backend,
                                         initialize_probe_parameters=False)
        jax = runtime['jax']
        lookup = {(str(ref.parent.landpoint_id), int(ref.parent.year)): ref
                  for ref in runtime['index'].select(spatial_split='train', temporal_split='train')}
        source = PreparedOneDayPool(plan['source_manifest'], cache_size=1)
        if source.identity != plan['source_identity']:
            raise ValueError('prepared retained-input source differs')
        for identifier in ids:
            item_started = time.perf_counter()
            entry = entries[identifier-1]
            ref = lookup[(entry['landpoint_id'], entry['year'])]
            if (ref.parent.sha256, ref.sha256, int(ref.day_count)) != (
                    entry['parent_shard_sha256'], entry['typed_shard_sha256'], entry['day_count']):
                raise ValueError('selected raw point-year differs from plan')
            origin = entry['retained_source']
            old_batch, old_entry = source.load(origin['artifact_id'])
            flattened = [(str(record['landpoint_id']), int(record['year']), int(row))
                         for record in old_entry['components']['one_day'] for row in record['start_rows']]
            if flattened[origin['batch_row']] != (entry['landpoint_id'], entry['year'], origin['reference_row']):
                raise ValueError('retained source row is not the declared point-year')
            old_row = jax.tree_util.tree_map(lambda value: np.asarray(value)[origin['batch_row']], old_batch)
            rows = entry['start_rows']
            if origin['reference_row'] not in rows:
                raise ValueError('old comparison row must be retained in expanded training rows')
            build_started = time.perf_counter()
            batch = prepare_one_day_rows(ref, rows, adapter=runtime['adapter'], schema=runtime['schema'],
                                         owner_contract=runtime['owner_contract'], day_inputs=old_row.day_inputs)
            build_seconds = time.perf_counter()-build_started
            comparison_row = jax.tree_util.tree_map(lambda value: np.asarray(value)[rows.index(origin['reference_row'])], batch)
            if not exact_tree(jax, comparison_row, old_row):
                raise ValueError('new raw reader differs from accepted prepared row')
            record = {key: entry[key] for key in ('landpoint_id', 'year', 'parent_shard_sha256', 'typed_shard_sha256', 'start_rows')}
            schedule_entry = {'update': identifier, 'runtime_year': entry['year'], 'components': {'one_day': [record]}}
            artifact = PreparedTargetOrientedUpdateArtifact(
                PREPARED_ARTIFACT_SCHEMA, source.identity['training_contract_sha256'], schedule_entry,
                'one_day_only_no_rollout_trace', PreparedTargetOrientedTrainingBatch(batch, None, None))
            path = args.output_dir/f'point-year-{identifier:06d}.pkl.gz'
            write_started = time.perf_counter()
            with gzip.open(path, 'wb', compresslevel=1) as handle:
                pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
            write_seconds = time.perf_counter()-write_started
            report['entries'].append({'update': identifier, 'path': path.name,
                'rows': len(rows), 'new_rows': len(entry['new_start_rows']),
                'landpoint_id': entry['landpoint_id'], 'year': entry['year'],
                'schedule_entry': schedule_entry, 'old_row_exact': True,
                'raw_shard_bytes': ref.parent.path.stat().st_size + ref.path.stat().st_size,
                'uncompressed_batch_bytes': sum(np.asarray(v).nbytes for v in jax.tree_util.tree_leaves(batch)),
                'size_bytes': path.stat().st_size, 'build_seconds': build_seconds,
                'write_seconds': write_seconds, 'elapsed_seconds': time.perf_counter()-item_started})
            emit('point_year_completed')
        # Use the unchanged training reader for a persisted batch readback.
        report['status'] = 'passed'
        _atomic_json(report_path, report)
        read_started = time.perf_counter()
        pool = PreparedOneDayPool(report_path, cache_size=1)
        readback, _ = pool.load(ids[-1])
        if not exact_tree(jax, readback, batch):
            raise ValueError('persisted one-day reader roundtrip differs')
        import resource
        emit('completed', status='passed', persisted_reader_exact=True,
             readback_seconds=time.perf_counter()-read_started,
             host_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    except Exception as error:
        emit('failed', status='failed', error=f'{type(error).__name__}: {error}')
        raise
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'dataset-manifest', 'acceptance-report', 'base-statistics',
                 'process-statistics', 'fast-statistics', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--entry-ids')
    parser.add_argument('--start-entry', type=int, default=1)
    parser.add_argument('--end-entry', type=int)
    run(parser.parse_args())
