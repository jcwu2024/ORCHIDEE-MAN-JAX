"""Build a reproducible one-day sampling catalogue from existing v6 metadata."""
from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path


def run(args):
    from research.daily_coarse_graining.daily_operator.one_day_sampling import OneDayArtifact, OneDaySampler
    from research.daily_coarse_graining.daily_operator.prepared_one_day import PreparedOneDayPool
    from research.daily_coarse_graining.daily_operator.target_oriented_training import (
        build_target_oriented_training_schedule,
        load_target_oriented_training_contract,
    )
    from research.daily_coarse_graining.typed_sidecar import load_coherent_typed_sidecar_index
    from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
    started = time.perf_counter()
    source = json.loads(args.prepared_manifest.read_text())
    index = load_coherent_typed_sidecar_index(args.dataset_manifest, verify_hashes=False)
    contract = load_target_oriented_training_contract(args.training_contract)
    schedule = build_target_oriented_training_schedule(index.select(spatial_split='train', temporal_split='train'),
                                                       contract, updates=source['updates'])
    if schedule['schedule_sha256'] != source['identity']['schedule_sha256']:
        raise ValueError('reconstructed data schedule differs from accepted packages')
    ids = {int(row['update']) for row in source['entries']}
    entries = {entry['update']: entry for entry in schedule['entries'] if entry['update'] in ids}
    artifacts = [OneDayArtifact.from_schedule_entry(identifier, entries[identifier]) for identifier in sorted(ids)]
    selection = OneDaySampler(artifacts, seed=args.seed+1, batch_size=32, microbatch_size=4, max_artifacts_per_batch=4)
    monitor, monitor_days = [], set()
    for batch in selection.iter_epoch(0):
        for row in batch.rows:
            if row.day_key not in monitor_days:
                monitor.append(row)
                monitor_days.add(row.day_key)
            if len(monitor) == args.monitor_days:
                break
        if len(monitor) == args.monitor_days:
            break
    if len(monitor) != args.monitor_days:
        raise ValueError('pool is too small for the requested fixed monitor')
    sampler = OneDaySampler(artifacts, seed=args.seed, batch_size=32, microbatch_size=4,
                            max_artifacts_per_batch=4, excluded_days=monitor_days)
    batches = list(sampler.iter_epoch(0))
    first = batches[0]
    pool = PreparedOneDayPool(args.prepared_manifest, cache_size=4)
    rows = [(row.artifact_id, row.batch_row) for row in first.rows]
    cold_started = time.perf_counter()
    cold = pool.batch(rows)
    cold_seconds = time.perf_counter()-cold_started
    for identifier in first.artifact_ids[:2]:
        _, actual_entry = pool.load(identifier)
        if actual_entry != entries[identifier]:
            raise ValueError('payload and reconstructed schedule differ')
    warm_started = time.perf_counter()
    warm = pool.batch(rows)
    warm_seconds = time.perf_counter()-warm_started
    import jax
    import numpy as np
    if not all(np.array_equal(a,b,equal_nan=True) for a,b in zip(jax.tree_util.tree_leaves(cold), jax.tree_util.tree_leaves(warm), strict=True)):
        raise ValueError('cached and freshly stacked batch differ')
    training_rows = [row for batch in batches for row in batch.rows]
    unique = {row.day_key for row in training_rows}
    if unique & monitor_days:
        raise ValueError('fixed monitor Teacher days overlap training')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    catalogue = args.output.with_suffix('.catalogue.json.gz')
    catalogue_payload = {
        'source_manifest': str(args.prepared_manifest.resolve()), 'source_identity': source['identity'],
        'seed': args.seed, 'batch_size': 32, 'microbatch_size': 4, 'max_artifacts_per_batch': 4,
        'monitor_rows': [row.as_metadata() for row in monitor],
        'one_day_entries': [{'update': e['update'], 'runtime_year': e['runtime_year'],
                             'components': {'one_day': e['components']['one_day']}} for e in entries.values()],
    }
    with gzip.open(catalogue, 'wt', encoding='utf-8') as handle:
        json.dump(catalogue_payload, handle, separators=(',',':'))
    report = {'status':'passed', 'schema_version':'prepared_one_day_catalogue_v1', 'training_updates':0,
              'teacher_regenerated':False, 'validation_accessed':False, 'catalogue':str(catalogue),
              'pool':sampler.pool_metadata(), 'source_artifact_count':len(ids),
              'source_one_day_draws':sum(len(a.rows) for a in artifacts), 'monitor_unique_days':len(monitor_days),
              'train_unique_locations':len({row.landpoint_id for row in training_rows}),
              'train_unique_point_years':len({(row.landpoint_id,row.year) for row in training_rows}),
              'train_years':sorted({row.year for row in training_rows}),
              'epoch_batches':len(batches), 'remainder_batches':sum(b.draw_count !=32 for b in batches),
              'first_batch':first.as_metadata(), 'reader':pool.statistics(),
              'cold_stack_seconds':cold_seconds, 'warm_stack_seconds':warm_seconds,
              'elapsed_seconds':time.perf_counter()-started}
    _atomic_json(args.output, report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('first_batch','train_years')}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-manifest', type=Path, required=True)
    parser.add_argument('--prepared-manifest', type=Path, required=True)
    parser.add_argument('--training-contract', type=Path, default=Path('manifests/coarse_graining/target_oriented_bounded_training_v1.json'))
    parser.add_argument('--seed', type=int, default=20260912)
    parser.add_argument('--monitor-days', type=int, default=64)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
