"""Bounded metadata coverage and persisted-read probe; no Teacher/model execution."""
import argparse
import gzip
import json
import time
from collections import Counter
from pathlib import Path

from scripts.hpc.run_fresh_start_one_day_pool import pool_plan


def run(args):
    started = time.perf_counter()
    with gzip.open(args.catalogue, 'rt') as handle:
        catalogue = json.load(handle)
    config = {'sampling': {'seed': 20260912, 'batch_size': 32,
                          'microbatch_size': 4, 'max_artifacts_per_batch': 4,
                          'batch_packing': 'fixed_replay_pad'}}
    sampler, _panels = pool_plan(catalogue, config)
    sizes, files, micros = Counter(), Counter(), Counter()
    selected = {}
    source_days, source_rows, draws = set(), 0, 0
    count = sampler.batch_count(0)
    for batch in sampler.iter_epoch(0):
        sizes[len(batch.rows)] += 1
        files[len({row.artifact_id for row in batch.rows})] += 1
        source_days.update(row.day_key for row in batch.rows)
        source_rows += batch.source_draw_count
        draws += len(batch.rows)
        for micro in batch.microbatches:
            micros[len(micro.rows)] += 1
        if batch.batch_index in (0, count // 2, count - 1):
            selected[batch.batch_index] = batch
    if sizes != {32: count} or micros != {4: count * 8}:
        raise ValueError('expanded fixed batch shape mismatch')
    if source_rows != 346539 or len(source_days) != 346539 or draws != 346560:
        raise ValueError('expanded fixed epoch coverage differs from declared pool')
    # Real reader exercises first/middle/terminal-padded batch without a network.
    import jax

    from research.daily_coarse_graining.daily_operator.prepared_one_day import PreparedOneDayPool

    pool = PreparedOneDayPool(Path(catalogue['source_manifest']), cache_size=4)
    readbacks = []
    for index, batch in selected.items():
        before = time.perf_counter()
        payload = pool.batch([(r.artifact_id, r.batch_row) for r in batch.rows])
        leaves = jax.tree_util.tree_leaves(payload)
        if not leaves or any(value.shape[0] != 32 for value in leaves):
            raise ValueError('saved batch does not have consistent32 leading rows')
        readbacks.append({'batch': index, 'seconds': time.perf_counter() - before,
                          'artifact_count': len({r.artifact_id for r in batch.rows}),
                          'source_rows': batch.source_draw_count,
                          'padding_rows': batch.padding_draw_count})
    result = {'status': 'passed', 'training_updates': 0, 'teacher_regenerated': False,
              'configuration': config, 'source_rows': source_rows, 'unique_days': len(source_days),
              'row_draws': draws, 'padding_draws': draws - source_rows, 'batches': count,
              'batch_size_histogram': sizes, 'microbatch_size_histogram': micros,
              'artifact_count_histogram': files, 'readbacks': readbacks,
              'reader': pool.statistics(), 'seconds': time.perf_counter() - started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalogue', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    run(parser.parse_args())
