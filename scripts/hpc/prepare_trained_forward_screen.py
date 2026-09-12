"""Select disjoint 7/30-day windows using existing training metadata only."""
import argparse
import gzip
import json
import time
from pathlib import Path


def run(args):
    from research.daily_coarse_graining.daily_operator.forward_screen_selection import select_windows, training_day_keys
    from research.daily_coarse_graining.daily_operator.target_oriented_training import (
        build_target_oriented_training_schedule,
        load_target_oriented_training_contract,
    )
    from research.daily_coarse_graining.typed_sidecar import load_coherent_typed_sidecar_index
    from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json

    started = time.perf_counter()
    with gzip.open(args.catalogue,'rt') as handle:
        catalogue = json.load(handle)
    source = json.loads(args.source_manifest.read_text())
    index = load_coherent_typed_sidecar_index(args.dataset_manifest,verify_hashes=False)
    schedule = build_target_oriented_training_schedule(
        index.select(spatial_split='train',temporal_split='train'),load_target_oriented_training_contract(),updates=source['updates'])
    if schedule['schedule_sha256'] != source['identity']['schedule_sha256']:
        raise ValueError('prepared schedule identity differs from source metadata')
    ids = {int(r['update']) for r in source['entries']}
    entries = [r for r in schedule['entries'] if r['update'] in ids]
    excluded = training_day_keys(catalogue)
    windows = select_windows(entries,excluded,seed=20260913,per_horizon=8)
    report = {'status':'passed','schema_version':'trained_forward_screen_selection_v1',
              'seed':20260913,'training_days_excluded':len(excluded),'windows':windows,
              'source_manifest':str(args.source_manifest.resolve()),'source_identity':source['identity'],
              'checkpoint_update':1998,'training_job':15048786,'teacher_regenerated':False,
              'spatial_temporal_split':'train/train','sealed_test_accessed':False,
              'parameter_updates':0,'elapsed_seconds':time.perf_counter()-started}
    _atomic_json(args.output,report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('windows','source_identity')},indent=2),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('catalogue','source-manifest','dataset-manifest','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
