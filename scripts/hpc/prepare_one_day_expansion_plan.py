"""Prepare expansion metadata from the existing TRAIN catalogue and v6 index."""
import argparse
import gzip
import json
import time
from pathlib import Path


def run(args):
    from research.daily_coarse_graining.daily_operator.one_day_expansion_plan import expansion_plan
    from research.daily_coarse_graining.typed_sidecar import load_coherent_typed_sidecar_index
    from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json

    started = time.perf_counter()
    with gzip.open(args.catalogue, 'rt') as handle:
        catalogue = json.load(handle)
    selection = json.loads(args.selection.read_text())
    if selection['status'] != 'passed':
        raise ValueError('development selection is not accepted')
    index = load_coherent_typed_sidecar_index(args.dataset_manifest, verify_hashes=False)
    result = expansion_plan(catalogue, index.select(spatial_split='train', temporal_split='train'),
                            selection['windows'], seed=args.seed,
                            rows_per_point_year=args.rows_per_point_year)
    result.update(dataset_manifest=str(args.dataset_manifest.resolve()),
                  catalogue=str(args.catalogue.resolve()), selection=str(args.selection.resolve()),
                  elapsed_seconds=time.perf_counter()-started)
    _atomic_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != 'entries'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset-manifest', 'catalogue', 'selection', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20260913)
    parser.add_argument('--rows-per-point-year', type=int, default=16)
    run(parser.parse_args())
