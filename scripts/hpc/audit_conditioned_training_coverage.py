"""Measure current package coverage and the existing schedule's 1001-3000 extension."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_persistent_scale_training import (
    persistent_training_pool,
)
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    build_target_oriented_training_schedule,
    load_target_oriented_training_contract,
)
from research.daily_coarse_graining.typed_sidecar import load_coherent_typed_sidecar_index
from scripts.hpc.audit_target_oriented_structured_gpp_data_coverage import _merged_day_keys, _summary


def run(dataset, training_contract, output):
    started = time.perf_counter()
    index = load_coherent_typed_sidecar_index(dataset, verify_hashes=False)
    references = tuple(index.select(spatial_split='train', temporal_split='train'))
    contract = load_target_oriented_training_contract(training_contract)
    schedule = build_target_oriented_training_schedule(references, contract, updates=3000)
    pool = persistent_training_pool()
    groups = {
        'current_678_package_pool': [schedule['entries'][u - 1] for u in pool],
        'original_1000_packages': schedule['entries'][:1000],
        'proposed_extra_2000_packages': schedule['entries'][1000:],
    }
    available = sum(int(r.day_count) - 1 for r in references)
    keys = {name: _merged_day_keys(entries) for name, entries in groups.items()}
    summaries = {}
    for name, group in keys.items():
        summaries[name] = _summary(group['all']) | {
            'eligible_day_fraction': len(group['all']) / available,
            'package_count': len(groups[name]),
            'by_component': {component: _summary(value) for component, value in group.items() if component != 'all'}}
    old, extra = keys['current_678_package_pool']['all'], keys['proposed_extra_2000_packages']['all']
    union = old | extra
    result = {
        'schema_version': 'conditioned_training_coverage_extension_v1',
        'status': 'completed', 'available_train_train': {
            'landpoints': len({r.parent.landpoint_id for r in references}),
            'years': sorted({int(r.parent.year) for r in references}),
            'point_years': len(references), 'eligible_days': available},
        'coverage': summaries,
        'combined_current_pool_and_extension': _summary(union) | {
            'eligible_day_fraction': len(union) / available,
            'new_unique_days_vs_current_pool': len(extra - old),
            'new_unique_days_vs_original_1000': len(extra - keys['original_1000_packages']['all']),
            'unique_day_coverage_factor_vs_current_pool': len(union) / len(old)},
        'proposed_package_range': [1001, 3000],
        'policy': {'teacher_regenerated': False, 'teacher_target_arrays_read': False,
                   'validation_or_test_targets_read': False, 'training_updates': 0,
                   'interpretation': 'Package availability/exposure only; not a lifetime audit of inherited pretraining.'},
        'elapsed_seconds': time.perf_counter() - started}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-manifest', type=Path, required=True)
    parser.add_argument('--training-contract', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.dataset_manifest, args.training_contract, args.output)
