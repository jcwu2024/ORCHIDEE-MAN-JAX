"""Metadata-only independent validation selection for corrected Attention continuation."""
from __future__ import annotations

from .target_oriented_structured_gpp_fresh_screen import (
    HORIZONS,
    SLICE_SPLITS,
    _reference_identity,
    canonical_sha256,
)
from .target_oriented_training import _rank, _start_rows


def consumed_point_years(value):
    """Collect point-years across known panel schemas without reading target arrays."""
    if isinstance(value, dict):
        found = {(str(value['landpoint_id']), int(value['year']))} if 'landpoint_id' in value and 'year' in value else set()
        return found | set().union(*(consumed_point_years(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(consumed_point_years(v) for v in value))
    return set()


def build_panel(index, target_contract, plan, consumed_panels):
    config = plan['independent_validation']
    excluded = set().union(*(consumed_point_years(p) for p in consumed_panels))
    seed, count = config['selection_seed'], config['windows_per_slice_per_horizon']
    slices, selected_point_years = {}, set()
    for name, (spatial, temporal) in SLICE_SPLITS.items():
        references = [r for r in index.select(spatial_split=spatial, temporal_split=temporal)
                      if (str(r.parent.landpoint_id), int(r.parent.year)) not in excluded]
        cells = {}
        for horizon in HORIZONS:
            ordered = sorted(references, key=lambda r: (_rank(seed, horizon, ('corrected_independent', name, *_reference_identity(r))), _reference_identity(r)))
            entries, seen = [], set()
            for ref in ordered:
                point, year, parent, typed = _reference_identity(ref)
                if point in seen or (point, year) in selected_point_years or ref.day_count <= horizon:
                    continue
                seen.add(point)
                selected_point_years.add((point, year))
                entries.append({'landpoint_id': point, 'year': year, 'parent_shard_sha256': parent,
                    'typed_shard_sha256': typed, 'day_count': ref.day_count, 'horizon_days': horizon,
                    'start_rows': list(_start_rows(ref, seed=seed, update=horizon,
                        component=f'corrected_independent/{name}', count=1, horizon=horizon, exclude_first=True))})
                if len(entries) == count:
                    break
            if len(entries) != count:
                raise ValueError(f'not enough independent references for {name}/{horizon}')
            cells[str(horizon)] = entries
        slices[name] = {'spatial_split': spatial, 'temporal_split': temporal, 'horizons': cells}
    panel = {'schema_version': 'conditioned_independent_validation_panel_v1', 'selection_seed': seed,
             'target_contract_sha256': target_contract.sha256, 'windows_per_slice_per_horizon': count,
             'excluded_known_point_year_count': len(excluded), 'excluded_panel_count': len(consumed_panels),
             'target_values_read_during_selection': False, 'sealed_test_used': False, 'slices': slices}
    panel['panel_sha256'] = canonical_sha256(panel, hash_key='panel_sha256')
    return panel


def independent_transfer(before, after, config):
    changes = {}
    counts_agree = before['evaluated_windows'] == after['evaluated_windows'] == 24
    for slice_id in SLICE_SPLITS:
        for horizon in ('7', '30'):
            first, last = (v['slices'][slice_id][horizon] for v in (before, after))
            a, b = first['science_macro_normalized_rmse'], last['science_macro_normalized_rmse']
            changes[f'{slice_id}.{horizon}'] = (b - a) / max(abs(a), 1e-12)
            counts_agree &= all(last['keys'][k]['evaluated_values'] == v['evaluated_values'] for k, v in first['keys'].items())
    a, b = before['science_macro_normalized_rmse'], after['science_macro_normalized_rmse']
    macro = (b - a) / max(abs(a), 1e-12)
    checks = {'macro_improves': macro < config['macro_relative_change_maximum'],
              'all_cells_within_regression_cap': max(changes.values()) <= config['cell_macro_relative_regression_maximum'],
              'physical': before['hard_failure_count'] == after['hard_failure_count'] == 0,
              'paired_counts': counts_agree}
    return {'passed': all(checks.values()), 'checks': checks, 'macro_relative_change': macro,
            'cell_relative_changes': changes, 'absolute_passed': after['absolute_short_horizon_passed']}
