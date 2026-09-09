"""Longer continuation schedule, bounded reporting, and independent split selection."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from test_target_oriented_structured_gpp_fresh_screen import Index, Parent, Reference

from research.daily_coarse_graining.daily_operator.conditioned_independent_panel import (
    SLICE_SPLITS,
    build_panel,
    consumed_point_years,
    independent_transfer,
)
from research.daily_coarse_graining.daily_operator.continuation_reporting import ContinuationReporter
from research.daily_coarse_graining.daily_operator.corrected_continuation import load_contract
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json

PLAN = Path(__file__).resolve().parents[2] / 'manifests/coarse_graining/conditioned_snow_free_continuation_v2.json'


def test_twenty_update_schedule_reuses_data_explicitly_without_source_collisions():
    contract = load_contract(PLAN)
    assert contract.logical_updates == tuple(range(15, 35))
    assert all(len(set(u for group in panel for u in group)) == 40 for panel in contract.train_panels)
    all_ids = [u for panel in contract.train_panels for group in panel for u in group]
    assert len(all_ids) == 800 and len(set(all_ids)) == 678
    assert set(all_ids).isdisjoint(contract.monitor_updates)
    assert all(len(set(registry)) == 43 for registry in contract.source_registries)


def test_progress_writes_are_small_throttled_and_final_evidence_is_complete(tmp_path):
    timer = [0.]
    report = {'records': [dict(big='X' * 100000)], 'status': 'running', 'monitors': []}
    writes = []
    def write(path, payload):
        writes.append((path.name, copy.deepcopy(payload)))
        _atomic_json(path, payload)
    emit = ContinuationReporter(tmp_path / 'report.json', report, write=write, clock=lambda: timer[0])
    emit('horizon_complete', {'progress': {'ordinal': 0}})
    timer[0] = 1.
    emit('horizon_complete', {'progress': {'ordinal': 1}})
    assert len(writes) == 1 and 'records' not in writes[0][1]
    timer[0] = 16.
    emit('reverse_day', {'progress': {'day': 1}})
    report['records'].append({'accepted': True})
    emit('update_retained', {'completed_logical_update': 15})
    emit('completed', {'status': 'completed'})
    assert len(writes) == 4 and writes[-1][1]['records'] == report['records']
    assert json.loads((tmp_path / 'report.json').read_text())['status'] == 'completed'
    assert (tmp_path / 'report.progress.json').stat().st_size < 1000


def test_panel_excludes_all_known_point_years_and_all_test_splits():
    plan = load_contract(PLAN).plan
    references = []
    for name, (spatial, temporal) in SLICE_SPLITS.items():
        references.extend(Reference(Parent(f'{name}-{i}', 2001, f'parent-{name}-{i}', spatial, temporal),
                                    f'typed-{name}-{i}') for i in range(30))
    references.append(Reference(Parent('test-secret', 2001, 'p', 'test', 'test'), 't'))
    consumed = [{'a': [{'landpoint_id': f'{name}-{i}', 'year': 2001} for name in SLICE_SPLITS for i in range(5)]}]
    panel = build_panel(Index(references), NS(sha256='target'), plan, consumed)
    entries = [entry for cell in panel['slices'].values() for values in cell['horizons'].values() for entry in values]
    assert len(entries) == 24 and len(consumed_point_years(panel)) == 24
    assert consumed_point_years(panel).isdisjoint(consumed_point_years(consumed))
    assert not any(e['landpoint_id'] == 'test-secret' for e in entries)
    assert all(1 <= e['start_rows'][0] <= 365 - e['horizon_days'] for e in entries)
    assert panel == build_panel(Index(references[::-1]), NS(sha256='target'), plan, consumed)


@pytest.mark.parametrize('failure', [None, 'count', 'cell', 'hard'])
def test_independent_transfer_uses_all_slices_and_equal_counts(failure):
    config = load_contract(PLAN).plan['independent_validation']
    before = {'evaluated_windows': 24, 'science_macro_normalized_rmse': 1., 'hard_failure_count': 0,
        'slices': {s: {h: {'science_macro_normalized_rmse': 1., 'keys': {'gpp': {'evaluated_values': 4}}}
                      for h in ('7', '30')} for s in SLICE_SPLITS}}
    after = copy.deepcopy(before)
    after.update(science_macro_normalized_rmse=.9, absolute_short_horizon_passed=False)
    for cells in after['slices'].values():
        for cell in cells.values():
            cell['science_macro_normalized_rmse'] = .9
    if failure == 'count':
        after['slices']['joint']['30']['keys']['gpp']['evaluated_values'] = 3
    if failure == 'cell':
        after['slices']['joint']['30']['science_macro_normalized_rmse'] = 1.03
    if failure == 'hard':
        after['hard_failure_count'] = 1
    assert independent_transfer(before, after, config)['passed'] == (failure is None)
