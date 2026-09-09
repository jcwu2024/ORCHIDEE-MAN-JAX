"""Longer continuation schedule, bounded reporting, and independent split selection."""
import copy
import json
from dataclasses import replace
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
from scripts.hpc import run_conditioned_snow_free_continuation as runner
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


@pytest.mark.parametrize('preflight', [True, False])
def test_stage_two_full_runner_parent_and_independent_lifecycle(tmp_path, monkeypatch, preflight):
    import jax
    import numpy as np
    from test_conditioned_snow_free_monitor import science

    contract = load_contract(PLAN)
    values = {'x': np.array([1.], dtype=np.float32)}
    baseline = science(contract, contract.monitor_updates)
    args = NS(output=tmp_path / 'report.json', continuation_contract=PLAN, preflight_only=preflight,
              artifact_aware_persistent_contract=tmp_path / 'parent.json', parent_persistent_training_report=None,
              parent_artifact_aware_recovery=None, conditioning_contract=None)
    args.artifact_aware_persistent_contract.write_text('{"contract_sha256":"old"}')
    monkeypatch.setattr(runner, 'starting_point', lambda c: (values, baseline, science(c, c.monitor_updates, 1.2)))
    monkeypatch.setattr(runner.window, 'TRAINABLE_PARAMETER_COUNT', 1)
    monkeypatch.setattr(runner.window, '_load_parent_state', lambda *a, **kw: (None, {}))
    monkeypatch.setattr(runner.window, 'load_coupled_gpp_conditioned_attention_contract', lambda *a: None)
    monkeypatch.setattr(runner.window, '_conditioned_bootstrap', lambda *a, **kw: {'context': {'runtime': {'jax': jax}}})
    monkeypatch.setattr(runner, 'artifact_readability', lambda *a: {'readable_artifacts': 686})
    monkeypatch.setattr(runner.persistent, '_monitor', lambda *a, trainable, **kw: science(contract, contract.monitor_updates, float(trainable['x'][0])))
    independent_calls, updates, callbacks = [], [], []
    def independent(v):
        factor = float(v['x'][0])
        independent_calls.append(factor)
        return {'evaluated_windows': 24, 'science_macro_normalized_rmse': factor, 'hard_failure_count': 0,
            'absolute_short_horizon_passed': False,
            'slices': {s: {h: {'science_macro_normalized_rmse': factor, 'keys': {'gpp': {'evaluated_values': 4}}}
                          for h in ('7', '30')} for s in SLICE_SPLITS}}
    monkeypatch.setattr(runner, 'independent_evaluator', lambda *a: independent)
    class Callbacks:
        def __init__(self, *a, **kw):
            self.calls, self.records, self.reverse = {7: 0, 30: 0}, [], NS(compile_count=1)
            callbacks.append(self)
        def attach(self):
            return self
    monkeypatch.setattr(runner.integration, 'StreamedHorizonCallbacks', Callbacks)
    def train(a, attached, c, **kw):
        assert not preflight and attached is callbacks[0]
        u = kw['logical_update']
        updates.append(u)
        attached.calls[7] += 64
        attached.calls[30] += 32
        attached.records.extend([{'linearized_score_agrees': True}] * 96)
        return {'record': {'logical_update': u, 'selection': {'accepted': True}},
                'updated_trainable': {'x': kw['trainable']['x'] * .99}}
    monkeypatch.setattr(runner.persistent, '_train_only_update', train)
    result = runner.run(args)
    if preflight:
        assert updates == [] and independent_calls == [1.] and result['passed']
    else:
        assert updates == list(range(15, 35)) and result['trained_updates'] == 20
        assert result['independent_gate']['passed'] and len(independent_calls) == 2
        assert all(m['original_gate']['passed'] for m in result['monitors'])
        runner.run(args)
        assert len(independent_calls) == 2 and updates == list(range(15, 35))


def test_stage_two_loads_actual_update14_recovery_instead_of_update4(tmp_path):
    import numpy as np

    from research.daily_coarse_graining.daily_operator.corrected_continuation import save_state
    base, stage = load_contract(), load_contract(PLAN)
    plan = dict(stage.plan, parent_recovery=str(tmp_path / 'parent.pkl'), parent_report=str(tmp_path / 'parent.json'))
    save_state(plan['parent_recovery'], {'schema_version': 'corrected_continuation_state_v1', 'plan': base.plan,
        'completed_logical_update': 14, 'stopped_reason': None,
        'trainable': {'x': np.array([14.], dtype=np.float32)}, 'initial_monitor': {'baseline': 4},
        'records': [{'logical_update': u} for u in range(5, 15)],
        'monitors': [{'panel': {'baseline': u}, 'local_gate': {'passed': True}, 'cumulative_gate': {'passed': True}}
                     for u in range(5, 15)]})
    Path(plan['parent_report']).write_text('{"status":"completed","completed_logical_update":14}')
    values, expected, original = runner.starting_point(replace(stage, plan=plan))
    assert float(values['x'][0]) == 14. and expected == {'baseline': 14} and original == {'baseline': 4}
