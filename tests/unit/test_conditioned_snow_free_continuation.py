"""Real schedule compatibility and interrupted continuation/monitor lifecycle."""
import copy
import json
from types import SimpleNamespace as NS

import jax
import numpy as np
import pytest
from test_conditioned_snow_free_monitor import science

from research.daily_coarse_graining.daily_operator import corrected_continuation as core
from scripts.hpc import run_conditioned_snow_free_continuation as runner


def initial(contract):
    return {'schema_version': 'corrected_continuation_state_v1', 'plan': contract.plan,
            'completed_logical_update': 4, 'trainable': {'x': np.array([1.], dtype=np.float32)},
            'initial_monitor': {'loss': 1.}, 'records': [], 'monitors': [], 'stopped_reason': None}


def hooks(calls):
    def propose(values, u, panel, registry):
        calls.append(u)
        return {'record': {'logical_update': u, 'selection': {'accepted': True}},
                'updated_trainable': {'x': values['x'] * .99}, 'gradient_admitted': True, 'gradient_evidence': {}}
    return dict(propose=propose, monitor=lambda v, u: {'loss': float(v['x'][0])},
                transfer=lambda a, b: {'passed': b['loss'] < a['loss']}, host=lambda v: v, emit=lambda *a: None)


def test_exact_schedule_uses_unchanged_43_source_order():
    contract = core.load_contract()
    old = core.load_conditioned_attention_persistent_training_contract()
    assert contract.train_panels[0] == old.train_panels[-1]
    assert contract.source_registries[0] == old.source_registries[-1]
    assert all(len(r) == 43 for r in contract.source_registries)
    assert contract.scales == old.scales
    assert len({u for p in contract.train_panels for g in p for u in g}) == 400


def test_ten_updates_atomic_recovery_and_completed_resume(tmp_path):
    contract, calls = core.load_contract(), []
    state = core.continue_training(contract, initial(contract), tmp_path, **hooks(calls))
    loaded = core.read_state(tmp_path / 'recovery.pkl', contract.plan)
    assert loaded['completed_logical_update'] == 14 and len(loaded['monitors']) == 10
    np.testing.assert_array_equal(state['trainable']['x'], loaded['trainable']['x'])
    core.continue_training(contract, loaded, tmp_path, **hooks(calls))
    assert calls == list(range(5, 15))


@pytest.mark.parametrize('interrupt_stage', ['pending_saved', 'update_retained'])
def test_interruption_resumes_without_repeating_proposal(tmp_path, interrupt_stage):
    contract, calls = core.load_contract(), []
    state = initial(contract)
    core.save_state(tmp_path / 'recovery.pkl', state)
    def interrupt(stage, data):
        if stage == interrupt_stage:
            raise RuntimeError('interrupted')
    kw = hooks(calls)
    kw['emit'] = interrupt
    with pytest.raises(RuntimeError, match='interrupted'):
        core.continue_training(contract, state, tmp_path, **kw)
    loaded = core.read_state(tmp_path / 'recovery.pkl', contract.plan)
    result = core.continue_training(contract, loaded, tmp_path, **hooks(calls))
    assert result['completed_logical_update'] == 14 and calls == list(range(5, 15))


@pytest.mark.parametrize('failure', ['gradient_score', 'train_only', 'monitor_transfer'])
def test_failed_candidate_preserves_last_accepted_and_stays_stopped(tmp_path, failure):
    contract, calls = core.load_contract(), []
    kw = hooks(calls)
    propose = kw['propose']
    def fail(values, u, panel, registry):
        result = propose(values, u, panel, registry)
        if u == 6:
            result['gradient_admitted'] = failure != 'gradient_score'
            result['record']['selection']['accepted'] = failure != 'train_only'
        return result
    kw['propose'] = fail
    kw['monitor'] = lambda v, u: {'loss': 2. if failure == 'monitor_transfer' and u == 6 else float(v['x'][0])}
    result = core.continue_training(contract, initial(contract), tmp_path, **kw)
    assert result['completed_logical_update'] == 5 and result['stopped_reason'] == failure
    loaded = core.read_state(tmp_path / 'recovery.pkl', contract.plan)
    core.continue_training(contract, loaded, tmp_path, **kw)
    assert calls == [5, 6]
    assert loaded['trainable']['x'][0] == np.float32(.99)


@pytest.mark.parametrize('mismatch', [None, 'evaluated_values', 'normalized_rmse', 'hard_gate'])
def test_baseline_compare_real_metric_schema(mismatch):
    contract = core.load_contract()
    expected = science(contract, contract.monitor_updates)
    actual = copy.deepcopy(expected)
    if mismatch == 'hard_gate':
        actual['hard_gate_passed'] = False
    elif mismatch:
        actual['acceptance']['30']['keys']['gpp'][mismatch] += 1
    assert runner.compare_monitor(actual, expected)['passed'] == (mismatch is None)


@pytest.mark.parametrize('preflight', [False, True])
def test_full_runner_reloads_saved_tree_and_reuses_callbacks_even_with_existing_report(tmp_path, monkeypatch, preflight):
    contract = core.load_contract()
    plan = copy.deepcopy(contract.plan)
    plan.update(candidate_snapshot=str(tmp_path / 'candidate.pkl'), monitor_reference=str(tmp_path / 'reference.json'))
    contract = core.CorrectedContinuationContract(contract.payload, contract.source_path, contract.sha256, plan)
    values = {'x': np.array([.9], dtype=np.float32)}
    core.save_state(plan['candidate_snapshot'], {'schema_version': 'corrected_train_selected_candidate_v1',
        'proposed_logical_update': 4, 'warm_snow_free_boundary': True, 'parent_identity': {}, 'trainable': values})
    baseline = science(contract, contract.monitor_updates)
    (tmp_path / 'reference.json').write_text(json.dumps({'status': 'completed', 'passed': True,
                                                      'parent_identity': {}, 'monitor_after': baseline}))
    args = NS(output=tmp_path / 'report.json', continuation_contract=None, preflight_only=preflight,
              artifact_aware_persistent_contract=tmp_path / 'contract.json', parent_persistent_training_report=None,
              parent_artifact_aware_recovery=None, conditioning_contract=None)
    args.output.write_text('{}')
    args.artifact_aware_persistent_contract.write_text('{"contract_sha256": "test"}')
    monkeypatch.setattr(core, 'load_contract', lambda *a: contract)
    monkeypatch.setattr(runner.window, 'TRAINABLE_PARAMETER_COUNT', 1)
    monkeypatch.setattr(runner.window, '_load_parent_state', lambda *a, **kw: (None, {}))
    monkeypatch.setattr(runner.window, 'load_coupled_gpp_conditioned_attention_contract', lambda *a: None)
    def bootstrap(a, **kw):
        assert not a.output.exists() and kw['warm_snow_free_boundary'] is True
        return {'context': {'runtime': {'jax': jax}}}
    monkeypatch.setattr(runner.window, '_conditioned_bootstrap', bootstrap)
    monkeypatch.setattr(runner, 'artifact_readability', lambda *a: {'readable_artifacts': 408})
    def monitor(a, b, c, *, trainable, label):
        if label == 'corrected_continuation_reload':
            np.testing.assert_array_equal(trainable['x'], values['x'])
        return science(contract, contract.monitor_updates, float(trainable['x'][0]) / float(values['x'][0]))
    monkeypatch.setattr(runner.persistent, '_monitor', monitor)
    instances, updates = [], []
    class Callbacks:
        def __init__(self, bootstrap, **kw):
            instances.append(self)
            self.calls, self.records, self.reverse = {7: 0, 30: 0}, [], NS(compile_count=1)
        def attach(self):
            return self
    def train(a, attached, c, **kw):
        assert not preflight and attached is instances[0]
        u = kw['logical_update']
        updates.append(u)
        attached.calls[7] += 64
        attached.calls[30] += 32
        attached.records.extend([{'linearized_score_agrees': True}] * 96)
        return {'record': {'logical_update': u, 'selection': {'accepted': True}},
                'updated_trainable': {'x': kw['trainable']['x'] * .99}}
    monkeypatch.setattr(runner.integration, 'StreamedHorizonCallbacks', Callbacks)
    monkeypatch.setattr(runner.persistent, '_train_only_update', train)
    result = runner.run(args)
    if preflight:
        assert result['passed'] and result['trained_updates'] == 0 and not instances
        assert not (tmp_path / 'recovery.pkl').exists()
    else:
        assert result['trained_updates'] == 10 and result['completed_logical_update'] == 14
        assert len(instances) == 1 and updates == list(range(5, 15))
        assert all(m['gradient_evidence']['calls'] == {7: 64, 30: 32} for m in result['monitors'])
        runner.run(args)
        assert updates == list(range(5, 15))  # A completed run does not train again.
