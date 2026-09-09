"""Replay numerical equivalence and train-before-monitor lifecycle."""
import copy
import json
import pickle
from pathlib import Path
from types import SimpleNamespace as NS

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.flatten_util import ravel_pytree
from test_conditioned_attention_streamed_integration import make_batch, setup_bootstrap

from scripts.hpc import audit_conditioned_snow_free_monitor as audit
from scripts.hpc.calibrate_target_oriented_structured_gpp_joint_gradients import _horizon_gradient

jax.config.update('jax_enable_x64', True)


def proposal_for(values, gradients, weights):
    flatten = audit.persistent._make_flattened_unit_gradient(jax)
    units, reports = {}, {}
    for name, gradient in gradients.items():
        unit, norm, _ = flatten(gradient)
        units[name] = unit
        reports[name] = {'l2_norm': float(norm)}
    _, norm, _ = audit.persistent._make_flat_direction(jax)(jnp.stack(list(units.values())), jnp.array(list(weights.values())))
    return {'direction': {'source_registry': list(gradients), 'source_gradients': reports,
            'selected_minimum_norm_direction': {'normalized_simplex_weights': weights},
            'combined_gradient_norm': float(norm), 'calibration': {'full_parameter_delta_l2': 0.00038},
            'constraint_losses': {n: .2 for n in gradients}},
            'selection': {'selected_scale': 1.}}, units


def test_sparse_replay_preserves_full_matrix_update_and_every_nonzero_weight():
    values = {'x': jnp.array([.1, -.3, .005], dtype=jnp.float32)}
    gradients = {'a': {'x': jnp.array([1., 2., 3.], dtype=jnp.float32)},
                 'zero': {'x': jnp.array([4., 5., 6.], dtype=jnp.float32)},
                 'tiny': {'x': jnp.array([7., 8., 9.], dtype=jnp.float32)},
                 'b': {'x': jnp.array([2., 1., 3.], dtype=jnp.float32)}}
    weights = {'a': .4, 'zero': 0., 'tiny': 1e-20, 'b': .6}
    proposal, units = proposal_for(values, gradients, weights)
    selected = {n: units[n] for n, _ in audit.active_sources(proposal)}
    assert set(selected) == {'a', 'tiny', 'b'}
    actual, _ = audit.combine_recorded(jax, jnp, values, proposal, selected)
    direction, norm, _ = audit.persistent._make_flat_direction(jax)(jnp.stack(list(units.values())), jnp.array(list(weights.values())))
    _, unravel = ravel_pytree(values)
    expected, _ = audit.persistent._make_dtype_preserving_direct_virtual_step(jax)(values, unravel(direction), norm, np.array(.00038))
    expected = audit.persistent._scaled_parameters(jax, values, expected, 1.)
    np.testing.assert_array_equal(actual['x'], expected['x'])
    with pytest.raises(ValueError, match='every nonzero'):
        audit.combine_recorded(jax, jnp, values, proposal, {'a': units['a'], 'b': units['b']})


def test_reconstruct_uses_actual_thirty_day_caller_and_preserves_outer_mean(monkeypatch):
    bootstrap = setup_bootstrap(monkeypatch)
    trace = audit.window._make_trace_evaluator(bootstrap)
    monkeypatch.setattr(audit.integration, 'make_compact_trace_evaluator', lambda b: trace)
    values = {'weight': jnp.array([.2, -.1], dtype=jnp.float32), 'bias': jnp.array(.03, dtype=jnp.float32)}
    batches = {u: NS(thirty_day=make_batch(30, .001 * u)) for u in range(8)}
    # Independent whole-window derivative as the reference for the streamed replay.
    def gradient(v, micro, total):
        def loss(p):
            stats = bootstrap['steps']['thirty_forward'](p, micro)[0]
            return audit.window.linearized_science_window_loss(stats, total, flux_basis=bootstrap['flux_basis'])
        return jax.value_and_grad(loss)(v)
    summed = audit.recal._zero_gradient(jax, jnp, values)
    losses = []
    for batch in batches.values():
        score, _, g, _ = _horizon_gradient(jax, jnp, values, values, batch, bootstrap['steps'],
            {'thirty_gradient': gradient}, horizon=30, flux_basis=bootstrap['flux_basis'])
        summed = audit.recal._add_gradient(jax, jnp, summed, g)
        losses.append(float(score.total))
    thirty = audit.recal._mean_gradient(jax, summed, 8)
    direct = jax.tree_util.tree_map(lambda v: jnp.ones_like(v) * .2, values)
    name = audit.DIRECT + '.artifact_555'
    proposal, units = proposal_for(values, {name: direct, audit.THIRTY: thirty}, {name: .4, audit.THIRTY: .6})
    proposal['train_only_panel'] = {'artifact_groups': [list(batches)]}
    proposal['direction']['constraint_losses'][audit.THIRTY] = float(np.mean(losses))
    monkeypatch.setattr(audit.window, '_load_artifact', lambda a, c, u: NS(batch=batches[u]))
    monkeypatch.setattr(audit.persistent, '_measure_direct_artifact_gradients', lambda *a, **kw: {
        'unit_gradients': [units[name]], 'gradient_reports': [proposal['direction']['source_gradients'][name]],
        'panel': {'artifact_losses': [{'loss': .2}]}})
    emitted = []
    candidate, replay = audit.reconstruct(None, bootstrap, values, proposal, emit=lambda e, r: emitted.append(e))
    assert replay['passed'] and len(replay['horizon_records']) == 32
    assert replay['reverse_compile_count'] == 1
    assert all(r['horizon'] == 30 for r in replay['horizon_records'])
    expected, _ = audit.combine_recorded(jax, jnp, values, proposal, units)
    for a, b in zip(jax.tree_util.tree_leaves(candidate), jax.tree_util.tree_leaves(expected), strict=True):
        np.testing.assert_allclose(a, b, atol=1e-8, rtol=1e-6)


def science(contract, updates, factor=1.):
    return {'hard_gate_passed': True, 'source_mean_losses': {audit.DIRECT: .8 * factor},
            'records': [{'artifact_update': u, 'source_losses': {audit.DIRECT: .8 * factor}} for u in updates],
            'acceptance': {h: {'keys': {k: {'normalized_rmse': .8 * factor, 'normalized_signed_bias': .2 * factor,
                                          'evaluated_values': 64.} for k in contract.required_science_keys}} for h in ('7', '30')}}


@pytest.mark.parametrize('mismatch', [False, True])
def test_full_runner_replay_before_monitor_and_candidate_snapshot(tmp_path, monkeypatch, mismatch):
    contract = audit.persistent.load_conditioned_attention_persistent_training_contract()
    panel = contract.train_panels[contract.logical_updates.index(4)]
    selected_science = science(contract, panel[0])
    groups = [{'artifact_losses': [{'artifact_update': u, 'loss': .2} for u in g]} for g in panel]
    expected_groups = [{'artifact_changes': [{'artifact_update': u, 'after': .2} for u in g]} for g in panel]
    identity = {'test': True}
    reference = {'schema_version': 'conditioned_snow_free_integration_v1', 'status': 'completed',
        'admission': {'passed': True}, 'policy': {'warm_snow_free_boundary': True}, 'parent_identity': identity,
        'logical_update': 4, 'proposal': {'train_only_panel': {'artifact_groups': panel},
        'selection': {'selected_scale': 1.}, 'scale_evaluations': [{'scale': 1., 'train_only_science': selected_science,
                                                                 'artifact_groups': expected_groups}]}}
    args = NS(output=tmp_path / 'result.json', conditioned_run_dir=tmp_path, integration_report=tmp_path / 'integration.json',
              artifact_aware_persistent_contract=tmp_path / 'contract.json', parent_persistent_training_report=None,
              parent_artifact_aware_recovery=None, conditioning_contract=None, conditioned_persistent_contract=contract.source_path)
    args.integration_report.write_text(json.dumps(reference))
    (tmp_path / 'report.json').write_text(json.dumps({'identity': identity}))
    args.artifact_aware_persistent_contract.write_text(json.dumps({'contract_sha256': 'test'}))
    baseline = {'x': np.array([1.], dtype=np.float32)}
    candidate = {'x': jnp.array([.99], dtype=jnp.float32)}
    bootstrap = {'context': {'runtime': {'jax': jax}}, 'fixed_parameters': {}, 'initialized_parameters': {},
                 'steps': {}, 'direct_steps': {}, 'state_steps': {}, 'layout': None, 'flux_basis': (), 'coefficients': {}}
    def initialize(*a, **kw):
        assert kw['warm_snow_free_boundary'] and not args.output.exists()
        return bootstrap
    monkeypatch.setattr(audit.window, '_conditioned_bootstrap', initialize)
    monkeypatch.setattr(audit.window, 'load_recovery', lambda *a, **kw: {'completed_logical_update': 3, 'trainable': baseline})
    monkeypatch.setattr(audit.window, '_load_parent_state', lambda *a, **kw: (None, {}))
    monkeypatch.setattr(audit.window, 'load_coupled_gpp_conditioned_attention_contract', lambda *a: None)
    monkeypatch.setattr(audit.window, 'replace_conditioned_coupled_gpp_trainable', lambda f, v: v)
    monkeypatch.setattr(audit, 'reconstruct', lambda *a, **kw: (candidate, {'passed': True}))
    actual_science = copy.deepcopy(selected_science)
    if mismatch:
        actual_science['acceptance']['7']['keys']['gpp']['evaluated_values'] = 63.
    monkeypatch.setattr(audit.persistent, '_evaluate_detailed_monitor', lambda *a, **kw: actual_science)
    monkeypatch.setattr(audit.persistent, '_evaluate_direct_groups', lambda *a, **kw: groups)
    calls = []
    def monitor(a, b, c, *, trainable, label):
        assert a.output.with_suffix('.candidate.pkl').exists()
        calls.append(label)
        return science(contract, contract.monitor_updates, 1. if 'baseline' in label else .99)
    monkeypatch.setattr(audit.persistent, '_monitor', monitor)
    result = audit.run(args)
    assert result['status'] == 'completed' and result['passed'] != mismatch
    if mismatch:
        assert calls == [] and not args.output.with_suffix('.candidate.pkl').exists()
        assert result['classification'] == 'proposal_replay_not_admitted'
    else:
        assert calls == ['corrected_monitor_baseline', 'corrected_monitor_candidate']
        assert not result['candidate_absolute_science']['passed']  # Transfer is not final accuracy.
        with args.output.with_suffix('.candidate.pkl').open('rb') as handle:
            snapshot = pickle.load(handle)
        np.testing.assert_array_equal(snapshot['trainable']['x'], candidate['x'])
        assert snapshot['warm_snow_free_boundary'] and not snapshot['promotion_eligible']
    assert not (tmp_path / 'recovery.pkl').exists()
    json.dumps(result, allow_nan=False)


def test_actual_result_schema_and_launcher():
    # Persisted reports use evaluated_values and artifact_changes.after.
    contract = audit.persistent.load_conditioned_attention_persistent_training_contract()
    expected = science(contract, (541,))
    groups = [{'artifact_losses': [{'artifact_update': 541, 'loss': .2}]}]
    reference = [{'artifact_changes': [{'artifact_update': 541, 'after': .2}]}]
    assert audit.verify_candidate(expected, expected, groups, reference)['passed']
    groups[0]['artifact_losses'][0]['loss'] = .3
    assert not audit.verify_candidate(expected, expected, groups, reference)['passed']
    root = Path(__file__).resolve().parents[2]
    shared = (root / 'scripts/hpc/run_target_oriented_coupled_gpp_multi_group_direction.sh').read_text()
    start = shared.index('elif [[ "$PYTHON_MODULE" == "scripts.hpc.run_target_oriented_coupled_gpp_conditioned_attention_persistent_training"')
    branch = shared[start:shared.index('elif [[', start + 8)]
    command = shared[shared.index('"$PYTHON" -m "$PYTHON_MODULE"'):]
    assert '"scripts.hpc.audit_conditioned_snow_free_monitor" ||' in branch
    assert all(o in branch or o in command for a in audit._parser()._actions if a.required for o in a.option_strings)
    wrapper = (root / 'scripts/hpc/audit_conditioned_snow_free_monitor.sh').read_text()
    assert 'RUN_TIMEOUT=${RUN_TIMEOUT:-90m}' in wrapper and 'sbatch' not in wrapper
