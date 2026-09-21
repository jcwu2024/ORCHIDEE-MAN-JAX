from collections import namedtuple
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from scripts.hpc.fresh_method_validation_pool import artifact_case, evaluate_validation_panel

Inputs = namedtuple('Inputs', 'continuous_state discrete_state day_index pft')
PFT = namedtuple('PFT', 'pft_fraction active_pft_mask')


@pytest.mark.parametrize('horizon', [7, 30])
def test_validation_adapter_keeps_teacher_state_days_pft_and_retained_bundle(horizon):
    states = np.arange(horizon * 3).reshape(1, horizon, 3)
    inputs = Inputs(states, {'phase': np.arange(horizon)[None]},
                    np.arange(100, 100 + horizon)[None],
                    PFT(np.ones((1, horizon, 2)) * .5, np.ones((1, horizon, 2), bool)))
    batch = SimpleNamespace(
        rollout_inputs=inputs, expected_next_state=states + 10,
        expected_next_state_defined=np.ones_like(states, bool),
        expected_fast_gpp=np.ones((1, horizon, 1, 2)),
        expected_fast_gpp_defined=np.ones((1, horizon, 1, 2), bool),
        expected_next_discrete_state={'phase': np.arange(horizon)[None] + 1})
    artifact = SimpleNamespace(
        slice_id='spatial', window_entry=dict(landpoint_id='x', year=2000,
            start_rows=[99], horizon_days=horizon), batch=batch,
        day_inputs={'monthly': np.arange(12), 'scalar': np.asarray(4.)})
    row = dict(slice_id='spatial', landpoint_id='x', year=2000, start_row=99, length=horizon)
    initial, discrete, (days, retained), targets, expected_discrete = artifact_case(jax, artifact, row)
    np.testing.assert_array_equal(initial, states[0, 0])
    np.testing.assert_array_equal(targets['state'], states[0] + 10)
    np.testing.assert_array_equal(days.day_index, inputs.day_index[0])
    np.testing.assert_array_equal(targets['fraction'], inputs.pft.pft_fraction[0])
    np.testing.assert_array_equal(retained['monthly'], np.broadcast_to(np.arange(12), (horizon, 12)))
    assert retained['scalar'].shape == (horizon,)
    assert discrete['phase'] == 0
    assert expected_discrete['phase'][-1] == horizon
    with pytest.raises(ValueError, match='selected window differ'):
        artifact_case(jax, artifact, dict(row, start_row=100))


def test_validation_report_does_not_mislabel_heldout_as_fixed_train_or_apply_old_gates():
    pool = SimpleNamespace(rows=[{'slice_id': name} for name in ('temporal', 'spatial', 'joint')],
                           panel={'panel_sha256': 'selection'})

    def monitor(parameters, source, rows, **options):
        assert source is pool and len(rows) == 1
        assert options['development'] is False
        assert options['apply_science_gates'] is False
        return dict(role='fixed_train_monitor', physical_gate_passed=True,
                    metric_population_complete=True)

    result = evaluate_validation_panel(monitor, None, pool)
    assert result['scientific_acceptance'] is False
    assert result['slices']['spatial']['role'] == 'held_out_validation'
    assert result['slices']['spatial']['spatial_split'] == 'validation'
    assert result['slices']['temporal']['spatial_split'] == 'train'
