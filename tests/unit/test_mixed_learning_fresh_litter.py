import json
from argparse import Namespace
from copy import deepcopy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scripts.hpc.probe_feedback_loss_calibration import _make_batch_objective
from scripts.hpc.run_mixed_learning import (
    FRESH_LITTER_PROTOCOL,
    daily_loss_composition,
    optional_litter_daily_diagnostics,
    requires_execution_policy,
    validate_configuration,
    validate_run_until,
)

jax.config.update('jax_enable_x64', True)

ROOT = Path(__file__).resolve().parents[2]
BASE_WEIGHTS = dict(
    independent_daily_flux_mse=1.,
    one_day_state_anchor=.1,
    necessary_process_supervision=.01,
)


def _fresh_protocols():
    root = ROOT / 'manifests' / 'coarse_graining'
    config = json.loads((root / 'mixed_learning_fresh_litter_v1.json').read_text())
    model = json.loads((root / config['model_protocol']).read_text())
    return config, model


def _objective(include_litter):
    def objective(parameter, sample):
        value = parameter * sample
        auxiliary = dict(
            flux_loss=value**2,
            state_loss=value,
            process_loss=value,
            canopy_stock_change={'loss': value**2},
            litterpart_feedback={'loss': value},
            hard={},
            physical={},
        )
        if include_litter:
            unweighted = (value - 3.)**2
            auxiliary['litter_stock_change'] = dict(
                unweighted_loss=unweighted,
                total=FRESH_LITTER_PROTOCOL['weight'] * unweighted,
            )
        return 0., auxiliary

    return objective


def test_runner_four_term_composition_matches_direct_loss_and_gradient_once_only():
    _, model = _fresh_protocols()
    composition = daily_loss_composition(model)
    assert composition == {
        'component_names': [
            'base', 'canopy_stock_change', 'litterpart_feedback',
            'litter_stock_change',
        ],
        'weights': [1., .1, .1, FRESH_LITTER_PROTOCOL['weight']],
        'optional_litter_stock_change': True,
    }
    samples = jnp.asarray([1., 2.], dtype=jnp.float64)
    coefficients = jnp.asarray(composition['weights'], dtype=jnp.float64)
    batch = _make_batch_objective(jax, jnp, _objective(True), BASE_WEIGHTS)

    def actual(parameter):
        return batch(parameter, samples, coefficients)[0]

    def expected(parameter):
        values = parameter * samples
        base = values**2 + .1 * values + .01 * values
        return jnp.mean(
            base + .1 * values**2 + .1 * values
            + FRESH_LITTER_PROTOCOL['weight'] * (values - 3.)**2
        )

    np.testing.assert_allclose(actual(1.), expected(1.), rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(
        jax.grad(actual)(1.), jax.grad(expected)(1.), rtol=1e-14, atol=1e-14
    )


def test_legacy_composition_remains_three_terms_and_fresh_schema_is_isolated():
    root = ROOT / 'manifests' / 'coarse_graining'
    old_config = json.loads((root / 'mixed_learning_epoch_v1.json').read_text())
    old_model = json.loads((root / old_config['model_protocol']).read_text())
    validate_configuration(old_config, old_model)
    assert daily_loss_composition(old_model) == {
        'component_names': ['base', 'canopy_stock_change', 'litterpart_feedback'],
        'weights': [1., .1, .1],
        'optional_litter_stock_change': False,
    }

    config, model = _fresh_protocols()
    validate_configuration(config, model)
    assert requires_execution_policy(config)
    assert requires_execution_policy(old_config)
    assert not requires_execution_policy({'schema_version': 'mixed_learning_v1'})

    inherited_schema = deepcopy(old_model)
    inherited_schema['litter_carbon_stock_change'] = dict(FRESH_LITTER_PROTOCOL)
    with pytest.raises(ValueError, match='declared fresh learning protocol'):
        validate_configuration(old_config, inherited_schema)

    changed = deepcopy(model)
    changed['litter_carbon_stock_change']['weight'] *= 2.
    with pytest.raises(ValueError, match='declared fresh learning protocol'):
        validate_configuration(config, changed)

    missing_seed = deepcopy(config)
    missing_seed.pop('initialization_seed')
    with pytest.raises(ValueError, match='declared fresh learning protocol'):
        validate_configuration(missing_seed, model)

    secondary_config, secondary_model = deepcopy(config), deepcopy(model)
    secondary_config['initialization_seed'] = 20260921
    secondary_model['seed'] = 20260921
    validate_configuration(secondary_config, secondary_model)
    secondary_config.pop('initialization_seed')
    with pytest.raises(ValueError, match='declared fresh learning protocol'):
        validate_configuration(secondary_config, secondary_model)


def _owner(loss, residual):
    return {
        'loss': np.asarray(loss),
        'valid': np.asarray([True, True]),
        'physical_residual': np.asarray(residual),
        'valid_count': np.asarray([1, 1]),
        'expected_count': np.asarray([1, 1]),
        'invalid_prediction_count': np.asarray([0, 0]),
        'invalid_target_count': np.asarray([0, 0]),
    }


def test_optional_litter_daily_diagnostics_are_compact_and_keep_hard_evidence():
    weight = FRESH_LITTER_PROTOCOL['weight']
    auxiliary = {'litter_stock_change': {
        'weight': np.asarray([weight, weight]),
        'scale_source_pft_index': np.asarray([13, 13]),
        'minimum_scale_gC_m2_day': np.asarray([.01, .01]),
        'unweighted_loss': np.asarray([2., 4.]),
        'total': weight * np.asarray([2., 4.]),
        'hard_invalid_count': np.asarray([0, 1]),
        'hard_empty_count': np.asarray([0, 0]),
        'per_owner': {
            'aboveground': _owner([1., 3.], [2., -2.]),
            'belowground': _owner([2., 4.], [1., -3.]),
        },
    }}
    result = optional_litter_daily_diagnostics(auxiliary)
    assert result['weight'] == weight
    assert result['scale_source_pft_index'] == 13
    assert result['minimum_scale_gC_m2_day'] == .01
    assert result['mean_unweighted_loss'] == 3.
    assert result['mean_weighted_loss'] == pytest.approx(3. * weight)
    assert result['hard_invalid_count'] == 1
    assert result['hard_empty_count'] == 0
    assert result['per_owner']['aboveground']['physical_residual_rmse'] == 2.
    assert optional_litter_daily_diagnostics({}) is None


def _segment_args(endpoint, *, integration=None, stop=None):
    return Namespace(
        run_until_update=endpoint,
        integration_updates=integration,
        stop_after_update=stop,
    )


def test_run_until_resource_segment_bounds_conflicts_and_resume_progress():
    validate_run_until(
        _segment_args(1000), 5415, profiling=False, completed_updates=0
    )
    validate_run_until(
        _segment_args(2500), 5415, profiling=False, completed_updates=1000
    )
    with pytest.raises(ValueError, match='fixed-checkpoint profiling'):
        validate_run_until(_segment_args(1000), 5415, profiling=True)
    with pytest.raises(ValueError, match='separate from integration-only'):
        validate_run_until(
            _segment_args(1000, integration=3), 5415, profiling=False
        )
    with pytest.raises(ValueError, match='separate from integration-only'):
        validate_run_until(_segment_args(2, stop=1), 5415, profiling=False)
    for endpoint in (0, 5416):
        with pytest.raises(ValueError, match='declared training budget'):
            validate_run_until(_segment_args(endpoint), 5415, profiling=False)
    with pytest.raises(ValueError, match='advance beyond'):
        validate_run_until(
            _segment_args(1000), 5415, profiling=False, completed_updates=1000
        )


def test_run_until_option_parses_as_distinct_training_limit():
    from scripts.hpc.run_mixed_learning import parser

    cli = parser()
    required = [action for action in cli._actions if action.required]
    arguments = [value for action in required for value in (action.option_strings[0], 'unused')]
    args = cli.parse_args(arguments + ['--run-until-update', '1000'])
    assert args.run_until_update == 1000
    assert args.integration_updates is None
    assert args.stop_after_update is None
