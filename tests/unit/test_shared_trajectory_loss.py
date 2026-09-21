from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.long_horizon_objective import MaskedTrajectory
from research.daily_coarse_graining.daily_operator.shared_trajectory_loss import (
    FLUX_KEYS,
    NONLIVING_STOCK_KEYS,
    STOCK_KEYS,
    SharedComponentTrajectories,
    build_shared_trajectory_scales,
    make_shared_trajectory_loss,
    shared_component_trajectories,
)


def _masked(value, defined=None):
    value = jnp.asarray(value, dtype=jnp.float64)
    if defined is None:
        defined = jnp.ones(value.shape, dtype=jnp.bool_)
    return MaskedTrajectory(value, jnp.asarray(defined, dtype=jnp.bool_))


def _trajectories(days, *, fluxes=None, stocks=None, defined=None):
    fluxes = {} if fluxes is None else fluxes
    stocks = {} if stocks is None else stocks
    shape = (days, 1)
    if defined is None:
        defined = np.ones(shape, dtype=bool)
    return SharedComponentTrajectories(
        fluxes={
            key: _masked(fluxes.get(key, np.zeros(shape, dtype=np.float64)), defined)
            for key in FLUX_KEYS
        },
        stocks={
            key: _masked(stocks.get(key, np.zeros(shape, dtype=np.float64)), defined)
            for key in STOCK_KEYS
        },
    )


def _extract(observation, _contract):
    return observation


def _loss(*, flux_scale=1.0, stock_scale=1.0):
    return make_shared_trajectory_loss(
        object(),
        {key: flux_scale for key in FLUX_KEYS},
        {key: stock_scale for key in STOCK_KEYS},
        trajectory_extractor=_extract,
    )


def _terms(predicted, target):
    return _loss().terms(None, None, predicted, target, None)


def test_constant_bias_prefix_and_stock_drift_penalties_grow_with_horizon():
    short_days = 7
    long_days = 30
    short = np.arange(1.0, short_days + 1.0)[:, None]
    long = np.arange(1.0, long_days + 1.0)[:, None]
    short_result = _terms(
        _trajectories(
            short_days,
            fluxes={"gpp": np.ones((short_days, 1))},
            stocks={"aboveground_biomass_carbon": short},
        ),
        _trajectories(short_days),
    )
    long_result = _terms(
        _trajectories(
            long_days,
            fluxes={"gpp": np.ones((long_days, 1))},
            stocks={"aboveground_biomass_carbon": long},
        ),
        _trajectories(long_days),
    )

    assert long_result.flux_prefix["gpp"].loss > short_result.flux_prefix["gpp"].loss
    assert (
        long_result.stock_trajectory["aboveground_biomass_carbon"].loss
        > short_result.stock_trajectory["aboveground_biomass_carbon"].loss
    )
    np.testing.assert_allclose(
        long_result.flux_prefix["gpp"].normalized_residual[-1], np.sqrt(long_days)
    )


def test_prefix_penalty_detects_alternating_error_that_cancels_at_endpoint():
    result = _terms(
        _trajectories(2, fluxes={"gpp": np.asarray([[1.0], [-1.0]])}),
        _trajectories(2),
    )

    np.testing.assert_allclose(result.flux_prefix["gpp"].normalized_residual[-1], 0.0)
    assert result.flux_prefix["gpp"].loss > 0.0


def test_opposite_nonliving_component_errors_cannot_cancel_in_macro():
    result = _terms(
        _trajectories(
            2,
            stocks={
                "aboveground_litter_carbon": np.ones((2, 1)),
                "belowground_litter_carbon": -np.ones((2, 1)),
            },
        ),
        _trajectories(2),
    )

    assert result.stock_change["aboveground_litter_carbon"].loss > 0.0
    assert result.stock_change["belowground_litter_carbon"].loss > 0.0
    assert result.nonliving_stock > 0.0


def test_living_and_nonliving_subgroups_have_equal_macro_weight():
    living = _terms(
        _trajectories(
            1,
            stocks={
                "aboveground_biomass_carbon": np.ones((1, 1)),
                "belowground_biomass_carbon": np.ones((1, 1)),
            },
        ),
        _trajectories(1),
    )
    nonliving = _terms(
        _trajectories(
            1,
            stocks={key: np.ones((1, 1)) for key in NONLIVING_STOCK_KEYS},
        ),
        _trajectories(1),
    )

    np.testing.assert_allclose(living.living_stock, nonliving.nonliving_stock)
    np.testing.assert_allclose(living.stock, nonliving.stock)


def test_nonfinite_defined_prediction_is_hard_failure_with_finite_gradient_and_jvp():
    target = _trajectories(2)

    def objective(parameter):
        predicted = _trajectories(
            2,
            fluxes={"gpp": jnp.reshape(jnp.stack((parameter, jnp.asarray(jnp.nan))), (2, 1))},
        )
        return _loss().terms(None, None, predicted, target, None).total

    parameter = jnp.asarray(0.5, dtype=jnp.float64)
    value, gradient = jax.jit(jax.value_and_grad(objective))(parameter)
    _, tangent = jax.jvp(objective, (parameter,), (jnp.asarray(1.0, dtype=jnp.float64),))
    result = _loss().terms(
        None,
        None,
        _trajectories(2, fluxes={"gpp": np.asarray([[0.5], [np.nan]])}),
        target,
        None,
    )

    assert np.isfinite(float(value))
    assert np.isfinite(float(gradient))
    assert np.isfinite(float(tangent))
    np.testing.assert_allclose(tangent, gradient, rtol=1e-12, atol=1e-12)
    assert int(result.hard_counts.nonfinite_defined_prediction) == 1
    assert int(result.hard_counts.missing_valid_term) == 0
    assert int(result.hard_counts.total) >= 1


def test_contiguous_prefix_and_adjacent_change_masks_are_counted_exactly():
    defined = np.asarray([[True], [False], [True]])
    result = _terms(_trajectories(3, defined=defined), _trajectories(3, defined=defined))

    assert int(result.flux_daily["gpp"].count) == 2
    assert int(result.flux_prefix["gpp"].count) == 1
    assert int(result.stock_change["aboveground_biomass_carbon"].count) == 1
    assert int(result.stock_trajectory["aboveground_biomass_carbon"].count) == 2


def test_factory_rejects_zero_scale_and_cpu_builder_applies_explicit_floors():
    bad_flux_scales = {key: 1.0 for key in FLUX_KEYS}
    bad_flux_scales["gpp"] = 0.0
    with pytest.raises(ValueError, match="flux_scales.gpp"):
        make_shared_trajectory_loss(
            object(),
            bad_flux_scales,
            {key: 1.0 for key in STOCK_KEYS},
            trajectory_extractor=_extract,
        )

    target = _trajectories(
        2,
        fluxes={"gpp": np.asarray([[3.0], [4.0]])},
        stocks={"aboveground_biomass_carbon": np.asarray([[2.0], [6.0]])},
    )
    initial = {key: _masked(np.asarray([0.0])) for key in STOCK_KEYS}
    scales = build_shared_trajectory_scales(
        [(target, initial)],
        split_label="train/train",
        flux_floors={key: 0.25 for key in FLUX_KEYS},
        stock_change_floors={key: 0.5 for key in STOCK_KEYS},
    )

    np.testing.assert_allclose(scales.flux_scales["gpp"], np.sqrt(12.5))
    np.testing.assert_allclose(scales.stock_change_scales["aboveground_biomass_carbon"], np.sqrt(10.0))
    assert scales.flux_scales["maintenance_respiration"] == 0.25
    assert scales.stock_change_scales["belowground_biomass_carbon"] == 0.5
    assert scales.flux_nonzero_counts["gpp"] == 2
    assert scales.flux_nonzero_counts["heterotrophic_respiration"] == 0

    with pytest.raises(ValueError, match="train/train"):
        build_shared_trajectory_scales(
            [(target, initial)],
            split_label="validation/spatial",
            flux_floors={key: 0.25 for key in FLUX_KEYS},
            stock_change_floors={key: 0.5 for key in STOCK_KEYS},
        )


def test_mapping_order_does_not_change_normalized_loss():
    predicted = _trajectories(
        3,
        fluxes={key: np.full((3, 1), index + 1.0) for index, key in enumerate(FLUX_KEYS)},
        stocks={key: np.full((3, 1), index + 0.5) for index, key in enumerate(STOCK_KEYS)},
    )
    reversed_prediction = SharedComponentTrajectories(
        fluxes=dict(reversed(tuple(predicted.fluxes.items()))),
        stocks=dict(reversed(tuple(predicted.stocks.items()))),
    )
    target = _trajectories(3)

    np.testing.assert_allclose(_terms(predicted, target).total, _terms(reversed_prediction, target).total)


def test_shared_extractor_preserves_split_litter_and_land_area_canopy(monkeypatch):
    import research.daily_coarse_graining.daily_operator.shared_trajectory_loss as module

    scalar = _masked(np.zeros((1, 1)))
    base = SharedComponentTrajectories(
        fluxes={key: scalar for key in FLUX_KEYS},
        stocks={
            "aboveground_biomass_carbon": scalar,
            "belowground_biomass_carbon": scalar,
            "soil_particulate_organic_carbon": scalar,
            "dissolved_organic_carbon": scalar,
        },
    )
    litter_above = np.zeros((1, 1, 1, 2, 1), dtype=np.float64)
    litter_above[0, 0, 0, :, 0] = [4.0, 8.0]
    litter_below = np.zeros_like(litter_above)
    litter_below[0, 0, 0, :, 0] = [2.0, 6.0]
    canopy = np.zeros((1, 1, 2, 1), dtype=np.float64)
    canopy[0, 0, :, 0] = [3.0, 5.0]
    fields = {
        "litter_above": litter_above,
        "litter_below": litter_below,
        "interception_storage": canopy,
    }

    monkeypatch.setattr(module, "extract_biomass_productivity_trajectories", lambda *args, **kwargs: base)
    monkeypatch.setattr(
        module,
        "_trajectory_field",
        lambda _state, _defined, _contract, field: (fields[field], np.ones_like(fields[field], dtype=bool)),
    )
    observation = {
        "state": np.zeros((1, 1)),
        "defined": np.ones((1, 1), dtype=bool),
        "gpp": np.zeros((1, 2)),
        "gpp_defined": np.ones((1, 2), dtype=bool),
        "fraction": np.asarray([[0.25, 0.75]]),
        "active": np.asarray([[True, True]]),
    }
    extracted = shared_component_trajectories(
        observation, SimpleNamespace(active_pft_indices=(0, 1))
    )

    np.testing.assert_allclose(extracted.stocks["aboveground_litter_carbon"].value, [[7.0]])
    np.testing.assert_allclose(extracted.stocks["belowground_litter_carbon"].value, [[5.0]])
    np.testing.assert_allclose(extracted.stocks["canopy_interception_organic_carbon"].value, [[8.0]])

    fields["litter_above"] = np.take(fields["litter_above"], [1, 0], axis=3)
    fields["litter_below"] = np.take(fields["litter_below"], [1, 0], axis=3)
    fields["interception_storage"] = np.take(fields["interception_storage"], [1, 0], axis=2)
    permuted_observation = dict(observation)
    permuted_observation["fraction"] = np.asarray([[0.75, 0.25]])
    permuted = shared_component_trajectories(
        permuted_observation, SimpleNamespace(active_pft_indices=(0, 1))
    )
    for key in (
        "aboveground_litter_carbon",
        "belowground_litter_carbon",
        "canopy_interception_organic_carbon",
    ):
        np.testing.assert_allclose(permuted.stocks[key].value, extracted.stocks[key].value)
