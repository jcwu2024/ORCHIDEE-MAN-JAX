import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.long_horizon_objective import MaskedTrajectory
from research.daily_coarse_graining.daily_operator.recurrent_science_loss import (
    CANOPY_KEY,
    ScienceTrajectories,
    make_recurrent_science_loss,
)
from research.daily_coarse_graining.daily_operator.rollout_loss import RolloutLoss


def _weights():
    flux = {key: 1.0 for key in ("gpp", "npp", "autotrophic_respiration", "heterotrophic_respiration")}
    stock = {key: 1.0 for key in ("aboveground_biomass_carbon", "belowground_biomass_carbon", CANOPY_KEY)}
    return {
        "daily_flux": flux,
        "cumulative_flux": flux,
        "stock_endpoint": stock,
        "stock_trajectory": stock,
    }


def _extract(observation, _contract):
    value = jnp.asarray(observation["value"], dtype=jnp.float64)
    defined = jnp.asarray(observation["defined"], dtype=bool)
    zero = jnp.zeros_like(value)
    # Fluxes are [day, landpoint]; scalar stock trajectories are valid [day].
    stock_value = value[:, 0]
    stock_defined = defined[:, 0]
    raw = {
        "gpp": MaskedTrajectory(value, defined),
        "maintenance_respiration": MaskedTrajectory(zero, defined),
        "growth_respiration": MaskedTrajectory(zero, defined),
        "heterotrophic_respiration": MaskedTrajectory(zero, defined),
    }
    stocks = {
        "aboveground_biomass_carbon": MaskedTrajectory(stock_value, stock_defined),
        "belowground_biomass_carbon": MaskedTrajectory(0.5 * stock_value, stock_defined),
        "total_live_biomass_carbon": MaskedTrajectory(stock_value, stock_defined),
        "ecosystem_carbon": MaskedTrajectory(stock_value, stock_defined),
        CANOPY_KEY: MaskedTrajectory(stock_value, stock_defined),
    }
    return ScienceTrajectories(fluxes=raw, stocks=stocks)


def _loss():
    return make_recurrent_science_loss(
        jax,
        jnp,
        contract=object(),
        flux_scales={key: 1.0 for key in ("gpp", "npp", "autotrophic_respiration", "heterotrophic_respiration")},
        stock_daily_change_scales={
            "aboveground_biomass_carbon": 1.0,
            "belowground_biomass_carbon": 1.0,
            CANOPY_KEY: 1.0,
        },
        weights=_weights(),
        trajectory_extractor=_extract,
    )


def _observation(value, defined=None):
    value = jnp.asarray(value, dtype=jnp.float64).reshape(-1, 1)
    if defined is None:
        defined = jnp.ones_like(value, dtype=bool)
    return {"value": value, "defined": jnp.asarray(defined, dtype=bool).reshape(-1, 1)}


def test_daily_flux_error_can_cancel_at_the_cumulative_horizon():
    loss = _loss()
    result = loss.terms(None, None, _observation([2.0, 0.0]), _observation([1.0, 1.0]), None)

    np.testing.assert_allclose(result.daily_flux["gpp"].loss, 1.0)
    np.testing.assert_allclose(result.cumulative_flux["gpp"].loss, 0.0)
    np.testing.assert_allclose(
        loss(None, None, _observation([2.0, 0.0]), _observation([1.0, 1.0]), None), result.total
    )
    # The common Teacher start makes the terminal stock difference an increment error.
    np.testing.assert_allclose(result.stock_endpoint["aboveground_biomass_carbon"].loss, 0.25)
    assert result.stock_trajectory[CANOPY_KEY].loss > 0.0


def test_masked_nan_is_reported_and_has_no_nonfinite_gradient_path():
    loss = _loss()
    target = _observation([1.0, 1.0])

    def scalar(value):
        prediction = _observation(jnp.stack((value, jnp.asarray(jnp.nan))))
        prediction["defined"] = jnp.asarray([[True], [False]])
        return loss.terms(None, None, prediction, target, None).total

    result = loss.terms(
        None,
        None,
        {"value": jnp.asarray([[1.2], [jnp.nan]]), "defined": jnp.asarray([[True], [False]])},
        target,
        None,
    )
    assert int(result.daily_flux["gpp"].invalid_prediction_count) == 1
    assert int(result.cumulative_flux["gpp"].invalid_prediction_count) == 1
    assert np.isfinite(float(result.total))
    assert np.isfinite(float(jax.grad(scalar)(jnp.asarray(1.2))))


def test_cumulative_undefined_target_excludes_window_without_false_invalid_target_count():
    loss = _loss()
    target = _observation([1.0, 1.0], defined=[[True], [False]])
    result = loss.terms(None, None, _observation([1.0, 1.0]), target, None)

    cumulative = result.cumulative_flux["gpp"]
    assert int(cumulative.expected_count) == 0
    assert int(cumulative.count) == 0
    assert int(cumulative.invalid_prediction_count) == 0
    assert int(cumulative.invalid_target_count) == 0


@pytest.mark.parametrize("horizon", (1, 2, 7))
def test_rollout_loss_native_gradient_and_jvp_are_finite_for_short_recurrences(horizon):
    loss_fn = _loss()

    def step(parameter, state, discrete, day):
        next_state = state + parameter * day
        return (
            next_state,
            discrete,
            {
                "value": next_state,
                "defined": jnp.ones_like(next_state, dtype=bool),
            },
        )

    component = RolloutLoss(jax, jnp, step, loss_fn)
    days = jnp.ones((horizon, 1), dtype=jnp.float64)
    targets = _observation(np.arange(1, horizon + 1, dtype=np.float64))

    def objective(parameter):
        return component.loss(
            parameter,
            jnp.zeros((1,), dtype=jnp.float64),
            jnp.asarray(0, dtype=jnp.int32),
            days,
            targets,
            None,
        )

    parameter = jnp.asarray(1.1, dtype=jnp.float64)
    value, gradient = jax.value_and_grad(objective)(parameter)
    _, tangent = jax.jvp(objective, (parameter,), (jnp.asarray(1.0),))
    checkpointed = component.checkpointed_loss(
        parameter,
        jnp.zeros((1,), dtype=jnp.float64),
        jnp.asarray(0, dtype=jnp.int32),
        days,
        targets,
        None,
    )
    assert np.isfinite(float(value))
    assert np.isfinite(float(gradient))
    assert np.isfinite(float(tangent))
    np.testing.assert_allclose(tangent, gradient, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(checkpointed, value, rtol=1e-12, atol=1e-12)
