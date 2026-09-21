from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.long_horizon_objective import MaskedTrajectory
from research.daily_coarse_graining.daily_operator.shared_training import (
    _validate_state_change_scales,
    derive_shared_state_change_scales,
    make_shared_daily_objective,
    one_day_observation,
    shared_trajectories_with_initial_mask,
)
from research.daily_coarse_graining.daily_operator.shared_trajectory_loss import (
    FLUX_KEYS,
    STOCK_KEYS,
    SharedComponentTrajectories,
    SharedTrajectoryHardCounts,
)
from research.daily_coarse_graining.daily_operator.types import StateDefinedMasks, StateFeatureGroups


def _state(value=0.0):
    return StateFeatureGroups(
        **{
            name: jnp.asarray([value], dtype=jnp.float64)
            for name in StateFeatureGroups._fields
        }
    )


def _defined(value=True):
    return StateDefinedMasks(
        **{
            name: jnp.asarray([value], dtype=jnp.bool_)
            for name in StateDefinedMasks._fields
        }
    )


def _scale_bundle(state=None):
    if state is None:
        state = _state(1.0)
    return SimpleNamespace(
        state=state,
        state_loss_eligible=_defined(True),
        state_declared_dynamic=_defined(False),
    )


def test_state_change_scale_derivation_uses_frozen_state_delta(monkeypatch):
    import research.daily_coarse_graining.daily_operator.shared_training as module

    observed = {}
    sentinel = object()

    def derive(state_scale, gpp_scale, contract, *, flux_basis):
        observed.update(
            state_scale=state_scale,
            gpp_scale=gpp_scale,
            contract=contract,
            flux_basis=flux_basis,
        )
        return sentinel

    monkeypatch.setattr(module, "derive_target_oriented_scales", derive)
    state_delta = object()
    runtime = {
        "base_statistics": SimpleNamespace(
            arrays={
                "state_delta": SimpleNamespace(scale=state_delta),
                "next_state": SimpleNamespace(scale=object()),
            }
        ),
        "process_statistics": SimpleNamespace(
            outputs={"daily.fast_gpp": SimpleNamespace(scale=3.0)}
        ),
    }
    contract = object()

    assert derive_shared_state_change_scales(runtime, SimpleNamespace(contract=contract)) is sentinel
    assert observed == {
        "state_scale": state_delta,
        "gpp_scale": 3.0,
        "contract": contract,
        "flux_basis": FLUX_KEYS,
    }


def test_state_change_scale_validation_rejects_standing_stock_values(monkeypatch):
    import research.daily_coarse_graining.daily_operator.shared_training as module

    expected = _scale_bundle()
    monkeypatch.setattr(module, "derive_shared_state_change_scales", lambda runtime, head: expected)
    supplied = _scale_bundle(state=expected.state._replace(carbon=jnp.asarray([2.0])))

    with pytest.raises(ValueError, match="state_delta for carbon"):
        _validate_state_change_scales({}, object(), supplied)


def test_initial_stock_masks_are_required_for_every_change_trajectory(monkeypatch):
    import research.daily_coarse_graining.daily_operator.shared_training as module

    flux = {key: MaskedTrajectory(jnp.zeros((3, 1)), jnp.ones((3, 1), dtype=bool)) for key in FLUX_KEYS}
    stocks = {
        key: MaskedTrajectory(jnp.zeros((3, 1)), jnp.ones((3, 1), dtype=bool))
        for key in STOCK_KEYS
    }
    monkeypatch.setattr(
        module,
        "shared_component_trajectories",
        lambda observation, contract: SharedComponentTrajectories(flux, stocks),
    )
    initial = {key: jnp.asarray([[True]]) for key in STOCK_KEYS}
    initial["aboveground_litter_carbon"] = jnp.asarray([[False]])
    masked = shared_trajectories_with_initial_mask(
        {"initial_stock_defined": initial}, object()
    )

    assert np.all(masked.stocks["aboveground_biomass_carbon"].defined)
    assert not np.any(masked.stocks["aboveground_litter_carbon"].defined)
    assert masked.fluxes is flux


def test_one_day_observation_preserves_teacher_gpp_and_pft_context_masks():
    inputs = SimpleNamespace(
        pft=SimpleNamespace(
            pft_fraction=jnp.asarray([0.2, 0.8]),
            active_pft_mask=jnp.asarray([True, False]),
        )
    )
    gpp_defined = jnp.asarray([[True, False]])
    observation = one_day_observation(
        _state(),
        _defined(),
        jnp.asarray([[1.0, 9.0]]),
        gpp_defined,
        inputs,
    )

    np.testing.assert_array_equal(observation["gpp_defined"], gpp_defined[None, ...])
    np.testing.assert_allclose(observation["fraction"], [[0.2, 0.8]])
    np.testing.assert_array_equal(observation["active"], [[True, False]])


def test_daily_objective_uses_three_groups_and_merges_science_hard_counts(monkeypatch):
    import research.daily_coarse_graining.daily_operator.shared_training as module

    scales = _scale_bundle()
    monkeypatch.setattr(module, "_validate_state_change_scales", lambda runtime, head, supplied: None)
    captured = {}

    predicted_state = _state()
    predicted_state = predicted_state._replace(thermal=jnp.asarray([jnp.nan]))
    predicted_defined = _defined()
    expected_state = _state()
    expected_defined = _defined()
    predicted_defined = predicted_defined._replace(canopy=jnp.asarray([False]))
    gpp = jnp.asarray([[2.0, jnp.nan]])
    gpp_target = SimpleNamespace(
        value=jnp.asarray([[1.5, 0.0]]),
        defined=jnp.asarray([[True, False]]),
    )
    prediction = SimpleNamespace(
        named_outputs={"daily.fast_gpp": gpp},
        auxiliary_outputs={},
    )
    result = SimpleNamespace(
        prediction=prediction,
        next_state=predicted_state,
        next_state_defined=predicted_defined,
        next_discrete_state={"flag": jnp.asarray(1)},
        diagnostics={
            "water_budget_residual": jnp.asarray(0.0),
            "carbon_budget_residual": jnp.asarray(0.0),
        },
    )
    monkeypatch.setattr(
        module,
        "bind_daily_operator_dynamic_retained_inputs",
        lambda **kwargs: lambda parameters, inputs, day_inputs: result,
    )
    monkeypatch.setattr(
        module,
        "masked_mechanistic_output_loss",
        lambda *args, **kwargs: SimpleNamespace(total=jnp.asarray(99.0)),
    )
    monkeypatch.setattr(
        module,
        "necessary_process_supervision_loss",
        lambda process, contract: SimpleNamespace(total=jnp.asarray(2.0)),
    )
    monkeypatch.setattr(
        module,
        "one_day_state_sufficient_statistics",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        module,
        "one_day_state_anchor_from_sufficient_statistics",
        lambda statistics: SimpleNamespace(total=jnp.asarray(4.0)),
    )
    initial_masks = {key: jnp.asarray([[True]]) for key in STOCK_KEYS}
    monkeypatch.setattr(
        module,
        "initial_component_stock_defined",
        lambda *args, **kwargs: initial_masks,
    )

    terms = SimpleNamespace(
        flux=jnp.asarray(6.0),
        stock=jnp.asarray(9.0),
        hard_counts=SharedTrajectoryHardCounts(
            nonfinite_defined_prediction=jnp.asarray(1),
            nonfinite_defined_target=jnp.asarray(2),
            defined_mask_mismatch=jnp.asarray(3),
            missing_valid_term=jnp.asarray(4),
            total=jnp.asarray(10),
        ),
    )

    def make_science(**kwargs):
        captured["factory"] = kwargs

        class Science:
            def terms(self, parameters, initial, predicted, expected, context):
                captured["predicted"] = predicted
                captured["expected"] = expected
                return terms

        return Science()

    monkeypatch.setattr(module, "make_shared_trajectory_loss", make_science)
    assembled = SimpleNamespace(state=_state(), state_defined=_defined())
    pft = SimpleNamespace(
        pft_fraction=jnp.asarray([0.2, 0.8]),
        active_pft_mask=jnp.asarray([True, False]),
    )
    inputs = SimpleNamespace(
        continuous_state=jnp.zeros((1,)),
        pft=pft,
    )
    head = SimpleNamespace(
        input_assembler=SimpleNamespace(assemble=lambda value: assembled),
        contract=object(),
        complete_schema=object(),
        learned_schema=SimpleNamespace(fields=(SimpleNamespace(name="daily.fast_gpp"),)),
        architecture_contract=object(),
        mechanism_contract=object(),
        target_contract=object(),
    )
    sample = SimpleNamespace(
        inputs=inputs,
        day_inputs=object(),
        process_targets={"daily.fast_gpp": gpp_target},
        expected_next_state=expected_state,
        expected_next_state_defined=expected_defined,
        expected_next_discrete_state={"flag": jnp.asarray(1)},
    )
    science_scales = SimpleNamespace(
        flux_scales={key: 1.0 for key in FLUX_KEYS},
        stock_change_scales={key: 1.0 for key in STOCK_KEYS},
    )
    objective = make_shared_daily_objective(
        {"loss_scales": {}}, head, object(), scales, science_scales
    )
    loss, diagnostics = objective(jnp.asarray(0.0), sample)

    np.testing.assert_allclose(loss, 6.0)
    np.testing.assert_array_equal(captured["predicted"]["gpp_defined"], gpp_target.defined[None, ...])
    np.testing.assert_allclose(captured["predicted"]["fraction"], [[0.2, 0.8]])
    assert captured["predicted"]["initial_stock_defined"] is initial_masks
    assert captured["factory"]["trajectory_extractor"] is shared_trajectories_with_initial_mask
    assert int(diagnostics["hard"]["state_nonfinite"]) == 1
    assert int(diagnostics["hard"]["mask_mismatch"]) == 1
    assert int(diagnostics["hard"]["prediction_nonfinite"]) == 0
    assert int(diagnostics["hard"]["science_nonfinite_prediction"]) == 1
    assert int(diagnostics["hard"]["science_nonfinite_target"]) == 2
    assert int(diagnostics["hard"]["science_mask_mismatch"]) == 3
    assert int(diagnostics["hard"]["science_missing_valid_term"]) == 4
