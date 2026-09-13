from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.daily_operator import fresh_start_training


class Field(SimpleNamespace):
    pass


class State(NamedTuple):
    other: object


def _objective_setup(
    monkeypatch,
    *,
    enabled,
    litterpart_enabled=False,
    canopy_weight=None,
    litterpart_weight=None,
    next_defined=None,
    active=(True, True),
):
    canopy_leaf = SimpleNamespace(
        key="slowproc_stomate_previous_step_state.interception_storage",
        start=0,
        stop=2,
        shape=(1, 2, 1),
        axis_names=("npts", "nvm", "nelements"),
        selected_pft_indices=(0, 13),
    )
    leaves = [canopy_leaf]
    start_values = [0.0, 10.0]
    expected_values = [0.0, 12.0]
    state_delta_scale = [1.0, 5.403926065530931]
    input_scale = [1.0, 1.0]
    if litterpart_enabled:
        leaves.append(SimpleNamespace(
            key="slowproc_stomate_previous_step_state.litterpart",
            start=2,
            stop=6,
            shape=(1, 2, 2),
            axis_names=("npts", "nvm", "nlitt"),
            selected_pft_indices=(0, 13),
        ))
        start_values += [0.0, 0.0, 0.0, 0.0]
        expected_values += [100.0, 200.0, 10.0, 20.0]
        state_delta_scale += [1.0, 1.0, 1.0, 1.0]
        input_scale += [0.5, 0.5, 0.5, 0.5]
    width = len(start_values)
    contract = SimpleNamespace(state_leaves=tuple(leaves), continuous_state_width=width)
    start = State(other=jnp.asarray(start_values))
    start_defined = State(other=jnp.ones(width, dtype=bool))
    expected = State(other=jnp.asarray(expected_values))
    expected_defined = State(other=jnp.ones(width, dtype=bool))
    if next_defined is None:
        next_defined = (True,) * width
    pft = SimpleNamespace(
        active_pft_mask=jnp.asarray(active),
        pft_traits=jnp.asarray([[1.0], [0.0]]),
        pft_fraction=jnp.asarray([0.0, 1.0]),
    )
    assembled = SimpleNamespace(state=start, state_defined=start_defined, pft=pft)
    adapter = SimpleNamespace(input_assembler=SimpleNamespace(assemble=lambda _: assembled))
    runtime = {
        "adapter": adapter,
        "loss_scales": {},
        "base_statistics": SimpleNamespace(
            arrays={"state_delta": SimpleNamespace(scale=np.asarray(state_delta_scale))}
        ),
    }
    target = SimpleNamespace(value=jnp.asarray([0.0, 0.0]), defined=jnp.asarray([True, True]))
    sample = SimpleNamespace(
        inputs=object(),
        day_inputs=object(),
        process_targets={"daily.fast_gpp": target},
        expected_next_state=expected,
        expected_next_state_defined=expected_defined,
        expected_next_discrete_state={},
    )
    head = SimpleNamespace(
        contract=contract,
        complete_schema=object(),
        learned_schema=SimpleNamespace(fields=()),
        architecture_contract=object(),
        mechanism_contract=object(),
        target_contract=object(),
        input_normalization=SimpleNamespace(continuous_scale=np.asarray(input_scale)),
    )
    scales = SimpleNamespace(
        state=object(),
        state_loss_eligible=object(),
        state_declared_dynamic=SimpleNamespace(other=jnp.zeros(width, dtype=bool)),
        fluxes={"gpp": 1.0},
    )
    protocol = {
        "loss": {
            "independent_daily_flux_mse": 0.0,
            "one_day_state_anchor": 0.0,
            "necessary_process_supervision": 0.0,
        },
        "fluxes": ("gpp",),
        "admission": {"budget_absolute_tolerance": 1.0e-12},
    }
    if enabled:
        protocol["canopy_carbon_stock_change"] = {
            "weight": 0.5 if canopy_weight is None else canopy_weight,
            "scale_source_pft_index": 13,
        }
    if litterpart_enabled or litterpart_weight is not None:
        protocol["loss"]["litterpart_feedback_mse"] = 0.25 if litterpart_weight is None else litterpart_weight

    def transition(parameters, *_):
        next_state = State(other=jnp.asarray(parameters))
        return SimpleNamespace(
            next_state=next_state,
            next_state_defined=State(other=jnp.asarray(next_defined, dtype=bool)),
            next_discrete_state={},
            prediction=SimpleNamespace(
                named_outputs={"daily.fast_gpp": jnp.asarray([0.0, 0.0])},
                auxiliary_outputs={},
            ),
            diagnostics={"water_budget_residual": jnp.asarray(0.0), "carbon_budget_residual": jnp.asarray(0.0)},
        )

    def trajectories(state, defined, gpp, gpp_defined, inputs, contract):
        value = jnp.asarray(gpp)
        defined = jnp.asarray(gpp_defined, dtype=bool)
        field = Field(value=value, defined=defined)
        stock = Field(value=jnp.asarray(0.0), defined=jnp.asarray(True))
        return SimpleNamespace(
            fluxes={"gpp": field, "maintenance_respiration": field, "growth_respiration": field},
            stocks={"aboveground_biomass_carbon": stock, "belowground_biomass_carbon": stock},
        )

    monkeypatch.setattr(fresh_start_training, "bind_daily_operator_dynamic_retained_inputs", lambda **_: transition)
    monkeypatch.setattr(fresh_start_training, "masked_mechanistic_output_loss", lambda *_, **__: object())
    monkeypatch.setattr(
        fresh_start_training,
        "necessary_process_supervision_loss",
        lambda *_: SimpleNamespace(total=jnp.asarray(0.0)),
    )
    monkeypatch.setattr(fresh_start_training, "one_day_state_sufficient_statistics", lambda *_, **__: object())
    monkeypatch.setattr(
        fresh_start_training,
        "one_day_state_anchor_from_sufficient_statistics",
        lambda *_: SimpleNamespace(total=jnp.asarray(0.0)),
    )
    monkeypatch.setattr(fresh_start_training, "extract_one_day_trajectories", trajectories)
    return fresh_start_training.make_objective(runtime, head, object(), scales, protocol), sample


def test_optional_canopy_term_preserves_default_and_adds_loss_diagnostics_and_ad(monkeypatch):
    default, sample = _objective_setup(monkeypatch, enabled=False)
    enabled, _ = _objective_setup(monkeypatch, enabled=True)
    parameters = jnp.asarray([1.0, 15.0])

    base_loss, base_auxiliary = default(parameters, sample)
    loss, auxiliary = enabled(parameters, sample)
    expected_residual = 4.0  # active bare change 1 plus vegetation change 5, versus Teacher change 2.
    expected_term = (expected_residual / 5.403926065530931) ** 2
    np.testing.assert_allclose(base_loss, 0.0)
    assert "canopy_stock_change" not in base_auxiliary
    np.testing.assert_allclose(loss, 0.5 * expected_term)
    canopy = auxiliary["canopy_stock_change"]
    np.testing.assert_allclose(canopy["predicted_delta"], 6.0)
    np.testing.assert_allclose(canopy["teacher_delta"], 2.0)
    np.testing.assert_allclose(canopy["bare_physical_residual"], 1.0)
    np.testing.assert_allclose(canopy["scale"], 5.403926065530931)
    assert int(auxiliary["hard"]["canopy_change_invalid"]) == 0

    gradient = jax.grad(lambda value: enabled(value, sample)[0])(parameters)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert np.any(np.asarray(gradient) != 0.0)


def test_enabled_canopy_term_marks_an_undefined_row_hard_invalid(monkeypatch):
    enabled, sample = _objective_setup(monkeypatch, enabled=True, next_defined=(True, False))
    loss, auxiliary = enabled(jnp.asarray([0.0, jnp.nan]), sample)
    np.testing.assert_allclose(loss, 0.0)
    assert not bool(auxiliary["canopy_stock_change"]["row_valid"])
    assert int(auxiliary["hard"]["canopy_change_invalid"]) > 0


def test_zero_optional_weights_preserve_the_base_objective_and_auxiliary_shape(monkeypatch):
    default, sample = _objective_setup(monkeypatch, enabled=False)
    zero, _ = _objective_setup(
        monkeypatch,
        enabled=True,
        canopy_weight=0.0,
        litterpart_weight=0.0,
    )
    parameters = jnp.asarray([1.0, 15.0])
    default_loss, default_auxiliary = default(parameters, sample)
    zero_loss, zero_auxiliary = zero(parameters, sample)
    np.testing.assert_allclose(zero_loss, default_loss)
    assert set(zero_auxiliary) == set(default_auxiliary)
    assert "canopy_change_invalid" not in zero_auxiliary["hard"]
    assert "litterpart_feedback_invalid" not in zero_auxiliary["hard"]


def test_optional_litterpart_feedback_uses_input_scale_and_propagates_validity(monkeypatch):
    enabled, sample = _objective_setup(monkeypatch, enabled=False, litterpart_enabled=True)
    parameters = jnp.asarray([0.0, 12.0, 100.5, 201.0, 8.0, 20.0])
    loss, auxiliary = enabled(parameters, sample)

    # Four active leaf coordinates have residuals [.5, 1, -2, 0] and scale .5.
    np.testing.assert_allclose(auxiliary["litterpart_feedback"]["loss"], 5.25)
    np.testing.assert_allclose(loss, 0.25 * 5.25)
    np.testing.assert_allclose(auxiliary["litterpart_feedback"]["physical_residual"], [[[0.5, 1.0], [-2.0, 0.0]]])
    assert int(auxiliary["litterpart_feedback"]["expected_count"]) == 4
    assert int(auxiliary["hard"]["litterpart_feedback_invalid"]) == 0
    assert int(auxiliary["hard"]["litterpart_feedback_empty"]) == 0
    gradient = jax.grad(lambda value: enabled(value, sample)[0])(parameters)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert np.any(np.asarray(gradient[2:]) != 0.0)

    invalid, invalid_sample = _objective_setup(
        monkeypatch,
        enabled=False,
        litterpart_enabled=True,
        next_defined=(True, True, True, True, True, False),
    )
    _, invalid_auxiliary = invalid(parameters, invalid_sample)
    assert int(invalid_auxiliary["hard"]["litterpart_feedback_invalid"]) == 1

    empty, empty_sample = _objective_setup(
        monkeypatch,
        enabled=False,
        litterpart_enabled=True,
        active=(False, False),
    )
    _, empty_auxiliary = empty(parameters, empty_sample)
    assert int(empty_auxiliary["hard"]["litterpart_feedback_empty"]) == 1
