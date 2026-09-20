from types import SimpleNamespace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator import fresh_start_training
from research.daily_coarse_graining.daily_operator.litter_change_loss import (
    LitterCarbonPools,
    derive_litter_change_scales,
    extract_litter_carbon_pools,
    litter_carbon_stock_change_loss,
    resolve_litter_carbon_layout,
)

jax.config.update("jax_enable_x64", True)


class _State(NamedTuple):
    other: object


def _contract():
    above = SimpleNamespace(
        key="slowproc_stomate_previous_step_state.litter_above",
        start=0,
        stop=12,
        shape=(1, 2, 3, 2),
        axis_names=("npts", "nlitt", "nvm", "nelements"),
        selected_pft_indices=(0, 13, 14),
    )
    below = SimpleNamespace(
        key="slowproc_stomate_previous_step_state.litter_below",
        start=12,
        stop=36,
        shape=(1, 2, 3, 2, 2),
        axis_names=("npts", "nlitt", "nvm", "ndeep", "nelements"),
        selected_pft_indices=(0, 13, 14),
    )
    return SimpleNamespace(state_leaves=(above, below), continuous_state_width=36)


def _state(above_carbon, below_carbon, *, inactive_nan=False):
    above = np.full((1, 2, 3, 2), 999.0)
    below = np.full((1, 2, 3, 2, 2), 777.0)
    above[..., 0] = np.asarray(above_carbon).reshape(1, 2, 3)
    below[..., 0] = np.asarray(below_carbon).reshape(1, 2, 3, 2)
    if inactive_nan:
        above[:, :, 0, :] = np.nan
        below[:, :, 0, :, :] = np.nan
    return np.concatenate((above.reshape(-1), below.reshape(-1)))


def _pools(value, *, defined=True, active_count=1):
    value = jnp.asarray(value, dtype=jnp.float64)
    return LitterCarbonPools(
        value=value,
        defined=jnp.broadcast_to(jnp.asarray(defined, dtype=bool), value.shape),
        active_pft_count=jnp.broadcast_to(jnp.asarray(active_count, dtype=jnp.int32), value.shape),
    )


def test_extracts_separate_area_weighted_carbon_owners_with_masks_and_elements():
    contract = _contract()
    state = _state(
        [[100.0, 10.0, 2.0], [200.0, 20.0, 4.0]],
        [
            [[100.0, 100.0], [1.0, 2.0], [5.0, 6.0]],
            [[100.0, 100.0], [3.0, 4.0], [7.0, 8.0]],
        ],
        inactive_nan=True,
    )
    # Only ICARBON contributes to this loss. Undefined/nonfinite coordinates
    # of other elements must neither contaminate arithmetic nor invalidate it.
    defined = np.ones(state.shape, dtype=bool)
    for leaf in contract.state_leaves:
        field = state[leaf.start : leaf.stop].reshape(leaf.shape)
        field_defined = defined[leaf.start : leaf.stop].reshape(leaf.shape)
        field[..., 1] = np.nan
        field_defined[..., 1] = False
    extracted = extract_litter_carbon_pools(
        state,
        defined,
        pft_fraction=jnp.asarray([0.2, 0.3, 0.5]),
        active_pft_mask=jnp.asarray([False, True, True]),
        contract=contract,
    )
    # Above: .3*(10+20)+.5*(2+4); below: .3*(1+2+3+4)+.5*(5+6+7+8).
    np.testing.assert_allclose(extracted.value, [12.0, 16.0])
    np.testing.assert_array_equal(extracted.defined, [True, True])
    np.testing.assert_array_equal(extracted.active_pft_count, [2, 2])


def test_scale_api_reports_owner_rss_raw_effective_and_coordinate_counts():
    contract = _contract()
    variance = np.ones(36)
    above = variance[:12].reshape(contract.state_leaves[0].shape)
    below = variance[12:].reshape(contract.state_leaves[1].shape)
    above[:, :, 1, 0] = [9.0, 16.0]
    below[:, :, 1, :, 0] = [[1.0, 4.0], [4.0, 16.0]]
    result = derive_litter_change_scales(
        variance,
        contract,
        scale_source_pft_index=13,
        minimum_scale_gC_m2_day=0.01,
    )
    assert result.owner_names == ("aboveground", "belowground")
    np.testing.assert_allclose(result.raw_scale, [5.0, 5.0])
    np.testing.assert_allclose(result.effective_scale, [5.0, 5.0])
    np.testing.assert_array_equal(result.coordinate_count, [2, 4])
    assert result.scale_compact_pft_index == 1
    assert result.scale_rule == "coordinate_variance_sum_sqrt"

    above[:, :, 1, 0] = 0.0
    below[:, :, 1, :, 0] = 0.0
    floored = derive_litter_change_scales(
        variance,
        contract,
        scale_source_pft_index=13,
        minimum_scale_gC_m2_day=0.01,
    )
    np.testing.assert_array_equal(floored.raw_scale, [0.0, 0.0])
    np.testing.assert_array_equal(floored.effective_scale, [0.01, 0.01])


@pytest.mark.parametrize("minimum", [0.0, np.nan])
def test_scale_api_rejects_nonpositive_or_nonfinite_declared_floor(minimum):
    with pytest.raises(ValueError, match="minimum litter-change scale"):
        derive_litter_change_scales(
            np.ones(36),
            _contract(),
            scale_source_pft_index=13,
            minimum_scale_gC_m2_day=minimum,
        )


def test_pseudo_huber_uses_shared_teacher_start_and_means_two_owner_terms():
    result = litter_carbon_stock_change_loss(
        initial_teacher=_pools([100.0, 200.0]),
        next_predicted=_pools([102.0, 198.0]),
        next_teacher=_pools([101.0, 204.0]),
        scale=jnp.asarray([2.0, 4.0]),
    )
    residual = np.asarray([0.5, -1.5])
    expected_owner = residual**2 / (np.sqrt(1.0 + residual**2) + 1.0)
    np.testing.assert_allclose(result.predicted_delta, [2.0, -2.0])
    np.testing.assert_allclose(result.teacher_delta, [1.0, 4.0])
    np.testing.assert_allclose(result.physical_residual, [1.0, -6.0])
    np.testing.assert_allclose(result.owner_loss, expected_owner)
    np.testing.assert_allclose(result.loss, np.mean(expected_owner))
    np.testing.assert_array_equal(result.valid_count, [1.0, 1.0])

    shifted = litter_carbon_stock_change_loss(
        initial_teacher=_pools([1100.0, -800.0]),
        next_predicted=_pools([1102.0, -802.0]),
        next_teacher=_pools([1101.0, -796.0]),
        scale=jnp.asarray([2.0, 4.0]),
    )
    np.testing.assert_allclose(shifted.loss, result.loss)
    np.testing.assert_allclose(shifted.physical_residual, result.physical_residual)


def test_valid_zero_target_is_supervised_and_invalid_or_empty_expectations_are_hard_counts():
    zero = litter_carbon_stock_change_loss(
        initial_teacher=_pools([4.0, 7.0]),
        next_predicted=_pools([4.0, 7.0]),
        next_teacher=_pools([4.0, 7.0]),
        scale=jnp.ones(2),
    )
    np.testing.assert_array_equal(zero.expected_count, [1.0, 1.0])
    np.testing.assert_array_equal(zero.valid_count, [1.0, 1.0])
    assert int(zero.invalid_count) == 0
    assert int(zero.empty_count) == 0
    assert float(zero.loss) == 0.0

    invalid = litter_carbon_stock_change_loss(
        initial_teacher=_pools([4.0, 7.0]),
        next_predicted=_pools([4.0, jnp.nan], defined=[True, False]),
        next_teacher=_pools([jnp.nan, 7.0], defined=[False, True]),
        scale=jnp.ones(2),
    )
    np.testing.assert_array_equal(invalid.invalid_target_count, [1, 0])
    np.testing.assert_array_equal(invalid.invalid_prediction_count, [0, 1])
    assert int(invalid.invalid_count) == 2
    assert np.isfinite(float(invalid.loss))

    empty = litter_carbon_stock_change_loss(
        initial_teacher=_pools([0.0, 0.0], active_count=0),
        next_predicted=_pools([0.0, 0.0], active_count=0),
        next_teacher=_pools([0.0, 0.0], active_count=0),
        scale=jnp.ones(2),
    )
    assert int(empty.empty_count) == 2
    assert int(empty.invalid_count) == 0


@pytest.mark.parametrize("scale", [[0.0, 1.0], [np.nan, 1.0]])
def test_loss_rejects_zero_or_nan_effective_scale_through_hard_invalid_count(scale):
    result = litter_carbon_stock_change_loss(
        initial_teacher=_pools([0.0, 0.0]),
        next_predicted=_pools([1.0, 1.0]),
        next_teacher=_pools([0.0, 0.0]),
        scale=jnp.asarray(scale),
    )
    assert int(result.invalid_count) == 1
    assert not bool(result.scale_valid[0])
    assert np.isfinite(float(result.loss))


def test_extracted_loss_has_finite_gradients_jvp_duality_finite_difference_and_fraction_gradient():
    contract = _contract()
    initial = _state(
        [[0.0, 2.0, 3.0], [0.0, 4.0, 5.0]],
        [
            [[0.0, 0.0], [1.0, 2.0], [2.0, 1.0]],
            [[0.0, 0.0], [3.0, 4.0], [4.0, 3.0]],
        ],
        inactive_nan=True,
    )
    teacher = initial + np.linspace(0.01, 0.36, 36)
    direction = jnp.linspace(-0.3, 0.4, 36)
    defined = jnp.ones(36, dtype=bool)
    active = jnp.asarray([False, True, True])

    def loss(predicted_state, fraction):
        args = dict(
            pft_fraction=fraction,
            active_pft_mask=active,
            contract=contract,
        )
        return litter_carbon_stock_change_loss(
            initial_teacher=extract_litter_carbon_pools(initial, defined, **args),
            next_predicted=extract_litter_carbon_pools(predicted_state, defined, **args),
            next_teacher=extract_litter_carbon_pools(teacher, defined, **args),
            scale=jnp.asarray([0.5, 0.75]),
        ).loss

    predicted = jnp.asarray(teacher + np.linspace(-0.02, 0.08, 36))
    fraction = jnp.asarray([0.0, 0.4, 0.6])
    gradient = jax.grad(loss, argnums=0)(predicted, fraction)
    fraction_gradient = jax.grad(loss, argnums=1)(predicted, fraction)
    _, tangent = jax.jvp(lambda value: loss(value, fraction), (predicted,), (direction,))
    np.testing.assert_allclose(tangent, jnp.vdot(gradient, direction), rtol=2e-12, atol=2e-12)
    epsilon = 1e-5
    finite_difference = (
        loss(predicted + epsilon * direction, fraction)
        - loss(predicted - epsilon * direction, fraction)
    ) / (2.0 * epsilon)
    np.testing.assert_allclose(tangent, finite_difference, rtol=2e-7, atol=2e-9)
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert bool(jnp.all(jnp.isfinite(fraction_gradient)))
    assert np.any(np.asarray(fraction_gradient[1:]) != 0.0)


def test_optional_plan_absence_and_zero_weight_preserve_disabled_default():
    contract = _contract()
    runtime = {
        "base_statistics": SimpleNamespace(
            arrays={"state_delta": SimpleNamespace(scale=np.ones(36), variance=np.ones(36))}
        )
    }
    assert fresh_start_training._litter_change_plan(runtime, contract, {}) is None
    zero = {
        "litter_carbon_stock_change": {
            "weight": 0.0,
            "penalty": "pseudo_huber",
            "scale_source_pft_index": 13,
            "minimum_scale_gC_m2_day": 0.01,
        }
    }
    # A disabled optional term does not require new owner/statistics fields.
    assert fresh_start_training._litter_change_plan({}, object(), zero) is None


def test_protocol_plan_exposes_declared_weight_and_train_derived_scale_audit():
    contract = _contract()
    variance = np.ones(36)
    above = variance[:12].reshape(contract.state_leaves[0].shape)
    below = variance[12:].reshape(contract.state_leaves[1].shape)
    above[:, :, 1, 0] = [0.009859231592171273, 0.0]
    below[:, :, 1, :, 0] = [[0.0035517143253187044, 0.0], [0.0, 0.0]]
    protocol = {
        "litter_carbon_stock_change": {
            "weight": 0.1,
            "penalty": "pseudo_huber",
            "scale_source_pft_index": 13,
            "minimum_scale_gC_m2_day": 0.01,
        }
    }
    runtime = {
        "base_statistics": SimpleNamespace(
            # Generic scales are one even for zero-variance coordinates. They
            # must not leak into the physical aggregate-change scale.
            arrays={"state_delta": SimpleNamespace(scale=np.ones(36), variance=variance)}
        )
    }
    plan = fresh_start_training._litter_change_plan(runtime, contract, protocol)
    assert plan.weight == pytest.approx(0.1)
    assert plan.penalty == "pseudo_huber"
    assert plan.scales.scale_source_pft_index == 13
    np.testing.assert_array_equal(plan.scales.coordinate_count, [2, 4])
    np.testing.assert_allclose(
        plan.scales.raw_scale,
        [0.09929366340392157, 0.059596261001162686],
        rtol=1e-14,
    )
    assert plan.layout == resolve_litter_carbon_layout(contract)


def test_make_objective_adds_enabled_term_and_keeps_disabled_return_tree_unchanged(monkeypatch):
    contract = _contract()
    start_values = _state(
        [[0.0, 2.0, 3.0], [0.0, 4.0, 5.0]],
        [
            [[0.0, 0.0], [1.0, 2.0], [2.0, 1.0]],
            [[0.0, 0.0], [3.0, 4.0], [4.0, 3.0]],
        ],
    )
    teacher_values = start_values.copy()
    predicted_values = start_values.copy()
    for leaf, teacher_increment, predicted_increment in zip(
        contract.state_leaves,
        (0.1, 0.2),
        (0.3, -0.1),
        strict=True,
    ):
        teacher = teacher_values[leaf.start : leaf.stop].reshape(leaf.shape)
        predicted = predicted_values[leaf.start : leaf.stop].reshape(leaf.shape)
        teacher[..., 0] += teacher_increment
        predicted[..., 0] += predicted_increment
    start = _State(jnp.asarray(start_values))
    expected = _State(jnp.asarray(teacher_values))
    all_defined = _State(jnp.ones(36, dtype=bool))
    pft = SimpleNamespace(
        active_pft_mask=jnp.asarray([False, True, True]),
        pft_fraction=jnp.asarray([0.0, 0.4, 0.6]),
        pft_traits=jnp.zeros((3, 1)),
    )
    assembled = SimpleNamespace(state=start, state_defined=all_defined, pft=pft)
    runtime = {
        "adapter": SimpleNamespace(input_assembler=SimpleNamespace(assemble=lambda _: assembled)),
        "loss_scales": {},
        "base_statistics": SimpleNamespace(
            arrays={"state_delta": SimpleNamespace(scale=np.ones(36), variance=np.ones(36))}
        ),
    }
    head = SimpleNamespace(
        contract=contract,
        complete_schema=object(),
        learned_schema=SimpleNamespace(fields=()),
        architecture_contract=object(),
        mechanism_contract=object(),
        target_contract=object(),
        input_normalization=SimpleNamespace(continuous_scale=np.ones(36)),
    )
    scales = SimpleNamespace(
        state=object(),
        state_loss_eligible=object(),
        state_declared_dynamic=_State(jnp.zeros(36, dtype=bool)),
        fluxes={"gpp": 1.0},
    )
    target = SimpleNamespace(value=jnp.zeros(3), defined=jnp.ones(3, dtype=bool))
    sample = SimpleNamespace(
        inputs=object(),
        day_inputs=object(),
        process_targets={"daily.fast_gpp": target},
        expected_next_state=expected,
        expected_next_state_defined=all_defined,
        expected_next_discrete_state={},
    )

    def transition(parameters, *_):
        return SimpleNamespace(
            next_state=_State(parameters),
            next_state_defined=all_defined,
            next_discrete_state={},
            prediction=SimpleNamespace(named_outputs={"daily.fast_gpp": jnp.zeros(3)}, auxiliary_outputs={}),
            diagnostics={"water_budget_residual": jnp.asarray(0.0), "carbon_budget_residual": jnp.asarray(0.0)},
        )

    def trajectories(state, defined, gpp, gpp_defined, inputs, contract):
        del state, defined, inputs, contract
        field = SimpleNamespace(value=jnp.asarray(gpp), defined=jnp.asarray(gpp_defined))
        stock = SimpleNamespace(value=jnp.asarray(0.0), defined=jnp.asarray(True))
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
    base_protocol = {
        "loss": {
            "independent_daily_flux_mse": 0.0,
            "one_day_state_anchor": 0.0,
            "necessary_process_supervision": 0.0,
        },
        "fluxes": ("gpp",),
        "admission": {"budget_absolute_tolerance": 1e-12},
    }
    augmented_protocol = {
        **base_protocol,
        "litter_carbon_stock_change": {
            "weight": 0.1,
            "penalty": "pseudo_huber",
            "scale_source_pft_index": 13,
            "minimum_scale_gC_m2_day": 0.01,
        },
    }
    default = fresh_start_training.make_objective(runtime, head, object(), scales, base_protocol)
    augmented = fresh_start_training.make_objective(runtime, head, object(), scales, augmented_protocol)
    base_loss, base_auxiliary = default(jnp.asarray(predicted_values), sample)
    loss, auxiliary = augmented(jnp.asarray(predicted_values), sample)

    assert "litter_stock_change" not in base_auxiliary
    assert "litter_change_invalid" not in base_auxiliary["hard"]
    np.testing.assert_allclose(base_loss, 0.0)
    litter = auxiliary["litter_stock_change"]
    np.testing.assert_allclose(loss, litter["total"])
    np.testing.assert_allclose(litter["total"], 0.1 * litter["unweighted_loss"])
    assert set(litter["per_owner"]) == {"aboveground", "belowground"}
    assert int(auxiliary["hard"]["litter_change_invalid"]) == 0
    assert int(auxiliary["hard"]["litter_change_empty"]) == 0
    gradient = jax.grad(lambda value: augmented(value, sample)[0])(jnp.asarray(predicted_values))
    assert bool(jnp.all(jnp.isfinite(gradient)))
    assert np.any(np.asarray(gradient) != 0.0)
