from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_objective import (
    DIRECT_SOURCE,
    GRADIENT_SOURCES,
    SEVEN_SOURCE,
    STATE_SOURCE,
    THIRTY_SOURCE,
    TRAINABLE_FAMILIES,
    add_reachable_state_anchor_statistics,
    build_reachable_state_anchor_layout,
    coupled_gpp_trainable_tree,
    field_grouped_state_anchor_from_sufficient_statistics,
    linearized_reachable_state_anchor_loss,
    load_coupled_gpp_objective_calibration_contract,
    reachable_state_anchor_sufficient_statistics,
    replace_coupled_gpp_trainable,
    resolve_coupled_gpp_objective_coefficients,
)


def _arrays(layout, *, batch_size: int = 2):
    width = int(np.max(layout.global_state_indices)) + 1
    return (
        np.zeros((batch_size, width), dtype=np.float64),
        np.zeros((batch_size, width), dtype=np.float64),
        np.zeros((batch_size, width), dtype=np.bool_),
        np.ones((width,), dtype=np.float64),
    )


def test_coupled_gpp_objective_contract_freezes_exact_reachability_inventory():
    contract = load_coupled_gpp_objective_calibration_contract()
    layout = build_reachable_state_anchor_layout(contract)

    assert contract.sha256 == "20986e2d4fdefdddf2c89bd5a8ab83f64c680804d5b2b5b497c011bc34dcb825"
    assert contract.calibration_updates == (352, 353, 354, 355, 356, 357, 359, 360)
    assert len(layout.field_names) == 16
    assert layout.global_state_indices.shape == (41,)
    assert np.unique(layout.global_state_indices).size == 41
    assert tuple(np.bincount(layout.field_index)) == (1, 1, 11, 7, 1, 4, 1, 1, 4, 4, 1, 1, 1, 1, 1, 1)


def test_reachable_state_anchor_is_equal_field_weighted_not_coordinate_weighted():
    contract = load_coupled_gpp_objective_calibration_contract()
    layout = build_reachable_state_anchor_layout(contract)
    predicted, expected, defined, scale = _arrays(layout)
    biomass = layout.field_names.index("slowproc_stomate_previous_step_state.biomass")
    npp = layout.field_names.index("slowproc_stomate_previous_step_state.npp_daily")
    biomass_indices = layout.global_state_indices[layout.field_index == biomass]
    npp_indices = layout.global_state_indices[layout.field_index == npp]
    predicted[:, biomass_indices] = 1.0
    predicted[:, npp_indices] = 2.0
    defined[:, biomass_indices] = True
    defined[:, npp_indices] = True

    statistics = reachable_state_anchor_sufficient_statistics(
        predicted,
        expected,
        defined,
        defined,
        scale,
        layout=layout,
    )
    result = field_grouped_state_anchor_from_sufficient_statistics(
        statistics,
        field_names=layout.field_names,
    )

    assert float(result.field_loss[layout.field_names[biomass]]) == pytest.approx(1.0)
    assert float(result.field_loss[layout.field_names[npp]]) == pytest.approx(4.0)
    assert float(result.total) == pytest.approx(2.5)


def test_linearized_reachable_state_loss_matches_full_batch_gradient():
    contract = load_coupled_gpp_objective_calibration_contract()
    layout = build_reachable_state_anchor_layout(contract)
    predicted, expected, defined, scale = _arrays(layout)
    defined[:, layout.global_state_indices] = True
    expected[:, layout.global_state_indices] = np.linspace(-0.5, 0.5, 41)
    predicted[:, layout.global_state_indices] = np.linspace(0.25, 1.25, 41)
    predicted = jnp.asarray(predicted)

    def statistics(value):
        return reachable_state_anchor_sufficient_statistics(
            value,
            expected,
            defined,
            defined,
            scale,
            layout=layout,
        )

    def full_loss(value):
        return field_grouped_state_anchor_from_sufficient_statistics(
            statistics(value),
            field_names=layout.field_names,
        ).total

    total = statistics(predicted)

    def linearized(value):
        current = statistics(value)
        return linearized_reachable_state_anchor_loss(
            current,
            total,
            field_names=layout.field_names,
        )

    np.testing.assert_allclose(
        np.asarray(jax.grad(full_loss)(predicted)),
        np.asarray(jax.grad(linearized)(predicted)),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    doubled = add_reachable_state_anchor_statistics(total, total)
    assert float(doubled[layout.field_names[0]].count) == 2.0 * float(total[layout.field_names[0]].count)


def test_objective_coefficient_resolution_and_trainable_boundary():
    contract = load_coupled_gpp_objective_calibration_contract()
    records = []
    for direct in range(10, 18):
        records.append(
            {
                "source_gradients": {
                    DIRECT_SOURCE: {"l2_norm": float(direct)},
                    STATE_SOURCE: {"l2_norm": 5.0},
                    SEVEN_SOURCE: {"l2_norm": 2.0},
                    THIRTY_SOURCE: {"l2_norm": 4.0},
                }
            }
        )
    coefficients = resolve_coupled_gpp_objective_coefficients(records, contract)
    assert tuple(coefficients.resolved) == GRADIENT_SOURCES
    assert coefficients.resolved[DIRECT_SOURCE] == 1.0
    assert coefficients.resolved[STATE_SOURCE] == pytest.approx(1.35)
    assert coefficients.resolved[SEVEN_SOURCE] == pytest.approx(1.6875)
    assert coefficients.resolved[THIRTY_SOURCE] == pytest.approx(0.84375)
    assert coefficients.inside_interval

    parent = {
        "fixed": object(),
        "target_oriented": {
            "structured_gpp": {"value": np.ones((2,))},
            "coupled_gpp_carbon_feedback": {"value": np.ones((3,))},
            "retained": object(),
        },
    }
    trainable = coupled_gpp_trainable_tree(parent)
    assert tuple(trainable) == TRAINABLE_FAMILIES
    fixed = replace_coupled_gpp_trainable(parent, {})
    assert not set(TRAINABLE_FAMILIES) & set(fixed["target_oriented"])
    replaced = replace_coupled_gpp_trainable(
        parent,
        {name: {"value": value["value"] + 1.0} for name, value in trainable.items()},
    )
    assert replaced["fixed"] is parent["fixed"]
    assert replaced["target_oriented"]["retained"] is parent["target_oriented"]["retained"]
    np.testing.assert_array_equal(replaced["target_oriented"]["structured_gpp"]["value"], 2.0)


def test_trainable_replacement_accepts_jax_canonical_key_order():
    parent = {
        "fixed": jnp.asarray(3.0),
        "target_oriented": {
            "structured_gpp": {"value": jnp.ones((2,))},
            "coupled_gpp_carbon_feedback": {"value": jnp.ones((3,))},
            "retained": jnp.asarray(4.0),
        },
    }
    trainable = coupled_gpp_trainable_tree(parent)
    fixed = replace_coupled_gpp_trainable(parent, {})
    replaced = jax.jit(replace_coupled_gpp_trainable)(fixed, trainable)

    np.testing.assert_array_equal(replaced["target_oriented"]["structured_gpp"]["value"], 1.0)
    np.testing.assert_array_equal(
        replaced["target_oriented"]["coupled_gpp_carbon_feedback"]["value"],
        1.0,
    )
    assert float(replaced["target_oriented"]["retained"]) == 4.0
