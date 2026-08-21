from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.daily_operator import (
    AUXILIARY_SUPERVISION,
    PRIMARY_SCIENCE_DIRECT,
    STATE_BUDGET_ESSENTIAL,
    MaskedTrajectory,
    build_mechanistic_learned_schema,
    compose_target_oriented_loss,
    derive_carbon_flux_identities,
    derive_carbon_stock_identities,
    load_mechanistic_output_contract,
    load_target_oriented_contract,
    necessary_process_supervision_loss,
    one_day_state_anchor,
    resolve_target_output_hierarchy,
    science_window_objective,
)
from research.daily_coarse_graining.daily_operator.training_loss import ProcessLossResult
from research.daily_coarse_graining.daily_operator.types import (
    StateDefinedMasks,
    StateFeatureGroups,
)
from tests.unit.test_daily_operator_successor_process_head import (
    extended_successor_setup,
    successor_setup,
)


def test_target_oriented_contract_is_frozen_and_absolute():
    contract = load_target_oriented_contract()
    assert contract.sha256 == contract.payload["contract_sha256"]
    assert contract.payload["acceptance"]["absolute_not_control_relative"] is True
    assert contract.loss_coefficients == {
        "one_day_state_anchor": 1.0,
        "seven_day_science": 1.0,
        "thirty_day_science": 1.0,
        "necessary_process_supervision": 0.1,
    }
    assert contract.payload["acceptance"]["short_horizon"]["7"] == {
        "normalized_rmse_maximum": 0.35,
        "absolute_normalized_signed_bias_maximum": 0.10,
    }
    assert contract.payload["acceptance"]["short_horizon"]["30"] == {
        "normalized_rmse_maximum": 0.25,
        "absolute_normalized_signed_bias_maximum": 0.07,
    }


def test_target_oriented_hierarchy_partitions_the_mechanistic_boundary():
    base = successor_setup.__wrapped__()
    extended = extended_successor_setup.__wrapped__(base)
    mechanism = load_mechanistic_output_contract()
    learned = build_mechanistic_learned_schema(extended["schema"], mechanism)
    contract = load_target_oriented_contract()
    hierarchy = resolve_target_output_hierarchy(learned, extended["schema"], contract)

    assert hierarchy.learned_field_count == 161
    assert hierarchy.learned_scalar_count == 39_006
    assert len(hierarchy.exact_or_diagnostic_fields) == 20
    assert hierarchy.learned_fields_by_category[PRIMARY_SCIENCE_DIRECT] == (
        "daily.fast_gpp",
        "daily.fast_maintenance_respiration",
    )
    assert {
        category: len(names)
        for category, names in hierarchy.learned_fields_by_category.items()
    } == {
        PRIMARY_SCIENCE_DIRECT: 2,
        STATE_BUDGET_ESSENTIAL: 93,
        AUXILIARY_SUPERVISION: 66,
    }
    classified = set().union(
        *(set(values) for values in hierarchy.learned_fields_by_category.values())
    )
    assert classified == {field.name for field in learned.fields}
    assert not classified & set(hierarchy.exact_or_diagnostic_fields)
    assert all(hierarchy.learned_fields_by_category.values())


def test_carbon_identities_use_an_independent_basis_without_duplicate_heads():
    flux = derive_carbon_flux_identities(
        {
            "gpp": jnp.asarray([10.0, 12.0]),
            "maintenance_respiration": jnp.asarray([2.0, 3.0]),
            "growth_respiration": jnp.asarray([1.0, 1.0]),
            "heterotrophic_respiration": jnp.asarray([3.0, 4.0]),
            "npp": jnp.asarray([7.0, 8.0]),
        }
    )
    np.testing.assert_allclose(flux.autotrophic_respiration, [3.0, 4.0])
    np.testing.assert_allclose(flux.npp_from_components, [7.0, 8.0])
    np.testing.assert_allclose(flux.nep, [4.0, 4.0])
    np.testing.assert_allclose(flux.nee, [-4.0, -4.0])
    np.testing.assert_allclose(flux.npp_identity_residual, 0.0)

    stock = derive_carbon_stock_identities(
        {
            "aboveground_biomass_carbon": jnp.asarray([10.0]),
            "belowground_biomass_carbon": jnp.asarray([5.0]),
            "total_live_biomass_carbon": jnp.asarray([15.0]),
            "ecosystem_carbon": jnp.asarray([25.0]),
        }
    )
    np.testing.assert_allclose(stock.total_live_biomass_carbon_from_parts, [15.0])
    np.testing.assert_allclose(stock.nonliving_ecosystem_carbon, [10.0])
    np.testing.assert_allclose(stock.total_live_identity_residual, 0.0)


def _state_groups(value: float, *, mask: bool = False):
    constructor = StateDefinedMasks if mask else StateFeatureGroups
    dtype = jnp.bool_ if mask else jnp.float64
    return constructor(
        water=jnp.asarray([value], dtype=dtype),
        carbon=jnp.asarray([value], dtype=dtype),
        thermal=jnp.asarray([value], dtype=dtype),
        canopy=jnp.asarray([value], dtype=dtype),
        soil=jnp.asarray([value], dtype=dtype),
        snow=jnp.asarray([value], dtype=dtype),
        other=jnp.asarray([value], dtype=dtype),
    )


def test_target_oriented_state_anchor_and_loss_composition_are_explicit():
    anchor = one_day_state_anchor(
        _state_groups(1.0),
        _state_groups(0.0),
        _state_groups(True, mask=True),
        _state_groups(True, mask=True),
        _state_groups(1.0),
    )
    np.testing.assert_allclose(anchor.total, 1.0)
    contract = load_target_oriented_contract()
    combined = compose_target_oriented_loss(
        one_day_state_anchor_value=anchor.total,
        seven_day_science=2.0,
        thirty_day_science=3.0,
        necessary_process_supervision=4.0,
        contract=contract,
    )
    np.testing.assert_allclose(combined.total, 6.4)
    np.testing.assert_allclose(
        combined.weighted_components["necessary_process_supervision"], 0.4
    )


def _trajectory(value: float) -> MaskedTrajectory:
    return MaskedTrajectory(
        value=jnp.full((30,), value, dtype=jnp.float64),
        defined=jnp.ones((30,), dtype=jnp.bool_),
    )


def test_science_window_optimizes_only_independent_flux_and_stock_coordinates():
    expected_fluxes = {
        "gpp": _trajectory(10.0),
        "npp": _trajectory(7.0),
        "maintenance_respiration": _trajectory(2.0),
        "growth_respiration": _trajectory(1.0),
        "heterotrophic_respiration": _trajectory(3.0),
    }
    predicted_fluxes = dict(expected_fluxes, gpp=_trajectory(11.0), npp=_trajectory(200.0))
    expected_stocks = {
        "aboveground_biomass_carbon": _trajectory(10.0),
        "belowground_biomass_carbon": _trajectory(5.0),
        "total_live_biomass_carbon": _trajectory(15.0),
        "litter_carbon": _trajectory(2.0),
        "soil_particulate_organic_carbon": _trajectory(7.0),
        "dissolved_organic_carbon": _trajectory(1.0),
        "ecosystem_carbon": _trajectory(25.0),
    }
    predicted_stocks = dict(
        expected_stocks,
        aboveground_biomass_carbon=_trajectory(11.0),
        total_live_biomass_carbon=_trajectory(16.0),
        ecosystem_carbon=_trajectory(26.0),
    )
    result = science_window_objective(
        predicted_fluxes,
        expected_fluxes,
        predicted_stocks,
        expected_stocks,
        flux_scales={
            "gpp": 1.0,
            "autotrophic_respiration": 1.0,
            "heterotrophic_respiration": 1.0,
        },
        stock_scales={
            "aboveground_biomass_carbon": 1.0,
            "belowground_biomass_carbon": 1.0,
            "nonliving_ecosystem_carbon": 1.0,
        },
        horizon=7,
    )
    np.testing.assert_allclose(result.flux_mse, 1.0 / 3.0)
    np.testing.assert_allclose(result.flux_bias_squared, 1.0 / 3.0)
    np.testing.assert_allclose(result.stock_mse, 1.0 / 3.0)
    np.testing.assert_allclose(result.stock_bias_squared, 1.0 / 3.0)
    np.testing.assert_allclose(result.total, 1.0 / 3.0)


def test_process_supervision_downweights_auxiliary_and_excludes_primary_duplicate():
    contract = load_target_oriented_contract()
    labels_by_category = contract.labels_by_category
    label_mse = {}
    label_count = {}
    category_values = {
        PRIMARY_SCIENCE_DIRECT: 1000.0,
        STATE_BUDGET_ESSENTIAL: 2.0,
        AUXILIARY_SUPERVISION: 4.0,
    }
    for category, labels in labels_by_category.items():
        for label in labels:
            label_mse[label] = jnp.asarray(category_values[category])
            label_count[label] = jnp.asarray(1.0)
    process = ProcessLossResult(
        total=jnp.asarray(0.0),
        label_mse=label_mse,
        label_count=label_count,
        field_mse={},
        field_count={},
    )
    result = necessary_process_supervision_loss(process, contract)
    expected = (17 * 2.0 + 29 * 0.25 * 4.0) / (17 + 29 * 0.25)
    np.testing.assert_allclose(result.total, expected)
    np.testing.assert_allclose(result.category_loss[PRIMARY_SCIENCE_DIRECT], 1000.0)
