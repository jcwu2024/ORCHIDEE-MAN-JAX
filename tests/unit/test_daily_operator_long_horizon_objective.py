from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_orchidee.stomate.carbon_kernels import NPARTS
from research.daily_coarse_graining.daily_operator.long_horizon_objective import (
    MaskedTrajectory,
    derive_area_weighted_biomass_carbon,
    derive_area_weighted_pft_carbon_pool,
    derive_ecosystem_carbon,
    long_horizon_biomass_productivity_objective,
    rollout_hard_gate_counts,
)

COEFFICIENTS = {
    "cumulative_flux_mse": 1.0,
    "cumulative_flux_bias_squared": 0.5,
    "terminal_stock_mse": 1.0,
    "terminal_stock_bias_squared": 0.5,
}

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _trajectory(value, defined=None):
    value = jnp.asarray(value, dtype=jnp.float64)
    if defined is None:
        defined = jnp.ones(value.shape, dtype=jnp.bool_)
    return MaskedTrajectory(value, jnp.asarray(defined, dtype=jnp.bool_))


def _objective(predicted_flux, predicted_stock, *, horizons=(1, 7, 30)):
    expected_flux = jnp.zeros_like(predicted_flux)
    expected_stock = jnp.zeros_like(predicted_stock)
    return long_horizon_biomass_productivity_objective(
        {"gpp": _trajectory(predicted_flux)},
        {"gpp": _trajectory(expected_flux)},
        {"agb": _trajectory(predicted_stock)},
        {"agb": _trajectory(expected_stock)},
        flux_scales={"gpp": jnp.asarray(2.0)},
        stock_scales={"agb": jnp.asarray(4.0)},
        coefficients=COEFFICIENTS,
        horizons=horizons,
    )


def test_biomass_derivation_uses_source_partition_and_pft_fraction():
    biomass = np.zeros((2, 3, NPARTS, 1), dtype=np.float64)
    biomass[0, 1, :, 0] = np.arange(1.0, NPARTS + 1.0)
    biomass[0, 2, :, 0] = 2.0
    biomass[1, 1, :, 0] = 5.0
    fraction = np.asarray([[0.0, 0.25, 0.75], [0.0, 0.5, 0.5]])
    active = np.asarray([[False, True, True], [False, True, False]])

    stocks = derive_area_weighted_biomass_carbon(biomass, fraction, active)

    np.testing.assert_allclose(stocks.total, [0.25 * 78.0 + 0.75 * 24.0, 30.0])
    np.testing.assert_allclose(stocks.aboveground + stocks.belowground, stocks.total)
    np.testing.assert_allclose(stocks.reserve, [0.25 * 8.0 + 0.75 * 2.0, 2.5])


def test_general_carbon_pool_reduction_and_ecosystem_identity():
    # [day, litter_pool, pft, element]
    litter = np.zeros((2, 2, 3, 1), dtype=np.float64)
    litter[:, :, 1, 0] = 2.0
    litter[:, :, 2, 0] = 4.0
    fraction = np.asarray([[0.0, 0.25, 0.75], [0.0, 0.5, 0.5]])
    active = np.asarray([[False, True, True], [False, True, False]])

    reduced = derive_area_weighted_pft_carbon_pool(
        litter,
        fraction,
        active,
        pft_axis=2,
        element_axis=3,
    )
    np.testing.assert_allclose(reduced, [7.0, 2.0])
    ecosystem = derive_ecosystem_carbon(
        {
            "live_biomass": reduced,
            "litter": reduced,
            "soil_particulate": reduced,
            "dissolved_organic": reduced,
            "canopy_interception_organic": reduced,
        }
    )
    np.testing.assert_allclose(ecosystem, 5.0 * reduced)

    with pytest.raises(ValueError, match="component drift"):
        derive_ecosystem_carbon(
            {
                "live_biomass": reduced,
                "litter": reduced,
                "soil_particulate": reduced,
                "dissolved_organic": reduced,
                "canopy_interception_organic": reduced,
                "deepC_peat": reduced,
            }
        )


def test_horizon_normalization_does_not_reward_longer_units():
    predicted_flux = jnp.ones((30, 2), dtype=jnp.float64)
    predicted_stock = jnp.full((30, 2), 2.0, dtype=jnp.float64)

    result = _objective(predicted_flux, predicted_stock)

    for horizon in (1, 7, 30):
        flux = result.by_horizon[horizon].cumulative_flux["gpp"]
        stock = result.by_horizon[horizon].terminal_stock["agb"]
        np.testing.assert_allclose(flux.mse, 0.25)
        np.testing.assert_allclose(flux.mean_bias, 0.5)
        np.testing.assert_allclose(stock.mse, 0.25)
        np.testing.assert_allclose(stock.mean_bias, 0.5)
    np.testing.assert_allclose(result.total, 0.75)


def test_cumulative_flux_requires_every_day_to_be_defined():
    value = jnp.ones((7, 2), dtype=jnp.float64)
    predicted_defined = np.ones((7, 2), dtype=bool)
    predicted_defined[3, 1] = False
    result = long_horizon_biomass_productivity_objective(
        {"npp": _trajectory(value, predicted_defined)},
        {"npp": _trajectory(jnp.zeros_like(value))},
        {"biomass": _trajectory(value)},
        {"biomass": _trajectory(jnp.zeros_like(value))},
        flux_scales={"npp": 1.0},
        stock_scales={"biomass": 1.0},
        coefficients=COEFFICIENTS,
        horizons=(7,),
    )

    assert int(result.by_horizon[7].cumulative_flux["npp"].count) == 1
    np.testing.assert_allclose(
        result.by_horizon[7].cumulative_flux["npp"].mean_bias,
        1.0,
    )


def test_objective_is_jittable_and_has_finite_long_horizon_gradient():
    expected_fluxes = {"gpp": _trajectory(jnp.zeros((30, 2)))}
    expected_stocks = {"agb": _trajectory(jnp.zeros((30, 2)))}

    def objective(parameter):
        result = long_horizon_biomass_productivity_objective(
            {"gpp": _trajectory(jnp.ones((30, 2)) * parameter)},
            expected_fluxes,
            {"agb": _trajectory(jnp.ones((30, 2)) * (2.0 * parameter))},
            expected_stocks,
            flux_scales={"gpp": 2.0},
            stock_scales={"agb": 4.0},
            coefficients=COEFFICIENTS,
        )
        return result.total

    value, gradient = jax.jit(jax.value_and_grad(objective))(jnp.asarray(1.0))

    assert np.isfinite(np.asarray(value))
    assert np.isfinite(np.asarray(gradient))
    assert float(gradient) > 0.0


def test_objective_rejects_registry_and_horizon_drift():
    value = jnp.zeros((7, 1))
    with pytest.raises(ValueError, match="registry drift"):
        long_horizon_biomass_productivity_objective(
            {"gpp": _trajectory(value)},
            {"npp": _trajectory(value)},
            {"agb": _trajectory(value)},
            {"agb": _trajectory(value)},
            flux_scales={"npp": 1.0},
            stock_scales={"agb": 1.0},
            coefficients=COEFFICIENTS,
            horizons=(7,),
        )
    with pytest.raises(ValueError, match="exceeds"):
        _objective(value, value, horizons=(30,))


def test_hard_gates_count_physical_failures_without_mixing_accuracy():
    predicted = {
        "agb": _trajectory(
            [[1.0, -1.0], [np.nan, 2.0]],
            [[True, True], [True, False]],
        )
    }
    expected = {
        "agb": _trajectory(
            [[1.0, 1.0], [1.0, 2.0]],
            [[True, True], [True, True]],
        )
    }
    counts = rollout_hard_gate_counts(
        predicted,
        expected,
        nonnegative_stock_fields=("agb",),
        budget_residuals={"carbon": jnp.asarray([0.0, 3.0e-12])},
        predicted_discrete_state={"flag": jnp.asarray([True, False])},
        expected_discrete_state={"flag": jnp.asarray([True, True])},
    )

    assert int(counts.nonfinite_defined_prediction) == 1
    assert int(counts.negative_declared_stock) == 1
    assert int(counts.defined_mask_mismatch) == 1
    assert int(counts.discrete_state_mismatch) == 1
    assert int(counts.budget_residual_failure) == 1


def test_frozen_manifest_binds_horizons_gaps_and_source_ranges():
    path = (
        REPOSITORY_ROOT
        / "manifests"
        / "coarse_graining"
        / "long_horizon_biomass_productivity_objective_v1.json"
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))

    expected_hash = manifest.pop("contract_sha256")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(canonical).hexdigest() == expected_hash
    assert manifest["horizons_days"] == [1, 7, 30]
    assert manifest["dataset"]["teacher_regeneration_forbidden"]
    assert manifest["dataset"]["historical_v5_forbidden"]
    assert manifest["coefficient_policy"]["implicit_defaults_forbidden"]
    assert all(
        value["coefficient"] == "architecture_preflight_required"
        for key, value in manifest["objective_components"].items()
        if key != "existing_one_day_process_anchor"
    )
    allocation = next(
        item
        for item in manifest["process_ledger"]
        if item["process"] == "biomass_allocation"
    )
    assert "not a frozen independent v6 process label" in allocation["supervision_gap"]

    references = []
    for section in ("cumulative_flux_fields", "terminal_stock_fields"):
        for item in manifest[section]:
            references.extend(item.get("fortran", ()))
    for item in manifest["process_ledger"]:
        references.extend(item.get("fortran", ()))
    for reference in references:
        relative, owner_and_span = reference.split("::", maxsplit=1)
        owner, span = owner_and_span.rsplit(":", maxsplit=1)
        del owner
        start, stop = (int(value) for value in span.split("-"))
        source = REPOSITORY_ROOT / "fortran_source" / "ORCHIDEE" / relative
        assert source.is_file(), reference
        line_count = len(source.read_bytes().splitlines())
        assert 0 < start <= stop <= line_count, reference
