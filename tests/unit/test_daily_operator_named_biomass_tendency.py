from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.gradient_feasibility import (
    GRADIENT_FEASIBILITY_LABELS,
    _minimum_norm_simplex_weights,
    classify_direct_trust_region,
    classify_gradient_feasibility,
    classify_realized_step_audit,
    gradient_direction_diagnostics,
)
from research.daily_coarse_graining.daily_operator.long_horizon_objective import (
    MaskedTrajectory,
)
from research.daily_coarse_graining.daily_operator.named_biomass_tendency import (
    NAMED_BIOMASS_TENDENCY_FIELDS,
    biomass_stock_identity_residual,
    biomass_stock_identity_roundoff_audit,
    fit_named_biomass_tendency_scales,
    named_biomass_tendency_objective,
)
from research.daily_coarse_graining.daily_operator.parameter_geometry import (
    SUCCESSOR_DECODER_LEAVES,
    SUCCESSOR_PARAMETER_BLOCKS,
    exclude_successor_decoder_leaf,
    exclude_successor_parameter_block,
    successor_decoder_leaf_geometry_report,
    successor_parameter_geometry_report,
    validate_successor_parameter_blocks,
)
from research.daily_coarse_graining.daily_operator.successor_training import (
    CANDIDATE_OBJECTIVE_COMPONENTS,
    CUMULATIVE_FLUX_FIELDS,
    TERMINAL_STOCK_FIELDS,
    BiomassProductivityTrajectories,
    calibrate_named_biomass_gradient_coefficient,
    candidate_successor_training_objective,
)
from research.daily_coarse_graining.daily_operator.successor_training_assets import (
    load_named_biomass_tendency_scale_asset,
)
from scripts.hpc.prepare_successor_named_biomass_tendency_scales import (
    _canonical_sha256,
    _select_records,
)
from scripts.hpc.probe_successor_direct_trust_region import (
    _make_combined_direction,
    _make_direct_virtual_step,
)
from scripts.hpc.probe_successor_multibatch_direction import (
    _empty_accumulators,
    _frozen_scalarization_seeds,
    _make_flat_direction,
    _make_flat_gram,
    _make_flattened_unit_gradient,
    _make_gradient_accumulator,
    _make_nonfinite_counter,
    classify_active_set_recalibration,
    classify_cumulative_microstep_smoke,
    classify_group_aware_direction,
    classify_repeated_active_set_smoke,
    classify_second_state_recalibration,
)
from scripts.hpc.probe_successor_parameter_geometry import (
    classify_parameter_geometry_attribution,
)
from scripts.hpc.probe_successor_realized_step import _make_proposed_directions
from scripts.hpc.run_successor_direct_direction_smoke import (
    classify_direct_direction_smoke,
)
from scripts.hpc.run_successor_gradient_balanced_smoke import (
    CANDIDATE,
    CONTROL,
    _make_candidate_apply_step,
    _make_gradient_gram,
    classify_gradient_balanced_smoke,
    selected_validation_subset,
)
from scripts.hpc.run_successor_named_biomass_objective_ab import (
    OBJECTIVES,
    _scale_asset_identity,
    classify_objective_ab,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
COEFFICIENTS = {"tendency_mse": 1.0, "tendency_bias_squared": 1.0}
SCALES = {
    horizon: {
        "aboveground_biomass_carbon": 2.0,
        "belowground_biomass_carbon": 4.0,
    }
    for horizon in (1, 7, 30)
}


def test_successor_parameter_geometry_blocks_are_complete_and_projectable():
    keys = tuple(
        key for members in SUCCESSOR_PARAMETER_BLOCKS.values() for key in members
    )
    parameters = {
        key: {"value": jnp.ones((2,), dtype=jnp.float64)} for key in keys
    }
    direction = {
        key: {"value": jnp.full((2,), index + 1.0, dtype=jnp.float64)}
        for index, key in enumerate(keys)
    }

    validate_successor_parameter_blocks(parameters)
    report = successor_parameter_geometry_report(
        parameters,
        direction,
        step_norm=5.0e-4,
    )
    projected = exclude_successor_parameter_block(direction, "output_decoder")

    assert report["block_count"] == 7
    assert report["top_level_key_count"] == len(keys) == 23
    assert report["parameter_count"] == 2 * len(keys)
    assert report["direction_squared_fraction_sum"] == pytest.approx(1.0)
    assert report["nonfinite_value_count"] == 0
    assert all(
        np.all(np.asarray(projected[key]["value"]) == 0.0)
        for key in SUCCESSOR_PARAMETER_BLOCKS["output_decoder"]
    )
    assert np.any(np.asarray(projected["process_blocks"]["value"]) != 0.0)


def test_successor_decoder_leaf_geometry_uses_full_direction_norm():
    keys = tuple(
        key for members in SUCCESSOR_PARAMETER_BLOCKS.values() for key in members
    )
    parameters = {key: jnp.ones((2,), dtype=jnp.float64) for key in keys}
    direction = {key: jnp.ones((2,), dtype=jnp.float64) for key in keys}
    for top_level in ("decoder_first", "decoder_second"):
        parameters[top_level] = {
            "kernel": jnp.ones((2, 2), dtype=jnp.float64),
            "bias": jnp.ones((2,), dtype=jnp.float64),
        }
        direction[top_level] = {
            "kernel": jnp.ones((2, 2), dtype=jnp.float64),
            "bias": jnp.ones((2,), dtype=jnp.float64),
        }

    report = successor_decoder_leaf_geometry_report(
        parameters,
        direction,
        step_norm=5.0e-4,
    )
    projected = exclude_successor_decoder_leaf(
        direction, "decoder_second.bias"
    )

    assert report["block_count"] == len(SUCCESSOR_DECODER_LEAVES) == 4
    assert report["top_level_key_count"] == 2
    assert 0.0 < report["direction_squared_fraction_sum"] < 1.0
    assert np.all(np.asarray(projected["decoder_second"]["bias"]) == 0.0)
    assert np.any(np.asarray(projected["decoder_second"]["kernel"]) != 0.0)


def _trajectory(value, defined=None):
    array = jnp.asarray(value, dtype=jnp.float64)
    mask = jnp.ones(array.shape, dtype=jnp.bool_) if defined is None else defined
    return MaskedTrajectory(array, jnp.asarray(mask, dtype=jnp.bool_))


def _stocks(above, below, *, above_defined=None, below_defined=None):
    return {
        "aboveground_biomass_carbon": _trajectory(above, above_defined),
        "belowground_biomass_carbon": _trajectory(below, below_defined),
    }


def test_teacher_teacher_named_biomass_tendency_loss_is_exact_zero():
    teacher = _stocks(jnp.ones((30, 3)), jnp.ones((30, 3)) * 2.0)
    result = named_biomass_tendency_objective(
        teacher,
        teacher,
        scales=SCALES,
        coefficients=COEFFICIENTS,
    )

    assert float(result.total) == 0.0
    assert float(result.tendency_mse) == 0.0
    assert float(result.tendency_bias_squared) == 0.0


def test_named_biomass_tendency_uses_equal_fields_and_horizons():
    expected = _stocks(jnp.zeros((30, 2)), jnp.zeros((30, 2)))
    predicted = _stocks(jnp.ones((30, 2)), jnp.ones((30, 2)) * 2.0)
    result = named_biomass_tendency_objective(
        predicted,
        expected,
        scales=SCALES,
        coefficients=COEFFICIENTS,
    )

    for horizon in (1, 7, 30):
        for name in NAMED_BIOMASS_TENDENCY_FIELDS:
            error = result.by_horizon[horizon].errors[name]
            assert float(error.mse) == pytest.approx(0.25)
            assert float(error.mean_bias) == pytest.approx(0.5)
        assert float(result.by_horizon[horizon].macro_mse) == pytest.approx(0.25)
        assert float(result.by_horizon[horizon].macro_bias_squared) == pytest.approx(
            0.25
        )
    assert float(result.total) == pytest.approx(0.5)


def test_named_biomass_tendency_excludes_undefined_field_from_macro():
    expected = _stocks(jnp.zeros((1, 2)), jnp.zeros((1, 2)))
    predicted = _stocks(
        jnp.ones((1, 2)) * 2.0,
        jnp.ones((1, 2)) * 100.0,
        below_defined=jnp.zeros((1, 2), dtype=jnp.bool_),
    )
    result = named_biomass_tendency_objective(
        predicted,
        expected,
        scales={1: SCALES[1]},
        coefficients=COEFFICIENTS,
        horizons=(1,),
    )

    assert int(
        result.by_horizon[1].errors["belowground_biomass_carbon"].count
    ) == 0
    assert float(result.tendency_mse) == pytest.approx(1.0)
    assert float(result.tendency_bias_squared) == pytest.approx(1.0)


def test_terminal_residual_is_exactly_the_shared_start_tendency_residual():
    start = jnp.asarray([10.0, 20.0])
    predicted = jnp.asarray([13.0, 19.0])
    teacher = jnp.asarray([11.0, 18.0])

    terminal_residual = predicted - teacher
    tendency_residual = (predicted - start) - (teacher - start)

    np.testing.assert_array_equal(terminal_residual, tendency_residual)


def test_fit_named_biomass_tendency_scales_uses_train_change_and_floor():
    initial = _stocks([10.0, 20.0, 30.0], [5.0, 10.0, 15.0])
    above = np.broadcast_to(np.asarray([10.0, 20.0, 30.0]), (30, 3)).copy()
    below = np.broadcast_to(np.asarray([5.0, 10.0, 15.0]), (30, 3)).copy()
    above[0] += np.asarray([1.0, 2.0, 3.0])
    above[6] += 0.01
    above[29] += np.asarray([2.0, 4.0, 6.0])
    below[0] += 0.01
    below[6] += np.asarray([2.0, 4.0, 6.0])
    below[29] += np.asarray([3.0, 6.0, 9.0])
    teacher = _stocks(above, below)

    result = fit_named_biomass_tendency_scales(
        initial,
        teacher,
        stock_scales={
            "aboveground_biomass_carbon": 100.0,
            "belowground_biomass_carbon": 200.0,
        },
        floor_ratio=1.0e-3,
    )

    assert result.scales[1]["aboveground_biomass_carbon"] == pytest.approx(1.0)
    assert result.scales[7]["aboveground_biomass_carbon"] == pytest.approx(0.1)
    assert result.scales[1]["belowground_biomass_carbon"] == pytest.approx(0.2)
    assert result.scales[7]["belowground_biomass_carbon"] == pytest.approx(2.0)
    assert result.audit["records"][7]["aboveground_biomass_carbon"][
        "floor_applied"
    ]
    assert result.audit["records"][1]["aboveground_biomass_carbon"]["count"] == 3


def test_named_biomass_tendency_is_jittable_with_finite_gradient():
    expected = _stocks(jnp.zeros((30, 2)), jnp.zeros((30, 2)))

    def objective(parameter):
        predicted = _stocks(
            jnp.ones((30, 2)) * parameter,
            jnp.ones((30, 2)) * (2.0 * parameter),
        )
        return named_biomass_tendency_objective(
            predicted,
            expected,
            scales=SCALES,
            coefficients=COEFFICIENTS,
        ).total

    value, gradient = jax.jit(jax.value_and_grad(objective))(jnp.asarray(1.0))

    assert np.isfinite(np.asarray(value))
    assert np.isfinite(np.asarray(gradient))
    assert float(gradient) > 0.0


def _candidate_trajectories(*, fluxes=None, stocks=None):
    flux_values = {} if fluxes is None else fluxes
    stock_values = {} if stocks is None else stocks
    return BiomassProductivityTrajectories(
        fluxes={
            name: _trajectory(flux_values.get(name, np.zeros((30, 1))))
            for name in CUMULATIVE_FLUX_FIELDS
        },
        stocks={
            name: _trajectory(stock_values.get(name, np.zeros((30, 1))))
            for name in TERMINAL_STOCK_FIELDS
        },
    )


def _candidate_objective(predicted, expected):
    return candidate_successor_training_objective(
        0.0,
        predicted,
        expected,
        flux_scales={name: 1.0 for name in CUMULATIVE_FLUX_FIELDS},
        stock_scales={name: 1.0 for name in TERMINAL_STOCK_FIELDS},
        tendency_scales={
            horizon: {
                "aboveground_biomass_carbon": 1.0,
                "belowground_biomass_carbon": 1.0,
            }
            for horizon in (1, 7, 30)
        },
        coefficients={name: 1.0 for name in CANDIDATE_OBJECTIVE_COMPONENTS},
    )


def test_candidate_objective_reports_but_does_not_optimize_long_maintenance():
    expected = _candidate_trajectories()
    predicted = _candidate_trajectories(
        fluxes={"maintenance_respiration": np.ones((30, 1))}
    )

    result = _candidate_objective(predicted, expected)

    assert float(result.total) == 0.0
    assert (
        float(
            result.long_horizon_diagnostics.by_horizon[
                30
            ].cumulative_flux["maintenance_respiration"].mse
        )
        > 0.0
    )


def test_candidate_objective_optimizes_npp_and_named_biomass():
    expected = _candidate_trajectories()
    npp = _candidate_objective(
        _candidate_trajectories(fluxes={"npp": np.ones((30, 1))}),
        expected,
    )
    biomass = _candidate_objective(
        _candidate_trajectories(
            stocks={"aboveground_biomass_carbon": np.ones((30, 1))}
        ),
        expected,
    )

    assert float(npp.total) > 0.0
    assert float(npp.cumulative_flux_mse) > 0.0
    assert float(biomass.total) > 0.0
    assert float(biomass.tendency_mse) > 0.0


def test_named_biomass_gradient_coefficient_equalizes_geometric_mean():
    result = calibrate_named_biomass_gradient_coefficient(
        [10.0, 10.0, 10.0],
        [2.0, 2.0, 2.0],
        [8.0, 8.0, 8.0],
        target_ratio_interval=(0.1, 10.0),
        coefficient_interval=(1.0e-6, 10.0),
    )

    assert result.coefficient == pytest.approx(2.5)
    assert result.achieved_stock_to_npp_ratios == pytest.approx(
        {
            "aboveground_biomass_carbon": 0.5,
            "belowground_biomass_carbon": 2.0,
        }
    )
    assert result.passed


def test_named_biomass_gradient_coefficient_rejects_imbalanced_stocks():
    result = calibrate_named_biomass_gradient_coefficient(
        [10.0, 10.0],
        [0.001, 0.001],
        [1000.0, 1000.0],
        target_ratio_interval=(0.1, 10.0),
        coefficient_interval=(1.0e-6, 10.0),
    )

    assert not result.ratios_within_interval
    assert not result.passed


def test_biomass_stock_identity_residual_is_exact_and_masked():
    stocks = _stocks([[1.0, 2.0]], [[3.0, 4.0]])
    stocks["total_live_biomass_carbon"] = _trajectory([[4.0, 6.0]])
    result = biomass_stock_identity_residual(stocks)

    np.testing.assert_array_equal(result.value, np.zeros((1, 2)))
    np.testing.assert_array_equal(result.defined, np.ones((1, 2), dtype=bool))


def test_biomass_stock_identity_roundoff_gate_accepts_exact_identity():
    stocks = _stocks([[400.0]], [[600.0]])
    stocks["total_live_biomass_carbon"] = _trajectory([[1000.0]])
    audit = biomass_stock_identity_roundoff_audit(stocks, roundoff_units=32)

    assert float(audit.tolerance_ratio[0, 0]) == 0.0
    assert float(audit.tolerance[0, 0]) > 0.0


def test_biomass_stock_identity_roundoff_gate_accepts_reduction_order_noise():
    stocks = _stocks([[400.0]], [[600.0]])
    stocks["total_live_biomass_carbon"] = _trajectory([[1000.0 + 3.0e-12]])
    audit = biomass_stock_identity_roundoff_audit(stocks, roundoff_units=32)

    assert float(audit.tolerance_ratio[0, 0]) <= 1.0


def test_biomass_stock_identity_roundoff_gate_rejects_material_error():
    stocks = _stocks([[400.0]], [[600.0]])
    stocks["total_live_biomass_carbon"] = _trajectory([[1000.0 + 1.0e-6]])
    audit = biomass_stock_identity_roundoff_audit(stocks, roundoff_units=32)

    assert float(audit.tolerance_ratio[0, 0]) > 1.0


def test_named_biomass_tendency_rejects_scale_and_horizon_drift():
    stocks = _stocks(jnp.zeros((7, 1)), jnp.zeros((7, 1)))
    with pytest.raises(ValueError, match="horizon registry drift"):
        named_biomass_tendency_objective(
            stocks,
            stocks,
            scales={1: SCALES[1]},
            coefficients=COEFFICIENTS,
            horizons=(1, 7),
        )
    with pytest.raises(ValueError, match="finite and positive"):
        named_biomass_tendency_objective(
            stocks,
            stocks,
            scales={1: dict(SCALES[1]) | {"aboveground_biomass_carbon": 0.0}},
            coefficients=COEFFICIENTS,
            horizons=(1,),
        )
    with pytest.raises(ValueError, match="exceeds"):
        named_biomass_tendency_objective(
            stocks,
            stocks,
            scales=SCALES,
            coefficients=COEFFICIENTS,
        )


def test_named_biomass_tendency_manifest_is_hash_bound_and_bounded():
    path = (
        REPOSITORY_ROOT
        / "manifests"
        / "coarse_graining"
        / "successor_named_biomass_tendency_v1.json"
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected_hash = manifest.pop("contract_sha256")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()

    assert hashlib.sha256(canonical).hexdigest() == expected_hash
    assert manifest["optimized_stock_fields"] == list(NAMED_BIOMASS_TENDENCY_FIELDS)
    assert manifest["scale_policy"]["stock_relative_floor_ratio"] == 1.0e-3
    assert manifest["preparation_gates"]["agb_plus_bgb_identity"] == {
        "floating_dtype": "float64",
        "summed_source_parts": 12,
        "roundoff_units": 32,
        "magnitude_floor": 1.0,
        "criterion": (
            "abs(total_live - aboveground - belowground) <= roundoff_units * "
            "float64_epsilon * max(abs(total_live), abs(aboveground) + "
            "abs(belowground), magnitude_floor)"
        ),
    }
    assert not manifest["negative_design"]["reopens_process_increment_v2"]
    assert manifest["preparation_gates"]["optimizer_updates"] == 0
    assert manifest["training_gate"]["status"].startswith("thresholds_require")


def test_named_biomass_tendency_scale_selection_is_deterministic_and_unique():
    entries = [
        {
            "update": index,
            "landpoint_id": f"point-{index % 4}",
            "year": 1961 + (index // 4),
            "sha256": f"{index:064x}",
        }
        for index in range(1, 21)
    ]
    first = _select_records(entries, point_year_count=8, seed=17)
    second = _select_records(tuple(reversed(entries)), point_year_count=8, seed=17)

    assert [record["update"] for record in first] == [
        record["update"] for record in second
    ]
    assert len({(record["landpoint_id"], record["year"]) for record in first}) == 8


def test_scale_asset_canonical_hash_survives_json_key_normalization():
    payload = {
        "scale_audit": {
            "records": {
                1: {"resolved_scale": 1.0},
                7: {"resolved_scale": 7.0},
                30: {"resolved_scale": 30.0},
            }
        }
    }
    digest = _canonical_sha256(payload)
    reloaded = json.loads(json.dumps(payload | {"asset_sha256": digest}))

    assert _canonical_sha256(reloaded) == digest


def test_named_biomass_tendency_scale_acceptance_manifest_is_hash_bound():
    path = (
        REPOSITORY_ROOT
        / "manifests"
        / "coarse_graining"
        / "successor_named_biomass_tendency_scale_acceptance_v1.json"
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected_hash = manifest.pop("contract_sha256")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()

    assert hashlib.sha256(canonical).hexdigest() == expected_hash
    assert manifest["status"] == "accepted"
    assert manifest["selection"]["total_rollout_windows"] == 256
    assert manifest["accepted_gates"]["training_updates"] == 0
    assert not manifest["accepted_gates"]["sealed_test_used"]


def test_named_biomass_gradient_preflight_manifest_freezes_train_only_rule():
    path = (
        REPOSITORY_ROOT
        / "manifests"
        / "coarse_graining"
        / "successor_named_biomass_gradient_preflight_v1.json"
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected_hash = manifest.pop("contract_sha256")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()

    assert hashlib.sha256(canonical).hexdigest() == expected_hash
    assert manifest["model"]["optimizer_updates"] == 0
    assert manifest["sampling"]["split"] == "train/train"
    assert manifest["sampling"]["validation_values_forbidden"]
    assert manifest["coefficient_rule"]["same_coefficient_for_agb_and_bgb"]
    assert manifest["coefficient_rule"][
        "each_stock_to_npp_median_gradient_ratio_interval"
    ] == [0.1, 10.0]
    assert (
        manifest["objective"]["maintenance_respiration"]
        == "reported_guard_excluded_from_summed_gradient"
    )


def test_named_biomass_candidate_coefficients_are_hash_bound_and_closed():
    path = (
        REPOSITORY_ROOT
        / "manifests"
        / "coarse_graining"
        / "successor_named_biomass_candidate_coefficients_v1.json"
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected_hash = manifest.pop("contract_sha256")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()

    assert hashlib.sha256(canonical).hexdigest() == expected_hash
    assert manifest["resolved_coefficients"]["tendency_mse"] == manifest[
        "resolved_coefficients"
    ]["tendency_bias_squared"]
    assert manifest["gradient_gate_result"]["passed"]
    assert manifest["gradient_gate_result"]["hard_failure_count"] == 0
    assert not manifest["training_policy"]["training_authorized"]
    assert "long_horizon_maintenance_respiration" in manifest[
        "excluded_from_summed_gradient"
    ]


def test_named_biomass_objective_ab_contract_freezes_exact_bounded_gates():
    path = (
        REPOSITORY_ROOT
        / "manifests"
        / "coarse_graining"
        / "successor_named_biomass_objective_ab_v1.json"
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected_hash = manifest.pop("contract_sha256")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()

    assert hashlib.sha256(canonical).hexdigest() == expected_hash
    assert manifest["model"]["architecture"] == "source_routed_process_attention_v2"
    assert manifest["model"]["architecture_change"] is False
    assert manifest["training"] == {
        "updates_per_objective": 100,
        "optimizer_updates": [1001, 1100],
        "sampling": "four_equal_point_year_contexts_per_update",
        "one_day_anchor_batch_size": 32,
        "rollout_windows_per_update": 4,
        "rollout_days": 30,
        "horizons_days": [1, 7, 30],
        "same_point_years_learning_rates_and_optimizer": True,
        "checkpoint_written": False,
        "recovery_written": False,
        "free_gln01_only": True,
        "paid_training_authorized": False,
    }
    assert manifest["gates"] == {
        "hard_failure_count": 0,
        "minimum_research_primary_composite_relative_improvement": 0.005,
        "each_research_primary_key_aggregate_no_worse": True,
        "minimum_improved_slice_horizon_cells": 4,
        "total_slice_horizon_cells": 6,
        "maximum_maintenance_aggregate_relative_regression": 0.05,
        "maximum_maintenance_single_cell_relative_regression": 0.10,
        "gradient_clipping_report_only": True,
    }
    assert manifest["fail_action"] == "stop_named_biomass_objective_without_more_updates"


def _objective_ab_validation(primary_candidate=0.99, maintenance_candidate=1.04):
    keys = (
        "cumulative.npp",
        "terminal.aboveground_biomass_carbon",
        "terminal.belowground_biomass_carbon",
        "terminal.total_live_biomass_carbon",
        "cumulative.maintenance_respiration",
    )

    def arm(primary, maintenance):
        return {
            "slices": {
                slice_id: {
                    "horizons": {
                        str(horizon): {
                            "keys": {
                                key: {
                                    "normalized_rmse": maintenance
                                    if key == "cumulative.maintenance_respiration"
                                    else primary
                                }
                                for key in keys
                            }
                        }
                        for horizon in (7, 30)
                    },
                    "hard_counts": {"budget_residual_failure": 0},
                }
                for slice_id in ("temporal", "spatial", "joint")
            }
        }

    return {
        OBJECTIVES[0]: arm(1.0, 1.0),
        OBJECTIVES[1]: arm(primary_candidate, maintenance_candidate),
    }


def test_named_biomass_objective_ab_classifier_requires_shared_improvement():
    gates = {
        "minimum_research_primary_composite_relative_improvement": 0.005,
        "minimum_improved_slice_horizon_cells": 4,
        "maximum_maintenance_aggregate_relative_regression": 0.05,
        "maximum_maintenance_single_cell_relative_regression": 0.10,
    }
    validation = _objective_ab_validation()
    admitted = classify_objective_ab(validation, gates=gates)

    assert admitted["status"] == "named_biomass_objective_admitted"
    assert admitted["research_primary"]["relative_improvement"] == pytest.approx(0.01)
    assert admitted["research_primary"]["improved_cell_count"] == 6

    validation[OBJECTIVES[1]]["slices"]["joint"]["horizons"]["30"]["keys"][
        "cumulative.npp"
    ]["normalized_rmse"] = 1.1
    rejected = classify_objective_ab(validation, gates=gates)

    assert rejected["status"] == "named_biomass_objective_rejected"
    assert rejected["checks"]["each_research_primary_key_no_worse"] is False


def test_named_biomass_objective_ab_launcher_is_free_and_no_checkpoint():
    launcher = (
        REPOSITORY_ROOT
        / "scripts"
        / "hpc"
        / "run_successor_named_biomass_objective_ab.sh"
    ).read_text()
    source = (
        REPOSITORY_ROOT
        / "scripts"
        / "hpc"
        / "run_successor_named_biomass_objective_ab.py"
    ).read_text()

    assert "must run on free gln01" in launcher
    assert "SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false" in launcher
    assert "#SBATCH" not in launcher
    assert '"checkpoint_written": False' in source
    assert '"recovery_written": False' in source
    assert "write_staged_recovery" not in source
    assert "write_staged_best_checkpoint" not in source


def test_gradient_feasibility_directions_find_orthogonal_common_descent():
    diagnostics = gradient_direction_diagnostics(np.eye(4, dtype=np.float64))

    for direction_name in ("equal_unit", "minimum_norm_unit"):
        direction = diagnostics["directions"][direction_name]
        assert direction["all_strictly_positive"]
        assert set(direction["normalized_directional_derivatives"]) == set(
            GRADIENT_FEASIBILITY_LABELS
        )
        assert tuple(direction["normalized_directional_derivatives"].values()) == (
            pytest.approx(0.5),
        ) * 4

    records = [
        {"group": group, "diagnostics": diagnostics}
        for group in range(4)
    ]
    result = classify_gradient_feasibility(
        records,
        gates={
            "minimum_group_component_cosine": 0.05,
            "minimum_component_median_cosine": 0.10,
            "minimum_passing_groups": 3,
        },
    )
    assert result["status"] == "common_descent_direction_supported"
    assert result["selected_direction"] == "equal_unit"


def test_gradient_feasibility_rejects_opposed_objectives():
    vectors = np.asarray(
        (
            (1.0, 0.0),
            (-1.0, 0.0),
            (0.0, 1.0),
            (0.0, 1.0),
        ),
        dtype=np.float64,
    )
    diagnostics = gradient_direction_diagnostics(vectors @ vectors.T)
    records = [
        {"group": group, "diagnostics": diagnostics}
        for group in range(4)
    ]
    result = classify_gradient_feasibility(
        records,
        gates={
            "minimum_group_component_cosine": 0.05,
            "minimum_component_median_cosine": 0.10,
            "minimum_passing_groups": 3,
        },
    )

    assert result["status"] == "common_descent_direction_not_supported"
    assert result["selected_direction"] is None


def test_minimum_norm_solver_falls_back_for_slow_projected_convergence():
    gram = np.asarray(
        (
            (1.0, 0.9, 0.2, 0.1),
            (0.9, 1.0, 0.3, 0.2),
            (0.2, 0.3, 1.0, 0.8),
            (0.1, 0.2, 0.8, 1.0),
        ),
        dtype=np.float64,
    )

    weights = _minimum_norm_simplex_weights(gram, maximum_iterations=1)

    assert np.all(weights >= 0.0)
    assert float(np.sum(weights)) == pytest.approx(1.0)
    gradient = gram @ weights
    level = float(weights @ gradient)
    support = weights > 1.0e-9
    assert np.max(np.abs(gradient[support] - level)) < 1.0e-8
    assert np.all(gradient[~support] >= level - 1.0e-8)


def test_minimum_norm_solver_scales_to_forty_constraints():
    normalized = np.eye(40, dtype=np.float64)

    weights = _minimum_norm_simplex_weights(normalized)

    np.testing.assert_allclose(weights, np.full(40, 1.0 / 40.0))
    assert float(np.sum(weights)) == pytest.approx(1.0)


@pytest.mark.parametrize("case", ("correlated", "low_rank", "opposed"))
def test_large_minimum_norm_solver_satisfies_kkt_for_hard_grams(case):
    rng = np.random.default_rng(20260815)
    if case == "correlated":
        vectors = rng.normal(size=(40, 128))
        vectors[:, :8] += 2.0
    elif case == "low_rank":
        basis = rng.normal(size=(8, 128))
        vectors = np.repeat(basis, 5, axis=0) + 1.0e-8 * rng.normal(size=(40, 128))
    else:
        half = rng.normal(size=(20, 128))
        vectors = np.concatenate((half, -half + 1.0e-8), axis=0)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    gram = vectors @ vectors.T

    weights = _minimum_norm_simplex_weights(gram)

    gradient = gram @ weights
    level = float(weights @ gradient)
    support = weights > 1.0e-9
    assert np.all(weights >= 0.0)
    assert float(np.sum(weights)) == pytest.approx(1.0)
    assert np.max(np.abs(gradient[support] - level)) <= 1.0e-8
    assert np.all(gradient[~support] >= level - 1.0e-8)


def test_gradient_feasibility_contract_is_hash_bound_and_zero_update():
    path = (
        REPOSITORY_ROOT
        / "manifests"
        / "coarse_graining"
        / "successor_gradient_feasibility_v1.json"
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected_hash = manifest.pop("contract_sha256")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()

    assert hashlib.sha256(canonical).hexdigest() == expected_hash
    assert manifest["components"] == list(GRADIENT_FEASIBILITY_LABELS)
    assert manifest["gates"]["minimum_group_component_cosine"] == 0.05
    assert manifest["gates"]["minimum_passing_groups"] == 3
    assert manifest["gates"]["minimum_component_median_cosine"] == 0.10
    assert manifest["execution"]["optimizer_updates"] == 0
    assert not manifest["execution"]["paid_training_authorized"]
    assert not manifest["execution"]["validation_values_evaluated"]


def test_gradient_feasibility_launcher_is_free_and_zero_update():
    launcher = (
        REPOSITORY_ROOT
        / "scripts"
        / "hpc"
        / "run_successor_gradient_feasibility.sh"
    ).read_text()
    source = (
        REPOSITORY_ROOT
        / "scripts"
        / "hpc"
        / "probe_successor_gradient_feasibility.py"
    ).read_text()

    assert "requires free gln01" in launcher
    assert "SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false" in launcher
    assert "#SBATCH" not in launcher
    assert '"training_updates": 0' in source
    assert '"checkpoint_written": False' in source
    assert '"validation_values_evaluated": False' in source
    assert "successor_adamw_update" not in source
    assert "write_staged_recovery" not in source


def test_gradient_balanced_contracts_are_hash_bound_and_bounded():
    manifest_root = REPOSITORY_ROOT / "manifests" / "coarse_graining"
    acceptance = json.loads(
        (
            manifest_root / "successor_gradient_feasibility_acceptance_v1.json"
        ).read_text(encoding="utf-8")
    )
    smoke = json.loads(
        (manifest_root / "successor_gradient_balanced_smoke_v1.json").read_text(
            encoding="utf-8"
        )
    )
    for payload in (acceptance, smoke):
        expected_hash = payload.pop("contract_sha256")
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        assert hashlib.sha256(canonical).hexdigest() == expected_hash

    assert smoke["training"]["updates_per_objective"] == 10
    assert smoke["training"]["optimizer_updates"] == [1001, 1010]
    assert smoke["training"]["one_day_anchor_batch_size"] == 32
    assert smoke["training"]["rollout_windows_per_update"] == 4
    assert smoke["training"]["checkpoint_written"] is False
    assert smoke["training"]["recovery_written"] is False
    assert smoke["training"]["free_gln01_only"] is True
    assert smoke["gates"]["minimum_direction_feasible_updates"] == 8
    assert smoke["gates"]["maximum_candidate_gradient_clip_updates"] == 2


def test_gradient_balanced_validation_subset_is_first_and_last_per_slice():
    slices = ("temporal", "spatial", "joint")
    panel = {
        "slices": {
            slice_id: {
                "entries": [
                    {"window": f"{slice_id}-{ordinal}"}
                    for ordinal in range(3)
                ]
            }
            for slice_id in slices
        }
    }
    prepared = {"entries": [{"artifact": ordinal} for ordinal in range(9)]}

    selected = selected_validation_subset(prepared, panel)

    assert [record["artifact"] for record, _entry in selected] == [0, 2, 3, 5, 6, 8]
    assert [slice_id for _record, (slice_id, _entry) in selected] == [
        "temporal",
        "temporal",
        "spatial",
        "spatial",
        "joint",
        "joint",
    ]


def test_gradient_balanced_classifier_requires_training_and_validation_gates():
    objective_validation = _objective_ab_validation()
    validation = {
        CONTROL: objective_validation[OBJECTIVES[0]],
        CANDIDATE: objective_validation[OBJECTIVES[1]],
    }
    gates = {
        "minimum_candidate_direction_component_cosine": 0.05,
        "minimum_direction_feasible_updates": 8,
        "maximum_candidate_gradient_clip_updates": 2,
        "minimum_research_primary_composite_relative_improvement": 0.005,
        "minimum_improved_slice_horizon_cells": 4,
        "maximum_maintenance_aggregate_relative_regression": 0.05,
        "maximum_maintenance_single_cell_relative_regression": 0.10,
    }
    updates = [
        {
            "candidate": {
                "minimum_component_cosine": 0.10,
                "combined_gradient_norm": 0.90,
            }
        }
        for _ in range(10)
    ]

    admitted = classify_gradient_balanced_smoke(
        validation,
        updates,
        gates=gates,
        gradient_clip=1.0,
    )
    assert admitted["status"] == "gradient_balanced_smoke_admitted"
    assert admitted["direction_feasible_updates"] == 10

    for update in updates[:3]:
        update["candidate"]["combined_gradient_norm"] = 1.01
    rejected = classify_gradient_balanced_smoke(
        validation,
        updates,
        gates=gates,
        gradient_clip=1.0,
    )
    assert rejected["status"] == "gradient_balanced_smoke_rejected"
    assert not rejected["training_checks"][
        "candidate_gradient_clip_within_limit"
    ]


def test_gradient_gram_compiles_for_parameter_pytrees():
    measure = _make_gradient_gram(jax)
    gradients = tuple(
        {"first": jnp.asarray([value, 1.0]), "second": jnp.asarray([2.0])}
        for value in (1.0, 2.0, 3.0, 4.0)
    )

    gram, nonfinite = measure(gradients)

    np.testing.assert_allclose(
        np.asarray(gram),
        np.asarray(
            [
                [6.0, 7.0, 8.0, 9.0],
                [7.0, 9.0, 11.0, 13.0],
                [8.0, 11.0, 14.0, 17.0],
                [9.0, 13.0, 17.0, 21.0],
            ]
        ),
    )
    assert np.asarray(nonfinite).tolist() == [0, 0, 0, 0]


def test_gradient_balanced_apply_step_compiles_with_existing_adamw():
    from research.daily_coarse_graining.daily_operator.successor_small_fit import (
        initialize_successor_adamw,
    )

    parameters = {"value": jnp.asarray([1.0, 2.0], dtype=jnp.float64)}
    optimizer = initialize_successor_adamw(parameters)
    contract = SimpleNamespace(
        optimizer={
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1.0e-8,
            "weight_decay": 0.0,
            "global_gradient_norm_clip": 1.0,
        }
    )
    apply_step = _make_candidate_apply_step(jax, contract)
    gradients = tuple(
        {"value": jnp.asarray([value, 0.0], dtype=jnp.float64)}
        for value in (1.0, 2.0, 3.0, 4.0)
    )

    next_parameters, next_optimizer, norm = apply_step(
        parameters,
        optimizer,
        gradients,
        jnp.full((4,), 0.25, dtype=jnp.float64),
        jnp.asarray(1.0e-3, dtype=jnp.float64),
    )

    assert float(norm) == pytest.approx(2.5)
    assert int(next_optimizer.step) == 1
    assert float(next_parameters["value"][0]) < 1.0
    assert float(next_parameters["value"][1]) == pytest.approx(2.0)


def test_gradient_balanced_launcher_is_free_and_no_checkpoint():
    launcher = (
        REPOSITORY_ROOT
        / "scripts"
        / "hpc"
        / "run_successor_gradient_balanced_smoke.sh"
    ).read_text()
    source = (
        REPOSITORY_ROOT
        / "scripts"
        / "hpc"
        / "run_successor_gradient_balanced_smoke.py"
    ).read_text()

    assert "must run on free gln01" in launcher
    assert "SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false" in launcher
    assert "#SBATCH" not in launcher
    assert '"checkpoint_written": False' in source
    assert '"recovery_written": False' in source
    assert "write_staged_recovery" not in source
    assert "write_staged_best_checkpoint" not in source


def _realized_step_records(inherited_cosine, reset_cosine, inherited_loss, reset_loss):
    return [
        {
            "group": group,
            "raw_direction": {"minimum_component_cosine": 0.2},
            "optimizers": {
                "inherited_adamw": {
                    "component_cosines": {
                        label: inherited_cosine for label in GRADIENT_FEASIBILITY_LABELS
                    },
                    "virtual_loss_relative_changes": {
                        label: inherited_loss for label in GRADIENT_FEASIBILITY_LABELS
                    },
                },
                "reset_adamw": {
                    "component_cosines": {
                        label: reset_cosine for label in GRADIENT_FEASIBILITY_LABELS
                    },
                    "virtual_loss_relative_changes": {
                        label: reset_loss for label in GRADIENT_FEASIBILITY_LABELS
                    },
                },
            },
        }
        for group in range(4)
    ]


def _realized_step_gates():
    return {
        "minimum_raw_component_cosine": 0.05,
        "minimum_realized_component_cosine": 0.01,
        "maximum_virtual_component_relative_increase": 1.0e-6,
        "minimum_passing_groups": 3,
        "total_groups": 4,
    }


@pytest.mark.parametrize(
    ("values", "status"),
    (
        (
            (0.2, 0.2, -0.01, -0.01),
            "inherited_adamw_handoff_supported",
        ),
        (
            (-0.2, 0.2, 0.01, -0.01),
            "inherited_adamw_moments_destroy_common_descent",
        ),
        (
            (-0.2, -0.2, 0.01, 0.01),
            "adamw_realization_not_supported",
        ),
    ),
)
def test_realized_step_classifier_has_decisive_optimizer_forks(values, status):
    result = classify_realized_step_audit(
        _realized_step_records(*values),
        gates=_realized_step_gates(),
    )

    assert result["status"] == status


def test_realized_step_classifier_stops_on_raw_direction_drift():
    records = _realized_step_records(0.2, 0.2, -0.01, -0.01)
    for record in records[:2]:
        record["raw_direction"]["minimum_component_cosine"] = -0.1

    result = classify_realized_step_audit(records, gates=_realized_step_gates())

    assert result["status"] == "raw_common_descent_drifted"


def test_realized_step_contract_is_hash_bound_and_zero_update():
    path = (
        REPOSITORY_ROOT
        / "manifests"
        / "coarse_graining"
        / "successor_realized_step_audit_v1.json"
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected_hash = manifest.pop("contract_sha256")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()

    assert hashlib.sha256(canonical).hexdigest() == expected_hash
    assert manifest["components"] == list(GRADIENT_FEASIBILITY_LABELS)
    assert manifest["gates"]["minimum_realized_component_cosine"] == 0.01
    assert manifest["gates"]["minimum_passing_groups"] == 3
    assert manifest["execution"]["optimizer_updates"] == 0
    assert manifest["virtual_step"]["proposed_parameters_retained"] is False
    assert manifest["virtual_step"]["optimizer_state_retained"] is False


def test_realized_step_launcher_is_free_and_zero_update():
    launcher = (
        REPOSITORY_ROOT / "scripts" / "hpc" / "run_successor_realized_step.sh"
    ).read_text()
    source = (
        REPOSITORY_ROOT / "scripts" / "hpc" / "probe_successor_realized_step.py"
    ).read_text()

    assert "requires free gln01" in launcher
    assert "SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false" in launcher
    assert "#SBATCH" not in launcher
    assert '"optimizer_updates": 0' in source
    assert '"proposed_parameters_retained": False' in source
    assert '"optimizer_state_retained": False' in source
    assert '"checkpoint_written": False' in source
    assert "write_staged_recovery" not in source
    assert "write_staged_best_checkpoint" not in source


def test_realized_step_proposal_compiles_and_distinguishes_optimizer_state():
    from research.daily_coarse_graining.daily_operator.successor_small_fit import (
        SuccessorAdamWState,
        initialize_successor_adamw,
    )

    parameters = {"value": jnp.asarray([1.0, 2.0], dtype=jnp.float64)}
    reset = initialize_successor_adamw(parameters)
    inherited = SuccessorAdamWState(
        jnp.asarray(1000, dtype=jnp.int32),
        {"value": jnp.asarray([0.5, 0.0], dtype=jnp.float64)},
        {"value": jnp.asarray([0.25, 0.0], dtype=jnp.float64)},
    )
    contract = SimpleNamespace(
        optimizer={
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1.0e-8,
            "weight_decay": 0.0,
            "global_gradient_norm_clip": 1.0,
        }
    )
    gradients = tuple(
        {"value": jnp.asarray([value, 0.0], dtype=jnp.float64)}
        for value in (1.0, 2.0, 3.0, 4.0)
    )
    propose = _make_proposed_directions(jax, contract)

    result = propose(
        parameters,
        inherited,
        reset,
        gradients,
        jnp.full((4,), 0.25, dtype=jnp.float64),
        jnp.asarray(1.0e-3, dtype=jnp.float64),
    )
    jax.block_until_ready(result)
    _, _, combined, inherited_direction, reset_direction, raw_norm = result

    np.testing.assert_allclose(combined["value"], np.asarray([2.5, 0.0]))
    assert float(raw_norm) == pytest.approx(2.5)
    assert np.all(np.isfinite(np.asarray(inherited_direction["value"])))
    assert np.all(np.isfinite(np.asarray(reset_direction["value"])))
    assert not np.allclose(
        np.asarray(inherited_direction["value"]),
        np.asarray(reset_direction["value"]),
    )


def _trust_region_records(changes_by_step):
    return [
        {
            "group": group,
            "steps": {
                format(step, ".12g"): {
                    "virtual_loss_relative_changes": {
                        label: values[group]
                        for label in GRADIENT_FEASIBILITY_LABELS
                    },
                    "hard_gate_counts": {"failure": 0},
                }
                for step, values in changes_by_step.items()
            },
        }
        for group in range(4)
    ]


def _trust_region_gates():
    return {
        "maximum_passing_group_component_relative_increase": 1.0e-6,
        "minimum_passing_groups": 3,
        "total_groups": 4,
        "maximum_any_group_component_relative_increase": 0.01,
        "minimum_each_component_median_relative_decrease": 1.0e-4,
    }


def test_direct_trust_region_selects_largest_shared_passing_norm():
    result = classify_direct_trust_region(
        _trust_region_records(
            {
                0.001: (-0.001, -0.001, -0.001, 0.005),
                0.003: (-0.003, -0.003, -0.003, 0.009),
                0.01: (-0.01, -0.01, 0.02, 0.02),
            }
        ),
        step_norms=(0.001, 0.003, 0.01),
        gates=_trust_region_gates(),
    )

    assert result["status"] == "direct_trust_region_supported"
    assert result["selected_step_norm"] == pytest.approx(0.003)


def test_direct_trust_region_rejects_only_tiny_or_unbounded_steps():
    result = classify_direct_trust_region(
        _trust_region_records(
            {
                0.0001: (-1.0e-6, -1.0e-6, -1.0e-6, -1.0e-6),
                0.1: (-0.1, -0.1, -0.1, 0.02),
            }
        ),
        step_norms=(0.0001, 0.1),
        gates=_trust_region_gates(),
    )

    assert result["status"] == "direct_trust_region_not_supported"
    assert result["selected_step_norm"] is None


def test_direct_trust_region_contract_is_hash_bound_and_zero_update():
    path = (
        REPOSITORY_ROOT
        / "manifests"
        / "coarse_graining"
        / "successor_direct_trust_region_audit_v1.json"
    )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected_hash = manifest.pop("contract_sha256")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()

    assert hashlib.sha256(canonical).hexdigest() == expected_hash
    assert manifest["step_norm_grid"] == [
        0.0001,
        0.0003,
        0.001,
        0.003,
        0.01,
        0.03,
        0.1,
    ]
    assert manifest["direction"]["momentum"] is False
    assert manifest["direction"]["coordinate_adaptive_scaling"] is False
    assert manifest["direction"]["weight_decay"] is False
    assert manifest["execution"]["optimizer_updates"] == 0


def test_direct_trust_region_step_has_frozen_global_norm():
    parameters = {"value": jnp.asarray([1.0, 2.0], dtype=jnp.float64)}
    gradients = tuple(
        {"value": jnp.asarray([value, 0.0], dtype=jnp.float64)}
        for value in (1.0, 2.0, 3.0, 4.0)
    )
    combine = _make_combined_direction(jax)
    virtual_step = _make_direct_virtual_step(jax)
    direction, norm, nonfinite = combine(
        gradients,
        jnp.full((4,), 0.25, dtype=jnp.float64),
    )
    proposed, proposed_nonfinite = virtual_step(
        parameters,
        direction,
        norm,
        jnp.asarray(0.03, dtype=jnp.float64),
    )

    delta = np.asarray(parameters["value"] - proposed["value"])
    assert np.linalg.norm(delta) == pytest.approx(0.03)
    assert int(nonfinite) == 0
    assert int(proposed_nonfinite) == 0


def test_direct_trust_region_launcher_is_free_and_zero_update():
    launcher = (
        REPOSITORY_ROOT
        / "scripts"
        / "hpc"
        / "run_successor_direct_trust_region.sh"
    ).read_text()
    source = (
        REPOSITORY_ROOT
        / "scripts"
        / "hpc"
        / "probe_successor_direct_trust_region.py"
    ).read_text()

    assert "requires free gln01" in launcher
    assert "SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false" in launcher
    assert "#SBATCH" not in launcher
    assert '"optimizer_updates": 0' in source
    assert '"proposed_parameters_retained": False' in source
    assert '"checkpoint_written": False' in source
    assert "successor_adamw_update" not in source
    assert "write_staged_recovery" not in source
    assert "_make_loss_evaluator" in source
    assert "_loss_record" not in source


def _direct_smoke_gates():
    return {
        "hard_failure_count": 0,
        "nonfinite_value_count": 0,
        "total_updates": 10,
        "minimum_all_component_nonincrease_updates": 8,
        "maximum_nonincrease_component_relative_increase": 1.0e-6,
        "maximum_any_update_component_relative_increase": 0.01,
        "minimum_each_component_median_relative_decrease": 1.0e-4,
        "minimum_research_primary_composite_relative_improvement": 0.005,
        "minimum_improved_slice_horizon_cells": 4,
        "maximum_maintenance_aggregate_relative_regression": 0.05,
        "maximum_maintenance_single_cell_relative_regression": 0.10,
    }


def _direct_smoke_updates(relative_change=-0.001):
    return [
        {
            "update_applied": True,
            "loss_relative_changes": {
                label: relative_change for label in GRADIENT_FEASIBILITY_LABELS
            },
            "parent_hard_gate_counts": {"failure": 0},
            "proposed_hard_gate_counts": {"failure": 0},
            "nonfinite_values": {"values": 0},
        }
        for _ in range(10)
    ]


def test_direct_direction_smoke_requires_bounded_updates_and_rollout_gain():
    objective_validation = _objective_ab_validation()
    validation = {
        CONTROL: objective_validation[OBJECTIVES[0]],
        CANDIDATE: objective_validation[OBJECTIVES[1]],
    }
    updates = _direct_smoke_updates()

    admitted = classify_direct_direction_smoke(
        validation,
        updates,
        gates=_direct_smoke_gates(),
    )
    assert admitted["status"] == "direct_direction_smoke_admitted"
    assert admitted["all_component_nonincrease_updates"] == 10

    updates[0]["loss_relative_changes"][GRADIENT_FEASIBILITY_LABELS[0]] = 0.02
    rejected = classify_direct_direction_smoke(
        validation,
        updates,
        gates=_direct_smoke_gates(),
    )
    assert rejected["status"] == "direct_direction_smoke_rejected"
    assert not rejected["training_checks"][
        "every_update_within_global_regression_bound"
    ]


def test_direct_direction_smoke_contract_and_launcher_are_bounded():
    manifest_path = (
        REPOSITORY_ROOT
        / "manifests"
        / "coarse_graining"
        / "successor_direct_direction_smoke_v1.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest.pop("contract_sha256")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    launcher = (
        REPOSITORY_ROOT
        / "scripts"
        / "hpc"
        / "run_successor_direct_direction_smoke.sh"
    ).read_text()
    source = (
        REPOSITORY_ROOT
        / "scripts"
        / "hpc"
        / "run_successor_direct_direction_smoke.py"
    ).read_text()

    assert hashlib.sha256(canonical).hexdigest() == expected_hash
    assert manifest["update"]["global_step_norm"] == pytest.approx(0.001)
    assert manifest["training"]["updates"] == 10
    assert manifest["update"]["adamw"] is False
    assert manifest["execution"]["checkpoint_written"] is False
    assert "requires free gln01" in launcher
    assert "SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false" in launcher
    assert "#SBATCH" not in launcher
    assert "successor_adamw_update" not in source
    assert '"checkpoint_written": False' in source
    assert '"final_parameters_retained": False' in source
    assert "write_staged_recovery" not in source


def test_multibatch_gradient_accumulation_matches_equal_group_mean():
    parameters = {"value": jnp.asarray([1.0, 2.0], dtype=jnp.float64)}
    accumulators = _empty_accumulators(jax, parameters)
    accumulate = _make_gradient_accumulator(jax)
    count_nonfinite = _make_nonfinite_counter(jax)
    gradient = {"value": jnp.asarray([3.0, 4.0], dtype=jnp.float64)}

    for _group in range(10):
        accumulators[0] = accumulate(
            accumulators[0],
            gradient,
            jnp.asarray(0.1, dtype=jnp.float64),
        )

    np.testing.assert_allclose(accumulators[0]["value"], gradient["value"])
    assert int(count_nonfinite(accumulators[0])) == 0


def test_multibatch_direction_contract_and_launcher_are_zero_update():
    manifest_path = (
        REPOSITORY_ROOT
        / "manifests"
        / "coarse_graining"
        / "successor_multibatch_direction_audit_v1.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest.pop("contract_sha256")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    launcher = (
        REPOSITORY_ROOT
        / "scripts"
        / "hpc"
        / "run_successor_multibatch_direction.sh"
    ).read_text()
    source = (
        REPOSITORY_ROOT
        / "scripts"
        / "hpc"
        / "probe_successor_multibatch_direction.py"
    ).read_text()

    assert hashlib.sha256(canonical).hexdigest() == expected_hash
    assert manifest["sampling"]["total_point_years"] == 40
    assert manifest["sampling"]["total_one_day_anchors"] == 320
    assert manifest["sampling"]["total_rollout_windows"] == 40
    assert manifest["execution"]["optimizer_updates"] == 0
    assert manifest["direction"]["gradient_accumulation"] is True
    assert "requires free gln01" in launcher
    assert "SINGULARITYENV_XLA_PYTHON_CLIENT_PREALLOCATE=false" in launcher
    assert "#SBATCH" not in launcher
    assert "successor_adamw_update" not in source
    assert '"checkpoint_written": False' in source
    assert '"proposed_parameters_retained": False' in source
    assert "write_staged_recovery" not in source


def test_group_aware_flat_gradient_geometry_compiles():
    flatten = _make_flattened_unit_gradient(jax)
    gradient = {"value": jnp.asarray([3.0, 4.0], dtype=jnp.float64)}
    unit, norm, nonfinite = flatten(gradient)
    matrix = jnp.stack((unit, -unit))
    gram = _make_flat_gram(jax)(matrix)
    direction, direction_norm, direction_nonfinite = _make_flat_direction(jax)(
        matrix,
        jnp.asarray([0.75, 0.25], dtype=jnp.float64),
    )

    assert float(norm) == pytest.approx(5.0)
    assert int(nonfinite) == 0
    np.testing.assert_allclose(gram, np.asarray([[1.0, -1.0], [-1.0, 1.0]]))
    np.testing.assert_allclose(direction, 0.5 * np.asarray(unit))
    assert float(direction_norm) == pytest.approx(0.5)
    assert int(direction_nonfinite) == 0


def test_group_aware_classifier_requires_direction_before_virtual_steps():
    diagnostics = gradient_direction_diagnostics(
        np.eye(40, dtype=np.float64),
        labels=tuple(f"constraint_{index}" for index in range(40)),
    )
    records = [
        {
            "group": group,
            "steps": {
                "0.0001": {
                    "virtual_loss_relative_changes": {
                        label: -0.001 for label in GRADIENT_FEASIBILITY_LABELS
                    },
                    "hard_gate_counts": {"failure": 0},
                }
            },
        }
        for group in range(10)
    ]
    contract = {
        "direction_gate": {"minimum_group_component_cosine": 0.001},
        "gates": {
            "maximum_passing_group_component_relative_increase": 1.0e-6,
            "minimum_passing_groups": 8,
            "total_groups": 10,
            "maximum_any_group_component_relative_increase": 0.01,
            "minimum_each_component_median_relative_decrease": 1.0e-4,
        },
        "pass_action": "pass",
        "direction_fail_action": "direction_fail",
        "step_fail_action": "step_fail",
    }

    result = classify_group_aware_direction(
        diagnostics,
        records,
        step_norms=(0.0001,),
        contract=contract,
    )

    assert result["status"] == "group_aware_finite_step_supported"
    assert result["selected_step_norm"] == pytest.approx(0.0001)
    assert result["next_action"] == "pass"


def test_group_aware_contract_and_launcher_are_zero_update():
    manifest_path = (
        REPOSITORY_ROOT
        / "manifests"
        / "coarse_graining"
        / "successor_group_aware_direction_audit_v1.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest.pop("contract_sha256")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    launcher = (
        REPOSITORY_ROOT
        / "scripts"
        / "hpc"
        / "run_successor_group_aware_direction.sh"
    ).read_text()
    source = (
        REPOSITORY_ROOT
        / "scripts"
        / "hpc"
        / "probe_successor_multibatch_direction.py"
    ).read_text()

    assert hashlib.sha256(canonical).hexdigest() == expected_hash
    assert manifest["sampling"]["group_component_constraints"] == 40
    assert manifest["direction_gate"]["minimum_group_component_cosine"] == 0.001
    assert manifest["execution"]["optimizer_updates"] == 0
    assert manifest["execution"]["geometry_sidecar_written"] is True
    assert "requires free gln01" in launcher
    assert "--group-aware" in launcher
    assert "#SBATCH" not in launcher
    assert "successor_adamw_update" not in source
    assert '"checkpoint_written": False' in source
    assert '"parameters_retained": False' in source


def test_group_aware_step_extension_is_frozen_and_zero_update():
    manifest_path = (
        REPOSITORY_ROOT
        / "manifests"
        / "coarse_graining"
        / "successor_group_aware_step_extension_v1.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest.pop("contract_sha256")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    launcher = (
        REPOSITORY_ROOT
        / "scripts"
        / "hpc"
        / "run_successor_group_aware_step_extension.sh"
    ).read_text()

    assert hashlib.sha256(canonical).hexdigest() == expected_hash
    assert manifest["step_norm_grid"] == [0.0005, 0.001, 0.0015, 0.002]
    assert manifest["sampling"]["group_component_constraints"] == 40
    assert manifest["execution"]["optimizer_updates"] == 0
    assert manifest["execution"]["proposed_parameters_retained"] is False
    assert "requires free gln01" in launcher
    assert "--group-aware-step-extension" in launcher
    assert "successor-group-aware-direction-b0ebda9/report.json" in launcher
    assert "#SBATCH" not in launcher


def test_frozen_group_aware_scalarization_uses_weight_over_raw_norm():
    weights = {
        f"group_{group + 1:02d}.{label}": 0.0
        for group in range(10)
        for label in GRADIENT_FEASIBILITY_LABELS
    }
    positive = []
    for group in range(8):
        count = 2 if group < 6 else 1
        for component in range(count):
            key = f"group_{group + 1:02d}.{GRADIENT_FEASIBILITY_LABELS[component]}"
            weights[key] = 1.0 / 14.0
            positive.append((group, component, key))
    predecessor = {
        "schema_version": "successor_group_aware_step_extension_report_v1",
        "status": "completed",
        "classification": {
            "status": "group_aware_finite_step_supported",
            "passed": True,
            "selected_step_norm": 0.002,
        },
        "aggregate_direction": {
            "normalized_simplex_weights": weights,
            "direction_norm": 0.05,
        },
        "records": [
            {
                "group": group,
                "gradient_norms": {
                    label: float(group + component + 1)
                    for component, label in enumerate(GRADIENT_FEASIBILITY_LABELS)
                },
            }
            for group in range(10)
        ],
    }

    seeds, summary = _frozen_scalarization_seeds(
        predecessor,
        expected_step_norm=0.002,
    )

    assert summary["positive_constraints"] == 14
    assert summary["active_groups"] == 8
    for group, component, key in positive:
        norm = predecessor["records"][group]["gradient_norms"][
            GRADIENT_FEASIBILITY_LABELS[component]
        ]
        assert seeds[group, component] == pytest.approx(weights[key] / norm)


def test_group_aware_scalarization_smoke_contract_is_bounded():
    manifest_path = (
        REPOSITORY_ROOT
        / "manifests"
        / "coarse_graining"
        / "successor_group_aware_scalarization_smoke_v1.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest.pop("contract_sha256")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    launcher = (
        REPOSITORY_ROOT
        / "scripts"
        / "hpc"
        / "run_successor_group_aware_scalarization_smoke.sh"
    ).read_text()

    assert hashlib.sha256(canonical).hexdigest() == expected_hash
    assert manifest["training"]["updates"] == 5
    assert manifest["training"]["reverse_passes_per_update"] == 8
    assert manifest["execution"]["checkpoint_written"] is False
    assert manifest["execution"]["final_parameters_retained"] is False
    assert "requires free gln01" in launcher
    assert "--frozen-group-aware-scalarization-smoke" in launcher
    assert "#SBATCH" not in launcher


def test_group_aware_backtracking_smoke_has_a_useful_minimum_step():
    manifest_path = (
        REPOSITORY_ROOT
        / "manifests"
        / "coarse_graining"
        / "successor_group_aware_backtracking_smoke_v1.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest.pop("contract_sha256")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    launcher = (
        REPOSITORY_ROOT
        / "scripts"
        / "hpc"
        / "run_successor_group_aware_backtracking_smoke.sh"
    ).read_text()

    assert hashlib.sha256(canonical).hexdigest() == expected_hash
    assert manifest["update"]["step_norm_grid"] == [0.002, 0.001, 0.0005]
    assert manifest["update"]["minimum_admissible_step_norm"] == 0.0005
    assert manifest["training"]["reverse_passes_per_update"] == 8
    assert manifest["execution"]["checkpoint_written"] is False
    assert "requires free gln01" in launcher
    assert "--group-aware-backtracking-scalarization-smoke" in launcher
    assert "--failed-scalarization-report" in launcher
    assert "#SBATCH" not in launcher


def test_second_state_recalibration_classifies_sparse_overlap():
    direction = {
        "direction_passed": True,
        "passed": True,
        "selected_step_norm": 0.001,
        "minimum_group_component_cosine": 0.04,
    }
    support = {"recalibrated_support_size": 12, "overlap_size": 6}
    result = classify_second_state_recalibration(
        direction,
        support,
        gates={
            "maximum_recalibrated_support_size": 20,
            "minimum_parent_support_overlap": 1,
        },
        pass_action="implement_active_set",
        direction_fail_action="redesign_objectives",
        step_fail_action="redesign_step",
        support_fail_action="redesign_recalibration",
    )

    assert result["status"] == "second_state_active_set_recalibration_supported"
    assert result["passed"] is True
    assert result["next_action"] == "implement_active_set"


def test_second_state_recalibration_contract_is_zero_retention():
    manifest_path = (
        REPOSITORY_ROOT
        / "manifests"
        / "coarse_graining"
        / "successor_second_state_recalibration_audit_v1.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest.pop("contract_sha256")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    launcher = (
        REPOSITORY_ROOT
        / "scripts"
        / "hpc"
        / "run_successor_second_state_recalibration_audit.sh"
    ).read_text()

    assert hashlib.sha256(canonical).hexdigest() == expected_hash
    assert manifest["first_update"]["reverse_passes"] == 8
    assert manifest["second_state"]["reverse_passes"] == 40
    assert manifest["step_norm_grid"] == [0.0005, 0.001, 0.0015, 0.002]
    assert manifest["gates"]["minimum_passing_groups"] == 10
    assert manifest["execution"]["new_updates_applied"] == 0
    assert manifest["execution"]["validation_values_evaluated"] is False
    assert manifest["execution"]["checkpoint_written"] is False
    assert "requires free gln01" in launcher
    assert "--second-state-recalibration-audit" in launcher
    assert "--backtracking-report" in launcher
    assert "#SBATCH" not in launcher


def test_active_set_recalibration_requires_direction_step_and_efficiency():
    result = classify_active_set_recalibration(
        {
            "nonpositive_constraint_count": 0,
            "minimum_relative_directional_derivative": 0.01,
        },
        {"passed": True, "selected_step_norm": 0.002},
        {
            "component_reverse_passes": 15,
            "directional_check_rounds": 2,
            "added_constraints": 1,
        },
        gates={
            "minimum_relative_directional_derivative": 0.0,
            "maximum_component_reverse_passes": 20,
            "maximum_directional_check_rounds": 3,
            "maximum_added_constraints": 6,
        },
        exact_selected_step_norm=0.002,
        pass_action="bounded_training_smoke",
        fail_action="redesign",
    )

    assert result["status"] == "active_set_recalibration_supported"
    assert result["passed"] is True
    assert result["reference_step_reproduced"] is True


def test_active_set_recalibration_contract_is_bounded():
    manifest_path = (
        REPOSITORY_ROOT
        / "manifests"
        / "coarse_graining"
        / "successor_active_set_recalibration_audit_v1.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest.pop("contract_sha256")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    launcher = (
        REPOSITORY_ROOT
        / "scripts"
        / "hpc"
        / "run_successor_active_set_recalibration_audit.sh"
    ).read_text()

    assert hashlib.sha256(canonical).hexdigest() == expected_hash
    assert manifest["active_set"]["initial_constraint_count"] == 14
    assert manifest["active_set"]["maximum_rounds"] == 3
    assert manifest["gates"]["maximum_component_reverse_passes"] == 20
    assert manifest["exact_reference"]["component_reverse_passes"] == 40
    assert manifest["execution"]["new_updates_applied"] == 0
    assert manifest["execution"]["validation_values_evaluated"] is False
    assert "requires free gln01" in launcher
    assert "--active-set-recalibration-audit" in launcher
    assert "--exact-recalibration-report" in launcher
    assert "#SBATCH" not in launcher


def test_repeated_active_set_smoke_separates_mechanism_and_promotion():
    objective_validation = _objective_ab_validation(
        primary_candidate=0.999,
        maintenance_candidate=1.01,
    )
    validation = {
        CONTROL: objective_validation[OBJECTIVES[0]],
        CANDIDATE: objective_validation[OBJECTIVES[1]],
    }
    updates = [
        {
            "update_applied": True,
            "direction_passed": True,
            "component_reverse_passes": 14,
            "rounds": [{}],
            "hard_failure_count": 0,
            "nonfinite_value_count": 0,
        }
        for _index in range(4)
    ]
    gates = {
        "training": {
            "expected_active_set_updates": 4,
            "maximum_component_reverse_passes_per_update": 20,
            "maximum_directional_check_rounds_per_update": 3,
            "hard_failure_count": 0,
            "nonfinite_value_count": 0,
        },
        "validation": {
            "minimum_research_primary_composite_relative_improvement": 0.0,
            "minimum_improved_slice_horizon_cells": 4,
            "maximum_maintenance_aggregate_relative_regression": 0.05,
            "maximum_maintenance_single_cell_relative_regression": 0.10,
        },
        "promotion": {
            "minimum_research_primary_composite_relative_improvement": 0.005,
        },
    }

    result = classify_repeated_active_set_smoke(
        validation,
        updates,
        gates=gates,
        pass_action="scale_bounded_updates",
        fail_action="redesign",
    )

    assert result["status"] == "repeated_active_set_smoke_supported"
    assert result["mechanism_passed"] is True
    assert result["directional_validation"]["passed"] is True
    assert result["promotion_threshold_met"] is False


def test_repeated_active_set_smoke_contract_is_bounded():
    manifest_path = (
        REPOSITORY_ROOT
        / "manifests"
        / "coarse_graining"
        / "successor_repeated_active_set_smoke_v1.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest.pop("contract_sha256")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    launcher = (
        REPOSITORY_ROOT
        / "scripts"
        / "hpc"
        / "run_successor_repeated_active_set_smoke.sh"
    ).read_text()

    assert hashlib.sha256(canonical).hexdigest() == expected_hash
    assert manifest["training"]["active_set_updates"] == 4
    assert manifest["training"]["total_direct_updates_after_completion"] == 5
    assert manifest["step_norm_grid"] == [0.002, 0.0015, 0.001, 0.0005]
    assert (
        manifest["gates"]["training"][
            "maximum_component_reverse_passes_per_update"
        ]
        == 20
    )
    assert manifest["execution"]["checkpoint_written"] is False
    assert manifest["execution"]["final_parameters_retained"] is False
    assert manifest["execution"]["validation_values_evaluated"] is True
    assert "requires free gln01" in launcher
    assert "--repeated-active-set-smoke" in launcher
    assert "--active-set-reference-report" in launcher
    assert "#SBATCH" not in launcher


def test_cumulative_microstep_smoke_requires_macro_effect_and_rollout():
    objective_validation = _objective_ab_validation(
        primary_candidate=0.999,
        maintenance_candidate=1.01,
    )
    validation = {
        CONTROL: objective_validation[OBJECTIVES[0]],
        CANDIDATE: objective_validation[OBJECTIVES[1]],
    }
    microsteps = [
        {
            "update_applied": True,
            "microstep_safe": True,
            "direction_passed": True,
            "component_reverse_passes": 13,
            "rounds": [{}],
            "hard_failure_count": 0,
            "nonfinite_value_count": 0,
            "cumulative": {
                "effect_gate_passed": index == 3,
                "component_median_relative_changes": {
                    label: -1.0e-4 if index == 3 else -2.5e-5 * (index + 1) for label in GRADIENT_FEASIBILITY_LABELS
                },
            },
        }
        for index in range(4)
    ]
    gates = {
        "training": {
            "maximum_component_reverse_passes_per_microstep": 20,
            "maximum_directional_check_rounds_per_microstep": 3,
            "hard_failure_count": 0,
            "nonfinite_value_count": 0,
        },
        "validation": {
            "minimum_research_primary_composite_relative_improvement": 0.0,
            "minimum_improved_slice_horizon_cells": 4,
            "maximum_maintenance_aggregate_relative_regression": 0.05,
            "maximum_maintenance_single_cell_relative_regression": 0.10,
        },
        "promotion": {
            "minimum_research_primary_composite_relative_improvement": 0.005,
        },
    }

    result = classify_cumulative_microstep_smoke(
        validation,
        microsteps,
        gates=gates,
        maximum_microsteps=4,
        pass_action="design_macro_training",
        fail_action="stop",
    )

    assert result["status"] == "cumulative_microstep_smoke_supported"
    assert result["mechanism_passed"] is True
    assert result["directional_validation"]["passed"] is True
    assert result["promotion_threshold_met"] is False

    microsteps[-1]["cumulative"]["effect_gate_passed"] = False
    rejected = classify_cumulative_microstep_smoke(
        None,
        microsteps,
        gates=gates,
        maximum_microsteps=4,
        pass_action="design_macro_training",
        fail_action="stop",
    )
    assert rejected["status"] == "cumulative_microstep_smoke_not_supported"
    assert rejected["mechanism_passed"] is False
    assert rejected["directional_validation"] is None


def test_cumulative_microstep_smoke_contract_is_bounded():
    manifest_path = REPOSITORY_ROOT / "manifests" / "coarse_graining" / "successor_cumulative_microstep_smoke_v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest.pop("contract_sha256")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    launcher = (REPOSITORY_ROOT / "scripts" / "hpc" / "run_successor_cumulative_microstep_smoke.sh").read_text()

    assert hashlib.sha256(canonical).hexdigest() == expected_hash
    assert manifest["microsteps"]["step_norm"] == pytest.approx(0.0005)
    assert manifest["microsteps"]["maximum_count"] == 4
    assert manifest["microsteps"]["recalibrate_after_every_microstep"] is True
    assert manifest["gates"]["macro_effect"]["minimum_each_component_median_relative_decrease"] == pytest.approx(1.0e-4)
    assert manifest["validation"]["run_only_if_macro_effect_passes"] is True
    assert manifest["execution"]["checkpoint_written"] is False
    assert manifest["execution"]["final_parameters_retained"] is False
    assert "requires free gln01" in launcher
    assert "--cumulative-microstep-smoke" in launcher
    assert "--repeated-active-set-report" in launcher
    assert "#SBATCH" not in launcher


def test_parameter_geometry_attribution_requires_safe_owned_projection():
    block_names = tuple(SUCCESSOR_PARAMETER_BLOCKS)
    geometry = {
        "block_count": 7,
        "top_level_key_count": 23,
        "direction_squared_fraction_sum": 1.0,
        "nonfinite_value_count": 0,
        "blocks": {name: {} for name in block_names},
    }
    reference = {
        "reference_reproduced": True,
        "direction_passed": True,
        "finite_step_safe": False,
        "hard_failure_count": 0,
        "nonfinite_value_count": 0,
    }
    projections = [
        {
            "excluded_block": name,
            "direction_passed": True,
            "finite_step_safe": name == "output_decoder",
            "hard_failure_count": 0,
            "nonfinite_value_count": 0,
        }
        for name in block_names
    ]
    gates = {
        "geometry": {
            "expected_block_count": 7,
            "expected_top_level_key_count": 23,
            "maximum_direction_squared_fraction_sum_error": 1.0e-12,
            "nonfinite_value_count": 0,
        },
        "projection": {
            "hard_failure_count": 0,
            "nonfinite_value_count": 0,
        },
        "attribution": {"minimum_safe_leave_one_out_projections": 1},
    }

    result = classify_parameter_geometry_attribution(
        geometry,
        reference,
        projections,
        gates=gates,
        pass_action="design_preconditioner",
        fail_action="review_boundary",
    )

    assert result["status"] == "parameter_geometry_block_attributed"
    assert result["safe_excluded_blocks"] == ["output_decoder"]
    projections[-1]["finite_step_safe"] = False
    rejected = classify_parameter_geometry_attribution(
        geometry,
        reference,
        projections,
        gates=gates,
        pass_action="design_preconditioner",
        fail_action="review_boundary",
    )
    assert rejected["status"] == "parameter_geometry_not_attributed"


def test_parameter_geometry_attribution_contract_is_bounded():
    manifest_path = (
        REPOSITORY_ROOT
        / "manifests"
        / "coarse_graining"
        / "successor_parameter_geometry_attribution_v1.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest.pop("contract_sha256")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    launcher = (
        REPOSITORY_ROOT
        / "scripts"
        / "hpc"
        / "run_successor_parameter_geometry_attribution.sh"
    ).read_text()

    assert hashlib.sha256(canonical).hexdigest() == expected_hash
    assert manifest["parameter_blocks"] == {
        name: list(keys) for name, keys in SUCCESSOR_PARAMETER_BLOCKS.items()
    }
    assert manifest["projections"]["candidate_count"] == 7
    assert manifest["projections"]["global_step_norm"] == pytest.approx(0.0005)
    assert manifest["execution"]["maximum_projection_candidates"] == 7
    assert manifest["execution"]["optimizer_updates"] == 0
    assert manifest["execution"]["validation_values_evaluated"] is False
    assert manifest["execution"]["final_parameters_retained"] is False
    assert "requires free gln01" in launcher
    assert "probe_successor_parameter_geometry" in launcher
    assert "#SBATCH" not in launcher


def test_decoder_leaf_attribution_contract_is_bounded():
    manifest_path = (
        REPOSITORY_ROOT
        / "manifests"
        / "coarse_graining"
        / "successor_decoder_leaf_attribution_v1.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash = manifest.pop("contract_sha256")
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    launcher = (
        REPOSITORY_ROOT
        / "scripts"
        / "hpc"
        / "run_successor_decoder_leaf_attribution.sh"
    ).read_text()

    assert hashlib.sha256(canonical).hexdigest() == expected_hash
    assert manifest["decoder_leaves"] == {
        name: list(path) for name, path in SUCCESSOR_DECODER_LEAVES.items()
    }
    assert manifest["projections"]["candidate_count"] == 4
    assert manifest["projections"]["global_step_norm"] == pytest.approx(0.0005)
    assert manifest["execution"]["maximum_projection_candidates"] == 4
    assert manifest["execution"]["optimizer_updates"] == 0
    assert manifest["execution"]["validation_values_evaluated"] is False
    assert "requires free gln01" in launcher
    assert "--decoder-leaf-attribution" in launcher
    assert "#SBATCH" not in launcher


def _write_scale_admission_fixture(tmp_path):
    scales = {
        str(horizon): {
            "aboveground_biomass_carbon": float(horizon),
            "belowground_biomass_carbon": float(horizon + 1),
        }
        for horizon in (1, 7, 30)
    }
    asset = {
        "schema_version": "successor_named_biomass_tendency_scale_asset_v1",
        "status": "passed",
        "git_head": "a" * 40,
        "identity": {"tendency_contract_sha256": "b" * 64},
        "scales": scales,
        "gates": {
            "teacher_teacher_loss_exact_zero": True,
            "agb_plus_bgb_identity_passed": True,
            "maximum_agb_plus_bgb_identity_tolerance_ratio": 0.25,
        },
        "training_updates": 0,
        "checkpoint_written": False,
        "teacher_data_regenerated": False,
        "historical_v5_used": False,
        "validation_values_evaluated": False,
        "sealed_test_used": False,
    }
    canonical = json.dumps(asset, sort_keys=True, separators=(",", ":")).encode()
    asset["asset_sha256"] = hashlib.sha256(canonical).hexdigest()
    asset_path = tmp_path / "scale_asset.json"
    asset_path.write_text(json.dumps(asset), encoding="utf-8")
    acceptance = {
        "schema_version": "successor_named_biomass_tendency_scale_acceptance_v1",
        "status": "accepted",
        "tendency_contract_sha256": "b" * 64,
        "asset": {
            "file_sha256": hashlib.sha256(asset_path.read_bytes()).hexdigest(),
            "asset_sha256": asset["asset_sha256"],
            "git_head": "a" * 40,
        },
        "scales": scales,
    }
    canonical = json.dumps(
        acceptance,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    acceptance["contract_sha256"] = hashlib.sha256(canonical).hexdigest()
    acceptance_path = tmp_path / "acceptance.json"
    acceptance_path.write_text(json.dumps(acceptance), encoding="utf-8")
    return asset_path, acceptance_path


def test_named_biomass_tendency_scale_loader_admits_hash_bound_asset(tmp_path):
    asset_path, acceptance_path = _write_scale_admission_fixture(tmp_path)

    result = load_named_biomass_tendency_scale_asset(asset_path, acceptance_path)

    assert result.scales[30]["aboveground_biomass_carbon"] == 30.0
    assert result.file_sha256 == hashlib.sha256(asset_path.read_bytes()).hexdigest()
    assert _scale_asset_identity(result) == {
        "tendency_scale_asset_sha256": result.payload["asset_sha256"],
        "tendency_scale_asset_file_sha256": result.file_sha256,
    }


def test_named_biomass_tendency_scale_loader_rejects_file_drift(tmp_path):
    asset_path, acceptance_path = _write_scale_admission_fixture(tmp_path)
    payload = json.loads(asset_path.read_text(encoding="utf-8"))
    payload["scales"]["30"]["aboveground_biomass_carbon"] = 31.0
    asset_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="identity drift"):
        load_named_biomass_tendency_scale_asset(asset_path, acceptance_path)


def test_named_biomass_tendency_scale_launcher_is_free_cpu_and_zero_update():
    launcher = (
        REPOSITORY_ROOT
        / "scripts"
        / "hpc"
        / "run_successor_named_biomass_tendency_scale_fit.sh"
    ).read_text()
    source = (
        REPOSITORY_ROOT
        / "scripts"
        / "hpc"
        / "prepare_successor_named_biomass_tendency_scales.py"
    ).read_text()

    assert "biomass tendency scales require a free CPU test node" in launcher
    assert "JAX_PLATFORMS=cpu" in launcher
    assert "#SBATCH" not in launcher
    assert '"training_updates": 0' in source
    assert '"checkpoint_written": False' in source
    assert '"validation_values_evaluated": False' in source
    assert "successor_adamw_update" not in source


def test_named_biomass_gradient_launcher_is_free_gpu_and_zero_update():
    launcher = (
        REPOSITORY_ROOT
        / "scripts"
        / "hpc"
        / "run_successor_named_biomass_gradient_preflight.sh"
    ).read_text()
    source = (
        REPOSITORY_ROOT
        / "scripts"
        / "hpc"
        / "probe_successor_named_biomass_gradients.py"
    ).read_text()

    assert "requires free gln01" in launcher
    assert "#SBATCH" not in launcher
    assert '"training_updates": 0' in source
    assert '"checkpoint_written": False' in source
    assert '"validation_values_evaluated": False' in source
    assert "successor_adamw_update" not in source
