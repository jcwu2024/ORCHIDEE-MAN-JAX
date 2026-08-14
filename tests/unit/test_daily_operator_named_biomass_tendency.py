from __future__ import annotations

import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.gradient_feasibility import (
    GRADIENT_FEASIBILITY_LABELS,
    classify_gradient_feasibility,
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
