from __future__ import annotations

import copy

import numpy as np

from scripts.dev.compare_physical_microbatches import (
    TOLERANCES,
    compare_numeric_trees,
    compare_physical_microbatches,
)


def _identity():
    return {
        "model_protocol": {"seed": 20260911, "loss": "physical-v2"},
        "benchmark_protocol": {
            "schema_version": "physical_microbatch_benchmark_v1",
            "timed_updates": 3,
            "compiled_memory_limit_gib": 24,
        },
        "rows": [{"artifact_id": index // 32, "batch_row": index} for index in range(64)],
        "coefficients": [1.0, 0.1, 0.1],
    }


def _report(microbatch_size, seconds):
    updates = [
        {
            "update": index + 1,
            "loss": 3.0 - index * 0.1,
            "compute_seconds": value,
            "parameters_optimizer_finite": True,
            "hard": {"budget": 0, "nonfinite": 0},
        }
        for index, value in enumerate(seconds)
    ]
    return {
        "schema_version": "physical_microbatch_probe_v1",
        "status": "passed",
        "source_git_head": "same-commit",
        "microbatch_size": microbatch_size,
        "parent_batch_size": 64,
        "identity": _identity(),
        "updates": updates,
        "completed_updates": 3,
        "median_compute_seconds": float(np.median(seconds)),
        "warmup_updates_discarded": 1,
        "same_batch_repeated": True,
        "all_hard_counts_zero": True,
        "all_updates_finite": True,
        "retained_model": False,
        "checkpoint_written": False,
        "checkpoint_inherited": False,
        "compiled_memory": {"estimated_live_bytes": 10 * 1024**3},
        "memory": {
            "available": True,
            "peak_bytes_in_use": 12 * 1024**3,
            "bytes_limit": 32 * 1024**3,
        },
    }


def _stage(*, gradient_shift=0.0, delta_shift=0.0):
    return {
        "identity": _identity(),
        "stage": "first_update",
        "result": {
            "loss": 3.0,
            "gradient": {
                "large": np.asarray([2.0, -1.0, 0.25], dtype=np.float32)
                + np.float32(gradient_shift),
                "tiny": [np.asarray([0.0, 2.0e-8], dtype=np.float32)],
            },
            "parameter_delta": {
                "large": np.asarray([-1.0e-5, 8.0e-6, 0.0], dtype=np.float32)
                + np.float32(delta_shift),
                "tiny": [np.asarray([0.0, 0.0], dtype=np.float32)],
            },
        },
    }


def test_equivalent_complete_updates_select_faster_feasible_mode():
    report8 = _report(8, [2.1, 2.0, 2.2])
    report16 = _report(16, [1.4, 1.5, 1.6])

    result = compare_physical_microbatches(report8, _stage(), report16, _stage())

    assert result["status"] == "passed"
    assert result["identity_matches"]
    assert result["first_update"]["gradient"]["structure_matches"]
    assert result["first_update"]["gradient"]["dtypes_match"]
    assert result["first_update"]["parameter_delta"]["passed"]
    assert result["recommended_microbatch_size"] == 16
    assert result["microbatch4_baseline_measured"] is False
    assert result["performance"]["observed_median_compute_speedup"] == 2.1 / 1.5


def test_gradient_and_adam_delta_gates_reject_material_reduction_order_drift():
    report8 = _report(8, [2.0, 2.0, 2.0])
    report16 = _report(16, [1.0, 1.0, 1.0])

    gradient_drift = compare_physical_microbatches(
        report8, _stage(), report16, _stage(gradient_shift=2.0e-3)
    )
    assert gradient_drift["status"] == "rejected"
    assert not gradient_drift["first_update"]["gradient"]["passed"]
    assert gradient_drift["first_update"]["gradient"]["failed_coordinates"] > 0
    assert gradient_drift["recommended_microbatch_size"] is None

    delta_drift = compare_physical_microbatches(
        report8, _stage(), report16, _stage(delta_shift=2.0e-6)
    )
    assert delta_drift["status"] == "rejected"
    assert not delta_drift["first_update"]["parameter_delta"]["passed"]


def test_tree_comparison_requires_exact_structure_shape_and_dtype():
    reference = {"weight": np.asarray([1.0, 2.0], dtype=np.float32)}
    changed_dtype = {"weight": np.asarray([1.0, 2.0], dtype=np.float64)}
    changed_shape = {"weight": np.asarray([[1.0, 2.0]], dtype=np.float32)}
    changed_structure = {"other": np.asarray([1.0, 2.0], dtype=np.float32)}

    assert not compare_numeric_trees(reference, changed_dtype, tolerance=TOLERANCES["gradient"])[
        "dtypes_match"
    ]
    assert not compare_numeric_trees(reference, changed_shape, tolerance=TOLERANCES["gradient"])[
        "shapes_match"
    ]
    assert not compare_numeric_trees(reference, changed_structure, tolerance=TOLERANCES["gradient"])[
        "structure_matches"
    ]

    nonfinite = compare_numeric_trees(
        reference,
        {"weight": np.asarray([1.0, np.nan], dtype=np.float32)},
        tolerance=TOLERANCES["gradient"],
    )
    assert not nonfinite["passed"]
    assert nonfinite["error_l2"] is None


def test_identity_and_resource_failures_prevent_a_speed_recommendation():
    report8 = _report(8, [2.0, 2.1, 2.2])
    report16 = _report(16, [1.0, 1.1, 1.2])
    different_stage = copy.deepcopy(_stage())
    different_stage["identity"]["rows"][0]["batch_row"] = 999

    identity_failure = compare_physical_microbatches(report8, _stage(), report16, different_stage)
    assert not identity_failure["identity_matches"]
    assert identity_failure["recommended_microbatch_size"] is None

    report16["memory"]["peak_bytes_in_use"] = report16["memory"]["bytes_limit"] + 1
    resource_failure = compare_physical_microbatches(report8, _stage(), report16, _stage())
    assert not resource_failure["both_modes_feasible"]
    assert resource_failure["recommended_microbatch_size"] is None


def test_near_tied_speed_prefers_lower_peak_memory():
    report8 = _report(8, [1.02, 1.03, 1.04])
    report16 = _report(16, [1.0, 1.0, 1.01])
    report16["memory"]["peak_bytes_in_use"] = 14 * 1024**3

    result = compare_physical_microbatches(report8, _stage(), report16, _stage())

    assert result["status"] == "passed"
    assert result["performance"]["faster_observed_microbatch_size"] == 16
    assert result["recommended_microbatch_size"] == 8
    assert "lower benchmark peak memory" in result["recommendation_basis"]
