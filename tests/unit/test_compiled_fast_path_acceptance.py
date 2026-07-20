from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.dev.audit_compiled_fast_path_acceptance import ComparisonAccumulator  # noqa: E402


def test_comparison_accumulator_requires_discrete_exact_and_reports_float_metrics() -> None:
    comparison = ComparisonAccumulator(atol=1.0e-8, rtol=1.0e-10)
    expected = {
        "float": np.asarray([1.0, 2.0], dtype=np.float64),
        "mask": np.asarray([True, False]),
        "index": np.asarray([1, 2], dtype=np.int32),
    }
    actual = {
        "float": np.nextafter(expected["float"], np.inf),
        "mask": expected["mask"].copy(),
        "index": expected["index"].copy(),
    }
    comparison.compare("state", actual, expected)
    report = comparison.report()

    assert report["passed"]
    assert report["float_leaves"] == 1
    assert report["discrete_leaves"] == 2
    assert report["max_ulp_error"] == 1


def test_comparison_accumulator_rejects_schema_and_discrete_differences() -> None:
    schema = ComparisonAccumulator(atol=1.0e-8, rtol=1.0e-10)
    schema.compare("state", {"a": 1}, {"b": 1})
    assert not schema.report()["passed"]
    assert schema.report()["failures"][0]["reason"] == "mapping_schema_mismatch"

    discrete = ComparisonAccumulator(atol=1.0e-8, rtol=1.0e-10)
    discrete.compare("state.mask", np.asarray([True]), np.asarray([False]))
    assert not discrete.report()["passed"]
    assert discrete.report()["failures"][0]["reason"] == "discrete_not_exact"


def test_comparison_accumulator_rejects_float_outside_fixed_tolerance() -> None:
    comparison = ComparisonAccumulator(atol=1.0e-8, rtol=1.0e-10)
    comparison.compare("state.float", np.asarray([1.0e-4]), np.asarray([0.0]))
    report = comparison.report()

    assert not report["passed"]
    assert report["max_abs_error"] == 1.0e-4
    assert report["failures"][0]["reason"] == "float_tolerance_failure"
