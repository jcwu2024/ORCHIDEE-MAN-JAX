"""Failure gates for the bounded cross-revision production-adapter audit."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location(
    "carbon_core_sync_audit", Path(__file__).resolve().parents[2] / "scripts/dev/verify_global_carbon_core_sync.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_fixed_tolerance_reports_zeros_separately_from_relative_error():
    reference = np.array([0.0, 100.0], dtype=np.float64)
    candidate = np.array([audit.ATOL / 2, 100.0 + audit.RTOL * 100.0], dtype=np.float64)
    record = audit.compare_leaf(reference, candidate)
    assert record["passed"]
    assert not record["exact"]
    assert record["reference_zero_count"] == record["changed_reference_zero_count"] == 1
    assert record["max_relative_nonzero"] == pytest.approx(audit.RTOL, rel=1e-5)
    assert record["max_bound_ratio"] == 0.5
    assert not audit.compare_leaf(np.zeros(1), np.array([2 * audit.ATOL]))["passed"]


@pytest.mark.parametrize("candidate", [np.zeros(2), np.zeros(1, dtype=np.float32)])
def test_schema_changes_fail(candidate):
    assert not audit.compare_leaf(np.zeros(1, dtype=np.float64), candidate)["passed"]


@pytest.mark.parametrize(
    "reference,candidate",
    [
        (np.array([False]), np.array([True])),
        (np.array([1], dtype=np.int32), np.array([2], dtype=np.int32)),
        (np.array(["a"]), np.array(["b"])),
    ],
)
def test_discrete_and_static_text_outputs_require_exactness(reference, candidate):
    assert audit.compare_leaf(reference, reference)["passed"]
    record = audit.compare_leaf(reference, candidate)
    assert not record["discrete_exact"]
    assert not record["passed"]


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_matching_nonfinite_values_still_fail(value):
    record = audit.compare_leaf(np.array([value]), np.array([value]))
    assert not record["finite"]
    assert not record["passed"]


def write_workers(directory, key="entry/mode/.biomass", delta=0.0):
    for name, pin, pid in (("baseline", "old", 1), ("candidate", "new", 2)):
        np.savez_compressed(
            directory / f"{name}.npz", **{key: np.array([1.0 + (delta if name == "candidate" else 0.0)])}
        )
        (directory / f"{name}.json").write_text(
            json.dumps(
                {
                    "pin": pin,
                    "pid": pid,
                    "fixture_sha256": "same",
                    "runtime": {"platform": "cpu"},
                    "structures": {"entry/mode": "same"},
                }
            ),
            encoding="utf-8",
        )


def test_stock_changes_fail_even_inside_fixed_forward_tolerance(tmp_path):
    write_workers(tmp_path, delta=audit.ATOL / 2)
    result = audit.compare_workers(tmp_path, "old", "new")
    assert result["metadata_gates_passed"]
    assert not result["passed"]
    assert result["failures"][0]["stock_output"]
    assert result["failures"][0]["max_bound_ratio"] < 1.0


def test_key_or_static_structure_mismatches_fail_metadata_gate(tmp_path):
    write_workers(tmp_path)
    assert audit.compare_workers(tmp_path, "old", "new")["passed"]
    np.savez_compressed(tmp_path / "candidate.npz", **{"other": np.array([1.0])})
    assert not audit.compare_workers(tmp_path, "old", "new")["metadata_gates_passed"]
    write_workers(tmp_path)
    metadata = json.loads((tmp_path / "candidate.json").read_text())
    metadata["structures"] = {"entry/mode": "different"}
    (tmp_path / "candidate.json").write_text(json.dumps(metadata), encoding="utf-8")
    assert not audit.compare_workers(tmp_path, "old", "new")["metadata_gates_passed"]
