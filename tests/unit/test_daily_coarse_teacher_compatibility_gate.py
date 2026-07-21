from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np

from research.daily_coarse_graining.gradient_training_ready import (
    deterministic_boundary_fields,
    deterministic_exact_metrics,
)
from research.daily_coarse_graining.teacher_compatibility_gate import (
    _array_metrics,
    compare_fingerprints,
)


def test_array_metrics_accepts_tolerance_and_matching_nonfinite_pattern():
    expected = np.asarray([1.0, np.nan, np.inf, -np.inf])
    actual = np.asarray([1.0 + 1.0e-12, np.nan, np.inf, -np.inf])
    result = _array_metrics(expected, actual, atol=1.0e-10, rtol=1.0e-10)
    assert result["passed"]
    assert result["nonfinite_pattern_equal"]


def test_array_metrics_rejects_real_difference():
    result = _array_metrics(
        np.asarray([1.0]), np.asarray([1.01]), atol=1.0e-10, rtol=1.0e-10
    )
    assert not result["passed"]


def _deterministic_sample(*, t2m_offset: float):
    forcing = SimpleNamespace(
        temp_air=np.linspace(295.0, 300.0, 48, dtype=np.float64)[:, None],
        precip_rain=np.zeros((48, 1), dtype=np.float64),
        precip_snow=np.zeros((48, 1), dtype=np.float64),
    )
    expected = deterministic_boundary_fields(
        forcing, dt_sechiba=1800.0, dt_stomate=86400.0
    )
    daily = {name: np.asarray(value) for name, value in expected.items()}
    daily["t2m_daily"] = daily["t2m_daily"] + t2m_offset
    return SimpleNamespace(
        day_index=4,
        forcing=forcing,
        record=SimpleNamespace(
            daily_fold=SimpleNamespace(daily_fields=daily),
            half_hour_transition=SimpleNamespace(
                completed_entry_payloads=({"t2mdiag": daily["t2mdiag"]},)
            ),
        ),
    )


def test_deterministic_float_gate_reports_ulp_difference_but_accepts_tolerance():
    result = deterministic_exact_metrics(
        (_deterministic_sample(t2m_offset=5.684341886080802e-14),),
        dt_sechiba=1800.0,
        dt_stomate=86400.0,
    )
    assert result["passed"]
    assert not result["fields"]["t2m_daily"]["exact"]
    assert result["fields"]["t2m_daily"]["within_tolerance"]


def test_deterministic_float_gate_rejects_difference_above_tolerance():
    result = deterministic_exact_metrics(
        (_deterministic_sample(t2m_offset=1.0e-9),),
        dt_sechiba=1800.0,
        dt_stomate=86400.0,
    )
    assert not result["passed"]


def _write_artifact(prefix, continuous, discrete, *, git_head="abc", leaves=None):
    np.savez(prefix.with_suffix(".npz"), continuous=continuous, discrete__mask=discrete)
    if leaves is None:
        leaves = [
            {
                "family": "hydrol",
                "component": "hydrol_previous_step_state",
                "name": "mc",
                "shape": [2],
                "start": 0,
                "stop": 2,
            }
        ]
    metadata = {
        "schema_version": "teacher_daily_boundary_fingerprint_v1",
        "teacher_commit": "teacher",
        "experiment_git_head": git_head,
        "jax": {"version": "test", "backend": "cpu", "devices": ["cpu"]},
        "capture": {
            "year": 1961,
            "days": 1,
            "state_cache_sha256": "state",
            "config_sha256": "config",
            "run_def_sha256": "run-def",
            "reference_restart_sha256": {
                "driver_start.nc": "driver",
                "sechiba_start.nc": "sechiba",
                "stomate_start.nc": "stomate",
                "stomate_restart.nc": "stomate-restart",
            },
            "leaves": leaves,
        },
    }
    prefix.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")


def test_compare_fingerprints_requires_numeric_discrete_and_contract_parity(tmp_path):
    expected = tmp_path / "expected"
    actual = tmp_path / "actual"
    _write_artifact(expected, np.asarray([[1.0, 2.0]]), np.asarray([[True, False]]))
    _write_artifact(actual, np.asarray([[1.0, 2.0]]), np.asarray([[True, False]]))
    report = compare_fingerprints(
        expected,
        actual,
        output=tmp_path / "report.json",
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    assert report["passed"]

    _write_artifact(actual, np.asarray([[1.0, 2.0]]), np.asarray([[False, False]]))
    report = compare_fingerprints(
        expected,
        actual,
        output=tmp_path / "failed.json",
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    assert not report["passed"]
    assert not report["discrete"]["discrete__mask"]["passed"]


def test_compare_fingerprints_aligns_continuous_leaves_by_identity(tmp_path):
    expected = tmp_path / "expected"
    actual = tmp_path / "actual"
    first = {
        "family": "hydrol",
        "component": "state",
        "name": "first",
        "shape": [1],
        "start": 0,
        "stop": 1,
    }
    second = {
        "family": "hydrol",
        "component": "state",
        "name": "second",
        "shape": [1],
        "start": 1,
        "stop": 2,
    }
    actual_second = {**second, "start": 0, "stop": 1}
    actual_first = {**first, "start": 1, "stop": 2}
    _write_artifact(
        expected,
        np.asarray([[1.0, 2.0]]),
        np.asarray([[True]]),
        leaves=[first, second],
    )
    _write_artifact(
        actual,
        np.asarray([[2.0, 1.0]]),
        np.asarray([[True]]),
        leaves=[actual_second, actual_first],
    )
    report = compare_fingerprints(
        expected,
        actual,
        output=tmp_path / "report.json",
        atol=0.0,
        rtol=0.0,
    )
    assert report["passed"]
