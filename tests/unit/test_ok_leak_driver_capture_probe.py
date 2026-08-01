from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from jax_orchidee.driver.orchestration import DriverCompiledOkLeakCarry
from scripts.dev.probe_ok_leak_driver_capture import (
    _array_comparison,
    _capture_probe_passes,
    _ok_leak_endpoint_comparisons,
    _replayed_ok_leak_endpoint_comparisons,
    _within_tolerance,
)


def _comparison(*, exact: bool = True, error: float = 0.0, status: int = 0):
    return {
        "shape_equal": True,
        "exact": exact,
        "max_absolute_error": error,
        "defined_status_mismatches": status,
    }


def test_array_comparison_tracks_value_and_defined_status_mismatches():
    value = _array_comparison(
        np.asarray([1.0, np.nan]),
        np.asarray([1.0 + 5.0e-13, np.nan]),
    )
    assert not value["exact"]
    assert value["max_absolute_error"] == pytest.approx(5.0e-13)
    assert value["defined_status_mismatches"] == 0
    assert _within_tolerance(value, 1.0e-12)

    status = _array_comparison(
        np.asarray([1.0, np.nan]),
        np.asarray([1.0, 2.0]),
    )
    assert status["defined_status_mismatches"] == 1
    assert not _within_tolerance(status, 1.0e-12)


def test_scaled_state_tolerance_is_relative_away_from_zero_and_absolute_near_zero():
    scaled = _array_comparison(
        np.asarray([2.0e6 + 9.313225746154785e-8]),
        np.asarray([2.0e6]),
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    assert scaled["max_absolute_error"] > 1.0e-12
    assert scaled["within_tolerance"]
    assert scaled["max_tolerance_ratio"] < 1.0

    near_zero = _array_comparison(
        np.asarray([2.0e-12]),
        np.asarray([0.0]),
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    assert not near_zero["within_tolerance"]


def test_capture_interface_gate_is_independent_of_full_state_reentry_diagnostic():
    assert _capture_probe_passes(
        {"soil_mc": _comparison()},
        {"exact": True},
        {"ok_leak.DOC": _comparison()},
        {"date": _comparison()},
    )


def test_probe_gate_ignores_unrelated_historical_fast_target_drift():
    all_targets = {
        "daily_interface.t2m_min_daily": _comparison(
            exact=False,
            error=299.47472890218097,
        ),
        "ok_leak.carbon_32l": _comparison(),
        "ok_leak.DOC": _comparison(),
        "ok_leak.deepC_peat": _comparison(),
    }
    ok_leak = _ok_leak_endpoint_comparisons(all_targets)

    assert tuple(ok_leak) == (
        "ok_leak.carbon_32l",
        "ok_leak.DOC",
        "ok_leak.deepC_peat",
    )
    assert _capture_probe_passes(
        {"soil_mc": _comparison()},
        {"exact": True},
        ok_leak,
        {"date": _comparison()},
    )


def test_probe_gate_rejects_ok_leak_or_empty_endpoint_evidence():
    arguments = (
        {"soil_mc": _comparison()},
        {"exact": True},
        {"ok_leak.DOC": _comparison(exact=False, error=2.0e-12)},
        {"date": _comparison()},
    )
    assert not _capture_probe_passes(*arguments)
    assert not _capture_probe_passes(
        arguments[0],
        arguments[1],
        {},
        arguments[3],
    )


def test_persisted_replay_compares_carry_and_derived_peat_endpoints():
    carry = DriverCompiledOkLeakCarry(
        litter_above=np.asarray([1.0, 2.0]),
        litter_below=np.asarray([0.0]),
        lignin_struc_above=np.asarray([0.0]),
        lignin_struc_below=np.asarray([0.0]),
        litterpart=np.asarray([0.0]),
        dead_leaves=np.asarray([0.0]),
        fuel_1hr=np.asarray([0.0]),
        fuel_10hr=np.asarray([0.0]),
        fuel_100hr=np.asarray([0.0]),
        fuel_1000hr=np.asarray([0.0]),
        carbon_32l=np.asarray([0.0]),
        doc=np.asarray([3.0]),
        interception_storage=np.asarray([0.0]),
    )
    replayed = (
        carry,
        SimpleNamespace(
            soilcarbon=SimpleNamespace(
                perma_peat=SimpleNamespace(deepc_peat=np.asarray([4.0]))
            )
        ),
    )
    leaves = (
        SimpleNamespace(
            family="ok_leak",
            path=("litter_above",),
            key="ok_leak.litter_above",
            start=0,
            stop=2,
        ),
        SimpleNamespace(
            family="ok_leak",
            path=("DOC",),
            key="ok_leak.DOC",
            start=2,
            stop=3,
        ),
        SimpleNamespace(
            family="ok_leak",
            path=("deepC_peat",),
            key="ok_leak.deepC_peat",
            start=3,
            stop=4,
        ),
    )

    comparisons = _replayed_ok_leak_endpoint_comparisons(
        replayed,
        np.asarray([1.0, 2.0, 3.0, 4.0]),
        leaves,
    )

    assert set(comparisons) == {
        "ok_leak.litter_above",
        "ok_leak.DOC",
        "ok_leak.deepC_peat",
    }
    assert all(item["exact"] for item in comparisons.values())
