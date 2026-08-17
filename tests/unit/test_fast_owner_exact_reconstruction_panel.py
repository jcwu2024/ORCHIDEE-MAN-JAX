from __future__ import annotations

import pytest

from research.daily_coarse_graining.fast_owner_exact_reconstruction_panel import (
    parse_selection,
    summarize_panel_fields,
)


def _metric(*, passed: bool, error: float, count: int = 4) -> dict:
    return {
        "passed": passed,
        "value_count": count,
        "maximum_absolute_error": error,
        "root_mean_squared_error": error / 2.0,
        "nonfinite_reconstructed_count": 0,
    }


def _record(*, z0m_passed: bool, z0m_error: float) -> dict:
    return {
        "source_reconstruction": {
            "condveg_roughness": {
                "fields": {
                    "z0m": _metric(passed=z0m_passed, error=z0m_error),
                    "z0h": _metric(passed=False, error=0.1),
                }
            },
            "hydrol_drysoil_fraction": _metric(passed=True, error=1.0e-15),
            "thermosoil_final_diagnostics": {"fields": {"gtemp": _metric(passed=True, error=0.0)}},
        }
    }


def test_parse_selection_uses_last_colon() -> None:
    assert parse_selection("001.0-071.0:1991") == ("001.0-071.0", 1991)
    with pytest.raises(ValueError, match="LANDPOINT:YEAR"):
        parse_selection("001.0-071.0")


def test_panel_summary_aggregates_passes_errors_and_counts() -> None:
    result = summarize_panel_fields(
        (
            _record(z0m_passed=True, z0m_error=1.0e-15),
            _record(z0m_passed=False, z0m_error=2.0e-4),
        )
    )

    z0m = result["condveg_roughness.z0m"]
    assert z0m["point_year_count"] == 2
    assert z0m["passed_point_year_count"] == 1
    assert z0m["failed_point_year_count"] == 1
    assert z0m["value_count"] == 8
    assert z0m["maximum_absolute_error"] == 2.0e-4


def test_panel_summary_rejects_field_registry_drift() -> None:
    first = _record(z0m_passed=True, z0m_error=0.0)
    second = _record(z0m_passed=True, z0m_error=0.0)
    del second["source_reconstruction"]["condveg_roughness"]["fields"]["z0h"]

    with pytest.raises(ValueError, match="registries drift"):
        summarize_panel_fields((first, second))
