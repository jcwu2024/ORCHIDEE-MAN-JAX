from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.stomate.trace_schema import (
    TRACE_COLUMN_GROUPS,
    columns_from_trace,
    require_trace_columns,
    validate_trace_columns,
)


def test_validate_trace_columns_accepts_complete_group_mapping():
    columns = {name: 1.0 for name in TRACE_COLUMN_GROUPS["gpp_accumulation"]}

    result = validate_trace_columns(columns, ("gpp_accumulation",))

    assert result.ok
    assert result.groups == ("gpp_accumulation",)
    assert result.missing_by_group["gpp_accumulation"] == ()
    assert result.unknown_columns == ()


def test_validate_trace_columns_reports_grouped_missing_and_unknown_columns():
    trace = {
        "itime": 1,
        "date": 19610101,
        "gpp_d_before_accu": 2.0,
        "extra_debug": 99.0,
    }

    result = validate_trace_columns(trace, ("gpp_accumulation", "resp_maint_part_accumulation"))

    assert not result.ok
    assert "dt_sechiba" in result.missing_by_group["gpp_accumulation"]
    assert "resp_maint_part_after_accum" in result.missing_by_group["resp_maint_part_accumulation"]
    assert result.unknown_columns == ("extra_debug",)


def test_require_trace_columns_raises_grouped_error_for_missing_columns():
    with pytest.raises(ValueError, match="gpp_accumulation"):
        require_trace_columns({"itime": 1}, ("gpp_accumulation",))


def test_columns_from_csv_reads_header_only(tmp_path: Path):
    path = tmp_path / "trace.csv"
    path.write_text("itime,date,do_slow\n1,19610101,T\n", encoding="utf-8")

    assert columns_from_trace(path) == ("itime", "date", "do_slow")


def test_unknown_schema_group_is_rejected():
    with pytest.raises(KeyError, match="unknown_group"):
        validate_trace_columns({}, ("unknown_group",))

