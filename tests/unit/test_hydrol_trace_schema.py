from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.hydrol_trace import (
    required_columns,
    trace_group,
    trace_group_names,
    validate_columns,
)


def test_hydrol_trace_schema_exposes_full_solve_groups():
    names = trace_group_names()

    assert names == (
        "setup",
        "pre_solve",
        "solve",
        "drainage_conservation",
        "post_solve",
        "alternate_path",
    )


def test_hydrol_trace_group_records_fortran_provenance_and_consumer():
    group = trace_group("drainage_conservation")

    assert "check_tr_ns" in group.columns
    assert "dr_corrnum_ns" in group.columns
    assert any("hydrol.f90:6007-6052" in item for item in group.fortran_provenance)
    assert "hydrol" in group.consumer


def test_hydrol_trace_required_columns_include_exact_state_unlocks():
    setup = required_columns("setup")
    post = required_columns("post_solve")
    alternate = required_columns("alternate_path")

    assert "a_raw" in setup
    assert "gp" in setup
    assert "mc_after_mcl_update" in post
    assert "mcl_final" in post
    assert "resolv_alternate" in alternate
    assert "rhs_residual" in alternate


def test_validate_columns_reports_missing_by_group_for_mapping():
    columns = {"kjit": [1], "ji": [1], "jst": [4], "check_tr_ns": [0.0]}

    result = validate_columns(columns, groups=("drainage_conservation",))

    assert not result.is_valid
    assert "dt_days" in result.missing_by_group["drainage_conservation"]
    assert "dr_corrnum_ns" in result.missing_by_group["drainage_conservation"]
    assert "check_tr_ns" not in result.missing_by_group["drainage_conservation"]


def test_validate_columns_accepts_csv_header():
    path = Path(__file__).with_name("_hydrol_trace_schema_test.csv")
    try:
        path.write_text(",".join(required_columns("solve")) + "\n", encoding="utf-8")

        result = validate_columns(path, groups=("solve",))

        assert result.is_valid
        assert result.missing_by_group == {"solve": ()}
    finally:
        path.unlink(missing_ok=True)
