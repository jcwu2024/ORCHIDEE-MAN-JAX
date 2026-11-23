from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.hydrol_trace import (  # noqa: E402
    POST_TILE_LIST_DIRECTED_COLUMNS,
    required_columns,
    trace_group,
    trace_group_names,
    validate_columns,
)
from jax_orchidee.trace.server_1961 import find_server_record, read_server_records, trace_schema  # noqa: E402


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


def test_server_hydrol_post_tile_reads_source_solve_mask(tmp_path):
    schema = trace_schema("hydrol_post")
    trace = tmp_path / schema.trace_file
    trace.write_text(
        # WRITE(912,*) emits kjit, ji, jst, resolv(ji), tmci, tmcf, then fluxes.
        " post_tile 1 1 1 F 610.0 609.0\n"
        "  2.0 3.0 4.0 5.0 6.0 7.0 8.0 0.02\n"
        " post_tile 1 1 4 T 600.0 599.0\n"
        "  12.0 13.0 14.0 15.0 16.0 17.0 18.0 0.02\n",
        encoding="ascii",
    )

    first, active = read_server_records("hydrol_post", tags="post_tile", limit=2, root=tmp_path)
    assert schema.tags["post_tile"].fields == POST_TILE_LIST_DIRECTED_COLUMNS
    assert first["resolv"] is False
    assert active["resolv"] is True
    assert "branch_peat" not in first and "branch_peat" not in active
    assert (active["start_line"], active["jst"], active["tmci"], active["dt_days"]) == (3, 4, 600.0, 0.02)
    assert find_server_record(
        "hydrol_post", tag="post_tile", criteria={"resolv": True, "jst": 4}, scan_limit=2, root=tmp_path
    ) == active
