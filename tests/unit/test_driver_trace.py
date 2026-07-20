from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.domain import read_domain_grid
from jax_orchidee.driver.trace import (
    apply_grid_geometry_trace_to_domain,
    coerce_trace_row_values,
    read_fixed_format_bbox_trace,
    read_fixed_format_driver_forcing_trace,
    read_fixed_format_grid_trace,
    read_fixed_format_intersurf_trace,
    read_fixed_format_soil_trace,
    read_trace_csv,
    static_soil_fields_from_fixed_format_trace,
    static_fields_from_trace,
    validate_trace_rows,
)
from jax_orchidee.driver.trace_schema import required_columns


CONFIG = ROOT / "configs" / "orchidee_man_250919.yaml"


def _complete_row(group: str, **overrides):
    row = {column: 1 for column in required_columns(group)}
    row.update(overrides)
    return row


def test_validate_trace_rows_reports_missing_columns_without_throwing():
    result = validate_trace_rows("grid_geometry_after_grid_stuff", [{"ik": 1, "kindex": 1}])

    assert not result.is_valid
    assert result.group_name == "grid_geometry_after_grid_stuff"
    assert result.row_count == 1
    assert "resolution_x" in result.missing_columns
    assert "neighbour_8" in result.missing_columns


def test_validate_trace_rows_marks_unknown_group():
    result = validate_trace_rows("not_a_group", [{"anything": 1}])

    assert result.unknown_group is True
    assert result.is_valid is False
    assert result.present_columns == ("anything",)


def test_coerce_trace_row_values_preserves_text_and_converts_numbers():
    row = coerce_trace_row_values({"field_name": "salinity", "ik": "1", "value": "1.25", "blank": ""})

    assert row["field_name"] == "salinity"
    assert row["ik"] == 1
    assert row["value"] == 1.25
    assert row["blank"] == ""


def test_read_trace_csv_validates_synthetic_file(tmp_path):
    row = _complete_row(
        "domain_forcing_selected_point",
        year=1961,
        tstep=0,
        nav_lon=109.0,
        nav_lat=21.0,
    )
    csv_path = tmp_path / "trace.csv"
    columns = required_columns("domain_forcing_selected_point")
    csv_path.write_text(",".join(columns) + "\n" + ",".join(str(row[col]) for col in columns) + "\n", encoding="utf-8")

    table = read_trace_csv(csv_path, "domain_forcing_selected_point")

    assert table.validation.is_valid
    assert table.rows[0]["year"] == 1961
    assert table.rows[0]["nav_lon"] == 109


def test_apply_grid_geometry_trace_to_domain_uses_explicit_values_only():
    domain = read_domain_grid(CONFIG, year=1961)
    row = _complete_row(
        "grid_geometry_after_grid_stuff",
        ik=1,
        kindex=1,
        area=123.0,
        resolution_x=111.0,
        resolution_y=222.0,
        **{f"neighbour_{idx}": -idx for idx in range(1, 9)},
    )

    updated = apply_grid_geometry_trace_to_domain(domain, [row])

    assert np.allclose(updated.resolution, [[111.0, 222.0]])
    assert np.array_equal(updated.neighbours, np.asarray([[-1, -2, -3, -4, -5, -6, -7, -8]], dtype=np.int32))
    assert np.allclose(updated.area, [123.0])
    assert updated.corners.shape == (1, 4, 2)
    assert updated.seglength.shape == (1, 4)


def test_static_fields_from_trace_packages_only_supplied_truth():
    soil_row = _complete_row(
        "soil_aggregation",
        ik=1,
        fopt=1,
        source_i=1,
        source_j=1,
        sub_area=10.0,
        soilclass_1=0.25,
        soilclass_2=0.75,
        soilclass_3=0.0,
        njsc=2,
        clay_frac=0.2,
        sand_frac=0.4,
        bulk_dens=1650.0,
        soil_ph=7.0,
        poor_soils=0.0,
    )
    salinity_row = _complete_row("salinity_bbox", field_name="salinity", ik=1, time_index=1, final_value=15.0)
    tide_row_a = _complete_row("tide_bbox", field_name="tide", ik=1, time_index=1, final_value=1.5)
    tide_row_b = _complete_row("tide_bbox", field_name="tide", ik=1, time_index=2, final_value=2.5)

    fields = static_fields_from_trace(soil_rows=[soil_row], salinity_rows=[salinity_row], tide_rows=[tide_row_b, tide_row_a])

    assert fields.blocked_fields == ()
    assert np.allclose(fields.soilclass, [[0.25, 0.75, 0.0]])
    assert np.array_equal(fields.njsc, np.asarray([2], dtype=np.int32))
    assert np.allclose(fields.salinity, [15.0])
    assert np.allclose(fields.tide_height, [[1.5, 2.5]])


def test_static_fields_from_trace_reports_absent_truth_as_blocked():
    fields = static_fields_from_trace()

    assert "soilclass" in fields.blocked_fields
    assert "salinity" in fields.blocked_fields
    assert "tide_height" in fields.blocked_fields
    assert fields.soilclass is None
    assert fields.salinity is None
    assert fields.tide_height is None


def test_fixed_format_parsers_read_minimal_server_layouts(tmp_path):
    driver_path = tmp_path / "driver.txt"
    driver_path.write_text(
        "first_forcing 1 1 1 1 109.0 21.0 0.5625 111106.0 111111.0 "
        "291.0 98422.0 0.008 -0.8 -3.8 0.0 0.0 362.0 338.0 2.0 10.0\n",
        encoding="utf-8",
    )
    assert read_fixed_format_driver_forcing_trace(driver_path)[0]["Wind_E"] == -3.8

    intersurf_path = tmp_path / "main.txt"
    intersurf_path.write_text(
        "main 1 1 1 109.0 21.0 0.5625 289.0 987.0 0.009 0.07 -4.1 0.0 0.0 119.0 359.0 317.27\n",
        encoding="utf-8",
    )
    assert read_fixed_format_intersurf_trace(intersurf_path)[0]["ccanopy"] == 317.27

    grid_path = tmp_path / "grid.txt"
    grid_path.write_text(
        "grid_scatter 1 123.0 1e20 111.0 222.0 -1 -1 -1 -1 -1 -1 -1 -1 "
        "10.0 20.0 30.0 40.0 108.0 21.5 109.5 21.5 109.5 20.5 108.0 20.5\n",
        encoding="utf-8",
    )
    grid_row = read_fixed_format_grid_trace(grid_path)[0]
    assert grid_row["resolution_y"] == 222.0
    assert grid_row["contfrac"] == 1e20
    assert grid_row["corner_4_lat"] == 20.5

    salinity_path = tmp_path / "salinity.txt"
    salinity_path.write_text("salinity 1 108.0 109.0 20.0 21.0 4 F 0 33.0\n", encoding="utf-8")
    assert read_fixed_format_bbox_trace(salinity_path, group_name="salinity_bbox")[0]["final_value"] == 33.0


def test_fixed_format_soil_trace_reconstructs_usda_soilclass_and_njsc(tmp_path):
    soil_path = tmp_path / "soil.txt"
    soil_path.write_text(
        "soil_overlap 1 1 usda 290 69 3075829029.14936 5.0 "
        "1.38594591617584 5.66892290115356 0.0008333333535119891 "
        "0.0 0.0 0.0 0.1 0.06 0.84 1.38594591617584 5.66892290115356 0.0008333333535119890\n",
        encoding="utf-8",
    )

    rows = read_fixed_format_soil_trace(soil_path)
    fields = static_soil_fields_from_fixed_format_trace(rows)

    expected = np.zeros((1, 12), dtype=np.float64)
    expected[0, 4] = 1.0
    assert np.allclose(fields.soilclass, expected)
    assert np.array_equal(fields.njsc, np.asarray([5], dtype=np.int32))
    assert np.allclose(fields.clay_frac, [0.1])
    assert np.allclose(fields.sand_frac, [0.06])
    assert np.allclose(fields.silt_frac, [0.84])
    assert np.allclose(fields.bulk_dens, [1.38594591617584])
