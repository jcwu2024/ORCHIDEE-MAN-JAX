from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.trace_schema import required_columns, trace_group, trace_group_names


def test_trace_schema_exposes_expected_groups_without_geometry_computation():
    names = trace_group_names()

    assert "domain_forcing_selected_point" in names
    assert "grid_geometry_after_grid_stuff" in names
    assert "soil_aggregation" in names
    assert "salinity_bbox" in names
    assert "tide_bbox" in names
    assert "intersurf_first_step_boundary" in names


def test_grid_geometry_trace_columns_require_exact_grid_stuff_outputs():
    columns = required_columns("grid_geometry_after_grid_stuff")

    assert "resolution_x" in columns
    assert "resolution_y" in columns
    assert "neighbour_1" in columns
    assert "neighbour_8" in columns
    assert "corner_1_lon" in columns
    assert "corner_4_lat" in columns
    assert "seglength_1" in columns
    assert "seglength_4" in columns


def test_soil_and_bbox_trace_columns_match_future_static_helper_inputs():
    soil_columns = required_columns("soil_aggregation")
    salinity_columns = required_columns("salinity_bbox")
    tide_columns = required_columns("tide_bbox")

    assert "sub_area" in soil_columns
    assert "source_i" in soil_columns
    assert "source_j" in soil_columns
    assert "soilclass_1" in soil_columns
    assert "soilclass_3" in soil_columns
    assert salinity_columns == tide_columns
    assert "lon_low" in salinity_columns
    assert "source_count" in salinity_columns
    assert "fallback_nearest" in salinity_columns
    assert "final_value" in salinity_columns


def test_trace_group_records_fortran_provenance_and_consumer():
    group = trace_group("intersurf_first_step_boundary")

    assert group.consumer == "jax_orchidee.driver.bundle"
    assert any("dim2_driver.f90" in item for item in group.fortran_provenance)
    assert any("intersurf.f90" in item for item in group.fortran_provenance)
