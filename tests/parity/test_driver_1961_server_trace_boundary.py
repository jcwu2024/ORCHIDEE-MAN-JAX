from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.bundle import load_paper_1961_first_step_bundle, load_paper_1961_step_bundle
from jax_orchidee.driver.sechiba_boundary import build_intersurf_first_step_payload
from jax_orchidee.driver.static import (
    paper_single_point_grid_geometry,
    read_paper_salinity_tide_static_fields,
    read_paper_usda_soil_static_fields,
)
from jax_orchidee.driver.trace import (
    read_fixed_format_bbox_trace,
    read_fixed_format_driver_forcing_trace,
    read_fixed_format_grid_trace,
    read_fixed_format_intersurf_trace,
    read_fixed_format_soil_trace,
    read_static_fields_from_fixed_format_trace_dir,
)


CONFIG = ROOT / "configs" / "orchidee_man_250919.yaml"
TRACE_DIR = ROOT / "outputs" / "server_1961_trace_full_20260623" / "traces"


def test_driver_first_forcing_domain_matches_server_trace_record():
    bundle = load_paper_1961_first_step_bundle(CONFIG, year=1961)
    trace = read_fixed_format_driver_forcing_trace(TRACE_DIR / "orchjax_driver_forcing_trace.txt")[0]

    assert bundle.domain.nbindex == 1
    assert np.array_equal(bundle.domain.kindex, np.asarray([1], dtype=np.int32))
    assert np.allclose(bundle.domain.lon.reshape(-1), [trace["nav_lon"]])
    assert np.allclose(bundle.domain.lat.reshape(-1), [trace["nav_lat"]])
    assert np.allclose(bundle.domain.lalo, [[trace["lalo_lat"], trace["lalo_lon"]]])
    assert np.allclose(bundle.domain.contfrac_land, [trace["contfrac"]])

    assert np.allclose(bundle.forcing.Tair, [[trace["Tair"]]])
    assert np.allclose(bundle.forcing.PSurf, [[trace["PSurf"]]])
    assert np.allclose(bundle.forcing.Qair, [[trace["Qair"]]])
    assert np.allclose(bundle.forcing.Wind_E, [[trace["Wind_E"]]])
    assert np.allclose(bundle.forcing.Wind_N, [[trace["Wind_N"]]])
    assert np.allclose(bundle.forcing.Rainf, [[trace["Rainf"]]])
    assert np.allclose(bundle.forcing.Snowf, [[trace["Snowf"]]])
    assert np.allclose(bundle.forcing.SWdown, [[trace["SWdown"]]])
    assert np.allclose(bundle.forcing.LWdown, [[trace["LWdown"]]])
    assert np.allclose(bundle.forcing.zlev, [[trace["Height_Lev1"]]])
    assert np.allclose(bundle.forcing.zlevuv, [[trace["Height_Levuv"]]])


def test_driver_boundary_first_intersurf_main_record_matches_server_trace():
    bundle = load_paper_1961_step_bundle(CONFIG, year=1961, tstep=0)
    payload = build_intersurf_first_step_payload(bundle, driver_z0_for_wind=999999.0)
    first_main = read_fixed_format_intersurf_trace(TRACE_DIR / "orchjax_intersurf_main_trace.txt")[0]

    assert first_main["call_name"] == "main"
    assert first_main["tstep_fortran"] == 1
    assert first_main["kindex"] == int(bundle.domain.kindex[0])
    assert np.allclose([first_main["lalo_lat"], first_main["lalo_lon"]], bundle.domain.lalo[0])
    assert np.allclose(first_main["contfrac"], bundle.domain.contfrac_land[0])
    assert np.allclose(first_main["ccanopy"], bundle.ccanopy[0])
    assert np.allclose(first_main["u"], payload.u[0])
    assert np.allclose(first_main["v"], payload.v[0])
    assert np.allclose(first_main["swdown"], payload.swdown[0])
    assert np.allclose(first_main["pb"], payload.pb[0])


def test_slowproc_salinity_and_tide_traces_close_bundle_missing_fields():
    static_truth = read_static_fields_from_fixed_format_trace_dir(TRACE_DIR)
    salinity_trace = read_fixed_format_bbox_trace(
        TRACE_DIR / "orchjax_slowproc_read_annual_trace.txt",
        group_name="salinity_bbox",
    )[0]
    tide_trace = read_fixed_format_bbox_trace(
        TRACE_DIR / "orchjax_slowproc_read_data_trace.txt",
        group_name="tide_bbox",
    )

    bundle = load_paper_1961_first_step_bundle(CONFIG, year=1961, fixed_format_trace_dir=TRACE_DIR)
    payload = build_intersurf_first_step_payload(bundle)

    assert np.allclose(static_truth.salinity, [salinity_trace["final_value"]])
    assert np.allclose(static_truth.tide_height[0, :3], [row["final_value"] for row in tide_trace[:3]])
    assert static_truth.tide_height.shape == (1, 584)

    assert "salinity" not in bundle.missing_static_fields
    assert "salinity_bbox" not in bundle.missing_static_fields
    assert "tide_height" not in bundle.missing_static_fields
    assert "tide_bbox" not in bundle.missing_static_fields
    assert "soilclass" not in bundle.missing_static_fields
    assert "soilclass_sub_index" not in bundle.missing_static_fields
    assert "soilclass_sub_area" not in bundle.missing_static_fields

    assert np.allclose(payload.salinity, static_truth.salinity)
    assert np.allclose(payload.tide_height, static_truth.tide_height)
    assert "salinity" not in payload.missing_fields
    assert "tide_height" not in payload.missing_fields
    assert "soilclass" not in payload.missing_fields
    assert "soilclass_sub_index" not in payload.missing_fields
    assert "soilclass_sub_area" not in payload.missing_fields


def test_local_salinity_and_tide_files_match_slowproc_bbox_trace_with_traced_geometry():
    bundle = load_paper_1961_first_step_bundle(CONFIG, year=1961, fixed_format_trace_dir=TRACE_DIR)
    salinity_trace = read_fixed_format_bbox_trace(
        TRACE_DIR / "orchjax_slowproc_read_annual_trace.txt",
        group_name="salinity_bbox",
    )[0]
    tide_trace = read_fixed_format_bbox_trace(
        TRACE_DIR / "orchjax_slowproc_read_data_trace.txt",
        group_name="tide_bbox",
    )

    salinity, tide_height, counts = read_paper_salinity_tide_static_fields(
        CONFIG,
        lalo=bundle.domain.lalo,
        resolution_m=bundle.domain.resolution,
    )

    assert counts["salinity_source_count"][0] == salinity_trace["source_count"]
    assert counts["tide_source_count"][0] == tide_trace[0]["source_count"]
    np.testing.assert_allclose(salinity, [salinity_trace["final_value"]])
    np.testing.assert_allclose(tide_height[0, :3], [row["final_value"] for row in tide_trace[:3]])
    assert tide_height.shape == (1, 584)


def test_local_single_point_grid_geometry_matches_grid_scatter_trace():
    grid_trace = read_fixed_format_grid_trace(TRACE_DIR / "orchjax_grid_trace.txt")[0]
    bundle = load_paper_1961_first_step_bundle(CONFIG, year=1961)

    resolution, neighbours, area = paper_single_point_grid_geometry(bundle.domain.lalo)

    np.testing.assert_allclose(resolution, [[grid_trace["resolution_x"], grid_trace["resolution_y"]]])
    np.testing.assert_array_equal(
        neighbours,
        np.asarray([[grid_trace[f"neighbour_{idx}"] for idx in range(1, 9)]], dtype=np.int32),
    )
    np.testing.assert_allclose(area, [grid_trace["area"]])
    np.testing.assert_allclose(bundle.domain.resolution, resolution)
    np.testing.assert_array_equal(bundle.domain.neighbours, neighbours)
    np.testing.assert_allclose(bundle.domain.area, area)


def test_grid_trace_closes_full_geometry_payload_fields():
    grid_trace = read_fixed_format_grid_trace(TRACE_DIR / "orchjax_grid_trace.txt")[0]
    bundle = load_paper_1961_first_step_bundle(CONFIG, year=1961, fixed_format_trace_dir=TRACE_DIR)
    payload = build_intersurf_first_step_payload(bundle)

    expected_resolution = [[grid_trace["resolution_x"], grid_trace["resolution_y"]]]
    expected_neighbours = [[grid_trace[f"neighbour_{idx}"] for idx in range(1, 9)]]

    assert np.allclose(bundle.domain.resolution, expected_resolution)
    assert np.array_equal(bundle.domain.neighbours, np.asarray(expected_neighbours, dtype=np.int32))
    assert np.allclose(bundle.domain.area, [grid_trace["area"]])
    assert "resolution" not in bundle.missing_geometry_fields
    assert "neighbours" not in bundle.missing_geometry_fields
    assert "corners" not in bundle.missing_geometry_fields
    assert "seglength" not in bundle.missing_geometry_fields

    assert np.allclose(payload.resolution, expected_resolution)
    assert np.array_equal(payload.neighbours, np.asarray(expected_neighbours, dtype=np.int32))
    assert np.allclose(payload.corners[0, 0], [grid_trace["corner_1_lon"], grid_trace["corner_1_lat"]])
    assert np.allclose(payload.seglength[0], [grid_trace[f"seglength_{idx}"] for idx in range(1, 5)])
    assert "resolution" not in payload.missing_fields
    assert "neighbours" not in payload.missing_fields
    assert "corners" not in payload.missing_fields
    assert "seglength" not in payload.missing_fields


def test_slowproc_soilt_trace_closes_soilclass_njsc_and_texture_scalars():
    soil_trace = read_fixed_format_soil_trace(TRACE_DIR / "orchjax_slowproc_soilt_trace.txt")[0]
    bundle = load_paper_1961_first_step_bundle(CONFIG, year=1961, fixed_format_trace_dir=TRACE_DIR)
    payload = build_intersurf_first_step_payload(bundle)

    expected_soilclass = np.zeros((1, 12), dtype=np.float64)
    expected_soilclass[0, int(round(float(soil_trace["soiltext"]))) - 1] = 1.0

    assert soil_trace["soil_classif"] == "usda"
    assert np.allclose(bundle.static_trace_fields.soilclass, expected_soilclass)
    assert np.array_equal(bundle.static_trace_fields.njsc, np.asarray([5], dtype=np.int32))
    assert np.allclose(bundle.static_trace_fields.clay_frac, [soil_trace["clay_frac"]])
    assert np.allclose(bundle.static_trace_fields.sand_frac, [soil_trace["sand_frac"]])
    assert np.allclose(bundle.static_trace_fields.silt_frac, [soil_trace["silt_frac"]])
    assert np.allclose(bundle.static_trace_fields.bulk_dens, [soil_trace["bulk_dens"]])
    assert np.allclose(bundle.static_trace_fields.soil_ph, [soil_trace["soil_ph"]])
    assert np.allclose(bundle.static_trace_fields.poor_soils, [soil_trace["poor_soils"]])

    for closed in ("soilclass", "njsc", "clay_frac", "sand_frac", "silt_frac", "bulk_dens", "soil_ph", "poor_soils"):
        assert closed not in bundle.missing_static_fields
        assert closed not in payload.missing_fields
    assert "soilclass_sub_index" not in bundle.missing_static_fields
    assert "soilclass_sub_area" not in bundle.missing_static_fields

    assert np.allclose(payload.soilclass, expected_soilclass)
    np.testing.assert_array_equal(payload.soilclass_sub_index[0, 0], [soil_trace["source_i"], soil_trace["source_j"]])
    np.testing.assert_allclose(payload.soilclass_sub_area[0, 0], soil_trace["sub_area"])
    assert np.array_equal(payload.njsc, np.asarray([5], dtype=np.int32))
    assert np.allclose(payload.clay_frac, [soil_trace["clay_frac"]])
    assert np.allclose(payload.sand_frac, [soil_trace["sand_frac"]])
    assert np.allclose(payload.silt_frac, [soil_trace["silt_frac"]])


def test_local_soilclass_file_matches_slowproc_soilt_trace():
    soil_trace = read_fixed_format_soil_trace(TRACE_DIR / "orchjax_slowproc_soilt_trace.txt")[0]
    bundle = load_paper_1961_first_step_bundle(CONFIG, year=1961)

    fields, overlap = read_paper_usda_soil_static_fields(
        CONFIG,
        lalo=bundle.domain.lalo,
        resolution_m=bundle.domain.resolution,
    )

    assert np.array_equal(overlap["soilclass_sub_index"][0, 0], [soil_trace["source_i"], soil_trace["source_j"]])
    np.testing.assert_allclose(overlap["soilclass_sub_area"][0, 0], soil_trace["sub_area"])
    expected_soilclass = np.zeros((1, 12), dtype=np.float64)
    expected_soilclass[0, int(round(float(soil_trace["soiltext"]))) - 1] = 1.0
    np.testing.assert_allclose(fields["soilclass"], expected_soilclass)
    np.testing.assert_array_equal(fields["njsc"], np.asarray([5], dtype=np.int32))
    np.testing.assert_allclose(fields["clay_frac"], [soil_trace["clay_frac"]])
    np.testing.assert_allclose(fields["sand_frac"], [soil_trace["sand_frac"]])
    np.testing.assert_allclose(fields["silt_frac"], [soil_trace["silt_frac"]])
    np.testing.assert_allclose(fields["bulk_dens"], [soil_trace["bulk_dens"]])
    np.testing.assert_allclose(fields["soil_ph"], [soil_trace["soil_ph"]])
    np.testing.assert_allclose(fields["poor_soils"], [soil_trace["poor_soils"]])
