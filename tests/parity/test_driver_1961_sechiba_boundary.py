from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.driver.bundle import load_paper_1961_first_step_bundle, load_paper_1961_step_bundle
from jax_orchidee.driver.sechiba_boundary import build_intersurf_first_step_payload
from jax_orchidee.driver.stomate_boundary import (
    paper_1961_first_step_stomate_boundary,
    paper_case_vertical_grids_from_used_run_def,
)
from jax_orchidee.driver.trace import StaticTraceFields
from jax_orchidee.driver.stomate_boundary import paper_case_cwrr_vertical_soil_grid_from_used_run_def
from jax_orchidee.sechiba.diffuco import cwrr_diaglev_from_vertical_soil_params


CONFIG = ROOT / "configs" / "orchidee_man_250919.yaml"
USED_RUN_DEF = ROOT / "outputs" / "server_1961_trace_full_20260623" / "run" / "used_run.def"


def _used_run_def_scalar(name: str) -> float:
    for line in USED_RUN_DEF.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{name} "):
            return float(stripped.split("=", maxsplit=1)[1].strip())
    raise KeyError(name)


def test_intersurf_first_step_payload_maps_ready_bundle_fields():
    bundle = load_paper_1961_first_step_bundle(CONFIG, year=1961)
    payload = build_intersurf_first_step_payload(bundle)

    assert payload.kjpindex == 1
    assert payload.nbindex == 1
    assert np.array_equal(payload.kindex, np.asarray([1], dtype=np.int32))
    assert np.allclose(payload.lalo, [[21.0, 109.0]])
    assert np.allclose(payload.contfrac, [0.5625])

    assert np.allclose(payload.temp_air, [291.3238830566406])
    assert np.allclose(payload.pb, [984.228671875])
    assert np.allclose(payload.qair, [0.008104387670755386])
    assert np.allclose(payload.u, [-0.5608849019156963])
    assert np.allclose(payload.v, [-2.5034796811461097])
    assert np.allclose(payload.precip_rain, [0.0])
    assert np.allclose(payload.precip_snow, [0.0])
    assert np.allclose(payload.swdown, [362.3620910644531])
    assert np.allclose(payload.lwdown, [338.0463562011719])
    assert np.allclose(payload.zlev, [2.0])
    assert np.allclose(payload.zlevuv, [10.0])
    assert np.allclose(payload.ccanopy, [317.27])

    assert payload.veget_max.shape == (1, 14)
    assert np.allclose(payload.veget_max[0, 13], 1.0)
    assert payload.veget.shape == (1, 14)
    assert np.allclose(payload.veget[0, :13], 0.0)
    assert np.allclose(payload.veget[0, 13], 1.0)
    assert np.allclose(payload.soiltile, [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])


def test_intersurf_payload_accepts_later_step_bundle_forcing_with_local_static_fields():
    bundle = load_paper_1961_step_bundle(CONFIG, year=1961, tstep=1)
    payload = build_intersurf_first_step_payload(bundle)

    assert payload.kjpindex == 1
    assert payload.nbindex == 1
    assert np.allclose(payload.temp_air, [289.9084014892578])
    assert np.allclose(payload.pb, [987.6242447916667])
    assert np.allclose(payload.qair, [0.009654206223785877])
    assert np.allclose(payload.u, [-0.011365802192769222])
    assert np.allclose(payload.v, [-2.36955126165179])
    assert np.allclose(payload.precip_rain, [0.0])
    assert np.allclose(payload.swdown, [186.07451159425887])
    assert np.allclose(payload.lwdown, [357.4541371663411])
    assert np.allclose(payload.zlev, [2.0])
    assert np.allclose(payload.zlevuv, [10.0])
    assert "soilclass" not in payload.missing_fields
    assert "salinity" not in payload.missing_fields
    assert "tide_height" not in payload.missing_fields


def test_intersurf_payload_accepts_multiland_driver_step_with_source_static_fields():
    bundle = load_paper_1961_step_bundle(
        CONFIG,
        year=1961,
        tstep=1,
        domain_override={"west": 107.0, "east": 111.0, "south": 19.0, "north": 23.0},
    )
    payload = build_intersurf_first_step_payload(bundle)

    assert payload.kjpindex == 9
    assert payload.nbindex == 9
    np.testing.assert_array_equal(payload.kindex, np.arange(1, 10, dtype=np.int32))
    assert payload.temp_air.shape == (9,)
    assert payload.pb.shape == (9,)
    assert payload.qair.shape == (9,)
    assert payload.swdown.shape == (9,)
    assert payload.veget_max.shape == (9, 14)
    assert payload.veget.shape == (9, 14)
    assert payload.soiltile.shape == (9, 6)
    assert payload.ccanopy.shape == (9,)

    np.testing.assert_allclose(payload.temp_air[:5], [287.29587809, 286.56400553, 286.63167318, 289.87373861, 289.90840149])
    np.testing.assert_allclose(payload.swdown[:5], [142.67387492, 157.9875261, 180.45878512, 175.82709401, 186.07451159])
    assert payload.resolution.shape == (9, 2)
    assert payload.neighbours.shape == (9, 8)
    assert payload.soilclass.shape == (9, 12)
    assert payload.salinity.shape == (9,)
    assert payload.tide_height.shape == (9, 584)
    np.testing.assert_allclose(payload.resolution[4], [207815.27471986, 222634.1993844])
    np.testing.assert_array_equal(payload.neighbours[4], np.asarray([2, 3, 6, 9, 8, 7, 4, 1], dtype=np.int32))
    np.testing.assert_allclose(payload.salinity[:3], [33.21248317, 33.2296896, 31.98728275])
    assert "resolution" not in payload.missing_fields
    assert "neighbours" not in payload.missing_fields
    assert "soilclass" not in payload.missing_fields
    assert "salinity" not in payload.missing_fields
    assert "tide_height" not in payload.missing_fields
    assert "veget" not in payload.missing_fields
    assert "corners" not in payload.missing_fields
    assert "seglength" not in payload.missing_fields
    assert "soilclass_sub_index" not in payload.missing_fields
    assert payload.corners.shape == (9, 4, 2)
    assert payload.seglength.shape == (9, 4)
    assert payload.soilclass_sub_index.shape[0] == 9


def test_intersurf_payload_converts_precipitation_rate_to_step_amount():
    bundle = load_paper_1961_step_bundle(CONFIG, year=1961, tstep=48)
    payload = build_intersurf_first_step_payload(bundle, dt_sechiba=1800.0)

    # Fortran dim2_driver.f90 passes precip_rain/snow multiplied by dt to
    # intersurf_main_2d (lines 869-871, 1073-1075, and 1259-1261).
    assert np.allclose(payload.precip_rain, bundle.forcing.precip_rain.reshape(-1) * 1800.0)
    assert np.allclose(payload.precip_snow, bundle.forcing.precip_snow.reshape(-1) * 1800.0)
    assert np.allclose(payload.precip_rain, [0.0009906776199386513])


def test_intersurf_first_step_payload_keeps_only_unported_geometry_metadata_missing():
    bundle = load_paper_1961_first_step_bundle(CONFIG, year=1961)
    payload = build_intersurf_first_step_payload(bundle)

    assert np.allclose(payload.resolution, [[111106.370838091, 111111.0]])
    assert np.array_equal(payload.neighbours, np.full((1, 8), -1, dtype=np.int32))
    assert payload.soilclass is not None
    assert payload.salinity is not None
    assert payload.tide_height is not None
    assert "resolution" not in payload.missing_fields
    assert "neighbours" not in payload.missing_fields
    assert "soilclass" not in payload.missing_fields
    assert "salinity" not in payload.missing_fields
    assert "tide_height" not in payload.missing_fields
    assert "corners" not in payload.missing_fields
    assert "seglength" not in payload.missing_fields
    assert "veget" not in payload.missing_fields


def test_intersurf_first_step_payload_prefers_local_slowproc_soil_fields_over_restart_anchors():
    bundle = load_paper_1961_first_step_bundle(CONFIG, year=1961)
    payload = build_intersurf_first_step_payload(bundle)

    assert np.array_equal(payload.njsc, np.asarray([5], dtype=np.int32))
    assert np.allclose(payload.clay_frac, [0.1])
    assert np.allclose(payload.sand_frac, [0.06])
    assert np.allclose(payload.silt_frac, [0.84])
    assert np.allclose(payload.bulk_dens, [1.38594591617584])
    assert np.allclose(payload.soil_ph, [5.66892290115356])
    assert np.allclose(payload.poor_soils, [8.333333535119891e-4])


def test_intersurf_first_step_payload_accepts_explicit_static_trace_truth():
    bundle = load_paper_1961_first_step_bundle(CONFIG, year=1961)
    static_truth = StaticTraceFields(
        soilclass=np.asarray([[0.1, 0.9, 0.0]], dtype=np.float64),
        silt_frac=np.asarray([0.4], dtype=np.float64),
        salinity=np.asarray([12.5], dtype=np.float64),
        tide_height=np.asarray([[1.0, 2.0]], dtype=np.float64),
    )

    payload = build_intersurf_first_step_payload(bundle, static_trace_fields=static_truth)

    assert np.allclose(payload.soilclass, [[0.1, 0.9, 0.0]])
    assert np.allclose(payload.silt_frac, [0.4])
    assert np.allclose(payload.salinity, [12.5])
    assert np.allclose(payload.tide_height, [[1.0, 2.0]])
    assert "soilclass" not in payload.missing_fields
    assert "silt_frac" not in payload.missing_fields
    assert "salinity" not in payload.missing_fields
    assert "tide_height" not in payload.missing_fields


def test_paper_1961_driver_restart_sources_build_pre_step_ok_leak_boundary():
    bundle = load_paper_1961_first_step_bundle(CONFIG, year=1961)
    assembled = paper_1961_first_step_stomate_boundary(
        CONFIG,
        bundle=bundle,
        used_run_def_path=USED_RUN_DEF,
        root=ROOT,
    )

    assert assembled.river_routing is True
    assert assembled.nflow == 6
    assert assembled.payload.kjpindex == 1
    boundary = assembled.pre_step.boundary_inputs
    assert boundary.ok_leak_inputs["pref_soil_veg"].shape == (14,)
    assert boundary.ok_leak_inputs["pref_soil_veg"][13] == 3
    assert bool(boundary.ok_leak_inputs["is_peat"][13]) is True
    assert boundary.ok_leak_inputs["z_soil"].shape == (12,)
    assert boundary.ok_leak_inputs["nslm"] == 11
    assert boundary.ok_leak_inputs["ndeep"] == 32
    np.testing.assert_allclose(boundary.ok_leak_inputs["clay"], [0.1])
    np.testing.assert_allclose(boundary.ok_leak_inputs["bulk_dens"], [1.38594591617584])
    np.testing.assert_allclose(boundary.ok_leak_inputs["poor_soils"], [8.333333535119891e-4])
    np.testing.assert_allclose(boundary.output_inputs["contfrac"], [0.5625])
    assert "soilwater_31mm" not in boundary.ok_leak_inputs
    assert "fbact_litter" not in boundary.ok_leak_inputs


def test_paper_1961_stomate_boundary_vertical_grid_is_shared_used_run_def_truth():
    expected_diaglev = np.asarray(
        cwrr_diaglev_from_vertical_soil_params(
            depth_max_h=_used_run_def_scalar("DEPTH_MAX_H"),
            depth_max_t=_used_run_def_scalar("DEPTH_MAX_T"),
            depth_topthickness=_used_run_def_scalar("DEPTH_TOPTHICK"),
            depth_cstthickness=_used_run_def_scalar("DEPTH_CSTTHICK"),
            depth_geom=_used_run_def_scalar("DEPTH_GEOM"),
            ratio_geom_below=_used_run_def_scalar("RATIO_GEOM_BELOW"),
        )
    )

    diaglev, zz_coef_deep, zz_deep = paper_case_vertical_grids_from_used_run_def(USED_RUN_DEF)
    grid = paper_case_cwrr_vertical_soil_grid_from_used_run_def(USED_RUN_DEF)

    np.testing.assert_allclose(diaglev, expected_diaglev)
    np.testing.assert_allclose(zz_coef_deep, np.asarray(grid.zlt))
    np.testing.assert_allclose(zz_deep, np.asarray(grid.znt))
    assert diaglev.shape == (11,)
    assert zz_coef_deep.shape == (32,)
    assert zz_deep.shape == (32,)
