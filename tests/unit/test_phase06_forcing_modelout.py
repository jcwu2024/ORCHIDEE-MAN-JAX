"""Portable original-source numerical interfaces; no model step/day/year.

External identities are separate from the always-running numeric regressions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
import pytest
import xarray as xr

from jax_orchidee.driver import static
from jax_orchidee.parameters.pft_catalog import build_pft_run_layout, load_pft_catalog
from jax_orchidee.sechiba import diffuco
from jax_orchidee.stomate import modelout
from scripts.dev.oracle_phase06_forcing_modelout import actual_forcing

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = json.loads((ROOT / "tests/fixtures/phase06_forcing_modelout.json").read_text(encoding="utf-8"))
STATIC = json.loads((ROOT / "tests/fixtures/phase06_salinity_tide.json").read_text(encoding="utf-8"))
CONSUMER = json.loads((ROOT / "tests/fixtures/phase06_forcing_consumer.json").read_text(encoding="utf-8"))
NEAREST = json.loads((ROOT / "tests/fixtures/phase06_static_nearest.json").read_text(encoding="utf-8"))
FIELDS = {"tair": "temp_air", "pb": "pb", "qair": "qair", "u": "u", "v": "v", "rainf": "precip_rain",
          "snowf": "precip_snow", "swdown": "swdown", "lwdown": "lwdown", "zlev": "zlev", "zlevuv": "zlevuv"}


@pytest.mark.parametrize("case", REFERENCE["forcing"]["cases"], ids=lambda case: case["id"])
def test_real_forcing_landpoint_component_matches_original_fortran_all_fields(case):
    for step, expected in case["expected"].items():
        actual = actual_forcing(REFERENCE, case, int(step) - 1)
        for name, attr in FIELDS.items():
            np.testing.assert_allclose(np.asarray(getattr(actual, attr)).reshape(-1), expected[name]["values"],
                                       rtol=1e-12, atol=1e-14, err_msg=f"{case['id']}:{step}:{name}")
        if case["watchout"]:
            for name in ("swnet", "eair", "petAcoef", "peqAcoef", "petBcoef", "peqBcoef", "cdrag", "ccanopy"):
                np.testing.assert_allclose(np.asarray(getattr(actual, name)).reshape(-1), expected[name.lower()]["values"],
                                           rtol=1e-12, atol=1e-14)


def _file_to_bundle(tmp_path, case, step):
    from jax_orchidee.driver.bundle import DriverStepBundle
    from jax_orchidee.driver.domain import DomainGrid, _forcing_model_step_from_cached_point, _read_forcing_land_cache
    from jax_orchidee.driver.init import ImposedVegetationState
    from jax_orchidee.driver.trace import StaticTraceFields

    inputs = REFERENCE["forcing"]["inputs"]
    variables = {name: (("time_counter", "y", "x"), np.asarray(value).reshape(24, 2, 2)) for name, value in inputs.items()}
    if not case["watchout"]:
        variables.pop("levels")
    lon = np.array([[-30., 30.], [0., 60.]])
    lat = np.array([[-20., 30.], [10., 60.]])
    variables.update(nav_lon=(("y", "x"), lon.T), nav_lat=(("y", "x"), lat.T),
                     Areas=(("y", "x"), np.ones((2, 2))), contfrac=(("y", "x"), np.ones((2, 2))),
                     resolutionX=(("y", "x"), np.full((2, 2), 1000.)),
                     resolutionY=(("y", "x"), np.full((2, 2), 2000.)),
                     Height_Lev1=((), 2.), Height_Levuv=((), 10.))
    path = tmp_path / "forcing.nc"
    ds = xr.Dataset(variables, coords={"time_counter": np.arange(24) * case["dt_force"]})
    ds.time_counter.attrs["units"] = "seconds since 2000-01-01 00:00:00"
    ds.to_netcdf(path)
    config = tmp_path / "case.json"
    config.write_text(json.dumps({"drivers": {"atmospheric_forcing": {"file_pattern": str(path)}}}), encoding="utf-8")
    land = ((0, 0), (1, 0), (0, 1), (1, 1))
    cache = _read_forcing_land_cache(str(config), 1961, land)
    forcing = _forcing_model_step_from_cached_point(
        cache, model_tstep=step - 1, split=case["split"], nb_spread=case["spread"],
        dt_force=case["dt_force"], netrad_cons=case["netrad_cons"], lon=lon, lat=lat, land_indices_key=land,
    )
    domain = DomainGrid(iim=2, jjm=2, nbindex=4, kindex=np.arange(1, 5), lon=lon, lat=lat,
                        lalo=np.stack((lat.ravel(order="F"), lon.ravel(order="F")), axis=1), contfrac=np.ones((2, 2)),
                        contfrac_land=np.ones(4), area=np.ones(4), forcing_indices_zero_based=np.asarray(land))
    veget = np.zeros((4, 14))
    veget[:, 13] = 1.
    vegetation = ImposedVegetationState(veget_max=veget, frac_nobio=np.zeros((4, 1)), totfrac_nobio=np.zeros(4),
                                       soiltile=np.tile([1., 0., 0., 0., 0., 0.], (4, 1)), veget=veget)
    # Real bundle/payload types; run scalars and water table are unused by
    # this adapter and explicitly outside this finite component witness.
    return DriverStepBundle(year=1961, tstep=step - 1, domain=domain, forcing=forcing, run_scalars=None,
                            vegetation=vegetation, co2_ppm=411.25, ccanopy=np.full(4, 411.25), water_table=None,
                            restart_anchors=None, static_trace_fields=StaticTraceFields(),
                            missing_geometry_fields=(), missing_static_fields=())


@pytest.mark.parametrize("case", CONSUMER["cases"], ids=lambda case: f"{case['case_id']}-{case['step']}-z0{case['roughness']}")
@pytest.mark.parametrize("jit", [False, True], ids=["eager", "narrow-jit"])
def test_real_netcdf_cache_payload_units_feed_finite_diffuco_consumers(tmp_path, case, jit):
    from jax_orchidee.driver.sechiba_boundary import build_intersurf_first_step_payload

    source_case = next(item for item in REFERENCE["forcing"]["cases"] if item["id"] == case["case_id"])
    bundle = _file_to_bundle(tmp_path, source_case, case["step"])
    payload = build_intersurf_first_step_payload(bundle, driver_z0_for_wind=case["roughness"], dt_sechiba=case["dt_sechiba"])
    for name in ("pb", "u", "v", "precip_rain", "precip_snow"):
        np.testing.assert_allclose(getattr(payload, name), case["expected"][name], rtol=1e-12, atol=1e-14)
    np.testing.assert_array_equal(payload.ccanopy, np.full(4, 411.25))
    qsat = jax.jit(diffuco.diffuco_qsatt_explicit) if jit else diffuco.diffuco_qsatt_explicit
    # Surface T equals supplied air T in this declared component tuple; no
    # claim of a surface-state predecessor or energy solve is made.
    np.testing.assert_allclose(qsat(temp_sol=payload.temp_air, pb=payload.pb), case["expected"]["qsatt"], rtol=1e-12, atol=1e-14)
    if case["watchout"]:
        drag = jax.jit(diffuco.diffuco_raerod_explicit) if jit else diffuco.diffuco_raerod_explicit
        np.testing.assert_allclose(drag(u=payload.u, v=payload.v, q_cdrag=payload.cdrag), case["expected"]["raero"], rtol=1e-12, atol=1e-14)


def _state_records():
    return [modelout.stomate_lpj_history_fields_from_state(**{name: np.asarray(value) for name, value in record.items()})
            for record in REFERENCE["modelout"]["records"]]


@pytest.mark.parametrize("record_index", range(3))
def test_real_output_state_mapper_matches_original_send_on_all_points_pfts(record_index):
    fields = _state_records()[record_index]
    for name in (*modelout.MODEL_OUTPUT_FIELD_NAMES, "FRUIT_M", "RESERVE_M"):
        source = "NPP_STOMATE" if name == "NPP" else name
        np.testing.assert_array_equal(fields[name], REFERENCE["modelout"]["expected_sends"][source][record_index])


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "narrow-jit"])
def test_finite_record_average_history_selection_and_original_paper_units(jit):
    records = _state_records()

    def aggregate(*values):
        daily = [dict(zip(modelout.MODEL_OUTPUT_FIELD_NAMES, record, strict=True)) for record in values]
        history = modelout.annual_history_mean_fields_from_daily_modelout(daily)
        selected = modelout.select_history_point_fields(history)
        return history, selected, modelout.compute_modelout_from_fields(selected)

    function = jax.jit(aggregate) if jit else aggregate
    history, selected, result = function(*[tuple(record[name] for name in modelout.MODEL_OUTPUT_FIELD_NAMES) for record in records])
    for name in modelout.MODEL_OUTPUT_FIELD_NAMES:
        source = "NPP_STOMATE" if name == "NPP" else name
        original = np.asarray(REFERENCE["modelout"]["expected_sends"][source])
        np.testing.assert_array_equal(history[name], original.mean(axis=0).T[None, :, None, :])
        np.testing.assert_array_equal(selected[name], original.mean(axis=0)[0, 13])
    for name, expected in REFERENCE["modelout"]["expected_finite_mean_modelout_point0"].items():
        np.testing.assert_allclose(getattr(result, name), expected, rtol=1e-12, atol=1e-14)
    # This is an average of three output records, never a natural-year sum.
    assert float(result.GPP_model) != float(np.sum(np.asarray(REFERENCE["modelout"]["expected_sends"]["GPP"])[:, 0, 13]))


def test_history_pft_identity_point_selection_and_age_route_keep_source_phase():
    records = _state_records()
    history = modelout.annual_history_mean_fields_from_daily_modelout(records)
    catalog = load_pft_catalog(ROOT / "configs/pft_catalogs/orchidee_man_paper_250919.json")
    legacy = build_pft_run_layout(catalog, layout_id="paper_250919_legacy14", fractions=[0.] * 13 + [1.])
    compact_catalog = replace(catalog, layouts={**catalog.layouts, "phase06_bare_mangrove": ("bare_soil", "mangrove_pft14")})
    compact = build_pft_run_layout(compact_catalog, layout_id="phase06_bare_mangrove", fractions=[0., 1.])
    point = modelout.select_history_point_fields(history, lon_index=1)
    selected, identity = modelout.select_history_point_fields_for_pft_id(
        {name: value[:, [0, 13], :, :] for name, value in history.items()}, layout=compact,
        pft_id="mangrove_pft14", lon_index=1,
    )
    assert identity.fortran_pft_id == 14 and identity.pft_index == 1
    assert modelout.modelout_pft_selection(legacy, "mangrove_pft14").pft_index == 13
    for name in modelout.MODEL_OUTPUT_FIELD_NAMES:
        source = "NPP_STOMATE" if name == "NPP" else name
        expected = np.asarray(REFERENCE["modelout"]["expected_sends"][source]).mean(axis=0)[1, 13]
        np.testing.assert_array_equal(selected[name], expected)
        np.testing.assert_array_equal(point[name], expected)
    for age, year in ((1, 1961), (2, 1962), (11, 1971)):
        assert modelout.history_year_for_age(age) == year
        assert modelout.history_filename_for_age(age) == f"stomate_history_{year}.nc"


def test_actual_modelout_driver_packet_keeps_computed_values_and_completed_boundary():
    from jax_orchidee.driver.orchestration import _driver_modelout_from_outputs
    from jax_orchidee.stomate.integration import ExplicitStomateLpjOutputsResult

    fields = _state_records()[1]
    computed = modelout.compute_modelout_from_fields(fields)
    # Upstream scientific diagnostics are deliberately not executed. This is
    # the actual typed output/result and driver packet transport boundary.
    packet = _driver_modelout_from_outputs(
        ExplicitStomateLpjOutputsResult(None, fields, computed, ()), day_index=3, start_tstep=96, steps_per_stomate=48,
    )
    assert (packet.day_index, packet.start_tstep, packet.end_tstep) == (3, 96, 143)
    for name in modelout.MODEL_OUTPUT_FIELD_NAMES:
        np.testing.assert_array_equal(packet.modelout_fields[name], fields[name])
    for name in computed._fields:
        np.testing.assert_array_equal(getattr(packet.modelout, name), getattr(computed, name))
    assert _driver_modelout_from_outputs(
        ExplicitStomateLpjOutputsResult(None, fields, computed, ("missing_upstream",)),
        day_index=3, start_tstep=96, steps_per_stomate=48,
    ) is None


def _static_file(tmp_path, fixture, name, *, values=None):
    path = tmp_path / f"{name}.nc"
    raw = np.asarray(fixture["raw"] if values is None else values, dtype=np.float64)
    xr.Dataset({name: (("time_counter", "lat", "lon"), raw)},
               coords={"time_counter": np.arange(raw.shape[0]), "lat": fixture["lat"], "lon": fixture["lon"]}).to_netcdf(path)
    width = fixture["bbox_width_degrees"] * np.pi / 180 * 6378000.
    return path, np.asarray(fixture["lalo"]), np.full((2, 2), width)


@pytest.mark.parametrize("static_fixture", [STATIC, NEAREST], ids=["inside-bbox", "nearest-fallback"])
def test_actual_salinity_file_preprocess_bbox_reaches_numeric_control(tmp_path, static_fixture):
    f = static_fixture["salinity"]
    path, lalo, resolution = _static_file(tmp_path, f, "salinity")
    values, counts = static.read_slowproc_annual_bbox_field(path, variable="salinity", lalo=lalo, resolution_m=resolution)
    np.testing.assert_array_equal(values, f["expected"])
    np.testing.assert_array_equal(counts, f["counts"])
    assembled = diffuco.assemble_pft14_control_salinity_first_step(salinity=values, control_salinity_min=static_fixture["controls"]["min"])
    np.testing.assert_allclose(assembled.payload["control_salinity"], static_fixture["controls"]["control_salinity"], rtol=1e-12, atol=1e-14)


def _tide_and_control(tmp_path, static_fixture):
    f, c = static_fixture["tide"], static_fixture["controls"]
    path, lalo, resolution = _static_file(tmp_path, f, "tide")
    tide, counts = static.read_slowproc_time_bbox_field(path, variable="tide", lalo=lalo, resolution_m=resolution)
    np.testing.assert_array_equal(tide, f["expected"])
    np.testing.assert_array_equal(counts, f["counts"])
    assert tide.shape == (2, static_fixture["itimetide"])
    rprof = np.full((2, 14), .5)
    rprof[:, 13] = c["rprof_pft14"]
    bio = np.ones((2, 14, 12, 1))
    bio[:, 13, 8, 0], bio[:, 13, 10, 0] = [20., 150.], [10., 100.]
    bio[:, 13, 9, 0], bio[:, 13, 11, 0] = [5., 30.], [3., 20.]
    bio[:, 1, 9, 0], bio[:, 1, 11, 0] = 0., 0.
    inputs = diffuco.ControlInundateInputs(z_soil=diffuco.z_soil_from_diaglev(np.asarray(c["diaglev"])),
                                         rprof=rprof, tide_height=tide, biomass=bio)
    kwargs = {name: c[name] for name in ("agb_agr_ven_all_st", "agb_agr_ven_all_pn", "h_agr_max_st", "h_agr_max_pn")}
    return inputs, {"control_inudate_min": c["min"], **kwargs}


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "narrow-jit"])
@pytest.mark.parametrize("static_fixture", [STATIC, NEAREST], ids=["inside-bbox", "nearest-fallback"])
def test_full_itimetide_file_operand_reaches_actual_inundation_and_height_component(tmp_path, jit, static_fixture):
    inputs, kwargs = _tide_and_control(tmp_path, static_fixture)
    def function(z_soil, rprof, tide_height, biomass):
        numeric_inputs = diffuco.ControlInundateInputs(z_soil=z_soil, rprof=rprof, tide_height=tide_height, biomass=biomass)
        return diffuco.mangrove_control_inundation(numeric_inputs, **kwargs)

    result = (jax.jit(function) if jit else function)(inputs.z_soil, inputs.rprof, inputs.tide_height, inputs.biomass)
    c = static_fixture["controls"]
    for name in ("control_inudate",):
        np.testing.assert_allclose(getattr(result, name), c[name], rtol=1e-12, atol=1e-14)
    for name in ("agb_st", "agb_pn", "h_st", "h_pn"):
        attr = {"agb_st": "agb_agr_st", "agb_pn": "agb_agr_pn", "h_st": "h_agr_st", "h_pn": "h_agr_pn"}[name]
        np.testing.assert_allclose(getattr(result.agr, attr), c[name], rtol=1e-12, atol=1e-14)
    for name, expected in c["samples"].items():
        actual = getattr(result.ventilation, name) if name == "frac_root_ventilate" else getattr(result, name)
        np.testing.assert_allclose(np.asarray(actual)[:, c["sample_indices"]], expected, rtol=1e-12, atol=1e-14)


def test_tide_file_mask_uses_first_slot_before_bbox_mean(tmp_path):
    f, c = STATIC["tide"], STATIC["tide_mask"]
    raw = np.ones_like(np.asarray(f["raw"]))
    raw[0, 0], raw[1, 0] = [2., 4., 8.], [-9100., 40., 80.]
    path, lalo, resolution = _static_file(tmp_path, {**f, **c}, "tide", values=raw)
    values, counts = static.read_slowproc_time_bbox_field(path, variable="tide", lalo=lalo, resolution_m=resolution)
    np.testing.assert_array_equal(values[:, 0], c["first"])
    np.testing.assert_array_equal(values[:, 1], c["second"])
    np.testing.assert_array_equal(counts, c["counts"])


@pytest.mark.external_data
def test_external_source_and_saved_build_identities_are_independent():
    sources = [*REFERENCE["forcing"]["source"], *REFERENCE["modelout"]["source"],
               *STATIC["source"], *CONSUMER["source"], *NEAREST["source"]]
    sources += [case["reference"] for case in REFERENCE["forcing"]["cases"]]
    retained = ROOT / "outputs/reviews/main-physics-20261120/physics-branch-closeout/phase06-remaining-arms-and-joins/forcing-modelout/reference"
    builds = [(build, retained) for build in REFERENCE["forcing"]["new_builds"]]
    builds += [(REFERENCE["modelout"]["build"], retained), (CONSUMER["build"], retained)]
    builds += [(build, retained) for build in STATIC["builds"]]
    builds += [(build, retained / "nearest") for build in NEAREST["builds"]]
    source_paths = {Path(item["path"]).name: item["path"] for item in sources}
    for build, directory in builds:
        assert build["compiler_version"]
        for unit in build["compile_units"]:
            path = ROOT / source_paths[unit["name"]] if unit["name"] in source_paths else directory / unit["name"]
            sources.append({"path": path.relative_to(ROOT).as_posix(), "sha256": unit["sha256"]})
    absent = [item["path"] for item in sources if not (ROOT / item["path"]).is_file()]
    if absent:
        pytest.skip("External source/reference identity only: " + ", ".join(absent))
    for item in sources:
        assert hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() == item["sha256"]
