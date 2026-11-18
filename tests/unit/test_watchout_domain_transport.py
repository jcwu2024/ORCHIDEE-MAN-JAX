"""Real NetCDF -> domain/cache/substep WATCHOUT transport regressions.

Expected payloads are saved original-Fortran observations, independently
extracted into the fixture; this module does not call the reader under test
to construct its expected values. No model transition is executed.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import MISSING, fields
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from jax_orchidee.driver.domain import (
    ForcingStep,
    _read_forcing_land_cache,
    read_domain_grid,
    read_forcing_first_step,
    read_forcing_model_step_cached,
    read_forcing_step,
    read_forcing_step_cached,
)
from jax_orchidee.driver.forcing_readers import ForcingReaderError

FIXTURE = json.loads(
    (Path(__file__).parents[1] / "fixtures/watchout_parameter_transport.json").read_text(encoding="utf-8")
)
PAYLOAD = ("swnet", "eair", "petAcoef", "peqAcoef", "petBcoef", "peqBcoef", "cdrag", "ccanopy")


def _case(tmp_path, *, daily=False, missing=None, records=None, watchout=True, multilanded=False):
    spec = FIXTURE["daily" if daily else "ordinary"]
    sequence = spec["record_sequence"] if records is None else records
    shape = (2, 2) if multilanded else (1, 1)
    lon, lat = FIXTURE["point"]["lon"], FIXTURE["point"]["lat"]
    contfrac = np.asarray([[0.5, 0.0], [0.3, 0.8]]) if multilanded else np.asarray([[0.5]])
    ds = xr.Dataset({
        "nav_lon": (("y", "x"), np.broadcast_to(lon + np.arange(shape[1]), shape)),
        "nav_lat": (("y", "x"), np.broadcast_to(lat + np.arange(shape[0])[:, None], shape)),
        "Areas": (("y", "x"), np.full(shape, 1234.0)),
        "contfrac": (("y", "x"), contfrac),
        "Height_Lev1": ((), 9.0),
        "Height_Levuv": ((), 17.0),
    })
    if watchout:
        for suffix in ("NN", "NE", "EE", "SE", "SS", "SW", "WW", "NW"):
            ds["neighbours" + suffix] = (("y", "x"), np.full(shape, -1.0))
        ds["resolutionX"] = (("y", "x"), np.full(shape, 1200.0))
        ds["resolutionY"] = (("y", "x"), np.full(shape, 2300.0))
    for name in FIXTURE["record_templates"]["1"]:
        if name == missing or (not watchout and name == "levels"):
            continue
        if daily and name in ("Tair", "Rainf", "Snowf"):
            continue
        values = np.asarray([FIXTURE["record_templates"][str(i)][name] for i in sequence])
        values = np.broadcast_to(values[:, None, None], (len(sequence), *shape)).copy()
        if multilanded:
            # Original backend's i/j coefficients, recorded independently in
            # fixture.input_recipe, make wrong selected-land gathers visible.
            dx, dy = {"levels": (0.1, 0.0), "SWnet": (1.0, 1.0), "Eair": (10.0, 1.0),
                      "petAcoef": (0.01, 0.0), "peqAcoef": (0.0, 0.01),
                      "petBcoef": (0.01, 0.0), "peqBcoef": (0.0, 0.01),
                      "cdrag": (0.001, 0.0), "ccanopy": (1.0, 1.0)}.get(name, (0.0, 0.0))
            values += dx * np.arange(shape[1])[None, None, :] + dy * np.arange(shape[0])[None, :, None]
        ds[name] = (("tstep", "y", "x"), values)
    ds = ds.assign_coords(tstep=("tstep", np.arange(len(sequence)) * spec["dt_force"]))
    ds["tstep"].attrs["units"] = "seconds since 1961-01-01 00:00:00"
    if missing in ds:
        ds = ds.drop_vars(missing)
    forcing_path = tmp_path / "forcing_1961.nc"
    ds.to_netcdf(forcing_path)
    config_path = tmp_path / "case.yaml"
    config_path.write_text(
        "drivers:\n  atmospheric_forcing:\n"
        f"    file_pattern: '{forcing_path.as_posix()}'\n"
        f"time:\n  forcing_step_hours: {spec['dt_force'] / 3600.0}\n"
        f"domain:\n  west: {lon - 0.1}\n  east: {lon + 1.1}\n  south: {lat - 0.1}\n  north: {lat + 1.1}\n",
        encoding="utf-8",
    )
    return config_path, read_domain_grid(config_path)


def _assert_expected(actual, expected):
    for name, value in expected.items():
        np.testing.assert_allclose(getattr(actual, name), [[value]], rtol=2e-14, atol=1e-13, err_msg=name)
    assert actual.is_watchout


@pytest.mark.parametrize("reader", [read_forcing_first_step, read_forcing_step, read_forcing_step_cached])
def test_raw_first_record_is_transported_by_real_high_level_readers(tmp_path, reader):
    config, domain = _case(tmp_path)
    actual = reader(config, domain=domain)
    _assert_expected(actual, FIXTURE["initialization"]["expected"])
    assert actual.Height_Lev1 == 9.0
    assert actual.Height_Levuv == 17.0


@pytest.mark.parametrize("case", FIXTURE["ordinary"]["cases"], ids=lambda c: str(c["model_tstep"]))
def test_ordinary_substeps_match_independent_original_fortran_payload(tmp_path, case):
    config, domain = _case(tmp_path)
    actual = read_forcing_model_step_cached(config, domain=domain, model_tstep=case["model_tstep"], split=2, nb_spread=1)
    _assert_expected(actual, case["expected"])


@pytest.mark.parametrize("case", FIXTURE["split1"]["cases"], ids=lambda c: str(c["model_tstep"]))
def test_split_one_uses_current_record_without_rescaling(tmp_path, case):
    config, domain = _case(tmp_path)
    actual = read_forcing_model_step_cached(config, domain=domain, model_tstep=case["model_tstep"], split=1, nb_spread=1)
    _assert_expected(actual, case["expected"])


def test_radiation_conservation_selects_current_radiation_without_changing_payload(tmp_path):
    """readdim2:1572-1578,1612-1618: NETRAD_CONS at dt_force<=3600."""

    config, domain = _case(tmp_path)
    actual = read_forcing_model_step_cached(config, domain=domain, split=2, nb_spread=1, netrad_cons=True)
    _assert_expected(actual, FIXTURE["ordinary"]["cases"][0]["expected"])
    np.testing.assert_array_equal(actual.SWdown, [[111.0]])
    np.testing.assert_array_equal(actual.LWdown, [[212.0]])


def test_daily_half_day_and_cross_day_payload_phase_matches_original_fortran(tmp_path):
    config, domain = _case(tmp_path, daily=True)
    for case in FIXTURE["daily"]["cases"]:
        actual = read_forcing_model_step_cached(config, domain=domain, model_tstep=case["model_tstep"], split=24, nb_spread=6)
        _assert_expected(actual, case["expected"])
        for name, expected in case["expected_meteorology"].items():
            public = {"tair": "temp_air", "rainf": "precip_rain", "snowf": "precip_snow"}.get(name, name)
            np.testing.assert_allclose(getattr(actual, public), [[expected]], rtol=2e-13, atol=1e-12, err_msg=name)


def test_daily_cursor_backwards_and_repeated_read_is_deterministic(tmp_path):
    config, domain = _case(tmp_path, daily=True)
    for case in reversed(FIXTURE["daily"]["cases"]):
        for _ in range(2):
            actual = read_forcing_model_step_cached(config, domain=domain, model_tstep=case["model_tstep"], split=24, nb_spread=6)
            _assert_expected(actual, case["expected"])


def test_watchout_file_geometry_is_retained_without_regular_grid_reconstruction(tmp_path):
    _, domain = _case(tmp_path)
    np.testing.assert_array_equal(domain.resolution, [[1200.0, 2300.0]])
    np.testing.assert_array_equal(domain.neighbours, np.full((1, 8), -1))
    np.testing.assert_array_equal(domain.area, [1200.0 * 2300.0])


def test_watchout_source_area_comes_from_resolution_without_areas_variable(tmp_path):
    config, domain = _case(tmp_path, missing="Areas")
    for reader in (read_forcing_step, read_forcing_step_cached):
        actual = reader(config, domain=domain)
        np.testing.assert_array_equal(actual.Areas, [[1200.0 * 2300.0]])
    np.testing.assert_array_equal(domain.area, [1200.0 * 2300.0])


def test_multiland_model_substeps_keep_land_order_and_latitude_geometry(tmp_path):
    config, domain = _case(tmp_path, daily=True, multilanded=True)
    actual = read_forcing_model_step_cached(config, domain=domain, model_tstep=11, split=24, nb_spread=6)
    for name in (*PAYLOAD, "zlev", "zlevuv", "swdown", "temp_air", "qair", "precip_snow"):
        assert getattr(actual, name).shape == (3, 1)
        assert np.all(np.isfinite(getattr(actual, name)))
    expected = FIXTURE["daily"]["cases"][1]["expected"]
    np.testing.assert_allclose(actual.petAcoef, [[expected["petAcoef"]], [expected["petAcoef"]], [expected["petAcoef"] + 0.01]])
    np.testing.assert_allclose(actual.peqAcoef, [[expected["peqAcoef"]], [expected["peqAcoef"] + 0.01], [expected["peqAcoef"] + 0.01]])
    # Same longitude and forcing SWdown, distinct latitude: solarang must
    # use each full-grid latitude rather than a gathered column's first row.
    assert not np.isclose(actual.swdown[0, 0], actual.swdown[1, 0])


@pytest.mark.parametrize("name", ["neighboursNN", "resolutionY"])
def test_watchout_missing_grid_payload_is_fatal(tmp_path, name):
    config, _ = _case(tmp_path)
    path = tmp_path / "forcing_1961.nc"
    with xr.open_dataset(path, decode_times=False) as source:
        ds = source.load().drop_vars(name)
    ds.to_netcdf(path, mode="w")
    with pytest.raises(ForcingReaderError, match=name):
        read_domain_grid(config)


@pytest.mark.parametrize("missing", ["SWnet", "Eair", "petAcoef", "peqAcoef", "petBcoef", "peqBcoef", "cdrag", "ccanopy"])
def test_watchout_missing_payload_is_fatal_in_raw_and_cache_readers(tmp_path, missing):
    config, domain = _case(tmp_path, missing=missing)
    for reader in (read_forcing_step, read_forcing_step_cached):
        with pytest.raises(ForcingReaderError, match=missing):
            reader(config, domain=domain)


def test_short_watchout_file_retains_source_first_substep_failure(tmp_path):
    config, domain = _case(tmp_path, records=[1, 2])
    assert read_forcing_first_step(config, domain=domain).is_watchout
    with pytest.raises(ValueError, match="less than 24 hours"):
        read_forcing_model_step_cached(config, domain=domain, split=2, nb_spread=1)


def test_selected_land_cache_retains_all_payload_and_order(tmp_path):
    config, domain = _case(tmp_path, multilanded=True)
    actual = read_forcing_step_cached(config, domain=domain, tstep=1)
    assert domain.nbindex == 3
    np.testing.assert_array_equal(domain.forcing_indices_zero_based, [[0, 0], [0, 1], [1, 1]])
    assert actual.is_watchout
    for name in PAYLOAD:
        assert getattr(actual, name).shape == (3, 1)
        expected = FIXTURE["ordinary"]["cases"][-1]["expected"][name]
        dx, dy = {"swnet": (1.0, 1.0), "eair": (10.0, 1.0),
                  "petAcoef": (0.01, 0.0), "peqAcoef": (0.0, 0.01),
                  "petBcoef": (0.01, 0.0), "peqBcoef": (0.0, 0.01),
                  "cdrag": (0.001, 0.0), "ccanopy": (1.0, 1.0)}[name]
        np.testing.assert_allclose(getattr(actual, name), [[expected], [expected + dy], [expected + dx + dy]])
    cache = _read_forcing_land_cache(str(config.resolve()), 1961, ((0, 0), (0, 1), (1, 1)))
    assert cache.petAcoef.shape == (24, 3, 1)


def test_non_watchout_detection_and_legacy_construction_pickle_remain_compatible(tmp_path):
    config, domain = _case(tmp_path, watchout=False)
    actual = read_forcing_step_cached(config, domain=domain)
    assert not actual.is_watchout
    assert all(getattr(actual, name) is None for name in PAYLOAD)
    legacy = ForcingStep(**{field.name: getattr(actual, field.name) for field in fields(actual) if field.default is MISSING})
    restored = pickle.loads(pickle.dumps(legacy))
    assert type(restored) is ForcingStep
    assert not restored.is_watchout
    np.testing.assert_array_equal(restored.zlev, [[9.0]])
    np.testing.assert_array_equal(restored.zlevuv, [[17.0]])
    # Historical pickle payloads have only the original instance state.
    old_state = object.__new__(ForcingStep)
    for field in fields(actual):
        if field.default is MISSING:
            object.__setattr__(old_state, field.name, getattr(actual, field.name))
    restored_old_state = pickle.loads(pickle.dumps(old_state))
    assert type(restored_old_state) is ForcingStep
    assert not restored_old_state.is_watchout
    assert all(getattr(restored_old_state, name) is None for name in PAYLOAD)
