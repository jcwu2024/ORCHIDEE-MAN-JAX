"""R3/R4: bounded real-file and public bundle transport repairs.

No physical-model transition runs. R3 expected values are literal results of
the original NINT(record5), rw=1/4 contract, independently specified here.
R4 observes the real iterator's loop boundary without iterating a model year.
"""

from __future__ import annotations

from types import SimpleNamespace as NS

import numpy as np
import pytest
import xarray as xr

from jax_orchidee.driver import bundle as b
from jax_orchidee.driver.domain import read_domain_grid, read_forcing_first_step
from jax_orchidee.driver.forcing_interpolation import (
    ForcingInterpolationRecord,
    ForcingInterpolationState,
    forcing_read_interpol_transition,
)

R3_COLUMNS = {
    "Tair": [270.0, 280.0, 290.0, 300.0, 310.0],
    "PSurf": [91000.0, 92000.0, 93000.0, 94000.0, 95000.0],
    "Qair": [0.001, 0.002, 0.003, 0.004, 0.005],
    "levels": [2.0, 4.0, 6.0, 8.0, 10.0],
    "SWnet": [31.0, 62.0, 93.0, 124.0, 155.0],
    "Eair": [1001.0, 2002.0, 3003.0, 4004.0, 5005.0],
    "petAcoef": [0.1, 0.2, 0.3, 0.4, 0.5],
    "peqAcoef": [0.2, 0.4, 0.6, 0.8, 1.0],
    "petBcoef": [201.0, 402.0, 603.0, 804.0, 1005.0],
    "peqBcoef": [0.001, 0.002, 0.003, 0.004, 0.005],
    "cdrag": [0.001, 0.002, 0.003, 0.004, 0.005],
    "ccanopy": [350.0, 360.0, 370.0, 380.0, 390.0],
    "Wind_N": [1.0, 2.0, 3.0, 4.0, 5.0],
    "Wind_E": [-1.0, -2.0, -3.0, -4.0, -5.0],
    "Rainf": [1e-5, 2e-5, 3e-5, 4e-5, 5e-5],
    "Snowf": [1e-6, 2e-6, 3e-6, 4e-6, 5e-6],
    "SWdown": [100.0, 200.0, 300.0, 400.0, 500.0],
    "LWdown": [200.0, 210.0, 220.0, 230.0, 240.0],
}
R3_EXPECTED = {
    "zlev": 8.0, "zlevuv": 8.0, "swnet": 124.0, "eair": 4004.0,
    "petAcoef": 0.4, "peqAcoef": 0.8, "petBcoef": 804.0,
    "peqBcoef": 0.004, "cdrag": 0.004, "ccanopy": 380.0,
    "temp_air": 300.0, "pb": 94000.0, "qair": 0.004, "u": 4.0, "v": -4.0,
}


def _write_case(tmp_path, *, r3=False, count=24, time_length=None):
    dt_force, dt = (19200.0, 4800.0) if r3 else (3600.0, 1800.0)
    if r3:
        columns = R3_COLUMNS
        count = 5
    else:
        record = np.arange(count, dtype=np.float64)
        columns = {name: np.full(count, values[0]) for name, values in R3_COLUMNS.items()}
        columns.update(Tair=280.0 + 0.01 * record, SWnet=10.0 + record, Eair=1000.0 + 100.0 * record)
    ds = xr.Dataset({
        "nav_lon": (("y", "x"), [[0.0]]), "nav_lat": (("y", "x"), [[0.0]]),
        "Height_Lev1": ((), 9.0), "Height_Levuv": ((), 17.0),
        "contfrac": (("y", "x"), [[1.0]]),
        "resolutionX": (("y", "x"), [[1000.0]]), "resolutionY": (("y", "x"), [[1000.0]]),
    })
    for suffix in ("NN", "NE", "EE", "SE", "SS", "SW", "WW", "NW"):
        ds["neighbours" + suffix] = (("y", "x"), [[-1.0]])
    for name, values in columns.items():
        ds[name] = (("tstep", "y", "x"), np.asarray(values).reshape(count, 1, 1))
    ds = ds.assign_coords(tstep=("tstep", np.arange(count) * dt_force))
    ds.tstep.attrs.update(units="seconds since 1961-01-01 00:00:00", calendar="noleap")
    forcing_path = tmp_path / "forcing_1961.nc"
    ds.to_netcdf(forcing_path)
    config_path = tmp_path / "case.yaml"
    # Intentionally retain stale paper cadence and record-count metadata.
    config_path.write_text(
        "drivers:\n  atmospheric_forcing:\n"
        f"    file_pattern: '{forcing_path.as_posix()}'\n"
        "    dimensions:\n      tstep: 1460\n"
        "time:\n  forcing_step_hours: 6\n  forcing_calendar: noleap\n"
        "domain:\n  west: -0.1\n  east: 0.1\n  south: -0.1\n  north: 0.1\n",
        encoding="utf-8",
    )
    domain = read_domain_grid(config_path)
    values = {"DT_SECHIBA": str(dt)}
    if time_length is not None:
        values["TIME_LENGTH"] = str(time_length)
    first = NS(
        domain=domain, forcing=read_forcing_first_step(config_path, domain=domain),
        run_scalars=NS(dt_sechiba=dt, raw_run_def=values), vegetation=None,
        co2_ppm=400.0, ccanopy=np.asarray([400.0]), water_table=None,
        restart_anchors=None, static_trace_fields=None, missing_geometry_fields=(), missing_static_fields=(),
    )
    return config_path, first


def _stub_static_bundle(monkeypatch, first):
    # The NetCDF, domain, forcing, effective controls and public bundle caller
    # remain real. Unrelated soil/vegetation/restart preparation is delegated.
    monkeypatch.setattr(b, "load_paper_1961_first_step_bundle", lambda *args, **kwargs: first)
    monkeypatch.setattr(b, "_load_paper_1961_first_step_bundle_cached", lambda *args, **kwargs: first)


def test_r3_source_nint_half_tie_selects_record_five():
    mapping = {"tair": "Tair", "pb": "PSurf", "qair": "Qair", "zlev": "levels", "zlevuv": "levels",
               "u": "Wind_N", "v": "Wind_E", "rainf": "Rainf", "snowf": "Snowf", "swdown": "SWdown", "lwdown": "LWdown",
               "swnet": "SWnet", "eair": "Eair", "petacoef": "petAcoef", "peqacoef": "peqAcoef",
               "petbcoef": "petBcoef", "peqbcoef": "peqBcoef", "cdrag": "cdrag", "ccanopy": "ccanopy"}
    records = tuple(ForcingInterpolationRecord.from_mapping({
        public: np.asarray([[R3_COLUMNS[file_name][index]]]) for public, file_name in mapping.items()
    }) for index in range(5))
    kwargs = dict(split=4, nb_spread=2, dt_force=19200.0, lon=np.asarray([[0.0]]), lat=np.asarray([[0.0]]), daily_interpol=False, is_watchout=True)
    initialized = forcing_read_interpol_transition(records, ForcingInterpolationState(), itauin=0, itau_split=0, **kwargs)
    actual = forcing_read_interpol_transition(records, initialized.state, itauin=1, itau_split=1, **kwargs)
    assert actual.state.itau_read_nm1 == 5  # NINT(4.5)=5; Python round(4.5)=4.
    np.testing.assert_array_equal(actual.values.eair, [[4004.0]])
    np.testing.assert_array_equal(actual.values.zlev, [[8.0]])


def test_r3_real_file_public_bundle_uses_original_record_five(tmp_path, monkeypatch):
    config, first = _write_case(tmp_path, r3=True)
    _stub_static_bundle(monkeypatch, first)
    actual = b.load_paper_1961_step_bundle(config, tstep=0).forcing
    assert actual.is_watchout
    assert actual.dt_force == 19200.0
    for name, expected in R3_EXPECTED.items():
        np.testing.assert_allclose(getattr(actual, name), [[expected]], rtol=2e-14, atol=1e-13, err_msg=name)


def test_r3_source_full_day_admission_is_not_only_a_rounded_index_check():
    record = ForcingInterpolationRecord.from_mapping({"tair": np.asarray([[280.0]])})
    records = (record,) * 4
    kwargs = dict(split=4, nb_spread=2, dt_force=20000.0, lon=np.asarray([[0.0]]), lat=np.asarray([[0.0]]), daily_interpol=False, is_watchout=False)
    initialized = forcing_read_interpol_transition(records, ForcingInterpolationState(), itauin=0, itau_split=0, **kwargs)
    # 4*20000 <86400-1 although NINT(86400/20000)=4 is in range.
    with pytest.raises(ValueError, match="less than 24 hours"):
        forcing_read_interpol_transition(records, initialized.state, itauin=1, itau_split=1, **kwargs)


@pytest.mark.parametrize("count,time_length,expected", [
    (24, None, 48), (48, None, 96), (8760, None, 17520),
    (24, "2", 4), (24, "1D", 48), (24, "2D", 96), (24, "1Y", 17520),
], ids=["one-day-file", "two-day-file", "year-file-metadata-only", "explicit-records", "one-day-duration", "cycle-two-days", "year-duration-count-only"])
def test_r4_public_default_iterator_plans_real_span_or_source_time_length(tmp_path, monkeypatch, count, time_length, expected):
    config, first = _write_case(tmp_path, count=count, time_length=time_length)
    _stub_static_bundle(monkeypatch, first)
    observed = []

    def range_boundary(stop):
        observed.append(stop)
        return ()

    monkeypatch.setattr(b, "range", range_boundary, raising=False)
    monkeypatch.setattr(b, "read_forcing_model_step_cached", lambda *args, **kwargs: pytest.fail("count-only boundary must not advance forcing"))
    assert list(b.iter_paper_year_step_bundles(config)) == []
    assert observed == [expected]


def test_r4_explicit_nsteps_retains_its_call_site_override(tmp_path, monkeypatch):
    config, first = _write_case(tmp_path, time_length="1Y")
    _stub_static_bundle(monkeypatch, first)
    observed = []
    monkeypatch.setattr(b, "range", lambda stop: observed.append(stop) or (), raising=False)
    assert list(b.iter_paper_year_step_bundles(config, nsteps=3)) == []
    assert observed == [3]


def test_r4_explicit_longer_time_length_uses_source_cyclic_records(tmp_path, monkeypatch):
    config, first = _write_case(tmp_path, time_length="2D")
    _stub_static_bundle(monkeypatch, first)
    # Observe the true iterator at its second-file boundary, without any
    # physical model callbacks. At most49 forcing-only substeps are read.
    iterator = b.iter_paper_year_step_bundles(config)
    first_value = next(iterator).forcing
    for _ in range(47):
        next(iterator)
    second_cycle = next(iterator).forcing
    assert second_cycle.tstep == 48
    np.testing.assert_array_equal(first_value.eair, [[2150.0]])
    np.testing.assert_array_equal(second_cycle.eair, [[2150.0]])
    iterator.close()


@pytest.mark.parametrize("time_length,nsteps", [("-1", None), (None, -1)])
def test_r4_negative_planned_length_is_rejected_before_any_forcing_step(tmp_path, monkeypatch, time_length, nsteps):
    config, first = _write_case(tmp_path, time_length=time_length)
    _stub_static_bundle(monkeypatch, first)
    monkeypatch.setattr(b, "read_forcing_model_step_cached", lambda *args, **kwargs: pytest.fail("negative count must not read a substep"))
    with pytest.raises(ValueError, match="non-negative"):
        next(b.iter_paper_year_step_bundles(config, nsteps=nsteps))
