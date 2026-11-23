"""Source-owner WATCHOUT boundaries; all physical process calls are witnesses."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from jax_orchidee.driver import bundle as b
from jax_orchidee.driver import dim2
from jax_orchidee.driver import orchestration as o
from jax_orchidee.driver.sechiba_boundary import build_intersurf_first_step_payload
from tests.unit.test_dim2_driver import _bundle, _output
from tests.unit.test_watchout_paper_compiled_transport import _clock_payload, _prepared_case

REFERENCE = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures/watchout_parameter_transport.json").read_text(encoding="utf-8")
)
COEFFICIENTS = ("petAcoef", "peqAcoef", "petBcoef", "peqBcoef")
WATCH_FIELDS = ("eair", "swnet", "cdrag", *COEFFICIENTS)


def watch_bundle(record=1, *, watchout=True):
    bundle = _bundle(0, npts=1)
    values = REFERENCE["record_templates"][str(record)]
    def vector(value):
        return np.array([value], dtype=np.float64)
    meteorology = {
        "temp_air": "Tair", "qair": "Qair", "pb": "PSurf", "u": "Wind_N", "v": "Wind_E",
        "swdown": "SWdown", "lwdown": "LWdown", "precip_rain": "Rainf", "precip_snow": "Snowf",
    }
    forcing = dict(bundle.forcing.__dict__)
    forcing.update({name: vector(values[source]) for name, source in meteorology.items()})
    forcing.update(
        is_watchout=watchout, zlev=vector(values["levels"]), zlevuv=vector(values["levels"]),
        eair=vector(values["Eair"]), swnet=vector(values["SWnet"]),
        **{name: vector(values[name]) for name in ("cdrag", "ccanopy", *COEFFICIENTS)},
    )
    bundle.forcing = NS(**forcing)
    bundle.ccanopy = vector(REFERENCE["driver"]["annual_atmco2"])
    return bundle


def test_watchout_schema_payload_keeps_tuple_and_annual_co2_override():
    bundle = watch_bundle()
    payload = build_intersurf_first_step_payload(bundle)
    assert payload.is_watchout
    for name in WATCH_FIELDS:
        np.testing.assert_array_equal(getattr(payload, name), getattr(bundle.forcing, name))
    np.testing.assert_array_equal(payload.ccanopy, [REFERENCE["driver"]["expected_ccanopy"]])
    assert payload.ccanopy[0] != bundle.forcing.ccanopy[0]
    np.testing.assert_array_equal(payload.u, bundle.forcing.u)
    np.testing.assert_array_equal(payload.v, bundle.forcing.v)


def test_watchout_is_admitted_but_original_relaxation_fatal_remains():
    dim2.validate_dim2_switches(dim2.Dim2Switches(watchout=True))
    with pytest.raises(NotImplementedError, match="RELAXATION"):
        dim2.validate_dim2_switches(dim2.Dim2Switches(watchout=True, relaxation=True))


@pytest.mark.parametrize("explicit_watchout", [False, True])
def test_actual_dim2_calls_observe_source_initialization_and_ordinary_phases(explicit_watchout):
    time = dim2.resolve_dim2_time_control(
        dt_force=3600.0, dt_sechiba=1800.0, forcing_length=2, date0=0.0, time_length=1,
    )
    seen_init, seen_main = [], []

    def initialize(boundary, state):
        seen_init.append(boundary)
        return _output(boundary, state, initialize=True)

    def main(boundary, state):
        seen_main.append(boundary)
        return _output(boundary, state)

    # Source mode comes from the file-derived bundle even when WATCHOUT is absent in run.def.
    result = dim2.run_dim2_driver(
        time=time, forcing_reader=lambda step: watch_bundle(step + 1),
        intersurf_initialize=initialize, intersurf_main=main,
        switches=dim2.Dim2Switches(watchout=explicit_watchout),
        restart=dim2.Dim2DriverRestart(
            **{name: np.array([999.0]) for name in COEFFICIENTS},
            albedo_vis=np.array([0.3]), albedo_nir=np.array([0.5]),
        ),
    )
    assert len(seen_init) == 1 and len(seen_main) == 2
    for boundary, record in ((seen_init[0], 1), (seen_main[0], 1), (seen_main[1], 2)):
        expected = REFERENCE["record_templates"][str(record)]
        for name in ("eair", "cdrag", *COEFFICIENTS):
            source = "Eair" if name == "eair" else name
            np.testing.assert_array_equal(getattr(boundary, name), [expected[source]])
        np.testing.assert_array_equal(boundary.payload.ccanopy, [REFERENCE["driver"]["expected_ccanopy"]])
    np.testing.assert_array_equal(seen_init[0].swnet, [52.0])
    # Source line1163 has no WATCHOUT guard: restored albedo applies to the first main call.
    np.testing.assert_array_equal(seen_main[0].swnet, [66.6])
    np.testing.assert_array_equal(seen_main[1].swnet, [102.0])
    for name in COEFFICIENTS:
        assert getattr(result.initial_restart, name) is None
        assert getattr(result.restart, name) is None


def test_actual_nonwatchout_boundary_keeps_source_recalculation():
    time = dim2.resolve_dim2_time_control(
        dt_force=1800.0, dt_sechiba=1800.0, forcing_length=1, date0=0.0,
    )
    result = dim2.run_dim2_driver(
        time=time, forcing_reader=lambda step: watch_bundle(watchout=False),
        intersurf_initialize=lambda b, s: _output(b, s, initialize=True),
        intersurf_main=lambda b, s: _output(b, s),
    )
    boundary = result.steps[0].main_boundary
    np.testing.assert_array_equal(boundary.petAcoef, [0.0])
    np.testing.assert_array_equal(boundary.peqAcoef, [0.0])
    np.testing.assert_array_equal(boundary.cdrag, [0.0])
    np.testing.assert_array_equal(boundary.petBcoef, boundary.eair)
    np.testing.assert_array_equal(boundary.peqBcoef, boundary.payload.qair)
    assert boundary.eair[0] != REFERENCE["record_templates"]["1"]["Eair"]


def test_actual_paper_energy_payload_preserves_watchout_and_first_main_exception():
    payload = build_intersurf_first_step_payload(watch_bundle())
    actual = o._enerbil_driver_coverage_payload(payload)
    np.testing.assert_array_equal(actual["epot_air"], [1011.0])
    np.testing.assert_array_equal(actual["swnet"], [52.0])
    for name in COEFFICIENTS:
        np.testing.assert_array_equal(actual[name], getattr(payload, name))
    initial = o._enerbil_driver_coverage_payload(
        payload, initializing=True, driver_albedo_payload={"albedo_vis": np.array([0.3]), "albedo_nir": np.array([0.5])},
    )
    np.testing.assert_array_equal(initial["swnet"], [66.6])


def test_compiled_projection_keeps_same_shape_record_changes_dynamic():
    a = build_intersurf_first_step_payload(watch_bundle(1))
    b = build_intersurf_first_step_payload(watch_bundle(2))
    no_tide_stream = {"source_clock": None, "hydrol_tide_forcing": None}
    first = o._compiled_half_hour_forcing(NS(base_payload=a, step=watch_bundle(1), **no_tide_stream))
    second = o._compiled_half_hour_forcing(NS(base_payload=b, step=watch_bundle(2), **no_tide_stream))
    assert first.watchout is not None and second.watchout is not None
    for name in WATCH_FIELDS:
        np.testing.assert_array_equal(getattr(first.watchout, name), getattr(a, name))
        np.testing.assert_array_equal(getattr(second.watchout, name), getattr(b, name))
        assert getattr(first.watchout, name)[0] != getattr(second.watchout, name)[0]
    assert not (set(WATCH_FIELDS) & set(o._COMPILED_INTERSURF_PAYLOAD_FIELDS))


def test_schema_incomplete_tuple_rejects_before_any_consumer():
    bundle = watch_bundle()
    bundle.forcing.petAcoef = None
    with pytest.raises(ValueError, match="petAcoef"):
        build_intersurf_first_step_payload(bundle)


@pytest.mark.parametrize("entry", ["bundle", "iterator", "prepared", "compiled"])
@pytest.mark.parametrize("values,conserved", [
    ({}, False),
    ({"INTER_LIN": "y"}, True),
    ({"INTER_LIN": "y", "NETRAD_CONS": "n"}, False),
    ({"INTER_LIN": "n", "NETRAD_CONS": "y"}, False),
])
def test_actual_forcing_entries_use_effective_radiation_controls(tmp_path, monkeypatch, entry, values, conserved):
    """Source DIM2620-696 controls actual LWdown transport, no physics.

    readdim21550-1583 at ordinary rw=.5 gives (212+222)/2=217;
    NETRAD_CONS instead retains the current file record, 222.
    """
    context = _prepared_case(tmp_path, monkeypatch)
    context.run_def_values = values
    first = context.first_step_bundle
    first.co2_ppm = REFERENCE["driver"]["annual_atmco2"]
    first.run_scalars.raw_run_def = {"DT_SECHIBA": "1800", **values}
    monkeypatch.setattr(b, "_load_paper_1961_first_step_bundle_cached", lambda *args: first)
    monkeypatch.setattr(b, "load_paper_1961_first_step_bundle", lambda *args, **kwargs: first)
    monkeypatch.setattr(o, "_driver_source_clock_payload", _clock_payload)
    if entry == "bundle":
        actual = b.load_paper_1961_step_bundle(context.config_path, tstep=2).forcing.lwdown
    elif entry == "iterator":
        actual = list(b.iter_paper_year_step_bundles(context.config_path, nsteps=3))[-1].forcing.lwdown
    elif entry == "prepared":
        actual = o._paper_1961_step_bundle_from_context(context, year=1961, tstep=2).forcing.lwdown
    else:
        actual = o._paper_compiled_forcing_day(context, year=1961, start_tstep=2, steps_per_stomate=1).lwdown
    np.testing.assert_array_equal(actual, np.full_like(actual, 222.0 if conserved else 217.0))


def test_watchout_file_cadence_and_spread_override_reach_actual_prepared_entry(tmp_path, monkeypatch):
    context = _prepared_case(tmp_path, monkeypatch)
    context.forcing_split = 12  # Deliberately stale case cadence; source file is hourly.
    context.forcing_nb_spread = 6
    context.run_def_values = {"SPRED_PREC": "1"}
    step = o._paper_1961_step_bundle_from_context(context, year=1961, tstep=2)
    np.testing.assert_array_equal(step.forcing.lwdown, [[217.0]])
    np.testing.assert_allclose(step.forcing.precip_rain, [[6e-5]], rtol=0, atol=1e-19)
    context.run_def_values = {}
    ordinary = o._paper_1961_step_bundle_from_context(context, year=1961, tstep=2)
    np.testing.assert_allclose(ordinary.forcing.precip_rain, [[3e-5]], rtol=0, atol=1e-19)


def test_watchout_split_one_ignores_unread_netrad_override(tmp_path, monkeypatch):
    context = _prepared_case(tmp_path, monkeypatch)
    context.dt_sechiba = 3600.0
    # DIM2647 disables INTER_LIN; unread NETRAD_CONS must not be parsed.
    context.run_def_values = {"INTER_LIN": "y", "NETRAD_CONS": "not-read-by-source"}
    step = o._paper_1961_step_bundle_from_context(context, year=1961, tstep=1)
    np.testing.assert_array_equal(step.forcing.lwdown, [[222.0]])


@pytest.mark.parametrize("watchout", [False, True])
@pytest.mark.parametrize("fast", [False, True])
def test_actual_runtime_writer_uses_mode_without_persisting_forcing_coefficients(monkeypatch, watchout, fast):
    """Run the shared state writer with component arrays, without physics."""
    previous = o.DriverPreviousStepStatePacket(0, {}, {})
    if fast:
        previous = o.fast_state_from_previous_packet(previous)
    values = {name: np.array([10.0 + i]) for i, name in enumerate(COEFFICIENTS)}
    values.update(fluxsens=np.array([3.0]), vevapp=np.array([1.8]), dt_sechiba=1800.0,
                  zlev=np.array([2.0]), qair=np.array([0.003]), temp_air=np.array([280.0]),
                  epot_air=np.array([12345.0]), rau=np.array([1.2]))
    monkeypatch.setattr(o, "_apply_driver_source_calendar", lambda state, **kwargs: state)
    result = o._runtime_next_state_from_components(
        tstep=1, previous_state=previous, enerbil_local=NS(payload=values), diffuco_local=None,
        thermosoil_module=None, condveg_module=None, hydrol_module=None,
        is_watchout=watchout, source_clock=object(),
    )
    if fast:
        result = o.previous_packet_from_fast_state(result)
    fields = result.fields_by_component["driver_previous_step_state"]
    for name in COEFFICIENTS:
        if watchout:
            assert name not in fields
        else:
            np.testing.assert_array_equal(fields[name], values[name])
    np.testing.assert_array_equal(fields["old_eair"], values["epot_air"] if watchout
                                  else o.CP_AIR * values["temp_air"] + o.GRAVITY * values["zlev"])
    np.testing.assert_allclose(fields["vevapp"], [0.001], rtol=0, atol=1e-18)
