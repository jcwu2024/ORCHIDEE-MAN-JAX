"""Bounded real component-result handoffs; never executes a model timestep.

Calendar admission and unrelated CONDVEG/THERMOSOIL readiness are isolated.
The selected HYDROL/DIFFUCO/ENERBIL result objects and shared state writers are
real. TOPMODEL=True is a separate diagnostics chain on a real HYDROL result,
not a newly admitted top-level wrapper mode.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import jax
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jax_orchidee.driver import orchestration as driver  # noqa: E402 - direct CLI root bootstrap
from jax_orchidee.driver.fast_state import fast_state_from_previous_packet  # noqa: E402

PAPER = ROOT / "reference/case_001_071/OUT/orc_calibrate_250919_sen/arg2_1.0/001.0-071.0/I10/S2_63.206_0.0876_0.2019_50.658"
ROUTES = ("first", "ordinary", "runtime", "compiled_fast")
BOUND_SOURCES = (
    "scripts/dev/check_hydrol_diffuco_state_handoff.py",
    "tests/unit/test_hydrol_diffuco_state_handoff.py",
    "tests/fixtures/hydrol_fwet_carry_reference.json",
    "scripts/dev/oracle_hydrol_component_owner.py",
    "scripts/dev/oracle_canopy_component_fixture.py",
    "jax_orchidee/driver/orchestration.py",
    "jax_orchidee/driver/fast_state.py",
    "jax_orchidee/driver/restart_bundle.py",
    "jax_orchidee/driver/restart_file_io.py",
    "jax_orchidee/sechiba/restart_io.py",
    "jax_orchidee/sechiba/hydrol.py",
    "jax_orchidee/sechiba/diffuco.py",
    "jax_orchidee/sechiba/enerbil.py",
    "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90",
    "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90",
)


def real_components():
    from scripts.dev.oracle_canopy_component_fixture import make_canopy_fixture, run_canopy_components
    from scripts.dev.oracle_hydrol_component_owner import run_owner_case, run_owner_diagnostics
    hydrol = run_owner_case(1, use_jit=False)
    enabled = run_owner_diagnostics(hydrol, topmodel_new=True)
    enabled_hydrol = hydrol._replace(diagnostics=enabled)
    canopy = run_canopy_components(make_canopy_fixture(indices=(0,)))
    return hydrol, enabled_hydrol, canopy


def real_hydrol_canopy_storage():
    """New synthetic rain/storage boundary through the actual HYDROL wrapper.

    This changes the source-bound fixture inputs and claims state selection
    only; it does not inherit the original zero-canopy numerical oracle.
    """
    from jax_orchidee.sechiba.hydrol import run_hydrol_first_step_module_from_precall
    from scripts.dev.oracle_hydrol_component_owner import make_owner_inputs
    values = make_owner_inputs(1)
    payload = dict(values["precall"].payload)
    payload["precip_rain"] = np.asarray([0.5])
    for name, amount in (("qsintmax", 0.3), ("qsintveg", 0.07), ("vevapwet", 0.02)):
        value = np.array(payload[name], copy=True)
        value[:, 13] = amount
        payload[name] = value
    result = run_hydrol_first_step_module_from_precall(
        values["precall"]._replace(payload=payload), **values["kwargs"], use_jit=False)
    if not result.ok:
        raise RuntimeError(result.missing_inputs)
    return result


def previous_packet(hydrol=None):
    fields = driver._empty_previous_step_state_fields()
    fields["hydrol_previous_step_state"] = {"fwet_new": np.asarray([0.123])}
    # Deliberately stale dynamic mirrors: the current writers must replace them.
    fields["diffuco_previous_step_state"] = {"qsintveg": np.asarray([[-31.0]]),
                                               "humrel": np.asarray([[-32.0]])}
    # The ordinary scaffold contract indexes the empty slowproc component.
    # Preserve that structural component without inventing scientific fields.
    return driver.DriverPreviousStepStatePacket(0, fields, {})


def carry(route, hydrol, canopy=None, *, previous=None):
    """Execute real selected-field writers and exact first/ordinary/fast route.

    Scaffold shells supply structural metadata only. No fake physics output is
    substituted for the three component objects. Full model/calendar admission
    and SECHIBA 93-field finalize transition are intentionally not executed.
    """
    previous = previous_packet(hydrol) if previous is None else previous
    energy = None if canopy is None else canopy.enerbil
    diffuse = None if canopy is None else canopy.diffuco
    with patch.object(driver, "_driver_source_clock_payload", lambda *a, **k: None), \
         patch.object(driver, "_apply_driver_source_calendar", lambda state, **k: state):
        if route == "first":
            return driver.driver_previous_step_state_from_timestep_scaffold(NS(
                step=NS(tstep=1), source_clock_context=None,
                enerbil_first_step_local=energy,
                diffuco_first_step_local_enerbil_precall=diffuse,
                diffuco_control_salinity=None, diffuco_control_assembly=None,
                diffuco_first_step_precall=None, thermosoil_first_step_module=None,
                hydrol_first_step_module=hydrol, condveg_first_step_module=None,
                first_step_restart_state=None))
        if route == "ordinary":
            return driver.driver_previous_step_state_from_next_step_hydrol_scaffold(NS(
                ready_for_next_state=True, tstep=1, source_clock_context=None,
                enerbil=NS(enerbil_local=energy, diffuco_local_enerbil_precall=diffuse,
                           previous_state=previous, payload=NS(is_watchout=False)), thermosoil_module=None,
                hydrol_module=hydrol, condveg_module=None))
        if route not in ("runtime", "compiled_fast"):
            raise ValueError(route)
        prior = fast_state_from_previous_packet(previous) if route == "compiled_fast" else previous
        return driver._runtime_next_state_from_components(
            tstep=1, previous_state=prior, enerbil_local=energy,
            diffuco_local=diffuse, thermosoil_module=None,
            condveg_module=None, hydrol_module=hydrol)


def check_selected(packet, hydrol, canopy):
    fields = packet.fields_by_component
    h, d, e = (fields[name] for name in ("hydrol_previous_step_state", "diffuco_previous_step_state", "enerbil_previous_step_state"))
    checked = []
    def same(actual, expected, label):
        np.testing.assert_array_equal(actual, expected, err_msg=label)
        checked.append(label)
    for name in ("humrel", "evap_bare_lim"):
        same(h[name], getattr(hydrol.diagnostics, name), "hydrol."+name)
        same(d[name], getattr(hydrol.diagnostics, name), "diffuco."+name)
    same(h["qsintveg"], hydrol.module.canop.qsintveg, "hydrol.qsintveg")
    same(d["qsintveg"], hydrol.module.canop.qsintveg, "diffuco.qsintveg")
    same(d["drysoil_frac"], hydrol.diagnostics.drysoil_frac, "diffuco.drysoil_frac")
    for name in ("us", "ae_ns"):
        same(h[name], getattr(hydrol.diagnostics, name), "hydrol."+name)
    same(h["profil_froz_hydro_ns"], hydrol.module.soil.profil_froz_hydro_ns, "hydrol.refreshed_profile")
    if hydrol.diagnostics.fwet_new is not None:
        same(h["fwet_new"], hydrol.diagnostics.fwet_new, "hydrol.current_fwet_new")
    for name in ("snow", "snowdz", "snowrho", "snowtemp", "snowheat", "snowgrain", "snow_age", "snow_nobio", "snow_nobio_age"):
        same(h[name], getattr(hydrol.snow_state, name), "hydrol."+name)
        same(d[name], getattr(hydrol.snow_state, name), "diffuco."+name)
    same(e["snowdz"], hydrol.snow_state.snowdz, "enerbil.current_snowdz")
    same(e["temp_sol_add"], h["temp_sol_add"], "enerbil.temp_sol_add")
    mapped, _, _ = driver._diffuco_restart_payload_from_previous_state(packet)
    for name in ("qsintveg", "humrel", "evap_bare_lim", "drysoil_frac"):
        same(mapped[name], d[name], "next_diffuco_mapper."+name)
    if canopy is not None:
        for name in ("gpp", "gsmean", "rveget", "rstruct", "cimean"):
            same(d[name], canopy.diffuco.result.payload.payload[name], "diffuco."+name)
        same(fields["driver_previous_step_state"]["vevapp"],
             canopy.enerbil.payload["vevapp"] / canopy.enerbil.payload["dt_sechiba"], "driver.rate_once")
    return checked


def compact_selected_jit(hydrol, canopy):
    """Compile selected outputs of the actual fast-layout state builder only."""
    prior = fast_state_from_previous_packet(previous_packet(hydrol))
    with patch.object(driver, "_apply_driver_source_calendar", lambda state, **k: state):
        def extract(old):
            packet = driver._runtime_next_state_from_components(
                tstep=1, previous_state=old, enerbil_local=canopy.enerbil,
                diffuco_local=canopy.diffuco, thermosoil_module=None,
                condveg_module=None, hydrol_module=hydrol, source_clock=NS(active=False))
            return tuple(packet.component_fields(c)[n] for c,n in (
                ("hydrol_previous_step_state", "fwet_new"),
                ("hydrol_previous_step_state", "ae_ns"),
                ("diffuco_previous_step_state", "humrel"),
                ("driver_previous_step_state", "vevapp")))
        values = jax.jit(extract)(prior)
    jax.block_until_ready(values)
    return values


def next_diffuco_consumer(packet, canopy):
    """HYDROL outputs are next-call inputs, never same-call DIFFUCO truth."""
    from scripts.dev.oracle_canopy_component_fixture import run_canopy_components
    mapped, _, _ = driver._diffuco_restart_payload_from_previous_state(packet)
    names = ("qsintveg", "humrel", "evap_bare_lim", "drysoil_frac")
    precall = canopy.fixture.diffuco_precall._replace(
        payload={**canopy.fixture.diffuco_precall.payload, **{n: mapped[n] for n in names}})
    fixture = canopy.fixture._replace(diffuco_precall=precall)
    return run_canopy_components(fixture, run_enerbil=True)


def single_point_roundtrip(packet, canopy, directory):
    """Overlay only matching real selected values on an existing 93-field file.

    This tests true codecs and raw/init reader phases, not regeneration of the
    unrelated old scientific state or a full model restart handoff.
    """
    from dataclasses import replace

    from jax_orchidee.driver.dim2 import Dim2DriverRestart
    from jax_orchidee.driver.restart_bundle import read_restart_physical_state
    from jax_orchidee.driver.restart_file_io import read_dim2_driver_restart, write_dim2_driver_restart
    from jax_orchidee.sechiba.restart_io import (
        SechibaRestartState,
        read_sechiba_init_restart_state,
        read_sechiba_restart_state,
        write_sechiba_restart_state,
    )
    source = PAPER / "sechiba_start.nc"
    original = read_sechiba_restart_state(source)
    components = packet.fields_by_component
    h = components["hydrol_previous_step_state"]
    d = components["diffuco_previous_step_state"]
    actual = {**{n: h[n] for n in ("qsintveg", "humrel", "evap_bare_lim", "ae_ns", "fwet_new", "us", "snow", "snowdz", "snow_nobio") if n in h},
              "drysoil_frac": d["drysoil_frac"], "rstruct": d["rstruct"],
              **{n: canopy.enerbil.payload[n] for n in ("qsurf", "evapot", "evapot_corr", "fluxsens", "fluxlat")},
              "evapora": canopy.enerbil.payload["vevapp"],
              "temp_sol": canopy.enerbil.payload["temp_sol_new"],
              "temp_sol_pft": canopy.enerbil.payload["temp_sol_new_pft"]}
    matched = {n: np.asarray(v) for n,v in actual.items() if np.shape(v) == original.fields[n].shape}
    excluded = {n: dict(actual=list(np.shape(v)), schema=list(original.fields[n].shape))
                for n,v in actual.items() if n not in matched}
    state = SechibaRestartState({**original.fields, **matched}, original.provenance_by_field)
    destination = Path(directory) / "sechiba_restart.nc"
    write_sechiba_restart_state(destination, state=state, physical_state=read_restart_physical_state(source))
    restored = read_sechiba_restart_state(destination)
    initialized = read_sechiba_init_restart_state(destination, npts=1, nvm=14, nstm=6, nslm=11,
                                                pref_soil_veg=np.ones(14, dtype=np.int32))
    for name, value in matched.items():
        np.testing.assert_array_equal(restored.fields[name], value)
        # Initialization is allowed to update US/vegstress, not the other saved fields.
        if name not in ("us", "vegstress"):
            np.testing.assert_array_equal(initialized.fields[name], value)
    driver_source = PAPER / "driver_start.nc"
    old_driver = read_dim2_driver_restart(driver_source)
    new_driver = replace(old_driver, **{n: v for n,v in components["driver_previous_step_state"].items()
                                       if n in Dim2DriverRestart.__dataclass_fields__})
    driver_path = Path(directory) / "driver_restart.nc"
    write_dim2_driver_restart(driver_path, state=new_driver, physical_state=read_restart_physical_state(driver_source))
    wire = read_dim2_driver_restart(driver_path)
    np.testing.assert_array_equal(wire.vevapp, new_driver.vevapp)
    np.testing.assert_array_equal(restored.fields["evapora"], wire.vevapp * canopy.enerbil.payload["dt_sechiba"])
    return dict(overlaid_real_fields=sorted(matched), shape_exclusions=excluded,
                complete_file_fields=len(restored.fields), driver_rate=list(np.asarray(wire.vevapp)),
                internal_amount=list(np.asarray(restored.fields["evapora"])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--before", action="store_true", help="Capture expected baseline failures and stop before JIT/IO")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    result = dict(schema="hydrol-diffuco-real-handoff-v1", route_results=[], limits=[
        "Selected real module results; calendar admission, unrelated readiness and 93-field finalize generation isolated",
        "TOPMODEL enabled diagnostics is an independent chain on a real module, outside the default wrapper admitted domain",
        "First route omits full restart restoration; current shared writer executed, original fallback guard source reused",
        "Canopy source-shaped input is synthetic; not actual thermal/CONDVEG upstream provenance"])
    try:
        assert jax.config.jax_enable_x64
        assert all(d.platform == "cpu" for d in jax.devices())
        hydrol, enabled, canopy = real_components()
        result["runtime"] = dict(x64=bool(jax.config.jax_enable_x64), devices=[str(d) for d in jax.devices()])
        result["real_types"] = [type(x).__name__ for x in (hydrol, canopy.diffuco, canopy.enerbil)]
        result["fwet"] = dict(prior=[0.123], actual=list(np.asarray(enabled.diagnostics.fwet_new)),
                               wrapper_default_is_none=hydrol.diagnostics.fwet_new is None)
        for route in ROUTES:
            packet = carry(route, enabled, canopy)
            try:
                checked = check_selected(packet, enabled, canopy)
                result["route_results"].append(dict(route=route, status="pass", fields=len(checked)))
            except AssertionError as exc:
                result["route_results"].append(dict(route=route, status="fail", error=str(exc),
                    carried_fwet=list(np.asarray(packet.fields_by_component["hydrol_previous_step_state"]["fwet_new"]))))
        if not args.before:
            if any(r["status"] == "fail" for r in result["route_results"]):
                raise AssertionError("route invariant failure; saved actual result")
            values = compact_selected_jit(enabled, canopy)
            np.testing.assert_array_equal(values[0], enabled.diagnostics.fwet_new)
            packet = carry("runtime", enabled, canopy)
            next_call = next_diffuco_consumer(packet, canopy)
            result["next_diffuco_actual_type"] = type(next_call.diffuco).__name__
            result["next_gpp_pft14"] = float(np.asarray(next_call.diffuco.result.payload.payload["gpp"])[0,13])
            result["io"] = single_point_roundtrip(packet, canopy, args.output)
            result["selected_fast_writer_jit"] = "pass"
    except Exception as exc:
        result["fatal_error"] = repr(exc)
        raise
    finally:
        result["elapsed_seconds"] = time.perf_counter()-start
        result["bindings"] = [dict(path=name, sha256=hashlib.sha256((ROOT / name).read_bytes()).hexdigest())
                              for name in BOUND_SOURCES]
        (args.output / "receipt.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
