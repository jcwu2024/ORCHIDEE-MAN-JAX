"""Bounded creation/restore phase interfaces; no model timestep executes.

Source: hydrol_initialize 625-676, hydrol_var_init 4071-4088;
sechiba_initialize 599-729; Job0_bio 277-322. Local AST entry/exit
blocks below execute current production statements and identify every replaced
delegate. Numerical checks use real HYDROL initialization owners only.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import pickle
from dataclasses import make_dataclass
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
from netCDF4 import Dataset

from jax_orchidee.driver import orchestration as o
from jax_orchidee.driver import restart_state as rs
from jax_orchidee.driver.source_clock import SourceClockRuntime, source_clock_from_inputs
from jax_orchidee.sechiba import hydrol as h
from jax_orchidee.sechiba import thermosoil
from jax_orchidee.sechiba.restart_io import (
    DEFAULT_SECHIBA_RESTART_SCHEMA,
    SechibaRestartState,
    sechiba_finalize_source_state_from_restart,
)

ROOT = Path(__file__).resolve().parents[2]


def _function(name):
    return ast.parse(inspect.getsource(getattr(o, name))).body[0]


def _execute(nodes, environment):
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 "<current-production-phase-block>", "exec"), environment)


@pytest.mark.parametrize("prepared", [True, False])
def test_actual_cold_ready_exit_binds_source_context_before_extracting_state(monkeypatch, prepared):
    """Conditional exit reachability, not proof of complete physical cold closure."""
    owner = _function("paper_1961_driver_cold_start_first_step_coverage")
    branch = [node for node in owner.body if isinstance(node, ast.If)
              and ast.unparse(node.test) == "not any(missing.values())"][-1]
    setup = object()
    supplied = NS(source_clock_setup=setup, run_def_values={"PC_LIM": "73"}) if prepared else None
    calls = []

    def clock_setup(config_path, **kwargs):
        calls.append((config_path, kwargs))
        return setup

    def extract(payloads, **kwargs):
        context = kwargs["source_clock_context"]
        assert context.source_clock_setup is setup
        assert context.run_def_values["PC_LIM"] == "73"
        if prepared:
            assert context is supplied
        return "closed-local-exit"

    monkeypatch.setattr(o, "source_clock_setup_from_config", clock_setup)
    monkeypatch.setattr(o, "_previous_step_state_from_cold_start_payloads", extract)
    bundle = NS(year=1961)
    environment = {**vars(o), "missing": {}, "initialized_payloads": {},
                   "prepared_context": supplied, "config_path": "cold.yaml",
                   "run_def_path": Path("cold.def"), "run_def_values": {"PC_LIM": "73"},
                   "init_step": bundle, "step": bundle}
    _execute([branch], environment)
    payloads = environment["initialized_payloads"]
    assert payloads["cold_start_first_step_previous_state"] == "closed-local-exit"
    assert payloads["source_clock_context"].source_clock_setup is setup
    if not prepared:
        assert len(calls) == 1
        assert calls[0][1]["first_step_bundle"] is bundle
        assert calls[0][1]["run_def_values"] == {"PC_LIM": "73"}


def _tiny_canonical_restart():
    schema = json.loads(DEFAULT_SECHIBA_RESTART_SCHEMA.read_text(encoding="utf8"))
    sizes = {name: row["size"] for name, row in schema["dimensions"].items()}
    fields = {}
    for name, row in schema["variables"].items():
        if name in {"nav_lon", "nav_lat", "nav_lev", "time", "time_steps"}:
            continue
        axes = [axis for axis in row["dimensions"] if axis not in {"time", "x", "y"}]
        fields[name] = np.zeros((1, *(sizes[axis] for axis in reversed(axes))), dtype=row["dtype"])
    fields["njsc"][:] = 2.
    fields["veget_year"][:] = -1.
    assert len(fields) == 93
    return SechibaRestartState(fields, {})


def _write_refsoc(path, value):
    with Dataset(path, "w") as ds:
        for name, size in (("time", 1), ("y", 1), ("x", np.size(value))):
            ds.createDimension(name, size)
        if value is not None:
            ds.createVariable("refSOC_1d", "f8", ("time", "y", "x"))[:] = np.asarray(value)


def test_actual_file_first_thermal_entry_receives_optional_refsoc_before_main(monkeypatch, tmp_path):
    """Execute the current tstep=0 local prefix through its first thermal call."""
    path = tmp_path / "sechiba_start.nc"
    _write_refsoc(path, [.37])
    restart = NS(sechiba_start=path, sechiba_restart_state=_tiny_canonical_restart(),
                 driver_albedo=NS(z0=.12), stomate=NS(altmax=np.zeros((1, 14))))
    values = {"use_refSOC_hydrol": "TRUE"}
    context = NS(first_step_restart_state=restart, reference_run_dir=tmp_path, run_def_values=values)
    domain = NS(lalo=np.zeros((1, 2)), resolution=np.ones((1, 2)))
    step = NS(domain=domain)
    observed = []

    def thermal(source, **kwargs):
        np.testing.assert_array_equal(source["refSOC_1d"], [.37])
        observed.append(source)
        return {"phase_witness": np.array([71.])}

    monkeypatch.setattr(o, "paper_case_vertical_grids_from_used_run_def", lambda *a: ((), (), ()))
    monkeypatch.setattr(o, "paper_case_cwrr_vertical_soil_grid_from_used_run_def", lambda *a: NS())
    monkeypatch.setattr(o, "build_intersurf_first_step_payload", lambda *a, **kw: NS(kjpindex=1))
    monkeypatch.setattr(o, "restart_thermosoil_initialization_supported", lambda context: True)
    monkeypatch.setattr(o, "restart_thermosoil_initial_fields", thermal)
    owner = _function("paper_1961_driver_timestep_scaffold")
    branch = next(node for node in owner.body if isinstance(node, ast.If)
                  and ast.unparse(node.test) == "int(tstep) == 0")
    stop = next(index for index, node in enumerate(branch.body)
                if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                    and target.id == "driver_source" for target in node.targets))
    environment = {**vars(o), "context": context, "config_path": "case.yaml",
                   "run_def_values": values, "run_def_path": "used.def", "root": ROOT,
                   "runtime": NS(dt_sechiba=1800.), "step": step, "tstep": 0}
    _execute(branch.body[:stop], environment)
    assert len(observed) == 1
    assert environment["restart_thermal_fields"]["phase_witness"][0] == 71.
    assert "refSOC_1d" not in restart.sechiba_restart_state.fields


class _InitializationBoundaryReached(Exception):
    pass


def _hydro_restart_case(*, depth_on_context=True):
    znh = np.geomspace(.001, 2., 7)
    dnh = np.diff(np.r_[0., znh])
    dlh = np.r_[(dnh[:-1] + dnh[1:]) / 2., dnh[-1] / 2.]
    values = {"OK_EXPLICITSNOW": "TRUE", "USE_SOILC_TEMPDIFF": "TRUE", "use_refSOC": "TRUE",
              "use_refSOC_hydrol": "TRUE", "CWRR_NKS_N0": "1.07", "KFACT_DECAY_RATE": "3.0"}
    context = NS(run_def_values=values, hydrol_soil_parameters=h.hydrol_soil_parameters(),
                 hydrol_ok_pc=False, hydrol_ok_leak=True, hydrol_soil_peat_hydro=False,
                 hydrol_tides=False, hydrol_agri_peat=False, ok_freeze_cwrr=False,
                 hydrol_depth_max_h=2., hydrol_humcste=np.ones(2),
                 hydrol_dz_mm=dnh * 1000., hydrol_dh_mm=dlh * 1000., thermosoil_satsoil=False,
                 cwrr_grid=NS(znh=znh, dnh=dnh, dlh=dlh, znt=znh, zlt=znh, dz5=np.ones(7)),
                 run_scalars=NS(pref_soil_veg=np.array([1, 3])),
                 first_step_bundle=NS(vegetation=NS(soiltile=np.array([[.2, .3, .3]]))))
    if depth_on_context:
        # A recognizable token permits a pre-interface baseline forwarding check.
        context.hydrol_depth_parameters = NS(marker="context-depth")
    source = dict(veget=np.array([[.2, .6]]), veget_max=np.array([[.2, .6]]),
                  njsc=np.array([2]), mc=np.full((1, 7, 3), .22), mcl=np.full((1, 7, 3), .22),
                  water2infilt=np.zeros((1, 3)), resdist=np.array([[.2, .3, .3]]),
                  vegtot_old=np.array([.8]), qsintveg=np.zeros((1, 2)), drysoil_frac=np.zeros(1),
                  refSOC_1d=np.array([h.SOILC_MAX / 2.]))
    return context, source


def test_actual_restart_initialization_forwards_the_prepared_depth_contract(monkeypatch):
    context, source = _hydro_restart_case()

    def inspect_hydrol(**kwargs):
        assert kwargs["depth_parameters"] is context.hydrol_depth_parameters
        raise _InitializationBoundaryReached

    monkeypatch.setattr(h, "hydrol_var_init", inspect_hydrol)
    with pytest.raises(_InitializationBoundaryReached):
        rs.restart_thermosoil_initial_fields(source, context=context, payload=NS(), altmax=np.zeros((1, 2)))


@pytest.mark.parametrize("depth_on_context", [True, False])
def test_actual_restart_init_depth_refsoc_and_mineral_background_reach_real_local_owners(monkeypatch, depth_on_context):
    context, source = _hydro_restart_case(depth_on_context=depth_on_context)
    expected_depth = h.hydrol_depth_parameters_from_run_def(context.run_def_values)
    if depth_on_context:
        context.hydrol_depth_parameters = expected_depth
    observed = {}
    original_hydro = h.hydrol_var_init
    original_moisture = h.hydrol_cold_start_thermosoil_moisture_inputs

    def hydro(**kwargs):
        observed["hydro_kwargs"] = kwargs
        observed["hydro"] = original_hydro(**kwargs)
        return observed["hydro"]

    def moisture(*args, **kwargs):
        observed["moisture_kwargs"] = kwargs
        observed["moisture"] = original_moisture(*args, **kwargs)
        return observed["moisture"]

    def stop_before_thermal(**kwargs):
        raise _InitializationBoundaryReached

    monkeypatch.setattr(h, "hydrol_var_init", hydro)
    monkeypatch.setattr(h, "hydrol_cold_start_thermosoil_moisture_inputs", moisture)
    monkeypatch.setattr(thermosoil, "thermosoil_humlev", stop_before_thermal)
    with pytest.raises(_InitializationBoundaryReached):
        rs.restart_thermosoil_initial_fields(source, context=context, payload=NS(), altmax=np.zeros((1, 2)))
    assert observed["hydro_kwargs"]["depth_parameters"].as_config_values() == expected_depth.as_config_values()
    assert observed["hydro_kwargs"]["refSOC_1d"] is source["refSOC_1d"]
    soil = context.hydrol_soil_parameters
    np.testing.assert_allclose(observed["hydro"].mcs,
                               [.5 * h.POROS_ORG + .5 * float(soil.mcs_mineral[1])], rtol=1.e-13)
    np.testing.assert_array_equal(observed["moisture_kwargs"]["mcs_mineral"], soil.mcs_mineral)
    expected_tables = h.build_mineral_cwrr_tables(
        njsc=source["njsc"], mcs=observed["hydro"].mcs, z_m=context.cwrr_grid.znh,
        mcr=soil.mcr, nvan=soil.nvan, avan=soil.avan, ks=soil.ks, **expected_depth.table_kwargs())
    np.testing.assert_array_equal(observed["hydro"].mineral_tables.nfact, expected_tables.nfact)


@pytest.mark.parametrize("file_values,fallback", [([.37], False), ([999999.], True), (None, True)])
def test_real_refsoc_file_initialization_uses_source_all_missing_rule(monkeypatch, tmp_path, file_values, fallback):
    path = tmp_path / "sechiba.nc"
    _write_refsoc(path, file_values)
    calls = []

    def static_reader(config_path, **kwargs):
        calls.append((config_path, kwargs))
        return np.array([.83]), {}

    monkeypatch.setattr("jax_orchidee.driver.static.read_paper_hydrol_refsoc_1d_static_field", static_reader)
    result = rs.resolve_hydrol_initialization_refsoc_1d(
        "case.yaml", run_def_values={"use_refSOC_hydrol": "TRUE"}, npts=1,
        restart_path=path, domain=NS(lalo=np.zeros((1, 2)), resolution=np.ones((1, 2))))
    np.testing.assert_array_equal(result, [.83] if fallback else [.37])
    assert len(calls) == int(fallback)


def test_refsoc_partial_missing_and_restored_mirror_never_trigger_static_read(monkeypatch, tmp_path):
    path = tmp_path / "sechiba.nc"
    _write_refsoc(path, [999999., .27])

    def forbidden(*args, **kwargs):
        pytest.fail("partial sentinel or explicit restored values must not use static fallback")

    monkeypatch.setattr("jax_orchidee.driver.static.read_paper_hydrol_refsoc_1d_static_field", forbidden)
    partial = rs.resolve_hydrol_initialization_refsoc_1d(
        "case.yaml", run_def_values={"use_refSOC_hydrol": "TRUE"}, npts=2, restart_path=path)
    np.testing.assert_array_equal(partial, [999999., .27])
    supplied = np.array([.41, .53])
    restored = rs.resolve_hydrol_initialization_refsoc_1d(
        "case.yaml", run_def_values={"use_refSOC_hydrol": "TRUE"}, npts=2,
        restored_refSOC_1d=supplied, restart_path=tmp_path / "not-opened.nc")
    np.testing.assert_array_equal(restored, supplied)


def test_disabled_refsoc_initialization_does_not_read_any_file():
    assert rs.resolve_hydrol_initialization_refsoc_1d(
        "not-opened.yaml", run_def_values={}, npts=1, restart_path="not-opened.nc") is None


def test_refsoc_all_missing_file_reaches_actual_small_static_reader(tmp_path):
    restart_path = tmp_path / "sechiba_start.nc"
    _write_refsoc(restart_path, [999999.])
    static_path = tmp_path / "refsoc_static.nc"
    with Dataset(static_path, "w") as dataset:
        dataset.createDimension("latitude", 2)
        dataset.createDimension("longitude", 2)
        dataset.createVariable("latitude", "f8", ("latitude",))[:] = [0., 1.]
        dataset.createVariable("longitude", "f8", ("longitude",))[:] = [0., 1.]
        dataset.createVariable("mask", "f8", ("latitude", "longitude"))[:] = 1.
        dataset.createVariable("soil_organic_carbon_1d", "f8", ("latitude", "longitude"))[:] = .4
    config_path = tmp_path / "case.yaml"
    config_path.write_text("static_inputs:\n  refsoc_1d_file: '" + static_path.as_posix() + "'\n", encoding="utf8")
    actual = rs.resolve_hydrol_initialization_refsoc_1d(
        config_path, run_def_values={"use_refSOC_hydrol": "TRUE"}, npts=1, restart_path=restart_path,
        domain=NS(lalo=np.array([[.5, .5]]), resolution=np.full((1, 2), 10000.)))
    np.testing.assert_allclose(actual, [.4], rtol=1.e-13)


def test_enabled_refsoc_without_restart_or_static_geometry_rejects_before_consumer():
    with pytest.raises(ValueError, match="domain/resolution"):
        rs.resolve_hydrol_initialization_refsoc_1d(
            "not-opened.yaml", run_def_values={"use_refSOC_hydrol": "TRUE"}, npts=1)


@pytest.mark.parametrize("value,npts,match", [([1., 2.], 1, "shape"), ([float("nan")], 1, "finite")])
def test_refsoc_input_validation_precedes_first_consumer(value, npts, match):
    with pytest.raises(ValueError, match=match):
        rs.resolve_hydrol_initialization_refsoc_1d(
            "case.yaml", run_def_values={"use_refSOC_hydrol": "TRUE"}, npts=npts,
            restored_refSOC_1d=np.array(value))


def test_complete_checkpoint_decode_keeps_initialized_values_and_legacy_class(tmp_path, monkeypatch):
    from jax_orchidee.runners.multiyear import _read_year_end_state_cache

    source = sechiba_finalize_source_state_from_restart(_tiny_canonical_restart())
    assert len(source) == 93
    source.update(us=np.full_like(source["us"], 999999.), vegstress=np.full_like(source["vegstress"], .219),
                  growth_day=np.full_like(source["growth_day"], 73.),
                  peatC_ok=np.full_like(source["peatC_ok"], True), refSOC_1d=np.array([.37]))
    components = {
        "sechiba_finalize_state": source,
        "thermosoil_previous_step_state": {"temp_sol_beg": np.array([278.25]), "pcapa_en": np.array([[[1234.]]])},
        "slowproc_stomate_previous_step_state": {"uo_0": np.array([[.73]])},
    }
    _, setup = _clock_setup(tmp_path, "new_native_run")
    state = o.DriverPreviousStepStatePacket(
        tstep=47, fields_by_component=components, provenance_by_component={}, source_clock_runtime=setup.runtime)
    path = tmp_path / "complete-checkpoint.pkl"
    with path.open("wb") as stream:
        pickle.dump({"state": state, "phase": "already_initialized"}, stream)
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    def forbidden(*args, **kwargs):
        pytest.fail("complete checkpoint decode must not replay initialization")

    for module, name in ((h, "hydrol_init_us_state"), (h, "hydrol_init_vegstress"),
                         (rs, "restart_thermosoil_initial_fields")):
        monkeypatch.setattr(module, name, forbidden)
    restored = _read_year_end_state_cache(path)["state"]
    assert type(restored) is o.DriverPreviousStepStatePacket
    assert type(restored).__module__ == "jax_orchidee.driver.orchestration"
    assert restored.tstep == 47
    for component, fields in components.items():
        for name, value in fields.items():
            np.testing.assert_array_equal(restored.fields_by_component[component][name], value)
    assert type(restored.source_clock_runtime) is type(setup.runtime)
    assert restored.source_clock_runtime.binding == setup.runtime.binding
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def _clock_setup(tmp_path, mode):
    for year in (1961, 1962):
        with Dataset(tmp_path / f"forcing-{year}.nc", "w") as dataset:
            dataset.createDimension("time", 3)
            time = dataset.createVariable("time", "f8", ("time",))
            time.units = f"seconds since {year}-01-01 00:00:00"
            time.calendar = "gregorian"
            time[:] = [0., 1800., 3600.]
    old = source_clock_from_inputs(tmp_path / "forcing-1961.nc", dt_sechiba=1800.,
                                   run_id="old-native", time_length=1)
    previous = SourceClockRuntime(old.binding, np.int64(17520), np.int64(17519), np.int64(17520))
    setup = source_clock_from_inputs(
        tmp_path / "forcing-1962.nc", dt_sechiba=1800., time_length=1,
        run_id="old-native" if mode == "continuous" else "new-native", run_mode=mode,
        previous_runtime=previous if mode == "continuous" else None)
    return previous, setup


@pytest.mark.parametrize("mode", ["new_native_run", "continuous"])
@pytest.mark.parametrize("all_missing_stress", [False, True])
def test_actual_year_entry_initializes_only_new_native_run_and_retains_gas_state(monkeypatch, tmp_path, mode, all_missing_stress):
    _exercise_actual_year_entry_phase(monkeypatch, tmp_path, mode, all_missing_stress)


def test_disabled_refsoc_new_native_entry_discards_stale_mirrors_before_default_export(monkeypatch, tmp_path):
    _exercise_actual_year_entry_phase(monkeypatch, tmp_path, "new_native_run", False, refsoc_enabled=False)


def _exercise_actual_year_entry_phase(monkeypatch, tmp_path, mode, all_missing_stress, *, refsoc_enabled=True):
    previous_runtime, setup = _clock_setup(tmp_path, mode)
    us = np.full((1, 14, 6, 11), .03125)
    stress_value = 999999. if all_missing_stress else .21875
    source = dict(us=us, vegstress=np.full((1, 14), stress_value), njsc=np.array([2.]),
                  temp_sol=np.array([278.25]))
    refsoc = np.array([.37])
    if not refsoc_enabled:
        source = {**sechiba_finalize_source_state_from_restart(_tiny_canonical_restart()),
                  **source, "refSOC_1d": refsoc}
    gas = np.full((1, 2, 14), .731)
    nroot = np.full((1, 14, 11), .071)
    packet = o.DriverPreviousStepStatePacket(
        tstep=17519, fields_by_component={
            "sechiba_finalize_state": source,
            "hydrol_previous_step_state": {"nroot": nroot, "refSOC_1d": refsoc,
                                            "temp_hydro": np.full((1, 11), 279.5)},
            "slowproc_stomate_previous_step_state": {"altmax": np.zeros((1, 14)), "uo_0": gas},
            "thermosoil_previous_step_state": {"temp_sol_beg": np.array([277.]), "pcapa_en": np.array([[[900.]]])},
            "driver_previous_step_state": {"z0": np.array([.12])},
        }, provenance_by_component={}, source_clock_runtime=previous_runtime)
    values = dict(OK_EXPLICITSNOW="TRUE", USE_SOILC_TEMPDIFF="TRUE", use_refSOC="TRUE",
                  use_refSOC_hydrol="TRUE" if refsoc_enabled else "FALSE")
    context_values = dict(config_path=tmp_path / "case.yaml", run_def_path=tmp_path / "used.def",
                          run_def_values=values, first_step_bundle=NS(), reference_run_dir=None,
                          first_step_restart_state=None, source_clock_setup=setup,
                          run_scalars=NS(pref_soil_veg=np.array([1] + [3] * 12 + [4])),
                          runtime=NS(dt_stomate=86400., dt_sechiba=1800.), dt_sechiba=1800.,
                          hydrol_ok_pc=False, hydrol_ok_leak=True)
    context_type = make_dataclass("LocalPhaseContext", [(name, object) for name in context_values])
    context = context_type(**context_values)
    events = []
    original_us = h.hydrol_init_us_state
    original_stress = h.hydrol_init_vegstress

    def us_init(**kwargs):
        events.append("US_INIT")
        return original_us(**kwargs)

    def stress_init(**kwargs):
        events.append("vegstress_initialize")
        return original_stress(**kwargs)

    def thermal(source, **kwargs):
        events.append("thermal_initialize")
        if refsoc_enabled:
            np.testing.assert_array_equal(source["refSOC_1d"], refsoc)
        else:
            assert "refSOC_1d" not in source
        assert source["temp_sol"] is packet.fields_by_component["sechiba_finalize_state"]["temp_sol"]
        return {"temp_sol_beg": source["temp_sol"], "pcapa_en": np.array([[[1200.]]])}

    def runtime(*args, **kwargs):
        events.append("runtime_boundary")
        return kwargs["previous_state"]

    monkeypatch.setattr(h, "hydrol_init_us_state", us_init)
    monkeypatch.setattr(h, "hydrol_init_vegstress", stress_init)
    monkeypatch.setattr(o, "restart_thermosoil_initial_fields", thermal)
    monkeypatch.setattr(o, "paper_1961_driver_later_day_runtime_result", runtime)
    monkeypatch.setattr(o, "driver_year_handoff_state_gaps", lambda packet: ())
    monkeypatch.setattr(o, "source_clock_setup_from_config", lambda *args, **kwargs: setup)
    monkeypatch.setattr(o, "_require_source_day_alignment", lambda *args, **kwargs: None)
    monkeypatch.setattr(o, "_paper_1961_step_bundle_from_context", lambda *args, **kwargs: NS(
        domain=NS(lalo=np.zeros((1, 2)), resolution=np.ones((1, 2)))))
    monkeypatch.setattr(o, "build_intersurf_first_step_payload", lambda *args, **kwargs: NS(kjpindex=1))
    result = o.paper_1961_driver_restart_year_start_day_result(
        "case.yaml", previous_year_end_state=packet, year=1962, prepared_context=context, run_mode=mode)
    assert events == (["US_INIT", "vegstress_initialize", "thermal_initialize", "runtime_boundary"]
                      if mode == "new_native_run" else ["runtime_boundary"])
    assert result.tstep == -1 and packet.tstep == 17519
    np.testing.assert_array_equal(source["vegstress"], stress_value)
    np.testing.assert_array_equal(result.fields_by_component["slowproc_stomate_previous_step_state"]["uo_0"], gas)
    if refsoc_enabled:
        np.testing.assert_array_equal(result.fields_by_component["hydrol_previous_step_state"]["refSOC_1d"], refsoc)
    else:
        for component in ("hydrol_previous_step_state", "sechiba_finalize_state"):
            assert "refSOC_1d" not in result.fields_by_component[component]
            np.testing.assert_array_equal(packet.fields_by_component[component]["refSOC_1d"], refsoc)
        from jax_orchidee.sechiba.restart_export import paper_sechiba_restart_state_from_finalize_state
        exported = paper_sechiba_restart_state_from_finalize_state(
            result.fields_by_component["sechiba_finalize_state"], kjit=0, use_refsoc_hydrol=False,
        )
        assert len(exported.fields) == 93
    if mode == "continuous":
        assert result.fields_by_component["hydrol_previous_step_state"]["nroot"] is nroot
        np.testing.assert_array_equal(result.fields_by_component["thermosoil_previous_step_state"]["pcapa_en"], [[[900.]]])
        np.testing.assert_array_equal(result.fields_by_component["sechiba_finalize_state"]["vegstress"], stress_value)
    else:
        assert "nroot" not in result.fields_by_component["hydrol_previous_step_state"]
        expected_stress = stress_value + 11 * .03125 if all_missing_stress else stress_value
        np.testing.assert_array_equal(result.fields_by_component["sechiba_finalize_state"]["vegstress"], expected_stress)
        np.testing.assert_array_equal(result.fields_by_component["thermosoil_previous_step_state"]["pcapa_en"], [[[1200.]]])


def test_actual_same_year_continuation_enters_runtime_binding_without_initialization(monkeypatch, tmp_path):
    _, setup = _clock_setup(tmp_path, "new_native_run")
    packet = o.DriverPreviousStepStatePacket(
        tstep=47, fields_by_component={"hydrol_previous_step_state": {"refSOC_1d": np.array([.37])}},
        provenance_by_component={}, source_clock_runtime=setup.runtime)
    context = NS(runtime=NS(dt_stomate=86400., dt_sechiba=1800.), run_def_path="used.def",
                 run_def_values={}, cwrr_grid=NS(), source_clock_setup=setup)

    def forbidden(*args, **kwargs):
        pytest.fail("same-year continuation must not replay initialization")

    def runtime_binding(**kwargs):
        assert kwargs["current_state"] is packet
        assert kwargs["context"] is context
        np.testing.assert_array_equal(kwargs["current_state"].fields_by_component["hydrol_previous_step_state"]["refSOC_1d"], [.37])
        raise _InitializationBoundaryReached

    monkeypatch.setattr(h, "hydrol_init_us_state", forbidden)
    monkeypatch.setattr(h, "hydrol_init_vegstress", forbidden)
    monkeypatch.setattr(o, "restart_thermosoil_initial_fields", forbidden)
    monkeypatch.setattr(o, "reference_case_first_step_restart_state", forbidden)
    monkeypatch.setattr(o, "_require_source_day_alignment", lambda *args, **kwargs: None)
    monkeypatch.setattr(o, "_paper_1961_later_day_transition_inputs", runtime_binding)
    with pytest.raises(_InitializationBoundaryReached):
        o.paper_1961_driver_later_day_runtime_result(
            "case.yaml", previous_state=packet, prepared_context=context,
            day_index=2, start_tstep=48, year=1962)
