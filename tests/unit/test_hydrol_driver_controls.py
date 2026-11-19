"""Real driver tide adapters, stopped at the HYDROL boundary, never a model step.

The source clock reads a tiny real NetCDF file; tides read versioned compact
paper records materialized as real text files. Physics initialization/DIFFUCO/ENERBIL and weather slabs
are explicitly substituted. Production prepare, adapter bodies, tide lookup,
compiled forcing construction, HYDROL assembler and incomplete owner guard
are real. No physical process transition is executed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, make_dataclass, replace
from pathlib import Path
from types import SimpleNamespace as NS

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import yaml
from netCDF4 import Dataset

from jax_orchidee.driver import orchestration as driver
from jax_orchidee.driver.domain import load_case_config
from jax_orchidee.driver.source_clock import (
    SourceClockRuntime,
    rebase_source_clock_runtime,
    source_clock_from_inputs,
    source_clock_global_attributes,
    source_clock_variable_attributes,
)
from jax_orchidee.driver.tides import TideForcingError, TideForcingValues
from jax_orchidee.sechiba import hydrol

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/orchidee_man_250919.yaml"
FIXTURE = json.loads((ROOT / "tests/fixtures/hydrol_driver_boundary.json").read_text())


@pytest.fixture(autouse=True)
def _binary64_boundary():
    with jax.enable_x64():
        yield


@dataclass(frozen=True)
class _Context:
    """Only unrelated prepared science constants are replaced by this fixture."""
    config_path: Path
    run_def_path: Path
    run_def_values: dict
    first_step_bundle: object
    extras: dict
    hydrol_tides: bool = True
    hydrol_check_cwrr: bool = False
    source_clock_setup: object = None
    hydrol_tide_run: object = None
    watchout_initialize_main_tstep: object = None
    diffuco_pft14_static_kwargs: object = None

    def __post_init__(self):
        if self.diffuco_pft14_static_kwargs is None:
            object.__setattr__(self, "diffuco_pft14_static_kwargs", self.extras.get("diffuco_pft14_static_kwargs", {}))

    def __getattr__(self, name):
        try:
            return self.extras[name]
        except KeyError:
            raise AttributeError(name) from None


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class _ProbeScanState:
    """Tiny source-clock carry; contains no model state or physical transition."""
    tstep: object
    source_clock_runtime: object
    value: object

    def tree_flatten(self):
        return (self.tstep, self.source_clock_runtime, self.value), None

    @classmethod
    def tree_unflatten(cls, _, children):
        return cls(*children)


def _set(values, **overrides):
    result = dict(values)
    for name, value in overrides.items():
        result = {key: item for key, item in result.items() if key.upper() != name.upper()}
        result[name] = str(value)
    return result


def _payload():
    vector = np.asarray([1.0])
    result = dict(kjpindex=1, nbindex=1, is_watchout=False,
                  lalo=np.asarray([[21.0, 109.0]]), neighbours=np.zeros((1, 8), dtype=int),
                  resolution=np.asarray([[200000.0, 200000.0]]),
                  salinity=vector * 20, tide_height=vector * 100,
                  soilclass=vector, clay_frac=vector * .2, sand_frac=vector * .5,
                  silt_frac=vector * .3, bulk_dens=vector * 1200,
                  soil_ph=vector * 6, poor_soils=vector * 0,
                  njsc=np.asarray([2], dtype=np.int32),
                  u=vector * 2, v=vector, zlev=vector * 10, zlevuv=vector * 10,
                  temp_air=vector * 280, qair=vector * .005, pb=vector * 1013,
                  swdown=vector * 100, lwdown=vector * 300, ccanopy=vector * 320,
                  precip_rain=vector * .1, precip_snow=vector * 0)
    return NS(**result)


def _grid():
    return NS(**{name: np.asarray([.01, .1, 1.]) for name in
                 ("znh", "dnh", "dlh", "znt", "zlt", "dlt", "dz1", "dz5")})


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    # Atmospheric metadata is real IO; only72 records and no spatial payload.
    weather = tmp_path / "weather-1961.nc"
    with Dataset(weather, "w") as dataset:
        dataset.createDimension("time", 72)
        time = dataset.createVariable("time", "f8", ("time",))
        time.units = "seconds since 1961-01-01 00:00:00"
        time.calendar = "noleap"
        time[:] = np.arange(72) * 3600.0
    config = load_case_config(CONFIG)
    config["drivers"]["atmospheric_forcing"]["file_pattern"] = str(tmp_path / "weather-{year}.nc")
    for key, records in (("positive_file", FIXTURE["positive_records"]),
                         ("differential_file", FIXTURE["differential_records"])):
        path = tmp_path / ("WTM_forcing_pos.txt" if key == "positive_file" else "WTM_forcing_diff.txt")
        path.write_text("\n".join(records) + "\n", encoding="ascii")
        config["drivers"]["water_table_text"][key] = str(path)
    config_path = tmp_path / "case.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf8")
    values = _set(FIXTURE["effective_run_def"], RESTART_FILEIN="NONE", SECHIBA_restart_in="NONE",
                  STOMATE_RESTART_FILEIN="NONE", TIME_SKIP="0", TIME_LENGTH="1D", CHECK_CWRR="FALSE", TIDES="TRUE",
                  READ_REFTEMP="FALSE", use_refSOC_hydrol="FALSE", ALB_BG_MODIS="FALSE")
    run_def_path = tmp_path / "used_run.def"
    run_def_path.write_text("\n".join(f"{key}={value}" for key, value in values.items()), encoding="utf8")
    # Preserve the real used log's empty filenames. Originating run.def is
    # explicit authority, exactly as the historical sibling file provides.
    origin = _set(values, **FIXTURE["originating_tide_declarations"])
    (tmp_path / "run.def").write_text("\n".join(f"{key}={value}" for key, value in origin.items()), encoding="utf8")
    # A single PFT14 axis is an unrelated shape fixture, not an architecture change.
    run = NS(nvm=1, nstm=6, fortran_pft_ids=(14,), pref_soil_veg=np.asarray([4]),
             impose_veg=True, pft_layout=NS(index_for_id=lambda _: 0,
                                          entries=(NS(fortran_pft_id=14),)))
    monkeypatch.setattr(driver, "read_run_scalars", lambda *a, **k: run)
    payload = _payload()
    weather_slab = NS(**{**vars(payload), "pb": payload.pb * 100, "dt_force": 3600.0})
    domain = NS(nbindex=1, lalo=payload.lalo, resolution=payload.resolution)
    first = NS(year=1961, tstep=0, domain=domain, run_scalars=run,
               forcing=weather_slab, static_trace_fields=NS(salinity=payload.salinity, tide_height=payload.tide_height))
    nvm = run.nvm
    extras = dict(reference_run_dir=None, runtime=NS(dt_sechiba=1800., dt_stomate=86400.),
                  dt_sechiba=1800., dt_sechiba_days=1 / 48, forcing_split=2, forcing_nb_spread=2,
                  nvm=nvm, run_scalars=run, cwrr_grid=_grid(), diaglev=np.asarray([.01, .1, 1.]),
                  zlt=_grid().zlt, znt=_grid().znt,
                  hydrol_soil_parameters=hydrol.hydrol_soil_parameters_from_run_def(values),
                  hydrol_humcste=np.ones(nvm), ext_coeff_vegetfrac=np.ones(nvm),
                  hydrol_throughfall_by_pft=np.zeros(nvm), hydrol_zz_mm=np.asarray([10., 100., 1000.]),
                  hydrol_dh_mm=np.asarray([10., 90., 900.]), hydrol_dz_mm=np.asarray([0., 90., 900.]),
                  hydrol_soil_peat_hydro=True, stomate_ok_dgvm=False, ok_explicitsnow=True,
                  ok_freeze_cwrr=True, ok_thermodynamical_freezing=True,
                  hydrol_fr_dt=2., hydrol_froz_frac_corr=1., hydrol_smtot_corr=2., hydrol_max_froz_hydro=1.,
                  ok_laidev=np.zeros(nvm, dtype=bool), min_wind=.1, river_routing=False,
                  domain_override=None, diffuco_pft14_static_kwargs={}, control_salinity_min=0.,
                  hydrol_cwrr_ks=np.ones(13), hydrol_reinf_slope=(0.,), hydrol_depth_max_h=2.,
                  hydrol_do_ponds=False, hydrol_ok_pc=False, hydrol_ok_leak=True,
                  hydrol_ok_ru2peat=True, hydrol_ok_wt_ab=True, hydrol_agri_peat=False,
                  hydrol_max_wt_ab=100., hydrol_dyn_nroot_larix=False)
    context = _Context(config_path, run_def_path, values, first, extras)
    # All scientific prepared construction is bypassed. The public prepare
    # function, effective switch gate, source clock factory and tide binder run.
    monkeypatch.setattr(driver, "_prepare_paper_1961_driver_context_cached", lambda *args: context)
    result = driver.prepare_paper_1961_driver_context(config_path, used_run_def_path=run_def_path,
                                                       run_id="adapter-cold")
    return NS(context=result, unbound=context, payload=payload, first=first,
              weather=weather, weather_slab=weather_slab, config=config, values=values)


class _ReachedHydrolBoundary(Exception):
    def __init__(self, received, assembly, closure):
        self.received = received
        self.assembly = assembly
        self.closure = closure


def _stop_at_real_assembler(monkeypatch):
    def boundary(**kwargs):
        # Preserve all three new production controls. Unrelated prognostic
        # prerequisites remain deliberately absent: this cannot run a model.
        real_kwargs = dict(kwargs)
        real_kwargs.update(enerbil_payload={}, hydrol_restart=NS(), slowproc_restart=NS(),
                           diffuco_payload={}, thermosoil_restart=None,
                           static_payload_template={"adapter_probe": np.asarray(0.)},
                           initialized_us=None, use_jit=False)
        assembly = hydrol.assemble_hydrol_first_step_precall_payload(**real_kwargs)
        closure = hydrol.run_hydrol_first_step_module_from_precall(
            assembly, humcste=np.ones(1), dz_mm=np.ones(3), dh_mm=np.ones(3),
            zz_mm=np.ones(3), tides=True)
        assert not assembly.ok and closure.module is None
        raise _ReachedHydrolBoundary(kwargs, assembly, closure)
    monkeypatch.setattr(driver, "assemble_hydrol_first_step_precall_payload", boundary)


def _assert_boundary(caught, expected):
    probe = caught.value
    assert (probe.received["wtp_tide"], probe.received["dwtp_tide"]) == expected
    assert (probe.assembly.payload["wtp_tide"], probe.assembly.payload["dwtp_tide"]) == expected
    assert probe.assembly.payload["check_cwrr"] is False
    assert {"wtp_tide", "dwtp_tide", "check_cwrr"} <= set(probe.assembly.driver_inputs)


def _previous(context, tstep=0):
    return NS(tstep=tstep, source_clock_runtime=context.source_clock_setup.runtime,
              fields_by_component={"slowproc_stomate_previous_step_state": {"lai": np.zeros((1, context.nvm))},
                                   "thermosoil_previous_step_state": {}})


def test_public_prepare_real_clock_and_assets_bind_first_record(prepared):
    context = prepared.context
    assert context.hydrol_tides and not context.hydrol_check_cwrr
    assert context.source_clock_setup.forcing.count == 72
    assert driver._driver_hydrol_controls(context, tstep=0) == dict(check_cwrr=False, wtp_tide=308.33, dwtp_tide=44.83)


def test_ordinary_entry_forwards_second_source_record_to_real_assembler_owner(prepared, monkeypatch):
    context = prepared.context
    payload = prepared.payload
    energetic = NS(enerbil_local=NS(payload={}), missing_previous_state_fields=(), step=prepared.first,
                   payload=payload, diffuco_precall=NS(payload={}),
                   diffuco_local_enerbil_precall=NS(result=NS(payload=NS(payload={}))))
    monkeypatch.setattr(driver, "paper_1961_next_step_enerbil_precall_scaffold", lambda *a, **k: energetic)
    monkeypatch.setattr(driver, "_hydrol_state_namespace_from_previous_state", lambda *a: (NS(), ()))
    _stop_at_real_assembler(monkeypatch)
    with pytest.raises(_ReachedHydrolBoundary) as caught:
        driver.paper_1961_next_step_hydrol_precall_scaffold(
            context.config_path, previous_state=_previous(context), tstep=1,
            prepared_context=context, hydrol_static_template={"probe": 0.}, module_jit=False)
    _assert_boundary(caught, (329., 20.67))


def _weather_stubs(monkeypatch, prepared):
    # Meteorological values are unrelated to READ103/104; their host producer
    # is exercised with an explicit tiny slab instead of a full annual file.
    monkeypatch.setattr(driver, "read_forcing_model_step_cached", lambda *a, **k: prepared.weather_slab)
    monkeypatch.setattr(driver, "read_annual_co2", lambda *a: 317.27)


def test_real_compiled_forcing_producer_has_validated_dynamic_tide_leaves(prepared, monkeypatch):
    context = prepared.context
    _weather_stubs(monkeypatch, prepared)
    forcing = driver._paper_compiled_forcing_day(context, year=1961, start_tstep=0, steps_per_stomate=2)
    np.testing.assert_array_equal(forcing.hydrol_tide_forcing.wtp_tide, [308.33, 329.])
    np.testing.assert_array_equal(forcing.hydrol_tide_forcing.dwtp_tide, [44.83, 20.67])
    previous = _previous(context, tstep=-1)
    driver._validate_compiled_source_clock_series(context, previous, forcing)
    # Same actual driver control owner and real HYDROL assembler under JIT;
    # deliberately incomplete physics fields prevent any full transition.
    def boundary(clock, pair):
        controls = driver._driver_hydrol_controls(context, tstep=clock.local_tstep,
                                                 source_clock=clock, tide_values=pair)
        assembly = hydrol.assemble_hydrol_first_step_precall_payload(
            **controls, enerbil_payload={}, hydrol_restart=NS(), slowproc_restart=NS(),
            pref_soil_veg=np.asarray([4]), nstm=6, ext_coeff_vegetfrac=np.asarray([1.]),
            static_payload_template={"adapter_probe": 0.})
        return jnp.stack([assembly.payload["wtp_tide"], assembly.payload["dwtp_tide"]])
    with jax.enable_x64():
        actual = jax.jit(jax.vmap(boundary))(forcing.source_clock, forcing.hydrol_tide_forcing)
    np.testing.assert_array_equal(actual, [[308.33, 44.83], [329., 20.67]])
    bad = forcing._replace(hydrol_tide_forcing=forcing.hydrol_tide_forcing._replace(dwtp_tide=np.zeros(2)))
    with pytest.raises(TideForcingError, match="disagree"):
        driver._validate_compiled_source_clock_series(context, previous, bad)
    with pytest.raises(TideForcingError, match="lacks bound"):
        driver._validate_compiled_source_clock_series(context, previous, forcing._replace(hydrol_tide_forcing=None))


def test_restart_origin_is_explicit_and_local_rebase_does_not_reset_it(prepared):
    context = prepared.context
    setup = context.source_clock_setup
    # DIM2 restart admission is at a completed forcing slab (split2).
    consumed = replace(setup.runtime, last_applied_istp=np.int64(2))
    rebased = rebase_source_clock_runtime(consumed, new_local_tstep=-1)
    continuous = source_clock_from_inputs(prepared.weather, dt_sechiba=1800., run_id=setup.binding.run_id,
                                         run_mode="continuous", previous_runtime=rebased, time_length="1D")
    with pytest.raises(TideForcingError, match="original TideForcingRun"):
        driver._bind_context_hydrol_tides(context, continuous, run_mode="continuous")
    resumed = driver._bind_context_hydrol_tides(context, continuous, previous=context.hydrol_tide_run, run_mode="continuous")
    assert resumed.hydrol_tide_run is context.hydrol_tide_run
    assert driver._driver_hydrol_controls(resumed, tstep=0)["wtp_tide"] == 349.67
    assert resumed.hydrol_tide_run.first_source_istp == 1
    fresh = source_clock_from_inputs(prepared.weather, dt_sechiba=1800., run_id="next-native-year",
                                    time_skip="1D", time_length="1D")
    restarted = driver._bind_context_hydrol_tides(context, fresh)
    assert fresh.time_control.istp == 49
    assert driver._driver_hydrol_controls(restarted, tstep=0)["wtp_tide"] == 308.33
    assert driver._driver_hydrol_controls(restarted, tstep=0)["dwtp_tide"] == 44.83


def test_actual_entry_rejects_missing_stream_and_eof(prepared, monkeypatch):
    context = prepared.context
    with pytest.raises(TideForcingError, match="bound native-run"):
        driver._driver_hydrol_controls(replace(context, hydrol_tide_run=None), tstep=0)
    with pytest.raises(TideForcingError, match="outside available"):
        driver._driver_hydrol_controls(context, tstep=4)
    config = dict(prepared.config)
    config["drivers"] = dict(config["drivers"])
    config["drivers"]["water_table_text"] = dict(config["drivers"]["water_table_text"])
    config["drivers"]["water_table_text"]["positive_file"] = str(context.config_path.parent / "missing/WTM_forcing_pos.txt")
    context.config_path.write_text(yaml.safe_dump(config), encoding="utf8")
    # Invalidate only this test's changed config path cache, not production data.
    from jax_orchidee.driver import domain
    domain._load_case_config_cached.cache_clear()
    with pytest.raises(TideForcingError, match="unavailable"):
        driver._bind_context_hydrol_tides(context, context.source_clock_setup)


def _physics_frontend_stubs(monkeypatch, prepared):
    """Only unrelated meteorology, DIFFUCO, ENERBIL and state prerequisites."""
    payload = prepared.payload
    monkeypatch.setattr(driver, "_paper_1961_step_bundle_from_context", lambda *a, **k: prepared.first)
    monkeypatch.setattr(driver, "build_intersurf_first_step_payload", lambda *a, **k: payload)
    monkeypatch.setattr(driver, "intersurf_payload_with_driver_wind_z0", lambda *a, **k: payload)
    monkeypatch.setattr(driver, "_driver_wind_z0_from_previous_state", lambda *a: .1)
    monkeypatch.setattr(driver, "_diffuco_restart_payload_from_previous_state", lambda *a: ({}, (), ()))
    monkeypatch.setattr(driver, "_enerbil_state_payload_from_previous_state", lambda *a: ({}, (), ()))
    monkeypatch.setattr(driver, "_hydrol_state_namespace_from_previous_state", lambda *a: (NS(), ()))
    monkeypatch.setattr(driver, "_diffuco_driver_coupling_fields", lambda *a, **k: {})
    monkeypatch.setattr(driver, "_enerbil_driver_coverage_payload", lambda *a, **k: {})
    monkeypatch.setattr(driver, "assemble_diffuco_first_step_precall_payload", lambda **k: NS(ok=True, payload={}))
    monkeypatch.setattr(driver, "run_pft14_local_enerbil_precall_from_first_step_precall",
                        lambda *a, **k: NS(result=NS(payload=NS(payload={}))))
    monkeypatch.setattr(driver, "assemble_enerbil_first_step_precall_payload", lambda **k: NS(ok=True, payload={}))
    monkeypatch.setattr(driver, "run_enerbil_first_step_local_from_precall", lambda *a, **k: NS(payload={}))
    return NS(control_salinity=NS(payload={"control_salinity": np.asarray([1.])}),
              control_inundation=NS(payload={"control_inudate": np.asarray([1.])}),
              slowproc_derivvar=None, slowproc_derivvar_payload={})


def test_compact_real_entry_forwards_prefetched_pair_to_real_boundary(prepared, monkeypatch):
    context = prepared.context
    cache = _physics_frontend_stubs(monkeypatch, prepared)
    _stop_at_real_assembler(monkeypatch)
    controls = driver._driver_hydrol_controls(context, tstep=1)
    prebuilt = driver.DriverRuntimeHalfHourInput(
        step=NS(**{**vars(prepared.first), "tstep": 1}),
        base_payload=prepared.payload,
        source_clock=driver._driver_source_clock_payload(context, None, tstep=1),
        hydrol_tide_forcing=TideForcingValues(controls["wtp_tide"], controls["dwtp_tide"]))
    with pytest.raises(_ReachedHydrolBoundary) as caught:
        driver._paper_1961_next_step_runtime_result_compact(
            context.config_path, previous_state=_previous(context), year=1961, tstep=1,
            fixed_format_trace_dir=None, static_trace_fields=None, used_run_def_path=None,
            prepared_context=context, hydrol_static_template={"probe": 0.},
            diffuco_day_static_cache=cache, module_jit=False, diffuco_local_jit=False,
            prebuilt_input=prebuilt)
    _assert_boundary(caught, (329., 20.67))


def test_compiled_compact_entry_consumes_produced_leaves_without_host_lookup(prepared, monkeypatch):
    context = prepared.context
    _weather_stubs(monkeypatch, prepared)
    produced = driver._paper_compiled_forcing_day(context, year=1961, start_tstep=0, steps_per_stomate=2)
    driver._validate_compiled_source_clock_series(context, _previous(context, -1), produced)
    cache = _physics_frontend_stubs(monkeypatch, prepared)
    _stop_at_real_assembler(monkeypatch)
    def compact_boundary(clock, pair):
        try:
            # This is the same runtime_run_scalars/runtime_forcing calling
            # convention used by the real compiled scan's compact adapter.
            driver._paper_1961_next_step_runtime_result_compact(
                context.config_path, previous_state=_previous(context, -1), year=clock.year,
                tstep=clock.local_tstep, fixed_format_trace_dir=None, static_trace_fields=None,
                used_run_def_path=None, prepared_context=context, hydrol_static_template={"probe": 0.},
                diffuco_day_static_cache=cache, module_jit=False, diffuco_local_jit=False,
                prebuilt_base_payload=prepared.payload, runtime_run_scalars=context.run_scalars,
                runtime_forcing=prepared.weather_slab, source_clock=clock, hydrol_tide_forcing=pair)
        except _ReachedHydrolBoundary as boundary:
            return jnp.stack((boundary.assembly.payload["wtp_tide"], boundary.assembly.payload["dwtp_tide"]))
        raise AssertionError("compact adapter failed to reach the HYDROL boundary")
    actual = jax.jit(jax.vmap(compact_boundary))(produced.source_clock, produced.hydrol_tide_forcing)
    np.testing.assert_array_equal(actual, [[308.33, 44.83], [329., 20.67]])
    with pytest.raises(TideForcingError, match="host-prefetched"):
        jax.jit(lambda clock: compact_boundary(clock, None))(
            jax.tree_util.tree_map(lambda array: array[0], produced.source_clock))


def test_real_compiled_scan_passes_dynamic_tides_and_cached_owner_readmits_check_cwrr(prepared, monkeypatch):
    """Real scan/cache/admission; only its physical transition is substituted."""
    _weather_stubs(monkeypatch, prepared)
    context = replace(prepared.context, extras={**prepared.context.extras, "peat_hydro": True},
        diffuco_pft14_static_kwargs={"trans_co2_pft_params": {}, "ldq_cdrag_from_gcm": False})
    context.run_scalars.is_tree = np.zeros(context.nvm, dtype=bool)
    one = driver._paper_compiled_forcing_day(context, year=1961, start_tstep=0, steps_per_stomate=1)
    two = driver._paper_compiled_forcing_day(context, year=1961, start_tstep=1, steps_per_stomate=1)
    attributes = {**vars(prepared.payload), **{name: None for name in driver.WATCHOUT_COUPLING_FIELDS}}
    payload_type = make_dataclass("TideScanPayload", [(name, object) for name in attributes])
    payload = payload_type(**attributes)
    monkeypatch.setattr(driver, "_COMPILED_SECHIBA_SCAN_CACHE", {})
    monkeypatch.setattr(driver, "_hydrol_runtime_static_tables_from_compiled_arrays", lambda *a, **k: None)
    traces = []
    def physical_transition_delegate(*args, **kwargs):
        traces.append("READ103/104 leaves reached compact transition")
        clock = kwargs["source_clock"]
        pair = kwargs["hydrol_tide_forcing"]
        controls = driver._driver_hydrol_controls(kwargs["prepared_context"], tstep=kwargs["tstep"],
                                                 source_clock=clock, tide_values=pair)
        assembly = hydrol.assemble_hydrol_first_step_precall_payload(
            **controls, enerbil_payload={}, hydrol_restart=NS(), slowproc_restart=NS(),
            pref_soil_veg=np.asarray([4]), nstm=6, ext_coeff_vegetfrac=np.asarray([1.]),
            static_payload_template={"adapter_probe": 0.})
        closure = hydrol.run_hydrol_first_step_module_from_precall(assembly, humcste=np.ones(1),
            dz_mm=np.ones(3), dh_mm=np.ones(3), zz_mm=np.ones(3), tides=True)
        assert not assembly.ok and closure.module is None
        value = jnp.stack((assembly.payload["wtp_tide"], assembly.payload["dwtp_tide"]))
        previous = kwargs["previous_state"]
        state = replace(previous, tstep=clock.local_tstep, value=value,
                        source_clock_runtime=replace(previous.source_clock_runtime, last_applied_istp=clock.istp))
        return NS(next_state=state, entry_payload={name: value for name in driver._COMPILED_STOMATE_ENTRY_FIELDS})
    monkeypatch.setattr(driver, "_paper_1961_next_step_runtime_result_compact", physical_transition_delegate)
    def executable():
        return driver._paper_compiled_sechiba_scan(config_path=context.config_path, context=context,
            base_payload_template=payload, run_scalars=context.run_scalars,
            runtime_spec=NS(components=(), field_names_by_component=()), hydrol_static_template={},
            hydrol_runtime_static_tables=NS(mineral=NS(imin=1, imax=2)))
    run = executable()
    assert run is executable()
    initial = _ProbeScanState(np.int64(-1), context.source_clock_setup.runtime, np.zeros(2))
    arguments = ({}, (), (), None, driver.DriverCompiledDiffucoParameterValues({}, jnp.ones(context.nvm)))
    first, output, _ = run(initial, one, *arguments)
    np.testing.assert_array_equal(first.value, [308.33, 44.83])
    second, output, _ = run(first, two, *arguments)
    np.testing.assert_array_equal(second.value, [329., 20.67])
    assert len(traces) == 1  # one compiled shape, second pair is dynamic data
    # Existing cached executable must re-admit effective mutable configuration.
    context.run_def_values["CHECK_CWRR"] = "TRUE"
    with pytest.raises(NotImplementedError, match="CHECK_CWRR"):
        run(first, two, *arguments)


def _first_entry_stubs(monkeypatch, prepared):
    _physics_frontend_stubs(monkeypatch, prepared)
    monkeypatch.setattr(driver, "stomate_driver_entry_source", lambda *a, **k: NS())
    monkeypatch.setattr(driver, "stomate_no_routing_entry_source", lambda *a, **k: NS())
    monkeypatch.setattr(driver, "enerbil_first_step_input_coverage", lambda **k: NS())
    monkeypatch.setattr(driver, "assemble_pft14_control_salinity_first_step", lambda **k: NS())
    monkeypatch.setattr(driver, "assemble_pft14_control_inundation_first_step", lambda **k: NS())
    monkeypatch.setattr(driver, "paper_case_cwrr_vertical_soil_grid_from_used_run_def", lambda *a: _grid())
    monkeypatch.setattr(driver, "paper_case_vertical_grids_from_used_run_def",
                        lambda *a: (np.asarray([.01, .1, 1.]), _grid().zlt, _grid().znt))
    derivative = NS(qsintmax=np.ones((1, prepared.context.nvm)), assim_param=np.ones((1, prepared.context.nvm)),
                    height=np.ones((1, prepared.context.nvm)), temp_growth=np.ones((1, prepared.context.nvm)) * 280.)
    monkeypatch.setattr(driver, "_paper_first_step_slowproc_initial_fields", lambda **k: derivative)
    return derivative


def test_first_step_scaffold_real_entry_forwards_first_pair(prepared, monkeypatch):
    context = prepared.context
    _first_entry_stubs(monkeypatch, prepared)
    monkeypatch.setattr(driver, "resolve_hydrol_initialization_refsoc_1d", lambda *a, **k: None)
    monkeypatch.setattr(driver, "restart_thermosoil_initialization_supported", lambda *a: False)
    methods = dict(as_coverage_payload=lambda: {}, as_payload=lambda: {})
    restart = NS(driver_albedo=NS(z0=np.asarray([.1]), **methods),
                 enerbil_thermal=NS(**methods), diffuco_precall=NS(**methods),
                 stomate=NS(biomass=np.zeros((1, context.nvm, 8))),
                 hydrol=NS(), slowproc=NS(), thermosoil=NS(), sechiba_start=Path("unused.nc"),
                 sechiba_restart_state=NS(fields={"us": np.zeros((1, context.nvm, 6, 3))}))
    context = replace(context, extras={**context.extras, "first_step_restart_state": restart})
    _stop_at_real_assembler(monkeypatch)
    with pytest.raises(_ReachedHydrolBoundary) as caught:
        driver.paper_1961_driver_timestep_scaffold(context.config_path, tstep=0, prepared_context=context)
    _assert_boundary(caught, (308.33, 44.83))


def _cold_initialization_stubs(monkeypatch, prepared):
    """Explicitly substitute unrelated cold science before the tide adapter."""
    derivative = _first_entry_stubs(monkeypatch, prepared)
    nvm = prepared.context.nvm
    vegetation = NS(veget=np.ones((1, nvm)), veget_max=np.ones((1, nvm)),
                    frac_nobio=np.zeros((1, 1)), totfrac_nobio=np.zeros(1),
                    soiltile=np.asarray([[0., 0., 0., 1., 0., 0.]]))
    veg_cold = NS(vegetation=vegetation, lai=np.ones((1, nvm)), height=np.ones((1, nvm)),
                  frac_age=np.ones((1, nvm, 4)), tot_bare_soil=np.zeros(1))
    full = NS(**vars(vegetation), lai=veg_cold.lai, height=veg_cold.height, frac_age=veg_cold.frac_age)
    explicit_snow = NS(snowdz=np.zeros((1, 3)), snowrho=np.zeros((1, 3)), snowtemp=np.ones((1, 3)) * 273.15)
    cold = NS(mc=np.ones((1, 3, 6)) * .3, mcl=np.ones((1, 3, 6)) * .3,
              explicit_snow=explicit_snow, ae_ns=np.zeros((1, 6)), water2infilt=np.zeros((1, 6)),
              free_drain_coef=np.ones((1, 6)), zwt_force=np.ones((1, 6)) * 1e20,
              evap_bare_lim_ns=np.zeros((1, 6)), us=np.zeros((1, nvm, 6, 3)),
              qsintveg=np.zeros((1, nvm)), evap_bare_lim=np.zeros(1),
              humrel=np.zeros((1, nvm)), profil_froz_hydro_ns=np.zeros((1, 3, 6)),
              temp_hydro=np.ones((1, 3)) * 280, wt_ab=np.zeros(1), wt_ab_tide=np.zeros(1),
              snow=np.zeros(1), snow_age=np.zeros(1), snow_nobio=np.zeros((1, 1)),
              snow_nobio_age=np.zeros((1, 1)))
    monkeypatch.setattr(driver, "load_paper_1961_step_bundle", lambda *a, **k: prepared.first)
    monkeypatch.setattr(driver, "initialize_imposed_vegetation_state", lambda *a, **k: vegetation)
    monkeypatch.setattr(driver, "slowproc_cold_start_vegetation_entry_state", lambda **k: veg_cold)
    monkeypatch.setattr(driver, "slowproc_init_pft14_explicit", lambda **k: full)
    for name in ("slowproc_static_entry_state", "slowproc_no_lcc_entry_state", "slowproc_fire_disabled_entry_state",
                 "slowproc_dyn_peat_disabled_entry_state", "slowproc_thermosoil_entry_init_state",
                 "hydrol_cold_start_waterbal_begin_state", "hydrol_cold_start_thermosoil_moisture_inputs",
                 "stomate_cold_start_daily_accumulator_state", "stomate_cold_start_season_state",
                 "stomate_cold_start_gas_state", "slowproc_erosion_daily_zero_entry_state"):
        monkeypatch.setattr(driver, name, lambda *a, **k: NS())
    monkeypatch.setattr(driver, "hydrol_cold_start_state", lambda **k: cold)
    monkeypatch.setattr(driver, "hydrol_refsoc_1d_thresholds", lambda **k: NS(mcs=np.ones(1) * .4))
    monkeypatch.setattr(driver, "_fresh_cold_thermosoil_missing_inputs", lambda *a, **k: ())
    monkeypatch.setattr(driver, "_fresh_cold_thermosoil_refsoc_input", lambda *a, **k: (None, None))
    monkeypatch.setattr(driver, "thermosoil_initial_ptn_constant", lambda **k: np.ones((1, 3, nvm)) * 280)
    surface = NS(temp_sol_new=np.ones(1) * 280, temp_sol_new_pft=np.ones((1, nvm)) * 280, as_payload=lambda: {})
    monkeypatch.setattr(driver, "enerbil_cold_start_surface_state", lambda **k: surface)
    monkeypatch.setattr(driver, "stomate_cold_start_entry_state",
                        lambda **k: NS(carbon_32l=np.zeros((1, nvm, 1, 32)), biomass=np.zeros((1, nvm, 8))))
    thermal = NS(stempdiag=np.ones((1, 3)) * 280)
    monkeypatch.setattr(driver, "thermosoil_cold_start_coef_closure", lambda **k: thermal)
    monkeypatch.setattr(driver, "_enerbil_soil_thermal_payload_from_thermosoil_cold_coef", lambda *a: {})
    monkeypatch.setattr(driver, "slowproc_derivvar_explicit", lambda **k: derivative)
    monkeypatch.setattr(driver, "condveg_main_minimal", lambda **k: NS(
        albedo=np.ones((1, 2)) * .2, roughheight=np.ones(1), roughheight_pft=np.ones((1, nvm)),
        z0h=np.ones(1) * .1, z0m=np.ones(1) * .1,
        frac_snow_veg=np.zeros(1), frac_snow_nobio=np.zeros((1, 1))))
    return cold


def test_fresh_cold_real_entry_builds_source_clock_and_first_pair_without_restart_constructor(prepared, monkeypatch):
    context = prepared.context
    _cold_initialization_stubs(monkeypatch, prepared)
    # The cold entry must build its own clock/tide context. Reusing the full
    # prepared restart constructor would hide a production cold-start gap.
    def forbidden(*a, **k):
        raise AssertionError("cold entry used the restart prepared constructor")
    monkeypatch.setattr(driver, "prepare_paper_1961_driver_context", forbidden)
    monkeypatch.setattr(driver, "_prepare_paper_1961_driver_context_cached", forbidden)
    _stop_at_real_assembler(monkeypatch)
    with pytest.raises(_ReachedHydrolBoundary) as caught:
        driver.paper_1961_driver_cold_start_first_step_coverage(
            context.config_path, used_run_def_path=context.run_def_path)
    _assert_boundary(caught, (308.33, 44.83))


@pytest.mark.parametrize("form", ["default", "indexed", "literal_vector"])
def test_cold_zwt_force_reaches_real_constructor(prepared, monkeypatch, form):
    """Public cold reader -> real initialization; stop before physical processes."""
    _cold_initialization_stubs(monkeypatch, prepared)
    desired = np.asarray([1e20, .25, .5, 1., 1.5, 1e20])
    values = {k: v for k, v in prepared.values.items() if not k.upper().startswith("ZWT_FORCE")}
    if form == "indexed":
        values.update({f"ZWT_FORCE__{i:05d}": str(v) for i, v in enumerate(desired, 1)})
    elif form == "literal_vector":
        values["ZWT_FORCE"] = " ".join(map(str, desired))
    else:
        desired = np.full(6, 1e20)
    prepared.context.run_def_path.write_text(
        "\n".join(f"{k}={v}" for k, v in values.items()), encoding="utf8")

    class ReachedColdState(Exception):
        def __init__(self, state):
            self.state = state

    def real_cold_boundary(**kwargs):
        raise ReachedColdState(hydrol.hydrol_cold_start_state(**kwargs))

    monkeypatch.setattr(driver, "hydrol_cold_start_state", real_cold_boundary)
    with pytest.raises(ReachedColdState) as caught:
        driver.paper_1961_driver_cold_start_first_step_coverage(
            prepared.context.config_path, used_run_def_path=prepared.context.run_def_path)
    np.testing.assert_array_equal(caught.value.state.zwt_force, desired[None, :])
    assert caught.value.state.zforce is (form != "default")


class _ReachedYearDispatch(Exception):
    def __init__(self, context, state, controls, clock):
        self.context, self.state, self.controls, self.clock = context, state, controls, clock


@pytest.mark.parametrize("mode,expected", [("continuous", (349.67, 20.67)),
                                             ("new_native_run", (308.33, 44.83))])
def test_actual_restart_year_owner_preserves_or_reopens_stream_origin(prepared, monkeypatch, mode, expected):
    """Real year entry, clock metadata/binder/rebase, stopped before day dispatch.

    Two synthetic12-hour steps cross Dec31; this is only a chronology/IO
    boundary fixture. No annual simulation or shortened scientific validation
    is performed or implied. Restart NetCDF files contain clock metadata only.
    """
    directory = prepared.context.config_path.parent
    def weather(path, origin):
        with Dataset(path, "w") as dataset:
            dataset.createDimension("time", 72)
            variable = dataset.createVariable("time", "f8", ("time",))
            variable.units = "seconds since " + origin
            variable.calendar = "noleap"
            variable[:] = np.arange(72) * 86400.
    weather(prepared.weather, "1961-12-31 00:00:00")
    weather(directory / "weather-1962.nc", "1962-01-01 00:00:00")
    values = _set(prepared.context.run_def_values, DT_SECHIBA="43200", TIME_LENGTH="1D")
    extras = {**prepared.context.extras, "dt_sechiba": 43200., "dt_sechiba_days": .5,
              "runtime": NS(dt_sechiba=43200., dt_stomate=86400.), "first_step_restart_state": None}
    seed_context = replace(prepared.unbound, run_def_values=values, extras=extras)
    seed_setup = driver.source_clock_setup_from_config(seed_context.config_path,
        run_def_path=seed_context.run_def_path, run_def_values=values,
        first_step_bundle=seed_context.first_step_bundle, run_id="year-origin")
    seed_context = driver._bind_context_hydrol_tides(seed_context, seed_setup)
    old_runtime = SourceClockRuntime(seed_setup.binding, np.int64(2), np.int64(0), np.int64(1))
    paths = {}
    for component, name in (("driver", "RESTART_FILEIN"), ("sechiba", "SECHIBA_restart_in"),
                            ("stomate", "STOMATE_RESTART_FILEIN")):
        path = directory / f"{component}_start.nc"
        with Dataset(path, "w") as dataset:
            dataset.createDimension("time", 1)
            time = dataset.createVariable("time", "f8", ("time",))
            time.setncatts(source_clock_variable_attributes(seed_setup.binding)["time"])
            time[:] = [2 * 43200.]
            steps = dataset.createVariable("time_steps", "i4", ("time",))
            steps.setncatts(source_clock_variable_attributes(seed_setup.binding)["time_steps"])
            steps[:] = [2]
            dataset.setncatts(source_clock_global_attributes(old_runtime))
        paths[name] = str(path)
    context = replace(seed_context, run_def_values=_set(values, **paths),
                      first_step_bundle=NS(**{**vars(prepared.first), "year": 1962}))
    packet = driver.DriverPreviousStepStatePacket(1, {
        "sechiba_finalize_state": {"us": np.zeros((1, 1, 6, 3)), "vegstress": np.ones((1, 1)),
                                    "njsc": np.asarray([2.])},
        "hydrol_previous_step_state": {}, "thermosoil_previous_step_state": {},
    }, {}, old_runtime)
    monkeypatch.setattr(driver, "driver_year_handoff_state_gaps", lambda state: ())
    monkeypatch.setattr(driver, "_paper_1961_step_bundle_from_context", lambda context, **k: context.first_step_bundle)
    monkeypatch.setattr(driver, "build_intersurf_first_step_payload", lambda *a, **k: prepared.payload)
    monkeypatch.setattr(driver, "_driver_wind_z0_from_previous_state", lambda *a: .1)
    monkeypatch.setattr(driver, "resolve_hydrol_initialization_refsoc_1d", lambda *a, **k: None)
    monkeypatch.setattr(driver, "restart_thermosoil_initialization_supported", lambda context: False)
    monkeypatch.setattr(hydrol, "hydrol_init_us_state", lambda **k: NS(us=k["us"], humrelv=np.ones((1, 1, 6))))
    monkeypatch.setattr(hydrol, "hydrol_init_vegstress", lambda **k: k["vegstress"])
    def stop_before_day(*args, **kwargs):
        admitted = kwargs["prepared_context"]
        previous = kwargs["previous_state"]
        clock = driver._driver_source_clock_payload(admitted, previous.source_clock_runtime, tstep=0)
        controls = driver._driver_hydrol_controls(admitted, tstep=0,
                                                 source_clock_runtime=previous.source_clock_runtime)
        raise _ReachedYearDispatch(admitted, previous, controls, clock)
    monkeypatch.setattr(driver, "paper_1961_driver_later_day_runtime_result", stop_before_day)
    with pytest.raises(_ReachedYearDispatch) as caught:
        driver.paper_1961_driver_restart_year_start_day_result(context.config_path,
            previous_year_end_state=packet, year=1962, prepared_context=context, run_mode=mode,
            run_id="next-native" if mode == "new_native_run" else "year-origin")
    probe = caught.value
    assert probe.state.tstep == -1 and int(probe.clock.local_tstep) == 0
    assert int(probe.clock.istp) == 3
    assert (int(probe.clock.year), int(probe.clock.month), int(probe.clock.day)) == (1962, 1, 1)
    assert (probe.controls["wtp_tide"], probe.controls["dwtp_tide"]) == expected
    if mode == "continuous":
        assert probe.context.hydrol_tide_run is seed_context.hydrol_tide_run
        assert probe.context.hydrol_tide_run.first_source_istp == 1
    else:
        assert probe.context.hydrol_tide_run.first_source_istp == 3
        assert probe.context.source_clock_setup.binding.run_id == "next-native"
