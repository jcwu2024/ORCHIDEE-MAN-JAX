"""Bounded real-file state handoff checks; no physical timestep executes.

The producer under test is the actual day-end packet/finalize adapter. Tagged
scientific arrays are interface probes, not physical trajectories or numerical
Fortran references. All file definitions come from the tracked paper schemas.
"""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from netCDF4 import Dataset

from jax_orchidee.driver.fast_state import (
    fast_state_from_previous_packet,
    previous_packet_from_fast_state,
)
from jax_orchidee.driver.init import read_run_scalars
from jax_orchidee.driver.orchestration import (
    LATER_DAY_STOMATE_ENTRY_STATE_OVERRIDES,
    STOMATE_DAY_SEASON_STATE_FIELDS,
    DriverFirstDayStomateStatePacket,
    DriverPreviousStepStatePacket,
    _paper_day_end_state_packet,
    _paper_later_day_stomate_entry_state_from_previous,
    _paper_later_day_stomate_season_state_from_previous,
)
from jax_orchidee.driver.restart_bundle import (
    PaperRestartBundlePhysicalState,
    PaperRestartBundleState,
    paper_restart_bundle_state_from_day_end_packet,
    write_paper_restart_start_bundle,
)
from jax_orchidee.driver.restart_file_io import read_dim2_driver_restart
from jax_orchidee.driver.restart_state import reference_case_first_step_restart_state
from jax_orchidee.sechiba.restart_io import (
    DEFAULT_SECHIBA_RESTART_SCHEMA,
    PHYSICAL_RESTART_VARIABLES,
    SechibaRestartState,
    read_sechiba_restart_state,
    sechiba_finalize_source_state_from_restart,
)
from jax_orchidee.stomate.reference import (
    StomateReadstartRemainderState,
    stomate_cold_start_daily_accumulator_state,
    stomate_cold_start_entry_state,
    stomate_cold_start_season_state,
)
from jax_orchidee.stomate.restart_io import (
    DEFAULT_STOMATE_RESTART_SCHEMA,
    StomateReadstartStates,
    StomateRestartPhysicalState,
    stomate_cold_start_gas_state,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "orchidee_man_250919.yaml"
CATALOG = ROOT / "configs" / "pft_catalogs" / "orchidee_man_paper_250919.json"


def _tag(value: object, index: int) -> np.ndarray:
    array = np.asarray(value)
    coordinates = np.arange(array.size).reshape(array.shape)
    if array.dtype.kind == "b":
        return (coordinates + index) % 2 == 0
    if array.dtype.kind in "iu":
        return np.asarray(coordinates + index + 1, dtype=array.dtype)
    return (index + 1) / 8.0 + coordinates / 1024.0


def _synthetic_stomate(nvm: int) -> StomateReadstartStates:
    """Create all public state identities using tracked shapes and cold owners."""
    t2m = np.array([280.0])
    states = {
        "entry_state": stomate_cold_start_entry_state(t2m=t2m, nvm=nvm, nslm=11),
        "season_state": stomate_cold_start_season_state(
            t2m=t2m, dt_days=1.0, nvm=nvm, nslm=11,
        ),
        "daily_state": stomate_cold_start_daily_accumulator_state(t2m=t2m, nvm=nvm, nslm=11),
        "gas_state": stomate_cold_start_gas_state(npts=1, nvm=nvm),
    }
    document = yaml.safe_load(
        (ROOT / "docs/source_audits/restart_state_lifecycle.yaml").read_text(encoding="utf-8")
    )
    sizes = {
        "npts": 1, "nvm": nvm, "ndeep": 32, "nvert": 171,
        "months_num": 360, "nlitt": 2, "ncarb": 3, "nbpools": 7,
    }
    axes = {
        field: group["axes"]
        for group in document["axis_groups"].values()
        for field in group["fields"]
    }
    remainder = {}
    for field in StomateReadstartRemainderState._fields:
        if field == "provenance":
            remainder[field] = ("synthetic state IO interface probe",)
        elif field.startswith("read_input_"):
            remainder[field] = False
        else:
            shape = tuple(sizes[axis] for axis in axes[field])
            remainder[field] = np.zeros(shape, dtype=bool if field == "ok_equilibrium" else float)
    states["remainder_state"] = StomateReadstartRemainderState(**remainder)
    names = sorted({name for state in states.values() for name in state._fields} - {"provenance"})
    tags = {field: i for i, field in enumerate(names)}
    for group, state in states.items():
        updates = {
            field: _tag(getattr(state, field), tags[field])
            for field in state._fields
            if field != "provenance" and not field.startswith("read_input_")
        }
        if group == "daily_state":
            for field in ("snowfall_daily", "snowmass_daily", "tmc_topgrass_daily"):
                updates[field] = np.zeros_like(updates[field])
        if group == "season_state":
            updates["date"] = 31
        if group == "remainder_state":
            # Global_years is an INTEGER source owner (stomate_io readstart).
            updates["Global_years"] = 7
        states[group] = state._replace(**updates)
    entry = states["entry_state"]
    states["entry_state"] = entry._replace(
        prod10_total=np.sum(entry.prod10, axis=(1, 2)),
        prod100_total=np.sum(entry.prod100, axis=(1, 2)),
        soilc_total=entry.deepC_a + entry.deepC_s + entry.deepC_p,
    )
    return StomateReadstartStates(**states, report=None)


def _synthetic_sechiba(nvm: int) -> SechibaRestartState:
    schema = json.loads(DEFAULT_SECHIBA_RESTART_SCHEMA.read_text(encoding="utf-8"))
    sizes = {name: item["size"] for name, item in schema["dimensions"].items()}
    sizes.update(z_a=nvm, l_d=nvm)
    fields = {}
    for index, (name, variable) in enumerate(schema["variables"].items()):
        if name in PHYSICAL_RESTART_VARIABLES:
            continue
        axes = [axis for axis in variable["dimensions"] if axis not in ("time", "y", "x")]
        shape = (1, *(sizes[axis] for axis in reversed(axes)))
        fields[name] = _tag(np.zeros(shape), index)
    fields["njsc"][:] = 12
    fields["veget_year"][:] = -1
    # One source marker in an otherwise identifiable local layer is retained:
    # the source US_INIT condition is ALL, not an elementwise replacement.
    fields["us"][0, 0, 0, 0] = 999999.0
    return SechibaRestartState(fields, {name: ("tagged interface input",) for name in fields})


def _physical(schema_path: Path) -> StomateRestartPhysicalState:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    sizes = {name: declaration["size"] for name, declaration in schema["dimensions"].items()}
    return StomateRestartPhysicalState(**{
        name: np.zeros(tuple(sizes[axis] for axis in schema["variables"][name]["dimensions"]))
        for name in schema["physical_variables"]
    })


def _layout_and_config(tmp_path: Path, compact: bool, monkeypatch, refsoc_enabled: bool):
    # All parsed inputs are local interface probes. No archived run.def or
    # materialized defaults under ignored outputs are needed by clean CI.
    run_def = tmp_path / "interface.def"
    values = {f"{name}__{pft:05d}": value
              for pft in range(1, 15)
              for name, value in (("EXT_COEFF_VEGETFRAC", .5),
                                  ("SLOWPROC_HEIGHT", .4), ("RATIO_Z0M_Z0H", 10.))}
    values["use_refSOC_hydrol"] = "TRUE" if refsoc_enabled else "FALSE"
    run_def.write_text("".join(f"{key} = {value}\n" for key, value in values.items()), encoding="utf-8")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["case"]["reference_run_def"] = str(run_def)
    if compact:
        payload = json.loads(CATALOG.read_text(encoding="utf-8"))
        payload["layouts"]["paper_compact"] = ["bare_soil", "mangrove_pft14"]
        catalog = tmp_path / "catalog.json"
        catalog.write_text(json.dumps(payload), encoding="utf-8")
        config["pft_catalog"] = {"path": str(catalog), "layout_id": "paper_compact"}
        config["structural_overrides"]["NVM"] = 2
    config_path = tmp_path / "interface.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setenv("ORCHIDEE_REPO_ROOT", str(ROOT))
    return read_run_scalars(config_path, run_def).pft_layout, config_path, run_def


def _packet_states(states):
    result = {}
    for group in ("entry_state", "season_state", "gas_state", "remainder_state"):
        result.update({k: v for k, v in getattr(states, group)._asdict().items() if k != "provenance"})
    result["daily_accumulators"] = {
        k: v for k, v in states.daily_state._asdict().items() if k != "provenance"
    }
    return result


def test_configured_thawed_humidity_in_ordinary_file_entry(tmp_path, monkeypatch):
    from jax_orchidee.driver.dim2 import Dim2DriverRestart

    layout, config_path, run_def = _layout_and_config(tmp_path, False, monkeypatch, False)
    states = _synthetic_stomate(layout.n_pft)
    entry = states.entry_state._replace(thawed_humidity=np.array([0.75]))
    states = replace(states, entry_state=entry)
    driver = Dim2DriverRestart(**{name: np.array([1.]) for name in Dim2DriverRestart.__dataclass_fields__})
    state = PaperRestartBundleState(driver=driver, sechiba=_synthetic_sechiba(layout.n_pft), stomate=states)
    physical = PaperRestartBundlePhysicalState(
        driver=_physical(ROOT / "docs/source_audits/driver_restart_netcdf_schema.json"),
        sechiba=_physical(DEFAULT_SECHIBA_RESTART_SCHEMA),
        stomate=_physical(DEFAULT_STOMATE_RESTART_SCHEMA),
    )
    report = write_paper_restart_start_bundle(
        tmp_path / "restart", state=state, physical_state=physical, pft_layout=layout,
    )
    base = run_def.read_text(encoding="utf-8")
    for options, expected in (
        ("reset_thawed_humidity=TRUE\nthawed_humidity_input=0.325D0\n", 0.325),
        ("RESET_THAWED_HUMIDITY=TRUE\n", 0.5),
        ("RESET_THAWED_HUMIDITY=FALSE\nTHAWED_HUMIDITY_INPUT=unused\n", 0.75),
        ("", 0.75),
    ):
        run_def.write_text(base + options, encoding="utf-8")
        restored = reference_case_first_step_restart_state(
            config_path, root=ROOT, run_dir=report.output_directory, run_def_path=run_def,
        )
        np.testing.assert_array_equal(restored.stomate_readstart.entry_state.thawed_humidity, [expected])
        np.testing.assert_array_equal(restored.stomate.thawed_humidity, [expected])


@pytest.mark.parametrize("compact", (False, True), ids=("legacy14", "compact2"))
@pytest.mark.parametrize("refsoc_enabled", (False, True), ids=("default93", "conditional_refsoc"))
def test_nondefault_day_end_fast_codec_real_triplet_and_next_consumers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, compact: bool, refsoc_enabled: bool,
) -> None:
    """Exercise production packet/finalize/IO/read/next-entry edges with tags."""
    layout, config_path, run_def_path = _layout_and_config(tmp_path, compact, monkeypatch, refsoc_enabled)
    states = _synthetic_stomate(layout.n_pft)
    sechiba = _synthetic_sechiba(layout.n_pft)
    source = sechiba_finalize_source_state_from_restart(sechiba)
    if refsoc_enabled:
        source["refSOC_1d"] = np.array([42500.125])
    from jax_orchidee.driver.dim2 import Dim2DriverRestart
    driver = {name: np.array([(i + 1) / 4.0]) for i, name in enumerate(Dim2DriverRestart.__dataclass_fields__)}
    previous = DriverPreviousStepStatePacket(
        tstep=17520,
        fields_by_component={
            "driver_previous_step_state": driver,
            "sechiba_finalize_state": source,
            "slowproc_stomate_previous_step_state": _packet_states(states),
        },
        provenance_by_component={},
    )
    daily_packet = DriverFirstDayStomateStatePacket(
        day_index=31, fields=_packet_states(states), provenance_by_field={},
    )
    packet = _paper_day_end_state_packet(previous_state=previous, stomate_state=daily_packet)
    fast = fast_state_from_previous_packet(packet)
    roundtripped = previous_packet_from_fast_state(fast)
    # Both ordinary and compact-state representations use the actual exporter.
    ordinary = paper_restart_bundle_state_from_day_end_packet(
        packet, base_stomate=states, kjit=17520, use_refsoc_hydrol=refsoc_enabled,
    )
    state = paper_restart_bundle_state_from_day_end_packet(
        roundtripped, base_stomate=states, kjit=17520, use_refsoc_hydrol=refsoc_enabled,
    )
    for field in ordinary.sechiba.fields:
        np.testing.assert_array_equal(state.sechiba.fields[field], ordinary.sechiba.fields[field])
    physical = PaperRestartBundlePhysicalState(
        driver=_physical(ROOT / "docs/source_audits/driver_restart_netcdf_schema.json"),
        sechiba=_physical(DEFAULT_SECHIBA_RESTART_SCHEMA),
        stomate=_physical(DEFAULT_STOMATE_RESTART_SCHEMA),
    )
    report = write_paper_restart_start_bundle(
        tmp_path / "next_year", state=state, physical_state=physical, pft_layout=layout,
    )
    restored = reference_case_first_step_restart_state(
        config_path, root=ROOT, run_dir=report.output_directory, run_def_path=run_def_path,
    )
    assert (len(report.driver.written_fields), len(report.sechiba.written_fields), len(report.stomate.written_fields)) == (13, 93 + int(refsoc_enabled), 161)
    assert restored.pft_layout == layout
    assert restored.sechiba_source_pft_layout.pft_ids == layout.pft_ids
    assert restored.stomate_source_pft_layout.pft_ids == layout.pft_ids
    raw = read_sechiba_restart_state(report.sechiba.output_path)
    for field, value in state.sechiba.fields.items():
        np.testing.assert_array_equal(raw.fields[field], value, err_msg=field)
        np.testing.assert_array_equal(restored.sechiba_restart_state.fields[field], value, err_msg=field)
    restored_driver = read_dim2_driver_restart(report.driver.output_path)
    for field in driver:
        np.testing.assert_array_equal(getattr(restored_driver, field), driver[field], err_msg=field)
    for group in ("entry_state", "season_state", "daily_state", "gas_state", "remainder_state"):
        expected = getattr(state.stomate, group)
        actual = getattr(restored.stomate_readstart, group)
        for field in expected._fields:
            if field == "provenance" or field.startswith("read_input_"):
                continue
            np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field), err_msg=f"{group}.{field}")
    next_packet = DriverPreviousStepStatePacket(
        tstep=17521,
        fields_by_component={"slowproc_stomate_previous_step_state": _packet_states(restored.stomate_readstart)},
        provenance_by_component={},
    )
    entry, gaps = _paper_later_day_stomate_entry_state_from_previous(
        template=stomate_cold_start_entry_state(t2m=np.array([280.0]), nvm=layout.n_pft, nslm=11),
        previous_state=next_packet,
    )
    assert gaps == ()
    for field in LATER_DAY_STOMATE_ENTRY_STATE_OVERRIDES:
        np.testing.assert_array_equal(getattr(entry, field), getattr(state.stomate.entry_state, field))
    season, gaps = _paper_later_day_stomate_season_state_from_previous(
        template=stomate_cold_start_season_state(t2m=np.array([280.0]), dt_days=1.0, nvm=layout.n_pft, nslm=11),
        previous_state=next_packet,
    )
    assert gaps == ()
    for field in STOMATE_DAY_SEASON_STATE_FIELDS:
        np.testing.assert_array_equal(getattr(season, field), getattr(state.stomate.season_state, field))
    for path in report.output_paths:
        with Dataset(path) as dataset:
            assert not {"KS", "MCS", "HYDROL_DEPTH_MAX_H", "KFACT_DECAY_RATE"} & set(dataset.variables)
    for role in ("sechiba", "stomate"):
        with Dataset(getattr(report, role).output_path) as dataset:
            assert json.loads(dataset.orchidee_jax_pft_ids_json) == list(layout.pft_ids)
    if refsoc_enabled:
        from jax_orchidee.sechiba.hydrol import hydrol_refsoc_1d_thresholds
        kwargs = dict(
            njsc=np.array([12]), use_refSOC_hydrol=True,
        )
        expected = hydrol_refsoc_1d_thresholds(refSOC_1d=source["refSOC_1d"], **kwargs)
        actual = hydrol_refsoc_1d_thresholds(
            refSOC_1d=restored.sechiba_restart_state.fields["refSOC_1d"], **kwargs,
        )
        for field in ("mcs", "mcw", "mcf"):
            np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
        with Dataset(report.sechiba.output_path) as dataset:
            assert dataset.variables["refSOC_1d"].dimensions == ("time", "y", "x")


def test_strict_sechiba_io_rejects_static_parameter_in_scientific_state(tmp_path: Path) -> None:
    from jax_orchidee.sechiba.restart_io import write_sechiba_restart_state
    state = _synthetic_sechiba(14)
    contaminated = SechibaRestartState({**state.fields, "KFACT_DECAY_RATE": np.array([2.0])}, state.provenance_by_field)
    with pytest.raises(ValueError, match="coverage mismatch.*KFACT_DECAY_RATE"):
        write_sechiba_restart_state(tmp_path / "bad.nc", state=contaminated, physical_state=_physical(DEFAULT_SECHIBA_RESTART_SCHEMA))
    assert not (tmp_path / "bad.nc").exists()


@pytest.mark.parametrize("enabled,present", ((True, False), (False, True)))
def test_conditional_refsoc_finalizer_requires_matching_source_switch(enabled: bool, present: bool) -> None:
    from jax_orchidee.sechiba.restart_export import paper_sechiba_restart_state_from_finalize_state
    source = sechiba_finalize_source_state_from_restart(_synthetic_sechiba(14))
    if present:
        source["refSOC_1d"] = np.array([42500.125])
    with pytest.raises(ValueError, match="refSOC_1d.*exactly when"):
        paper_sechiba_restart_state_from_finalize_state(source, kjit=17520, use_refsoc_hydrol=enabled)


def test_conditional_refsoc_lifecycle_requires_explicit_phase_ownership() -> None:
    from jax_orchidee.sechiba.restart_lifecycle import transition_sechiba_finalize_source_state
    previous = sechiba_finalize_source_state_from_restart(_synthetic_sechiba(14))
    previous["refSOC_1d"] = np.array([42500.125])
    base = tuple(name for name in previous if name != "refSOC_1d")
    provenance = {name: ("interface probe declared carry",) for name in previous}
    with pytest.raises(ValueError, match="ownership mismatch.*refSOC_1d"):
        transition_sechiba_finalize_source_state(
            previous, updates={}, carried_fields=base, update_provenance={}, carry_provenance=provenance,
        )
    transition = transition_sechiba_finalize_source_state(
        previous, updates={}, carried_fields=tuple(previous), update_provenance={}, carry_provenance=provenance,
    )
    np.testing.assert_array_equal(transition.fields["refSOC_1d"], previous["refSOC_1d"])
    assert "refSOC_1d" in transition.carried_fields


def test_initialized_refsoc_first_writer_scaffold_ordinary_and_compact_carry(monkeypatch) -> None:
    _exercise_initialized_refsoc_transport(monkeypatch, enabled=True)


def test_disabled_refsoc_first_writer_drops_stale_restart_field_before_default_export(monkeypatch) -> None:
    _exercise_initialized_refsoc_transport(monkeypatch, enabled=False)


def _exercise_initialized_refsoc_transport(monkeypatch, *, enabled: bool) -> None:
    """Execute transport owners; replace only the orthogonal physical finalizer."""
    from jax_orchidee.driver import orchestration as orch
    states = _synthetic_stomate(14)
    state = _synthetic_sechiba(14)
    initialized = np.array([42500.125])
    if not enabled:
        state = SechibaRestartState(
            {**state.fields, "refSOC_1d": np.array([12345.875])}, state.provenance_by_field,
        )
    source = sechiba_finalize_source_state_from_restart(state)
    from jax_orchidee.driver.source_clock import SourceClockBinding, SourceClockRuntime, SourceClockSetup
    from jax_orchidee.sechiba.ioipsl_calendar import IOIPSLCalendar
    calendar = IOIPSLCalendar("noleap")
    date0 = calendar.ymds2ju(2001, 1, 10, 0.0)
    lock = IOIPSLCalendar.from_scalar_startdate("noleap", date0)
    binding = SourceClockBinding("refsoc-interface", "noleap", date0, 1800.0,
                                 lock.startdate_day, lock.startdate_second, 0)
    clock_runtime = SourceClockRuntime(binding, np.int64(0), np.int64(0), np.int64(1))
    context = SimpleNamespace(source_clock_setup=SourceClockSetup(binding, clock_runtime, None, None, None),
                              run_def_values={"use_refSOC_hydrol": "TRUE" if enabled else "FALSE"})
    slowproc = SimpleNamespace(**{
        name: source[name] for name in ("veget", "veget_max", "lai", "frac_nobio", "height", "frac_age")
    })
    snapshot = SimpleNamespace(
        slowproc=slowproc, stomate=states.entry_state, stomate_readstart=states,
        stomate_input=None, sechiba_restart_state=state,
        diffuco_precall=SimpleNamespace(soilalbedo_bg=state.fields["soilalbedo_bg"]),
        hydrol=SimpleNamespace(),
        thermosoil=SimpleNamespace(refsoc=source["refsoc"], e_soil_lat=source["e_soil_lat"]),
    )
    monkeypatch.setattr(orch, "_sechiba_finalize_state_from_components", lambda *, previous, **kwargs: dict(previous))
    first = orch.driver_previous_step_state_from_timestep_scaffold(SimpleNamespace(
        step=SimpleNamespace(tstep=0), first_step_restart_state=snapshot,
        payload=SimpleNamespace(is_watchout=False),
        source_clock_context=context,
        enerbil_first_step_local=None, diffuco_first_step_local_enerbil_precall=None,
        diffuco_control_salinity=None, diffuco_control_assembly=None,
        thermosoil_first_step_module=None, condveg_first_step_module=None,
        hydrol_first_step_module=None,
        diffuco_first_step_precall=SimpleNamespace(payload={"temp_growth": np.array([280.0])}),
        stomate_first_step_daily_accumulation=None,
        hydrol_first_step_precall=SimpleNamespace(payload={"refSOC_1d": initialized} if enabled else {}),
    ))
    for component in ("sechiba_finalize_state", "hydrol_previous_step_state"):
        if enabled:
            np.testing.assert_array_equal(first.fields_by_component[component]["refSOC_1d"], initialized)
        else:
            assert "refSOC_1d" not in first.fields_by_component.get(component, {})
    scaffold = SimpleNamespace(
        ready_for_next_state=True, tstep=1, source_clock_context=context,
        enerbil=SimpleNamespace(previous_state=first, enerbil_local=None, diffuco_local_enerbil_precall=None,
                               payload=SimpleNamespace(is_watchout=False)),
        thermosoil_module=None, condveg_module=None, hydrol_module=None,
    )
    first_step = orch.driver_previous_step_state_from_next_step_hydrol_scaffold(scaffold)
    kwargs = dict(enerbil_local=None, diffuco_local=None, thermosoil_module=None,
                  condveg_module=None, hydrol_module=None, source_clock_context=context)
    ordinary = orch._runtime_next_state_from_components(tstep=2, previous_state=first_step, **kwargs)
    compact = orch._runtime_next_state_from_components(tstep=3, previous_state=fast_state_from_previous_packet(ordinary), **kwargs)
    for packet in (first_step, ordinary, compact):
        for component in ("sechiba_finalize_state", "hydrol_previous_step_state"):
            if enabled:
                np.testing.assert_array_equal(packet.fields_by_component[component]["refSOC_1d"], initialized)
            else:
                assert "refSOC_1d" not in packet.fields_by_component.get(component, {})
    from jax_orchidee.sechiba.restart_export import paper_sechiba_restart_state_from_finalize_state
    exported = paper_sechiba_restart_state_from_finalize_state(
        compact.fields_by_component["sechiba_finalize_state"], kjit=4, use_refsoc_hydrol=enabled,
    )
    assert len(exported.fields) == 93 + int(enabled)
    assert ("refSOC_1d" in state.fields) is (not enabled)


@pytest.mark.parametrize("enabled", (False, True))
def test_actual_file_split_caller_preserves_effective_run_def_and_source_flag(tmp_path, monkeypatch, enabled) -> None:
    """Stop at the exporter; no long acceptance runner or physics executes."""
    import importlib

    import jax_orchidee.runtime as runtime
    monkeypatch.setattr(runtime, "configure_jax_compilation_cache", lambda *args, **kwargs: None)
    caller = importlib.import_module("scripts.dev.audit_compiled_fast_path_acceptance")
    custom = tmp_path / "nondefault.used"
    custom.write_text("use_refSOC_hydrol = TRUE\n", encoding="utf-8")
    base = object()
    context = SimpleNamespace(run_def_values={"use_refSOC_hydrol": "TRUE" if enabled else "FALSE"},
                              first_step_restart_state=SimpleNamespace(stomate_readstart=base))
    seen = {}
    def prepare(config, **kwargs):
        seen.update(kwargs)
        return context
    class ReachedExporter(Exception):
        pass
    def export(packet, **kwargs):
        assert kwargs["base_stomate"] is base
        assert kwargs["use_refsoc_hydrol"] is enabled
        assert kwargs["kjit"] == 49
        raise ReachedExporter
    monkeypatch.setattr(caller, "prepare_paper_1961_driver_context", prepare)
    monkeypatch.setattr(caller, "paper_restart_bundle_state_from_day_end_packet", export)
    with pytest.raises(ReachedExporter):
        caller._split_restart_packet(
            SimpleNamespace(output_dir=tmp_path, landpoint_id="synthetic"),
            SimpleNamespace(tstep=48), tmp_path / "output", used_run_def_path=custom,
        )
    assert seen["used_run_def_path"] == custom
    assert seen["reference_run_dir"] == tmp_path


@pytest.mark.parametrize("enabled", (False, True))
def test_stage5_export_adapter_uses_effective_source_flag(monkeypatch, enabled) -> None:
    import scripts.dev.audit_stage5_lifecycle as caller
    context = SimpleNamespace(run_def_values={"use_refSOC_hydrol": "TRUE" if enabled else "FALSE"},
                              first_step_restart_state=SimpleNamespace(stomate_readstart="base"))
    seen = {}
    def export(packet, **kwargs):
        seen.update(kwargs)
        return "exported"
    monkeypatch.setattr(caller, "paper_restart_bundle_state_from_day_end_packet", export)
    assert caller._restart_bundle_from_context("packet", context, kjit=96) == "exported"
    assert seen == {"base_stomate": "base", "kjit": 96, "use_refsoc_hydrol": enabled}
