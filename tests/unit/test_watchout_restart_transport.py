"""Source-conditional WATCHOUT restart IO; no reference data or model execution."""

from __future__ import annotations

import json
import pickle
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from netCDF4 import Dataset

import jax_orchidee.driver.restart_bundle as bundle_io
from jax_orchidee.driver.dim2 import Dim2DriverRestart
from jax_orchidee.driver.restart_file_io import (
    DEFAULT_DRIVER_RESTART_SCHEMA,
    DRIVER_RESTART_VARIABLES,
    read_dim2_driver_restart,
    write_dim2_driver_restart,
)
from jax_orchidee.parameters.pft_catalog import PFTCatalogEntry, PFTRunLayout
from jax_orchidee.stomate.restart_io import (
    StomateRestartPhysicalState,
    StomateRestartWriteReport,
    create_stomate_restart_skeleton_from_schema,
)

COEFFICIENTS = ("petAcoef", "petBcoef", "peqAcoef", "peqBcoef")
RETAINED = tuple(name for name in DRIVER_RESTART_VARIABLES if name not in COEFFICIENTS)


def _physical(**overrides):
    return StomateRestartPhysicalState(
        nav_lon=np.array([[-30.0]]), nav_lat=np.array([[-20.0]]),
        nav_lev=np.array([0.0]), time=np.array([12.0]), time_steps=np.array([1800.0]),
        **overrides,
    )


def _values():
    return {name: np.array([13.0 + ordinal / 8.0]) for ordinal, name in enumerate(DRIVER_RESTART_VARIABLES)}


def _source_file(tmp_path, *, coefficients):
    """Build an actual small file from tracked schema, optionally source-conditional."""
    schema = json.loads(DEFAULT_DRIVER_RESTART_SCHEMA.read_text())
    if not coefficients:
        for name in COEFFICIENTS:
            schema["variables"].pop(name)
    schema_path = tmp_path / "fixture-schema.json"
    schema_path.write_text(json.dumps(schema))
    path = create_stomate_restart_skeleton_from_schema(
        tmp_path / "source.nc", _physical(), schema_path=schema_path,
    )
    with Dataset(path, "r+") as dataset:
        for name, value in _values().items():
            wire = DRIVER_RESTART_VARIABLES[name]
            if wire in dataset.variables:
                dataset.variables[wire][0] = value.reshape(1, 1, 1)
    return path


@pytest.mark.parametrize("coefficients", (False, True), ids=("absent-four", "stale-four"))
def test_watchout_reader_ignores_restart_coefficients(tmp_path, coefficients):
    # dim2_driver.f90:769-788 does not execute coefficient restget for WATCHOUT.
    source = _source_file(tmp_path, coefficients=coefficients)
    state = read_dim2_driver_restart(source, is_watchout=True)
    for name in RETAINED:
        np.testing.assert_array_equal(getattr(state, name), _values()[name])
    assert all(getattr(state, name) is None for name in COEFFICIENTS)
    if coefficients:
        original = read_dim2_driver_restart(source)
        assert all(getattr(original, name) is not None for name in COEFFICIENTS)
    else:
        with pytest.raises(KeyError, match="missing variables"):
            read_dim2_driver_restart(source)


@pytest.mark.parametrize("stale_coefficients", (False, True))
def test_watchout_writer_omits_four_variables_and_preserves_schema(tmp_path, stale_coefficients):
    values = _values()
    if not stale_coefficients:
        values = {name: value for name, value in values.items() if name in RETAINED}
    state = Dim2DriverRestart(**values)
    report = write_dim2_driver_restart(
        tmp_path / "watchout.nc", state=state, physical_state=_physical(), is_watchout=True,
    )
    assert report.written_fields == RETAINED
    schema = json.loads(DEFAULT_DRIVER_RESTART_SCHEMA.read_text())
    with Dataset(report.output_path) as dataset:
        assert not set(COEFFICIENTS) & set(dataset.variables)
        assert set(dataset.variables) == set(schema["variables"]) - set(COEFFICIENTS)
        assert dataset.file_name == "watchout.nc"
        for name in RETAINED:
            variable = dataset.variables[DRIVER_RESTART_VARIABLES[name]]
            declaration = schema["variables"][DRIVER_RESTART_VARIABLES[name]]
            assert variable.dimensions == tuple(declaration["dimensions"])
            for key, expected in declaration["attributes"].items():
                np.testing.assert_equal(variable.getncattr(key), expected)
    restored = read_dim2_driver_restart(report.output_path, is_watchout=True)
    for name in RETAINED:
        np.testing.assert_array_equal(getattr(restored, name), values[name])
    for name in COEFFICIENTS:
        assert getattr(restored, name) is None
        if stale_coefficients:
            np.testing.assert_array_equal(getattr(state, name), _values()[name])


def test_default_nonwatchout_keeps_all_thirteen_strictly(tmp_path):
    state = Dim2DriverRestart(**_values())
    report = write_dim2_driver_restart(tmp_path / "ordinary.nc", state=state, physical_state=_physical())
    assert report.written_fields == tuple(DRIVER_RESTART_VARIABLES)
    restored = read_dim2_driver_restart(report.output_path)
    for name in DRIVER_RESTART_VARIABLES:
        np.testing.assert_array_equal(getattr(restored, name), getattr(state, name))
    with pytest.raises(ValueError, match="undefined"):
        write_dim2_driver_restart(tmp_path / "missing.nc", state=replace(state, peqAcoef=None), physical_state=_physical())


@pytest.mark.parametrize("name", RETAINED)
def test_watchout_keeps_nine_field_missing_and_shape_gates(tmp_path, name):
    values = {key: value for key, value in _values().items() if key in RETAINED}
    with pytest.raises(ValueError, match="undefined"):
        write_dim2_driver_restart(
            tmp_path / "missing.nc", state=Dim2DriverRestart(**{**values, name: None}),
            physical_state=_physical(), is_watchout=True,
        )
    with pytest.raises(ValueError, match="normalized shape"):
        write_dim2_driver_restart(
            tmp_path / "shape.nc", state=Dim2DriverRestart(**{**values, name: np.zeros(2)}),
            physical_state=_physical(), is_watchout=True,
        )


@pytest.mark.parametrize("excluded", (("nav_lon",), ("unknown_coefficient",)))
def test_schema_exclusion_is_limited_to_known_science(tmp_path, excluded):
    with pytest.raises(ValueError, match="science"):
        create_stomate_restart_skeleton_from_schema(
            tmp_path / "invalid.nc", _physical(), schema_path=DEFAULT_DRIVER_RESTART_SCHEMA,
            excluded_science_variables=excluded,
        )


def test_watchout_exclusion_does_not_bypass_physical_metadata_gate(tmp_path):
    state = Dim2DriverRestart(**{name: value for name, value in _values().items() if name in RETAINED})
    with pytest.raises(ValueError, match="physical metadata overlay cannot modify science"):
        write_dim2_driver_restart(
            tmp_path / "overlay.nc", state=state,
            physical_state=_physical(variable_attributes={"peqAcoef": {"units": "invented"}}),
            is_watchout=True,
        )


def _export_stubs(monkeypatch):
    # Retain the real bundle's mode/driver/93-field gates while isolating unrelated science conversion.
    monkeypatch.setattr(bundle_io, "validate_source_clock_restart_packet", lambda packet, source_istp: None)
    monkeypatch.setattr(bundle_io, "stomate_full_restart_states_from_day_end_packet", lambda *args, **kwargs: ("stomate", None))
    monkeypatch.setattr(bundle_io, "paper_sechiba_restart_state_from_finalize_state", lambda *args, **kwargs: "sechiba")


def _packet(*, coefficients=False):
    driver = {name: value for name, value in _values().items() if coefficients or name in RETAINED}
    return SimpleNamespace(fields_by_component={
        "driver_previous_step_state": driver,
        "sechiba_finalize_state": {f"science_{index}": np.array([index]) for index in range(93)},
    })


@pytest.mark.parametrize("coefficients", (False, True))
def test_bundle_export_requires_nine_for_watchout_and_ignores_stale_four(monkeypatch, coefficients):
    _export_stubs(monkeypatch)
    packet = _packet(coefficients=coefficients)
    original_keys = tuple(packet.fields_by_component["driver_previous_step_state"])
    state = bundle_io.paper_restart_bundle_state_from_day_end_packet(packet, base_stomate=None, kjit=4, is_watchout=True)
    assert state.is_watchout is True
    assert all(getattr(state.driver, name) is None for name in COEFFICIENTS)
    for name in RETAINED:
        np.testing.assert_array_equal(getattr(state.driver, name), _values()[name])
    assert tuple(packet.fields_by_component["driver_previous_step_state"]) == original_keys
    with pytest.raises(ValueError, match="missing driver restart"):
        bundle_io.paper_restart_bundle_state_from_day_end_packet(_packet(), base_stomate=None, kjit=4)


def test_watchout_bundle_does_not_relax_remaining_science_gates(monkeypatch):
    _export_stubs(monkeypatch)
    packet = _packet()
    packet.fields_by_component["driver_previous_step_state"].pop("old_eair")
    with pytest.raises(ValueError, match="old_eair"):
        bundle_io.paper_restart_bundle_state_from_day_end_packet(packet, base_stomate=None, kjit=4, is_watchout=True)
    packet = _packet()
    packet.fields_by_component["sechiba_finalize_state"].pop("science_92")
    with pytest.raises(ValueError, match="denominator must be 93"):
        bundle_io.paper_restart_bundle_state_from_day_end_packet(packet, base_stomate=None, kjit=4, is_watchout=True)


def _layout():
    entry = PFTCatalogEntry("bare", 1, 1, "bare_soil", "source_validated", {}, "source", (), ())
    return PFTRunLayout("synthetic", "one", (entry,), (1.0,))


@pytest.mark.parametrize("watchout", (False, True))
def test_actual_bundle_writer_uses_persisted_mode(tmp_path, monkeypatch, watchout):
    def report(path, **kwargs):
        return StomateRestartWriteReport(Path(path), (), (), ())

    monkeypatch.setattr(bundle_io, "write_sechiba_restart_state", report)
    monkeypatch.setattr(bundle_io, "write_stomate_full_writerestart_states", report)
    state = bundle_io.PaperRestartBundleState(
        driver=Dim2DriverRestart(**{name: value for name, value in _values().items() if not watchout or name in RETAINED}),
        sechiba=None,
        stomate=SimpleNamespace(**{name: None for name in ("entry_state", "season_state", "daily_state", "gas_state", "remainder_state")}),
        is_watchout=watchout,
    )
    state = pickle.loads(pickle.dumps(state))
    physical = bundle_io.PaperRestartBundlePhysicalState(_physical(), _physical(), _physical())
    written = bundle_io.write_paper_restart_start_bundle(tmp_path / "bundle", state=state, physical_state=physical, pft_layout=_layout())
    with Dataset(written.driver.output_path) as dataset:
        assert bool(set(COEFFICIENTS) & set(dataset.variables)) is (not watchout)
    restored = read_dim2_driver_restart(written.driver.output_path, is_watchout=watchout)
    assert (restored.peqAcoef is None) is watchout


def test_old_bundle_constructor_and_pickle_default_to_nonwatchout():
    state = bundle_io.PaperRestartBundleState(Dim2DriverRestart(), None, None)
    assert state.is_watchout is False
    # Old pickle dictionaries have no mode field; the dataclass's class default remains available.
    object.__delattr__(state, "is_watchout")
    assert pickle.loads(pickle.dumps(state)).is_watchout is False


@pytest.mark.parametrize("watchout", (False, True))
def test_stage5_actual_export_adapter_uses_forcing_schema(monkeypatch, watchout):
    from scripts.dev import audit_stage5_lifecycle as caller

    captured = {}
    monkeypatch.setattr(caller, "paper_restart_bundle_state_from_day_end_packet", lambda *args, **kwargs: captured.update(kwargs))
    context = SimpleNamespace(
        first_step_restart_state=SimpleNamespace(stomate_readstart="base"),
        first_step_bundle=SimpleNamespace(forcing=SimpleNamespace(is_watchout=watchout)),
        run_def_values={"WATCHOUT": not watchout},
    )
    caller._restart_bundle_from_context("packet", context, kjit=4)
    assert captured.get("is_watchout", False) is watchout


@pytest.mark.parametrize("watchout", (False, True))
def test_stage5_readback_adapter_keeps_forcing_coefficients_out_of_packet(tmp_path, monkeypatch, watchout):
    from scripts.dev import audit_stage5_lifecycle as caller

    source = _source_file(tmp_path, coefficients=not watchout)
    driver = read_dim2_driver_restart(source, is_watchout=watchout)
    components = {name: {} for name in (
        "driver_previous_step_state", "sechiba_finalize_state", "diffuco_previous_step_state",
        "enerbil_previous_step_state", "hydrol_previous_step_state", "thermosoil_previous_step_state",
        "slowproc_stomate_previous_step_state",
    )}
    template = SimpleNamespace(tstep=4, fields_by_component=components, provenance_by_component={})
    monkeypatch.setattr(caller, "sechiba_finalize_source_state_from_restart", lambda _: {})
    group = SimpleNamespace(_asdict=lambda: {})
    restored = caller._packet_from_restart_readers(
        template, driver=driver, sechiba=None,
        stomate=SimpleNamespace(entry_state=group, season_state=group, daily_state=group),
        is_watchout=watchout,
    )
    fields = restored.fields_by_component["driver_previous_step_state"]
    assert set(COEFFICIENTS).issubset(fields) is (not watchout)
    assert not components["driver_previous_step_state"]
    for name in RETAINED:
        np.testing.assert_array_equal(fields[name], _values()[name])


@pytest.mark.parametrize("watchout", (False, True))
def test_compiled_split_actual_adapter_carries_mode_without_running_audit(tmp_path, monkeypatch, watchout):
    import importlib

    import jax_orchidee.runtime as runtime

    monkeypatch.setattr(runtime, "configure_jax_compilation_cache", lambda *_: None)
    caller = importlib.import_module("scripts.dev.audit_compiled_fast_path_acceptance")
    context = SimpleNamespace(
        first_step_bundle=SimpleNamespace(forcing=SimpleNamespace(is_watchout=watchout)),
        first_step_restart_state=SimpleNamespace(stomate_readstart="base", pft_layout=_layout()),
        run_def_values={"WATCHOUT": not watchout},
    )
    monkeypatch.setattr(caller, "prepare_paper_1961_driver_context", lambda *args, **kwargs: context)
    seen = {}

    def export(packet, **kwargs):
        seen["export_mode"] = kwargs.get("is_watchout", False)
        return SimpleNamespace(
            is_watchout=seen["export_mode"],
            driver=Dim2DriverRestart(**{name: value for name, value in _values().items() if not watchout or name in RETAINED}),
        )

    def write(output, *, state, **kwargs):
        output.mkdir()
        write_dim2_driver_restart(
            output / "driver_start.nc", state=state.driver, physical_state=_physical(),
            is_watchout=state.is_watchout,
        )
        (output / "stomate_start.nc").write_bytes(b"transport-only placeholder")
        return SimpleNamespace(output_directory=output)

    def readback(template, *, driver, is_watchout, **kwargs):
        seen["readback_mode"] = is_watchout
        seen["driver"] = driver
        return "restored_packet"

    monkeypatch.setattr(caller, "paper_restart_bundle_state_from_day_end_packet", export)
    monkeypatch.setattr(caller, "read_restart_physical_state", lambda _: _physical())
    monkeypatch.setattr(caller, "write_paper_restart_start_bundle", write)
    monkeypatch.setattr(caller, "reference_case_first_step_restart_state", lambda *args, **kwargs: SimpleNamespace(sechiba_restart_state=None, stomate_readstart=None))
    monkeypatch.setattr(caller, "_packet_from_restart_readers", readback)
    restored, output = caller._split_restart_packet(
        SimpleNamespace(output_dir=tmp_path, landpoint_id="synthetic"),
        SimpleNamespace(tstep=4), tmp_path / "split", used_run_def_path=tmp_path / "used.def",
    )
    assert restored == "restored_packet"
    assert seen["export_mode"] is watchout
    assert seen["readback_mode"] is watchout
    assert (seen["driver"].peqAcoef is None) is watchout
    with Dataset(output / "driver_start.nc") as dataset:
        assert bool(set(COEFFICIENTS) & set(dataset.variables)) is (not watchout)
