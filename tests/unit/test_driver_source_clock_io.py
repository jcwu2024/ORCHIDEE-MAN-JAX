"""Synthetic metadata contracts for the declared IOIPSL source clock.

These tests exercise host-side NetCDF metadata and restart serialization only;
they do not construct or advance a physical model state.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from netCDF4 import Dataset

from jax_orchidee.driver.dim2 import Dim2RestartTime, resolve_dim2_time_control
from jax_orchidee.driver.restart_bundle import (
    PaperRestartBundlePhysicalState,
    PaperRestartBundleState,
    read_restart_physical_state,
    paper_restart_bundle_state_from_day_end_packet,
    write_paper_restart_start_bundle,
)
from jax_orchidee.driver.restart_file_io import read_dim2_driver_restart
from jax_orchidee.driver.restart_state import reference_case_first_step_restart_state
from jax_orchidee.driver.source_clock import (
    PREFIX,
    SourceClockBinding,
    SourceClockError,
    SourceClockRuntime,
    read_forcing_clock_metadata,
    read_restart_clock_metadata,
    read_source_clock_restart_bundle,
    rebase_source_clock_runtime,
    source_clock_global_attributes,
    source_clock_from_inputs,
    source_clock_variable_attributes,
    source_clock_setup_from_config,
    source_istp_for_local_tstep,
)
from jax_orchidee.sechiba.ioipsl_calendar import IOIPSLCalendar
from jax_orchidee.sechiba.restart_io import sechiba_finalize_source_state_from_restart

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "orchidee_man_250919.yaml"


def _origin(calendar: str = "noleap", year: int = 1961) -> str:
    return f"{year:04d}-01-01 00:00:00"


def _binding(*, run_id: str = "run-a", dt: float = 3600.0) -> SourceClockBinding:
    calendar = IOIPSLCalendar("noleap")
    date0 = calendar.ymds2ju(1961, 1, 1, 0)
    lock = IOIPSLCalendar.from_scalar_startdate("noleap", date0)
    return SourceClockBinding(
        run_id=run_id,
        calendar="noleap",
        date0=date0,
        dt=dt,
        startdate_day=lock.startdate_day,
        startdate_second=lock.startdate_second,
        initialization_istp=0,
    )


def _write_forcing(
    path: Path,
    *,
    calendar: str | None = "noleap",
    timestep_axis: bool = False,
    conflicting_origin: bool = False,
    steps_calendar: str = "noleap",
    physical_first_axis_value: float = 0.0,
    timestep_first_axis_value: int = 0,
    count: int = 8,
) -> Path:
    with Dataset(path, "w") as dataset:
        dataset.createDimension("time", count)
        time = dataset.createVariable("time", "f8", ("time",))
        time_origin = _origin(year=1961)
        time.units = f"seconds since {time_origin}"
        time.time_origin = _origin(year=1962) if conflicting_origin else time_origin
        if calendar is not None:
            time.calendar = calendar
        time[:] = physical_first_axis_value + np.arange(count, dtype=np.float64) * 7200.0
        if timestep_axis:
            steps = dataset.createVariable("time_steps", "i4", ("time",))
            steps.units = f"timesteps since {_origin(year=1962)}"
            steps.time_origin = _origin(year=1962)
            steps.tstep_sec = 7200.0
            steps.calendar = steps_calendar
            steps[:] = timestep_first_axis_value + np.arange(count, dtype=np.int32)
    return path


def _write_restart(
    path: Path,
    runtime: SourceClockRuntime,
    *,
    last_applied_istp: int | None = None,
    runtime_override: SourceClockRuntime | None = None,
) -> Path:
    binding = runtime.binding
    cursor = int(runtime.last_applied_istp if last_applied_istp is None else last_applied_istp)
    with Dataset(path, "w") as dataset:
        dataset.createDimension("time", 1)
        physical = dataset.createVariable("time", "f8", ("time",))
        physical_attrs = source_clock_variable_attributes(binding)["time"]
        physical.setncatts(physical_attrs)
        physical[:] = [cursor * binding.dt]
        steps = dataset.createVariable("time_steps", "i4", ("time",))
        steps.setncatts(source_clock_variable_attributes(binding)["time_steps"])
        steps[:] = [cursor]
        attrs = source_clock_global_attributes(runtime_override or runtime)
        dataset.setncatts(attrs)
    return path


def _reference_run() -> Path:
    root = ROOT / "reference" / "OUT" / "orc_calibrate_250919_sen" / "arg2_1.0" / "001.0-071.0"
    candidates = sorted(path.parent for path in root.rglob("driver_start.nc"))
    if not candidates:
        pytest.skip("paper restart triplet is unavailable")
    return candidates[0]


def test_forcing_timestep_axis_precedes_physical_axis_and_owns_its_origin(tmp_path: Path):
    forcing = _write_forcing(tmp_path / "forcing.nc", timestep_axis=True)

    metadata = read_forcing_clock_metadata(forcing)

    expected = IOIPSLCalendar("noleap").ymds2ju(1962, 1, 1, 0)
    assert metadata.selected_axis == "time_steps"
    assert metadata.date0 == expected
    assert metadata.dt_force == 7200.0
    assert metadata.count == 8
    assert metadata.origin_ymds == (1962, 1, 1, 0.0)


def test_forcing_uses_last_seconds_coordinate_in_netcdf_order(tmp_path: Path):
    path = tmp_path / "forcing.nc"
    with Dataset(path, "w") as dataset:
        dataset.createDimension("tstep", 8)
        # The variable name does not determine precedence.
        for name, offset in (("timeplussix", 0), ("time", 21600)):
            coordinate = dataset.createVariable(name, "f8", ("tstep",))
            coordinate.units = f"seconds since {_origin()}"
            coordinate.calendar = "noleap"
            coordinate[:] = offset + np.arange(8) * 21600

    setup = source_clock_from_inputs(path, dt_sechiba=1800, run_id="cold", time_length="2D")

    assert setup.forcing.selected_axis == "time"
    assert setup.forcing.first_axis_value == 21600
    assert setup.forcing.dt_force == 21600
    assert setup.forcing.count == 8
    assert setup.forcing.origin_ymds == (1961, 1, 1, 0.0)
    assert setup.binding.date0 == IOIPSLCalendar("noleap").ymds2ju(1961, 1, 1, 0)
    assert setup.binding.clock_at_istp(1).second == 1800
    assert setup.time_control.itau_fin == 8


def test_forcing_last_timestep_coordinate_uses_last_seconds_calendar(tmp_path: Path):
    path = _write_forcing(tmp_path / "forcing.nc", calendar="gregorian", steps_calendar="gregorian",
                          timestep_axis=True)
    with Dataset(path, "a") as dataset:
        steps = dataset.createVariable("driver_index", "i4", ("time",))
        steps.units = f"timesteps since {_origin(year=1963)}"
        steps.tstep_sec = 3600.0
        steps[:] = 12 + np.arange(8)
        physical = dataset.createVariable("time_shifted", "f8", ("time",))
        physical.units = f"seconds since {_origin(year=1981)}"
        physical.calendar = "noleap"
        physical[:] = 21600 + np.arange(8) * 7200

    metadata = read_forcing_clock_metadata(path)

    assert metadata.selected_axis == "driver_index"
    assert metadata.calendar == "noleap"
    assert metadata.origin_ymds == (1963, 1, 1, 0.0)
    assert metadata.date0 == IOIPSLCalendar("noleap").ymds2ju(1963, 1, 1, 0)
    assert metadata.first_axis_value == 12
    assert metadata.dt_force == 3600


def test_forcing_rejects_malformed_selected_seconds_coordinate(tmp_path: Path):
    path = _write_forcing(tmp_path / "forcing.nc")
    with Dataset(path, "a") as dataset:
        physical = dataset.createVariable("time_shifted", "f8", ("time",))
        physical.units = f"seconds since {_origin()}"
        physical.calendar = "noleap"
        physical[:] = np.arange(8) ** 2 * 7200

    with pytest.raises(SourceClockError, match="forcing time axis is not regular"):
        read_forcing_clock_metadata(path)


@pytest.mark.parametrize("units", ("legacy", "timesteps since 1961-01-01 00:00:00"))
def test_forcing_cold_entry_requires_seconds_coordinate_calendar(tmp_path: Path, units: str):
    path = tmp_path / "forcing.nc"
    with Dataset(path, "w") as dataset:
        dataset.createDimension("time", 8)
        coordinate = dataset.createVariable("tstep", "i4", ("time",))
        coordinate.units = units
        coordinate.calendar = "noleap"
        coordinate.tstep_sec = 7200
        coordinate[:] = np.arange(8)

    with pytest.raises(SourceClockError, match="requires a seconds-since time axis"):
        read_forcing_clock_metadata(path)


@pytest.mark.parametrize("duplicate_axis", ("time", "time_steps"))
def test_restart_rejects_duplicate_axis_even_in_netcdf_order(tmp_path: Path, duplicate_axis: str):
    runtime = SourceClockRuntime(_binding(), np.int64(4), np.int64(0), np.int64(1))
    path = _write_restart(tmp_path / "restart.nc", runtime)
    with Dataset(path, "a") as dataset:
        original = dataset.variables[duplicate_axis]
        duplicate = dataset.createVariable("duplicate", original.dtype, ("time",))
        duplicate.setncatts({name: original.getncattr(name) for name in original.ncattrs()})
        duplicate[:] = original[:]

    with pytest.raises(SourceClockError, match="missing/ambiguous physical time axis or multiple time_steps axes"):
        read_restart_clock_metadata(path)


def test_declared_config_factory_reads_actual_forcing_and_cold_restart_controls(tmp_path: Path, monkeypatch):
    # The deliberately different filename year selects a file, never its date.
    forcing = _write_forcing(tmp_path / "forcing-2099.nc", timestep_axis=True, count=32)
    from jax_orchidee.driver import domain
    monkeypatch.setattr(domain, "load_case_config", lambda _: {
        "drivers": {"atmospheric_forcing": {"file_pattern": str(tmp_path / "forcing-{year}.nc")}}
    })
    values = {"DT_SECHIBA": "3600", "RESTART_FILEIN": "NONE",
              "SECHIBA_restart_in": "NONE", "STOMATE_RESTART_FILEIN": "NONE",
              "TIME_SKIP": "1D", "TIME_LENGTH": "1D"}
    setup = source_clock_setup_from_config(tmp_path / "case.yaml", run_def_values=values,
                                          first_step_bundle=SimpleNamespace(year=2099), run_id="declared-cold")
    assert forcing.is_file()
    assert setup.forcing.origin_ymds == (1962, 1, 1, 0.0)
    assert setup.binding.date0 == IOIPSLCalendar("noleap").ymds2ju(1962, 1, 1, 0.0)
    assert setup.time_control.itau_dep == 12
    assert setup.time_control.istp == 25
    assert setup.binding.initialization_istp == 0
    with pytest.raises(SourceClockError, match="missing/conflicting declared run.def STOMATE_RESTART_FILEIN"):
        source_clock_setup_from_config(tmp_path / "case.yaml",
            run_def_values={key: value for key, value in values.items() if key != "STOMATE_RESTART_FILEIN"},
            first_step_bundle=SimpleNamespace(year=2099), run_id="missing-input")


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"calendar": None}, "missing calendar"),
        ({"conflicting_origin": True}, "conflicts with its units"),
    ],
)
def test_forcing_clock_requires_declared_calendar_and_consistent_origin(
    tmp_path: Path, kwargs, match: str
):
    forcing = _write_forcing(tmp_path / "forcing.nc", **kwargs)

    with pytest.raises(SourceClockError, match=match):
        read_forcing_clock_metadata(forcing)


def test_forcing_clock_rejects_conflicting_calendar_on_optional_steps_axis(tmp_path: Path):
    forcing = _write_forcing(
        tmp_path / "forcing.nc", calendar="gregorian", timestep_axis=True,
        steps_calendar="noleap",
    )

    with pytest.raises(SourceClockError, match="time_steps calendar conflicts"):
        read_forcing_clock_metadata(forcing)


@pytest.mark.parametrize(
    "timestep_axis,physical_first,timestep_first,axis,expected_origin,expected_first",
    [
        (False, 3600.0, 0, "time", (1961, 1, 1, 0.0), 3600.0),
        (True, 3600.0, 5, "time_steps", (1962, 1, 1, 0.0), 5.0),
    ],
)
def test_nonzero_first_axis_value_does_not_shift_declared_units_origin(
    tmp_path: Path,
    timestep_axis: bool,
    physical_first: float,
    timestep_first: int,
    axis: str,
    expected_origin: tuple,
    expected_first: float,
):
    forcing = _write_forcing(
        tmp_path / "forcing.nc",
        timestep_axis=timestep_axis,
        physical_first_axis_value=physical_first,
        timestep_first_axis_value=timestep_first,
    )

    metadata = read_forcing_clock_metadata(forcing)

    assert metadata.selected_axis == axis
    assert metadata.origin_ymds == expected_origin
    assert metadata.date0 == IOIPSLCalendar("noleap").ymds2ju(*expected_origin)
    assert metadata.first_axis_value == expected_first


def test_restart_writer_roundtrips_one_locked_clock_across_three_real_netcdf_files(
    tmp_path: Path,
):
    binding = _binding()
    runtime = SourceClockRuntime(
        binding,
        np.int64(4),
        np.int64(7),
        np.int64(5),
    )
    run = _reference_run()
    original = reference_case_first_step_restart_state(
        CONFIG, root=ROOT, run_dir=run, stomate_filename="stomate_start.nc"
    )
    driver = read_dim2_driver_restart(run / "driver_start.nc")
    final = sechiba_finalize_source_state_from_restart(original.sechiba_restart_state)
    slow = {name: np.array(final[name], copy=True) for name in ("growth_day", "peatC_ok")}
    slow["daily_accumulators"] = original.stomate_readstart.daily_state._asdict()
    packet = SimpleNamespace(
        tstep=6,  # admitted affine map: anchor(7,5) -> local6/source4
        fields_by_component={
            "driver_previous_step_state": {name: getattr(driver, name) for name in driver.__dataclass_fields__},
            "sechiba_finalize_state": final,
            "slowproc_stomate_previous_step_state": slow,
        },
        source_clock_runtime=runtime,
    )
    state = paper_restart_bundle_state_from_day_end_packet(packet, base_stomate=original.stomate_readstart, kjit=4)
    physical = PaperRestartBundlePhysicalState(**{
        name: read_restart_physical_state(run / f"{name}_start.nc")
        for name in ("driver", "sechiba", "stomate")
    })

    report = write_paper_restart_start_bundle(
        tmp_path / "source_clock_handoff",
        state=state,
        physical_state=physical,
        pft_layout=original.pft_layout,
    )
    bundle = read_source_clock_restart_bundle(report.output_paths)
    restored = reference_case_first_step_restart_state(
        CONFIG,
        root=ROOT,
        run_dir=report.output_directory,
        stomate_filename="stomate_start.nc",
    )

    assert bundle.runtime == runtime
    assert restored.source_clock_runtime == runtime
    restored_driver = read_dim2_driver_restart(report.driver.output_path)
    assert len(original.sechiba_restart_state.fields) == 93
    for name in state.driver.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(restored_driver, name), getattr(state.driver, name))
    for name, expected in original.sechiba_restart_state.fields.items():
        np.testing.assert_array_equal(restored.sechiba_restart_state.fields[name], expected)
    np.testing.assert_array_equal(restored.stomate.age, original.stomate.age)
    for name in ("growth_day", "peatC_ok"):
        # There is one authoritative persisted SECHIBA value. Export validates
        # both RAM mirrors against it rather than writing divergent copies.
        np.testing.assert_array_equal(restored.sechiba_restart_state.fields[name], slow[name])
        np.testing.assert_array_equal(restored.sechiba_restart_state.fields[name], final[name])
    for path in report.output_paths:
        with Dataset(path) as dataset:
            assert dataset.getncattr(PREFIX + "binding_sha256") == binding.binding_key
            assert dataset.variables["time"].units == source_clock_variable_attributes(binding)["time"]["units"]
            assert dataset.variables["time"].calendar == "noleap"
            assert dataset.variables["time_steps"].tstep_sec == binding.dt
    assert read_restart_clock_metadata(report.driver.output_path).itau == 4


@pytest.mark.parametrize("fault", ["growth_day", "peatC_ok", "missing_mirror", "local_cursor", "source_kjit"])
def test_versioned_packet_export_rejects_mirror_and_cursor_conflicts_before_finalize(fault):
    # Only the export admission runs. Dummy science is deliberately unsuitable
    # for a physical finalizer so a successful negative cannot hide execution.
    runtime = SourceClockRuntime(_binding(), np.int64(4), np.int64(7), np.int64(5))
    slow = {"growth_day": np.asarray([3.0]), "peatC_ok": np.asarray([1.0])}
    final = {name: value.copy() for name, value in slow.items()}
    packet = SimpleNamespace(tstep=6, source_clock_runtime=runtime, fields_by_component={
        "slowproc_stomate_previous_step_state": slow, "sechiba_finalize_state": final,
    })
    kjit = 4
    if fault in ("growth_day", "peatC_ok"):
        final[fault] = final[fault] + 1
        error = f"restart calendar science mirrors conflict: {fault}"
    elif fault == "missing_mirror":
        slow.pop("growth_day")
        error = "missing calendar science mirror: growth_day"
    elif fault == "local_cursor":
        packet.tstep = 7
        error = "local tstep conflicts with source clock cursor"
    else:
        kjit = 5
        error = "source kjit conflicts with source clock cursor"
    with pytest.raises(SourceClockError, match=error):
        paper_restart_bundle_state_from_day_end_packet(packet, base_stomate=None, kjit=kjit)


def test_restart_metadata_rejects_axis_runtime_and_cross_file_conflicts(tmp_path: Path):
    binding = _binding()
    runtime = SourceClockRuntime(binding, np.int64(4), np.int64(0), np.int64(5))
    good = _write_restart(tmp_path / "good.nc", runtime)
    bad_axis = _write_restart(tmp_path / "bad-axis.nc", runtime, last_applied_istp=5)
    other_runtime = SourceClockRuntime(
        SourceClockBinding(
            run_id="run-b",
            calendar=binding.calendar,
            date0=binding.date0,
            dt=binding.dt,
            startdate_day=binding.startdate_day,
            startdate_second=binding.startdate_second,
            initialization_istp=binding.initialization_istp,
        ),
        np.int64(4),
        np.int64(0),
        np.int64(5),
    )
    other = _write_restart(tmp_path / "other.nc", runtime, runtime_override=other_runtime)

    with pytest.raises(SourceClockError, match="selected axis conflicts"):
        read_restart_clock_metadata(bad_axis)
    with pytest.raises(SourceClockError, match="three-file restart.*conflicts"):
        read_source_clock_restart_bundle((good, good, other))


def test_legacy_restart_requires_explicit_new_native_metadata_policy(tmp_path: Path):
    runtime = SourceClockRuntime(_binding(), np.int64(4), np.int64(0), np.int64(5))
    legacy = _write_restart(tmp_path / "legacy.nc", runtime)
    with Dataset(legacy, "r+") as dataset:
        for name in tuple(dataset.ncattrs()):
            if name.startswith(PREFIX):
                dataset.delncattr(name)

    with pytest.raises(SourceClockError, match="lacks versioned source clock runtime"):
        read_restart_clock_metadata(legacy)
    metadata = read_restart_clock_metadata(legacy, legacy_policy="native_metadata_v1")
    assert metadata.runtime is None
    assert metadata.itau == 4


def test_three_file_restart_rejects_missing_clock_field_and_mixed_legacy_member(tmp_path: Path):
    runtime = SourceClockRuntime(_binding(), np.int64(4), np.int64(0), np.int64(5))
    first = _write_restart(tmp_path / "driver.nc", runtime)
    second = _write_restart(tmp_path / "sechiba.nc", runtime)
    third = _write_restart(tmp_path / "stomate.nc", runtime)

    with Dataset(second, "r+") as dataset:
        dataset.delncattr(PREFIX + "anchor_local_tstep")
    with pytest.raises(SourceClockError, match="incomplete/unsupported versioned"):
        read_source_clock_restart_bundle((first, second, third))

    with Dataset(second, "r+") as dataset:
        dataset.setncattr(PREFIX + "anchor_local_tstep", np.int64(0))
    with Dataset(third, "r+") as dataset:
        for name in tuple(dataset.ncattrs()):
            if name.startswith(PREFIX):
                dataset.delncattr(name)
    with pytest.raises(SourceClockError, match="legacy restart lacks versioned"):
        read_source_clock_restart_bundle((first, second, third))


def test_source_clock_lifecycle_rescales_restart_resets_and_requires_new_run_id(
    tmp_path: Path,
):
    forcing = _write_forcing(tmp_path / "forcing.nc")
    binding = _binding()
    previous = SourceClockRuntime(
        binding,
        np.int64(4),
        np.int64(0),
        np.int64(5),
    )
    restart_path = _write_restart(tmp_path / "restart.nc", previous)
    restart_paths = (restart_path, restart_path, restart_path)

    restarted = source_clock_from_inputs(
        forcing,
        dt_sechiba=3600.0,
        run_id="run-b",
        restart_paths=restart_paths,
        time_length=1,
    )
    assert restarted.time_control.dt_rest == 3600.0
    assert restarted.time_control.itau_dep == 2
    assert restarted.time_control.istp == 5
    assert int(restarted.runtime.last_applied_istp) == 4
    assert restarted.binding.run_id == "run-b"

    reset = source_clock_from_inputs(
        forcing,
        dt_sechiba=3600.0,
        run_id="run-c",
        restart_paths=restart_paths,
        driver_reset_time=True,
        time_length=1,
    )
    assert reset.time_control.itau_dep == 0
    assert reset.time_control.istp == 1
    assert int(reset.runtime.last_applied_istp) == 0

    with pytest.raises(SourceClockError, match="new run_id"):
        source_clock_from_inputs(
            forcing,
            dt_sechiba=3600.0,
            run_id="run-a",
            previous_runtime=previous,
            time_length=1,
        )


def test_continuous_clock_advances_without_rebinding_and_local_rebase_preserves_source(
    tmp_path: Path,
):
    forcing = _write_forcing(tmp_path / "forcing.nc")
    initial = source_clock_from_inputs(
        forcing, dt_sechiba=7200.0, run_id="continuous", time_length=1
    )
    previous = SourceClockRuntime(
        initial.binding,
        np.int64(1),
        initial.runtime.anchor_local_tstep,
        initial.runtime.anchor_source_istp,
    )

    continued = source_clock_from_inputs(
        forcing,
        dt_sechiba=7200.0,
        run_id="continuous",
        run_mode="continuous",
        previous_runtime=previous,
        time_length=1,
    )
    assert continued.binding == initial.binding
    assert continued.time_control.istp == 2
    assert int(continued.runtime.last_applied_istp) == 1

    rebased = rebase_source_clock_runtime(previous, new_local_tstep=23)
    assert int(rebased.anchor_local_tstep) == 23
    assert int(rebased.anchor_source_istp) == 1
    assert int(source_istp_for_local_tstep(rebased, 25)) == 3


def test_continuous_gregorian_year_retains_first_call_constants_and_new_run_reinitializes(tmp_path: Path):
    def annual(year):
        path = tmp_path / f"calendar-{year}.nc"
        with Dataset(path, "w") as dataset:
            dataset.createDimension("time", 3)
            time = dataset.createVariable("time", "f8", ("time",))
            time.units = f"seconds since {year}-01-01 00:00:00"
            time.calendar = "gregorian"
            time[:] = np.arange(3) * 86400.0
        return path
    first = source_clock_from_inputs(annual(2000), dt_sechiba=86400, run_id="leap-run", time_length=1)
    end = SourceClockRuntime(first.binding, np.int64(366), np.int64(0), np.int64(1))
    continued = source_clock_from_inputs(annual(2001), dt_sechiba=86400, run_id="leap-run",
        run_mode="continuous", previous_runtime=end, time_length=1)
    # A separately declared restart origin must establish its own lifecycle;
    # it must not unconditionally inherit the previous process's leap lock.
    lock = IOIPSLCalendar.from_ymds("gregorian", 2001, 1, 1)
    restart_binding = SourceClockBinding("declared-2001-producer", "gregorian",
        lock.ymds2ju(2001, 1, 1), 86400.0, lock.startdate_day, lock.startdate_second, 0)
    restart = _write_restart(tmp_path / "year-2001-restart.nc",
        SourceClockRuntime(restart_binding, np.int64(0), np.int64(0), np.int64(1)))
    fresh = source_clock_from_inputs(tmp_path / "calendar-2001.nc", dt_sechiba=86400,
        run_id="next-native-run", previous_runtime=end, restart_paths=(restart,) * 3, time_length=1)
    assert continued.binding == first.binding
    assert continued.binding.year_length == 366
    assert continued.binding.clock_at_istp(367).year_length == 366
    assert continued.time_control.istp == 367
    assert continued.time_control.for_offset == -366
    assert fresh.binding.year_length == 365
    assert fresh.binding.startdate_day != first.binding.startdate_day
    assert fresh.binding.run_id != first.binding.run_id


@pytest.mark.parametrize("year,expected_steps", [(2000, 366), (2001, 365)])
def test_gregorian_time_length_one_year_uses_date_aware_duration(
    year: int, expected_steps: int
):
    calendar = IOIPSLCalendar("gregorian")
    date0 = calendar.ymds2ju(year, 1, 1, 0)

    control = resolve_dim2_time_control(
        dt_force=86400,
        dt_sechiba=86400,
        forcing_length=0,
        date0=date0,
        time_length="1Y",
        calendar="gregorian",
    )

    assert control.itau_len == expected_steps


def test_gregorian_combined_duration_rounds_each_component_and_itau_one_is_not_rescaled():
    calendar = IOIPSLCalendar("gregorian")
    date0 = calendar.ymds2ju(2001, 1, 1, 0)
    length = resolve_dim2_time_control(
        dt_force=172800,
        dt_sechiba=172800,
        forcing_length=0,
        date0=date0,
        time_length="1Y1D",
        calendar="gregorian",
    )
    boundary = resolve_dim2_time_control(
        dt_force=7200,
        dt_sechiba=3600,
        forcing_length=0,
        date0=date0,
        restart=Dim2RestartTime(1, date0, 3600),
        time_length=0,
        calendar="gregorian",
    )

    assert length.itau_len == 184
    assert boundary.itau_dep == 1
    assert boundary.istp == 3
