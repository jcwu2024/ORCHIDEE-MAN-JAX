"""Declared IOIPSL clock binding and versioned restart metadata.

Source: IOIPSL calendar.f90136-466/718-783, flincom.f90581-665,
restcom.f90178-400/636-829; DIM2 driver361-401/464-529. The accepted
20261114 native scalar fixture establishes the provider, not original001.
No physical process, calendar assignment schedule, or inferred local year is
implemented here. All filesystem access is ordinary NetCDF metadata IO.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from os import PathLike
import re
from typing import Any, Mapping

import numpy as np
from netCDF4 import Dataset
from jax import tree_util

from jax_orchidee.sechiba.ioipsl_calendar import IOIPSLCalendar, fortran_nint
from jax_orchidee.sechiba.lifecycle_completion import IntersurfTimeState, intersurf_time

AUTHORITY = "ioipsl-calendar-4143df77-native-127957445-v1"
PREFIX = "orchidee_jax_source_clock_"
VERSION = 1
LEGACY_NATIVE_METADATA = "native_metadata_v1"


class SourceClockError(ValueError):
    """A declared source clock is missing, conflicting, or unsupported."""


@dataclass(frozen=True)
class SourceClockBinding:
    """One immutable native-run lock, separate from forcing-file indexing."""

    run_id: str
    calendar: str
    date0: float
    dt: float
    startdate_day: int
    startdate_second: float
    initialization_istp: int
    authority: str = AUTHORITY
    schema_version: int = VERSION

    def __post_init__(self):
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise SourceClockError("source clock requires an explicit nonempty run_id")
        if self.authority != AUTHORITY or self.schema_version != VERSION:
            raise SourceClockError("unsupported source clock provider/schema identity")
        if self.calendar not in {"gregorian", "noleap"}:
            raise SourceClockError("source clock binding requires canonical gregorian/noleap calendar")
        if not math.isfinite(self.date0) or not math.isfinite(self.dt) or self.dt <= 0:
            raise SourceClockError("source clock date0/dt must be finite, with dt positive")
        if int(self.initialization_istp) != self.initialization_istp:
            raise SourceClockError("initialization_istp must be an integer")
        self.provider  # validate the explicitly independent split lock

    @property
    def provider(self) -> IOIPSLCalendar:
        return IOIPSLCalendar(self.calendar, self.startdate_day, self.startdate_second)

    @property
    def binding_key(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf8")).hexdigest()

    @property
    def one_year_days(self) -> float:
        return self.provider.one_year_days

    @property
    def one_day_seconds(self) -> float:
        return self.provider.one_day_seconds

    @property
    def year_length(self) -> int:
        return self.provider.year_steps(self.dt, self.date0)

    @property
    def year_spread(self) -> float:
        return 1.0 if self.calendar == "gregorian" else self.one_year_days / 365.2425

    def clock_at_istp(self, istp: int, previous: IntersurfTimeState | None = None) -> IntersurfTimeState:
        """Host-side owner payload, seeded by this run's initialization call.

        A caller-supplied previous state is always validated by the accepted
        immutable helper. A new run is represented by a new binding/run_id,
        never by dropping previous state at an ordinary year boundary.
        """
        kwargs = dict(date0=self.date0, dt_sechiba=self.dt, calendar=self.calendar,
                      one_year=self.one_year_days, one_day=self.one_day_seconds,
                      clock=self.provider, date0_convention="ioipsl_calendar")
        if previous is None:
            previous = intersurf_time(istp=self.initialization_istp, **kwargs)
        return intersurf_time(istp=istp, previous=previous, **kwargs)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False)

    @classmethod
    def from_json(cls, value: str) -> SourceClockBinding:
        try:
            data = json.loads(value)
            if set(data) != set(cls.__dataclass_fields__):
                raise SourceClockError("source clock binding fields/version are incomplete or unknown")
            return cls(**data)
        except (TypeError, json.JSONDecodeError) as exc:
            raise SourceClockError("invalid source clock binding JSON") from exc


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SourceClockRuntime:
    """Three numeric leaves; immutable run binding is pytree auxiliary data.

    The initial cursor is a DIM2 admission boundary (first active istp-1),
    not a claim that skipped initialization steps executed slowproc_main.
    """

    binding: SourceClockBinding
    last_applied_istp: Any
    anchor_local_tstep: Any
    anchor_source_istp: Any

    @property
    def binding_key(self) -> str:
        return self.binding.binding_key

    def tree_flatten(self):
        return (self.last_applied_istp, self.anchor_local_tstep, self.anchor_source_istp), self.binding

    @classmethod
    def tree_unflatten(cls, binding, children):
        return cls(binding, *children)


def _integer(value: Any, label: str) -> int:
    try:
        a = np.asarray(value)
        if a.shape != () or not np.isfinite(a).item() or int(a) != a.item():
            raise SourceClockError(f"{label} must be a finite integer scalar")
        return int(a)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SourceClockError(f"{label} must be a finite integer scalar") from exc


def validate_source_clock_runtime(runtime: SourceClockRuntime, binding: SourceClockBinding | None = None) -> None:
    """Host admission only; never branch on tracers inside a compiled scan."""
    if not isinstance(runtime, SourceClockRuntime) or not isinstance(runtime.binding, SourceClockBinding):
        raise SourceClockError("missing typed source clock runtime/binding")
    if binding is not None and runtime.binding != binding:
        raise SourceClockError("source clock binding changed within a native run")
    for name in ("last_applied_istp", "anchor_local_tstep", "anchor_source_istp"):
        _integer(getattr(runtime, name), name)


def source_istp_for_local_tstep(runtime: SourceClockRuntime, tstep: Any):
    """Explicit affine frame; no modulo/day/year inference (DIM2519-529)."""
    return runtime.anchor_source_istp + (tstep - runtime.anchor_local_tstep)


def validate_source_clock_restart_packet(packet, *, source_istp: int | None = None):
    """Check versioned export mirrors and chronology before serialization.

    growth_day/peatC_ok are one slowproc556-610 state mirrored for finalize;
    source kjit is distinct from the packet's local label (DIM2519-529).
    A legacy science-only packet has no clock claim and remains unchanged.
    """
    runtime = getattr(packet, "source_clock_runtime", None)
    if runtime is None:
        return None
    validate_source_clock_runtime(runtime)
    local = _integer(getattr(packet, "tstep", None), "restart packet local tstep")
    cursor = _integer(runtime.last_applied_istp, "last_applied_istp")
    if _integer(source_istp_for_local_tstep(runtime, local), "packet source istp") != cursor:
        raise SourceClockError("restart packet local tstep conflicts with source clock cursor")
    if source_istp is not None and _integer(source_istp, "restart source kjit") != cursor:
        raise SourceClockError("restart source kjit conflicts with source clock cursor")
    components = getattr(packet, "fields_by_component", None)
    if not isinstance(components, Mapping):
        raise SourceClockError("versioned restart packet requires both calendar science mirrors")
    slow = components.get("slowproc_stomate_previous_step_state", {})
    final = components.get("sechiba_finalize_state", {})
    for name in ("growth_day", "peatC_ok"):
        if name not in slow or name not in final:
            raise SourceClockError(f"versioned restart packet is missing calendar science mirror: {name}")
        if not np.array_equal(np.asarray(slow[name]), np.asarray(final[name])):
            raise SourceClockError(f"restart calendar science mirrors conflict: {name}")
    return runtime


def rebase_source_clock_runtime(runtime: SourceClockRuntime, *, new_local_tstep: int) -> SourceClockRuntime:
    """Rebase the current stored packet's local label, preserving its source cursor."""
    validate_source_clock_runtime(runtime)
    return replace(runtime, anchor_local_tstep=np.int64(_integer(new_local_tstep, "new_local_tstep")),
                   anchor_source_istp=np.int64(_integer(runtime.last_applied_istp, "last_applied_istp")))


@contextmanager
def _dataset(source):
    if isinstance(source, (str, bytes, PathLike)):
        with Dataset(source) as dataset:
            yield dataset
    else:
        yield source


def _attrs(value) -> dict:
    if hasattr(value, "ncattrs"):
        return {name: value.getncattr(name) for name in value.ncattrs()}
    return dict(value.attrs)


def _values(variable, label: str) -> np.ndarray:
    value = np.ma.asarray(variable[:])
    if np.any(np.ma.getmaskarray(value)):
        raise SourceClockError(f"{label} contains missing values")
    a = np.asarray(value)
    if a.ndim != 1 or not a.size or not np.all(np.isfinite(a)):
        raise SourceClockError(f"{label} must be a nonempty finite one-dimensional axis")
    return a


_ORIGIN = re.compile(r"^(\d{4})-([A-Za-z]{3}|\d{2})-(\d{2})[ T](\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)$")
_MONTHS = {s: i + 1 for i, s in enumerate(("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"))}


def _origin(value: str, provider: IOIPSLCalendar) -> tuple[int, int, int, float]:
    match = _ORIGIN.fullmatch(str(value).strip())
    if not match:
        raise SourceClockError(f"unsupported clock origin {value!r}; require explicit YYYY-MM-DD HH:MM:SS")
    year, month, day, hour, minute, sec = match.groups()
    month = _MONTHS.get(month.upper(), month)
    y, m, d, h, mm, s = int(year), int(month), int(day), int(hour), int(minute), float(sec)
    if not (1 <= m <= 12 and 1 <= d <= provider.month_length(y, m) and 0 <= h < 24 and 0 <= mm < 60 and 0 <= s < 60):
        raise SourceClockError("invalid clock origin date/time")
    return y, m, d, h * 3600.0 + mm * 60.0 + s


def _axis_origin(attrs: Mapping, prefix: str, provider: IOIPSLCalendar) -> tuple[str, tuple, float]:
    units = str(attrs.get("units", ""))
    marker = prefix + " since "
    if not units.startswith(marker):
        raise SourceClockError(f"missing/unsupported {prefix} variable units")
    date = _origin(units[len(marker):], provider)
    if "time_origin" in attrs and _origin(attrs["time_origin"], provider) != date:
        raise SourceClockError(f"{prefix} time_origin conflicts with its units")
    return units, date, provider.ymds2ju(*date)


def _axes(dataset):
    physical, steps = [], []
    for name, variable in dataset.variables.items():
        units = str(_attrs(variable).get("units", ""))
        if units.startswith("seconds since "):
            physical.append((name, variable))
        if units.startswith("timesteps since "):
            steps.append((name, variable))
    if len(physical) != 1 or len(steps) > 1:
        raise SourceClockError("missing/ambiguous physical time axis or multiple time_steps axes")
    return physical[0], steps[0] if steps else None


@dataclass(frozen=True)
class ForcingClockMetadata:
    calendar: str
    date0: float
    dt_force: float
    count: int
    selected_axis: str
    units: str
    origin_ymds: tuple
    first_axis_value: float


def read_forcing_clock_metadata(source) -> ForcingClockMetadata:
    """flinopen581-665 timestep-axis precedence and variable calendar/origin.

    The first axis value is an observation, never an origin adjustment:
    flinopen614 reads it separately from selected-unit origin636/648.
    readdim2 forcing_info115-121 derives spacing, not a replacement origin.
    """
    with _dataset(source) as ds:
        (_, physical), step_axis = _axes(ds)
        attrs = _attrs(physical)
        if "calendar" not in attrs:
            raise SourceClockError("forcing time variable is missing calendar")
        provider = IOIPSLCalendar(str(attrs["calendar"]))
        _axis_origin(attrs, "seconds", provider)
        if step_axis is not None:
            name, variable = step_axis
            a = _attrs(variable)
            if "calendar" in a and str(a["calendar"]) != provider.calendar:
                raise SourceClockError("forcing time_steps calendar conflicts with physical calendar")
            units, date, date0 = _axis_origin(a, "timesteps", provider)
            values = _values(variable, name)
            dt = float(a.get("tstep_sec", math.nan))
            if len(values) < 2 or not np.all(np.diff(values) == 1):
                raise SourceClockError("forcing time_steps axis must contain consecutive indices")
        else:
            name, variable = _axes(ds)[0]
            units, date, date0 = _axis_origin(attrs, "seconds", provider)
            values = _values(variable, name)
            if len(values) < 2:
                raise SourceClockError("forcing clock requires at least two time values")
            delta = np.diff(values.astype(np.float64))
            if not np.all(delta == delta[0]):
                raise SourceClockError("forcing time axis is not regular")
            dt = float(delta[0])
        if not math.isfinite(dt) or dt <= 0:
            raise SourceClockError("forcing timestep/tstep_sec must be finite and positive")
        return ForcingClockMetadata(provider.calendar, date0, dt, len(values), name, units, date, float(values[0]))


@dataclass(frozen=True)
class RestartClockMetadata:
    calendar: str
    date0: float
    itau: int
    dt: float
    physical_origin: float
    physical_time: float
    physical_dtype: str
    runtime: SourceClockRuntime | None


def _runtime_from_attrs(attrs: Mapping) -> SourceClockRuntime | None:
    names = ("version", "binding_json", "binding_sha256", "last_applied_istp", "anchor_local_tstep", "anchor_source_istp")
    present = [PREFIX + name in attrs for name in names]
    if not any(present):
        return None
    if not all(present) or _integer(attrs[PREFIX + "version"], "clock version") != VERSION:
        raise SourceClockError("incomplete/unsupported versioned source clock metadata")
    binding = SourceClockBinding.from_json(str(attrs[PREFIX + "binding_json"]))
    if attrs[PREFIX + "binding_sha256"] != binding.binding_key:
        raise SourceClockError("source clock binding checksum conflicts")
    runtime = SourceClockRuntime(binding, *(np.int64(_integer(attrs[PREFIX + name], name)) for name in names[3:]))
    validate_source_clock_runtime(runtime)
    return runtime


def read_restart_clock_metadata(source, *, legacy_policy: str = "reject") -> RestartClockMetadata:
    """restsett707-781: time_steps origin/tstep_sec dominate physical time.

    The native fixture's different physical/timestep origins are legitimate.
    Physical-only index construction784-787 is unimplemented source text and
    remains unsupported. Legacy admission restores declared axes for a NEW
    run only; it never manufactures a same-run cursor or historical identity.
    """
    if legacy_policy not in {"reject", LEGACY_NATIVE_METADATA}:
        raise SourceClockError("unknown legacy source clock policy")
    with _dataset(source) as ds:
        (_, physical), step_axis = _axes(ds)
        if step_axis is None:
            raise SourceClockError("restart time_steps required; physical-only conversion is not implemented in restsett784-787")
        attrs = _attrs(physical)
        if "calendar" not in attrs:
            raise SourceClockError("restart time variable is missing calendar")
        provider = IOIPSLCalendar(str(attrs["calendar"]))
        _, _, physical_origin = _axis_origin(attrs, "seconds", provider)
        step_name, steps = step_axis
        a = _attrs(steps)
        if "calendar" in a and a["calendar"] != provider.calendar:
            raise SourceClockError("time_steps calendar conflicts with physical calendar")
        _, _, date0 = _axis_origin(a, "timesteps", provider)
        values, times = _values(steps, step_name), _values(physical, "time")
        if len(values) != 1 or len(times) != 1:
            raise SourceClockError("restart clock requires exactly one selected record")
        itau = _integer(values[0], "restart time_steps")
        dt = float(a.get("tstep_sec", math.nan))
        if not math.isfinite(dt) or dt <= 0:
            raise SourceClockError("restart time_steps tstep_sec is missing/invalid")
        runtime = _runtime_from_attrs(_attrs(ds))
        if runtime is None and legacy_policy == "reject":
            raise SourceClockError("legacy restart lacks versioned source clock runtime; declare native_metadata_v1 for a new native run")
        result = RestartClockMetadata(provider.calendar, date0, itau, dt, physical_origin,
                                      float(times[0]), str(times.dtype), runtime)
        if runtime is not None:
            b = runtime.binding
            if (provider.calendar, date0, dt, itau) != (b.calendar, b.date0, b.dt, int(runtime.last_applied_istp)):
                raise SourceClockError("restart selected axis conflicts with versioned source clock runtime")
            expected = np.asarray(int(runtime.last_applied_istp) * b.dt, dtype=times.dtype).item()
            if physical_origin != b.date0 or float(times[0]) != expected:
                raise SourceClockError("restart physical axis conflicts with versioned source clock chronology")
        return result


@dataclass(frozen=True)
class RestartClockBundle:
    driver: RestartClockMetadata
    sechiba: RestartClockMetadata
    stomate: RestartClockMetadata

    @property
    def runtime(self):
        return self.driver.runtime


def read_source_clock_restart_bundle(paths, *, legacy_policy: str = "reject") -> RestartClockBundle:
    if isinstance(paths, Mapping):
        if set(paths) != {"driver", "sechiba", "stomate"}:
            raise SourceClockError("source clock requires driver/sechiba/stomate paths")
        paths = tuple(paths[name] for name in ("driver", "sechiba", "stomate"))
    if len(paths) != 3:
        raise SourceClockError("source clock requires exactly three restart files")
    values = [read_restart_clock_metadata(path, legacy_policy=legacy_policy) for path in paths]
    first = values[0]
    for value in values[1:]:
        if (value.calendar, value.date0, value.itau, value.dt, value.runtime) != (first.calendar, first.date0, first.itau, first.dt, first.runtime):
            raise SourceClockError("three-file restart source clock metadata conflicts")
    return RestartClockBundle(*values)


@dataclass(frozen=True)
class SourceClockSetup:
    binding: SourceClockBinding
    runtime: SourceClockRuntime
    time_control: Any
    forcing: ForcingClockMetadata
    restart: RestartClockBundle | None


def source_clock_from_inputs(forcing, *, dt_sechiba: float, run_id: str,
                             restart_paths=None, run_mode: str = "new_native_run",
                             previous_runtime: SourceClockRuntime | None = None,
                             driver_reset_time: bool = False, overwrite_time: bool = False,
                             time_skip=0, time_length=None, local_tstep_origin: int = 0,
                             legacy_policy: str = "reject") -> SourceClockSetup:
    """Actual declared metadata -> restini precedence -> DIM2 -> run lifecycle."""
    from jax_orchidee.driver.dim2 import Dim2RestartTime, resolve_dim2_time_control
    info = read_forcing_clock_metadata(forcing)
    bundle = None if restart_paths is None else read_source_clock_restart_bundle(restart_paths, legacy_policy=legacy_policy)
    if bundle is not None and bundle.driver.calendar != info.calendar:
        raise SourceClockError("forcing and restart calendars conflict")
    if run_mode not in {"new_native_run", "continuous"}:
        raise SourceClockError("declare run_mode='new_native_run' or 'continuous'")
    if run_mode == "continuous":
        if previous_runtime is None or driver_reset_time or overwrite_time:
            raise SourceClockError("continuous run requires previous runtime and cannot reset/overwrite its locked chronology")
        validate_source_clock_runtime(previous_runtime)
        if previous_runtime.binding.run_id != run_id or previous_runtime.binding.calendar != info.calendar or previous_runtime.binding.dt != dt_sechiba:
            raise SourceClockError("continuous native-run binding changed")
        if bundle is not None and bundle.runtime != previous_runtime:
            raise SourceClockError("continuous restart requires the exact versioned runtime, not legacy axes")
        rt = previous_runtime
        restart_time = Dim2RestartTime(int(rt.last_applied_istp), rt.binding.date0, rt.binding.dt)
    elif bundle is None or driver_reset_time or overwrite_time:
        # IOIPSL overwrite uses the explicitly initialized caller tuple0/date0/dt.
        restart_time = Dim2RestartTime(0, info.date0, dt_sechiba, driver_reset_time)
    else:
        r = bundle.driver
        restart_time = Dim2RestartTime(r.itau, r.date0, r.dt, driver_reset_time)
    control = resolve_dim2_time_control(dt_force=info.dt_force, dt_sechiba=dt_sechiba,
        forcing_length=info.count, date0=info.date0, restart=restart_time,
        time_skip=time_skip, time_length=time_length, calendar=info.calendar)
    if run_mode == "continuous":
        binding = previous_runtime.binding
        if control.istp != int(previous_runtime.last_applied_istp) + 1:
            raise SourceClockError("continuous DIM2 chronology skips/reverses the previous source cursor")
    else:
        old_runtime = bundle.runtime if bundle is not None else previous_runtime
        if old_runtime is not None and old_runtime.binding.run_id == run_id:
            raise SourceClockError("new_native_run requires a new run_id, not silent lifecycle reinitialization")
        lock = IOIPSLCalendar.from_scalar_startdate(info.calendar, control.date0_rest)
        binding = SourceClockBinding(run_id, info.calendar, control.date0_rest, float(dt_sechiba),
                                     lock.startdate_day, lock.startdate_second, control.istp_old)
    runtime = SourceClockRuntime(binding, np.int64(control.istp - 1),
                                 np.int64(_integer(local_tstep_origin, "local_tstep_origin")), np.int64(control.istp))
    validate_source_clock_runtime(runtime)
    return SourceClockSetup(binding, runtime, control, info, bundle)


def source_clock_variable_attributes(binding: SourceClockBinding) -> dict[str, dict]:
    y, m, d, second = binding.provider.ju2ymds(binding.date0)
    # restopenout formats whole seconds; recover the declared whole-second
    # origin only when re-encoding is exactly the same binary64 scalar.
    second = float(fortran_nint(second))
    if binding.provider.ymds2ju(y, m, d, second) != binding.date0:
        raise SourceClockError("fractional metadata origin is outside source fixed-format serialization")
    hour, residual = divmod(second, 3600.0)
    minute, sec = divmod(residual, 60.0)
    # Metadata origins are source fixed-format whole seconds; no silent loss.
    if sec != int(sec):
        raise SourceClockError("fractional metadata origin needs an explicitly supported serialization format")
    origin = f"{y:04d}-{m:02d}-{d:02d} {int(hour):02d}:{int(minute):02d}:{int(sec):02d}"
    return {"time": {"units": "seconds since " + origin, "calendar": binding.calendar, "time_origin": origin},
            "time_steps": {"units": "timesteps since " + origin, "tstep_sec": binding.dt, "time_origin": origin}}


def source_clock_global_attributes(runtime: SourceClockRuntime) -> dict:
    validate_source_clock_runtime(runtime)
    return {PREFIX + "version": np.int32(VERSION), PREFIX + "binding_json": runtime.binding.to_json(),
            PREFIX + "binding_sha256": runtime.binding_key,
            **{PREFIX + name: np.int64(_integer(getattr(runtime, name), name))
               for name in ("last_applied_istp", "anchor_local_tstep", "anchor_source_istp")}}


def physical_state_with_source_clock(physical_state, runtime: SourceClockRuntime):
    """Overwrite physical chronology and attributes, never science coordinates."""
    validate_source_clock_runtime(runtime)
    attrs = dict(physical_state.global_attributes or {})
    attrs.update(source_clock_global_attributes(runtime))
    variables = {name: dict(value) for name, value in (physical_state.variable_attributes or {}).items()}
    for name, value in source_clock_variable_attributes(runtime.binding).items():
        variables.setdefault(name, {}).update(value)
    step = int(runtime.last_applied_istp)
    if not np.iinfo(np.int32).min <= step <= np.iinfo(np.int32).max:
        raise SourceClockError("source time_steps exceeds the audited int32 restart schema")
    return replace(physical_state, time=np.asarray([step * runtime.binding.dt], dtype=np.float64),
                   time_steps=np.asarray([step], dtype=np.int32), global_attributes=attrs, variable_attributes=variables)


def source_clock_setup_from_config(config_path, *, run_def_path=None, run_def_values=None,
                                   first_step_bundle, reference_run_dir=None,
                                   run_mode="new_native_run", previous_runtime=None,
                                   run_id=None) -> SourceClockSetup:
    """Use the same actual forcing file as the prepared bundle and declared IO.

    The bundle's year selects a filename only. Calendar dates come exclusively
    from that file's variable attributes. Missing restart declarations are not
    interpreted as NONE. No original001 producer identity is inferred.
    """
    from pathlib import Path
    from jax_orchidee.driver.domain import load_case_config, forcing_file_for_year
    from jax_orchidee.driver.init import parse_run_def, parse_run_def_bool
    from jax_orchidee.driver.restart import reference_restart_path
    values = dict(run_def_values if run_def_values is not None else parse_run_def(run_def_path))
    def declared(name):
        matches = [str(v).strip() for k, v in values.items() if k.upper() == name.upper()]
        if not matches or len(set(matches)) != 1:
            raise SourceClockError(f"missing/conflicting declared run.def {name}")
        return matches[0]
    config = load_case_config(config_path)
    forcing = forcing_file_for_year(config, first_step_bundle.year)
    names = [declared(key) for key in ("RESTART_FILEIN", "SECHIBA_restart_in", "STOMATE_RESTART_FILEIN")]
    absent = [name.upper() == "NONE" for name in names]
    if any(absent) and not all(absent):
        raise SourceClockError("partial three-file restart chronology is unsupported; declare all components or all NONE")
    paths = None
    if not all(absent):
        paths = []
        for name in names:
            path = Path(name)
            if not path.is_absolute():
                path = Path(reference_run_dir) / path if reference_run_dir is not None else reference_restart_path(config_path, name)
            if not path.is_file():
                raise SourceClockError(f"declared restart file is unavailable: {path}")
            paths.append(path)
    if run_id is None:
        if run_mode == "continuous" and previous_runtime is not None:
            run_id = previous_runtime.binding.run_id
        else:
            identity = json.dumps([str(Path(config_path).resolve()), str(run_def_path), str(forcing),
                                   None if paths is None else [str(p) for p in paths], sorted(values.items())])
            run_id = "declared-input-" + hashlib.sha256(identity.encode()).hexdigest()
    # DRIVER_reset_time defaults FALSE in source; missing is a source default,
    # unlike absent file identity. TIME_SKIP defaults0 and TIME_LENGTH uses
    # the source forcing span when absent (DIM2 378-401,498-556).
    reset = next((parse_run_def_bool(v) for k, v in values.items() if k.upper() == "DRIVER_RESET_TIME"), False)
    skip = next((v for k, v in values.items() if k.upper() == "TIME_SKIP"), 0)
    length = next((v for k, v in values.items() if k.upper() == "TIME_LENGTH"), None)
    dt = float(declared("DT_SECHIBA"))
    return source_clock_from_inputs(forcing, dt_sechiba=dt,
        run_id=run_id, restart_paths=paths, run_mode=run_mode, previous_runtime=previous_runtime,
        driver_reset_time=reset, overwrite_time=reset, time_skip=skip, time_length=length,
        legacy_policy=LEGACY_NATIVE_METADATA)
