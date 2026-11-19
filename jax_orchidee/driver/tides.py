"""Host producer for HYDROL's sequential tide text streams.

Fortran provenance: dim2_driver.f90::driver147-151 opens units103/104;
hydrol.f90::hydrol_soil5541-5542/5645-5655 reads one F10.3 record per
main call, and6276-6288 uses the values as mm and mm per model step.
The supplied differential is authoritative, including its first record.
This module does not infer a derivative, interpolate, round to three decimal
places, clip negative levels, cycle a stream, or apply a timestep multiplier.

The paper Job_001.0-071.0.sh81-82/308-379 invokes a new executable each year,
reopening the SAME files at record zero. An ordinary year boundary within one
native run does not reopen them. Calendar labels never select tide records.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import NamedTuple

import numpy as np


class TideForcingError(ValueError):
    """A declared tide stream or its native-run position is invalid."""


class TideForcingValues(NamedTuple):
    """Two host-prefetched numeric leaves suitable for a compiled consumer."""

    wtp_tide: object
    dwtp_tide: object


@dataclass(frozen=True, eq=False)
class PreparedTideForcing:
    """Immutable prepared records; text parsing is outside the model/JIT loop."""

    positive: np.ndarray
    differential: np.ndarray
    positive_path: str
    differential_path: str

    def __post_init__(self):
        for name in ("positive", "differential"):
            source = np.asarray(getattr(self, name), dtype=np.float64)
            if source.ndim != 1 or source.size == 0 or not np.all(np.isfinite(source)):
                raise TideForcingError("tide records must be nonempty finite one-dimensional arrays")
            # Immutable bytes backing prevents callers from reenabling writeability.
            value = np.frombuffer(source.tobytes(), dtype=np.float64)
            object.__setattr__(self, name, value)


@dataclass(frozen=True, eq=False)
class TideForcingRun:
    """Record zero is tied to an explicit first main source istp, not a year."""

    prepared: PreparedTideForcing
    source_clock_binding_key: str
    first_source_istp: int


_REAL_FIELD = re.compile(r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+))(?:(?:[eEdD])?([+-]\d+)|[eEdD](\d+))?$")


def parse_fortran_f10_3_record(record: str) -> np.float64:
    """Fortran F10.3 input: width10, blank-null, implied decimal if absent.

    Source READ103/1045648/5654; the packaged bounded gfortran reference
    checks decimal, integer, signed exponent, short, blank and width edges.
    An explicit decimal point overrides ``.3``: this is input, not F10.3
    output formatting, so 1.234567 is retained rather than rounded.
    """
    field = record.rstrip("\r\n")[:10].replace(" ", "").split(",", 1)[0]
    if not field:
        return np.float64(0.0)
    match = _REAL_FIELD.fullmatch(field)
    if match is None:
        raise TideForcingError(f"unsupported/invalid F10.3 numeric field: {field!r}")
    coefficient, signed_exponent, exponent = match.groups()
    power = int(signed_exponent or exponent or 0) - (0 if "." in coefficient else 3)
    try:
        value = np.float64(coefficient + "e" + str(power))
    except (ValueError, OverflowError) as exc:
        raise TideForcingError(f"invalid F10.3 numeric field: {field!r}") from exc
    if not np.isfinite(value):
        raise TideForcingError("nonfinite tide forcing is unsupported")
    return value


@lru_cache(maxsize=16)
def _read_prepared(positive_path: str, positive_stat: tuple, differential_path: str,
                   differential_stat: tuple) -> PreparedTideForcing:
    del positive_stat, differential_stat  # cache invalidation, not a scientific identity
    def read(path):
        with Path(path).open("r", encoding="ascii", newline=None) as handle:
            return np.asarray([parse_fortran_f10_3_record(row) for row in handle], dtype=np.float64)
    return PreparedTideForcing(read(positive_path), read(differential_path), positive_path, differential_path)


def prepare_tide_forcing(config_path, *, run_def_values=None, source_run_def_path=None) -> PreparedTideForcing:
    """Load the configured local equivalents of the declared103/104 files once.

    The paper run.def contains legacy server paths. Local YAML explicitly
    maps those two assets; a different declared basename requires a matching
    local mapping, rather than silently continuing with the paper streams.
    If an effective used_run.def omits a filename, callers may explicitly
    supply the originating run.def as declaration authority. No sibling or
    default file is inferred here, and nonempty conflicts never fall back.
    """
    from jax_orchidee.driver.domain import load_case_config
    config = load_case_config(config_path)
    mapping = config["drivers"]["water_table_text"]
    paths = [Path(mapping[key]).resolve() for key in ("positive_file", "differential_file")]
    if run_def_values is not None:
        source_values = None
        def declaration(values, key):
            declarations = [str(value).strip().strip("\"'") for name, value in values.items()
                            if name.lower() == key]
            if len(set(declarations)) > 1:
                raise TideForcingError(f"conflicting declared {key}")
            return declarations[0] if declarations else ""
        for key, path in zip(("wtd_filename", "wtd_filename2"), paths):
            selected = declaration(run_def_values, key)
            if not selected and source_run_def_path is not None:
                if source_values is None:
                    from jax_orchidee.driver.init import parse_run_def
                    try:
                        source_values = parse_run_def(source_run_def_path)
                    except OSError as exc:
                        raise TideForcingError(f"originating tide run.def unavailable: {source_run_def_path}") from exc
                selected = declaration(source_values, key)
            if not selected:
                raise TideForcingError(f"missing declared {key}; no nonempty explicit source authority")
            basename = PurePosixPath(selected.replace("\\", "/")).name
            if basename != path.name:
                raise TideForcingError(f"declared {key} disagrees with the configured local tide asset")
    try:
        stats = [(path.stat().st_size, path.stat().st_mtime_ns) for path in paths]
        return _read_prepared(str(paths[0]), stats[0], str(paths[1]), stats[1])
    except OSError as exc:
        raise TideForcingError(f"declared tide forcing unavailable: {exc}") from exc


def _integer(value, label):
    try:
        a = np.asarray(value)
        if a.shape != () or not np.isfinite(a).item() or int(a) != a.item():
            raise TideForcingError(f"{label} must be an integer scalar")
        return int(a)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TideForcingError(f"{label} must be an integer scalar") from exc


def bind_tide_forcing_run(prepared: PreparedTideForcing, source_clock_setup, *,
                          previous: TideForcingRun | None = None) -> TideForcingRun:
    """New native process resets streams; same-run continuation retains origin.

    DIM2 opens before restini361-401/TIME_SKIP498-529 and does no seek.
    Its initialization1051-1125 performs no hydrol_soil call; first main1293
    consumes record zero even for restart or nonzero TIME_SKIP.
    """
    key = source_clock_setup.binding.binding_key
    if previous is not None and previous.source_clock_binding_key == key:
        if previous.prepared is not prepared:
            if (previous.prepared.positive_path, previous.prepared.differential_path) != (prepared.positive_path, prepared.differential_path) or not (
                np.array_equal(previous.prepared.positive, prepared.positive) and
                np.array_equal(previous.prepared.differential, prepared.differential)
            ):
                raise TideForcingError("tide streams changed within a continuous native run")
        return previous
    return TideForcingRun(prepared, key, _integer(source_clock_setup.time_control.istp, "first source istp"))


def tide_forcing_for_source_istp(run: TideForcingRun, istp) -> TideForcingValues:
    """Prefetch exactly one source READ pair, failing at EOF without wrapping."""
    index = _integer(istp, "source istp") - run.first_source_istp
    count = min(run.prepared.positive.size, run.prepared.differential.size)
    if not 0 <= index < count:
        raise TideForcingError(f"tide stream record {index} outside available paired records [0,{count})")
    return TideForcingValues(run.prepared.positive[index], run.prepared.differential[index])


def tide_forcing_for_local_tstep(run: TideForcingRun, source_clock_runtime, tstep) -> TideForcingValues:
    """Resolve a rebased local label using the accepted explicit source frame."""
    from jax_orchidee.driver.source_clock import source_istp_for_local_tstep, validate_source_clock_runtime
    validate_source_clock_runtime(source_clock_runtime)
    if source_clock_runtime.binding_key != run.source_clock_binding_key:
        raise TideForcingError("tide native-run origin disagrees with source clock binding")
    return tide_forcing_for_source_istp(run, source_istp_for_local_tstep(source_clock_runtime, tstep))
