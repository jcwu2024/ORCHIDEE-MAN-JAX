"""Metadata-only selection for a training-day-disjoint 7/30-day screen.

``start_row`` is always the actual row in the Teacher shard. A window starting
at row ``r`` therefore owns the semantic Teacher-day keys ``r`` through
``r + length - 1``; no calendar or prepared-day offset is applied here.
"""

from __future__ import annotations

import operator
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np

DayKey: TypeAlias = tuple[str, int, int]
ArtifactId: TypeAlias = int | str

_COMPONENT_LENGTHS = (("seven_day", 7), ("thirty_day", 30))
_MISSING = object()
_NO_DEFAULT = object()


def _value(container: Any, name: str, *, default: Any = _NO_DEFAULT) -> Any:
    if isinstance(container, Mapping):
        if name in container:
            return container[name]
    elif hasattr(container, name):
        return getattr(container, name)
    if default is not _NO_DEFAULT:
        return default
    raise ValueError(f"metadata is missing {name!r}")


def _items(value: Any, *, name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        raise TypeError(f"{name} must be a sequence")
    try:
        return tuple(value)
    except TypeError as error:
        raise TypeError(f"{name} must be a sequence") from error


def _integer(value: Any, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        result = int(operator.index(value))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _landpoint_id(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError("landpoint_id must be a non-empty string")
    return value


def _artifact_id(entry: Any) -> ArtifactId:
    declared = _value(entry, "artifact_id", default=_MISSING)
    update = _value(entry, "update", default=_MISSING)
    if declared is _MISSING and update is _MISSING:
        raise ValueError("schedule entry is missing artifact_id/update")
    if declared is not _MISSING and update is not _MISSING and declared != update:
        raise ValueError("schedule entry artifact_id and update disagree")
    value = update if declared is _MISSING else declared
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise TypeError("artifact_id must be an integer or string")
    if isinstance(value, str) and not value:
        raise ValueError("artifact_id must not be empty")
    return value


def _artifact_order_key(value: ArtifactId) -> tuple[int, int | str]:
    if isinstance(value, int):
        return (0, value)
    return (1, value)


def _day_key(value: Any) -> DayKey:
    if isinstance(value, Mapping) or hasattr(value, "landpoint_id"):
        landpoint_id = _landpoint_id(_value(value, "landpoint_id"))
        year = _integer(_value(value, "year"), name="year")
        start_row = _integer(_value(value, "start_row"), name="start_row", minimum=0)
        return (landpoint_id, year, start_row)
    raw = _items(value, name="excluded day key")
    if len(raw) != 3:
        raise ValueError("excluded day keys must contain landpoint_id, year, and start_row")
    return (
        _landpoint_id(raw[0]),
        _integer(raw[1], name="year"),
        _integer(raw[2], name="start_row", minimum=0),
    )


def _record_day_keys(record: Any) -> set[DayKey]:
    landpoint_id = _landpoint_id(_value(record, "landpoint_id"))
    year = _integer(_value(record, "year"), name="year")
    starts = _items(_value(record, "start_rows"), name="start_rows")
    return {
        (landpoint_id, year, _integer(start, name="start_row", minimum=0))
        for start in starts
    }


def training_day_keys(catalogue: Any) -> set[DayKey]:
    """Return every one-day and fixed-monitor Teacher row in ``catalogue``."""

    result: set[DayKey] = set()
    entries = _items(_value(catalogue, "one_day_entries"), name="one_day_entries")
    for entry in entries:
        components = _value(entry, "components")
        records = _items(_value(components, "one_day"), name="components.one_day")
        for record in records:
            result.update(_record_day_keys(record))

    monitor_rows = _items(_value(catalogue, "monitor_rows", default=()), name="monitor_rows")
    result.update(_day_key(row) for row in monitor_rows)
    return result


@dataclass(frozen=True, slots=True)
class _Candidate:
    artifact_id: ArtifactId
    component: str
    window_index: int
    landpoint_id: str
    year: int
    start_row: int
    length: int

    @property
    def point_year(self) -> tuple[str, int]:
        return (self.landpoint_id, self.year)

    @property
    def teacher_day_keys(self) -> tuple[DayKey, ...]:
        return tuple(
            (self.landpoint_id, self.year, row)
            for row in range(self.start_row, self.start_row + self.length)
        )

    def as_metadata(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "component": self.component,
            "window_index": self.window_index,
            "landpoint_id": self.landpoint_id,
            "year": self.year,
            "start_row": self.start_row,
            "length": self.length,
            "teacher_day_keys": self.teacher_day_keys,
        }


def _component_candidates(
    entry: Any,
    artifact_id: ArtifactId,
    component: str,
    length: int,
) -> tuple[_Candidate, ...]:
    components = _value(entry, "components")
    records = _items(_value(components, component, default=()), name=f"components.{component}")
    candidates = []
    window_index = 0
    for record in records:
        landpoint_id = _landpoint_id(_value(record, "landpoint_id"))
        year = _integer(_value(record, "year"), name="year")
        starts = _items(_value(record, "start_rows"), name=f"{component}.start_rows")
        for raw_start in starts:
            candidates.append(
                _Candidate(
                    artifact_id=artifact_id,
                    component=component,
                    window_index=window_index,
                    landpoint_id=landpoint_id,
                    year=year,
                    start_row=_integer(raw_start, name="start_row", minimum=0),
                    length=length,
                )
            )
            window_index += 1
    return tuple(candidates)


def _shuffled_candidate_pool(
    schedule_entries: Sequence[Any],
    generator: np.random.Generator,
) -> dict[str, tuple[tuple[ArtifactId, tuple[_Candidate, ...]], ...]]:
    identified = tuple((_artifact_id(entry), entry) for entry in schedule_entries)
    artifact_ids = tuple(artifact_id for artifact_id, _entry in identified)
    if len(set(artifact_ids)) != len(artifact_ids):
        raise ValueError("schedule artifact IDs must be unique")
    canonical = tuple(sorted(identified, key=lambda item: _artifact_order_key(item[0])))
    artifact_permutation = generator.permutation(len(canonical))
    ordered = tuple(canonical[int(index)] for index in artifact_permutation)

    mutable: dict[str, list[tuple[ArtifactId, tuple[_Candidate, ...]]]] = {
        component: [] for component, _length in _COMPONENT_LENGTHS
    }
    for artifact_id, entry in ordered:
        for component, length in _COMPONENT_LENGTHS:
            canonical_candidates = _component_candidates(entry, artifact_id, component, length)
            candidate_permutation = generator.permutation(len(canonical_candidates))
            shuffled = tuple(canonical_candidates[int(index)] for index in candidate_permutation)
            mutable[component].append((artifact_id, shuffled))
    return {component: tuple(values) for component, values in mutable.items()}


def _fill_component(
    pool: tuple[tuple[ArtifactId, tuple[_Candidate, ...]], ...],
    count: int,
    excluded_days: set[DayKey],
    selected_days: set[DayKey],
    used_point_years: set[tuple[str, int]],
    *,
    prefer_distinct: bool,
) -> list[_Candidate]:
    selected: list[_Candidate] = []
    used_artifacts: set[ArtifactId] = set()
    passes = (True, False) if prefer_distinct else (False,)
    for require_new_point_year in passes:
        for artifact_id, candidates in pool:
            if len(selected) == count:
                return selected
            if artifact_id in used_artifacts:
                continue
            for candidate in candidates:
                days = set(candidate.teacher_day_keys)
                if days & excluded_days or days & selected_days:
                    continue
                if require_new_point_year and candidate.point_year in used_point_years:
                    continue
                selected.append(candidate)
                used_artifacts.add(artifact_id)
                selected_days.update(days)
                used_point_years.add(candidate.point_year)
                break
    return selected


def _attempt_selection(
    pools: Mapping[str, tuple[tuple[ArtifactId, tuple[_Candidate, ...]], ...]],
    excluded_days: set[DayKey],
    per_horizon: int,
    component_order: tuple[str, str],
    *,
    prefer_distinct: bool,
) -> dict[str, list[_Candidate]]:
    selected_days: set[DayKey] = set()
    used_point_years: set[tuple[str, int]] = set()
    result = {component: [] for component, _length in _COMPONENT_LENGTHS}
    for component in component_order:
        result[component] = _fill_component(
            pools[component],
            per_horizon,
            excluded_days,
            selected_days,
            used_point_years,
            prefer_distinct=prefer_distinct,
        )
    return result


def _selection_score(selection: Mapping[str, Sequence[_Candidate]]) -> tuple[int, int]:
    values = tuple(
        candidate
        for component, _length in _COMPONENT_LENGTHS
        for candidate in selection[component]
    )
    return (len(values), len({candidate.point_year for candidate in values}))


def select_windows(
    schedule_entries: Iterable[Any] | Mapping[str, Any],
    excluded_days: Iterable[Any],
    seed: int = 20260913,
    per_horizon: int = 8,
) -> list[dict[str, Any]]:
    """Select shuffled, metadata-only, mutually day-disjoint 7/30-day windows.

    The result is flat and ordered with all ``seven_day`` records before all
    ``thirty_day`` records. ``window_index`` is the candidate's position in
    the artifact component before filtering or shuffling.
    """

    seed = _integer(seed, name="seed", minimum=0)
    per_horizon = _integer(per_horizon, name="per_horizon", minimum=1)
    if isinstance(schedule_entries, Mapping):
        entries = _items(_value(schedule_entries, "entries"), name="schedule entries")
    else:
        entries = _items(schedule_entries, name="schedule entries")
    if not entries:
        raise ValueError("metadata-only window shortage: the schedule has no entries")
    exclusions = {_day_key(value) for value in excluded_days}

    generator = np.random.Generator(np.random.PCG64(seed))
    pools = _shuffled_candidate_pool(entries, generator)
    attempts = [
        _attempt_selection(
            pools,
            exclusions,
            per_horizon,
            component_order,
            prefer_distinct=prefer_distinct,
        )
        for prefer_distinct in (True, False)
        for component_order in (("thirty_day", "seven_day"), ("seven_day", "thirty_day"))
    ]
    complete = [
        attempt
        for attempt in attempts
        if all(
            len(attempt[component]) == per_horizon
            for component, _length in _COMPONENT_LENGTHS
        )
    ]
    if not complete:
        best = max(attempts, key=_selection_score)
        counts = ", ".join(
            f"{component} required={per_horizon} selected={len(best[component])}"
            for component, _length in _COMPONENT_LENGTHS
        )
        raise ValueError(f"metadata-only window shortage after exclusions and disjointness: {counts}")

    selected = max(complete, key=lambda value: _selection_score(value)[1])
    ordered = [
        candidate
        for component, _length in _COMPONENT_LENGTHS
        for candidate in selected[component]
    ]
    return [candidate.as_metadata() for candidate in ordered]


__all__ = ["DayKey", "select_windows", "training_day_keys"]
