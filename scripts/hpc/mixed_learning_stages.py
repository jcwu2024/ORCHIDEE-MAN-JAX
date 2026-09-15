"""Strict phase scheduling for opt-in mixed-learning comparisons."""

from __future__ import annotations

import json
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = "mixed_learning_stages_v1"
OBJECTIVES = ("daily", "mixed")


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        result = int(operator.index(value))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    return result


def _json_record(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be JSON-compatible") from error
    return json.loads(encoded)


@dataclass(frozen=True)
class LearningPhase:
    """One contiguous, one-based optimizer-update interval."""

    name: str
    objective: str
    start_update: int
    end_update: int
    outer_weights: dict[str, float]


@dataclass(frozen=True)
class LearningSchedule:
    """Validated schedule plus deterministic progress/cursor arithmetic."""

    record: dict[str, Any]
    name: str
    total_updates: int
    milestones: tuple[int, ...]
    phases: tuple[LearningPhase, ...]

    def phase_for_update(self, update: int) -> LearningPhase:
        update = _integer(update, name="update")
        if not 1 <= update <= self.total_updates:
            raise ValueError("update is outside the learning schedule")
        return next(phase for phase in self.phases if update <= phase.end_update)

    def recurrent_cursor(self, completed_updates: int) -> int:
        completed = _integer(completed_updates, name="completed_updates")
        if not 0 <= completed <= self.total_updates:
            raise ValueError("completed_updates is outside the learning schedule")
        consumed = 0
        for phase in self.phases:
            phase_completed = max(
                0,
                min(completed, phase.end_update) - phase.start_update + 1,
            )
            if phase.objective == "mixed":
                consumed += phase_completed
        return consumed

    def progress(self, completed_updates: int) -> dict[str, Any]:
        completed = _integer(completed_updates, name="completed_updates")
        recurrent = self.recurrent_cursor(completed)
        completed_phase = self.phase_for_update(completed).name if completed else None
        next_phase = (
            self.phase_for_update(completed + 1).name
            if completed < self.total_updates
            else None
        )
        return {
            "completed_updates": completed,
            "completed_phase": completed_phase,
            "next_phase": next_phase,
            "daily_sampler_cursor": {"epoch": 0, "batch_offset": completed},
            "recurrent_schedule_cursor": recurrent,
            "consumed": {
                "daily_batches": completed,
                "recurrent_updates": recurrent,
                "recurrent_windows": {"T7": recurrent, "T30": recurrent},
            },
        }

    def validate_progress(self, progress: Any, *, completed_updates: int) -> None:
        if _json_record(progress, name="learning progress") != self.progress(
            completed_updates
        ):
            raise ValueError("learning schedule phase/cursor progress mismatch")

    def data_consumption(self, completed_updates: int, batches: Any) -> dict[str, Any]:
        completed = _integer(completed_updates, name="completed_updates")
        progress = self.progress(completed)
        try:
            consumed = batches[:completed]
        except (TypeError, KeyError) as error:
            raise TypeError("training batches must support ordered slicing") from error
        if len(consumed) != completed:
            raise ValueError("training batches do not cover completed updates")
        result = dict(progress["consumed"])
        result.update(
            daily_draws=sum(
                _integer(batch.draw_count, name="batch draw_count")
                for batch in consumed
            ),
            daily_source_draws=sum(
                _integer(batch.source_draw_count, name="batch source_draw_count")
                for batch in consumed
            ),
            daily_padding_draws=sum(
                _integer(batch.padding_draw_count, name="batch padding_draw_count")
                for batch in consumed
            ),
        )
        if result["daily_source_draws"] + result["daily_padding_draws"] != result[
            "daily_draws"
        ]:
            raise ValueError("daily source and padding draws do not sum to actual draws")
        return result

    def validate_data_consumption(
        self, value: Any, *, completed_updates: int, batches: Any
    ) -> None:
        if _json_record(value, name="data consumption") != self.data_consumption(
            completed_updates, batches
        ):
            raise ValueError("learning report data consumption mismatch")

    def validate_update_records(
        self, records: Any, *, completed_updates: int
    ) -> None:
        completed = _integer(completed_updates, name="completed_updates")
        if not isinstance(records, list) or len(records) != completed:
            raise ValueError("learning report update records disagree with progress")
        for update, record in enumerate(records, start=1):
            if not isinstance(record, Mapping):
                raise TypeError("learning report update record must be a mapping")
            phase = self.phase_for_update(update)
            recurrent_before = self.recurrent_cursor(update - 1)
            recurrent_index = recurrent_before if phase.objective == "mixed" else None
            expected = {
                "update": update,
                "phase": phase.name,
                "objective": phase.objective,
                "daily_batch_index": update - 1,
                "recurrent_schedule_index": recurrent_index,
                "recurrent_schedule_cursor_after": self.recurrent_cursor(update),
            }
            if any(record.get(key) != value for key, value in expected.items()):
                raise ValueError("learning report phase/cursor update record mismatch")


def validate_learning_schedule(
    value: Any,
    *,
    outer_weights: Mapping[str, Any],
    available_updates: int,
) -> LearningSchedule:
    """Validate the complete opt-in schedule without changing objective weights."""

    record = _json_record(value, name="learning schedule")
    expected_keys = {
        "schema_version",
        "name",
        "total_updates",
        "milestones",
        "phases",
    }
    if set(record) != expected_keys or record["schema_version"] != SCHEMA_VERSION:
        raise ValueError("learning schedule schema or keys drift")
    if not isinstance(record["name"], str) or not record["name"].strip():
        raise ValueError("learning schedule name must be nonempty")

    total = _integer(record["total_updates"], name="total_updates")
    available = _integer(available_updates, name="available_updates")
    if total <= 0 or total > available:
        raise ValueError("learning schedule exceeds available daily batches")

    milestones_value = record["milestones"]
    if not isinstance(milestones_value, list):
        raise TypeError("learning schedule milestones must be a list")
    milestones = tuple(
        _integer(value, name="milestone") for value in milestones_value
    )
    if (
        not milestones
        or milestones[0] != 0
        or milestones[-1] != total
        or tuple(sorted(set(milestones))) != milestones
    ):
        raise ValueError("learning schedule milestones must be unique from zero to total")

    declared_weights = _json_record(outer_weights, name="declared outer weights")
    if set(declared_weights) != {"daily", "T7", "T30"}:
        raise ValueError("declared mixed outer weights are incomplete")
    daily_weights = {"daily": declared_weights["daily"]}

    phase_values = record["phases"]
    if not isinstance(phase_values, list) or not phase_values:
        raise ValueError("learning schedule phases must be a nonempty list")
    phases = []
    names = set()
    start = 1
    for phase_value in phase_values:
        phase = _json_record(phase_value, name="learning phase")
        if set(phase) != {"name", "objective", "end_update", "outer_weights"}:
            raise ValueError("learning phase keys drift")
        name = phase["name"]
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("learning phase names must be unique and nonempty")
        names.add(name)
        objective = phase["objective"]
        if objective not in OBJECTIVES:
            raise ValueError("learning phase objective must be daily or mixed")
        end = _integer(phase["end_update"], name="phase end_update")
        if end < start or end > total:
            raise ValueError("learning phase endpoints must be increasing and bounded")
        weights = _json_record(phase["outer_weights"], name="phase outer weights")
        expected_weights = daily_weights if objective == "daily" else declared_weights
        if weights != expected_weights:
            raise ValueError(f"{objective} phase changes the declared outer weights")
        phases.append(
            LearningPhase(
                name=name,
                objective=objective,
                start_update=start,
                end_update=end,
                outer_weights=weights,
            )
        )
        start = end + 1

    endpoints = {phase.end_update for phase in phases}
    if phases[-1].end_update != total:
        raise ValueError("learning phases do not reach total_updates")
    if not endpoints.issubset(milestones):
        raise ValueError("every learning phase endpoint must be a milestone")

    return LearningSchedule(
        record=record,
        name=record["name"],
        total_updates=total,
        milestones=milestones,
        phases=tuple(phases),
    )


def load_learning_schedule(
    path: str | Path,
    *,
    outer_weights: Mapping[str, Any],
    available_updates: int,
) -> LearningSchedule:
    return validate_learning_schedule(
        json.loads(Path(path).read_text(encoding="utf-8")),
        outer_weights=outer_weights,
        available_updates=available_updates,
    )


__all__ = [
    "LearningPhase",
    "LearningSchedule",
    "OBJECTIVES",
    "SCHEMA_VERSION",
    "load_learning_schedule",
    "validate_learning_schedule",
]
