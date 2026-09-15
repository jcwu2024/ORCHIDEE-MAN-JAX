"""Atomic full-state recovery for the fixed 200-update mixed training run."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Mapping

from .one_day_training_state import (
    _atomic_pickle,
    _cursor,
    _method,
    _nonnegative_integer,
)

CHECKPOINT_SCHEMA = "fresh_mixed_learning_checkpoint_v1"
MAXIMUM_UPDATES = 200
MILESTONE_UPDATES = (0, 50, 100, 200)
_CHECKPOINT_KEYS = {
    "schema_version",
    "parameters",
    "optimizer_state",
    "completed_updates",
    "daily_sampler_cursor",
    "recurrent_schedule_cursor",
    "method",
    "source_identities",
    "schedule_identity",
    "fresh_initialization_identity",
}


def _identity(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an identity mapping")
    return _method(dict(value), name=name)


def _optimizer_step(optimizer_state: Any) -> int:
    if not hasattr(optimizer_state, "step"):
        raise TypeError("optimizer_state must expose its AdamW step")
    return _nonnegative_integer(optimizer_state.step, name="optimizer step")


def _declared_maximum_updates(
    method: Mapping[str, Any], maximum_updates: Any | None
) -> int:
    schedule = method.get("learning_schedule")
    configuration = method.get("configuration")
    if schedule is not None:
        if not isinstance(schedule, Mapping):
            raise TypeError("method learning_schedule must be a mapping")
        declared = _nonnegative_integer(
            schedule.get("total_updates"), name="learning schedule total_updates"
        )
    elif isinstance(configuration, Mapping) and "updates" in configuration:
        declared = _nonnegative_integer(
            configuration["updates"], name="configuration updates"
        )
    elif "updates" in method:
        declared = _nonnegative_integer(method["updates"], name="method updates")
    else:
        declared = MAXIMUM_UPDATES
    if declared <= 0:
        raise ValueError("declared maximum updates must be positive")
    if maximum_updates is not None:
        requested = _nonnegative_integer(maximum_updates, name="maximum_updates")
        if requested != declared:
            raise ValueError("maximum_updates differs from the checkpoint method")
    return declared


def _declared_recurrent_cursor(method: Mapping[str, Any], completed: int) -> int:
    schedule = method.get("learning_schedule")
    if schedule is None:
        return completed
    if not isinstance(schedule, Mapping):
        raise TypeError("method learning_schedule must be a mapping")
    phases = schedule.get("phases")
    total = _nonnegative_integer(
        schedule.get("total_updates"), name="learning schedule total_updates"
    )
    if not isinstance(phases, list) or not phases:
        raise ValueError("method learning_schedule phases are invalid")
    start, recurrent = 1, 0
    for phase in phases:
        if not isinstance(phase, Mapping):
            raise TypeError("method learning phase must be a mapping")
        end = _nonnegative_integer(phase.get("end_update"), name="phase end_update")
        objective = phase.get("objective")
        if end < start or end > total or objective not in ("daily", "mixed"):
            raise ValueError("method learning phase is invalid")
        phase_completed = max(0, min(completed, end) - start + 1)
        if objective == "mixed":
            recurrent += phase_completed
        start = end + 1
    if start != total + 1:
        raise ValueError("method learning phases do not reach total_updates")
    return recurrent


def _progress(
    *,
    completed_updates: Any,
    daily_sampler_cursor: Any,
    recurrent_schedule_cursor: Any,
    optimizer_state: Any,
    method: Mapping[str, Any],
    maximum_updates: Any | None,
) -> tuple[int, dict[str, int], int]:
    completed = _nonnegative_integer(completed_updates, name="completed_updates")
    maximum = _declared_maximum_updates(method, maximum_updates)
    if completed > maximum:
        raise ValueError("completed_updates exceeds the declared training protocol")

    daily = _cursor(daily_sampler_cursor)
    recurrent = _nonnegative_integer(
        recurrent_schedule_cursor,
        name="recurrent_schedule_cursor",
    )
    if daily != {"epoch": 0, "batch_offset": completed}:
        raise ValueError(
            "daily sampler cursor disagrees with fixed mixed-training progress"
        )
    if recurrent != _declared_recurrent_cursor(method, completed):
        raise ValueError(
            "recurrent schedule cursor disagrees with declared phase progress"
        )
    if _optimizer_step(optimizer_state) != completed:
        raise ValueError("AdamW step disagrees with mixed-training progress")
    return completed, daily, recurrent


def save_mixed_training_checkpoint(
    path: str | Path,
    *,
    jax: Any,
    parameters: Any,
    optimizer_state: Any,
    completed_updates: int,
    daily_sampler_cursor: dict[str, int],
    recurrent_schedule_cursor: int,
    method: Mapping[str, Any],
    source_identities: Mapping[str, Any],
    schedule_identity: Mapping[str, Any],
    fresh_initialization_identity: Mapping[str, Any],
    maximum_updates: int | None = None,
    replace: bool = False,
) -> Mapping[str, Any]:
    """Atomically save parameters, AdamW moments, identities, and paired cursors.

    A unique path with ``replace=False`` is appropriate for retained milestone
    checkpoints. A single rolling recovery path may opt into ``replace=True``.
    """

    if not isinstance(replace, bool):
        raise TypeError("replace must be a bool")
    normalized_method = _identity(method, name="method")
    completed, daily, recurrent = _progress(
        completed_updates=completed_updates,
        daily_sampler_cursor=daily_sampler_cursor,
        recurrent_schedule_cursor=recurrent_schedule_cursor,
        optimizer_state=optimizer_state,
        method=normalized_method,
        maximum_updates=maximum_updates,
    )
    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "parameters": parameters,
        "optimizer_state": optimizer_state,
        "completed_updates": completed,
        "daily_sampler_cursor": daily,
        "recurrent_schedule_cursor": recurrent,
        "method": normalized_method,
        "source_identities": _identity(
            source_identities, name="source_identities"
        ),
        "schedule_identity": _identity(
            schedule_identity, name="schedule_identity"
        ),
        "fresh_initialization_identity": _identity(
            fresh_initialization_identity,
            name="fresh_initialization_identity",
        ),
    }
    host_payload = jax.device_get(payload)
    target = Path(path).resolve()
    if target.exists() and not replace:
        raise FileExistsError(target)
    _atomic_pickle(target, host_payload)
    return host_payload


def load_mixed_training_checkpoint(
    path: str | Path,
    *,
    expected_method: Mapping[str, Any],
    expected_source_identities: Mapping[str, Any],
    expected_schedule_identity: Mapping[str, Any],
    expected_fresh_initialization_identity: Mapping[str, Any],
    maximum_updates: int | None = None,
) -> Mapping[str, Any]:
    """Load a full host-resident state and fail closed on identity/cursor drift."""

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or set(payload) != _CHECKPOINT_KEYS:
        raise ValueError("mixed training checkpoint structure drift")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("mixed training checkpoint schema drift")

    identity_pairs = (
        ("method", expected_method),
        ("source_identities", expected_source_identities),
        ("schedule_identity", expected_schedule_identity),
        ("fresh_initialization_identity", expected_fresh_initialization_identity),
    )
    normalized = {}
    for name, expected in identity_pairs:
        observed = _identity(payload.get(name), name=f"checkpoint {name}")
        required = _identity(expected, name=f"expected_{name}")
        if observed != required:
            raise ValueError(f"mixed training checkpoint {name} mismatch")
        normalized[name] = observed

    completed, daily, recurrent = _progress(
        completed_updates=payload.get("completed_updates"),
        daily_sampler_cursor=payload.get("daily_sampler_cursor"),
        recurrent_schedule_cursor=payload.get("recurrent_schedule_cursor"),
        optimizer_state=payload.get("optimizer_state"),
        method=normalized["method"],
        maximum_updates=maximum_updates,
    )
    return {
        **payload,
        "completed_updates": completed,
        "daily_sampler_cursor": daily,
        "recurrent_schedule_cursor": recurrent,
        **normalized,
    }


__all__ = [
    "CHECKPOINT_SCHEMA",
    "MAXIMUM_UPDATES",
    "MILESTONE_UPDATES",
    "load_mixed_training_checkpoint",
    "save_mixed_training_checkpoint",
]
