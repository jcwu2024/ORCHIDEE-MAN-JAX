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


def _progress(
    *,
    completed_updates: Any,
    daily_sampler_cursor: Any,
    recurrent_schedule_cursor: Any,
    optimizer_state: Any,
) -> tuple[int, dict[str, int], int]:
    completed = _nonnegative_integer(completed_updates, name="completed_updates")
    if completed > MAXIMUM_UPDATES:
        raise ValueError("completed_updates exceeds the 200-update protocol")

    daily = _cursor(daily_sampler_cursor)
    recurrent = _nonnegative_integer(
        recurrent_schedule_cursor,
        name="recurrent_schedule_cursor",
    )
    if daily != {"epoch": 0, "batch_offset": completed}:
        raise ValueError(
            "daily sampler cursor disagrees with fixed mixed-training progress"
        )
    if recurrent != completed:
        raise ValueError(
            "recurrent schedule cursor disagrees with mixed-training progress"
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
    replace: bool = False,
) -> Mapping[str, Any]:
    """Atomically save parameters, AdamW moments, identities, and paired cursors.

    A unique path with ``replace=False`` is appropriate for retained milestone
    checkpoints. A single rolling recovery path may opt into ``replace=True``.
    """

    if not isinstance(replace, bool):
        raise TypeError("replace must be a bool")
    completed, daily, recurrent = _progress(
        completed_updates=completed_updates,
        daily_sampler_cursor=daily_sampler_cursor,
        recurrent_schedule_cursor=recurrent_schedule_cursor,
        optimizer_state=optimizer_state,
    )
    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "parameters": parameters,
        "optimizer_state": optimizer_state,
        "completed_updates": completed,
        "daily_sampler_cursor": daily,
        "recurrent_schedule_cursor": recurrent,
        "method": _identity(method, name="method"),
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
