"""Atomic parameter-only recovery for persistent scale-selected training."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .successor_fit_checkpoint import _tree_schema, _validate_finite_tree, sha256_file
from .target_oriented_coupled_gpp_objective import GRADIENT_SOURCES
from .target_oriented_coupled_gpp_persistent_scale_training import (
    ALL_REPORTED_KEYS,
    MILESTONES,
)

RECOVERY_SCHEMA = "target_oriented_coupled_gpp_persistent_recovery_v1"
CHECKPOINT_SCHEMA = "target_oriented_coupled_gpp_persistent_checkpoint_v1"
MONITOR_ARTIFACT_COUNT = 8


def _atomic_pickle(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validate_monitor(panel: Mapping[str, Any]) -> Mapping[str, Any]:
    records = tuple(panel.get("records", ()))
    losses = panel.get("source_mean_losses", {})
    acceptance = panel.get("acceptance", {})
    gpp = panel.get("cumulative_gpp_scores", {})
    valid_acceptance = set(acceptance) == {"7", "30"} and all(
        set(acceptance[horizon].get("keys", {})) == set(ALL_REPORTED_KEYS) for horizon in ("7", "30")
    )
    if (
        len(records) != MONITOR_ARTIFACT_COUNT
        or set(losses) != set(GRADIENT_SOURCES)
        or not all(np.isfinite(float(value)) and float(value) >= 0.0 for value in losses.values())
        or not valid_acceptance
        or set(gpp) != {"7", "30"}
        or not all(np.isfinite(float(value)) and float(value) >= 0.0 for value in gpp.values())
        or panel.get("hard_gate_passed") is not True
    ):
        raise ValueError("persistent coupled GPP monitor drift")
    return dict(panel)


def _validate_records(records: Sequence[Mapping[str, Any]], completed_updates: int) -> None:
    expected = list(range(1, int(completed_updates) + 1))
    if [int(record.get("persistent_update", -1)) for record in records] != expected:
        raise ValueError("persistent coupled GPP update coverage drift")
    for record in records:
        scale = record.get("selected_scale")
        if record.get("finite_step_accepted") is not True or scale is None or not np.isfinite(float(scale)):
            raise ValueError("persistent coupled GPP retained update is invalid")


def _validate_milestones(
    monitors: Sequence[Mapping[str, Any]],
    completed_updates: int,
) -> tuple[Mapping[str, Any], ...]:
    values = tuple(monitors)
    expected = tuple(value for value in MILESTONES if value <= int(completed_updates))
    observed = tuple(int(value.get("persistent_update", -1)) for value in values)
    if observed != expected:
        raise ValueError("persistent coupled GPP milestone coverage drift")
    for value in values:
        _validate_monitor(value)
        gate = value.get("gate", {})
        if not isinstance(gate.get("checkpoint_passed"), bool) or not isinstance(gate.get("stop_stage"), bool):
            raise ValueError("persistent coupled GPP milestone gate drift")
    return values


def _payload(
    *,
    schema: str,
    identity: Mapping[str, Any],
    trainable: Any,
    completed_updates: int,
    parameter_count: int,
    update_records: Sequence[Mapping[str, Any]],
    initial_monitor: Mapping[str, Any],
    milestone_monitors: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    completed = int(completed_updates)
    if completed <= 0:
        raise ValueError("persistent coupled GPP persistence requires progress")
    _validate_records(update_records, completed)
    observed = _validate_finite_tree(trainable, label="persistent coupled GPP trainable parameters")
    if observed != int(parameter_count):
        raise ValueError("persistent coupled GPP parameter count drift")
    milestones = _validate_milestones(milestone_monitors, completed)
    return {
        "schema_version": schema,
        "promotion_eligible": False,
        "identity": dict(identity),
        "completed_updates": completed,
        "parameter_count": int(parameter_count),
        "parameter_tree_schema": _tree_schema(trainable),
        "trainable": trainable,
        "update_records": [dict(value) for value in update_records],
        "initial_monitor": _validate_monitor(initial_monitor),
        "milestone_monitors": [dict(value) for value in milestones],
    }


def write_persistent_recovery(path: str | Path, **kwargs: Any) -> Mapping[str, Any]:
    payload = _payload(schema=RECOVERY_SCHEMA, **kwargs)
    target = Path(path).resolve()
    _atomic_pickle(target, payload)
    return {
        "path": str(target),
        "sha256": sha256_file(target),
        "completed_updates": int(payload["completed_updates"]),
        "promotion_eligible": False,
    }


def write_persistent_checkpoint(path: str | Path, **kwargs: Any) -> Mapping[str, Any]:
    payload = _payload(schema=CHECKPOINT_SCHEMA, **kwargs)
    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError(target)
    _atomic_pickle(target, payload)
    return {
        "path": str(target),
        "sha256": sha256_file(target),
        "completed_updates": int(payload["completed_updates"]),
        "parameter_count": int(payload["parameter_count"]),
        "promotion_eligible": False,
    }


def _load(
    path: str | Path,
    *,
    schema: str,
    identity: Mapping[str, Any],
    parameter_count: int,
    maximum_updates: int,
    expected_sha256: str | None = None,
) -> Mapping[str, Any] | None:
    source = Path(path).resolve()
    if not source.exists() and schema == RECOVERY_SCHEMA:
        return None
    if not source.is_file() or (expected_sha256 is not None and sha256_file(source) != expected_sha256):
        raise ValueError("persistent coupled GPP persistence file identity drift")
    with source.open("rb") as handle:
        payload = pickle.load(handle)
    completed = int(payload.get("completed_updates", -1))
    if (
        payload.get("schema_version") != schema
        or payload.get("promotion_eligible") is not False
        or payload.get("identity") != dict(identity)
        or int(payload.get("parameter_count", -1)) != int(parameter_count)
        or not 0 < completed <= int(maximum_updates)
    ):
        raise ValueError("persistent coupled GPP persistence identity drift")
    _validate_records(payload.get("update_records", ()), completed)
    observed = _validate_finite_tree(payload["trainable"], label="persistent coupled GPP trainable parameters")
    if observed != int(parameter_count) or _tree_schema(payload["trainable"]) != payload.get("parameter_tree_schema"):
        raise ValueError("persistent coupled GPP persisted parameter tree drift")
    _validate_monitor(payload.get("initial_monitor", {}))
    _validate_milestones(payload.get("milestone_monitors", ()), completed)
    return payload


def load_persistent_recovery(
    path: str | Path,
    *,
    identity: Mapping[str, Any],
    parameter_count: int,
    maximum_updates: int,
) -> Mapping[str, Any] | None:
    return _load(
        path,
        schema=RECOVERY_SCHEMA,
        identity=identity,
        parameter_count=parameter_count,
        maximum_updates=maximum_updates,
    )


def load_persistent_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str,
    identity: Mapping[str, Any],
    completed_updates: int,
    parameter_count: int,
) -> Mapping[str, Any]:
    payload = _load(
        path,
        schema=CHECKPOINT_SCHEMA,
        identity=identity,
        parameter_count=parameter_count,
        maximum_updates=completed_updates,
        expected_sha256=expected_sha256,
    )
    if payload is None or int(payload["completed_updates"]) != int(completed_updates):
        raise ValueError("persistent coupled GPP checkpoint progress drift")
    return payload


__all__ = [
    "CHECKPOINT_SCHEMA",
    "RECOVERY_SCHEMA",
    "load_persistent_checkpoint",
    "load_persistent_recovery",
    "write_persistent_checkpoint",
    "write_persistent_recovery",
]
