"""Atomic recovery and stage checkpoints for bounded coupled-GPP training."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .successor_fit_checkpoint import _tree_schema, _validate_finite_tree, sha256_file

RECOVERY_SCHEMA = "target_oriented_coupled_gpp_training_recovery_v1"
CHECKPOINT_SCHEMA = "target_oriented_coupled_gpp_training_checkpoint_v1"
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


def _optimizer_step(optimizer: Any) -> int:
    value = optimizer["step"] if isinstance(optimizer, Mapping) else optimizer.step
    return int(np.asarray(value))


def _validate_monitor(panel: Mapping[str, Any]) -> Mapping[str, Any]:
    records = tuple(panel.get("records", ()))
    losses = panel.get("source_mean_losses", {})
    weighted = panel.get("weighted_objective", {})
    if (
        len(records) != MONITOR_ARTIFACT_COUNT
        or len(losses) != 4
        or not all(np.isfinite(float(value)) and float(value) >= 0.0 for value in losses.values())
        or not np.isfinite(float(weighted.get("joint", np.nan)))
    ):
        raise ValueError("coupled GPP checkpoint monitor drift")
    return dict(panel)


def _validate_records(records: Sequence[Mapping[str, Any]], completed_updates: int) -> None:
    if [int(record.get("optimizer_update", -1)) for record in records] != list(range(1, int(completed_updates) + 1)):
        raise ValueError("coupled GPP checkpoint update coverage drift")


def _validate_state(
    trainable: Any,
    optimizer: Any,
    *,
    parameter_count: int,
    completed_updates: int,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    observed = _validate_finite_tree(trainable, label="coupled GPP trainable parameters")
    if observed != int(parameter_count):
        raise ValueError("coupled GPP checkpoint parameter count drift")
    _validate_finite_tree(optimizer, label="coupled GPP optimizer")
    if _optimizer_step(optimizer) != int(completed_updates):
        raise ValueError("coupled GPP checkpoint optimizer-step drift")
    return _tree_schema(trainable), _tree_schema(optimizer)


def _payload(
    *,
    schema: str,
    identity: Mapping[str, Any],
    trainable: Any,
    optimizer: Any,
    completed_updates: int,
    parameter_count: int,
    clip_threshold: float,
    update_records: Sequence[Mapping[str, Any]],
    initial_monitor: Mapping[str, Any],
    milestone_monitors: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    completed = int(completed_updates)
    _validate_records(update_records, completed)
    parameter_schema, optimizer_schema = _validate_state(
        trainable,
        optimizer,
        parameter_count=parameter_count,
        completed_updates=completed,
    )
    monitors = tuple(dict(value) for value in milestone_monitors)
    if len(monitors) != completed // 4:
        raise ValueError("coupled GPP checkpoint milestone coverage drift")
    return {
        "schema_version": schema,
        "promotion_eligible": False,
        "identity": dict(identity),
        "completed_updates": completed,
        "parameter_count": int(parameter_count),
        "clip_threshold": float(clip_threshold),
        "parameter_tree_schema": parameter_schema,
        "optimizer_tree_schema": optimizer_schema,
        "trainable": trainable,
        "optimizer": optimizer,
        "update_records": list(update_records),
        "initial_monitor": _validate_monitor(initial_monitor),
        "milestone_monitors": [_validate_monitor(value) for value in monitors],
    }


def write_coupled_gpp_recovery(path: str | Path, **kwargs: Any) -> Mapping[str, Any]:
    if int(kwargs["completed_updates"]) <= 0:
        raise ValueError("coupled GPP recovery requires progress")
    payload = _payload(schema=RECOVERY_SCHEMA, **kwargs)
    target = Path(path).resolve()
    _atomic_pickle(target, payload)
    return {
        "path": str(target),
        "sha256": sha256_file(target),
        "completed_updates": int(payload["completed_updates"]),
        "promotion_eligible": False,
    }


def write_coupled_gpp_checkpoint(path: str | Path, **kwargs: Any) -> Mapping[str, Any]:
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
    clip_threshold: float,
    maximum_updates: int,
    expected_sha256: str | None = None,
) -> Mapping[str, Any] | None:
    source = Path(path).resolve()
    if not source.exists() and schema == RECOVERY_SCHEMA:
        return None
    if not source.is_file() or (expected_sha256 is not None and sha256_file(source) != expected_sha256):
        raise ValueError("coupled GPP checkpoint file identity drift")
    with source.open("rb") as handle:
        payload = pickle.load(handle)
    completed = int(payload.get("completed_updates", -1))
    if (
        payload.get("schema_version") != schema
        or payload.get("promotion_eligible") is not False
        or payload.get("identity") != dict(identity)
        or int(payload.get("parameter_count", -1)) != int(parameter_count)
        or float(payload.get("clip_threshold", np.nan)) != float(clip_threshold)
        or not 0 < completed <= int(maximum_updates)
    ):
        raise ValueError("coupled GPP checkpoint identity drift")
    _validate_records(payload.get("update_records", ()), completed)
    parameter_schema, optimizer_schema = _validate_state(
        payload["trainable"],
        payload["optimizer"],
        parameter_count=parameter_count,
        completed_updates=completed,
    )
    if parameter_schema != payload.get("parameter_tree_schema") or optimizer_schema != payload.get(
        "optimizer_tree_schema"
    ):
        raise ValueError("coupled GPP checkpoint tree schema drift")
    _validate_monitor(payload.get("initial_monitor", {}))
    monitors = tuple(payload.get("milestone_monitors", ()))
    if len(monitors) != completed // 4:
        raise ValueError("coupled GPP checkpoint milestone coverage drift")
    for value in monitors:
        _validate_monitor(value)
    return payload


def load_coupled_gpp_recovery(
    path: str | Path,
    *,
    identity: Mapping[str, Any],
    parameter_count: int,
    clip_threshold: float,
    maximum_updates: int,
) -> Mapping[str, Any] | None:
    return _load(
        path,
        schema=RECOVERY_SCHEMA,
        identity=identity,
        parameter_count=parameter_count,
        clip_threshold=clip_threshold,
        maximum_updates=maximum_updates,
    )


def load_coupled_gpp_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str,
    identity: Mapping[str, Any],
    completed_updates: int,
    parameter_count: int,
    clip_threshold: float,
) -> Mapping[str, Any]:
    payload = _load(
        path,
        schema=CHECKPOINT_SCHEMA,
        identity=identity,
        parameter_count=parameter_count,
        clip_threshold=clip_threshold,
        maximum_updates=completed_updates,
        expected_sha256=expected_sha256,
    )
    if payload is None or int(payload["completed_updates"]) != int(completed_updates):
        raise ValueError("coupled GPP stage-checkpoint progress drift")
    return payload


__all__ = [
    "CHECKPOINT_SCHEMA",
    "RECOVERY_SCHEMA",
    "load_coupled_gpp_checkpoint",
    "load_coupled_gpp_recovery",
    "write_coupled_gpp_checkpoint",
    "write_coupled_gpp_recovery",
]
