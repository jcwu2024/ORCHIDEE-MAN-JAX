"""Atomic recovery and milestone checkpoints for target-oriented training."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .successor_fit_checkpoint import _tree_schema, _validate_finite_tree, sha256_file

RECOVERY_SCHEMA = "target_oriented_training_recovery_v1"
STAGE_SCHEMA = "target_oriented_training_stage_checkpoint_v1"
BEST_SCHEMA = "target_oriented_training_best_checkpoint_v1"


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


def _validate_state(
    parameters: Any,
    optimizer: Any,
    *,
    parameter_count: int,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    if _validate_finite_tree(parameters, label="target-oriented parameters") != int(parameter_count):
        raise ValueError("target-oriented checkpoint parameter count drift")
    _validate_finite_tree(optimizer, label="target-oriented optimizer")
    return _tree_schema(parameters), _tree_schema(optimizer)


def write_target_oriented_recovery(
    path: str | Path,
    *,
    identity: Mapping[str, Any],
    parameters: Any,
    optimizer: Any,
    completed_updates: int,
    parameter_count: int,
    clip_threshold: float,
    initial_validation: Mapping[str, Any],
    update_records: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if int(completed_updates) <= 0 or len(update_records) != int(completed_updates):
        raise ValueError("target-oriented recovery update coverage drift")
    parameter_schema, optimizer_schema = _validate_state(
        parameters,
        optimizer,
        parameter_count=parameter_count,
    )
    payload = {
        "schema_version": RECOVERY_SCHEMA,
        "promotion_eligible": False,
        "identity": dict(identity),
        "completed_updates": int(completed_updates),
        "parameter_count": int(parameter_count),
        "clip_threshold": float(clip_threshold),
        "parameter_tree_schema": parameter_schema,
        "optimizer_tree_schema": optimizer_schema,
        "parameters": parameters,
        "optimizer": optimizer,
        "initial_validation": dict(initial_validation),
        "update_records": list(update_records),
    }
    target = Path(path).resolve()
    _atomic_pickle(target, payload)
    return {
        "path": str(target),
        "sha256": sha256_file(target),
        "completed_updates": int(completed_updates),
        "promotion_eligible": False,
    }


def load_target_oriented_recovery(
    path: str | Path,
    *,
    identity: Mapping[str, Any],
    parameter_count: int,
    clip_threshold: float,
    maximum_updates: int,
) -> Mapping[str, Any] | None:
    source = Path(path).resolve()
    if not source.exists():
        return None
    with source.open("rb") as handle:
        payload = pickle.load(handle)
    completed = int(payload.get("completed_updates", -1))
    if (
        payload.get("schema_version") != RECOVERY_SCHEMA
        or payload.get("promotion_eligible") is not False
        or payload.get("identity") != dict(identity)
        or int(payload.get("parameter_count", -1)) != int(parameter_count)
        or float(payload.get("clip_threshold", np.nan)) != float(clip_threshold)
        or not 0 < completed <= int(maximum_updates)
        or len(payload.get("update_records", ())) != completed
        or not payload.get("initial_validation")
    ):
        raise ValueError("target-oriented recovery identity or coverage drift")
    parameter_schema, optimizer_schema = _validate_state(
        payload["parameters"],
        payload["optimizer"],
        parameter_count=parameter_count,
    )
    if (
        parameter_schema != payload.get("parameter_tree_schema")
        or optimizer_schema != payload.get("optimizer_tree_schema")
    ):
        raise ValueError("target-oriented recovery tree schema drift")
    return payload


def write_target_oriented_stage_checkpoint(
    path: str | Path,
    *,
    identity: Mapping[str, Any],
    parameters: Any,
    optimizer: Any,
    completed_updates: int,
    parameter_count: int,
    clip_threshold: float,
    validation: Mapping[str, Any],
    update_records: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if int(completed_updates) <= 0 or len(update_records) != int(completed_updates):
        raise ValueError("target-oriented stage checkpoint update coverage drift")
    parameter_schema, optimizer_schema = _validate_state(
        parameters,
        optimizer,
        parameter_count=parameter_count,
    )
    payload = {
        "schema_version": STAGE_SCHEMA,
        "promotion_eligible": False,
        "identity": dict(identity),
        "completed_updates": int(completed_updates),
        "parameter_count": int(parameter_count),
        "clip_threshold": float(clip_threshold),
        "parameter_tree_schema": parameter_schema,
        "optimizer_tree_schema": optimizer_schema,
        "parameters": parameters,
        "optimizer": optimizer,
        "validation": dict(validation),
        "update_records": list(update_records),
    }
    target = Path(path).resolve()
    _atomic_pickle(target, payload)
    return {
        "path": str(target),
        "sha256": sha256_file(target),
        "completed_updates": int(completed_updates),
        "promotion_eligible": False,
    }


def write_target_oriented_best_checkpoint(
    path: str | Path,
    *,
    identity: Mapping[str, Any],
    parameters: Any,
    completed_updates: int,
    parameter_count: int,
    validation: Mapping[str, Any],
) -> Mapping[str, Any]:
    if int(validation.get("hard_failure_count", -1)) != 0:
        raise ValueError("target-oriented best checkpoint requires zero hard failures")
    parameter_schema, _optimizer_schema = _validate_state(
        parameters,
        {"step": np.asarray(int(completed_updates), dtype=np.int32)},
        parameter_count=parameter_count,
    )
    payload = {
        "schema_version": BEST_SCHEMA,
        "status": "selected_zero-hard-failure_validation",
        "promotion_eligible": False,
        "identity": dict(identity),
        "completed_updates": int(completed_updates),
        "parameter_count": int(parameter_count),
        "parameter_tree_schema": parameter_schema,
        "parameters": parameters,
        "validation": dict(validation),
    }
    target = Path(path).resolve()
    _atomic_pickle(target, payload)
    return {
        "path": str(target),
        "sha256": sha256_file(target),
        "completed_updates": int(completed_updates),
        "promotion_eligible": False,
    }


__all__ = [
    "BEST_SCHEMA",
    "RECOVERY_SCHEMA",
    "STAGE_SCHEMA",
    "load_target_oriented_recovery",
    "write_target_oriented_best_checkpoint",
    "write_target_oriented_recovery",
    "write_target_oriented_stage_checkpoint",
]
