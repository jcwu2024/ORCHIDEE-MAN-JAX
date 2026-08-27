"""Atomic branch-only recovery and checkpoint for source-shaped training."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .successor_fit_checkpoint import _tree_schema, _validate_finite_tree, sha256_file

RECOVERY_SCHEMA = "target_oriented_source_shaped_recovery_v1"
CHECKPOINT_SCHEMA = "target_oriented_source_shaped_checkpoint_v1"


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


def _validate_state(
    branches: Any,
    optimizer: Any,
    *,
    parameter_count: int,
    completed_updates: int,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    observed = _validate_finite_tree(branches, label="source-shaped branches")
    if observed != int(parameter_count):
        raise ValueError("source-shaped checkpoint parameter count drift")
    _validate_finite_tree(optimizer, label="source-shaped optimizer")
    if _optimizer_step(optimizer) != int(completed_updates):
        raise ValueError("source-shaped checkpoint optimizer step drift")
    return _tree_schema(branches), _tree_schema(optimizer)


def _validate_records(records: Sequence[Mapping[str, Any]], completed_updates: int) -> None:
    expected = list(range(1, int(completed_updates) + 1))
    if [int(record.get("optimizer_update", -1)) for record in records] != expected:
        raise ValueError("source-shaped checkpoint update coverage drift")


def write_source_shaped_recovery(
    path: str | Path,
    *,
    identity: Mapping[str, Any],
    branches: Any,
    optimizer: Any,
    completed_updates: int,
    parameter_count: int,
    clip_threshold: float,
    update_records: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    completed = int(completed_updates)
    if completed <= 0:
        raise ValueError("source-shaped recovery requires a positive update count")
    _validate_records(update_records, completed)
    branch_schema, optimizer_schema = _validate_state(
        branches,
        optimizer,
        parameter_count=parameter_count,
        completed_updates=completed,
    )
    payload = {
        "schema_version": RECOVERY_SCHEMA,
        "promotion_eligible": False,
        "identity": dict(identity),
        "completed_updates": completed,
        "parameter_count": int(parameter_count),
        "clip_threshold": float(clip_threshold),
        "branch_tree_schema": branch_schema,
        "optimizer_tree_schema": optimizer_schema,
        "branches": branches,
        "optimizer": optimizer,
        "update_records": list(update_records),
    }
    target = Path(path).resolve()
    _atomic_pickle(target, payload)
    return {
        "path": str(target),
        "sha256": sha256_file(target),
        "completed_updates": completed,
        "promotion_eligible": False,
    }


def load_source_shaped_recovery(
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
    ):
        raise ValueError("source-shaped recovery identity drift")
    _validate_records(payload.get("update_records", ()), completed)
    branch_schema, optimizer_schema = _validate_state(
        payload["branches"],
        payload["optimizer"],
        parameter_count=parameter_count,
        completed_updates=completed,
    )
    if branch_schema != payload.get("branch_tree_schema") or optimizer_schema != payload.get("optimizer_tree_schema"):
        raise ValueError("source-shaped recovery tree schema drift")
    return payload


def write_source_shaped_checkpoint(
    path: str | Path,
    *,
    identity: Mapping[str, Any],
    branches: Any,
    optimizer: Any,
    completed_updates: int,
    parameter_count: int,
    clip_threshold: float,
    update_records: Sequence[Mapping[str, Any]],
    candidate_groups: Sequence[Mapping[str, Any]],
    screen_comparison: Mapping[str, Any],
) -> Mapping[str, Any]:
    completed = int(completed_updates)
    _validate_records(update_records, completed)
    branch_schema, optimizer_schema = _validate_state(
        branches,
        optimizer,
        parameter_count=parameter_count,
        completed_updates=completed,
    )
    if len(candidate_groups) != 8 or not screen_comparison:
        raise ValueError("source-shaped checkpoint requires the complete final screen")
    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "promotion_eligible": False,
        "identity": dict(identity),
        "completed_updates": completed,
        "parameter_count": int(parameter_count),
        "clip_threshold": float(clip_threshold),
        "branch_tree_schema": branch_schema,
        "optimizer_tree_schema": optimizer_schema,
        "branches": branches,
        "optimizer": optimizer,
        "update_records": list(update_records),
        "candidate_groups": list(candidate_groups),
        "screen_comparison": dict(screen_comparison),
    }
    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError(target)
    _atomic_pickle(target, payload)
    return {
        "path": str(target),
        "sha256": sha256_file(target),
        "completed_updates": completed,
        "parameter_count": int(parameter_count),
        "promotion_eligible": False,
    }


def load_source_shaped_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str,
    identity: Mapping[str, Any],
    completed_updates: int,
    parameter_count: int,
    clip_threshold: float,
) -> Mapping[str, Any]:
    source = Path(path).resolve()
    if not source.is_file() or sha256_file(source) != str(expected_sha256):
        raise ValueError("source-shaped checkpoint hash drift")
    with source.open("rb") as handle:
        payload = pickle.load(handle)
    completed = int(completed_updates)
    if (
        payload.get("schema_version") != CHECKPOINT_SCHEMA
        or payload.get("promotion_eligible") is not False
        or payload.get("identity") != dict(identity)
        or int(payload.get("completed_updates", -1)) != completed
        or int(payload.get("parameter_count", -1)) != int(parameter_count)
        or float(payload.get("clip_threshold", np.nan)) != float(clip_threshold)
        or len(payload.get("candidate_groups", ())) != 8
        or not payload.get("screen_comparison")
    ):
        raise ValueError("source-shaped checkpoint identity or coverage drift")
    _validate_records(payload.get("update_records", ()), completed)
    branch_schema, optimizer_schema = _validate_state(
        payload["branches"],
        payload["optimizer"],
        parameter_count=parameter_count,
        completed_updates=completed,
    )
    if branch_schema != payload.get("branch_tree_schema") or optimizer_schema != payload.get("optimizer_tree_schema"):
        raise ValueError("source-shaped checkpoint tree schema drift")
    return payload


__all__ = [
    "CHECKPOINT_SCHEMA",
    "RECOVERY_SCHEMA",
    "load_source_shaped_checkpoint",
    "load_source_shaped_recovery",
    "write_source_shaped_checkpoint",
    "write_source_shaped_recovery",
]
