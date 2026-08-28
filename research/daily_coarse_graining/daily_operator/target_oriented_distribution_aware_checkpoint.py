"""Atomic adapter-only recovery and checkpoint for bounded training."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .successor_fit_checkpoint import _tree_schema, _validate_finite_tree, sha256_file

RECOVERY_SCHEMA = "target_oriented_distribution_aware_recovery_v1"
CHECKPOINT_SCHEMA = "target_oriented_distribution_aware_checkpoint_v1"


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
    adapter: Any,
    optimizer: Any,
    *,
    parameter_count: int,
    completed_updates: int,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    observed = _validate_finite_tree(adapter, label="distribution-aware adapter")
    if observed != int(parameter_count):
        raise ValueError("distribution-aware checkpoint parameter count drift")
    _validate_finite_tree(optimizer, label="distribution-aware optimizer")
    if _optimizer_step(optimizer) != int(completed_updates):
        raise ValueError("distribution-aware checkpoint optimizer step drift")
    return _tree_schema(adapter), _tree_schema(optimizer)


def _validate_records(records: Sequence[Mapping[str, Any]], completed_updates: int) -> None:
    expected = list(range(1, int(completed_updates) + 1))
    if [int(record.get("optimizer_update", -1)) for record in records] != expected:
        raise ValueError("distribution-aware checkpoint update coverage drift")


def _screen_payload(screen: Mapping[str, Any]) -> Mapping[str, Any]:
    if len(screen.get("candidate_groups", ())) != 8 or not screen.get("comparison_to_fixed_parent"):
        raise ValueError("distribution-aware checkpoint requires a complete screen")
    return dict(screen)


def write_distribution_aware_recovery(
    path: str | Path,
    *,
    identity: Mapping[str, Any],
    adapter: Any,
    optimizer: Any,
    completed_updates: int,
    parameter_count: int,
    clip_threshold: float,
    update_records: Sequence[Mapping[str, Any]],
    initial_screen: Mapping[str, Any],
) -> Mapping[str, Any]:
    completed = int(completed_updates)
    if completed <= 0:
        raise ValueError("distribution-aware recovery requires a positive update count")
    _validate_records(update_records, completed)
    adapter_schema, optimizer_schema = _validate_state(
        adapter,
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
        "adapter_tree_schema": adapter_schema,
        "optimizer_tree_schema": optimizer_schema,
        "adapter": adapter,
        "optimizer": optimizer,
        "update_records": list(update_records),
        "initial_screen": _screen_payload(initial_screen),
    }
    target = Path(path).resolve()
    _atomic_pickle(target, payload)
    return {
        "path": str(target),
        "sha256": sha256_file(target),
        "completed_updates": completed,
        "promotion_eligible": False,
    }


def load_distribution_aware_recovery(
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
        raise ValueError("distribution-aware recovery identity drift")
    _validate_records(payload.get("update_records", ()), completed)
    adapter_schema, optimizer_schema = _validate_state(
        payload["adapter"],
        payload["optimizer"],
        parameter_count=parameter_count,
        completed_updates=completed,
    )
    if (
        adapter_schema != payload.get("adapter_tree_schema")
        or optimizer_schema != payload.get("optimizer_tree_schema")
    ):
        raise ValueError("distribution-aware recovery tree schema drift")
    _screen_payload(payload.get("initial_screen", {}))
    return payload


def write_distribution_aware_checkpoint(
    path: str | Path,
    *,
    identity: Mapping[str, Any],
    adapter: Any,
    optimizer: Any,
    completed_updates: int,
    parameter_count: int,
    clip_threshold: float,
    update_records: Sequence[Mapping[str, Any]],
    initial_screen: Mapping[str, Any],
    final_screen: Mapping[str, Any],
) -> Mapping[str, Any]:
    completed = int(completed_updates)
    _validate_records(update_records, completed)
    adapter_schema, optimizer_schema = _validate_state(
        adapter,
        optimizer,
        parameter_count=parameter_count,
        completed_updates=completed,
    )
    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "promotion_eligible": False,
        "identity": dict(identity),
        "completed_updates": completed,
        "parameter_count": int(parameter_count),
        "clip_threshold": float(clip_threshold),
        "adapter_tree_schema": adapter_schema,
        "optimizer_tree_schema": optimizer_schema,
        "adapter": adapter,
        "optimizer": optimizer,
        "update_records": list(update_records),
        "initial_screen": _screen_payload(initial_screen),
        "final_screen": _screen_payload(final_screen),
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


def load_distribution_aware_checkpoint(
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
        raise ValueError("distribution-aware checkpoint hash drift")
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
    ):
        raise ValueError("distribution-aware checkpoint identity drift")
    _validate_records(payload.get("update_records", ()), completed)
    adapter_schema, optimizer_schema = _validate_state(
        payload["adapter"],
        payload["optimizer"],
        parameter_count=parameter_count,
        completed_updates=completed,
    )
    if (
        adapter_schema != payload.get("adapter_tree_schema")
        or optimizer_schema != payload.get("optimizer_tree_schema")
    ):
        raise ValueError("distribution-aware checkpoint tree schema drift")
    _screen_payload(payload.get("initial_screen", {}))
    _screen_payload(payload.get("final_screen", {}))
    return payload


__all__ = [
    "CHECKPOINT_SCHEMA",
    "RECOVERY_SCHEMA",
    "load_distribution_aware_checkpoint",
    "load_distribution_aware_recovery",
    "write_distribution_aware_checkpoint",
    "write_distribution_aware_recovery",
]
