"""Specialist-only recovery and checkpoint for structured-GPP warm start."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .successor_fit_checkpoint import _tree_schema, _validate_finite_tree, sha256_file

RECOVERY_SCHEMA = "target_oriented_structured_gpp_recovery_v1"
CHECKPOINT_SCHEMA = "target_oriented_structured_gpp_checkpoint_v1"


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


def _validate_records(records: Sequence[Mapping[str, Any]], completed_updates: int) -> None:
    if [int(record.get("optimizer_update", -1)) for record in records] != list(range(1, int(completed_updates) + 1)):
        raise ValueError("structured GPP checkpoint update coverage drift")


def _validate_panel(panel: Mapping[str, Any]) -> Mapping[str, Any]:
    records = tuple(panel.get("records", ()))
    loss = float(panel.get("aggregate_direct_gpp_loss", np.nan))
    if len(records) != 8 or not np.isfinite(loss) or loss < 0.0:
        raise ValueError("structured GPP checkpoint requires a complete direct panel")
    return dict(panel)


def _validate_state(
    specialist: Any,
    optimizer: Any,
    *,
    parameter_count: int,
    completed_updates: int,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    observed = _validate_finite_tree(specialist, label="structured GPP specialist")
    if observed != int(parameter_count):
        raise ValueError("structured GPP checkpoint parameter count drift")
    _validate_finite_tree(optimizer, label="structured GPP optimizer")
    if _optimizer_step(optimizer) != int(completed_updates):
        raise ValueError("structured GPP checkpoint optimizer step drift")
    return _tree_schema(specialist), _tree_schema(optimizer)


def _payload(
    *,
    schema: str,
    identity: Mapping[str, Any],
    specialist: Any,
    optimizer: Any,
    completed_updates: int,
    parameter_count: int,
    clip_threshold: float,
    update_records: Sequence[Mapping[str, Any]],
    initial_panel: Mapping[str, Any],
    final_panel: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    completed = int(completed_updates)
    _validate_records(update_records, completed)
    specialist_schema, optimizer_schema = _validate_state(
        specialist,
        optimizer,
        parameter_count=parameter_count,
        completed_updates=completed,
    )
    result = {
        "schema_version": schema,
        "promotion_eligible": False,
        "identity": dict(identity),
        "completed_updates": completed,
        "parameter_count": int(parameter_count),
        "clip_threshold": float(clip_threshold),
        "specialist_tree_schema": specialist_schema,
        "optimizer_tree_schema": optimizer_schema,
        "specialist": specialist,
        "optimizer": optimizer,
        "update_records": list(update_records),
        "initial_direct_panel": _validate_panel(initial_panel),
    }
    if final_panel is not None:
        result["final_direct_panel"] = _validate_panel(final_panel)
    return result


def write_structured_gpp_recovery(
    path: str | Path,
    **kwargs: Any,
) -> Mapping[str, Any]:
    if int(kwargs["completed_updates"]) <= 0:
        raise ValueError("structured GPP recovery requires a positive update count")
    payload = _payload(schema=RECOVERY_SCHEMA, **kwargs)
    target = Path(path).resolve()
    _atomic_pickle(target, payload)
    return {
        "path": str(target),
        "sha256": sha256_file(target),
        "completed_updates": int(payload["completed_updates"]),
        "promotion_eligible": False,
    }


def load_structured_gpp_recovery(
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
        raise ValueError("structured GPP recovery identity drift")
    _validate_records(payload.get("update_records", ()), completed)
    specialist_schema, optimizer_schema = _validate_state(
        payload["specialist"],
        payload["optimizer"],
        parameter_count=parameter_count,
        completed_updates=completed,
    )
    if specialist_schema != payload.get("specialist_tree_schema") or optimizer_schema != payload.get(
        "optimizer_tree_schema"
    ):
        raise ValueError("structured GPP recovery tree schema drift")
    _validate_panel(payload.get("initial_direct_panel", {}))
    return payload


def write_structured_gpp_checkpoint(
    path: str | Path,
    **kwargs: Any,
) -> Mapping[str, Any]:
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


def load_structured_gpp_checkpoint(
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
        raise ValueError("structured GPP checkpoint hash drift")
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
        raise ValueError("structured GPP checkpoint identity drift")
    _validate_records(payload.get("update_records", ()), completed)
    specialist_schema, optimizer_schema = _validate_state(
        payload["specialist"],
        payload["optimizer"],
        parameter_count=parameter_count,
        completed_updates=completed,
    )
    if specialist_schema != payload.get("specialist_tree_schema") or optimizer_schema != payload.get(
        "optimizer_tree_schema"
    ):
        raise ValueError("structured GPP checkpoint tree schema drift")
    _validate_panel(payload.get("initial_direct_panel", {}))
    _validate_panel(payload.get("final_direct_panel", {}))
    return payload


__all__ = [
    "CHECKPOINT_SCHEMA",
    "RECOVERY_SCHEMA",
    "load_structured_gpp_checkpoint",
    "load_structured_gpp_recovery",
    "write_structured_gpp_checkpoint",
    "write_structured_gpp_recovery",
]
