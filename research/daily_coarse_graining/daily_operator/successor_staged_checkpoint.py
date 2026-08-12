"""Atomic recovery and best-parameter state for staged successor training."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

from .successor_fit_checkpoint import _validate_finite_tree

STAGED_RECOVERY_SCHEMA = "successor_staged_recovery_v1"
STAGED_BEST_CHECKPOINT_SCHEMA = "successor_staged_best_checkpoint_v1"


def write_staged_recovery(
    path: str | Path,
    *,
    identity: Mapping[str, Any],
    parameters: Mapping[str, Any],
    optimizers: Mapping[str, Any],
    completed_updates: int,
    parameter_counts: Mapping[str, int],
    validation_state: Mapping[str, Any] | None,
    best_parameters: Mapping[str, Any],
    expected_arms: Sequence[str],
) -> Mapping[str, Any]:
    arms = tuple(expected_arms)
    expected = set(arms)
    if (
        int(completed_updates) <= 0
        or set(parameters) != expected
        or set(optimizers) != expected
        or set(parameter_counts) != expected
        or set(best_parameters) != expected
    ):
        raise ValueError("successor staged recovery inventory drift")
    for arm in arms:
        count = _validate_finite_tree(parameters[arm], label=f"{arm} staged parameter")
        _validate_finite_tree(optimizers[arm], label=f"{arm} staged optimizer")
        best_count = _validate_finite_tree(
            best_parameters[arm],
            label=f"{arm} staged best parameter",
        )
        if count != int(parameter_counts[arm]) or best_count != count:
            raise ValueError("successor staged recovery parameter count drift")
    payload = {
        "schema_version": STAGED_RECOVERY_SCHEMA,
        "promotion_eligible": False,
        "identity": dict(identity),
        "arms": list(arms),
        "completed_updates": int(completed_updates),
        "parameter_counts": {arm: int(parameter_counts[arm]) for arm in arms},
        "parameters": dict(parameters),
        "optimizers": dict(optimizers),
        "validation_state": None if validation_state is None else dict(validation_state),
        "best_parameters": dict(best_parameters),
    }
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "path": str(target),
        "completed_updates": int(completed_updates),
        "promotion_eligible": False,
    }


def load_staged_recovery(
    path: str | Path,
    *,
    identity: Mapping[str, Any],
    expected_arms: Sequence[str],
    parameter_counts: Mapping[str, int],
    minimum_completed_updates: int,
    maximum_completed_updates: int,
) -> Mapping[str, Any] | None:
    source = Path(path).resolve()
    if not source.exists():
        return None
    with source.open("rb") as handle:
        payload = pickle.load(handle)
    arms = tuple(expected_arms)
    expected = set(arms)
    if (
        payload.get("schema_version") != STAGED_RECOVERY_SCHEMA
        or payload.get("promotion_eligible") is not False
        or payload.get("identity") != dict(identity)
        or payload.get("arms") != list(arms)
        or not int(minimum_completed_updates)
        <= int(payload.get("completed_updates", -1))
        <= int(maximum_completed_updates)
        or payload.get("parameter_counts")
        != {arm: int(parameter_counts[arm]) for arm in arms}
        or set(payload.get("parameters", {})) != expected
        or set(payload.get("optimizers", {})) != expected
        or set(payload.get("best_parameters", {})) != expected
    ):
        raise ValueError("successor staged recovery identity drift")
    for arm in arms:
        count = _validate_finite_tree(
            payload["parameters"][arm],
            label=f"{arm} staged parameter",
        )
        _validate_finite_tree(
            payload["optimizers"][arm],
            label=f"{arm} staged optimizer",
        )
        best_count = _validate_finite_tree(
            payload["best_parameters"][arm],
            label=f"{arm} staged best parameter",
        )
        if count != int(parameter_counts[arm]) or best_count != count:
            raise ValueError("successor staged recovery parameter count drift")
    return payload


def write_staged_best_checkpoint(
    path: str | Path,
    *,
    identity: Mapping[str, Any],
    parameters: Mapping[str, Any],
    parameter_counts: Mapping[str, int],
    validation_state: Mapping[str, Any],
    expected_arms: Sequence[str],
) -> Mapping[str, Any]:
    arms = tuple(expected_arms)
    if set(parameters) != set(arms) or set(parameter_counts) != set(arms):
        raise ValueError("successor staged best-checkpoint inventory drift")
    for arm in arms:
        count = _validate_finite_tree(
            parameters[arm],
            label=f"{arm} staged best checkpoint",
        )
        if count != int(parameter_counts[arm]):
            raise ValueError("successor staged best-checkpoint parameter count drift")
        if validation_state["arms"][arm]["best_milestone"] is None:
            raise ValueError("successor staged best checkpoint requires a selected milestone")
    payload = {
        "schema_version": STAGED_BEST_CHECKPOINT_SCHEMA,
        "status": "accepted_validation_selection",
        "identity": dict(identity),
        "arms": list(arms),
        "parameter_counts": {arm: int(parameter_counts[arm]) for arm in arms},
        "best_milestones": {
            arm: int(validation_state["arms"][arm]["best_milestone"])
            for arm in arms
        },
        "best_primary_composites": {
            arm: float(validation_state["arms"][arm]["best_primary_composite"])
            for arm in arms
        },
        "parameters": dict(parameters),
    }
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "path": str(target),
        "best_milestones": payload["best_milestones"],
        "promotion_eligible": False,
    }
