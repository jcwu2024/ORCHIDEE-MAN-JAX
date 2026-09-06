"""Identity-exact persistence for conditioned-Attention training smoke."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .successor_fit_checkpoint import _tree_schema, _validate_finite_tree, sha256_file
from .target_oriented_coupled_gpp_conditioned_attention_persistent_training import (
    FINAL_LOGICAL_UPDATE,
    LOGICAL_UPDATES,
    START_COMPLETED_LOGICAL_UPDATE,
)
from .target_oriented_coupled_gpp_direct_generalization import MONITOR_UPDATES
from .target_oriented_coupled_gpp_objective import GRADIENT_SOURCES
from .target_oriented_coupled_gpp_persistent_scale_training import ALL_REPORTED_KEYS

START_SCHEMA = "target_oriented_coupled_gpp_conditioned_attention_start_v1"
RECOVERY_SCHEMA = "target_oriented_coupled_gpp_conditioned_attention_recovery_v1"
CHECKPOINT_SCHEMA = "target_oriented_coupled_gpp_conditioned_attention_checkpoint_v1"


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


def parameter_trees_identical(left: Any, right: Any) -> bool:
    import jax

    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    return (
        left_tree == right_tree
        and len(left_leaves) == len(right_leaves)
        and all(
            np.array_equal(np.asarray(first), np.asarray(second))
            for first, second in zip(left_leaves, right_leaves, strict=True)
        )
    )


def _validate_parameter_tree(trainable: Any, parameter_count: int) -> list[Mapping[str, Any]]:
    observed = _validate_finite_tree(trainable, label="conditioned attention trainable parameters")
    if observed != int(parameter_count):
        raise ValueError("conditioned attention persistent parameter count drift")
    return _tree_schema(trainable)


def _validate_monitor_panel(panel: Mapping[str, Any]) -> Mapping[str, Any]:
    artifact_updates = tuple(int(value) for value in panel.get("artifact_updates", ()))
    records = tuple(panel.get("records", ()))
    record_updates = tuple(int(value.get("artifact_update", -1)) for value in records)
    source_losses = panel.get("source_mean_losses", {})
    key_scores = panel.get("absolute_key_scores", {})
    acceptance = panel.get("acceptance", {})
    valid_key_scores = set(key_scores) == {"7", "30"} and all(
        set(key_scores[horizon]) == set(ALL_REPORTED_KEYS)
        and all(np.isfinite(float(value)) and float(value) >= 0.0 for value in key_scores[horizon].values())
        for horizon in ("7", "30")
    )
    valid_acceptance = set(acceptance) == {"7", "30"} and all(
        set(acceptance[horizon].get("keys", {})) == set(ALL_REPORTED_KEYS) for horizon in ("7", "30")
    )
    if (
        artifact_updates != MONITOR_UPDATES
        or record_updates != MONITOR_UPDATES
        or set(source_losses) != set(GRADIENT_SOURCES)
        or not all(np.isfinite(float(value)) and float(value) >= 0.0 for value in source_losses.values())
        or not valid_key_scores
        or not valid_acceptance
        or panel.get("hard_gate_passed") is not True
    ):
        raise ValueError("conditioned attention persistent monitor drift")
    return dict(panel)


def _validate_update_records(
    records: Sequence[Mapping[str, Any]],
    completed_logical_update: int,
) -> tuple[Mapping[str, Any], ...]:
    values = tuple(records)
    expected = tuple(range(LOGICAL_UPDATES[0], int(completed_logical_update) + 1))
    observed = tuple(int(value.get("logical_update", -1)) for value in values)
    if observed != expected:
        raise ValueError("conditioned attention persistent update coverage drift")
    for value in values:
        selection = value.get("selection", {})
        scale = selection.get("selected_scale")
        if selection.get("accepted") is not True or scale is None or not np.isfinite(float(scale)):
            raise ValueError("conditioned attention persistent retained update is invalid")
    return values


def _validate_update_monitors(
    monitors: Sequence[Mapping[str, Any]],
    completed_logical_update: int,
) -> tuple[Mapping[str, Any], ...]:
    values = tuple(monitors)
    expected = tuple(range(LOGICAL_UPDATES[0], int(completed_logical_update) + 1))
    observed = tuple(int(value.get("logical_update", -1)) for value in values)
    if observed != expected:
        raise ValueError("conditioned attention persistent monitor coverage drift")
    for value in values:
        _validate_monitor_panel(value.get("panel", {}))
        for name in ("local_gate", "cumulative_gate"):
            gate = value.get(name, {})
            if gate.get("passed") is not True or gate.get("absolute_science", {}).get("passed") not in (True, False):
                raise ValueError("conditioned attention persistent accepted monitor gate drift")
    return values


def write_start_snapshot(
    path: str | Path,
    *,
    identity: Mapping[str, Any],
    trainable: Any,
    parameter_count: int,
) -> Mapping[str, Any]:
    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError(target)
    payload = {
        "schema_version": START_SCHEMA,
        "promotion_eligible": False,
        "identity": dict(identity),
        "completed_logical_update": START_COMPLETED_LOGICAL_UPDATE,
        "parameter_count": int(parameter_count),
        "parameter_tree_schema": _validate_parameter_tree(trainable, parameter_count),
        "trainable": trainable,
    }
    _atomic_pickle(target, payload)
    return {
        "path": str(target),
        "sha256": sha256_file(target),
        "completed_logical_update": START_COMPLETED_LOGICAL_UPDATE,
        "parameter_count": int(parameter_count),
        "promotion_eligible": False,
    }


def load_start_snapshot(
    path: str | Path,
    *,
    identity: Mapping[str, Any],
    parameter_count: int,
) -> Mapping[str, Any]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("rb") as handle:
        payload = pickle.load(handle)
    if (
        payload.get("schema_version") != START_SCHEMA
        or payload.get("promotion_eligible") is not False
        or payload.get("identity") != dict(identity)
        or int(payload.get("completed_logical_update", -1)) != START_COMPLETED_LOGICAL_UPDATE
        or int(payload.get("parameter_count", -1)) != int(parameter_count)
    ):
        raise ValueError("conditioned attention persistent start identity drift")
    schema = _validate_parameter_tree(payload["trainable"], parameter_count)
    if schema != payload.get("parameter_tree_schema"):
        raise ValueError("conditioned attention persistent start parameter tree drift")
    return payload


def _training_payload(
    *,
    schema_version: str,
    identity: Mapping[str, Any],
    trainable: Any,
    completed_logical_update: int,
    parameter_count: int,
    update_records: Sequence[Mapping[str, Any]],
    initial_monitor: Mapping[str, Any],
    update_monitors: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    completed = int(completed_logical_update)
    if completed < START_COMPLETED_LOGICAL_UPDATE or completed > FINAL_LOGICAL_UPDATE:
        raise ValueError("conditioned attention persistent completed update is invalid")
    records = _validate_update_records(update_records, completed)
    monitors = _validate_update_monitors(update_monitors, completed)
    return {
        "schema_version": schema_version,
        "promotion_eligible": False,
        "identity": dict(identity),
        "completed_logical_update": completed,
        "parameter_count": int(parameter_count),
        "parameter_tree_schema": _validate_parameter_tree(trainable, parameter_count),
        "trainable": trainable,
        "update_records": [dict(value) for value in records],
        "initial_monitor": _validate_monitor_panel(initial_monitor),
        "update_monitors": [dict(value) for value in monitors],
    }


def write_recovery(path: str | Path, **kwargs: Any) -> Mapping[str, Any]:
    payload = _training_payload(schema_version=RECOVERY_SCHEMA, **kwargs)
    target = Path(path).resolve()
    _atomic_pickle(target, payload)
    return {
        "path": str(target),
        "sha256": sha256_file(target),
        "completed_logical_update": int(payload["completed_logical_update"]),
        "promotion_eligible": False,
    }


def _load_training_state(
    path: str | Path,
    *,
    schema_version: str,
    identity: Mapping[str, Any],
    parameter_count: int,
    expected_sha256: str | None = None,
) -> Mapping[str, Any] | None:
    source = Path(path).resolve()
    if not source.exists() and schema_version == RECOVERY_SCHEMA:
        return None
    if not source.is_file() or (expected_sha256 is not None and sha256_file(source) != expected_sha256):
        raise ValueError("conditioned attention persistent file identity drift")
    with source.open("rb") as handle:
        payload = pickle.load(handle)
    if (
        payload.get("schema_version") != schema_version
        or payload.get("promotion_eligible") is not False
        or payload.get("identity") != dict(identity)
        or int(payload.get("parameter_count", -1)) != int(parameter_count)
    ):
        raise ValueError("conditioned attention persistent state identity drift")
    completed = int(payload.get("completed_logical_update", -1))
    if completed < START_COMPLETED_LOGICAL_UPDATE or completed > FINAL_LOGICAL_UPDATE:
        raise ValueError("conditioned attention persistent completed update is invalid")
    _validate_update_records(payload.get("update_records", ()), completed)
    _validate_monitor_panel(payload.get("initial_monitor", {}))
    _validate_update_monitors(payload.get("update_monitors", ()), completed)
    schema = _validate_parameter_tree(payload["trainable"], parameter_count)
    if schema != payload.get("parameter_tree_schema"):
        raise ValueError("conditioned attention persistent state parameter tree drift")
    return payload


def load_recovery(
    path: str | Path,
    *,
    identity: Mapping[str, Any],
    parameter_count: int,
) -> Mapping[str, Any] | None:
    return _load_training_state(
        path,
        schema_version=RECOVERY_SCHEMA,
        identity=identity,
        parameter_count=parameter_count,
    )


def write_checkpoint(path: str | Path, **kwargs: Any) -> Mapping[str, Any]:
    payload = _training_payload(schema_version=CHECKPOINT_SCHEMA, **kwargs)
    if int(payload["completed_logical_update"]) != FINAL_LOGICAL_UPDATE:
        raise ValueError("conditioned attention checkpoint requires the final logical update")
    final_monitor = payload["update_monitors"][-1]
    if (
        final_monitor["local_gate"].get("passed") is not True
        or final_monitor["cumulative_gate"].get("passed") is not True
    ):
        raise ValueError("conditioned attention checkpoint requires passing monitor gates")
    target = Path(path).resolve()
    if target.exists():
        raise FileExistsError(target)
    _atomic_pickle(target, payload)
    return {
        "path": str(target),
        "sha256": sha256_file(target),
        "completed_logical_update": FINAL_LOGICAL_UPDATE,
        "parameter_count": int(payload["parameter_count"]),
        "promotion_eligible": False,
    }


def load_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str,
    identity: Mapping[str, Any],
    parameter_count: int,
) -> Mapping[str, Any]:
    payload = _load_training_state(
        path,
        schema_version=CHECKPOINT_SCHEMA,
        identity=identity,
        parameter_count=parameter_count,
        expected_sha256=expected_sha256,
    )
    if payload is None or int(payload["completed_logical_update"]) != FINAL_LOGICAL_UPDATE:
        raise ValueError("conditioned attention checkpoint progress drift")
    return payload


__all__ = [
    "CHECKPOINT_SCHEMA",
    "RECOVERY_SCHEMA",
    "START_SCHEMA",
    "load_checkpoint",
    "load_recovery",
    "load_start_snapshot",
    "parameter_trees_identical",
    "write_checkpoint",
    "write_recovery",
    "write_start_snapshot",
]
