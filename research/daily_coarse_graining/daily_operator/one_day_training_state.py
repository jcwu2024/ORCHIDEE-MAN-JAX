"""Gradient accumulation and exact recovery for prepared one-day training."""

from __future__ import annotations

import json
import operator
import os
import pickle
from pathlib import Path
from typing import Any, Mapping

CHECKPOINT_SCHEMA = "fresh_one_day_pool_checkpoint_v1"
_CHECKPOINT_KEYS = {
    "schema_version",
    "parameters",
    "optimizer_state",
    "completed_updates",
    "cursor",
    "method",
}


def make_gradient_accumulator(jax: Any, jnp: Any) -> tuple[Any, Any]:
    """Return jitted weighted-add and finish functions for mean gradients.

    ``total_gradient`` is initialized with zeros shaped like the parameters.
    Each supplied microgradient is already a mean over ``count`` samples, so
    ``add`` converts it back to a sample-weighted sum. Optimizer clipping and
    updates belong after ``finish`` has divided by the complete sample count.
    """

    def add(total_gradient: Any, mean_microgradient: Any, count: Any) -> Any:
        return jax.tree_util.tree_map(
            lambda total, gradient: total
            + gradient * jnp.asarray(count, dtype=gradient.dtype),
            total_gradient,
            mean_microgradient,
        )

    def finish(total_gradient: Any, total_count: Any) -> Any:
        return jax.tree_util.tree_map(
            lambda total: total / jnp.asarray(total_count, dtype=total.dtype),
            total_gradient,
        )

    return jax.jit(add), jax.jit(finish)


def _nonnegative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        result = int(operator.index(value))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if result < 0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _cursor(value: Any) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != {"epoch", "batch_offset"}:
        raise TypeError("cursor must be a dict with epoch and batch_offset")
    return {
        "epoch": _nonnegative_integer(value["epoch"], name="cursor epoch"),
        "batch_offset": _nonnegative_integer(
            value["batch_offset"], name="cursor batch_offset"
        ),
    }


def _method(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a method dict")
    if not value:
        raise ValueError(f"{name} must not be empty")
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be JSON-compatible") from error
    return json.loads(encoded)


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


def save_training_checkpoint(
    path: str | Path,
    *,
    jax: Any,
    parameters: Any,
    optimizer_state: Any,
    completed_updates: int,
    cursor: dict[str, int],
    method: dict[str, Any],
) -> Mapping[str, Any]:
    """Atomically save a host-resident full training state."""

    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "parameters": parameters,
        "optimizer_state": optimizer_state,
        "completed_updates": _nonnegative_integer(
            completed_updates, name="completed_updates"
        ),
        "cursor": _cursor(cursor),
        "method": _method(method, name="method"),
    }
    host_payload = jax.device_get(payload)
    _atomic_pickle(Path(path).resolve(), host_payload)
    return host_payload


def load_training_checkpoint(
    path: str | Path,
    *,
    expected_method: dict[str, Any],
) -> Mapping[str, Any]:
    """Load and validate a complete host-resident one-day training state."""

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict) or set(payload) != _CHECKPOINT_KEYS:
        raise ValueError("one-day training checkpoint structure drift")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise ValueError("one-day training checkpoint schema drift")
    observed_method = _method(payload.get("method"), name="checkpoint method")
    required_method = _method(expected_method, name="expected_method")
    if observed_method != required_method:
        raise ValueError("one-day training checkpoint method mismatch")
    completed_updates = _nonnegative_integer(
        payload.get("completed_updates"), name="completed_updates"
    )
    cursor = _cursor(payload.get("cursor"))
    return {
        **payload,
        "completed_updates": completed_updates,
        "cursor": cursor,
        "method": observed_method,
    }


__all__ = [
    "CHECKPOINT_SCHEMA",
    "load_training_checkpoint",
    "make_gradient_accumulator",
    "save_training_checkpoint",
]
