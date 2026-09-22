"""Execution-only microbatching for the shared one-day training objective.

The process objective remains a scalar single-row function.  This module only
changes how its exact equal-row mean and gradient are evaluated: a small vmap
is differentiated once per microbatch, then equal-sized microbatch gradients
are accumulated on device.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


class MicrobatchFailure(ValueError):
    """A microbatch violated an existing numerical or physical training gate."""

    def __init__(self, reason: str, *, microbatch_index: int, **details: Any):
        super().__init__(reason)
        self.reason = reason
        self.microbatch_index = int(microbatch_index)
        self.details = details


def _path_names(path: tuple[Any, ...]) -> tuple[str, ...]:
    names = []
    for entry in path:
        if hasattr(entry, "key"):
            names.append(str(entry.key))
        elif hasattr(entry, "name"):
            names.append(str(entry.name))
        elif hasattr(entry, "idx"):
            names.append(str(entry.idx))
        else:
            names.append(str(entry))
    return tuple(names)


def _is_count_path(path: tuple[Any, ...]) -> bool:
    """Identify additive diagnostic counts in the shared daily auxiliary."""

    names = tuple(name.lower() for name in _path_names(path))
    leaf = names[-1] if names else ""
    return (
        "hard" in names
        or "hard_counts" in names
        or leaf == "count"
        or leaf.endswith("_count")
        or leaf.endswith("_counts")
        or leaf == "declared_mask_change"
    )


def _aggregate_auxiliary(jax: Any, jnp: Any, auxiliary: Any) -> Any:
    def reduce(path: tuple[Any, ...], value: Any) -> Any:
        if _is_count_path(path):
            return jnp.sum(value, axis=0)
        return jnp.mean(value, axis=0)

    return jax.tree_util.tree_map_with_path(reduce, auxiliary)


def make_microbatch_objective(
    jax: Any,
    jnp: Any,
    daily_objective: Callable[[Any, Any], tuple[Any, Any]],
) -> Callable[[Any, Any], tuple[Any, Any]]:
    """Lift the scalar single-row daily objective to an equal-row mean.

    The returned objective is differentiated outside the vmap, so no
    ``microbatch_size x parameter_tree`` collection of per-row gradients is
    materialized.  Loss-like auxiliary leaves are means.  Diagnostic counts
    are sums, matching serial accumulation across the same rows.
    """

    evaluate_rows = jax.vmap(daily_objective, in_axes=(None, 0))

    def objective(parameters: Any, samples: Any) -> tuple[Any, Any]:
        values, auxiliary = evaluate_rows(parameters, samples)
        return jnp.mean(values, axis=0), _aggregate_auxiliary(jax, jnp, auxiliary)

    return objective


def compile_microbatch_evaluator(
    *,
    jax: Any,
    parameters: Any,
    microbatch_objective: Callable[[Any, Any], tuple[Any, Any]],
    microbatch_example: Any,
) -> Callable[[Any, Any], tuple[tuple[Any, Any], Any]]:
    """Compile one fixed-shape value-and-mean-gradient executable."""

    backward = jax.jit(jax.value_and_grad(microbatch_objective, has_aux=True))
    return backward.lower(parameters, microbatch_example).compile()


@dataclass(frozen=True)
class _AccumulatorOperations:
    add: Callable[[Any, Any], Any]
    finish: Callable[[Any, Any, Any], tuple[tuple[Any, Any], Any]]
    finite: Callable[[Any], Any]


def _make_accumulator_operations(jax: Any, jnp: Any, microbatch_count: int) -> _AccumulatorOperations:
    add = jax.jit(
        lambda left, right: jax.tree_util.tree_map(lambda x, y: x + y, left, right)
    )

    def finish(value_total: Any, auxiliary_total: Any, gradient_total: Any) -> Any:
        mean_value = value_total / microbatch_count
        mean_gradient = jax.tree_util.tree_map(
            lambda value: value / microbatch_count, gradient_total
        )

        def finish_auxiliary(path: tuple[Any, ...], value: Any) -> Any:
            if _is_count_path(path):
                return value
            return value / microbatch_count

        aggregate_auxiliary = jax.tree_util.tree_map_with_path(
            finish_auxiliary, auxiliary_total
        )
        return (mean_value, aggregate_auxiliary), mean_gradient

    def finite(tree: Any) -> Any:
        leaves = jax.tree_util.tree_leaves(tree)
        return jnp.all(jnp.stack([jnp.all(jnp.isfinite(value)) for value in leaves]))

    return _AccumulatorOperations(add=add, finish=jax.jit(finish), finite=jax.jit(finite))


def _batch_size(jax: Any, daily_batch: Any) -> int:
    leaves = jax.tree_util.tree_leaves(daily_batch)
    if not leaves:
        raise ValueError("daily batch cannot have an empty pytree")
    sizes = []
    for leaf in leaves:
        shape = np.shape(leaf)
        if not shape:
            raise ValueError("every daily batch leaf must have a leading row axis")
        sizes.append(int(shape[0]))
    if len(set(sizes)) != 1:
        raise ValueError("daily batch leaves have unequal row counts")
    return sizes[0]


def _hard_counts(auxiliary: Any) -> dict[str, int]:
    if not isinstance(auxiliary, dict) or "hard" not in auxiliary:
        raise ValueError("daily auxiliary must contain the declared hard-count mapping")
    hard = auxiliary["hard"]
    if not isinstance(hard, dict):
        raise ValueError("daily auxiliary hard counts must be a mapping")
    return {name: int(np.asarray(value)) for name, value in hard.items()}


def accumulate_daily_gradients(
    *,
    jax: Any,
    jnp: Any,
    parameters: Any,
    daily_batch: Any,
    microbatch_size: int,
    evaluator: Callable[[Any, Any], tuple[tuple[Any, Any], Any]],
    expected_count: int,
    _operations: _AccumulatorOperations | None = None,
) -> tuple[tuple[Any, Any], Any]:
    """Evaluate and accumulate an exact equal-sample daily mean gradient.

    Full gradient pytrees remain on device.  Only scalar finite and hard-gate
    results are inspected on the host after each microbatch, preserving the
    existing training rejection behavior inside the timed execution path.
    Repeated callers should use :func:`make_daily_accumulator` so the small
    accumulation operations are compiled once and cached by the closure.
    """

    size = int(microbatch_size)
    count = int(expected_count)
    if size <= 0 or count <= 0:
        raise ValueError("microbatch size and expected daily count must be positive")
    actual_count = _batch_size(jax, daily_batch)
    if actual_count != count:
        raise ValueError("daily batch row count differs from the declared mean")
    if count % size:
        raise ValueError("daily batch must divide into equal-sized microbatches")

    microbatch_count = count // size
    operations = _operations or _make_accumulator_operations(jax, jnp, microbatch_count)
    value_total = None
    auxiliary_total = None
    gradient_total = None
    for microbatch_index, start in enumerate(range(0, count, size)):
        microbatch = jax.tree_util.tree_map(
            lambda value, lower=start: value[lower : lower + size], daily_batch
        )
        (value, auxiliary), gradient = evaluator(parameters, microbatch)
        hard = _hard_counts(auxiliary)
        finite_value = bool(np.isfinite(float(np.asarray(value))))
        finite_gradient = bool(np.asarray(operations.finite(gradient)))
        if not finite_value or not finite_gradient or any(value != 0 for value in hard.values()):
            raise MicrobatchFailure(
                "daily_numerical_or_physical",
                microbatch_index=microbatch_index,
                row_start=start,
                row_stop=start + size,
                value=float(np.asarray(value)),
                finite_gradient=finite_gradient,
                hard=hard,
            )
        if value_total is None:
            value_total, auxiliary_total, gradient_total = value, auxiliary, gradient
        else:
            value_total = value_total + value
            auxiliary_total = operations.add(auxiliary_total, auxiliary)
            gradient_total = operations.add(gradient_total, gradient)

    return operations.finish(value_total, auxiliary_total, gradient_total)


def make_daily_accumulator(
    *,
    jax: Any,
    jnp: Any,
    microbatch_size: int,
    evaluator: Callable[[Any, Any], tuple[tuple[Any, Any], Any]],
    expected_count: int,
) -> Callable[[Any, Any], tuple[tuple[Any, Any], Any]]:
    """Bind one compiled evaluator and cached device accumulation operations."""

    size = int(microbatch_size)
    count = int(expected_count)
    if size <= 0 or count <= 0 or count % size:
        raise ValueError("declared daily count must divide into equal-sized microbatches")
    operations = _make_accumulator_operations(jax, jnp, count // size)

    def evaluate(parameters: Any, daily_batch: Any) -> tuple[tuple[Any, Any], Any]:
        return accumulate_daily_gradients(
            jax=jax,
            jnp=jnp,
            parameters=parameters,
            daily_batch=daily_batch,
            microbatch_size=size,
            evaluator=evaluator,
            expected_count=count,
            _operations=operations,
        )

    return evaluate


__all__ = [
    "MicrobatchFailure",
    "accumulate_daily_gradients",
    "compile_microbatch_evaluator",
    "make_daily_accumulator",
    "make_microbatch_objective",
]
