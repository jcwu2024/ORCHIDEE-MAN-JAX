"""Optimizer and exact-epoch helpers for strong-baseline qualification."""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class StrongBaselineAdamWState(NamedTuple):
    step: Any
    first_moment: Any
    second_moment: Any


def padded_epoch_batches(
    sample_count: int,
    batch_size: int,
    *,
    seed: int,
    epoch: int,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    """Shuffle every real sample once and zero-weight only final padding."""

    if sample_count < 1 or batch_size < 1 or epoch < 1:
        raise ValueError("sample count, batch size, and epoch must be positive")
    rng = np.random.default_rng(np.random.SeedSequence((int(seed), int(epoch))))
    order = rng.permutation(sample_count).astype(np.int32, copy=False)
    padding = (-sample_count) % batch_size
    if padding:
        order = np.concatenate((order, order[:padding]))
    result = []
    for start in range(0, order.size, batch_size):
        indices = order[start : start + batch_size]
        weights = np.ones((batch_size,), dtype=np.float32)
        if padding and start + batch_size == order.size:
            weights[-padding:] = 0.0
        result.append((indices, weights))
    return tuple(result)


def qualification_learning_rate(
    update: int,
    maximum_updates: int,
    *,
    peak: float = 1.0e-4,
    final: float = 1.0e-5,
    warmup_fraction: float = 0.02,
) -> float:
    """Linear warmup followed by cosine decay over the qualification budget."""

    if not 1 <= update <= maximum_updates:
        raise ValueError("qualification update lies outside its declared budget")
    if not 0.0 < warmup_fraction < 1.0 or not 0.0 < final <= peak:
        raise ValueError("invalid qualification learning-rate schedule")
    warmup = max(1, int(np.ceil(maximum_updates * warmup_fraction)))
    if update <= warmup:
        return float(peak * update / warmup)
    progress = (update - warmup) / max(maximum_updates - warmup, 1)
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return float(final + (peak - final) * cosine)


def _path_key(path: tuple[Any, ...]) -> str:
    if not path:
        return ""
    item = path[-1]
    return str(getattr(item, "key", getattr(item, "idx", item)))


def strong_baseline_weight_decay_mask(parameters: Any) -> Any:
    """Exclude all biases and normalization scales from AdamW decay."""

    def admitted(path: tuple[Any, ...], _value: Any) -> bool:
        key = _path_key(path)
        return not (
            key == "bias"
            or key.endswith("_bias")
            or key == "scale"
            or key.endswith("_norm_scale")
        )

    return jax.tree_util.tree_map_with_path(admitted, parameters)


def initialize_strong_baseline_adamw(parameters: Any) -> StrongBaselineAdamWState:
    zeros = jax.tree_util.tree_map(jnp.zeros_like, parameters)
    return StrongBaselineAdamWState(
        step=jnp.asarray(0, dtype=jnp.int32),
        first_moment=zeros,
        second_moment=zeros,
    )


def strong_baseline_adamw_update(
    parameters: Any,
    gradients: Any,
    state: StrongBaselineAdamWState,
    learning_rate: Any,
    decay_mask: Any,
    *,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1.0e-8,
    weight_decay: float = 1.0e-4,
) -> tuple[Any, StrongBaselineAdamWState]:
    """Apply one unclipped float32 AdamW update."""

    step = state.step + jnp.asarray(1, dtype=jnp.int32)
    first = jax.tree_util.tree_map(
        lambda prior, gradient: beta1 * prior + (1.0 - beta1) * gradient,
        state.first_moment,
        gradients,
    )
    second = jax.tree_util.tree_map(
        lambda prior, gradient: beta2 * prior + (1.0 - beta2) * jnp.square(gradient),
        state.second_moment,
        gradients,
    )
    first_correction = 1.0 - jnp.power(jnp.asarray(beta1, dtype=jnp.float32), step)
    second_correction = 1.0 - jnp.power(jnp.asarray(beta2, dtype=jnp.float32), step)
    rate = jnp.asarray(learning_rate, dtype=jnp.float32)

    def update(parameter, first_value, second_value, apply_decay):
        direction = (first_value / first_correction) / (
            jnp.sqrt(second_value / second_correction) + epsilon
        )
        direction = direction + (
            jnp.asarray(apply_decay, dtype=parameter.dtype)
            * jnp.asarray(weight_decay, dtype=parameter.dtype)
            * parameter
        )
        return parameter - rate * direction

    next_parameters = jax.tree_util.tree_map(
        update,
        parameters,
        first,
        second,
        decay_mask,
    )
    return next_parameters, StrongBaselineAdamWState(step, first, second)


__all__ = [
    "StrongBaselineAdamWState",
    "initialize_strong_baseline_adamw",
    "padded_epoch_batches",
    "qualification_learning_rate",
    "strong_baseline_adamw_update",
    "strong_baseline_weight_decay_mask",
]
