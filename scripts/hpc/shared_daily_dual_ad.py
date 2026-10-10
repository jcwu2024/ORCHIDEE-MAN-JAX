"""Expose only the four fixed daily input states as an additional AD input."""
from __future__ import annotations

from typing import Any, Callable


EXPECTED_ROWS = 4
EXPECTED_STATE_WIDTH = 3854


def replace_initial_states(daily_batch: Any, initial_states: Any) -> Any:
    expected = (EXPECTED_ROWS, EXPECTED_STATE_WIDTH)
    if tuple(daily_batch.inputs.continuous_state.shape) != expected:
        raise ValueError(
            f"daily input state must have shape {expected}, got "
            f"{tuple(daily_batch.inputs.continuous_state.shape)}"
        )
    if tuple(initial_states.shape) != expected:
        raise ValueError(
            f"explicit state must have shape {expected}, got {tuple(initial_states.shape)}"
        )
    return daily_batch._replace(
        inputs=daily_batch.inputs._replace(continuous_state=initial_states)
    )


def make_daily_dual_objective(
    jax: Any,
    daily_objective: Callable[[Any, Any], Any],
) -> Callable[[Any, Any, Any], Any]:
    """Expose parameters, four states, and every daily sample as dynamic inputs."""

    def objective(parameters: Any, states: Any, daily_samples: Any) -> Any:
        daily_batch = replace_initial_states(daily_samples, states)
        return daily_objective(parameters, daily_batch)

    return objective


def make_daily_dual_value_and_grad(jax: Any, objective: Callable[[Any, Any, Any], Any]):
    """Return native AD for parameters and initial states only."""
    return jax.jit(jax.value_and_grad(objective, argnums=(0, 1), has_aux=True))


def make_scaled_daily_dual_value_and_grad(jax: Any, objective):
    """Native AD of S*loss, returning original values and dtype-preserving g/S.

    S is a dynamic fourth argument. Every gradient divide stays within the
    timed compiled call. This is conditioning, with no custom AD, loss
    reweighting, frozen parameters or shortened state derivatives.
    """
    import jax.numpy as jnp

    def scaled(parameters, initial_states, daily_samples, scale):
        loss, auxiliary = objective(parameters, initial_states, daily_samples)
        return loss * jnp.asarray(scale, dtype=loss.dtype), (loss, auxiliary)

    native = jax.value_and_grad(scaled, argnums=(0, 1), has_aux=True)

    def unscaled(parameters, initial_states, daily_samples, scale):
        (_scaled_loss, original), gradients = native(parameters, initial_states, daily_samples, scale)
        gradients = jax.tree_util.tree_map(
            lambda value: value / jnp.asarray(scale, dtype=value.dtype), gradients)
        return original, gradients

    return jax.jit(unscaled)


__all__ = [
    "EXPECTED_ROWS",
    "EXPECTED_STATE_WIDTH",
    "make_daily_dual_objective",
    "make_daily_dual_value_and_grad",
    "make_scaled_daily_dual_value_and_grad",
    "replace_initial_states",
]
