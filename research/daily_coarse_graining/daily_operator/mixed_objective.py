"""Minimal gradient composition for the declared mixed training objective."""

from __future__ import annotations

from typing import Any, Callable


def make_mixed_gradient(jax: Any, jnp: Any) -> Callable[[Any, Any, Any, Any], Any]:
    """Return a jitted weighted sum of two independently averaged gradients.

    The caller owns each component's sample accumulation and denominator.  In
    particular, a daily-example mean and a recurrent-window mean must be
    finished separately before this function is called.  Gradient clipping
    and the optimizer update belong after the returned mixed gradient.
    """

    def combine(
        daily_mean: Any,
        recurrent_mean: Any,
        daily_weight: Any,
        recurrent_weight: Any,
    ) -> Any:
        return jax.tree_util.tree_map(
            lambda daily, recurrent: (
                jnp.asarray(daily_weight, dtype=daily.dtype) * daily
                + jnp.asarray(recurrent_weight, dtype=recurrent.dtype) * recurrent
            ),
            daily_mean,
            recurrent_mean,
        )

    return jax.jit(combine)


__all__ = ["make_mixed_gradient"]
