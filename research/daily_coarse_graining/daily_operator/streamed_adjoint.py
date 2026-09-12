"""Full recurrent parameter pullback evaluated through cached one-day graphs.

This is an autodiff execution strategy; callers supply the unchanged physical
transition and observation-loss cotangents. It does not define process logic.
"""

from __future__ import annotations

from typing import Any, NamedTuple


class AdjointState(NamedTuple):
    other: Any


class AdjointPrediction(NamedTuple):
    named_outputs: Any


class AdjointResult(NamedTuple):
    next_state: AdjointState
    next_state_defined: AdjointState
    next_discrete_state: Any
    prediction: AdjointPrediction


class AdjointTrace(NamedTuple):
    result: AdjointResult
    pft_fraction: Any
    active_pft_mask: Any


def compact_adjoint_trace(trace):
    """Keep the complete recurrent carry and science observations, without auxiliaries."""
    result = trace.result
    return AdjointTrace(
        AdjointResult(
            AdjointState(result.next_state.other), AdjointState(result.next_state_defined.other),
            result.next_discrete_state, AdjointPrediction({"daily.fast_gpp": result.prediction.named_outputs["daily.fast_gpp"]}),
        ),
        trace.pft_fraction, trace.active_pft_mask,
    )


class StreamedAdjoint:
    """Reuse one reverse executable for equal-shaped windows and parameter trees."""

    def __init__(self, jax, jnp, step, *, limit_bytes=24 * 1024**3):
        self.jax, self.jnp = jax, jnp
        self.limit_bytes = limit_bytes
        self.compiled = None
        self.memory = None
        self.compile_count = 0

        def reverse(values, state, fixed, day, cotangent):
            primal, pullback = jax.vjp(lambda v, s: step(v, s, fixed, day), values, state)
            parameter_gradient, state_gradient = pullback(cotangent)
            return primal, parameter_gradient, state_gradient

        self.reverse = jax.jit(reverse)
        self.add = jax.jit(lambda a, b: jax.tree_util.tree_map(lambda x, y: x + y.astype(x.dtype), a, b))
        self.nonfinite = jax.jit(
            lambda tree: sum(jnp.sum(~jnp.isfinite(v)) for v in jax.tree_util.tree_leaves(tree))
        )

    def gradient(self, values, fixed, states, days, cotangents, *, callback=None):
        """Return the complete recurrent parameter gradient.

        This preserves the original public behavior. Use
        :meth:`gradient_with_initial_state` when the day-one carry adjoint is
        also required.
        """
        gradient, _ = self.gradient_with_initial_state(
            values, fixed, states, days, cotangents, callback=callback
        )
        return gradient

    def gradient_with_initial_state(self, values, fixed, states, days, cotangents, *, callback=None, primal_callback=None):
        """Return parameter gradient and the adjoint of ``states[0]``."""
        jax, jnp = self.jax, self.jnp
        emit = callback or (lambda event, data: None)
        if not states or len(states) != len(days) or len(states) != len(cotangents):
            raise ValueError("streamed pullback requires one carry/day/cotangent per transition")
        if self.compiled is None:
            emit("reverse_compile", {})
            compiled = self.reverse.lower(values, states[-1], fixed, days[-1], cotangents[-1]).compile()
            memory = compiled.memory_analysis()
            if memory is None:
                raise RuntimeError("one-day reverse memory estimate unavailable")
            sizes = {
                key: int(getattr(memory, key))
                for key in ("argument_size_in_bytes", "output_size_in_bytes", "temp_size_in_bytes", "alias_size_in_bytes")
            }
            sizes["estimated_peak_bytes"] = (
                sizes["argument_size_in_bytes"] + sizes["output_size_in_bytes"]
                + sizes["temp_size_in_bytes"] - sizes["alias_size_in_bytes"]
            )
            if sizes["estimated_peak_bytes"] > self.limit_bytes:
                raise RuntimeError("one-day reverse exceeds diagnostic memory envelope")
            self.compiled, self.memory = compiled, sizes
            self.compile_count += 1
            emit("reverse_memory", sizes)
        future = jnp.zeros_like(states[-1])
        accumulated = jax.tree_util.tree_map(lambda p: jnp.zeros_like(p, dtype=jnp.float64), values)
        for index in reversed(range(len(states))):
            cotangent = dict(cotangents[index], state=cotangents[index]["state"] + future)
            primal, gradient, future = self.compiled(values, states[index], fixed, days[index], cotangent)
            if primal_callback is not None:
                primal_callback(index, primal)
            accumulated = self.add(accumulated, gradient)
            if int(self.nonfinite((gradient, future))) != 0:
                raise RuntimeError(f"nonfinite streamed derivative at day {index + 1}")
            emit("reverse_day", {"day": index + 1})
        jax.block_until_ready((accumulated, future))
        return accumulated, future
