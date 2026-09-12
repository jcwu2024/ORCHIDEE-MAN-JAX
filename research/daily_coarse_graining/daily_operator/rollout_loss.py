"""Public multi-day loss and bounded streamed reverse differentiation.

The ordinary and checkpointed loss methods are pure JAX functions built from
``lax.scan``. ``streamed_value_and_grad`` is deliberately host-managed: it
retains daily carries, differentiates the loss with respect to the complete
observation trajectory, and reuses a bounded one-day VJP executable for full
backpropagation through time. It is not a ``custom_vjp`` and therefore does
not replace the ordinary loss's native AD or JVP behavior.
"""

from __future__ import annotations

from typing import Any, Callable

from .streamed_adjoint import StreamedAdjoint


class RolloutLoss:
    """Compose a daily transition and a trajectory loss.

    ``step`` has signature
    ``step(params, continuous_state, discrete_state, day)`` and returns
    ``(next_continuous_state, next_discrete_state, observations_dict)``. This
    component always sets ``observations["state"]`` to the returned next
    continuous state.

    ``initial_continuous`` is one floating array, not a PyTree. ``loss_fn`` has signature
    ``loss_fn(params, initial_continuous, observations, targets, loss_context)``
    and returns a scalar. Consequently, direct parameter and initial-state
    terms remain part of both ordinary and streamed gradients.

    The streamed method differentiates only ``params`` and
    ``initial_continuous``. ``initial_discrete``, ``days``, ``targets``, and
    ``loss_context`` are fixed inputs to that derivative. Their values remain
    dynamic method arguments; equal-shaped new values are not captured as
    factory constants.
    """

    def __init__(
        self,
        jax: Any,
        jnp: Any,
        step: Callable[..., tuple[Any, Any, dict[str, Any]]],
        loss_fn: Callable[..., Any],
        *,
        limit_bytes: int = 24 * 1024**3,
    ):
        self.jax = jax
        self.jnp = jnp
        self.step = step
        self.loss_fn = loss_fn

        def observed_step(params, continuous_state, _fixed, discrete_and_day):
            discrete_state, day = discrete_and_day
            next_continuous, _, observations = step(params, continuous_state, discrete_state, day)
            return self._with_state(observations, next_continuous)

        self._streamed_adjoint = StreamedAdjoint(jax, jnp, observed_step, limit_bytes=limit_bytes)
        self._streamed_forward = jax.jit(self._forward_with_discrete_history)
        self._loss_partials = jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 1, 2), allow_int=True))

    @property
    def reverse_compile_count(self) -> int:
        """Number of bounded one-day reverse executables compiled so far."""
        return self._streamed_adjoint.compile_count

    @property
    def reverse_memory(self) -> dict[str, int] | None:
        """Most recent one-day compiled-memory estimate, if available."""
        return self._streamed_adjoint.memory

    def _with_state(self, observations, next_continuous):
        if not isinstance(observations, dict):
            raise TypeError("daily step observations must be a dict")
        complete = dict(observations)
        complete["state"] = next_continuous
        return complete

    def _scan(self, params, initial_continuous, initial_discrete, days, *, checkpointed):
        def transition(carry, day):
            continuous_state, discrete_state = carry
            next_continuous, next_discrete, observations = self.step(
                params, continuous_state, discrete_state, day
            )
            complete = self._with_state(observations, next_continuous)
            return (next_continuous, next_discrete), (next_discrete, complete)

        if checkpointed:
            transition = self.jax.checkpoint(transition)
        final, (discrete_history, observations) = self.jax.lax.scan(
            transition, (initial_continuous, initial_discrete), days
        )
        return final, discrete_history, observations

    def _forward_with_discrete_history(self, params, initial_continuous, initial_discrete, days):
        return self._scan(params, initial_continuous, initial_discrete, days, checkpointed=False)

    def rollout(self, params, initial_continuous, initial_discrete, days, *, checkpointed=False):
        """Return ``((final_continuous, final_discrete), observations)``.

        ``checkpointed`` is a Python boolean selecting native rematerialization
        of the daily scan body. The returned observation dictionary includes
        the component-owned next-state trajectory under ``"state"``.
        """
        self._require_continuous_array(initial_continuous)
        final, _, observations = self._scan(
            params,
            initial_continuous,
            initial_discrete,
            days,
            checkpointed=checkpointed,
        )
        return final, observations

    def loss(self, params, initial_continuous, initial_discrete, days, targets, loss_context):
        """Evaluate the ordinary pure-JAX ``lax.scan`` reference loss."""
        _, observations = self.rollout(
            params, initial_continuous, initial_discrete, days, checkpointed=False
        )
        return self.loss_fn(params, initial_continuous, observations, targets, loss_context)

    def checkpointed_loss(self, params, initial_continuous, initial_discrete, days, targets, loss_context):
        """Evaluate the same pure-JAX loss with a rematerialized scan body."""
        _, observations = self.rollout(
            params, initial_continuous, initial_discrete, days, checkpointed=True
        )
        return self.loss_fn(params, initial_continuous, observations, targets, loss_context)

    def _leading_length(self, tree, name):
        leaves = self.jax.tree_util.tree_leaves(tree)
        if not leaves:
            raise ValueError(f"{name} must contain at least one array leaf")
        shapes = [getattr(leaf, "shape", ()) for leaf in leaves]
        if any(not shape for shape in shapes):
            raise ValueError(f"every {name} leaf must have a leading day axis")
        length = shapes[0][0]
        if any(shape[0] != length for shape in shapes[1:]):
            raise ValueError(f"all {name} leaves must have the same leading day length")
        return int(length)

    def _require_floating(self, tree, name):
        leaves = self.jax.tree_util.tree_leaves(tree)
        if not leaves:
            raise ValueError(f"{name} must contain at least one floating array leaf")
        for leaf in leaves:
            dtype = getattr(leaf, "dtype", None)
            if dtype is None or not self.jnp.issubdtype(dtype, self.jnp.floating):
                raise TypeError(f"every {name} leaf must have a floating dtype")

    def _require_continuous_array(self, value):
        dtype = getattr(value, "dtype", None)
        shape = getattr(value, "shape", None)
        if shape is None or dtype is None or not self.jnp.issubdtype(dtype, self.jnp.floating):
            raise TypeError("initial continuous state must be one floating array, not a PyTree")

    def _require_discrete(self, tree, name):
        for leaf in self.jax.tree_util.tree_leaves(tree):
            dtype = getattr(leaf, "dtype", None)
            if dtype is None or not (
                self.jnp.issubdtype(dtype, self.jnp.integer)
                or self.jnp.issubdtype(dtype, self.jnp.bool_)
            ):
                raise TypeError(f"every {name} leaf must have a bool or integer dtype")

    def _slice(self, tree, index):
        # float0 cotangents for scalar bool/int observations live in NumPy.
        # Integer indexing creates numpy.void, rejected by JAX 0.4 AOT calls;
        # preserve a zero-dimensional array and the original tangent dtype.
        return self.jax.tree_util.tree_map(
            lambda value: value[index:index + 1].reshape(value.shape[1:]), tree)

    def streamed_value_and_grad(
        self,
        params,
        initial_continuous,
        initial_discrete,
        days,
        targets,
        loss_context,
        *,
        callback=None,
        primal_callback=None,
    ):
        """Return loss and full gradients using bounded one-day reverse graphs.

        This method is host-managed and must not be placed inside ``jax.jit``.
        It requires one or more equal-shaped daily transitions. Parameter
        leaves and the continuous initial state must be floating. Discrete
        carry leaves must remain bool/integer because their derivative is held
        fixed rather than silently omitted.

        ``primal_callback(index, replay, forward)`` optionally observes the
        already-returned one-day VJP primal and its saved forward observation.
        It is diagnostic only and cannot replace state or cotangent values.
        """
        jax = self.jax
        horizon = self._leading_length(days, "days")
        if horizon < 1:
            raise ValueError("streamed differentiation requires at least one day")
        self._require_floating(params, "parameter")
        self._require_continuous_array(initial_continuous)
        self._require_discrete(initial_discrete, "initial discrete state")

        _, discrete_history, observations = self._streamed_forward(
            params, initial_continuous, initial_discrete, days
        )
        self._require_discrete(discrete_history, "propagated discrete state")
        loss, (direct_parameter, direct_initial, observation_cotangents) = self._loss_partials(
            params, initial_continuous, observations, targets, loss_context
        )

        state_history = observations["state"]
        continuous_starts = [initial_continuous] + [state_history[index] for index in range(horizon - 1)]
        discrete_starts = [initial_discrete] + [
            self._slice(discrete_history, index) for index in range(horizon - 1)
        ]
        day_records = [
            (discrete_starts[index], self._slice(days, index)) for index in range(horizon)
        ]
        cotangent_days = [self._slice(observation_cotangents, index) for index in range(horizon)]

        recurrent_parameter, recurrent_initial = self._streamed_adjoint.gradient_with_initial_state(
            params,
            (),
            continuous_starts,
            day_records,
            cotangent_days,
            callback=callback,
            primal_callback=(None if primal_callback is None else
                lambda index, replay: primal_callback(index, replay, self._slice(observations, index))),
        )
        parameter_gradient = self._streamed_adjoint.add(recurrent_parameter, direct_parameter)
        initial_gradient = jax.tree_util.tree_map(
            lambda recurrent, direct: recurrent + direct.astype(recurrent.dtype),
            recurrent_initial,
            direct_initial,
        )
        jax.block_until_ready((loss, parameter_gradient, initial_gradient))
        if int(self._streamed_adjoint.nonfinite((loss, parameter_gradient, initial_gradient))) != 0:
            raise RuntimeError("nonfinite streamed loss or derivative")
        return loss, (parameter_gradient, initial_gradient)


def make_rollout_loss(jax, jnp, step, loss_fn, *, limit_bytes=24 * 1024**3):
    """Construct a :class:`RolloutLoss` without capturing sample values."""
    return RolloutLoss(jax, jnp, step, loss_fn, limit_bytes=limit_bytes)
