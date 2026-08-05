"""Composition root for one true-daily Gate-E1 operator transition."""

from __future__ import annotations

from typing import Any, Callable, Protocol

import jax.numpy as jnp

from .conservative_update import apply_conservative_update
from .native_forcing import encode_native_forcing
from .retained_tail import RetainedTailAdapter, retained_tail_input
from .state_features import (
    assemble_local_pft_features,
    fraction_weighted_pft_context,
    summarize_state_groups,
)
from .types import (
    AssembledDailyOperatorInput,
    DailyOperatorInput,
    DailyOperatorParameters,
    DailyOperatorResult,
)


class ProcessHead(Protocol):
    """Injectable scientific output boundary shared by model and Oracle heads."""

    def __call__(
        self,
        parameters: DailyOperatorParameters,
        context: Any,
        inputs: AssembledDailyOperatorInput,
    ) -> Any: ...


class InputAssembler(Protocol):
    """Unique canonical-to-derived-view assembly boundary."""

    def assemble(self, value: DailyOperatorInput) -> AssembledDailyOperatorInput: ...


def assemble_operator_context(
    parameters: DailyOperatorParameters,
    inputs: AssembledDailyOperatorInput,
) -> Any:
    """Assemble ordered forcing, grouped state, PFT, and named static context."""

    forcing = encode_native_forcing(inputs.forcing, parameters.forcing)
    state = summarize_state_groups(inputs.state, inputs.state_defined)
    local_pft = assemble_local_pft_features(inputs.pft)
    pft = fraction_weighted_pft_context(local_pft, inputs.pft)
    static = jnp.asarray(inputs.static_conditions, dtype=jnp.float64).reshape(-1)
    annual = jnp.asarray(inputs.annual_conditions, dtype=jnp.float64).reshape(-1)
    return jnp.concatenate((forcing, state, pft, static, annual))


def daily_operator_transition(
    parameters: DailyOperatorParameters,
    inputs: DailyOperatorInput,
    *,
    input_assembler: InputAssembler,
    process_head: ProcessHead,
    retained_tail: RetainedTailAdapter,
) -> DailyOperatorResult:
    """Execute one constrained daily update followed by the retained exact tail."""

    assembled = input_assembler.assemble(inputs)
    context = assemble_operator_context(parameters, assembled)
    prediction = process_head(parameters, context, assembled)
    fast_update = apply_conservative_update(assembled, prediction)
    tail = retained_tail(retained_tail_input(assembled, prediction, fast_update))
    diagnostics = {
        **tail.diagnostics,
        "water_budget_residual": fast_update.water.budget_residual,
        "carbon_budget_residual": fast_update.carbon.budget_residual,
        "water_external_input": jnp.sum(fast_update.water.external_input),
        "water_external_output": jnp.sum(fast_update.water.external_output),
        "carbon_external_input": jnp.sum(fast_update.carbon.external_input),
        "carbon_external_output": jnp.sum(fast_update.carbon.external_output),
    }
    return DailyOperatorResult(
        prediction=prediction,
        fast_update=fast_update,
        next_state=tail.state,
        next_state_defined=tail.state_defined,
        next_discrete_state=tail.discrete_state,
        diagnostics=diagnostics,
    )


def bind_daily_operator(
    *,
    input_assembler: InputAssembler,
    process_head: ProcessHead,
    retained_tail: RetainedTailAdapter,
) -> Callable[[DailyOperatorParameters, DailyOperatorInput], DailyOperatorResult]:
    """Bind static scientific ownership while keeping all arrays explicit."""

    def transition(parameters: DailyOperatorParameters, inputs: DailyOperatorInput) -> DailyOperatorResult:
        return daily_operator_transition(
            parameters,
            inputs,
            input_assembler=input_assembler,
            process_head=process_head,
            retained_tail=retained_tail,
        )

    return transition
