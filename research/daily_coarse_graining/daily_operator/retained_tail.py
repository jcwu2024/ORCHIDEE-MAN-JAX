"""Clean callable boundary around source-backed retained exact daily owners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from .types import (
    AssembledDailyOperatorInput,
    ProcessPrediction,
    RetainedTailInput,
    RetainedTailResult,
)


@dataclass(frozen=True)
class ExactRetainedTailAdapter:
    """Bind a pure retained-tail transition without exposing historical targets.

    ``transition`` must execute the accepted source-backed daily owners. The
    operator sees only the constrained fast update and compact named inputs;
    it cannot pass an anonymous fast-day target or a Teacher endpoint.
    """

    transition: Callable[[RetainedTailInput], RetainedTailResult]
    source_owner: str

    def __post_init__(self) -> None:
        if not self.source_owner.strip():
            raise ValueError("retained-tail adapters require a source owner")

    def __call__(self, value: RetainedTailInput) -> RetainedTailResult:
        return self.transition(value)


class RetainedTailAdapter(Protocol):
    """Typed callable implemented by plumbing and canonical exact adapters."""

    source_owner: str

    def __call__(self, value: RetainedTailInput) -> RetainedTailResult: ...


def retained_tail_input(
    inputs: AssembledDailyOperatorInput,
    prediction: ProcessPrediction,
    fast_update,
) -> RetainedTailInput:
    """Build the only product-facing handoff into retained exact processes."""

    return RetainedTailInput(
        fast_update=fast_update,
        process_prediction=prediction,
        canonical_input=inputs.canonical_input,
        pft_parameters=inputs.pft.pft_parameters,
        pft_traits=inputs.pft.pft_traits,
        static_conditions=inputs.static_conditions,
        annual_conditions=inputs.annual_conditions,
        year=inputs.year,
        day_index=inputs.day_index,
    )


def identity_retained_tail_for_plumbing_tests(value: RetainedTailInput) -> RetainedTailResult:
    """Named test-only identity tail; never a scientific retained-tail claim."""

    update = value.fast_update
    return RetainedTailResult(
        state=update.state,
        state_defined=update.state_defined,
        discrete_state=update.discrete_state,
        diagnostics={
            "retained_tail_executed": value.day_index * 0 + 1,
        },
    )


def plumbing_test_retained_tail_adapter() -> ExactRetainedTailAdapter:
    """Return the explicitly named identity adapter used by local E1 tests."""

    return ExactRetainedTailAdapter(
        transition=identity_retained_tail_for_plumbing_tests,
        source_owner="Gate E1 plumbing identity; not valid for training or scientific rollout",
    )
