"""Frozen zero-training Teacher-GPP causal bridge."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .conservative_update import apply_conservative_update
from .retained_tail import retained_tail_input
from .successor_training import DailyOperatorRolloutTrace, rollout_input_from_template
from .target_oriented_distribution_aware_training import (
    aggregate_target_oriented_acceptance,
)
from .target_oriented_source_shaped_coupling import contract_sha256
from .types import DailyOperatorResult

SCHEMA_VERSION = "target_oriented_distribution_aware_gpp_bridge_v1"
FROZEN_CONTRACT_SHA256 = "ce2eefb1360bb5b6f49f727ab470d1faef3952a7649d411539d81ba3b28c5647"
GPP_OUTPUT_KEY = "daily.fast_gpp"
EXPECTED_EVALUATION_UPDATES = (251, 358, 465, 572, 679, 786, 893, 1000)
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_distribution_aware_gpp_bridge_v1.json"
)


@dataclass(frozen=True)
class DistributionAwareGPPBridgeContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def evaluation_updates(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.payload["data"]["evaluation_updates"])

    @property
    def reported_keys(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.payload["metrics"]["reported_keys"])

    @property
    def downstream_keys(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.payload["metrics"]["causal_downstream_keys"])

    @property
    def absolute_required_keys(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.payload["metrics"]["absolute_required_keys"])


def _validate_contract(payload: Mapping[str, Any]) -> None:
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "frozen_before_zero_training_execution"
    ):
        raise ValueError("distribution-aware GPP-bridge schema drift")
    data = payload.get("data", {})
    if (
        data.get("release_id") != "pft14-daily-teacher-v6"
        or data.get("split") != "train/train"
        or tuple(data.get("evaluation_updates", ())) != EXPECTED_EVALUATION_UPDATES
        or bool(data.get("validation_values_allowed"))
        or bool(data.get("sealed_test_allowed"))
    ):
        raise ValueError("distribution-aware GPP-bridge data boundary drift")
    intervention = payload.get("intervention", {})
    if (
        intervention.get("learned_output_key") != GPP_OUTPUT_KEY
        or intervention.get("replacement_occurs_before_conservative_update") is not True
        or intervention.get("replacement_occurs_before_retained_exact_tail") is not True
        or intervention.get("model_maintenance_and_all_other_learned_outputs_remain_predicted") is not True
        or intervention.get("diagnostic_only") is not True
        or intervention.get("allowed_in_training") is not False
        or intervention.get("allowed_in_final_inference") is not False
    ):
        raise ValueError("distribution-aware GPP-bridge intervention drift")
    metrics = payload.get("metrics", {})
    if (
        tuple(metrics.get("causal_downstream_keys", ())) != ("growth_respiration", "npp")
        or tuple(metrics.get("required_horizons", ())) != (7, 30)
        or set(metrics.get("absolute_required_keys", ())) != set(metrics.get("reported_keys", ()))
    ):
        raise ValueError("distribution-aware GPP-bridge metric drift")
    execution = payload.get("execution", {})
    forbidden = (
        "optimizer_allowed",
        "checkpoint_write_allowed",
        "teacher_data_regeneration_allowed",
        "historical_v5_allowed",
        "architecture_edit_allowed",
        "validation_read_authorized",
        "sealed_test_authorized",
        "365_day_rollout_allowed",
        "gate_d2_allowed",
        "product_acceptance_allowed",
    )
    if (
        execution.get("free_gln01_allowed") is not True
        or execution.get("paid_execution_authorized") is not False
        or int(execution.get("training_updates", -1)) != 0
        or int(execution.get("parameter_steps", -1)) != 0
        or any(bool(execution.get(name)) for name in forbidden)
        or int(execution.get("forcing_records_reconstructed", -1)) != 0
        or int(execution.get("subdaily_state_scan_steps", -1)) != 0
    ):
        raise ValueError("distribution-aware GPP-bridge execution boundary drift")


def load_distribution_aware_gpp_bridge_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> DistributionAwareGPPBridgeContract:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    _validate_contract(payload)
    observed = contract_sha256(payload)
    if payload.get("contract_sha256") != observed or observed != FROZEN_CONTRACT_SHA256:
        raise ValueError("distribution-aware GPP-bridge contract is not frozen v1")
    return DistributionAwareGPPBridgeContract(payload, source, observed)


def replace_teacher_fast_gpp(
    learned_outputs: Mapping[str, Any],
    teacher_fast_gpp: Any,
) -> Mapping[str, Any]:
    """Return learned coordinates with exactly one Teacher-owned replacement."""

    if GPP_OUTPUT_KEY not in learned_outputs:
        raise ValueError(f"learned output boundary lacks {GPP_OUTPUT_KEY}")
    predicted = jnp.asarray(learned_outputs[GPP_OUTPUT_KEY])
    teacher = jnp.asarray(teacher_fast_gpp, dtype=predicted.dtype)
    if teacher.shape != predicted.shape:
        raise ValueError(
            f"Teacher GPP shape {teacher.shape} does not match predicted shape {predicted.shape}"
        )
    return dict(learned_outputs, **{GPP_OUTPUT_KEY: teacher})


def teacher_gpp_bridge_transition(
    parameters: Any,
    inputs: Any,
    teacher_fast_gpp: Any,
    *,
    input_assembler: Any,
    process_head: Any,
    retained_tail: Any,
) -> DailyOperatorResult:
    """Execute one ordinary daily transition after substituting only fast GPP."""

    assembled = input_assembler.assemble(inputs)
    context = process_head.assemble_context(parameters, assembled)
    learned = process_head.learned_outputs(parameters, context, assembled)
    bridged = replace_teacher_fast_gpp(learned, teacher_fast_gpp)
    prediction = process_head.prediction_from_learned_outputs(bridged, assembled)
    fast_update = apply_conservative_update(assembled, prediction)
    prediction = process_head.materialize_post_update(
        parameters,
        context,
        assembled,
        prediction,
        fast_update,
    )
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


def rollout_with_teacher_fast_gpp(
    parameters: Any,
    templates: Any,
    teacher_fast_gpp: Any,
    *,
    transition: Callable[[Any, Any, Any], DailyOperatorResult],
    contract: Any,
    rematerialize: bool = False,
) -> DailyOperatorRolloutTrace:
    """Run a free state recurrence while scanning matched Teacher GPP by day."""

    continuous = jnp.asarray(templates.continuous_state, dtype=jnp.float64)
    teacher = jnp.asarray(teacher_fast_gpp)
    if continuous.ndim != 2 or continuous.shape[1] != contract.continuous_state_width:
        raise ValueError("Teacher-GPP rollout templates require [day, continuous_state]")
    if teacher.shape[0] != continuous.shape[0]:
        raise ValueError("Teacher-GPP rollout day counts differ")
    initial_discrete = jax.tree_util.tree_map(
        lambda value: jnp.asarray(value)[0],
        templates.discrete_state,
    )

    def step(carry, daily_values):
        template, daily_gpp = daily_values
        current = rollout_input_from_template(template, carry[0], carry[1], contract)
        result = transition(parameters, current, daily_gpp)
        trace = DailyOperatorRolloutTrace(
            result=result,
            pft_fraction=current.pft.pft_fraction,
            active_pft_mask=current.pft.active_pft_mask,
        )
        return (result.next_state.other, result.next_discrete_state), trace

    scan_step = jax.checkpoint(step) if rematerialize else step
    _, trace = jax.lax.scan(
        scan_step,
        (continuous[0], initial_discrete),
        (templates, teacher),
    )
    return trace


def _acceptance_score(record: Mapping[str, Any]) -> float:
    rmse = float(record["normalized_rmse"])
    bias = float(record["normalized_signed_bias"])
    return 0.5 * (rmse * rmse + bias * bias)


def _group_lookup(
    groups: Sequence[Mapping[str, Any]],
    expected: Sequence[int],
) -> Mapping[int, Mapping[str, Any]]:
    result = {int(group["update"]): group for group in groups}
    if tuple(result) != tuple(expected):
        raise ValueError("distribution-aware GPP-bridge group identity or order drift")
    return result


def compare_gpp_bridge_modes(
    free_groups: Sequence[Mapping[str, Any]],
    intervention_groups: Sequence[Mapping[str, Any]],
    contract: DistributionAwareGPPBridgeContract,
) -> Mapping[str, Any]:
    """Compare one diagnostic mode with the accepted update-250 free rollout."""

    free = _group_lookup(free_groups, contract.evaluation_updates)
    intervention = _group_lookup(intervention_groups, contract.evaluation_updates)
    horizons = {}
    all_finite = True
    for horizon in ("7", "30"):
        keys = {}
        for name in contract.reported_keys:
            reductions = []
            for update in contract.evaluation_updates:
                before = _acceptance_score(free[update]["acceptance"][horizon]["keys"][name])
                after = _acceptance_score(
                    intervention[update]["acceptance"][horizon]["keys"][name]
                )
                reduction = 1.0 - after / before if before > 1.0e-15 else float("nan")
                reductions.append(reduction)
            all_finite = all_finite and all(np.isfinite(value) for value in reductions)
            keys[name] = {
                "median_score_reduction": float(np.median(reductions)),
                "group_score_reductions": reductions,
                "improved_group_count": int(sum(value > 0.0 for value in reductions)),
            }
        horizons[horizon] = {"keys": keys}
    return {
        "all_finite": bool(all_finite),
        "horizons": horizons,
        "aggregate_acceptance": aggregate_target_oriented_acceptance(
            tuple(intervention.values())
        ),
    }


def classify_gpp_bridge(
    bridge_comparison: Mapping[str, Any],
    *,
    hard_failure_count: int,
    contract: DistributionAwareGPPBridgeContract,
) -> Mapping[str, Any]:
    """Decide whether the next architecture discussion may be GPP-scoped."""

    thresholds = contract.payload["classification"]
    downstream_reductions = {
        horizon: {
            name: float(
                bridge_comparison["horizons"][horizon]["keys"][name][
                    "median_score_reduction"
                ]
            )
            for name in contract.downstream_keys
        }
        for horizon in ("7", "30")
    }
    aggregate = bridge_comparison["aggregate_acceptance"]
    absolute_cells = {
        horizon: {
            name: bool(aggregate[horizon]["keys"][name]["passed"])
            for name in contract.absolute_required_keys
        }
        for horizon in ("7", "30")
    }
    hard_passed = int(hard_failure_count) == 0
    finite = bool(bridge_comparison["all_finite"])
    absolute_passed = all(value for cells in absolute_cells.values() for value in cells.values())
    minimum_scope = float(
        thresholds["minimum_median_downstream_score_reduction_for_gpp_scope"]
    )
    downstream_scope_passed = all(
        value >= minimum_scope
        for values in downstream_reductions.values()
        for value in values.values()
    )
    sufficient = finite and hard_passed and absolute_passed and downstream_scope_passed
    minimum_contribution = float(
        thresholds["minimum_median_downstream_score_reduction_for_contribution"]
    )
    maximum_regression = float(
        thresholds["maximum_median_downstream_score_regression_for_contribution"]
    )
    downstream_values = tuple(
        value for values in downstream_reductions.values() for value in values.values()
    )
    contributes = bool(
        finite
        and hard_passed
        and any(value >= minimum_contribution for value in downstream_values)
        and all(value >= -maximum_regression for value in downstream_values)
    )
    classification = (
        "gpp_boundary_supported_as_next_scope"
        if sufficient
        else (
            "gpp_boundary_contributes_but_not_sufficient"
            if contributes
            else "shared_attention_or_state_scope_required"
        )
    )
    return {
        "classification": classification,
        "hard_gate_passed": hard_passed,
        "all_scores_finite": finite,
        "absolute_required_cells": absolute_cells,
        "absolute_required_gate_passed": absolute_passed,
        "downstream_median_score_reductions": downstream_reductions,
        "downstream_scope_gate_passed": downstream_scope_passed,
        "contribution_gate_passed": contributes,
        "next_action": "architecture_discussion_required_after_teacher_gpp_bridge",
    }


__all__ = [
    "DEFAULT_CONTRACT_PATH",
    "EXPECTED_EVALUATION_UPDATES",
    "FROZEN_CONTRACT_SHA256",
    "GPP_OUTPUT_KEY",
    "SCHEMA_VERSION",
    "DistributionAwareGPPBridgeContract",
    "classify_gpp_bridge",
    "compare_gpp_bridge_modes",
    "load_distribution_aware_gpp_bridge_contract",
    "replace_teacher_fast_gpp",
    "rollout_with_teacher_fast_gpp",
    "teacher_gpp_bridge_transition",
]
