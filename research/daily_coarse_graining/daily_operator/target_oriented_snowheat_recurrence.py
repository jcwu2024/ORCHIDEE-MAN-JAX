"""Frozen contract and diagnostic snowheat-only recurrence intervention."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import jax
import jax.numpy as jnp

from research.daily_coarse_graining.daily_markov_contract import (
    DailyMarkovContract,
    StateLeafSpec,
)

from .successor_training import DailyOperatorRolloutTrace, rollout_input_from_template
from .target_oriented_flux_basis import contract_sha256
from .types import DailyOperatorInput, DailyOperatorResult

SCHEMA_VERSION = "target_oriented_snowheat_recurrence_intervention_v1"
FROZEN_CONTRACT_SHA256 = "006e9d3d8ff63253ea2e70c995735856e9c3e93285934934fbbe3f7b16cab122"
SNOWHEAT_STATE_KEY = "hydrol_previous_step_state.snowheat"
SNOWHEAT_TENDENCY_FIELD = "thermal.hydrol_previous_step_state.snowheat.normalized_tendency"
WINDOW_NAMES = (
    "seven_day_extreme",
    "seven_day_matched_control",
    "thirty_day_extreme",
    "thirty_day_matched_control",
)
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_snowheat_recurrence_intervention_v1.json"
)


@dataclass(frozen=True)
class TargetOrientedSnowheatRecurrenceContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def windows(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.payload["data"]["windows"])


def _validate(payload: Mapping[str, Any]) -> None:
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "frozen_before_zero_training_execution"
    ):
        raise ValueError("snowheat recurrence intervention schema drift")
    identity = payload.get("identity", {})
    if (
        identity.get("gradient_path_report_sha256")
        != "a2e9a88575563f832bf705827c869dcbe920d15abd656f06b0eb08068ffa870d"
        or identity.get("gradient_path_contract_sha256")
        != "d683b87ae2658c4aeb5658cf42ab0b55f6e9cef94cd18af5419a376e3a821900"
        or identity.get("update_1000_checkpoint_sha256")
        != "a566f9d95726984dc03cbf1f1a183dd0d1eae9e3479ce5055c64e571ef6a16c7"
        or identity.get("prepared_manifest_sha256")
        != "8205140ddeae34967a287d41d204318659a84a8574de954e9b0be14f8998fc31"
        or int(identity.get("parameter_count", 0)) != 83_755_173
    ):
        raise ValueError("snowheat recurrence intervention identity drift")
    data = payload.get("data", {})
    windows = tuple(data.get("windows", ()))
    if (
        data.get("release_id") != "pft14-daily-teacher-v6"
        or data.get("split") != "train/train"
        or tuple(record.get("name") for record in windows) != WINDOW_NAMES
        or tuple(int(record.get("horizon", 0)) for record in windows) != (7, 7, 30, 30)
        or any(
            bool(data.get(name))
            for name in (
                "teacher_regeneration_allowed",
                "historical_v5_allowed",
                "validation_artifact_values_allowed",
                "sealed_test_allowed",
            )
        )
    ):
        raise ValueError("snowheat recurrence intervention data boundary drift")
    intervention = payload.get("intervention", {})
    if (
        intervention.get("state_key") != SNOWHEAT_STATE_KEY
        or intervention.get("learned_tendency_field") != SNOWHEAT_TENDENCY_FIELD
        or intervention.get("diagnostic_only") is not True
        or intervention.get("allowed_in_training") is not False
        or intervention.get("allowed_in_final_inference") is not False
        or any(
            intervention.get(name) is not True
            for name in (
                "first_day_is_identity",
                "other_continuous_state_coordinates_remain_predicted",
                "discrete_state_remains_predicted",
                "forcing_unchanged",
                "parameters_unchanged",
                "operator_unchanged",
                "retained_exact_tail_unchanged",
            )
        )
    ):
        raise ValueError("snowheat recurrence intervention boundary drift")
    measurement = payload.get("measurement", {})
    if (
        measurement.get("field") != "maintenance_respiration"
        or tuple(measurement.get("modes", ())) != ("free_reference", "snowheat_teacher_carry")
        or int(measurement.get("gradient_count", 0)) != 8
        or int(measurement.get("gradient_coverage", 0)) != 83_755_173
    ):
        raise ValueError("snowheat recurrence measurement drift")
    execution = payload.get("execution", {})
    if (
        execution.get("free_gln01_only") is not True
        or execution.get("paid_training_authorized") is not False
        or int(execution.get("optimizer_updates", -1)) != 0
        or int(execution.get("parameter_steps", -1)) != 0
        or any(
            bool(execution.get(name))
            for name in (
                "proposed_parameters_retained",
                "optimizer_state_retained",
                "checkpoint_written",
                "recovery_written",
                "teacher_data_regenerated",
                "historical_v5_used",
                "validation_values_evaluated",
                "sealed_test_used",
                "architecture_edit_allowed",
                "365_day_rollout_allowed",
                "gate_d2_allowed",
            )
        )
    ):
        raise ValueError("snowheat recurrence execution boundary drift")


def load_target_oriented_snowheat_recurrence_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> TargetOrientedSnowheatRecurrenceContract:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    _validate(payload)
    observed = contract_sha256(payload)
    if payload.get("contract_sha256") != observed or observed != FROZEN_CONTRACT_SHA256:
        raise ValueError("snowheat recurrence intervention contract is not frozen v1")
    return TargetOrientedSnowheatRecurrenceContract(payload, source, observed)


def resolve_continuous_state_leaf(
    contract: DailyMarkovContract,
    key: str,
) -> StateLeafSpec:
    matches = tuple(leaf for leaf in contract.state_leaves if leaf.key == key)
    if len(matches) != 1:
        raise ValueError(f"state intervention owner drift for {key}")
    leaf = matches[0]
    if leaf.discrete or leaf.start is None or leaf.stop is None or leaf.stop <= leaf.start:
        raise ValueError(f"state intervention owner is not a continuous slice for {key}")
    return leaf


def replace_state_leaf_with_teacher(
    predicted_state: Any,
    teacher_state: Any,
    leaf: StateLeafSpec,
) -> Any:
    predicted = jnp.asarray(predicted_state, dtype=jnp.float64)
    teacher = jnp.asarray(teacher_state, dtype=jnp.float64)
    if predicted.ndim != 1 or teacher.shape != predicted.shape:
        raise ValueError("state intervention vectors must have the same one-dimensional shape")
    if leaf.start is None or leaf.stop is None or not (0 <= leaf.start < leaf.stop <= predicted.size):
        raise ValueError("state intervention slice is outside the continuous state")
    return predicted.at[leaf.start : leaf.stop].set(teacher[leaf.start : leaf.stop])


def rollout_with_teacher_state_leaf(
    parameters: Any,
    templates: DailyOperatorInput,
    *,
    transition: Callable[[Any, DailyOperatorInput], DailyOperatorResult],
    contract: DailyMarkovContract,
    state_key: str = SNOWHEAT_STATE_KEY,
    rematerialize: bool = False,
) -> DailyOperatorRolloutTrace:
    """Run a diagnostic free rollout with one Teacher state slice intervened."""

    leaf = resolve_continuous_state_leaf(contract, state_key)
    continuous = jnp.asarray(templates.continuous_state, dtype=jnp.float64)
    if continuous.ndim != 2 or continuous.shape[1] != contract.continuous_state_width:
        raise ValueError("state intervention templates require [day, continuous_state]")
    if continuous.shape[0] < 1:
        raise ValueError("state intervention rollout window cannot be empty")
    initial_discrete = jax.tree_util.tree_map(
        lambda value: jnp.asarray(value)[0],
        templates.discrete_state,
    )

    def step(carry, template):
        intervened = replace_state_leaf_with_teacher(
            carry[0],
            template.continuous_state,
            leaf,
        )
        current = rollout_input_from_template(template, intervened, carry[1], contract)
        result = transition(parameters, current)
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
        templates,
    )
    return trace


def classify_snowheat_recurrence(
    records: Mapping[str, Mapping[str, Any]],
    contract: TargetOrientedSnowheatRecurrenceContract,
) -> Mapping[str, Any]:
    if set(records) != set(WINDOW_NAMES):
        raise ValueError("snowheat recurrence result window registry drift")
    policy = contract.payload["classification"]
    causal_ratio = float(policy["extreme_gradient_reduction_ratio_for_causal_confirmation"])
    head_ratio = float(policy["snowheat_head_reduction_ratio_for_causal_confirmation"])
    no_effect_ratio = float(policy["gradient_reduction_ratio_for_no_effect"])
    extreme = {}
    confirmed = []
    no_effect = []
    for name in ("seven_day_extreme", "thirty_day_extreme"):
        record = records[name]
        gradient_reduction = float(record["gradient_reduction_ratio"])
        snowheat_reduction = float(record["snowheat_head_reduction_ratio"])
        is_confirmed = gradient_reduction >= causal_ratio and snowheat_reduction >= head_ratio
        is_no_effect = gradient_reduction < no_effect_ratio
        confirmed.append(is_confirmed)
        no_effect.append(is_no_effect)
        extreme[name] = {
            "gradient_reduction_ratio": gradient_reduction,
            "snowheat_head_reduction_ratio": snowheat_reduction,
            "causal_threshold_passed": is_confirmed,
            "no_effect_threshold_met": is_no_effect,
        }
    if all(confirmed):
        classification = "snowheat_recurrence_causal"
    elif all(no_effect):
        classification = "snowheat_recurrence_not_causal"
    else:
        classification = "snowheat_recurrence_partial"
    return {
        "extreme_windows": extreme,
        "classification": classification,
        "action": str(contract.payload["decision"][classification]),
    }


__all__ = [
    "DEFAULT_CONTRACT_PATH",
    "FROZEN_CONTRACT_SHA256",
    "SCHEMA_VERSION",
    "SNOWHEAT_STATE_KEY",
    "SNOWHEAT_TENDENCY_FIELD",
    "TargetOrientedSnowheatRecurrenceContract",
    "WINDOW_NAMES",
    "classify_snowheat_recurrence",
    "load_target_oriented_snowheat_recurrence_contract",
    "replace_state_leaf_with_teacher",
    "resolve_continuous_state_leaf",
    "rollout_with_teacher_state_leaf",
]
