"""Strict loader for the bounded causal-carbon adapter experiment."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from research.daily_coarse_graining.causal_carbon_adapter_daily_model import (
    CAUSAL_CARBON_ADAPTER_V1,
)
from research.daily_coarse_graining.causal_carbon_objective import (
    FAST_INTERFACE_FIELDS,
    FLUX_BIAS_FIELDS,
    NO_REGRESSION_GUARD_FIELDS,
    PRIMARY_NEXT_STATE_FIELDS,
    STOCK_TENDENCY_FIELDS,
)
from research.daily_coarse_graining.daily_model_architecture import (
    AXIS_PROCESS_COUPLED_V1,
)

SCHEMA_VERSION = "causal_carbon_adapter_experiment_v1"
ARM_IDS = (
    "causal_interface_one_step_control",
    "causal_interface_rollout_candidate",
)
COMPONENT_IDS = (
    "L_interface",
    "L_next",
    "L_rollout",
    "L_flux_bias",
    "L_stock_tendency_bias",
)
EXPECTED_HORIZONS = (1, 3, 7)
EXPECTED_SELECTION_SLICES = ("temporal", "spatial", "joint")


@dataclass(frozen=True)
class CausalCarbonAdapterProtocol:
    path: Path
    raw: Mapping[str, Any]
    sha256: str


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    context: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{context} key inventory drift")


def _validate_parent_architecture(raw: Mapping[str, Any]) -> None:
    predecessor = raw["rejected_predecessor"]
    _require_keys(
        predecessor,
        {
            "experiment_id",
            "candidate_arm",
            "screening_report_sha256",
            "decision",
            "reuse_candidate_checkpoint",
        },
        context="rejected predecessor",
    )
    if predecessor["decision"] != "rejected":
        raise ValueError("Experiment C requires a rejected predecessor")
    if predecessor["reuse_candidate_checkpoint"] is not False:
        raise ValueError("the rejected candidate checkpoint must not be reused")

    parent = raw["parent"]
    _require_keys(
        parent,
        {
            "arm_id",
            "model_architecture",
            "checkpoint_sha256",
            "training_report_sha256",
            "parameters_frozen",
            "optimizer_state_reused",
        },
        context="causal adapter parent",
    )
    if parent["arm_id"] != "one_step_continuation_control":
        raise ValueError("causal adapter must use the accepted one-step control")
    if parent["model_architecture"] != AXIS_PROCESS_COUPLED_V1:
        raise ValueError("causal adapter parent architecture drift")
    if parent["parameters_frozen"] is not True:
        raise ValueError("causal adapter parent must remain frozen")
    if parent["optimizer_state_reused"] is not False:
        raise ValueError("parent optimizer state must not be reused")

    architecture = raw["architecture"]
    _require_keys(
        architecture,
        {
            "id",
            "pft_index",
            "adapter_hidden_width",
            "adapter_layout_sha256",
            "objective_layout_sha256",
            "output_initialization",
            "base_prediction_at_initialization",
            "protected_fast_day_columns",
            "adapted_fast_day_columns",
            "hidden_cross_day_memory",
        },
        context="causal adapter architecture",
    )
    if architecture["id"] != CAUSAL_CARBON_ADAPTER_V1:
        raise ValueError("causal adapter architecture ID drift")
    if int(architecture["pft_index"]) != 13:
        raise ValueError("causal adapter must remain restricted to PFT14")
    if int(architecture["adapter_hidden_width"]) < 1:
        raise ValueError("causal adapter hidden width must be positive")
    if int(architecture["protected_fast_day_columns"]) + int(architecture["adapted_fast_day_columns"]) != 2855:
        raise ValueError("causal adapter target-column inventory drift")
    if architecture["output_initialization"] != "exact_zero":
        raise ValueError("causal adapter must begin at the parent prediction")
    if architecture["base_prediction_at_initialization"] != "bit_exact":
        raise ValueError("causal adapter parent prediction must be bit-exact")
    if architecture["hidden_cross_day_memory"] is not False:
        raise ValueError("causal adapter cannot add hidden cross-day memory")


def _validate_data_comparison(raw: Mapping[str, Any]) -> None:
    data = raw["data_policy"]
    if data["optimization_and_coefficient_calibration"] != {
        "spatial_split": "train",
        "temporal_split": "train",
    }:
        raise ValueError("adapter optimization must use train/train data only")
    if tuple(data["model_selection_slices"]) != EXPECTED_SELECTION_SLICES:
        raise ValueError("adapter model-selection slice inventory drift")
    for key in (
        "sealed_test_used",
        "landpoint_identity_as_feature",
        "new_teacher_data_required",
        "cross_year_training_windows",
    ):
        if data[key] is not False:
            raise ValueError(f"adapter data policy requires {key}=false")

    comparison = raw["comparison"]
    observed_arms = tuple(item["id"] for item in comparison["arms"])
    if observed_arms != ARM_IDS:
        raise ValueError("causal adapter arm inventory or order drift")
    expected_objectives = (
        "field_balanced_causal_interface_v1",
        "causal_interface_next_rollout_signed_bias_v1",
    )
    if tuple(item["objective"] for item in comparison["arms"]) != expected_objectives:
        raise ValueError("causal adapter arm objective drift")
    for key in (
        "same_frozen_parent",
        "same_zero_initialized_adapter",
        "same_anchor_batches",
        "same_optimizer_updates",
        "same_seed",
    ):
        if comparison[key] is not True:
            raise ValueError(f"matched adapter experiment requires {key}")
    if comparison["candidate_only_treatment"] != ("causal_next_state_rollout_and_signed_bias"):
        raise ValueError("causal adapter treatment drift")


def _validate_objective(raw: Mapping[str, Any]) -> None:
    objective = raw["objective"]
    expected_fast = [f"{family}.{field}" for family, field in FAST_INTERFACE_FIELDS]
    if objective["fast_interface_fields_equal_weight"] != expected_fast:
        raise ValueError("causal fast-interface field inventory drift")
    if objective["primary_next_state_fields_equal_weight"] != list(PRIMARY_NEXT_STATE_FIELDS):
        raise ValueError("causal primary next-state field inventory drift")
    if objective["flux_bias_fields"] != list(FLUX_BIAS_FIELDS):
        raise ValueError("causal flux-bias field inventory drift")
    if objective["stock_tendency_bias_fields"] != list(STOCK_TENDENCY_FIELDS):
        raise ValueError("causal stock-tendency field inventory drift")
    if objective["no_regression_guard_fields"] != list(NO_REGRESSION_GUARD_FIELDS):
        raise ValueError("causal no-regression guard inventory drift")
    if objective["day_end_gpp_state_supervision"] is not False:
        raise ValueError("reset day-end GPP state must not be a supervision target")

    components = objective["loss_components"]
    if tuple(item["id"] for item in components) != COMPONENT_IDS:
        raise ValueError("causal loss-component inventory or order drift")
    if components[0] != {
        "id": "L_interface",
        "present_in_both_arms": True,
        "coefficient": 1.0,
    }:
        raise ValueError("causal interface anchor definition drift")
    for component in components[1:]:
        if component != {
            "id": component["id"],
            "present_in_both_arms": False,
            "coefficient_source": "train_only_gradient_calibration",
        }:
            raise ValueError(f"causal treatment definition drift for {component['id']}")

    calibration = objective["coefficient_calibration"]
    if calibration["method"] != "train_only_median_gradient_norm_ratio_v1":
        raise ValueError("causal coefficient calibration method drift")
    if int(calibration["batches_per_horizon"]) < 1:
        raise ValueError("causal coefficient calibration budget must be positive")
    if set(calibration["target_gradient_norm_ratios"]) != set(COMPONENT_IDS[1:]):
        raise ValueError("causal coefficient calibration component drift")
    if any(float(value) <= 0 for value in calibration["target_gradient_norm_ratios"].values()):
        raise ValueError("causal target gradient ratios must be positive")
    if calibration["coefficient_clip"] != [0.001, 1000.0]:
        raise ValueError("causal coefficient clip drift")
    for key in ("validation_data_forbidden", "frozen_before_matched_training"):
        if calibration[key] is not True:
            raise ValueError(f"causal coefficient calibration requires {key}")


def _validate_execution_and_gates(raw: Mapping[str, Any]) -> None:
    optimization = raw["optimization"]
    if optimization["optimizer"] != "adam":
        raise ValueError("causal adapter optimizer drift")
    for key in (
        "learning_rate",
        "screening_updates",
        "anchor_batch_size",
        "checkpoint_every_updates",
        "progress_every_updates",
    ):
        if float(optimization[key]) <= 0:
            raise ValueError(f"causal adapter {key} must be positive")

    horizons = raw["mixed_horizon_sampling"]
    if tuple(int(item["days"]) for item in horizons) != EXPECTED_HORIZONS:
        raise ValueError("causal adapter horizon inventory or order drift")
    if not math.isclose(
        math.fsum(float(item["probability"]) for item in horizons),
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("causal adapter horizon probabilities must sum to one")
    for item in horizons:
        if int(item["batch_size"]) < 1:
            raise ValueError("causal adapter rollout batch size must be positive")

    feasibility = raw["train_only_feasibility_gate"]
    if tuple(feasibility["required_horizon_coverage"]) != EXPECTED_HORIZONS:
        raise ValueError("causal feasibility horizon coverage drift")
    for key in (
        "real_shards_required",
        "finite_loss_and_gradients",
        "every_update_applied",
        "frozen_parent_bit_exact",
        "protected_fast_day_columns_bit_exact",
    ):
        if feasibility[key] is not True:
            raise ValueError(f"causal feasibility gate requires {key}")
    if feasibility["sealed_test_used"] is not False:
        raise ValueError("causal feasibility gate cannot use sealed test data")
    for key in (
        "unexpected_defined_status_mismatches",
        "discrete_state_mismatches",
        "nonfinite_defined_values",
        "negative_source_nonnegative_carbon_stocks",
    ):
        if int(feasibility[key]) != 0:
            raise ValueError(f"causal feasibility hard constraint {key} must be zero")

    screening = raw["screening_validation"]
    if tuple(screening["required_slices"]) != EXPECTED_SELECTION_SLICES:
        raise ValueError("causal screening slice inventory drift")
    if tuple(screening["horizons_days"]) != (1, 7, 30):
        raise ValueError("causal screening horizon inventory drift")
    if screening["all_model_selection_samples"] is not True:
        raise ValueError("causal screening must cover all model-selection samples")
    if screening["all_hard_constraints_required"] is not True:
        raise ValueError("causal screening requires all hard constraints")
    if any(float(value) <= 0 for value in screening["gates_relative_to_control"].values()):
        raise ValueError("causal screening ratios must be positive")

    stop = raw["stop_rules"]
    for key in (
        "validation_weight_search",
        "inspect_sealed_test_after_failure",
        "train_through_365_days_or_50_years",
        "fallback_to_rejected_predecessor",
    ):
        if stop[key] is not False:
            raise ValueError(f"causal stop rule requires {key}=false")


def load_causal_carbon_adapter_protocol(
    path: str | Path,
) -> CausalCarbonAdapterProtocol:
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "experiment_id",
        "status",
        "scientific_hypothesis",
        "rejected_predecessor",
        "parent",
        "artifacts",
        "architecture",
        "data_policy",
        "comparison",
        "objective",
        "optimization",
        "mixed_horizon_sampling",
        "train_only_feasibility_gate",
        "screening_validation",
        "promotion_order",
        "stop_rules",
    }
    _require_keys(raw, required, context="causal adapter protocol")
    if raw["schema_version"] != SCHEMA_VERSION:
        raise ValueError("unsupported causal adapter protocol schema")
    if raw["status"] != "frozen_before_feasibility_and_validation":
        raise ValueError("causal adapter protocol is not frozen")
    _validate_parent_architecture(raw)
    _validate_data_comparison(raw)
    _validate_objective(raw)
    _validate_execution_and_gates(raw)
    return CausalCarbonAdapterProtocol(
        path=path,
        raw=raw,
        sha256=_canonical_sha256(raw),
    )
