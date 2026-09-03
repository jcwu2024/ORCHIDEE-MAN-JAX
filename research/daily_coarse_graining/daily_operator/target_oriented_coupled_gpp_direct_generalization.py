"""Frozen direct-GPP generalization attribution for persistent training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .target_oriented_coupled_gpp_objective import TRAINABLE_PARAMETER_COUNT
from .target_oriented_coupled_gpp_persistent_scale_training import (
    FROZEN_CONTRACT_SHA256 as PERSISTENT_CONTRACT_SHA256,
)
from .target_oriented_coupled_gpp_persistent_scale_training import (
    SCHEDULE_SHA256,
    build_persistent_scale_training_schedule,
)
from .target_oriented_source_shaped_maintenance import contract_sha256

SCHEMA_VERSION = "target_oriented_coupled_gpp_direct_generalization_audit_v1"
FROZEN_CONTRACT_SHA256 = "017b8a1de69260970cb78f500afc8984cb86afb2a57cfb53bdb111e1fd55ff5f"
PERSISTENT_RESULT_SHA256 = "764b14e169345ba119407b9e3dcf3a7e9141900c2710c8afa2775dd0fe881168"
AUDITED_PERSISTENT_UPDATES = tuple(range(1, 11))
SELECTED_SCALES = (1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0)
MONITOR_UPDATES = (251, 358, 465, 572, 679, 786, 893, 1000)
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_coupled_gpp_direct_generalization_audit_v1.json"
)


@dataclass(frozen=True)
class CoupledGPPDirectGeneralizationContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def monitor_updates(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.payload["data"]["monitor_artifact_updates"])

    @property
    def audited_updates(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.payload["trajectory"]["persistent_updates"])


def first_adjacent_regression(
    losses: Sequence[float],
    *,
    relative_tolerance: float = 0.0,
) -> int | None:
    """Return the one-based transition index of the first adjacent increase."""
    values = tuple(float(value) for value in losses)
    tolerance = float(relative_tolerance)
    if len(values) < 2 or tolerance < 0.0:
        raise ValueError("direct-GPP trajectory requires two losses and a nonnegative tolerance")
    if not all(np.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("direct-GPP trajectory losses must be finite and positive")
    for transition, (before, after) in enumerate(zip(values, values[1:]), start=1):
        if after > before * (1.0 + tolerance):
            return transition
    return None


def cosine_matrix_from_gram(gram: Any) -> np.ndarray:
    """Normalize a finite positive-diagonal Gram matrix to cosines."""
    values = np.asarray(gram, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1] or not np.all(np.isfinite(values)):
        raise ValueError("direct-GPP gradient Gram matrix is invalid")
    diagonal = np.diag(values)
    if np.any(diagonal <= 0.0):
        raise ValueError("direct-GPP gradient Gram diagonal must be positive")
    denominator = np.sqrt(np.outer(diagonal, diagonal))
    return np.clip(values / denominator, -1.0, 1.0)


def finite_loss_changes(before: Sequence[float], after: Sequence[float]) -> tuple[Mapping[str, Any], ...]:
    first = tuple(float(value) for value in before)
    second = tuple(float(value) for value in after)
    if len(first) != len(second) or not first:
        raise ValueError("direct-GPP finite changes require paired nonempty losses")
    records = []
    for old, new in zip(first, second, strict=True):
        if not np.isfinite(old) or not np.isfinite(new) or old <= 0.0 or new < 0.0:
            raise ValueError("direct-GPP finite changes contain an invalid loss")
        records.append(
            {
                "before": old,
                "after": new,
                "relative_change": (new - old) / old,
                "improved": new < old,
            }
        )
    return tuple(records)


def directional_attribution(
    *,
    training_monitor_mean_cosine: float,
    training_applied_descent_cosine: float,
    monitor_applied_descent_cosine: float,
    monitor_mean_relative_change: float,
    negative_within_training_pairs: int,
    negative_within_monitor_pairs: int,
    negative_cross_pairs: int,
) -> Mapping[str, Any]:
    """Classify sign-based evidence without imposing a statistical threshold."""
    scalars = (
        training_monitor_mean_cosine,
        training_applied_descent_cosine,
        monitor_applied_descent_cosine,
        monitor_mean_relative_change,
    )
    counts = (negative_within_training_pairs, negative_within_monitor_pairs, negative_cross_pairs)
    if not all(np.isfinite(float(value)) for value in scalars) or any(int(value) < 0 for value in counts):
        raise ValueError("direct-GPP directional attribution inputs are invalid")

    monitor_regressed = float(monitor_mean_relative_change) > 0.0
    if not monitor_regressed:
        primary = "no_finite_monitor_regression_at_audited_step"
    elif float(monitor_applied_descent_cosine) > 0.0:
        primary = "finite_step_curvature_or_nonlinearity"
    elif float(training_monitor_mean_cosine) <= 0.0:
        primary = "scheduled_batch_monitor_direction_mismatch"
    else:
        primary = "multi_source_direction_rotated_away_from_monitor_direct_gpp"

    return {
        "primary": primary,
        "scheduled_batch_monitor_mean_direction_conflict": float(training_monitor_mean_cosine) <= 0.0,
        "training_direct_objective_supported_applied_step": float(training_applied_descent_cosine) > 0.0,
        "monitor_direct_objective_supported_applied_step": float(monitor_applied_descent_cosine) > 0.0,
        "finite_monitor_regression": monitor_regressed,
        "artifact_direction_conflict_observed": any(int(value) > 0 for value in counts),
        "sampling_variance_separable_from_population_heterogeneity": False,
    }


def load_coupled_gpp_direct_generalization_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> CoupledGPPDirectGeneralizationContract:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    observed = contract_sha256(payload)
    identity = payload.get("identity", {})
    data = payload.get("data", {})
    trajectory = payload.get("trajectory", {})
    attribution = payload.get("attribution", {})
    acceptance = payload.get("acceptance", {})
    retention = payload.get("retention", {})
    execution = payload.get("execution", {})
    schedule = build_persistent_scale_training_schedule()
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "frozen_before_direct_gpp_generalization_values_are_evaluated"
        or payload.get("contract_sha256") != observed
        or observed != FROZEN_CONTRACT_SHA256
        or identity.get("persistent_contract_sha256") != PERSISTENT_CONTRACT_SHA256
        or identity.get("persistent_result_sha256") != PERSISTENT_RESULT_SHA256
        or identity.get("persistent_schedule_sha256") != SCHEDULE_SHA256
        or schedule.sha256 != SCHEDULE_SHA256
        or int(identity.get("candidate_parameter_count", -1)) != 109_464_359
        or int(identity.get("trainable_parameter_count", -1)) != TRAINABLE_PARAMETER_COUNT
        or int(identity.get("fixed_parameter_count", -1)) != 93_504_164
        or tuple(int(value) for value in data.get("monitor_artifact_updates", ())) != MONITOR_UPDATES
        or data.get("training_split") != "train/train"
        or data.get("monitor_split") != "train/train_non_test_monitor"
        or data.get("monitor_used_by_persistent_gradients_or_scale_selection") is not False
        or tuple(int(value) for value in trajectory.get("persistent_updates", ())) != AUDITED_PERSISTENT_UPDATES
        or tuple(float(value) for value in trajectory.get("selected_scales_from_result", ())) != SELECTED_SCALES
        or trajectory.get("start_state") != "reconstructed_selected_state_after_update_7"
        or trajectory.get("persistent_recovery_read") is not False
        or trajectory.get("recompute_four_source_direction_per_update") is not True
        or trajectory.get("repeat_scale_search") is not False
        or trajectory.get("evaluate_fixed_monitor_direct_gpp_after_every_update") is not True
        or float(trajectory.get("adjacent_regression_relative_tolerance", -1.0)) != 0.0
        or float(trajectory.get("historical_independent_initial_monitor_relative_spread", -1.0)) != 0.00125583158782248
        or float(trajectory.get("bootstrap_persistent_result_relative_difference_maximum", -1.0)) != 0.01
        or float(trajectory.get("same_replay_internal_loss_match_rtol", -1.0)) != 1.0e-6
        or float(trajectory.get("same_replay_internal_loss_match_atol", -1.0)) != 1.0e-9
        or trajectory.get("per_update_and_endpoint_persistent_result_loss_comparison") != "report_only"
        or int(attribution.get("training_artifact_gradient_count", -1)) != 8
        or int(attribution.get("monitor_artifact_gradient_count", -1)) != 8
        or attribution.get("gradient_objective") != "direct_gpp_208_teacher_days"
        or attribution.get("gradient_state") != "immediately_before_first_adjacent_monitor_regression"
        or attribution.get("applied_step_direction_included") is not True
        or attribution.get("pairwise_cosine_matrix_required") is not True
        or attribution.get("finite_step_per_artifact_changes_required") is not True
        or attribution.get("sign_based_classification_only") is not True
        or acceptance.get("all_ten_updates_replayed") is not True
        or acceptance.get("first_adjacent_regression_located") is not True
        or acceptance.get("all_sixteen_direct_gradients_finite_nonzero") is not True
        or acceptance.get("trajectory_direct_gpp_outcome_direction_matches_persistent_result") is not True
        or acceptance.get("scientific_model_promotion_implied") is not False
    ):
        raise ValueError("coupled GPP direct-generalization contract drift")
    if any(
        bool(data[name])
        for name in (
            "fresh_validation_values_allowed",
            "teacher_data_regeneration_allowed",
            "historical_v5_allowed",
            "sealed_test_allowed",
        )
    ) or any(bool(value) for value in retention.values()):
        raise ValueError("coupled GPP direct-generalization audit opens a forbidden data or retention path")
    if (
        execution.get("free_gln01_allowed") is not True
        or execution.get("paid_partition_authorized") is not False
        or int(execution.get("timeout_minutes", -1)) != 180
        or int(execution.get("forcing_records_reconstructed", -1)) != 0
        or int(execution.get("subdaily_state_scan_steps", -1)) != 0
        or any(
            bool(execution[name])
            for name in (
                "fresh_validation_read_authorized",
                "365_day_rollout_authorized",
                "gate_d2_authorized",
                "sealed_test_authorized",
                "product_acceptance_authorized",
            )
        )
    ):
        raise ValueError("coupled GPP direct-generalization execution boundary drift")
    return CoupledGPPDirectGeneralizationContract(payload, source, observed)


__all__ = [
    "AUDITED_PERSISTENT_UPDATES",
    "DEFAULT_CONTRACT_PATH",
    "FROZEN_CONTRACT_SHA256",
    "MONITOR_UPDATES",
    "PERSISTENT_RESULT_SHA256",
    "SCHEMA_VERSION",
    "SELECTED_SCALES",
    "CoupledGPPDirectGeneralizationContract",
    "cosine_matrix_from_gram",
    "directional_attribution",
    "finite_loss_changes",
    "first_adjacent_regression",
    "load_coupled_gpp_direct_generalization_contract",
]
