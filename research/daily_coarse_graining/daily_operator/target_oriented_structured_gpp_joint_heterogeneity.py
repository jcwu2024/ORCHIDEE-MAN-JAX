"""Grouped gradient geometry for the structured-GPP joint objective."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .target_oriented_source_shaped_maintenance import contract_sha256
from .target_oriented_structured_gpp_joint_calibration import (
    DIRECT_SOURCE,
    GRADIENT_SOURCES,
    SEVEN_SOURCE,
    THIRTY_SOURCE,
)
from .target_oriented_structured_gpp_training import SPECIALIST_PARAMETER_COUNT

SCHEMA_VERSION = "target_oriented_structured_gpp_joint_gradient_heterogeneity_audit_v1"
FROZEN_CONTRACT_SHA256 = "0905b11467e982cdff72c624f2c61c2849e4e2dd90cbee860b9996c9588d945b"
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_structured_gpp_joint_gradient_heterogeneity_audit_v1.json"
)
ARTIFACT_UPDATES = (512, 548, 584, 620, 656, 692, 728, 763)
EFFECTIVE_BATCH_SIZES = (1, 2, 4, 8)


@dataclass(frozen=True)
class StructuredGPPJointHeterogeneityContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def artifact_updates(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.payload["data"]["artifact_updates"])

    @property
    def effective_batch_sizes(self) -> tuple[int, ...]:
        return tuple(
            int(value) for value in self.payload["effective_batch_analysis"]["candidate_sizes"]
        )

    @property
    def coefficients(self) -> Mapping[str, float]:
        return {
            str(name): float(value)
            for name, value in self.payload["objective"]["coefficients"].items()
        }


def grouped_artifact_updates(
    updates: Sequence[int],
    size: int,
) -> tuple[tuple[int, ...], ...]:
    values = tuple(int(value) for value in updates)
    size = int(size)
    if size <= 0 or len(values) % size != 0:
        raise ValueError("structured GPP heterogeneity group size does not partition artifacts")
    return tuple(values[start : start + size] for start in range(0, len(values), size))


def minimum_consistent_effective_batch_size(
    grouped_reports: Mapping[int, Sequence[Mapping[str, Any]]],
    candidate_sizes: Sequence[int] = EFFECTIVE_BATCH_SIZES,
) -> int | None:
    for size in tuple(int(value) for value in candidate_sizes):
        reports = tuple(grouped_reports.get(size, ()))
        if reports and all(
            all(float(value) > 0.0 for value in report["source_to_combined_cosines"].values())
            for report in reports
        ):
            return size
    return None


def load_structured_gpp_joint_heterogeneity_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> StructuredGPPJointHeterogeneityContract:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    observed = contract_sha256(payload)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("structured GPP heterogeneity audit schema drift")
    if (
        payload.get("status") != "frozen_before_free_gpu_train_only_gradient_heterogeneity_audit"
        or payload.get("contract_sha256") != observed
        or observed != FROZEN_CONTRACT_SHA256
    ):
        raise ValueError("structured GPP heterogeneity audit contract is not frozen")

    identity = payload["identity"]
    data = payload["data"]
    objective = payload["objective"]
    analysis = payload["effective_batch_analysis"]
    execution = payload["execution"]
    coefficients = {str(name): float(value) for name, value in objective["coefficients"].items()}
    if (
        int(identity["warm_start_checkpoint_update"]) != 250
        or int(identity["candidate_parameter_count"]) != 106_383_527
        or int(identity["differentiated_specialist_parameter_count"]) != SPECIALIST_PARAMETER_COUNT
        or int(identity["retained_parent_parameter_count"]) != 93_504_164
        or tuple(int(value) for value in data["artifact_updates"]) != ARTIFACT_UPDATES
        or tuple(int(value) for value in data["artifact_runtime_years"])
        != (1973, 1969, 1989, 1988, 1972, 2003, 2002, 1987)
        or int(data["direct_gpp_teacher_days_per_artifact"]) != 208
        or int(data["seven_day_windows_per_artifact"]) != 8
        or int(data["thirty_day_windows_per_artifact"]) != 4
        or data["calibration_artifacts_reused"] is not False
        or data["evaluation_artifacts_reused"] is not False
        or any(
            bool(data[name])
            for name in (
                "teacher_data_regeneration_allowed",
                "historical_v5_allowed",
                "validation_values_allowed",
                "sealed_test_allowed",
            )
        )
    ):
        raise ValueError("structured GPP heterogeneity identity or data boundary drift")
    if (
        tuple(objective["gradient_sources"]) != GRADIENT_SOURCES
        or set(coefficients) != set(GRADIENT_SOURCES)
        or coefficients[DIRECT_SOURCE] != 1.0
        or coefficients[SEVEN_SOURCE] != 5.065197169020285
        or coefficients[THIRTY_SOURCE] != 4.594433600967992
        or objective["coefficients_recalibrated"] is not False
        or objective["validation_weight_tuning_allowed"] is not False
        or tuple(int(value) for value in analysis["candidate_sizes"]) != EFFECTIVE_BATCH_SIZES
        or analysis["grouping"] != "contiguous_nonoverlapping_groups_in_frozen_artifact_order"
        or analysis["source_gradient_aggregation"] != "arithmetic_mean_across_artifacts_in_group"
        or analysis["optimizer_created"] is not False
        or int(analysis["optimizer_updates_executed"]) != 0
    ):
        raise ValueError("structured GPP heterogeneity objective or analysis drift")
    if (
        execution["free_gln01_allowed"] is not True
        or any(
            bool(execution[name])
            for name in (
                "paid_training_authorized",
                "joint_training_authorized",
                "optimizer_update_authorized",
                "teacher_data_regeneration_allowed",
                "historical_v5_allowed",
                "validation_read_authorized",
                "365_day_rollout_authorized",
                "gate_d2_authorized",
                "sealed_test_authorized",
                "product_acceptance_authorized",
                "cross_day_neural_hidden_state",
            )
        )
        or int(execution["forcing_records_reconstructed"]) != 0
        or int(execution["subdaily_state_scan_steps"]) != 0
    ):
        raise ValueError("structured GPP heterogeneity execution boundary drift")
    for size in EFFECTIVE_BATCH_SIZES:
        grouped_artifact_updates(ARTIFACT_UPDATES, size)
    return StructuredGPPJointHeterogeneityContract(payload, source, observed)


__all__ = [
    "ARTIFACT_UPDATES",
    "DEFAULT_CONTRACT_PATH",
    "EFFECTIVE_BATCH_SIZES",
    "FROZEN_CONTRACT_SHA256",
    "SCHEMA_VERSION",
    "StructuredGPPJointHeterogeneityContract",
    "grouped_artifact_updates",
    "load_structured_gpp_joint_heterogeneity_contract",
    "minimum_consistent_effective_batch_size",
]
