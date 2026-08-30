"""Frozen first broad-data training stage for the structured-GPP specialist."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .target_oriented_source_shaped_maintenance import contract_sha256
from .target_oriented_structured_gpp_training import SPECIALIST_PARAMETER_COUNT

SCHEMA_VERSION = "target_oriented_structured_gpp_broad_training_v1"
FROZEN_CONTRACT_SHA256 = "c54233c118e206200d1e8e5ec827355a469228c5a5e383786a720dd4af5340e1"
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_structured_gpp_broad_training_v1.json"
)
MONITOR_UPDATES = (0, 250, 500, 1000)
CHECKPOINT_UPDATES = (250, 500, 1000)


@dataclass(frozen=True)
class StructuredGPPBroadTrainingContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def optimizer(self) -> Mapping[str, Any]:
        return self.payload["optimizer"]

    @property
    def clip_threshold(self) -> float:
        return float(self.optimizer["global_gradient_norm_clip"])


def load_structured_gpp_broad_training_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> StructuredGPPBroadTrainingContract:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    observed = contract_sha256(payload)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("structured GPP broad-training schema drift")
    if (
        payload.get("status")
        != "frozen_before_runner_qualification_free_admission_and_paid_approval"
        or payload.get("contract_sha256") != observed
        or observed != FROZEN_CONTRACT_SHA256
    ):
        raise ValueError("structured GPP broad-training contract is not frozen")

    identity = payload["identity"]
    data = payload["data"]
    ownership = payload["parameter_ownership"]
    optimizer = payload["optimizer"]
    bounded = payload["bounded_stage"]
    monitor = payload["train_only_monitor"]
    execution = payload["execution"]
    if (
        identity["broad_preflight_contract_sha256"]
        != "f0e50031715b26898d581d3d99a17ae3c01a6a27e693f2e0130fb9e6c63096a3"
        or identity["schedule_report_sha256"]
        != "7de9e840eb8773b4ab9eaeb136cf97c295a13079b6589eebfea970e83679e252"
        or identity["broad_schedule_sha256"]
        != "b4295558406fd627837b012897585992ccd4e75042f7e76171b2a286bc7d6b8b"
        or identity["broad_preflight_report_sha256"]
        != "0962b18d30e5f1b718554897064f7ca8d859234d24b93068f24ed034e7a39b0d"
        or identity["selected_checkpoint_sha256"]
        != "77cb405fc7cbcea3a394041a141c7d5180bd7068eaa309ccdc33a3995d295f2b"
        or int(identity["selected_checkpoint_optimizer_step"]) != 498
        or int(identity["candidate_parameter_count"]) != 106_383_527
        or int(identity["trainable_specialist_parameter_count"])
        != SPECIALIST_PARAMETER_COUNT
        or int(identity["retained_parent_parameter_count"]) != 93_504_164
        or data["release_id"] != "pft14-daily-teacher-v6"
        or data["split"] != "train/train"
        or int(data["schedule_update_start"]) != 1
        or int(data["schedule_update_end"]) != 1000
        or int(data["optimizer_updates_maximum"]) != 1000
        or int(data["days_per_update"]) != 208
        or int(data["unique_training_days"]) != 208_000
        or int(data["point_years_per_update"]) != 52
        or int(data["days_per_point_year_per_update"]) != 4
        or data["training_days_repeat"] is not False
        or data["monitor_point_years_excluded"] is not True
        or data["prior_exposed_days_excluded"] is not True
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
        raise ValueError("structured GPP broad-training identity or data drift")
    if (
        ownership["trainable_parameter_count"] != SPECIALIST_PARAMETER_COUNT
        or int(ownership["retained_parent_parameter_count"]) != 93_504_164
        or ownership["parent_parameters_frozen"] is not True
        or ownership["fresh_optimizer_state"] is not True
        or optimizer["name"] != "adamw"
        or optimizer["state_source"] != "fresh_zero_moments_at_selected_step_498_parameters"
        or float(optimizer["learning_rate"]) != 4e-6
        or optimizer["learning_rate_schedule"] != "constant"
        or float(optimizer["global_gradient_norm_clip"]) != 20.98786557796449
        or int(optimizer["initial_optimizer_step"]) != 0
        or int(optimizer["maximum_optimizer_step"]) != 1000
    ):
        raise ValueError("structured GPP broad-training ownership or optimizer drift")
    if (
        tuple(int(value) for value in bounded["checkpoint_updates"]) != CHECKPOINT_UPDATES
        or int(bounded["recovery_interval_updates"]) != 25
        or tuple(int(value) for value in monitor["evaluate_at_optimizer_updates"])
        != MONITOR_UPDATES
        or int(monitor["point_years"]) != 128
        or int(monitor["sampled_days"]) != 2048
        or int(monitor["early_stop_update"]) != 500
        or float(monitor["minimum_final_aggregate_loss_reduction"]) != 0.05
        or int(monitor["minimum_improved_point_years"]) != 96
        or monitor["checkpoint_selection_uses_validation"] is not False
        or monitor["absolute_aggregate_normalized_bias_must_not_increase"] is not True
    ):
        raise ValueError("structured GPP broad-training bounded monitor drift")
    if (
        execution["paid_training_requires_explicit_resource_authorization"] is not True
        or execution["paid_training_authorized_by_contract_alone"] is not False
        or execution["free_gln01_runner_admission_allowed"] is not True
        or any(
            bool(execution[name])
            for name in (
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
        raise ValueError("structured GPP broad-training execution boundary drift")
    return StructuredGPPBroadTrainingContract(payload, source, observed)


__all__ = [
    "CHECKPOINT_UPDATES",
    "DEFAULT_CONTRACT_PATH",
    "FROZEN_CONTRACT_SHA256",
    "MONITOR_UPDATES",
    "SCHEMA_VERSION",
    "StructuredGPPBroadTrainingContract",
    "load_structured_gpp_broad_training_contract",
]
