"""Frozen fresh-moment admission contract for coupled-GPP training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .target_oriented_coupled_gpp_bounded_training import (
    FROZEN_CONTRACT_SHA256 as BOUNDED_TRAINING_SHA256,
)
from .target_oriented_coupled_gpp_bounded_training import (
    SCHEDULE_SHA256,
    build_coupled_gpp_training_schedule,
)
from .target_oriented_source_shaped_maintenance import contract_sha256

SCHEMA_VERSION = "target_oriented_coupled_gpp_fresh_moment_admission_v1"
FROZEN_CONTRACT_SHA256 = "a8c362fefb8512cdfe5204dccb2fba4d585f7024366ebe64bf9e5c5a031eda11"
GEOMETRY_REPORT_SHA256 = "0920927313ecba7378128142cce142c9a66bfe5592fe0cf01a37ab4848b60956"
GEOMETRY_CONTRACT_SHA256 = "7585cc5b46a76fb19d08703c8e88839f8758258888a1f302d5448dbf7e79f198"
ADMISSION_UPDATES = (2, 3)
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_coupled_gpp_fresh_moment_admission_v1.json"
)


@dataclass(frozen=True)
class CoupledGPPFreshMomentAdmissionContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def updates(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.payload["data"]["science_optimizer_updates"])

    @property
    def groups(self) -> tuple[tuple[int, ...], ...]:
        return tuple(tuple(int(value) for value in group) for group in self.payload["data"]["artifact_groups"])


def fresh_moment_admission_passed(records: Sequence[Mapping[str, Any]]) -> bool:
    values = tuple(records)
    return bool(
        len(values) == len(ADMISSION_UPDATES)
        and tuple(int(value["optimizer_update"]) for value in values) == ADMISSION_UPDATES
        and all(bool(value["finite_step_accepted"]) for value in values)
    )


def load_coupled_gpp_fresh_moment_admission_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> CoupledGPPFreshMomentAdmissionContract:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    observed = contract_sha256(payload)
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "frozen_before_free_consecutive_update_admission"
        or payload.get("contract_sha256") != observed
        or observed != FROZEN_CONTRACT_SHA256
    ):
        raise ValueError("coupled GPP fresh-moment admission contract is not frozen")
    identity = payload["identity"]
    data = payload["data"]
    optimizer = payload["optimizer_amendment"]
    admission = payload["admission"]
    retention = payload["retention"]
    execution = payload["execution"]
    schedule = build_coupled_gpp_training_schedule()
    updates = tuple(int(value) for value in data["science_optimizer_updates"])
    groups = tuple(tuple(int(value) for value in group) for group in data["artifact_groups"])
    if (
        identity["geometry_report_sha256"] != GEOMETRY_REPORT_SHA256
        or identity["geometry_contract_sha256"] != GEOMETRY_CONTRACT_SHA256
        or identity["bounded_training_contract_sha256"] != BOUNDED_TRAINING_SHA256
        or identity["schedule_sha256"] != SCHEDULE_SHA256
        or int(identity["parent_completed_updates"]) != 1
        or updates != ADMISSION_UPDATES
        or groups != (schedule.groups[1], schedule.groups[2])
        or int(data["macro_batch_artifact_count"]) != 8
        or optimizer["name"] != "fresh_adamw_moments_per_macro_update"
        or optimizer["parameters_continue_between_updates"] is not True
        or optimizer["first_and_second_moments_reset_before_every_macro_update"] is not True
        or int(optimizer["optimizer_bias_correction_step_for_every_proposal"]) != 1
        or optimizer["accepted_optimizer_moments_retained_for_next_update"] is not False
        or float(optimizer["learning_rate"]) != 2.0e-6
        or float(optimizer["global_gradient_norm_clip"]) != 12.808193184117403
        or float(admission["all_four_source_losses_must_not_increase_relative_tolerance"]) != 1.0e-6
        or admission["scientific_rejection_writes_report_instead_of_raising"] is not True
    ):
        raise ValueError("coupled GPP fresh-moment admission identity or method drift")
    if (
        any(
            bool(data[name])
            for name in (
                "fresh_validation_values_allowed",
                "teacher_data_regeneration_allowed",
                "historical_v5_allowed",
                "sealed_test_allowed",
            )
        )
        or any(
            bool(retention[name])
            for name in (
                "parent_recovery_modified",
                "optimizer_updates_persisted",
                "parameter_updates_persisted",
                "checkpoint_written",
            )
        )
        or any(
            bool(execution[name])
            for name in (
                "paid_partition_authorized",
                "fresh_validation_read_authorized",
                "365_day_rollout_authorized",
                "gate_d2_authorized",
                "sealed_test_authorized",
                "product_acceptance_authorized",
            )
        )
    ):
        raise ValueError("coupled GPP fresh-moment admission opens a forbidden path")
    if not bool(execution["free_gln01_allowed"]):
        raise ValueError("coupled GPP fresh-moment free admission is closed")
    if int(execution["forcing_records_reconstructed"]) != 0 or int(execution["subdaily_state_scan_steps"]) != 0:
        raise ValueError("coupled GPP fresh-moment inference boundary drift")
    return CoupledGPPFreshMomentAdmissionContract(payload, source, observed)


__all__ = [
    "ADMISSION_UPDATES",
    "DEFAULT_CONTRACT_PATH",
    "FROZEN_CONTRACT_SHA256",
    "GEOMETRY_CONTRACT_SHA256",
    "GEOMETRY_REPORT_SHA256",
    "SCHEMA_VERSION",
    "CoupledGPPFreshMomentAdmissionContract",
    "fresh_moment_admission_passed",
    "load_coupled_gpp_fresh_moment_admission_contract",
]
