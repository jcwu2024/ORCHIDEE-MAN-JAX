"""Frozen non-test 7/30-day transfer audit for scale-selected coupled GPP."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .target_oriented_coupled_gpp_bounded_training import (
    MONITOR_UPDATES,
    SCHEDULE_SHA256,
)
from .target_oriented_coupled_gpp_objective import (
    DIRECT_SOURCE,
    SEVEN_SOURCE,
    STATE_SOURCE,
    THIRTY_SOURCE,
    TRAINABLE_PARAMETER_COUNT,
)
from .target_oriented_coupled_gpp_scale_selecting_recalibration import (
    FROZEN_CONTRACT_SHA256 as SCALE_SELECTING_CONTRACT_SHA256,
)
from .target_oriented_source_shaped_maintenance import contract_sha256

SCHEMA_VERSION = "target_oriented_coupled_gpp_scale_selected_transfer_audit_v1"
FROZEN_CONTRACT_SHA256 = "110d3b4a158d4ab9a76858ce1afab082df97a842874c248a0094f01869ab571d"
SCALE_SELECTING_RESULT_SHA256 = "c8ae64e5767c1ead579dbb84def10fdab992857abb3a8d676f075ba17c42a0b8"
PRIMARY_SOURCES = (SEVEN_SOURCE, THIRTY_SOURCE)
REPORT_ONLY_SOURCES = (DIRECT_SOURCE, STATE_SOURCE)
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_coupled_gpp_scale_selected_transfer_audit_v1.json"
)


@dataclass(frozen=True)
class CoupledGPPScaleSelectedTransferContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def monitor_updates(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.payload["data"]["monitor_artifact_updates"])


def load_coupled_gpp_scale_selected_transfer_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> CoupledGPPScaleSelectedTransferContract:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    observed = contract_sha256(payload)
    identity = payload.get("identity", {})
    data = payload.get("data", {})
    evaluation = payload.get("evaluation", {})
    retention = payload.get("retention", {})
    execution = payload.get("execution", {})
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "frozen_before_non_test_transfer_values_are_evaluated"
        or payload.get("contract_sha256") != observed
        or observed != FROZEN_CONTRACT_SHA256
        or identity.get("scale_selecting_contract_sha256") != SCALE_SELECTING_CONTRACT_SHA256
        or identity.get("scale_selecting_result_sha256") != SCALE_SELECTING_RESULT_SHA256
        or identity.get("training_schedule_sha256") != SCHEDULE_SHA256
        or int(identity.get("candidate_parameter_count", -1)) != 109_464_359
        or int(identity.get("trainable_parameter_count", -1)) != TRAINABLE_PARAMETER_COUNT
        or int(identity.get("fixed_parameter_count", -1)) != 93_504_164
        or tuple(int(value) for value in data.get("monitor_artifact_updates", ())) != MONITOR_UPDATES
        or data.get("split") != "train_train_non_test_monitor"
        or data.get("gradient_or_scale_selection_use_of_monitor_artifacts") is not False
        or int(data.get("seven_day_window_count", -1)) != 64
        or int(data.get("thirty_day_window_count", -1)) != 32
        or tuple(evaluation.get("primary_sources", ())) != PRIMARY_SOURCES
        or tuple(evaluation.get("report_only_sources", ())) != REPORT_ONLY_SOURCES
        or tuple(int(value) for value in evaluation.get("horizons_days", ())) != (7, 30)
        or float(evaluation.get("relative_improvement_threshold", -1.0)) != 0.0
        or evaluation.get("both_primary_mean_losses_must_strictly_improve") is not True
        or evaluation.get("per_artifact_unanimity_required") is not False
        or evaluation.get("before_and_after_physical_hard_gates_must_pass") is not True
    ):
        raise ValueError("coupled GPP scale-selected transfer contract drift")
    if any(
        bool(data[name])
        for name in (
            "fresh_validation_values_allowed",
            "teacher_data_regeneration_allowed",
            "historical_v5_allowed",
            "sealed_test_allowed",
        )
    ) or any(bool(value) for value in retention.values()):
        raise ValueError("coupled GPP scale-selected transfer opens a forbidden data or retention path")
    if (
        execution.get("free_gln01_allowed") is not True
        or execution.get("paid_partition_authorized") is not False
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
        or int(execution.get("forcing_records_reconstructed", -1)) != 0
        or int(execution.get("subdaily_state_scan_steps", -1)) != 0
        or int(execution.get("timeout_minutes", -1)) != 120
    ):
        raise ValueError("coupled GPP scale-selected transfer execution boundary drift")
    return CoupledGPPScaleSelectedTransferContract(payload, source, observed)


def transfer_changes(
    before: Mapping[str, float],
    after: Mapping[str, float],
) -> Mapping[str, Mapping[str, float]]:
    result = {}
    for name in (*PRIMARY_SOURCES, *REPORT_ONLY_SOURCES):
        first = float(before[name])
        second = float(after[name])
        if first <= 0.0 or second < 0.0:
            raise ValueError("coupled GPP transfer loss is invalid")
        result[name] = {
            "before": first,
            "after": second,
            "relative_change": (second - first) / first,
        }
    return result


def transfer_accepted(changes: Mapping[str, Mapping[str, float]]) -> bool:
    return all(float(changes[name]["relative_change"]) < 0.0 for name in PRIMARY_SOURCES)


__all__ = [
    "DEFAULT_CONTRACT_PATH",
    "FROZEN_CONTRACT_SHA256",
    "PRIMARY_SOURCES",
    "REPORT_ONLY_SOURCES",
    "SCALE_SELECTING_RESULT_SHA256",
    "CoupledGPPScaleSelectedTransferContract",
    "load_coupled_gpp_scale_selected_transfer_contract",
    "transfer_accepted",
    "transfer_changes",
]
