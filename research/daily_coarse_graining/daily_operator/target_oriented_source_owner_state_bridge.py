"""Frozen four-owner state-bridge causal audit."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .target_oriented_flux_basis import contract_sha256

SCHEMA_VERSION = "target_oriented_source_owner_state_bridge_v1"
FROZEN_CONTRACT_SHA256 = "c8bcadc2ed94a38afaa8788ba5775070725ed1dc00d58db29b2753099ae86a2a"
SOURCE_OWNER_STATE_KEYS = (
    "slowproc_stomate_previous_step_state.biomass",
    "slowproc_stomate_previous_step_state.lai",
    "slowproc_stomate_previous_step_state.veget_max",
    "slowproc_stomate_previous_step_state.t2m_longterm",
)
REQUIRED_REVERSAL_KEYS = (
    "growth_respiration",
    "npp",
    "aboveground_biomass_carbon",
    "belowground_biomass_carbon",
    "total_live_biomass_carbon",
)
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_source_owner_state_bridge_v1.json"
)


@dataclass(frozen=True)
class SourceOwnerStateBridgeContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def evaluation_updates(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.payload["data"]["evaluation_updates"])

    @property
    def state_owner_keys(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.payload["intervention"]["state_owner_keys"])

    @property
    def required_reversal_keys(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.payload["metrics"]["required_reversal_keys"])


def _validate(payload: Mapping[str, Any]) -> None:
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "frozen_before_zero_training_execution"
    ):
        raise ValueError("source-owner state-bridge schema drift")
    identity = payload.get("identity", {})
    if (
        identity.get("source_coupling_contract_sha256")
        != "12b327c81851551dfd7ef5e4878b03845290d78b5d51d26168046fe25dce93b2"
        or identity.get("source_coupling_report_sha256")
        != "e2a0b90f3972887da3e494da943f0a9d9fb8389eb8c551587ff71563426a0d6b"
        or identity.get("branch_checkpoint_sha256")
        != "fd6b866ec7a7c74f13dd7485a11d14126a7c39a689e1b9e2836bfd11079bd153"
        or identity.get("parent_checkpoint_sha256")
        != "a566f9d95726984dc03cbf1f1a183dd0d1eae9e3479ce5055c64e571ef6a16c7"
    ):
        raise ValueError("source-owner state-bridge identity drift")
    data = payload.get("data", {})
    if (
        data.get("release_id") != "pft14-daily-teacher-v6"
        or data.get("split") != "train/train"
        or tuple(data.get("evaluation_updates", ())) != (251, 358, 465, 572, 679, 786, 893, 1000)
        or bool(data.get("validation_values_allowed"))
        or bool(data.get("sealed_test_allowed"))
    ):
        raise ValueError("source-owner state-bridge data boundary drift")
    intervention = payload.get("intervention", {})
    if (
        tuple(intervention.get("state_owner_keys", ())) != SOURCE_OWNER_STATE_KEYS
        or intervention.get("diagnostic_only") is not True
        or intervention.get("allowed_in_training") is not False
        or intervention.get("allowed_in_final_inference") is not False
    ):
        raise ValueError("source-owner state-bridge intervention drift")
    metrics = payload.get("metrics", {})
    if tuple(metrics.get("required_reversal_keys", ())) != REQUIRED_REVERSAL_KEYS or tuple(
        metrics.get("required_horizons", ())
    ) != (7, 30):
        raise ValueError("source-owner state-bridge metric drift")
    execution = payload.get("execution", {})
    if (
        execution.get("free_gln01_allowed") is not True
        or execution.get("paid_execution_authorized") is not False
        or execution.get("optimizer_allowed") is not False
        or int(execution.get("training_updates", -1)) != 0
        or int(execution.get("parameter_steps", -1)) != 0
        or any(
            bool(execution.get(name))
            for name in (
                "checkpoint_write_allowed",
                "teacher_data_regeneration_allowed",
                "historical_v5_allowed",
                "architecture_edit_allowed",
                "validation_read_authorized",
                "sealed_test_authorized",
                "365_day_rollout_allowed",
                "gate_d2_allowed",
            )
        )
        or int(execution.get("forcing_records_reconstructed", -1)) != 0
        or int(execution.get("subdaily_state_scan_steps", -1)) != 0
    ):
        raise ValueError("source-owner state-bridge execution boundary drift")


def load_source_owner_state_bridge_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> SourceOwnerStateBridgeContract:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    _validate(payload)
    observed = contract_sha256(payload)
    if payload.get("contract_sha256") != observed or observed != FROZEN_CONTRACT_SHA256:
        raise ValueError("source-owner state-bridge contract is not frozen v1")
    return SourceOwnerStateBridgeContract(payload, source, observed)


def classify_source_owner_state_bridge(
    free_comparison: Mapping[str, Any],
    bridge_comparison: Mapping[str, Any],
    teacher_day_start_comparison: Mapping[str, Any],
    contract: SourceOwnerStateBridgeContract,
) -> Mapping[str, Any]:
    """Classify whether four exact source owners explain all recurrence reversals."""

    threshold = float(contract.payload["classification"]["material_median_relative_score_increase"])
    cells = []
    for horizon in ("7", "30"):
        for name in contract.required_reversal_keys:
            free = float(free_comparison["horizons"][horizon]["keys"][name]["median_relative_score_change"])
            bridge = float(bridge_comparison["horizons"][horizon]["keys"][name]["median_relative_score_change"])
            teacher = float(
                teacher_day_start_comparison["horizons"][horizon]["keys"][name]["median_relative_score_change"]
            )
            if free <= threshold or teacher > 0.0:
                raise ValueError("source-owner bridge prerequisite reversal drift")
            denominator = free - teacher
            progress = (free - bridge) / denominator if abs(denominator) > 1.0e-15 else float("nan")
            cells.append(
                {
                    "horizon": int(horizon),
                    "key": name,
                    "free_relative_score_change": free,
                    "bridge_relative_score_change": bridge,
                    "teacher_day_start_relative_score_change": teacher,
                    "bridge_progress_from_free_to_teacher": progress,
                    "reversal_reproduced": bridge <= 0.0,
                    "remains_materially_regressed": bridge > threshold,
                }
            )
    reversed_count = sum(bool(record["reversal_reproduced"]) for record in cells)
    total = len(cells)
    if reversed_count == total:
        classification = "source_owner_state_drift_sufficient"
        action = "discuss_source_owner_recurrence_architecture"
    elif reversed_count == 0:
        classification = "source_owner_state_drift_not_explanatory"
        action = "discuss_broader_attention_context_or_other_recurrent_state_paths"
    else:
        classification = "source_owner_state_drift_contributes_but_not_sufficient"
        action = "separate_remaining_state_paths_before_architecture_revision"
    return {
        "material_relative_score_increase": threshold,
        "required_cell_count": total,
        "reversed_cell_count": reversed_count,
        "materially_regressed_cell_count": sum(bool(record["remains_materially_regressed"]) for record in cells),
        "cells": cells,
        "classification": classification,
        "action": action,
    }


__all__ = [
    "DEFAULT_CONTRACT_PATH",
    "FROZEN_CONTRACT_SHA256",
    "REQUIRED_REVERSAL_KEYS",
    "SCHEMA_VERSION",
    "SOURCE_OWNER_STATE_KEYS",
    "SourceOwnerStateBridgeContract",
    "classify_source_owner_state_bridge",
    "load_source_owner_state_bridge_contract",
]
