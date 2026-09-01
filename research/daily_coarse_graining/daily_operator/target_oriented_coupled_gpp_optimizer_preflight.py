"""Frozen macro-batch optimizer proposal policy for coupled GPP."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .target_oriented_coupled_gpp_objective import (
    DIRECT_SOURCE,
    GRADIENT_SOURCES,
    SEVEN_SOURCE,
    STATE_SOURCE,
    THIRTY_SOURCE,
    TRAINABLE_PARAMETER_COUNT,
)
from .target_oriented_source_shaped_maintenance import contract_sha256

SCHEMA_VERSION = "target_oriented_coupled_gpp_optimizer_preflight_v1"
BACKOFF_SCHEMA_VERSION = "target_oriented_coupled_gpp_optimizer_backoff_v1"
FROZEN_CONTRACT_SHA256 = "7932efa132780c1c687434980a05fab17d14b4b73cc81926d9eaab92640aa3cf"
BACKOFF_FROZEN_CONTRACT_SHA256 = "129d2892c384aee8d206b61a93e642020a7106dd436fbe88dbe902f90aadc315"
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_coupled_gpp_optimizer_preflight_v1.json"
)


@dataclass(frozen=True)
class CoupledGPPOptimizerPreflightContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def artifact_updates(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.payload["data"]["macro_batch_artifact_updates"])

    @property
    def coefficients(self) -> Mapping[str, float]:
        return {str(name): float(value) for name, value in self.payload["objective"]["coefficients"].items()}

    @property
    def clip_threshold(self) -> float:
        return float(self.payload["objective"]["combined_gradient_p90_clip_threshold"])

    @property
    def learning_rate(self) -> float:
        return float(self.payload["optimizer_proposal"]["learning_rate"])


def load_coupled_gpp_optimizer_preflight_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> CoupledGPPOptimizerPreflightContract:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    observed = contract_sha256(payload)
    schema_version = payload.get("schema_version")
    expected_status_and_hash = {
        SCHEMA_VERSION: (
            "frozen_before_discarded_train_only_macro_batch_proposal",
            FROZEN_CONTRACT_SHA256,
        ),
        BACKOFF_SCHEMA_VERSION: (
            "frozen_before_single_halved_learning_rate_proposal",
            BACKOFF_FROZEN_CONTRACT_SHA256,
        ),
    }
    expected = expected_status_and_hash.get(schema_version)
    if (
        expected is None
        or payload.get("status") != expected[0]
        or payload.get("contract_sha256") != observed
        or observed != expected[1]
    ):
        raise ValueError("coupled GPP optimizer-preflight contract is not frozen")

    identity = payload["identity"]
    data = payload["data"]
    objective = payload["objective"]
    proposal = payload["optimizer_proposal"]
    execution = payload["execution"]
    coefficients = {str(name): float(value) for name, value in objective["coefficients"].items()}
    if (
        identity["objective_calibration_report_sha256"]
        != "4f209b773d020964e485dfdfb1bfbc247be6106dc39540d4fde5be40800b5138"
        or int(identity["trainable_parameter_count"]) != TRAINABLE_PARAMETER_COUNT
        or int(identity["fixed_parameter_count"]) != 93_504_164
        or tuple(int(value) for value in data["macro_batch_artifact_updates"])
        != (352, 353, 354, 355, 356, 357, 359, 360)
        or int(data["artifact_count"]) != 8
        or int(data["direct_gpp_teacher_days"]) != 1_664
        or int(data["one_day_state_transitions"]) != 256
        or int(data["seven_day_windows"]) != 64
        or int(data["thirty_day_windows"]) != 32
        or tuple(objective["gradient_sources"]) != GRADIENT_SOURCES
        or coefficients
        != {
            DIRECT_SOURCE: 1.0,
            STATE_SOURCE: 0.6524275555272667,
            SEVEN_SOURCE: 0.4072260558556481,
            THIRTY_SOURCE: 0.28249951035415655,
        }
        or float(objective["combined_gradient_p90_clip_threshold"]) != 12.808193184117403
        or proposal["state_source"] != "fresh_for_both_trainable_families"
        or bool(proposal["inherited_partial_optimizer_state_allowed"])
        or int(proposal["initial_optimizer_step"]) != 0
        or int(proposal["proposed_optimizer_step"]) != 1
        or float(proposal["learning_rate"]) != (2.0e-6 if schema_version == BACKOFF_SCHEMA_VERSION else 4.0e-6)
        or int(proposal["independent_proposals"]) != 1
        or int(proposal["chained_updates"]) != 0
        or int(proposal["updates_persisted"]) != 0
        or bool(proposal["checkpoint_written"])
        or bool(proposal["recovery_written"])
    ):
        raise ValueError("coupled GPP optimizer-preflight identity or policy drift")
    if schema_version == BACKOFF_SCHEMA_VERSION and (
        identity.get("rejected_optimizer_preflight_contract_sha256") != FROZEN_CONTRACT_SHA256
        or identity.get("rejected_optimizer_preflight_execution_git_head") != "b54c50200318aaebf593dc4f1ea53b5049917dc8"
        or identity.get("rejected_optimizer_preflight_report_sha256")
        != "dd43a960fc077284fc5d5925ed55df4141c20e55d04a9489a820d0db58d146ab"
        or proposal.get("learning_rate_selection") != "single_exact_halving_after_rejected_4e-6_finite_step"
    ):
        raise ValueError("coupled GPP optimizer-backoff predecessor drift")
    if any(
        bool(data[name])
        for name in (
            "teacher_data_regeneration_allowed",
            "historical_v5_allowed",
            "fresh_validation_values_allowed",
            "sealed_test_allowed",
        )
    ) or any(
        bool(execution[name])
        for name in (
            "training_authorized",
            "teacher_data_regeneration_allowed",
            "historical_v5_allowed",
            "fresh_validation_read_authorized",
            "365_day_rollout_authorized",
            "gate_d2_authorized",
            "sealed_test_authorized",
            "product_acceptance_authorized",
            "cross_day_neural_hidden_state",
        )
    ):
        raise ValueError("coupled GPP optimizer preflight opens a forbidden path")
    if int(execution["forcing_records_reconstructed"]) or int(execution["subdaily_state_scan_steps"]):
        raise ValueError("coupled GPP optimizer-preflight execution drift")
    return CoupledGPPOptimizerPreflightContract(payload, source, observed)


def four_source_macro_objectives(
    source_losses: Mapping[str, float],
    coefficients: Mapping[str, float],
) -> Mapping[str, Any]:
    if set(source_losses) != set(GRADIENT_SOURCES) or set(coefficients) != set(GRADIENT_SOURCES):
        raise ValueError("coupled GPP macro source registry drift")
    values = {name: float(source_losses[name]) for name in GRADIENT_SOURCES}
    weights = {name: float(coefficients[name]) for name in GRADIENT_SOURCES}
    if any(not np.isfinite(value) or value <= 0.0 for value in values.values()) or any(
        not np.isfinite(value) or value <= 0.0 for value in weights.values()
    ):
        raise ValueError("coupled GPP macro source loss or coefficient is invalid")
    weighted = {name: weights[name] * values[name] for name in GRADIENT_SOURCES}
    return dict(weighted_sources=weighted, joint=float(sum(weighted.values())))


def macro_source_changes(
    before_records: Sequence[Mapping[str, float]],
    after_records: Sequence[Mapping[str, float]],
) -> Mapping[str, Mapping[str, float]]:
    before_records = tuple(before_records)
    after_records = tuple(after_records)
    if not before_records or len(before_records) != len(after_records):
        raise ValueError("coupled GPP macro source record count drift")
    result = {}
    for name in GRADIENT_SOURCES:
        before = np.asarray([float(record[name]) for record in before_records], dtype=np.float64)
        after = np.asarray([float(record[name]) for record in after_records], dtype=np.float64)
        if not np.all(np.isfinite(before)) or np.any(before <= 0.0) or not np.all(np.isfinite(after)):
            raise ValueError("coupled GPP macro source record is invalid")
        first = float(np.mean(before))
        second = float(np.mean(after))
        result[name] = {
            "before": first,
            "after": second,
            "relative_change": (second - first) / first,
        }
    return result


__all__ = [
    "BACKOFF_FROZEN_CONTRACT_SHA256",
    "BACKOFF_SCHEMA_VERSION",
    "DEFAULT_CONTRACT_PATH",
    "FROZEN_CONTRACT_SHA256",
    "SCHEMA_VERSION",
    "CoupledGPPOptimizerPreflightContract",
    "four_source_macro_objectives",
    "load_coupled_gpp_optimizer_preflight_contract",
    "macro_source_changes",
]
