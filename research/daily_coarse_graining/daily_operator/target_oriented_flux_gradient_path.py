"""Frozen contract and host analysis for flux-gradient path attribution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .target_oriented_flux_basis import contract_sha256

SCHEMA_VERSION = "target_oriented_flux_gradient_path_attribution_v1"
FROZEN_CONTRACT_SHA256 = "d683b87ae2658c4aeb5658cf42ab0b55f6e9cef94cd18af5419a376e3a821900"
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_flux_gradient_path_attribution_v1.json"
)
FIELD_BASIS = ("gpp", "maintenance_respiration", "growth_respiration")
WINDOW_NAMES = (
    "seven_day_extreme",
    "seven_day_matched_control",
    "thirty_day_extreme",
    "thirty_day_matched_control",
)
TARGET_PARAMETER_FAMILIES = (
    "domain_embedding",
    "domain_blocks",
    "science_query",
    "science_embedding",
    "science_cross_blocks",
    "science_mixer_blocks",
    "category_embedding",
    "primary_blocks",
    "state_blocks",
    "primary_decoder",
    "state_decoder",
    "primary_heads",
    "state_heads",
    "auxiliary_science_projection",
    "auxiliary_bottleneck",
    "auxiliary_label_heads",
)


@dataclass(frozen=True)
class TargetOrientedFluxGradientPathContract:
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
        raise ValueError("target-oriented gradient-path schema drift")
    identity = payload.get("identity", {})
    if (
        identity.get("residual_tail_report_sha256")
        != "e1c094e2dba5f2eeb4ea8becfdf6623ef58fadd3e9d02518312a71d693d27d9e"
        or identity.get("update_1000_checkpoint_sha256")
        != "a566f9d95726984dc03cbf1f1a183dd0d1eae9e3479ce5055c64e571ef6a16c7"
        or identity.get("prepared_manifest_sha256")
        != "8205140ddeae34967a287d41d204318659a84a8574de954e9b0be14f8998fc31"
        or int(identity.get("parameter_count", 0)) != 83_755_173
    ):
        raise ValueError("target-oriented gradient-path identity drift")
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
        raise ValueError("target-oriented gradient-path data boundary drift")
    measurement = payload.get("measurement", {})
    if (
        tuple(measurement.get("field_basis", ())) != FIELD_BASIS
        or tuple(measurement.get("fine_target_families", ())) != TARGET_PARAMETER_FAMILIES
        or int(measurement.get("gradient_coverage", 0)) != 83_755_173
        or float(measurement.get("aggregate_reconstruction_weight_per_field", 0.0)) != 0.125
    ):
        raise ValueError("target-oriented gradient-path measurement drift")
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
        raise ValueError("target-oriented gradient-path execution boundary drift")


def load_target_oriented_flux_gradient_path_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> TargetOrientedFluxGradientPathContract:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    _validate(payload)
    observed = contract_sha256(payload)
    if payload.get("contract_sha256") != observed or observed != FROZEN_CONTRACT_SHA256:
        raise ValueError("target-oriented gradient-path contract is not frozen v1")
    return TargetOrientedFluxGradientPathContract(payload, source, observed)


def fine_parameter_family_trees(gradient: Mapping[str, Any]) -> Mapping[str, Any]:
    target = gradient.get("target_oriented")
    if not isinstance(target, Mapping) or set(target) != set(TARGET_PARAMETER_FAMILIES):
        raise ValueError("target-oriented fine parameter family registry drift")
    return {
        "base_source_encoder": {
            name: value for name, value in gradient.items() if name != "target_oriented"
        },
        **{name: target[name] for name in TARGET_PARAMETER_FAMILIES},
    }


def gram_cosine_summary(gram: Any, labels: Sequence[str]) -> Mapping[str, Any]:
    matrix = np.asarray(gram, dtype=np.float64)
    names = tuple(str(label) for label in labels)
    if matrix.shape != (len(names), len(names)) or np.any(~np.isfinite(matrix)):
        raise ValueError("gradient-path Gram matrix is invalid")
    norms = np.sqrt(np.maximum(np.diag(matrix), 0.0))
    denominator = np.outer(norms, norms)
    cosine = np.divide(
        matrix,
        denominator,
        out=np.full_like(matrix, np.nan),
        where=denominator > 0.0,
    )
    return {
        "labels": list(names),
        "gradient_norms": norms.tolist(),
        "cosines": cosine.tolist(),
    }


def classify_gradient_path(
    records: Mapping[str, Mapping[str, Any]],
    contract: TargetOrientedFluxGradientPathContract,
) -> Mapping[str, Any]:
    if set(records) != set(WINDOW_NAMES):
        raise ValueError("gradient-path result window registry drift")
    thresholds = contract.payload["classification"]
    recurrence_ratio = float(thresholds["free_to_teacher_gradient_ratio_for_recurrence"])
    local_ratio = float(thresholds["free_to_teacher_gradient_ratio_for_local_path"])
    alignment_minimum = float(thresholds["field_alignment_cosine_minimum"])
    matched_minimum = float(thresholds["matched_control_gradient_ratio_minimum"])
    family_fraction_minimum = float(
        thresholds["dominant_parameter_family_squared_norm_fraction"]
    )
    pair_summaries = {}
    mechanisms = []
    for horizon in ("seven", "thirty"):
        extreme = records[f"{horizon}_day_extreme"]
        control = records[f"{horizon}_day_matched_control"]
        fields = extreme["fields"]
        dominant = max(
            FIELD_BASIS,
            key=lambda name: float(fields[name]["free_alignment_to_aggregate"]),
        )
        field_record = fields[dominant]
        free_norm = float(field_record["free_gradient"]["l2_norm"])
        teacher_norm = float(field_record["teacher_state_gradient"]["l2_norm"])
        free_teacher_ratio = free_norm / max(teacher_norm, 1.0e-30)
        control_norm = float(control["fields"][dominant]["free_gradient"]["l2_norm"])
        matched_ratio = free_norm / max(control_norm, 1.0e-30)
        alignment = float(field_record["free_alignment_to_aggregate"])
        family_fraction = float(
            field_record["dominant_fine_parameter_family_squared_norm_fraction"]
        )
        if matched_ratio < matched_minimum or alignment < alignment_minimum:
            mechanism = "mixed_or_unresolved"
        elif free_teacher_ratio >= recurrence_ratio:
            mechanism = "rollout_recurrence_dominant"
        elif free_teacher_ratio <= local_ratio and family_fraction >= family_fraction_minimum:
            dominant_family = str(field_record["dominant_fine_parameter_family"])
            if dominant_family in {
                "primary_blocks",
                "primary_decoder",
                "primary_heads",
                "state_blocks",
                "state_decoder",
                "state_heads",
            }:
                mechanism = "primary_or_state_output_path_dominant"
            else:
                mechanism = "shared_backbone_dominant"
        else:
            mechanism = "mixed_or_unresolved"
        mechanisms.append(mechanism)
        pair_summaries[horizon] = {
            "dominant_field": dominant,
            "field_alignment_to_aggregate": alignment,
            "extreme_to_matched_control_gradient_ratio": matched_ratio,
            "free_to_teacher_state_gradient_ratio": free_teacher_ratio,
            "dominant_fine_parameter_family_squared_norm_fraction": family_fraction,
            "mechanism": mechanism,
        }
    action_key = mechanisms[0] if len(set(mechanisms)) == 1 else "mixed_or_unresolved"
    return {
        "pairs": pair_summaries,
        "classification": action_key,
        "action": str(contract.payload["decision"][action_key]),
    }


__all__ = [
    "DEFAULT_CONTRACT_PATH",
    "FIELD_BASIS",
    "FROZEN_CONTRACT_SHA256",
    "SCHEMA_VERSION",
    "TARGET_PARAMETER_FAMILIES",
    "TargetOrientedFluxGradientPathContract",
    "WINDOW_NAMES",
    "classify_gradient_path",
    "fine_parameter_family_trees",
    "gram_cosine_summary",
    "load_target_oriented_flux_gradient_path_contract",
]
