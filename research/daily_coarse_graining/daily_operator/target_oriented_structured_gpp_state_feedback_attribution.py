"""Frozen train-only source-component attribution for structured-GPP feedback."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .target_oriented_flux_basis import contract_sha256

SCHEMA_VERSION = "target_oriented_structured_gpp_state_feedback_attribution_v1"
FROZEN_CONTRACT_SHA256 = "858b59077d4a9dbdc7ecaeb06345725bd94c2a570f1b78d258d8dee480bc69ed"
SOURCE_COMPONENTS = (
    ("state_canopy", "diffuco_previous_step_state"),
    ("state_energy", "enerbil_previous_step_state"),
    ("state_water", "hydrol_previous_step_state"),
    ("state_thermal", "thermosoil_previous_step_state"),
    ("state_carbon", "slowproc_stomate_previous_step_state"),
    ("state_driver", "driver_previous_step_state"),
)
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_structured_gpp_state_feedback_attribution_v1.json"
)


@dataclass(frozen=True)
class StructuredGPPStateFeedbackAttributionContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def evaluation_updates(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.payload["data"]["prepared_updates"])

    @property
    def source_components(self) -> tuple[tuple[str, str], ...]:
        return tuple((str(name), str(component)) for name, component in self.payload["source_components"].items())


def _validate(payload: Mapping[str, Any]) -> None:
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "frozen_before_zero_training_execution"
    ):
        raise ValueError("structured-GPP state-feedback attribution schema drift")
    identity = payload.get("identity", {})
    if (
        identity.get("candidate_contract_sha256") != "ce211fa7c6f5a09bd900e02e31a864b61fb32f6b06ee96d4262402c513d620ea"
        or identity.get("broad_training_report_sha256")
        != "5cfab5a693be4c58fd2726420106f2844f93703e5ca8e2910268498154b2bee9"
        or identity.get("selected_checkpoint_sha256")
        != "3b173b8980c37b49db3d1547dddad2ab98ec00c733e2f564469490ee382d4ec0"
        or identity.get("prepared_manifest_sha256")
        != "8205140ddeae34967a287d41d204318659a84a8574de954e9b0be14f8998fc31"
        or int(identity.get("candidate_parameter_count", 0)) != 106_383_527
    ):
        raise ValueError("structured-GPP state-feedback attribution identity drift")
    data = payload.get("data", {})
    if (
        data.get("release_id") != "pft14-daily-teacher-v6"
        or data.get("split") != "train/train"
        or tuple(data.get("prepared_updates", ())) != (301, 601, 901)
        or int(data.get("seven_day_windows_per_update", 0)) != 8
        or int(data.get("thirty_day_windows_per_update", 0)) != 4
        or bool(data.get("fresh_validation_values_allowed"))
        or bool(data.get("sealed_test_allowed"))
    ):
        raise ValueError("structured-GPP state-feedback attribution data drift")
    if tuple(payload.get("source_components", {}).items()) != SOURCE_COMPONENTS:
        raise ValueError("structured-GPP state-feedback source-component registry drift")
    measurement = payload.get("measurement", {})
    classification = payload.get("classification", {})
    if (
        tuple(measurement.get("horizons_days", ())) != (7, 30)
        or float(measurement.get("minimum_positive_free_to_teacher_score_gap", -1.0)) != 1.0e-12
        or int(classification.get("minimum_eligible_group_count_per_horizon", 0)) != 2
        or float(classification.get("material_median_capture_fraction", -1.0)) != 0.1
        or int(classification.get("material_positive_capture_group_count", 0)) != 2
    ):
        raise ValueError("structured-GPP state-feedback attribution measurement drift")
    execution = payload.get("execution", {})
    if (
        execution.get("free_gln01_allowed") is not True
        or execution.get("paid_execution_authorized") is not False
        or any(
            int(execution.get(name, -1)) != 0 for name in ("optimizer_updates", "parameter_steps", "checkpoint_writes")
        )
        or any(
            bool(execution.get(name))
            for name in (
                "teacher_data_regeneration_allowed",
                "historical_v5_allowed",
                "architecture_or_loss_edit_allowed",
                "fresh_validation_read_allowed",
                "sealed_test_allowed",
                "365_day_rollout_allowed",
                "gate_d2_allowed",
                "product_acceptance_allowed",
            )
        )
        or int(execution.get("forcing_records_reconstructed", -1)) != 0
        or int(execution.get("subdaily_state_scan_steps", -1)) != 0
    ):
        raise ValueError("structured-GPP state-feedback attribution execution drift")


def load_structured_gpp_state_feedback_attribution_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> StructuredGPPStateFeedbackAttributionContract:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    _validate(payload)
    observed = contract_sha256(payload)
    if payload.get("contract_sha256") != observed or observed != FROZEN_CONTRACT_SHA256:
        raise ValueError("structured-GPP state-feedback attribution contract is not frozen v1")
    return StructuredGPPStateFeedbackAttributionContract(payload, source, observed)


def resolve_component_state_masks(
    markov_contract: Any,
    components: Sequence[tuple[str, str]] = SOURCE_COMPONENTS,
) -> Mapping[str, Mapping[str, Any]]:
    """Resolve each source component to one runtime mask over continuous state."""

    width = int(markov_contract.continuous_state_width)
    result = {}
    for name, component in tuple(components):
        leaves = tuple(
            leaf for leaf in markov_contract.state_leaves if leaf.component == component and not leaf.discrete
        )
        if not leaves:
            raise ValueError(f"source component has no continuous state leaves: {component}")
        mask = np.zeros((width,), dtype=np.bool_)
        for leaf in leaves:
            if leaf.start is None or leaf.stop is None or not (0 <= leaf.start < leaf.stop <= width):
                raise ValueError(f"invalid continuous state slice for {leaf.key}")
            mask[leaf.start : leaf.stop] = True
        result[name] = {
            "source_component": component,
            "state_keys": tuple(leaf.key for leaf in leaves),
            "state_coordinate_count": int(np.sum(mask)),
            "mask": mask,
        }
    return result


def acceptance_score(record: Mapping[str, Any]) -> float:
    rmse = float(record["normalized_rmse"])
    bias = float(record["normalized_signed_bias"])
    return 0.5 * (rmse * rmse + bias * bias)


def classify_component_feedback(
    free_groups: Sequence[Mapping[str, Any]],
    teacher_groups: Sequence[Mapping[str, Any]],
    component_groups: Mapping[str, Sequence[Mapping[str, Any]]],
    contract: StructuredGPPStateFeedbackAttributionContract,
) -> Mapping[str, Any]:
    """Classify source components by captured free-to-Teacher GPP score gap."""

    updates = contract.evaluation_updates

    def lookup(records: Sequence[Mapping[str, Any]]) -> Mapping[int, Mapping[str, Any]]:
        values = {int(record["update"]): record for record in records}
        if tuple(values) != updates:
            raise ValueError("state-feedback attribution group identity drift")
        return values

    free = lookup(free_groups)
    teacher = lookup(teacher_groups)
    component_lookup = {name: lookup(records) for name, records in component_groups.items()}
    if tuple(component_lookup) != tuple(name for name, _ in contract.source_components):
        raise ValueError("state-feedback attribution component result registry drift")

    measurement = contract.payload["measurement"]
    policy = contract.payload["classification"]
    gap_minimum = float(measurement["minimum_positive_free_to_teacher_score_gap"])
    eligible_minimum = int(policy["minimum_eligible_group_count_per_horizon"])
    capture_minimum = float(policy["material_median_capture_fraction"])
    positive_minimum = int(policy["material_positive_capture_group_count"])
    components = {}
    selected = []
    for name, source_component in contract.source_components:
        horizons = {}
        material_any = False
        for horizon in ("7", "30"):
            records = []
            captures = []
            for update in updates:
                free_key = free[update]["acceptance"][horizon]["keys"]["gpp"]
                teacher_key = teacher[update]["acceptance"][horizon]["keys"]["gpp"]
                component_key = component_lookup[name][update]["acceptance"][horizon]["keys"]["gpp"]
                free_score = acceptance_score(free_key)
                teacher_score = acceptance_score(teacher_key)
                component_score = acceptance_score(component_key)
                gap = free_score - teacher_score
                capture = (free_score - component_score) / gap if gap > gap_minimum else None
                if capture is not None:
                    captures.append(capture)
                records.append(
                    {
                        "update": update,
                        "free_score": free_score,
                        "full_teacher_score": teacher_score,
                        "component_score": component_score,
                        "free_to_teacher_score_gap": gap,
                        "capture_fraction": capture,
                    }
                )
            median = float(np.median(captures)) if captures else None
            positive_count = sum(value > 0.0 for value in captures)
            material = bool(
                len(captures) >= eligible_minimum
                and median is not None
                and median >= capture_minimum
                and positive_count >= positive_minimum
            )
            material_any = material_any or material
            horizons[horizon] = {
                "eligible_group_count": len(captures),
                "positive_capture_group_count": positive_count,
                "median_capture_fraction": median,
                "material": material,
                "groups": records,
            }
        if material_any:
            selected.append(name)
        components[name] = {
            "source_component": source_component,
            "material_at_either_horizon": material_any,
            "horizons": horizons,
        }
    classification = (
        "material_source_components_identified" if selected else "no_individually_material_source_component"
    )
    action_key = "on_material_components" if selected else "on_no_material_component"
    return {
        "selected_components": selected,
        "selected_source_components": [dict(contract.source_components)[name] for name in selected],
        "components": components,
        "classification": classification,
        "action": contract.payload["next_gate"][action_key],
    }


__all__ = [
    "DEFAULT_CONTRACT_PATH",
    "FROZEN_CONTRACT_SHA256",
    "SCHEMA_VERSION",
    "SOURCE_COMPONENTS",
    "StructuredGPPStateFeedbackAttributionContract",
    "acceptance_score",
    "classify_component_feedback",
    "load_structured_gpp_state_feedback_attribution_contract",
    "resolve_component_state_masks",
]
