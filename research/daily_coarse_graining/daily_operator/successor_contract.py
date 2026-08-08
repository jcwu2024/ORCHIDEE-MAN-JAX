"""Frozen local-admission contract for the successor process-head pair."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = "successor_process_head_candidate_v1"
FROZEN_CONTRACT_SHA256 = "bb8bb34972e6fdff713f06f7ce48a2faa9eb4991af9f269c9aab26d96b592b5c"
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "successor_process_head_candidate_v1.json"
)


def contract_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("contract_sha256", None)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SuccessorCandidateContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def candidate_id(self) -> str:
        return str(self.payload["candidate"]["id"])

    @property
    def control_id(self) -> str:
        return str(self.payload["control"]["id"])


def _require_sha256(value: Any, label: str) -> None:
    text = str(value)
    if (
        len(text) != 64
        or text.lower() != text
        or any(character not in "0123456789abcdef" for character in text)
    ):
        raise ValueError(f"successor data identity requires lowercase SHA256 {label}")


def _flatten_groups(groups: Mapping[str, Any], label: str) -> tuple[str, ...]:
    values = tuple(str(value) for members in groups.values() for value in members)
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(f"successor {label} groups overlap: {duplicates}")
    return values


def load_successor_candidate_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> SuccessorCandidateContract:
    """Load and fail closed on architecture, source-route, or gate drift."""

    source_path = Path(path).resolve()
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    observed = contract_sha256(payload)
    if payload.get("contract_sha256") != observed:
        raise ValueError("successor-candidate contract hash mismatch")
    if observed != FROZEN_CONTRACT_SHA256:
        raise ValueError("successor-candidate v1 frozen hash drift")

    data = payload.get("data", {})
    for name in (
        "dataset_manifest_sha256",
        "acceptance_report_sha256",
        "training_statistics_json_sha256",
        "training_statistics_npz_sha256",
        "markov_contract_sha256",
        "typed_contract_sha256",
    ):
        _require_sha256(data.get(name), name)
    if (
        data.get("historical_v5_allowed") is not False
        or data.get("teacher_regeneration_allowed") is not False
        or data.get("test_split_sealed") is not True
    ):
        raise ValueError("successor data policy drift")

    candidate = payload.get("candidate", {})
    if candidate.get("family") != "source_routed_field_query_attention":
        raise ValueError("successor candidate family drift")
    width = int(candidate.get("d_model", 0))
    heads = int(candidate.get("attention_heads", 0))
    if width < 1 or heads < 1 or width % heads:
        raise ValueError("successor d_model must be divisible by attention heads")
    if int(candidate.get("forcing_blocks", 0)) != 2 or int(candidate.get("process_blocks", 0)) != 3:
        raise ValueError("successor depth drift")
    if candidate.get("global_all_token_attention") is not False:
        raise ValueError("global all-token attention is forbidden")
    if candidate.get("cross_day_hidden_state") is not False:
        raise ValueError("cross-day hidden state is forbidden")
    minimum = int(candidate.get("parameter_count_minimum", 0))
    maximum = int(candidate.get("parameter_count_maximum", 0))
    if minimum < 1 or maximum < minimum:
        raise ValueError("successor parameter interval is invalid")

    control = payload.get("control", {})
    if control.get("same_tokens_edges_queries_decoders_and_loss") is not True:
        raise ValueError("successor control must share the scientific boundary")
    tolerance = float(control.get("maximum_parameter_count_relative_difference", -1.0))
    if not 0.0 <= tolerance <= 0.02:
        raise ValueError("successor parameter-match tolerance exceeds two percent")
    if control.get("unused_candidate_parameters_allowed") is not False:
        raise ValueError("successor control cannot carry unused candidate parameters")

    families = payload.get("input_source_families", {})
    forcing = _flatten_groups(families.get("forcing_fields", {}), "forcing-field")
    if forcing != (
        "Tair",
        "PSurf",
        "Qair",
        "Wind_E",
        "Wind_N",
        "Rainf",
        "Snowf",
        "SWdown",
        "LWdown",
    ):
        raise ValueError("successor forcing registry drift")
    _flatten_groups(families.get("state_components", {}), "state-component")
    _flatten_groups(families.get("static_fields", {}), "static-field")
    _flatten_groups(families.get("annual_fields", {}), "annual-field")
    _flatten_groups(families.get("process_static_fields", {}), "process-static")

    declared_families = set(families.get("structural", ()))
    for key in (
        "forcing_fields",
        "state_components",
        "static_fields",
        "annual_fields",
        "process_static_fields",
    ):
        declared_families.update(families.get(key, {}))
    routes = payload.get("output_source_routes", {})
    if set(routes) != {
        "water",
        "carbon.fast_gpp",
        "carbon.fast_maintenance_respiration",
        "carbon.other",
        "energy",
    }:
        raise ValueError("successor output source-route registry drift")
    unknown = sorted(
        {
            str(family)
            for route in routes.values()
            for family in route
            if family not in declared_families
        }
    )
    if unknown:
        raise ValueError(f"successor output routes reference unknown families: {unknown}")

    loss = payload.get("loss", {})
    if float(loss.get("one_day_process_anchor_coefficient", 0.0)) != 1.0:
        raise ValueError("successor one-day anchor must remain one")
    if tuple(loss.get("horizons_days", ())) != (1, 7, 30):
        raise ValueError("successor rollout horizons drift")
    if sum(float(value) for value in loss.get("gradient_ratio_targets", {}).values()) != 0.5:
        raise ValueError("successor long-horizon gradient budget must total 0.5")
    if not 0.0 < float(loss.get("coefficient_minimum", 0.0)) < float(
        loss.get("coefficient_maximum", 0.0)
    ):
        raise ValueError("successor coefficient interval is invalid")
    if loss.get("single_train_train_preflight") is not True or loss.get(
        "validation_weight_tuning_allowed"
    ) is not False:
        raise ValueError("successor coefficient policy drift")

    small = payload.get("bounded_small_fit", {})
    if int(small.get("maximum_updates", 0)) != 500 or small.get("checkpoint_written") is not False:
        raise ValueError("successor small-fit budget drift")
    gates = payload.get("gates", {})
    if float(gates.get("budget_absolute_tolerance", 0.0)) != 2.0e-12:
        raise ValueError("successor budget tolerance drift")
    if int(gates.get("hard_failure_count", -1)) != 0:
        raise ValueError("successor hard failures cannot be admitted")

    return SuccessorCandidateContract(payload=payload, source_path=source_path, sha256=observed)
