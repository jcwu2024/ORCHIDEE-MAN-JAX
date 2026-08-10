"""Admission for the frozen successor small-fit objective coefficients."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = "successor_small_fit_frozen_coefficients_v1"
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "successor_small_fit_frozen_coefficients_v1.json"
)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("contract_sha256", None)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class FrozenSuccessorCoefficients:
    payload: Mapping[str, Any]
    source_path: Path
    source_report_path: Path
    sha256: str
    coefficients: Mapping[str, float]
    source_preflight: Mapping[str, Any]


def load_frozen_successor_coefficients(
    path: str | Path = DEFAULT_CONTRACT_PATH,
    *,
    source_report_path: str | Path,
    runner_contract_sha256: str,
    schedule_sha256: str,
    successor_contract_sha256: str,
    fast_owner_contract_sha256: str,
    first_prepared_artifact_sha256: str,
    components: Sequence[str],
    gradient_ratio_targets: Mapping[str, float],
    coefficient_minimum: float,
    coefficient_maximum: float,
) -> FrozenSuccessorCoefficients:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("status") != "accepted":
        raise ValueError("successor frozen-coefficient contract is not accepted")
    observed = _canonical_sha256(payload)
    if payload.get("contract_sha256") != observed:
        raise ValueError("successor frozen-coefficient contract hash mismatch")

    identity = payload.get("identity", {})
    if (
        identity.get("runner_contract_sha256") != runner_contract_sha256
        or identity.get("schedule_sha256") != schedule_sha256
        or identity.get("successor_candidate_contract_sha256") != successor_contract_sha256
        or identity.get("fast_owner_contract_sha256") != fast_owner_contract_sha256
        or identity.get("first_prepared_artifact_sha256") != first_prepared_artifact_sha256
    ):
        raise ValueError("successor frozen-coefficient identity drift")

    objective = payload.get("objective", {})
    expected_components = tuple(components)
    if tuple(objective.get("components", ())) != expected_components:
        raise ValueError("successor frozen-coefficient component order drift")
    if objective.get("split_label") != "train/train":
        raise ValueError("successor frozen coefficients must come from train/train")
    if objective.get("gradient_ratio_targets") != dict(gradient_ratio_targets):
        raise ValueError("successor frozen-coefficient target ratio drift")
    if (
        objective.get("same_coefficients_for_both_arms") is not True
        or objective.get("recalibrate_during_fit") is not False
        or objective.get("validation_weight_tuning_allowed") is not False
    ):
        raise ValueError("successor frozen-coefficient policy drift")
    if float(objective.get("coefficient_minimum", np.nan)) != float(coefficient_minimum) or float(
        objective.get("coefficient_maximum", np.nan)
    ) != float(coefficient_maximum):
        raise ValueError("successor frozen-coefficient bound drift")
    coefficients = {name: float(objective["resolved_coefficients"][name]) for name in expected_components}
    if set(objective.get("resolved_coefficients", {})) != set(expected_components) or any(
        not np.isfinite(value) or not coefficient_minimum <= value <= coefficient_maximum
        for value in coefficients.values()
    ):
        raise ValueError("successor frozen coefficients are incomplete or out of bounds")

    report_path = Path(source_report_path).resolve()
    source_evidence = payload.get("source_evidence", {})
    if _sha256_file(report_path) != source_evidence.get("runner_report_sha256"):
        raise ValueError("successor frozen-coefficient source report hash drift")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report_identity = report.get("identity", {})
    if (
        report.get("status") != "passed"
        or report.get("run_kind") != "bounded_reuse_probe"
        or report_identity.get("git_head") != source_evidence.get("git_head")
        or report_identity.get("runner_contract_sha256") != runner_contract_sha256
        or report_identity.get("schedule_sha256") != schedule_sha256
        or report_identity.get("successor_candidate_contract_sha256") != successor_contract_sha256
        or report_identity.get("fast_owner_contract_sha256") != fast_owner_contract_sha256
        or report_identity.get("prepared_manifest_sha256") != source_evidence.get("prepared_manifest_sha256")
        or int(report.get("training_updates_per_arm", -1)) != int(source_evidence.get("training_updates_per_arm", -2))
    ):
        raise ValueError("successor frozen-coefficient source report identity drift")
    preflight = report.get("coefficient_preflight", {})
    if preflight.get("resolved_coefficients") != objective["resolved_coefficients"]:
        raise ValueError("successor frozen coefficients differ from source report")
    if preflight.get("target_gradient_ratios") != objective["gradient_ratio_targets"]:
        raise ValueError("successor frozen-coefficient source targets drift")
    if preflight.get("split_label") != objective["split_label"]:
        raise ValueError("successor frozen-coefficient source split drift")
    for field in ("component_gradient_norms", "component_losses"):
        source_values = preflight.get(field, {})
        values = np.asarray(tuple(source_values.values()), dtype=np.float64)
        if values.size != len(expected_components) or np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError(f"successor frozen-coefficient source {field} is invalid")
        if set(source_values) != set(expected_components):
            raise ValueError(f"successor frozen-coefficient source {field} registry drift")
    if (
        not np.isfinite(float(preflight.get("anchor_gradient_norm", np.nan)))
        or float(preflight["anchor_gradient_norm"]) <= 0.0
    ):
        raise ValueError("successor frozen-coefficient anchor gradient norm is invalid")

    return FrozenSuccessorCoefficients(
        payload=payload,
        source_path=source,
        source_report_path=report_path,
        sha256=observed,
        coefficients=coefficients,
        source_preflight=preflight,
    )
