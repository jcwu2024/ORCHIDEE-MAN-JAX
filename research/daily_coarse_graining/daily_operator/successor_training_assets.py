"""Admission for frozen successor sufficiency-training assets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

COEFFICIENT_SCHEMA_VERSION = "successor_training_frozen_coefficients_v2"
VALIDATION_PANEL_SCHEMA_VERSION = "successor_training_validation_panel_v1"


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("contract_sha256", None)
    canonical.pop("panel_sha256", None)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SuccessorTrainingAssets:
    coefficients: Mapping[str, float]
    coefficient_contract: Mapping[str, Any]
    validation_panel: Mapping[str, Any]


def load_successor_training_assets(
    coefficient_path: str | Path,
    validation_panel_path: str | Path,
    *,
    coefficient_report_path: str | Path,
    prepared_manifest_path: str | Path,
    runner_contract: Any,
    components: Sequence[str],
) -> SuccessorTrainingAssets:
    coefficient_source = Path(coefficient_path).resolve()
    coefficients = json.loads(coefficient_source.read_text(encoding="utf-8"))
    objective = runner_contract.payload["objective"]
    identity = coefficients.get("identity", {})
    source = coefficients.get("source_evidence", {})
    coefficient_objective = coefficients.get("objective", {})
    report_path = Path(coefficient_report_path).resolve()
    prepared_path = Path(prepared_manifest_path).resolve()
    if (
        coefficients.get("schema_version") != COEFFICIENT_SCHEMA_VERSION
        or coefficients.get("status") != "accepted"
        or coefficients.get("contract_sha256") != _canonical_sha256(coefficients)
        or identity.get("staged_training_contract_sha256")
        != runner_contract.sha256
        or identity.get("coefficient_probe_runner_contract_sha256")
        != runner_contract.payload["identity"][
            "coefficient_probe_runner_contract_sha256"
        ]
        or identity.get("prepared_manifest_sha256") != _sha256_file(prepared_path)
        or source.get("probe_report_sha256") != _sha256_file(report_path)
        or source.get("reported_coefficient_aggregation")
        != "shared_median_gradient_ratio"
        or int(source.get("record_count", 0)) != 32
        or int(source.get("hard_failure_count", -1)) != 0
        or tuple(coefficient_objective.get("components", ())) != tuple(components)
        or coefficient_objective.get("gradient_ratio_targets")
        != objective["gradient_ratio_targets"]
        or coefficient_objective.get("coefficient_aggregation")
        != objective["coefficient_aggregation"]
        or float(coefficient_objective.get("coefficient_quantile", -1.0))
        != float(objective["coefficient_quantile"])
        or coefficient_objective.get("quantile_method") != "lower"
        or coefficient_objective.get("same_coefficients_for_both_arms") is not True
        or coefficient_objective.get("recalibrate_during_fit") is not False
        or coefficient_objective.get("validation_weight_tuning_allowed") is not False
    ):
        raise ValueError("successor frozen training coefficients drift")
    resolved = {
        name: float(coefficient_objective["resolved_coefficients"][name])
        for name in components
    }
    if set(coefficient_objective.get("resolved_coefficients", {})) != set(components) or any(
        not np.isfinite(value)
        or value < float(objective["coefficient_minimum"])
        or value > float(objective["coefficient_maximum"])
        for value in resolved.values()
    ):
        raise ValueError("successor frozen coefficient values drift")

    panel_source = Path(validation_panel_path).resolve()
    panel = json.loads(panel_source.read_text(encoding="utf-8"))
    validation = runner_contract.payload["validation"]
    if (
        panel.get("schema_version") != VALIDATION_PANEL_SCHEMA_VERSION
        or panel.get("panel_sha256") != _canonical_sha256(panel)
        or panel.get("runner_contract_sha256") != runner_contract.sha256
        or panel.get("sealed_test_used") is not False
        or tuple(panel.get("reported_horizons_days", ()))
        != tuple(validation["reported_horizons_days"])
        or tuple(panel.get("milestones", ()))
        != tuple(runner_contract.sampling["stage_milestones"])
        or panel.get("excluded_formal_window_plan_file_sha256")
        != validation["excluded_formal_window_plan_file_sha256"]
        or panel.get("excluded_formal_window_plan_sha256")
        != validation["excluded_formal_window_plan_sha256"]
        or set(panel.get("slices", {})) != {"temporal", "spatial", "joint"}
    ):
        raise ValueError("successor validation panel drift")
    return SuccessorTrainingAssets(resolved, coefficients, panel)
