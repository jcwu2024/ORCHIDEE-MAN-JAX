"""Frozen source-shaped update-0/update-100 coupling attribution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = "target_oriented_source_shaped_coupling_attribution_v1"
FROZEN_SHA256 = "12b327c81851551dfd7ef5e4878b03845290d78b5d51d26168046fe25dce93b2"
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_source_shaped_coupling_attribution_v1.json"
)


def contract_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("contract_sha256", None)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SourceShapedCouplingAttributionContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def evaluation_updates(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.payload["data"]["evaluation_updates"])

    @property
    def protected_keys(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.payload["metrics"]["protected_keys"])

    @property
    def comparison_keys(self) -> tuple[str, ...]:
        reported = tuple(str(value) for value in self.payload["metrics"]["reported_fluxes"])
        return tuple(dict.fromkeys((*reported, *self.protected_keys)))


def load_source_shaped_coupling_attribution_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> SourceShapedCouplingAttributionContract:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    observed = contract_sha256(payload)
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "frozen_before_zero_training_execution"
    ):
        raise ValueError("source-shaped coupling attribution schema drift")
    if payload.get("contract_sha256") != observed or observed != FROZEN_SHA256:
        raise ValueError("source-shaped coupling attribution contract SHA256 drift")
    if tuple(payload["data"]["evaluation_updates"]) != (
        251,
        358,
        465,
        572,
        679,
        786,
        893,
        1000,
    ):
        raise ValueError("source-shaped coupling evaluation panel drift")
    if payload["execution"]["training_updates"] != 0 or payload["execution"]["optimizer_allowed"]:
        raise ValueError("source-shaped coupling attribution must remain zero-training")
    return SourceShapedCouplingAttributionContract(payload, source, observed)


def _acceptance_score(record: Mapping[str, Any]) -> float:
    rmse = float(record["normalized_rmse"])
    bias = float(record["normalized_signed_bias"])
    return 0.5 * (rmse * rmse + bias * bias)


def _relative_change(after: float, before: float) -> float:
    if not np.isfinite(after) or not np.isfinite(before):
        return float("nan")
    if before > 1.0e-15:
        return after / before - 1.0
    return 0.0 if abs(after - before) <= 1.0e-15 else float("inf")


def _group_lookup(
    groups: Sequence[Mapping[str, Any]],
    expected: Sequence[int],
) -> Mapping[int, Mapping[str, Any]]:
    lookup = {int(group["update"]): group for group in groups}
    if tuple(lookup) != tuple(expected):
        raise ValueError("source-shaped coupling group identity or order drift")
    return lookup


def compare_update_states(
    update_0_groups: Sequence[Mapping[str, Any]],
    update_100_groups: Sequence[Mapping[str, Any]],
    contract: SourceShapedCouplingAttributionContract,
) -> Mapping[str, Any]:
    """Compare accepted scores at update 100 against source-shaped update 0."""

    expected = contract.evaluation_updates
    before = _group_lookup(update_0_groups, expected)
    after = _group_lookup(update_100_groups, expected)
    keys = contract.comparison_keys
    horizons = {}
    all_finite = True
    for horizon in ("7", "30"):
        key_records = {}
        for name in keys:
            changes = []
            score_deltas = []
            bias_deltas = []
            for update in expected:
                left = before[update]["acceptance"][horizon]["keys"][name]
                right = after[update]["acceptance"][horizon]["keys"][name]
                left_score = _acceptance_score(left)
                right_score = _acceptance_score(right)
                changes.append(_relative_change(right_score, left_score))
                score_deltas.append(right_score - left_score)
                bias_deltas.append(float(right["normalized_signed_bias"]) - float(left["normalized_signed_bias"]))
            values = (*changes, *score_deltas, *bias_deltas)
            all_finite = all_finite and all(np.isfinite(value) for value in values)
            key_records[name] = {
                "median_relative_score_change": float(np.median(changes)),
                "median_absolute_score_change": float(np.median(score_deltas)),
                "median_normalized_signed_bias_change": float(np.median(bias_deltas)),
                "regressed_group_count": int(sum(value > 0.0 for value in score_deltas)),
                "improved_group_count": int(sum(value < 0.0 for value in score_deltas)),
                "group_relative_score_changes": changes,
            }
        horizons[horizon] = {"keys": key_records}
    return {"all_finite": all_finite, "horizons": horizons}


def carbon_bias_identity_report(
    groups_by_state: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    flux_scales: Mapping[str, float],
    contract: SourceShapedCouplingAttributionContract,
) -> Mapping[str, Any]:
    """Check raw mean cumulative-bias identities behind derived NPP."""

    required = {
        "gpp",
        "maintenance_respiration",
        "growth_respiration",
        "autotrophic_respiration",
        "npp",
    }
    if not required <= set(flux_scales):
        raise ValueError("source-shaped coupling flux-scale registry is incomplete")
    maximum_autotrophic = 0.0
    maximum_npp = 0.0
    records = []
    for state_name, groups in groups_by_state.items():
        lookup = _group_lookup(groups, contract.evaluation_updates)
        for update in contract.evaluation_updates:
            for horizon_text in ("7", "30"):
                horizon = int(horizon_text)
                keys = lookup[update]["acceptance"][horizon_text]["keys"]
                raw_bias = {
                    name: float(keys[name]["normalized_signed_bias"]) * horizon * float(flux_scales[name])
                    for name in required
                }
                autotrophic_residual = raw_bias["autotrophic_respiration"] - (
                    raw_bias["maintenance_respiration"] + raw_bias["growth_respiration"]
                )
                npp_residual = raw_bias["npp"] - (
                    raw_bias["gpp"] - raw_bias["maintenance_respiration"] - raw_bias["growth_respiration"]
                )
                maximum_autotrophic = max(maximum_autotrophic, abs(autotrophic_residual))
                maximum_npp = max(maximum_npp, abs(npp_residual))
                records.append(
                    {
                        "state": state_name,
                        "update": update,
                        "horizon": horizon,
                        "autotrophic_raw_bias_identity_residual": autotrophic_residual,
                        "npp_raw_bias_identity_residual": npp_residual,
                    }
                )
    tolerance = float(contract.payload["classification"]["raw_bias_identity_absolute_tolerance"])
    maximum = max(maximum_autotrophic, maximum_npp)
    return {
        "maximum_autotrophic_raw_bias_identity_residual": maximum_autotrophic,
        "maximum_npp_raw_bias_identity_residual": maximum_npp,
        "maximum_absolute_identity_residual": maximum,
        "absolute_tolerance": tolerance,
        "passed": bool(np.isfinite(maximum) and maximum <= tolerance),
        "records": records,
    }


def classify_source_shaped_coupling(
    free_comparison: Mapping[str, Any],
    teacher_day_start_comparison: Mapping[str, Any],
    identity: Mapping[str, Any],
    contract: SourceShapedCouplingAttributionContract,
) -> Mapping[str, Any]:
    threshold = float(contract.payload["classification"]["material_median_relative_score_increase"])

    def regressed(comparison: Mapping[str, Any], horizon: str) -> tuple[str, ...]:
        return tuple(
            name
            for name in contract.protected_keys
            if float(comparison["horizons"][horizon]["keys"][name]["median_relative_score_change"]) > threshold
        )

    free = {horizon: regressed(free_comparison, horizon) for horizon in ("7", "30")}
    direct = {horizon: regressed(teacher_day_start_comparison, horizon) for horizon in ("7", "30")}
    recurrence_required = {
        horizon: tuple(name for name in free[horizon] if name not in direct[horizon]) for horizon in ("7", "30")
    }
    maintenance_direct = {
        horizon: float(
            teacher_day_start_comparison["horizons"][horizon]["keys"]["maintenance_respiration"][
                "median_relative_score_change"
            ]
        )
        for horizon in ("7", "30")
    }
    direct_npp = any("npp" in direct[horizon] for horizon in ("7", "30"))
    maintenance_improved = all(value < -threshold for value in maintenance_direct.values())
    joint_scope = bool(identity["passed"] and direct_npp and maintenance_improved)
    if joint_scope:
        classification = "direct_daily_carbon_coupling_requires_joint_flux_scope"
    elif any(direct.values()):
        classification = "direct_daily_coupling_confirmed"
    elif any(free.values()):
        classification = "rollout_recurrence_required"
    else:
        classification = "no_material_branch_training_regression"
    return {
        "material_relative_score_increase": threshold,
        "free_rollout_regressed_keys": free,
        "teacher_day_start_regressed_keys": direct,
        "recurrence_required_regressed_keys": recurrence_required,
        "teacher_day_start_maintenance_score_changes": maintenance_direct,
        "teacher_day_start_maintenance_improved": maintenance_improved,
        "joint_flux_scope_required": joint_scope,
        "classification": classification,
    }


__all__ = [
    "DEFAULT_CONTRACT_PATH",
    "FROZEN_SHA256",
    "SCHEMA_VERSION",
    "SourceShapedCouplingAttributionContract",
    "carbon_bias_identity_report",
    "classify_source_shaped_coupling",
    "compare_update_states",
    "contract_sha256",
    "load_source_shaped_coupling_attribution_contract",
]
