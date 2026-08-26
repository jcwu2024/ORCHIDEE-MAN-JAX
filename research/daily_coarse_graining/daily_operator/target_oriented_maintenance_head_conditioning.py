"""Frozen host contract for the maintenance final-head conditioning audit."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .target_oriented_flux_basis import contract_sha256

SCHEMA_VERSION = "target_oriented_maintenance_head_conditioning_audit_v1"
FROZEN_CONTRACT_SHA256 = "b354c4e8309e878899ce61c83f851b08279977a9984eccf0b9a60bc7089fc369"
MAINTENANCE_FIELD = "daily.fast_maintenance_respiration"
AUDIT_UPDATES = (251, 358, 465, 572, 679, 786, 893, 1000)
HORIZONS = (7, 30)
HEAD_PARAMETER_COUNT = 769
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_maintenance_head_conditioning_audit_v1.json"
)


@dataclass(frozen=True)
class TargetOrientedMaintenanceHeadConditioningContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def predicted_fractions(self) -> tuple[float, ...]:
        return tuple(
            float(value)
            for value in self.payload["step_calibration"][
                "predicted_median_relative_decrease_fractions"
            ]
        )

    @property
    def gates(self) -> Mapping[str, Any]:
        return self.payload["gates"]


def _validate(payload: Mapping[str, Any]) -> None:
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "frozen_before_zero_training_execution"
    ):
        raise ValueError("maintenance-head conditioning schema drift")
    identity = payload.get("identity", {})
    if (
        identity.get("maintenance_state_recurrence_report_sha256")
        != "a95ae1291a1d4cc77bb026448b4c31e8244604dee9485886c5f7079272f809b9"
        or identity.get("flux_basis_amendment_sha256")
        != "6c81e8f56c2c81fc2ac024a71da98f3e875c4ebc429f9e3d1ff4e8e3ef6bfb45"
        or identity.get("update_1000_checkpoint_sha256")
        != "a566f9d95726984dc03cbf1f1a183dd0d1eae9e3479ce5055c64e571ef6a16c7"
        or identity.get("prepared_manifest_sha256")
        != "8205140ddeae34967a287d41d204318659a84a8574de954e9b0be14f8998fc31"
        or identity.get("candidate")
        != "target_oriented_hierarchical_coordinate_attention_v1"
        or int(identity.get("parameter_count", 0)) != 83_755_173
        or identity.get("head_field") != MAINTENANCE_FIELD
        or int(identity.get("head_parameter_count", 0)) != HEAD_PARAMETER_COUNT
    ):
        raise ValueError("maintenance-head conditioning identity drift")
    data = payload.get("data", {})
    if (
        data.get("release_id") != "pft14-daily-teacher-v6"
        or data.get("split") != "train/train"
        or tuple(data.get("audit_updates", ())) != AUDIT_UPDATES
        or int(data.get("total_seven_day_windows", 0)) != 64
        or int(data.get("total_thirty_day_windows", 0)) != 32
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
        raise ValueError("maintenance-head conditioning data boundary drift")
    direction = payload.get("direction", {})
    if (
        int(direction.get("constraint_count", 0)) != 16
        or int(direction.get("head_parameter_count", 0)) != HEAD_PARAMETER_COUNT
        or direction.get("normalization")
        != "unit_l2_per_group_horizon_head_gradient"
        or direction.get("combination")
        != "minimum_norm_simplex_common_descent"
        or float(direction.get("minimum_constraint_cosine", 0.0)) != 0.01
        or any(
            bool(direction.get(name))
            for name in (
                "momentum",
                "coordinate_adaptive_scaling",
                "weight_decay",
                "gradient_clipping",
            )
        )
    ):
        raise ValueError("maintenance-head conditioning direction drift")
    calibration = payload.get("step_calibration", {})
    fractions = tuple(
        float(value)
        for value in calibration.get(
            "predicted_median_relative_decrease_fractions", ()
        )
    )
    if (
        fractions != (0.001, 0.003, 0.01, 0.03, 0.1)
        or float(calibration.get("maximum_parameter_step_norm", 0.0)) != 0.01
        or calibration.get("out_of_bound_candidates_evaluated") is not False
        or calibration.get("grid_extended_after_result") is not False
    ):
        raise ValueError("maintenance-head conditioning calibration drift")
    execution = payload.get("execution", {})
    if (
        execution.get("free_gln01_only") is not True
        or execution.get("paid_training_authorized") is not False
        or int(execution.get("optimizer_updates", -1)) != 0
        or int(execution.get("parameter_steps_retained", -1)) != 0
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
                "input_mask_edit_allowed",
                "source_routing_edit_allowed",
                "architecture_edit_allowed",
                "365_day_rollout_allowed",
                "gate_d2_allowed",
            )
        )
    ):
        raise ValueError("maintenance-head conditioning execution drift")


def load_target_oriented_maintenance_head_conditioning_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> TargetOrientedMaintenanceHeadConditioningContract:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    _validate(payload)
    observed = contract_sha256(payload)
    if payload.get("contract_sha256") != observed or observed != FROZEN_CONTRACT_SHA256:
        raise ValueError("maintenance-head conditioning contract is not frozen v1")
    return TargetOrientedMaintenanceHeadConditioningContract(payload, source, observed)


def resolve_maintenance_head_index(schema: Any) -> int:
    matches = tuple(
        index
        for index, field in enumerate(schema.fields)
        if str(field.name) == MAINTENANCE_FIELD
    )
    if len(matches) != 1:
        raise ValueError("maintenance final-head field must resolve exactly once")
    return matches[0]


def maintenance_head(parameters: Mapping[str, Any], field_index: int) -> Mapping[str, Any]:
    head = parameters["target_oriented"]["primary_heads"][field_index]
    if set(head) != {"kernel", "bias"}:
        raise ValueError("maintenance final-head parameter structure drift")
    size = sum(int(np.prod(np.shape(value), dtype=np.int64)) for value in head.values())
    if size != HEAD_PARAMETER_COUNT:
        raise ValueError("maintenance final-head parameter count drift")
    return head


def replace_maintenance_head(
    parameters: Mapping[str, Any],
    field_index: int,
    head: Mapping[str, Any],
) -> Mapping[str, Any]:
    current = maintenance_head(parameters, field_index)
    if set(head) != set(current) or any(
        np.shape(head[name]) != np.shape(current[name]) for name in current
    ):
        raise ValueError("replacement maintenance head structure drift")
    heads = list(parameters["target_oriented"]["primary_heads"])
    heads[field_index] = dict(head)
    target = dict(parameters["target_oriented"])
    target["primary_heads"] = tuple(heads)
    result = dict(parameters)
    result["target_oriented"] = target
    return result


def calibrated_step_records(
    parent_scores: Sequence[float],
    directional_derivatives: Sequence[float],
    contract: TargetOrientedMaintenanceHeadConditioningContract,
) -> tuple[Mapping[str, Any], ...]:
    scores = np.asarray(parent_scores, dtype=np.float64)
    derivatives = np.asarray(directional_derivatives, dtype=np.float64)
    if scores.shape != (16,) or derivatives.shape != (16,):
        raise ValueError("maintenance-head calibration requires 16 constraints")
    if (
        np.any(~np.isfinite(scores))
        or np.any(~np.isfinite(derivatives))
        or np.any(scores <= 0.0)
        or np.any(derivatives <= 0.0)
    ):
        raise ValueError("maintenance-head calibration inputs must be finite and positive")
    base = float(np.median(scores / derivatives))
    maximum = float(contract.payload["step_calibration"]["maximum_parameter_step_norm"])
    return tuple(
        {
            "predicted_fraction": fraction,
            "step_norm": fraction * base,
            "within_parameter_bound": fraction * base <= maximum,
        }
        for fraction in contract.predicted_fractions
    )


def _relative_matrix(
    groups: Sequence[Mapping[str, Any]],
    section: str,
    key: str,
) -> Mapping[int, np.ndarray]:
    return {
        horizon: np.asarray(
            [float(group[section][str(horizon)][key]) for group in groups],
            dtype=np.float64,
        )
        for horizon in HORIZONS
    }


def classify_maintenance_head_steps(
    direction: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    contract: TargetOrientedMaintenanceHeadConditioningContract,
) -> Mapping[str, Any]:
    minimum_cosine = float(direction["minimum_constraint_cosine"])
    direction_passed = minimum_cosine >= float(
        contract.payload["direction"]["minimum_constraint_cosine"]
    )
    actions = contract.payload["decision"]
    if not direction_passed:
        return {
            "classification": "common_direction_not_supported",
            "passed": False,
            "direction_passed": False,
            "minimum_constraint_cosine": minimum_cosine,
            "selected_predicted_fraction": None,
            "selected_step_norm": None,
            "candidates": [],
            "action": actions["common_direction_not_supported"],
        }

    gates = contract.gates
    records = []
    for candidate in candidates:
        groups = tuple(candidate["groups"])
        if tuple(int(group["update"]) for group in groups) != AUDIT_UPDATES:
            raise ValueError("maintenance-head candidate group coverage drift")
        maintenance = _relative_matrix(
            groups,
            "independent_flux_relative_changes",
            "maintenance_respiration",
        )
        gpp = _relative_matrix(groups, "independent_flux_relative_changes", "gpp")
        growth = _relative_matrix(
            groups,
            "independent_flux_relative_changes",
            "growth_respiration",
        )
        derived_keys = tuple(groups[0]["derived_flux_relative_changes"]["7"])
        derived = {
            (horizon, key): np.asarray(
                [
                    float(group["derived_flux_relative_changes"][str(horizon)][key])
                    for group in groups
                ],
                dtype=np.float64,
            )
            for horizon in HORIZONS
            for key in derived_keys
        }
        stock_keys = tuple(groups[0]["stock_relative_changes"]["7"])
        stocks = {
            (horizon, key): np.asarray(
                [
                    float(group["stock_relative_changes"][str(horizon)][key])
                    for group in groups
                ],
                dtype=np.float64,
            )
            for horizon in HORIZONS
            for key in stock_keys
        }
        one_day = np.asarray(
            [float(group["one_day_relative_change"]) for group in groups],
            dtype=np.float64,
        )
        all_values = (
            *maintenance.values(),
            *gpp.values(),
            *growth.values(),
            *derived.values(),
            *stocks.values(),
            one_day,
        )
        finite = all(np.all(np.isfinite(value)) for value in all_values)
        hard = sum(int(group["hard_failure_count"]) for group in groups)
        nonfinite = int(candidate["nonfinite_value_count"]) + int(not finite)
        maintenance_medians = {
            str(horizon): float(np.median(values))
            for horizon, values in maintenance.items()
        }
        growth_medians = {
            str(horizon): float(np.median(values)) for horizon, values in growth.items()
        }
        derived_medians = {
            f"{horizon}.{key}": float(np.median(values))
            for (horizon, key), values in derived.items()
        }
        stock_medians = {
            f"{horizon}.{key}": float(np.median(values))
            for (horizon, key), values in stocks.items()
        }
        parent_retained = all(
            bool(group["parent_passing_acceptance_keys_retained"])
            for group in groups
        )
        failing_stock = max(
            float(group["parent_failing_stock_rmse_relative_increase_max"])
            for group in groups
        )
        passed = (
            bool(candidate["within_parameter_bound"])
            and hard == int(gates["hard_failure_count"])
            and nonfinite == int(gates["nonfinite_value_count"])
            and max(float(np.max(value)) for value in maintenance.values())
            <= float(gates["maximum_any_group_maintenance_relative_increase"])
            and all(
                value
                <= -float(
                    gates[
                        "minimum_each_horizon_maintenance_median_relative_decrease"
                    ]
                )
                for value in maintenance_medians.values()
            )
            and max(float(np.max(value)) for value in gpp.values())
            <= float(gates["maximum_any_group_gpp_relative_increase"])
            and all(
                float(np.median(value))
                <= float(gates["maximum_gpp_median_relative_increase"])
                for value in gpp.values()
            )
            and max(float(np.max(value)) for value in growth.values())
            <= float(gates["maximum_any_group_growth_respiration_relative_increase"])
            and all(
                value
                <= float(gates["maximum_growth_respiration_median_relative_increase"])
                for value in growth_medians.values()
            )
            and max(float(np.max(value)) for value in derived.values())
            <= float(gates["maximum_any_group_derived_flux_relative_increase"])
            and all(
                value
                <= float(gates["maximum_derived_flux_median_relative_increase"])
                for value in derived_medians.values()
            )
            and float(np.max(one_day))
            <= float(gates["maximum_any_group_one_day_relative_increase"])
            and float(np.median(one_day))
            <= float(gates["maximum_one_day_median_relative_increase"])
            and max(float(np.max(value)) for value in stocks.values())
            <= float(gates["maximum_any_group_stock_relative_increase"])
            and all(
                value <= float(gates["maximum_stock_median_relative_increase"])
                for value in stock_medians.values()
            )
            and (
                parent_retained
                or not bool(
                    gates["require_parent_passing_acceptance_keys_remain_passing"]
                )
            )
            and failing_stock
            <= float(gates["maximum_parent_failing_stock_rmse_relative_increase"])
        )
        records.append(
            {
                "predicted_fraction": float(candidate["predicted_fraction"]),
                "step_norm": float(candidate["step_norm"]),
                "within_parameter_bound": bool(candidate["within_parameter_bound"]),
                "maintenance_medians": maintenance_medians,
                "maximum_maintenance_relative_increase": max(
                    float(np.max(value)) for value in maintenance.values()
                ),
                "gpp_medians": {
                    str(horizon): float(np.median(value))
                    for horizon, value in gpp.items()
                },
                "maximum_gpp_relative_increase": max(
                    float(np.max(value)) for value in gpp.values()
                ),
                "growth_medians": growth_medians,
                "maximum_growth_relative_increase": max(
                    float(np.max(value)) for value in growth.values()
                ),
                "derived_medians": derived_medians,
                "maximum_derived_relative_increase": max(
                    float(np.max(value)) for value in derived.values()
                ),
                "one_day_median": float(np.median(one_day)),
                "one_day_maximum": float(np.max(one_day)),
                "stock_medians": stock_medians,
                "maximum_stock_relative_increase": max(
                    float(np.max(value)) for value in stocks.values()
                ),
                "hard_failure_count": hard,
                "nonfinite_value_count": nonfinite,
                "passed": passed,
            }
        )

    passing = [record for record in records if bool(record["passed"])]
    selected = passing[-1] if passing else None
    classification = (
        "protected_head_step_supported"
        if selected is not None
        else "finite_head_step_not_protected"
    )
    return {
        "classification": classification,
        "passed": selected is not None,
        "direction_passed": True,
        "minimum_constraint_cosine": minimum_cosine,
        "selected_predicted_fraction": (
            selected["predicted_fraction"] if selected is not None else None
        ),
        "selected_step_norm": selected["step_norm"] if selected is not None else None,
        "candidates": records,
        "action": actions[classification],
    }


__all__ = [
    "AUDIT_UPDATES",
    "DEFAULT_CONTRACT_PATH",
    "FROZEN_CONTRACT_SHA256",
    "HEAD_PARAMETER_COUNT",
    "HORIZONS",
    "MAINTENANCE_FIELD",
    "SCHEMA_VERSION",
    "TargetOrientedMaintenanceHeadConditioningContract",
    "calibrated_step_records",
    "classify_maintenance_head_steps",
    "load_target_oriented_maintenance_head_conditioning_contract",
    "maintenance_head",
    "replace_maintenance_head",
    "resolve_maintenance_head_index",
]
