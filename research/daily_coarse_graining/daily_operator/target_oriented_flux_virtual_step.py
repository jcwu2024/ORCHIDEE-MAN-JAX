"""Frozen protected virtual-step contract for target-oriented flux training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .target_oriented_flux_basis import contract_sha256

SCHEMA_VERSION = "target_oriented_flux_protected_virtual_step_v1"
FROZEN_CONTRACT_SHA256 = "0781cd758a3c30c9558801beae451301cb66ad794a5ad8aaf47d8f3473a9053b"
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_flux_protected_virtual_step_v1.json"
)

AUDIT_UPDATES = (251, 358, 465, 572, 679, 786, 893, 1000)
OPTIMIZATION_SOURCES = (
    "seven_day_component_resolved_flux",
    "thirty_day_component_resolved_flux",
)
OPTIMIZED_FLUX_KEYS = (
    "gpp",
    "maintenance_respiration",
    "growth_respiration",
)
DERIVED_FLUX_KEYS = (
    "autotrophic_respiration",
    "npp",
    "nep",
    "nee",
)
PROTECTION_SOURCES = (
    "one_day_state_and_process",
    "seven_day_independent_stock",
    "thirty_day_independent_stock",
)
HORIZONS = (7, 30)


@dataclass(frozen=True)
class TargetOrientedFluxVirtualStepContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def step_norms(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.payload["step_norm_grid"])

    @property
    def gates(self) -> Mapping[str, Any]:
        return self.payload["gates"]


def _validate(payload: Mapping[str, Any]) -> None:
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "frozen_zero_update_flux_protected_virtual_step"
    ):
        raise ValueError("target-oriented protected virtual-step schema drift")
    identity = payload.get("identity", {})
    if (
        identity.get("geometry_report_sha256")
        != "414aa5433baeab85ef824de34ad93822a035e4ee4c61375cab09e83d05ff2a05"
        or identity.get("flux_basis_amendment_sha256")
        != "6c81e8f56c2c81fc2ac024a71da98f3e875c4ebc429f9e3d1ff4e8e3ef6bfb45"
        or identity.get("update_1000_checkpoint_sha256")
        != "a566f9d95726984dc03cbf1f1a183dd0d1eae9e3479ce5055c64e571ef6a16c7"
        or identity.get("prepared_manifest_sha256")
        != "8205140ddeae34967a287d41d204318659a84a8574de954e9b0be14f8998fc31"
        or identity.get("candidate")
        != "target_oriented_hierarchical_coordinate_attention_v1"
        or int(identity.get("parameter_count", 0)) != 83_755_173
    ):
        raise ValueError("target-oriented protected virtual-step identity drift")
    data = payload.get("data", {})
    if (
        data.get("release_id") != "pft14-daily-teacher-v6"
        or data.get("split") != "train/train"
        or tuple(data.get("audit_updates", ())) != AUDIT_UPDATES
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
        raise ValueError("target-oriented protected virtual-step data boundary drift")
    direction = payload.get("direction", {})
    if (
        tuple(direction.get("optimization_sources", ())) != OPTIMIZATION_SOURCES
        or int(direction.get("constraint_count", 0)) != 16
        or direction.get("normalization") != "unit_l2_per_group_horizon_gradient"
        or direction.get("combination") != "minimum_norm_simplex_common_descent"
        or float(direction.get("minimum_constraint_cosine", 0.0)) != 0.001
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
        raise ValueError("target-oriented protected virtual-step direction drift")
    step_norms = tuple(float(value) for value in payload.get("step_norm_grid", ()))
    if step_norms != (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1):
        raise ValueError("target-oriented protected virtual-step grid drift")
    flux = payload.get("flux_evaluation", {})
    if (
        tuple(flux.get("independent_optimized_keys", ())) != OPTIMIZED_FLUX_KEYS
        or flux.get("exact_retained_key") != "heterotrophic_respiration"
        or tuple(flux.get("derived_required_keys", ())) != DERIVED_FLUX_KEYS
        or tuple(flux.get("horizons", ())) != HORIZONS
        or tuple(payload.get("protection_sources", ())) != PROTECTION_SOURCES
    ):
        raise ValueError("target-oriented protected virtual-step target registry drift")
    execution = payload.get("execution", {})
    if (
        execution.get("free_gln01_only") is not True
        or execution.get("paid_training_authorized") is not False
        or int(execution.get("optimizer_updates", -1)) != 0
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
                "365_day_rollout_allowed",
                "gate_d2_allowed",
            )
        )
    ):
        raise ValueError("target-oriented protected virtual-step execution boundary drift")


def load_target_oriented_flux_virtual_step_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> TargetOrientedFluxVirtualStepContract:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    _validate(payload)
    observed = contract_sha256(payload)
    if payload.get("contract_sha256") != observed or observed != FROZEN_CONTRACT_SHA256:
        raise ValueError("target-oriented protected virtual-step contract is not frozen v1")
    return TargetOrientedFluxVirtualStepContract(payload, source, observed)


def classify_flux_protected_virtual_steps(
    direction: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    contract: TargetOrientedFluxVirtualStepContract,
) -> Mapping[str, Any]:
    """Select the largest direct step satisfying every frozen flux/protection gate."""

    gates = contract.gates
    minimum_cosine = float(direction["minimum_constraint_cosine"])
    direction_passed = minimum_cosine >= float(
        contract.payload["direction"]["minimum_constraint_cosine"]
    )
    if not direction_passed:
        return {
            "status": "common_flux_direction_not_supported",
            "passed": False,
            "direction_passed": False,
            "minimum_constraint_cosine": minimum_cosine,
            "selected_step_norm": None,
            "candidates": [],
            "next_action": contract.payload["fail_action"],
        }
    candidates = []
    for expected_norm, record in zip(contract.step_norms, records, strict=True):
        if float(record["step_norm"]) != expected_norm:
            raise ValueError("target-oriented protected virtual-step record order drift")
        groups = tuple(record["groups"])
        if tuple(int(group["update"]) for group in groups) != AUDIT_UPDATES:
            raise ValueError("target-oriented protected virtual-step group coverage drift")

        independent = {
            (horizon, key): np.asarray(
                [
                    float(group["independent_flux_relative_changes"][str(horizon)][key])
                    for group in groups
                ],
                dtype=np.float64,
            )
            for horizon in HORIZONS
            for key in OPTIMIZED_FLUX_KEYS
        }
        derived = {
            (horizon, key): np.asarray(
                [
                    float(group["derived_flux_relative_changes"][str(horizon)][key])
                    for group in groups
                ],
                dtype=np.float64,
            )
            for horizon in HORIZONS
            for key in DERIVED_FLUX_KEYS
        }
        one_day = np.asarray(
            [float(group["one_day_relative_change"]) for group in groups],
            dtype=np.float64,
        )
        stock_names = tuple(groups[0]["stock_relative_changes"][str(HORIZONS[0])])
        stock = {
            (horizon, key): np.asarray(
                [
                    float(group["stock_relative_changes"][str(horizon)][key])
                    for group in groups
                ],
                dtype=np.float64,
            )
            for horizon in HORIZONS
            for key in stock_names
        }
        finite = all(
            np.all(np.isfinite(values))
            for values in (*independent.values(), *derived.values(), one_day, *stock.values())
        )
        effect_limit = -float(
            gates["minimum_each_optimized_or_derived_key_median_relative_decrease"]
        )
        independent_medians = {
            f"{horizon}.{key}": float(np.median(values))
            for (horizon, key), values in independent.items()
        }
        derived_medians = {
            f"{horizon}.{key}": float(np.median(values))
            for (horizon, key), values in derived.items()
        }
        stock_medians = {
            f"{horizon}.{key}": float(np.median(values))
            for (horizon, key), values in stock.items()
        }
        maximum_independent = max(float(np.max(values)) for values in independent.values())
        maximum_derived = max(float(np.max(values)) for values in derived.values())
        maximum_stock = max(float(np.max(values)) for values in stock.values())
        acceptance_retained = all(
            bool(group["parent_passing_acceptance_keys_retained"])
            for group in groups
        )
        failing_stock_increase = max(
            float(group["parent_failing_stock_rmse_relative_increase_max"])
            for group in groups
        )
        hard = int(sum(int(group["hard_failure_count"]) for group in groups))
        nonfinite = int(record["nonfinite_value_count"]) + int(not finite)
        passed = (
            direction_passed
            and hard == int(gates["hard_failure_count"])
            and nonfinite == int(gates["nonfinite_value_count"])
            and maximum_independent
            <= float(gates["maximum_any_group_independent_flux_relative_increase"])
            and maximum_derived
            <= float(gates["maximum_any_group_derived_flux_relative_increase"])
            and all(value <= effect_limit for value in independent_medians.values())
            and all(value <= effect_limit for value in derived_medians.values())
            and float(np.max(one_day))
            <= float(gates["maximum_any_group_one_day_relative_increase"])
            and float(np.median(one_day))
            <= float(gates["maximum_one_day_median_relative_increase"])
            and maximum_stock <= float(gates["maximum_any_group_stock_relative_increase"])
            and all(
                value <= float(gates["maximum_stock_median_relative_increase"])
                for value in stock_medians.values()
            )
            and acceptance_retained
            and failing_stock_increase
            <= float(gates["maximum_parent_failing_stock_rmse_relative_increase"])
        )
        candidates.append(
            {
                "step_norm": expected_norm,
                "passed": bool(passed),
                "maximum_independent_flux_relative_increase": maximum_independent,
                "maximum_derived_flux_relative_increase": maximum_derived,
                "independent_flux_median_relative_changes": independent_medians,
                "derived_flux_median_relative_changes": derived_medians,
                "maximum_one_day_relative_increase": float(np.max(one_day)),
                "one_day_median_relative_change": float(np.median(one_day)),
                "maximum_stock_relative_increase": maximum_stock,
                "stock_median_relative_changes": stock_medians,
                "parent_passing_acceptance_keys_retained": acceptance_retained,
                "parent_failing_stock_rmse_relative_increase_max": failing_stock_increase,
                "hard_failure_count": hard,
                "nonfinite_value_count": nonfinite,
            }
        )

    admitted = tuple(candidate for candidate in candidates if candidate["passed"])
    selected = admitted[-1]["step_norm"] if admitted else None
    if selected is None:
        status = "finite_flux_step_not_protected"
    else:
        status = "protected_flux_virtual_step_supported"
    return {
        "status": status,
        "passed": selected is not None,
        "direction_passed": direction_passed,
        "minimum_constraint_cosine": minimum_cosine,
        "selected_step_norm": selected,
        "candidates": candidates,
        "next_action": (
            contract.payload["pass_action"]
            if selected is not None
            else contract.payload["fail_action"]
        ),
    }


__all__ = [
    "AUDIT_UPDATES",
    "DEFAULT_CONTRACT_PATH",
    "DERIVED_FLUX_KEYS",
    "FROZEN_CONTRACT_SHA256",
    "HORIZONS",
    "OPTIMIZATION_SOURCES",
    "OPTIMIZED_FLUX_KEYS",
    "PROTECTION_SOURCES",
    "SCHEMA_VERSION",
    "TargetOrientedFluxVirtualStepContract",
    "classify_flux_protected_virtual_steps",
    "load_target_oriented_flux_virtual_step_contract",
]
