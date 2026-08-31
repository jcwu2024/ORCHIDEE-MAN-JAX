"""Frozen statistics and classification for fresh structured-GPP attribution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, NamedTuple

import jax.numpy as jnp
import numpy as np

from .target_oriented_structured_gpp_fresh_screen import canonical_sha256

SCHEMA_VERSION = "target_oriented_structured_gpp_fresh_attribution_v1"
FROZEN_CONTRACT_SHA256 = "2c93dbe5d31791474c9c02bbf08efd76f831c1f7e99bb5c1458b4cd8f880112a"
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_structured_gpp_fresh_attribution_v1.json"
)


@dataclass(frozen=True)
class StructuredGPPFreshAttributionContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str


class ResidualDecompositionStatistics(NamedTuple):
    local_sum_squared: Any
    feedback_sum_squared: Any
    total_sum_squared: Any
    local_feedback_sum_product: Any
    local_sum_signed: Any
    feedback_sum_signed: Any
    total_sum_signed: Any
    count: Any
    maximum_absolute_closure_error: Any


def load_structured_gpp_fresh_attribution_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> StructuredGPPFreshAttributionContract:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    observed = canonical_sha256(payload, hash_key="contract_sha256")
    identity = payload.get("identity", {})
    data = payload.get("data", {})
    modes = payload.get("modes", {})
    execution = payload.get("execution", {})
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "frozen_before_attribution_execution"
        or payload.get("contract_sha256") != observed
        or observed != FROZEN_CONTRACT_SHA256
        or identity.get("fresh_screen_contract_sha256")
        != "894a0b2dc329179566e624e7c04b3a63522588aed34feb8dc00e3187eb73e6a7"
        or identity.get("fresh_formal_report_sha256")
        != "a9b9f6dcaf9aca349e738f096609e683342286d14d870d2c140afe2a3253798b"
        or identity.get("selected_checkpoint_sha256")
        != "3b173b8980c37b49db3d1547dddad2ab98ec00c733e2f564469490ee382d4ec0"
        or int(data.get("formal_window_count", -1)) != 96
        or int(data.get("admission_window_count", -1)) != 6
        or tuple(data.get("slices", ())) != ("temporal", "spatial", "joint")
        or tuple(data.get("horizons_days", ())) != (7, 30)
        or set(modes)
        != {
            "ordinary_free_rollout",
            "teacher_day_start_direct_gpp",
            "teacher_gpp_bridge",
        }
        or execution.get("free_gln01_allowed") is not True
        or execution.get("paid_execution_authorized") is not False
        or any(
            int(execution.get(name, -1)) != 0 for name in ("optimizer_updates", "parameter_steps", "checkpoint_writes")
        )
        or any(
            execution.get(name) is not False
            for name in (
                "teacher_data_regeneration_allowed",
                "historical_v5_allowed",
                "sealed_test_allowed",
                "architecture_edit_allowed",
                "loss_edit_allowed",
                "365_day_rollout_allowed",
                "gate_d2_allowed",
                "product_acceptance_allowed",
            )
        )
        or int(execution.get("forcing_records_reconstructed", -1)) != 0
        or int(execution.get("subdaily_state_scan_steps", -1)) != 0
    ):
        raise ValueError("structured GPP fresh-attribution contract drift")
    return StructuredGPPFreshAttributionContract(payload, source, observed)


def residual_decomposition_sufficient_statistics(
    expected: Any,
    teacher_state_predicted: Any,
    free_state_predicted: Any,
    defined: Any,
    scale: Any,
) -> ResidualDecompositionStatistics:
    expected = jnp.asarray(expected, dtype=jnp.float64)
    teacher = jnp.asarray(teacher_state_predicted, dtype=jnp.float64)
    free = jnp.asarray(free_state_predicted, dtype=jnp.float64)
    defined = jnp.asarray(defined, dtype=jnp.bool_)
    if teacher.shape != expected.shape or free.shape != expected.shape or defined.shape != expected.shape:
        raise ValueError("fresh-attribution residual shapes differ")
    scale = jnp.broadcast_to(jnp.asarray(scale, dtype=jnp.float64), expected.shape)
    valid = (
        defined
        & jnp.isfinite(expected)
        & jnp.isfinite(teacher)
        & jnp.isfinite(free)
        & jnp.isfinite(scale)
        & (scale > 0.0)
    )
    safe_scale = jnp.where(valid, scale, 1.0)
    local = jnp.where(valid, (teacher - expected) / safe_scale, 0.0)
    feedback = jnp.where(valid, (free - teacher) / safe_scale, 0.0)
    total = jnp.where(valid, (free - expected) / safe_scale, 0.0)
    closure = jnp.where(valid, jnp.abs(total - local - feedback), 0.0)
    return ResidualDecompositionStatistics(
        jnp.sum(local * local, dtype=jnp.float64),
        jnp.sum(feedback * feedback, dtype=jnp.float64),
        jnp.sum(total * total, dtype=jnp.float64),
        jnp.sum(local * feedback, dtype=jnp.float64),
        jnp.sum(local, dtype=jnp.float64),
        jnp.sum(feedback, dtype=jnp.float64),
        jnp.sum(total, dtype=jnp.float64),
        jnp.sum(valid, dtype=jnp.float64),
        jnp.max(closure),
    )


def add_residual_decomposition_statistics(
    left: ResidualDecompositionStatistics,
    right: ResidualDecompositionStatistics,
) -> ResidualDecompositionStatistics:
    return ResidualDecompositionStatistics(
        *(left[index] + right[index] for index in range(8)),
        jnp.maximum(left.maximum_absolute_closure_error, right.maximum_absolute_closure_error),
    )


def finalize_residual_decomposition_statistics(
    statistics: ResidualDecompositionStatistics,
) -> Mapping[str, Any]:
    values = np.asarray(statistics, dtype=np.float64)
    count = float(values[7])
    if count <= 0.0 or not np.all(np.isfinite(values)):
        raise ValueError("fresh-attribution residual statistics are empty or nonfinite")
    local_sq, feedback_sq, total_sq, product = (float(value) for value in values[:4])
    local_rms = float(np.sqrt(max(local_sq / count, 0.0)))
    feedback_rms = float(np.sqrt(max(feedback_sq / count, 0.0)))
    total_rms = float(np.sqrt(max(total_sq / count, 0.0)))
    denominator = float(np.sqrt(max(local_sq * feedback_sq, 0.0)))
    squared_closure = local_sq + feedback_sq + 2.0 * product - total_sq
    relative_squared_closure = abs(squared_closure) / max(total_sq, 1.0)
    return {
        "evaluated_values": count,
        "local": {
            "normalized_rms": local_rms,
            "normalized_signed_bias": float(values[4] / count),
        },
        "state_feedback": {
            "normalized_rms": feedback_rms,
            "normalized_signed_bias": float(values[5] / count),
            "rms_to_total_ratio": feedback_rms / total_rms if total_rms > 0.0 else 0.0,
        },
        "total": {
            "normalized_rms": total_rms,
            "normalized_signed_bias": float(values[6] / count),
        },
        "local_feedback_cosine": product / denominator if denominator > 0.0 else 0.0,
        "squared_error_cross_fraction": 2.0 * product / total_sq if total_sq > 0.0 else 0.0,
        "squared_decomposition_relative_error": relative_squared_closure,
        "maximum_absolute_residual_closure_error": float(values[8]),
    }


def acceptance_score(record: Mapping[str, Any]) -> float:
    rmse = float(record["normalized_rmse"])
    bias = float(record["normalized_signed_bias"])
    return 0.5 * (rmse * rmse + bias * bias)


def classify_fresh_attribution(
    direct_cells: Mapping[str, Mapping[str, Mapping[str, Any]]],
    decomposition_cells: Mapping[str, Mapping[str, Mapping[str, Any]]],
    free_cells: Mapping[str, Mapping[str, Mapping[str, Any]]],
    bridge_cells: Mapping[str, Mapping[str, Mapping[str, Any]]],
    contract: StructuredGPPFreshAttributionContract,
) -> Mapping[str, Any]:
    thresholds = contract.payload["classification"]
    failing_direct = 0
    material_feedback = 0
    reductions = {name: [] for name in contract.payload["measurements"]["causal_downstream_keys"]}
    bridge_pass = {name: 0 for name in reductions}
    for slice_id in ("temporal", "spatial", "joint"):
        for horizon in ("7", "30"):
            if not bool(direct_cells[slice_id][horizon]["passed"]):
                failing_direct += 1
            ratio = float(decomposition_cells[slice_id][horizon]["state_feedback"]["rms_to_total_ratio"])
            material_feedback += int(
                ratio >= float(thresholds["state_feedback_material_when_feedback_to_total_rms_ratio_minimum"])
            )
            for name in reductions:
                before = acceptance_score(free_cells[slice_id][horizon]["keys"][name])
                after = acceptance_score(bridge_cells[slice_id][horizon]["keys"][name])
                reductions[name].append(1.0 - after / before if before > 1.0e-15 else float("nan"))
                bridge_pass[name] += int(bridge_cells[slice_id][horizon]["keys"][name]["passed"])
    medians = {name: float(np.median(values)) for name, values in reductions.items()}
    finite = all(np.isfinite(value) for values in reductions.values() for value in values)
    sufficient = finite and all(count == 6 for count in bridge_pass.values())
    contributes = finite and all(
        value >= float(thresholds["teacher_gpp_contributes_when_median_downstream_score_reduction_minimum"])
        for value in medians.values()
    )
    direct_insufficient = failing_direct >= int(
        thresholds["direct_mapping_insufficient_when_teacher_state_cumulative_gpp_failing_cells_minimum"]
    )
    feedback_is_material = material_feedback >= int(thresholds["state_feedback_material_cell_count_minimum"])
    return {
        "teacher_state_cumulative_gpp_failing_cells": failing_direct,
        "direct_mapping_insufficient": direct_insufficient,
        "material_state_feedback_cells": material_feedback,
        "state_feedback_material": feedback_is_material,
        "teacher_gpp_bridge_downstream_passing_cells": bridge_pass,
        "teacher_gpp_bridge_median_score_reductions": medians,
        "teacher_gpp_sufficient_for_required_downstream": sufficient,
        "teacher_gpp_contributes_to_required_downstream": contributes,
        "classification": (
            "direct_and_feedback_gpp_with_sufficient_downstream_cause"
            if direct_insufficient and feedback_is_material and sufficient
            else "direct_gpp_with_sufficient_downstream_cause"
            if direct_insufficient and sufficient
            else "gpp_contributes_but_additional_scope_required"
            if contributes
            else "gpp_not_sufficient_and_additional_scope_required"
        ),
    }


__all__ = [
    "DEFAULT_CONTRACT_PATH",
    "FROZEN_CONTRACT_SHA256",
    "ResidualDecompositionStatistics",
    "SCHEMA_VERSION",
    "StructuredGPPFreshAttributionContract",
    "acceptance_score",
    "add_residual_decomposition_statistics",
    "classify_fresh_attribution",
    "finalize_residual_decomposition_statistics",
    "load_structured_gpp_fresh_attribution_contract",
    "residual_decomposition_sufficient_statistics",
]
