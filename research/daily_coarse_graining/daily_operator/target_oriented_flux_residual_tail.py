"""Frozen host analysis for the target-oriented flux residual-tail audit."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Hashable, Mapping, Sequence

import numpy as np

from .target_oriented_flux_basis import contract_sha256

SCHEMA_VERSION = "target_oriented_flux_residual_tail_audit_v1"
FROZEN_CONTRACT_SHA256 = "bafcb2daa0bcfc0e7108022b44d8af4741d87a1c5e73e371104e5985819bb763"
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_flux_residual_tail_audit_v1.json"
)
AUDIT_UPDATES = (251, 358, 465, 572, 679, 786, 893, 1000)
HORIZONS = (7, 30)
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


@dataclass(frozen=True)
class TargetOrientedFluxResidualTailContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def top_counts(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.payload["measurement"]["top_contribution_counts"])

    @property
    def scale_multipliers(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.payload["candidate_influence"]["scale_multipliers"])


def _validate(payload: Mapping[str, Any]) -> None:
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "frozen_before_zero_training_execution"
    ):
        raise ValueError("target-oriented residual-tail audit schema drift")
    identity = payload.get("identity", {})
    if (
        identity.get("protected_virtual_step_report_sha256")
        != "6a9ed9c93141ec1d40f582a6fcfe314b6ade2cc1c52432f36fe57145339f1fba"
        or identity.get("flux_basis_amendment_sha256")
        != "6c81e8f56c2c81fc2ac024a71da98f3e875c4ebc429f9e3d1ff4e8e3ef6bfb45"
        or identity.get("update_1000_checkpoint_sha256")
        != "a566f9d95726984dc03cbf1f1a183dd0d1eae9e3479ce5055c64e571ef6a16c7"
        or identity.get("prepared_manifest_sha256")
        != "8205140ddeae34967a287d41d204318659a84a8574de954e9b0be14f8998fc31"
        or int(identity.get("parameter_count", 0)) != 83_755_173
    ):
        raise ValueError("target-oriented residual-tail audit identity drift")
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
        raise ValueError("target-oriented residual-tail audit data boundary drift")
    measurement = payload.get("measurement", {})
    if (
        tuple(measurement.get("horizons", ())) != HORIZONS
        or tuple(measurement.get("optimized_flux_keys", ())) != OPTIMIZED_FLUX_KEYS
        or tuple(measurement.get("derived_report_only", ())) != DERIVED_FLUX_KEYS
        or int(measurement.get("gradient_coverage", 0)) != 83_755_173
    ):
        raise ValueError("target-oriented residual-tail audit measurement drift")
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
                "365_day_rollout_allowed",
                "gate_d2_allowed",
            )
        )
    ):
        raise ValueError("target-oriented residual-tail audit execution boundary drift")


def load_target_oriented_flux_residual_tail_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> TargetOrientedFluxResidualTailContract:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    _validate(payload)
    observed = contract_sha256(payload)
    if payload.get("contract_sha256") != observed or observed != FROZEN_CONTRACT_SHA256:
        raise ValueError("target-oriented residual-tail audit is not frozen v1")
    return TargetOrientedFluxResidualTailContract(payload, source, observed)


def contribution_summary(
    values: Sequence[float],
    *,
    top_counts: Sequence[int] = (1, 5, 10),
) -> Mapping[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("contribution summary requires a nonempty vector")
    if np.any(~np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError("contribution summary values must be finite and nonnegative")
    total = float(np.sum(array, dtype=np.float64))
    ordered = np.sort(array)[::-1]
    if total > 0.0:
        top_shares = {
            str(int(count)): float(np.sum(ordered[: min(int(count), array.size)]) / total) for count in top_counts
        }
        denominator = float(np.sum(array * array, dtype=np.float64))
        effective = total * total / denominator if denominator > 0.0 else float(array.size)
    else:
        top_shares = {str(int(count)): 0.0 for count in top_counts}
        effective = float(array.size)
    positive = array[array > 0.0]
    return {
        "count": int(array.size),
        "total": total,
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "positive_median": float(np.median(positive)) if positive.size else 0.0,
        "p90": float(np.quantile(array, 0.9)),
        "maximum": float(np.max(array)),
        "top_shares": top_shares,
        "effective_sample_size": float(effective),
        "effective_sample_fraction": float(effective / array.size),
    }


def bounded_influence_weights(
    scores: Sequence[float],
    *,
    family: str,
    tau: float | None = None,
) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("bounded influence requires a nonempty score vector")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("bounded influence scores must be finite and nonnegative")
    if family == "equal_window":
        raw = np.ones_like(values)
    else:
        resolved_tau = float(tau) if tau is not None else 0.0
        if not np.isfinite(resolved_tau) or resolved_tau <= 0.0:
            raise ValueError("bounded influence tau must be finite and positive")
        ratio = values / resolved_tau
        if family == "soft_l1":
            raw = 1.0 / np.sqrt(1.0 + ratio)
        elif family == "log1p":
            raw = 1.0 / (1.0 + ratio)
        else:
            raise ValueError(f"unsupported bounded influence family: {family}")
    total = float(np.sum(raw, dtype=np.float64))
    if not np.isfinite(total) or total <= 0.0:
        raise FloatingPointError("bounded influence weights do not have positive mass")
    return raw / total


def influence_candidates(
    scores: Sequence[float],
    contract: TargetOrientedFluxResidualTailContract,
) -> Mapping[str, Mapping[str, Any]]:
    summary = contribution_summary(scores, top_counts=contract.top_counts)
    median = float(summary["positive_median"])
    if median <= 0.0:
        median = 1.0
    result: dict[str, Mapping[str, Any]] = {
        "equal_window": {
            "family": "equal_window",
            "tau": None,
            "weights": bounded_influence_weights(scores, family="equal_window"),
        }
    }
    for family in ("soft_l1", "log1p"):
        for multiplier in contract.scale_multipliers:
            tau = median * multiplier
            name = f"{family}_x{multiplier:g}"
            result[name] = {
                "family": family,
                "tau": tau,
                "weights": bounded_influence_weights(scores, family=family, tau=tau),
            }
    return result


def gram_direction_summary(gram: Any, weights: Sequence[float]) -> Mapping[str, Any]:
    matrix = np.asarray(gram, dtype=np.float64)
    coefficients = np.asarray(weights, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or coefficients.shape != (matrix.shape[0],):
        raise ValueError("Gram direction dimensions differ")
    if np.any(~np.isfinite(matrix)) or np.any(~np.isfinite(coefficients)):
        raise ValueError("Gram direction inputs must be finite")
    total = float(np.sum(coefficients, dtype=np.float64))
    if total <= 0.0 or np.any(coefficients < 0.0):
        raise ValueError("Gram direction weights must be nonnegative with positive mass")
    coefficients = coefficients / total
    diagonal = np.maximum(np.diag(matrix), 0.0)
    norms = np.sqrt(diagonal)
    direction_squared = float(coefficients @ matrix @ coefficients)
    direction_norm = float(np.sqrt(max(direction_squared, 0.0)))
    if direction_norm <= 0.0 or np.any(norms <= 0.0):
        raise FloatingPointError("Gram direction contains a zero gradient")
    cosines = matrix @ coefficients / norms / direction_norm
    return {
        "weights": coefficients.tolist(),
        "direction_norm": direction_norm,
        "constraint_cosines": cosines.tolist(),
        "minimum_constraint_cosine": float(np.min(cosines)),
        "median_constraint_cosine": float(np.median(cosines)),
        "maximum_constraint_cosine": float(np.max(cosines)),
        "negative_constraint_fraction": float(np.mean(cosines < 0.0)),
    }


def classify_tail_mechanism(
    losses: Sequence[float],
    gradient_norms: Sequence[float],
    group_labels: Sequence[Hashable],
    contract: TargetOrientedFluxResidualTailContract,
) -> Mapping[str, Any]:
    loss = np.asarray(losses, dtype=np.float64)
    gradient = np.asarray(gradient_norms, dtype=np.float64)
    labels = np.asarray(group_labels, dtype=object)
    if loss.shape != gradient.shape or labels.shape != loss.shape:
        raise ValueError("tail classification vectors differ")
    loss_summary = contribution_summary(loss, top_counts=contract.top_counts)
    gradient_summary = contribution_summary(gradient, top_counts=contract.top_counts)
    gates = contract.payload["classification"]
    localized_loss = bool(
        float(loss_summary["top_shares"]["1"]) >= float(gates["localized_loss_top1_share_minimum"])
        or float(loss_summary["top_shares"]["5"]) >= float(gates["localized_loss_top5_share_minimum"])
        or float(loss_summary["effective_sample_fraction"]) <= float(gates["localized_loss_ess_fraction_maximum"])
    )
    gradient_tail = bool(
        float(gradient_summary["top_shares"]["1"]) >= float(gates["gradient_tail_top1_norm_share_minimum"])
        or float(gradient_summary["effective_sample_fraction"]) <= float(gates["gradient_tail_ess_fraction_maximum"])
    )

    total_loss = float(np.sum(loss, dtype=np.float64))
    global_median = float(np.median(loss))
    broad_groups = []
    for label in dict.fromkeys(group_labels):
        active = labels == label
        local = loss[active]
        local_total = float(np.sum(local, dtype=np.float64))
        share = local_total / total_loss if total_loss > 0.0 else 0.0
        top1 = float(np.max(local) / local_total) if local_total > 0.0 else 0.0
        above = float(np.mean(local > global_median))
        if (
            share >= float(gates["broad_group_share_minimum"])
            and top1 <= float(gates["broad_group_top1_within_group_maximum"])
            and above >= float(gates["broad_group_above_global_median_fraction_minimum"])
        ):
            broad_groups.append(
                {
                    "group": str(label),
                    "loss_share": share,
                    "top1_within_group_share": top1,
                    "above_global_median_fraction": above,
                }
            )

    gradient_order = np.argsort(-gradient, kind="stable")
    loss_order = np.argsort(-loss, kind="stable")
    gradient_rank = np.empty_like(gradient_order)
    loss_rank = np.empty_like(loss_order)
    gradient_rank[gradient_order] = np.arange(loss.size)
    loss_rank[loss_order] = np.arange(loss.size)
    gradient_rank_fraction = (gradient_rank + 1) / loss.size
    loss_rank_fraction = (loss_rank + 1) / loss.size
    amplified = np.flatnonzero(
        (gradient_rank_fraction <= float(gates["jacobian_amplification_gradient_rank_fraction_maximum"]))
        & (loss_rank_fraction >= float(gates["jacobian_amplification_loss_rank_fraction_minimum"]))
    )
    return {
        "loss_concentration": loss_summary,
        "gradient_norm_concentration": gradient_summary,
        "localized_loss_tail": localized_loss,
        "gradient_norm_tail": gradient_tail,
        "broad_groups": broad_groups,
        "jacobian_amplified_window_indices": amplified.astype(int).tolist(),
        "loss_rank_fractions": loss_rank_fraction.tolist(),
        "gradient_norm_rank_fractions": gradient_rank_fraction.tolist(),
    }


__all__ = [
    "AUDIT_UPDATES",
    "DEFAULT_CONTRACT_PATH",
    "DERIVED_FLUX_KEYS",
    "FROZEN_CONTRACT_SHA256",
    "HORIZONS",
    "OPTIMIZED_FLUX_KEYS",
    "SCHEMA_VERSION",
    "TargetOrientedFluxResidualTailContract",
    "bounded_influence_weights",
    "classify_tail_mechanism",
    "contribution_summary",
    "gram_direction_summary",
    "influence_candidates",
    "load_target_oriented_flux_residual_tail_contract",
]
