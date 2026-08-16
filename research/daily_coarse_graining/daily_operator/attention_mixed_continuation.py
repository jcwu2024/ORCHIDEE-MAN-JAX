"""Attention-only mixed-point continuation policy and checkpoint state."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .successor_fit_checkpoint import _validate_finite_tree
from .successor_staged_training import (
    REQUIRED_VALIDATION_HORIZONS,
    REQUIRED_VALIDATION_SLICES,
    validation_hard_failure_count,
)

SCHEMA_VERSION = "successor_attention_mixed_annealed_continuation_v1"
STATE_SCHEMA_VERSION = "successor_attention_mixed_continuation_state_v1"
FROZEN_CONTRACT_SHA256 = "a6708373e4bb9066ad1795d88e5e58a3cfc764a34d5a8532307c02a2bd696ebc"
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "successor_attention_mixed_annealed_continuation_v1.json"
)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("contract_sha256", None)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class AttentionMixedContinuationContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def training(self) -> Mapping[str, Any]:
        return self.payload["training"]

    @property
    def gates(self) -> Mapping[str, Any]:
        return self.payload["gates"]


def load_attention_mixed_continuation_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> AttentionMixedContinuationContract:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    observed = _canonical_sha256(payload)
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "frozen_for_attention_candidate_training"
        or payload.get("contract_sha256") != observed
        or observed != FROZEN_CONTRACT_SHA256
    ):
        raise ValueError("attention mixed-continuation contract drift")
    training = payload.get("training", {})
    data = payload.get("data", {})
    if (
        int(training.get("continuation_updates", 0)) != 250
        or int(training.get("point_years_per_update", 0)) != 4
        or int(training.get("one_day_batch_size", 0)) != 32
        or int(training.get("rollout_batch_size", 0)) != 4
        or int(training.get("rollout_days", 0)) != 30
        or tuple(training.get("objective_horizons_days", ())) != (1, 7, 30)
        or training.get("schedule") != "compressed_cosine_decay"
        or data.get("teacher_data_regenerated") is not False
        or data.get("historical_v5_used") is not False
        or data.get("sealed_test_used") is not False
    ):
        raise ValueError("attention mixed-continuation training policy drift")
    return AttentionMixedContinuationContract(payload, source, observed)


def continuation_learning_rate(
    contract: AttentionMixedContinuationContract,
    continuation_update: int,
    *,
    initial_learning_rate: float,
) -> float:
    total = int(contract.training["continuation_updates"])
    update = int(continuation_update)
    if not 1 <= update <= total:
        raise ValueError("attention continuation update lies outside its budget")
    initial = float(initial_learning_rate)
    final = float(contract.training["final_learning_rate"])
    if not np.isfinite(initial) or initial <= final or final <= 0.0:
        raise ValueError("attention continuation learning-rate endpoints are invalid")
    progress = (update - 1) / (total - 1)
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return float(final + (initial - final) * cosine)


def mixed_update_groups(
    entries: Sequence[Mapping[str, Any]],
    *,
    point_years_per_update: int,
    expected_updates: int,
) -> tuple[tuple[Mapping[str, Any], ...], ...]:
    width = int(point_years_per_update)
    count = int(expected_updates)
    if width <= 1 or len(entries) != width * count:
        raise ValueError("attention continuation prepared coverage drift")
    groups = []
    seen: set[tuple[str, int]] = set()
    for offset in range(0, len(entries), width):
        group = tuple(entries[offset : offset + width])
        identities = {
            (str(entry["landpoint_id"]), int(entry["year"])) for entry in group
        }
        if len(identities) != width or seen.intersection(identities):
            raise ValueError("attention continuation repeated a point-year")
        seen.update(identities)
        groups.append(group)
    return tuple(groups)


def _rms(values: Sequence[float]) -> float:
    result = float(np.sqrt(np.mean(np.square(np.asarray(values, dtype=np.float64)))))
    if not np.isfinite(result):
        raise ValueError("attention continuation validation metric is nonfinite")
    return result


def classify_attention_continuation(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    contract: AttentionMixedContinuationContract,
) -> Mapping[str, Any]:
    keys = tuple(contract.payload["validation"]["research_primary_keys"])
    maintenance_key = str(contract.payload["validation"]["maintenance_key"])
    cells = tuple(
        (slice_id, str(horizon))
        for slice_id in REQUIRED_VALIDATION_SLICES
        for horizon in REQUIRED_VALIDATION_HORIZONS
    )
    key_aggregates = {}
    for key in keys:
        before = _rms(
            [
                baseline["slices"][slice_id]["horizons"][horizon]["keys"][key][
                    "normalized_rmse"
                ]
                for slice_id, horizon in cells
            ]
        )
        after = _rms(
            [
                candidate["slices"][slice_id]["horizons"][horizon]["keys"][key][
                    "normalized_rmse"
                ]
                for slice_id, horizon in cells
            ]
        )
        key_aggregates[key] = {
            "baseline_rms": before,
            "candidate_rms": after,
            "relative_change": after / before - 1.0,
            "no_worse": after <= before,
        }

    baseline_values = []
    candidate_values = []
    cell_records = []
    for slice_id, horizon in cells:
        before_metrics = baseline["slices"][slice_id]["horizons"][horizon]["keys"]
        after_metrics = candidate["slices"][slice_id]["horizons"][horizon]["keys"]
        before = _rms([before_metrics[key]["normalized_rmse"] for key in keys])
        after = _rms([after_metrics[key]["normalized_rmse"] for key in keys])
        baseline_values.extend(before_metrics[key]["normalized_rmse"] for key in keys)
        candidate_values.extend(after_metrics[key]["normalized_rmse"] for key in keys)
        cell_records.append(
            {
                "slice": slice_id,
                "horizon_days": int(horizon),
                "baseline_composite": before,
                "candidate_composite": after,
                "relative_change": after / before - 1.0,
                "improved": after < before,
            }
        )
    baseline_composite = _rms(baseline_values)
    candidate_composite = _rms(candidate_values)

    maintenance_cells = []
    for slice_id, horizon in cells:
        before = float(
            baseline["slices"][slice_id]["horizons"][horizon]["keys"][
                maintenance_key
            ]["normalized_rmse"]
        )
        after = float(
            candidate["slices"][slice_id]["horizons"][horizon]["keys"][
                maintenance_key
            ]["normalized_rmse"]
        )
        maintenance_cells.append(
            {
                "slice": slice_id,
                "horizon_days": int(horizon),
                "baseline_rmse": before,
                "candidate_rmse": after,
                "relative_change": after / before - 1.0,
            }
        )
    maintenance_before = _rms([item["baseline_rmse"] for item in maintenance_cells])
    maintenance_after = _rms([item["candidate_rmse"] for item in maintenance_cells])
    gates = contract.gates
    hard_failures = validation_hard_failure_count(candidate)
    checks = {
        "hard_failures_zero": hard_failures == int(gates["hard_failure_count"]),
        "research_primary_composite_improved": 1.0
        - candidate_composite / baseline_composite
        >= float(gates["minimum_research_primary_composite_relative_improvement"]),
        "each_research_primary_key_no_worse": all(
            value["no_worse"] for value in key_aggregates.values()
        ),
        "cell_majority_improved": sum(item["improved"] for item in cell_records)
        >= int(gates["minimum_improved_slice_horizon_cells"]),
        "maintenance_aggregate_within_limit": maintenance_after
        <= maintenance_before
        * (1.0 + float(gates["maximum_maintenance_aggregate_relative_regression"])),
        "maintenance_cells_within_limit": all(
            item["candidate_rmse"]
            <= item["baseline_rmse"]
            * (1.0 + float(gates["maximum_maintenance_single_cell_relative_regression"]))
            for item in maintenance_cells
        ),
    }
    passed = all(checks.values())
    return {
        "status": (
            "attention_mixed_continuation_admitted"
            if passed
            else "attention_mixed_continuation_not_admitted"
        ),
        "passed": passed,
        "checks": checks,
        "hard_failure_count": hard_failures,
        "research_primary": {
            "baseline_composite": baseline_composite,
            "candidate_composite": candidate_composite,
            "relative_improvement": 1.0 - candidate_composite / baseline_composite,
            "key_aggregates": key_aggregates,
            "cells": cell_records,
            "improved_cell_count": sum(item["improved"] for item in cell_records),
        },
        "maintenance": {
            "baseline_rms": maintenance_before,
            "candidate_rms": maintenance_after,
            "relative_change": maintenance_after / maintenance_before - 1.0,
            "cells": maintenance_cells,
        },
    }


def write_attention_continuation_state(
    path: str | Path,
    *,
    identity: Mapping[str, Any],
    parameters: Any,
    optimizer: Any,
    completed_updates: int,
    parameter_count: int,
    classification: Mapping[str, Any] | None = None,
    compute_sha256: bool = True,
) -> Mapping[str, Any]:
    completed = int(completed_updates)
    if completed <= 0:
        raise ValueError("attention continuation state requires a positive update count")
    observed = _validate_finite_tree(parameters, label="attention continuation parameter")
    _validate_finite_tree(optimizer, label="attention continuation optimizer")
    if observed != int(parameter_count):
        raise ValueError("attention continuation parameter count drift")
    payload = {
        "schema_version": STATE_SCHEMA_VERSION,
        "identity": dict(identity),
        "architecture": "source_routed_process_attention_v2",
        "completed_updates": completed,
        "parameter_count": observed,
        "promotion_eligible": bool(classification and classification.get("passed")),
        "classification": None if classification is None else dict(classification),
        "parameters": parameters,
        "optimizer": optimizer,
    }
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {
        "path": str(target),
        "sha256": sha256_file(target) if compute_sha256 else None,
        "completed_updates": completed,
        "promotion_eligible": payload["promotion_eligible"],
    }


def load_attention_continuation_state(
    path: str | Path,
    *,
    identity: Mapping[str, Any],
    parameter_count: int,
    maximum_updates: int,
) -> Mapping[str, Any] | None:
    source = Path(path).resolve()
    if not source.exists():
        return None
    with source.open("rb") as handle:
        payload = pickle.load(handle)
    if (
        payload.get("schema_version") != STATE_SCHEMA_VERSION
        or payload.get("identity") != dict(identity)
        or payload.get("architecture") != "source_routed_process_attention_v2"
        or int(payload.get("parameter_count", -1)) != int(parameter_count)
        or not 0 < int(payload.get("completed_updates", 0)) <= int(maximum_updates)
    ):
        raise ValueError("attention continuation state identity drift")
    if (
        _validate_finite_tree(payload["parameters"], label="attention continuation parameter")
        != int(parameter_count)
    ):
        raise ValueError("attention continuation state parameter count drift")
    _validate_finite_tree(payload["optimizer"], label="attention continuation optimizer")
    return payload
