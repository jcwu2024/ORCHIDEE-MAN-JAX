"""Target-blind broad-data schedule and preflight contract for structured GPP."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .target_oriented_source_shaped_maintenance import contract_sha256
from .target_oriented_structured_gpp_training import SPECIALIST_PARAMETER_COUNT

SCHEMA_VERSION = "target_oriented_structured_gpp_broad_preflight_v1"
FROZEN_CONTRACT_SHA256 = "f0e50031715b26898d581d3d99a17ae3c01a6a27e693f2e0130fb9e6c63096a3"
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "target_oriented_structured_gpp_broad_preflight_v1.json"
)

PointYearKey = tuple[str, int, str, str]
DayKey = tuple[str, int, str, str, int]


@dataclass(frozen=True)
class BroadTrainingSelection:
    reference: Any
    rows: tuple[int, ...]


@dataclass(frozen=True)
class BroadTrainingBatch:
    update: int
    selections: tuple[BroadTrainingSelection, ...]


@dataclass(frozen=True)
class BroadTrainingSchedule:
    batches: tuple[BroadTrainingBatch, ...]
    sha256: str
    unique_days: int
    unique_point_years: int
    unique_landpoints: int


@dataclass(frozen=True)
class StructuredGPPBroadPreflightContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str


def reference_identity(reference: Any) -> PointYearKey:
    return (
        str(reference.parent.landpoint_id),
        int(reference.parent.year),
        str(reference.parent.sha256),
        str(reference.sha256),
    )


def _rank(seed: int, domain: str, value: Any) -> str:
    payload = json.dumps(
        [int(seed), str(domain), value],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def build_broad_training_schedule(
    references: Sequence[Any],
    *,
    prior_exposed_days: set[DayKey],
    monitor_point_years: set[PointYearKey],
    seed: int,
    optimizer_updates: int,
    point_years_per_update: int,
    days_per_point_year: int,
) -> BroadTrainingSchedule:
    """Build deterministic unique-day batches without consulting target values."""

    updates = int(optimizer_updates)
    points_per_update = int(point_years_per_update)
    days_per_point = int(days_per_point_year)
    if updates <= 0 or points_per_update <= 0 or days_per_point <= 0:
        raise ValueError("structured GPP broad schedule dimensions must be positive")

    by_landpoint: dict[str, list[Any]] = {}
    row_orders: dict[PointYearKey, tuple[int, ...]] = {}
    for reference in references:
        identity = reference_identity(reference)
        if identity in monitor_point_years:
            continue
        rows = tuple(
            sorted(
                (
                    row
                    for row in range(1, int(reference.day_count))
                    if (*identity, row) not in prior_exposed_days
                ),
                key=lambda row: (_rank(seed, "day", [*identity, row]), row),
            )
        )
        if rows:
            by_landpoint.setdefault(identity[0], []).append(reference)
            row_orders[identity] = rows

    landpoints = tuple(
        sorted(by_landpoint, key=lambda value: (_rank(seed, "landpoint", value), value))
    )
    if len(landpoints) < points_per_update:
        raise ValueError("structured GPP broad schedule lacks distinct landpoints")
    for landpoint, values in by_landpoint.items():
        values.sort(
            key=lambda value: (
                _rank(seed, "point_year", reference_identity(value)),
                reference_identity(value),
            )
        )
        if not values:
            raise ValueError(f"structured GPP broad schedule lacks point-years for {landpoint}")

    point_year_cursors = {landpoint: 0 for landpoint in landpoints}
    row_cursors = {identity: 0 for identity in row_orders}
    batches = []
    used_point_years: set[PointYearKey] = set()
    digest = hashlib.sha256()
    for update in range(1, updates + 1):
        start = ((update - 1) * points_per_update) % len(landpoints)
        selected_landpoints = tuple(
            landpoints[(start + offset) % len(landpoints)]
            for offset in range(points_per_update)
        )
        if len(set(selected_landpoints)) != points_per_update:
            raise ValueError("structured GPP broad schedule repeated a landpoint within one update")
        selections = []
        canonical_entries = []
        for landpoint in selected_landpoints:
            candidates = by_landpoint[landpoint]
            reference = candidates[point_year_cursors[landpoint] % len(candidates)]
            point_year_cursors[landpoint] += 1
            identity = reference_identity(reference)
            cursor = row_cursors[identity]
            rows = row_orders[identity][cursor : cursor + days_per_point]
            if len(rows) != days_per_point:
                raise ValueError("structured GPP broad schedule exhausted a point-year")
            row_cursors[identity] += days_per_point
            selections.append(BroadTrainingSelection(reference=reference, rows=rows))
            used_point_years.add(identity)
            canonical_entries.append({"identity": list(identity), "rows": list(rows)})
        batch = BroadTrainingBatch(update=update, selections=tuple(selections))
        batches.append(batch)
        digest.update(
            json.dumps(
                {"update": update, "entries": canonical_entries},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        digest.update(b"\n")
    return BroadTrainingSchedule(
        batches=tuple(batches),
        sha256=digest.hexdigest(),
        unique_days=updates * points_per_update * days_per_point,
        unique_point_years=len(used_point_years),
        unique_landpoints=len(landpoints),
    )


def load_structured_gpp_broad_preflight_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> StructuredGPPBroadPreflightContract:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    observed = contract_sha256(payload)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("structured GPP broad-preflight schema drift")
    if (
        payload.get("status")
        != "frozen_before_target_blind_schedule_materialization_and_free_gpu_preflight"
        or payload.get("contract_sha256") != observed
        or observed != FROZEN_CONTRACT_SHA256
    ):
        raise ValueError("structured GPP broad-preflight contract is not frozen")

    identity = payload["identity"]
    data = payload["data"]
    optimizer = payload["optimizer_preflight"]
    execution = payload["execution"]
    if (
        identity["coverage_report_sha256"]
        != "a53330a621cd63cd863798caea012a1517ef8552205839cb6f7f93aafff0528b"
        or identity["throughput_report_sha256"]
        != "06731a560786a02dd2de5e323a3d3b395eaacc89e51f69ceb89a3a6ed0210761"
        or identity["selected_checkpoint_sha256"]
        != "77cb405fc7cbcea3a394041a141c7d5180bd7068eaa309ccdc33a3995d295f2b"
        or int(identity["selected_checkpoint_optimizer_step"]) != 498
        or int(identity["candidate_parameter_count"]) != 106_383_527
        or int(identity["trainable_specialist_parameter_count"])
        != SPECIALIST_PARAMETER_COUNT
        or int(identity["retained_parent_parameter_count"]) != 93_504_164
        or data["release_id"] != "pft14-daily-teacher-v6"
        or data["split"] != "train/train"
        or int(data["optimizer_updates_maximum"]) != 5000
        or int(data["point_years_per_update"]) != 52
        or int(data["days_per_point_year_per_update"]) != 4
        or int(data["effective_day_batch_size"]) != 208
        or int(data["unique_training_days"]) != 1_040_000
        or int(data["monitor"]["point_years"]) != 128
        or int(data["monitor"]["sampled_days"]) != 2048
        or data["monitor"]["reserve_whole_point_years"] is not True
        or data["exclude_prior_exposed_days"] is not True
        or data["distinct_landpoints_within_update"] is not True
        or data["selection_uses_target_values"] is not False
        or any(
            bool(data[name])
            for name in (
                "teacher_data_regeneration_allowed",
                "historical_v5_allowed",
                "validation_values_allowed",
                "sealed_test_allowed",
            )
        )
    ):
        raise ValueError("structured GPP broad-preflight identity or data drift")
    if (
        optimizer["name"] != "adamw"
        or optimizer["state"] != "fresh_zero_moments_at_selected_step_498_parameters"
        or float(optimizer["learning_rate"]) != 4e-6
        or int(optimizer["gradient_batches"]) != 16
        or int(optimizer["independent_finite_step_batches"]) != 4
        or optimizer["chained_training_updates"] is not False
    ):
        raise ValueError("structured GPP broad-preflight optimizer drift")
    if (
        execution["free_gln01_preflight_allowed"] is not True
        or execution["paid_training_authorized"] is not False
        or execution["optimizer_checkpoint_writes_allowed"] is not False
        or any(
            bool(execution[name])
            for name in (
                "teacher_data_regeneration_allowed",
                "historical_v5_allowed",
                "validation_read_authorized",
                "365_day_rollout_authorized",
                "gate_d2_authorized",
                "sealed_test_authorized",
                "product_acceptance_authorized",
                "cross_day_neural_hidden_state",
            )
        )
        or int(execution["forcing_records_reconstructed"]) != 0
        or int(execution["subdaily_state_scan_steps"]) != 0
    ):
        raise ValueError("structured GPP broad-preflight execution boundary drift")
    return StructuredGPPBroadPreflightContract(payload, source, observed)


__all__ = [
    "DEFAULT_CONTRACT_PATH",
    "FROZEN_CONTRACT_SHA256",
    "SCHEMA_VERSION",
    "BroadTrainingBatch",
    "BroadTrainingSchedule",
    "BroadTrainingSelection",
    "StructuredGPPBroadPreflightContract",
    "build_broad_training_schedule",
    "load_structured_gpp_broad_preflight_contract",
    "reference_identity",
]
