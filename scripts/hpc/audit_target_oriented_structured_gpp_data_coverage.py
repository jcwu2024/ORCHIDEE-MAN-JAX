"""Audit exact direct-GPP exposure without reading Teacher target values."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file

SCHEMA_VERSION = "target_oriented_structured_gpp_data_coverage_audit_v1"
HORIZONS = {"one_day": 1, "seven_day": 7, "thirty_day": 30}
MONITOR_POINT_YEARS = 128
MONITOR_DAYS_PER_POINT_YEAR = 16
MONITOR_SEED = 20260830

DayKey = tuple[str, int, str, str, int]
PointYearKey = tuple[str, int, str, str]


def _direct_day_keys(entry: Mapping[str, Any]) -> Mapping[str, set[DayKey]]:
    result = {name: set() for name in HORIZONS}
    for component, horizon in HORIZONS.items():
        for record in entry["components"][component]:
            identity: PointYearKey = (
                str(record["landpoint_id"]),
                int(record["year"]),
                str(record["parent_shard_sha256"]),
                str(record["typed_shard_sha256"]),
            )
            for start in record["start_rows"]:
                for offset in range(horizon):
                    result[component].add((*identity, int(start) + offset))
    return result


def _merged_day_keys(entries: Iterable[Mapping[str, Any]]) -> Mapping[str, set[DayKey]]:
    result = {name: set() for name in HORIZONS}
    for entry in entries:
        local = _direct_day_keys(entry)
        for component in HORIZONS:
            result[component].update(local[component])
    result["all"] = set().union(*result.values())
    return result


def _summary(keys: set[DayKey]) -> Mapping[str, int]:
    return {
        "unique_days": len(keys),
        "unique_point_years": len({key[:4] for key in keys}),
        "unique_landpoints": len({key[0] for key in keys}),
        "unique_years": len({key[1] for key in keys}),
    }


def _rank(seed: int, value: Any) -> str:
    encoded = json.dumps([int(seed), value], sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _reference_identity(reference: Any) -> PointYearKey:
    return (
        str(reference.parent.landpoint_id),
        int(reference.parent.year),
        str(reference.parent.sha256),
        str(reference.sha256),
    )


def _select_monitor_entries(
    references: Sequence[Any],
    *,
    exposed_point_years: set[PointYearKey],
    point_year_count: int = MONITOR_POINT_YEARS,
    days_per_point_year: int = MONITOR_DAYS_PER_POINT_YEAR,
    seed: int = MONITOR_SEED,
) -> list[Mapping[str, Any]]:
    eligible = [
        reference
        for reference in references
        if _reference_identity(reference) not in exposed_point_years
        and int(reference.day_count) > days_per_point_year
    ]
    eligible.sort(key=lambda value: (_rank(seed, _reference_identity(value)), _reference_identity(value)))
    selected = []
    seen_landpoints = set()
    for reference in eligible:
        identity = _reference_identity(reference)
        if identity[0] in seen_landpoints:
            continue
        rows = sorted(
            sorted(
                range(1, int(reference.day_count)),
                key=lambda row: (_rank(seed, [*identity, row]), row),
            )[:days_per_point_year]
        )
        selected.append(
            {
                "landpoint_id": identity[0],
                "year": identity[1],
                "parent_shard_sha256": identity[2],
                "typed_shard_sha256": identity[3],
                "day_count": int(reference.day_count),
                "monitor_rows": rows,
            }
        )
        seen_landpoints.add(identity[0])
        if len(selected) == point_year_count:
            break
    if len(selected) != point_year_count:
        raise ValueError("structured GPP coverage audit cannot select the monitor point-years")
    return selected


def _contract_updates(warm: Any, joint: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    warm_settings = warm.payload["data"]["training_updates"]
    excluded = {int(value) for value in warm_settings["excluded_artifact_updates"]}
    warm_updates = tuple(
        value
        for value in range(
            int(warm_settings["artifact_candidate_start"]),
            int(warm_settings["artifact_candidate_end"]) + 1,
        )
        if value not in excluded
    )
    joint_data = joint.payload["data"]
    joint_excluded = {int(value) for value in joint_data["excluded_evaluation_artifacts"]}
    joint_updates = tuple(
        value
        for value in range(
            int(joint_data["artifact_candidate_start"]),
            int(joint_data["artifact_candidate_end"]) + 1,
        )
        if value not in joint_excluded
    )
    if len(warm_updates) != 250 or len(joint_updates) != 250 or set(warm_updates) & set(joint_updates):
        raise ValueError("structured GPP coverage audit training ranges drift")
    return warm_updates, joint_updates


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    started = time.perf_counter()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)

    from research.daily_coarse_graining.daily_markov_contract import (
        daily_markov_contract_from_metadata,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_joint_training import (
        build_joint_training_schedule,
        load_structured_gpp_joint_training_contract,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_training import (
        load_structured_gpp_warm_start_contract,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_training import (
        build_target_oriented_training_schedule,
        load_target_oriented_training_contract,
    )
    from research.daily_coarse_graining.typed_sidecar import (
        load_coherent_typed_sidecar_index,
    )

    dataset_manifest = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    dataset_sha256 = sha256_file(args.dataset_manifest)
    training = load_target_oriented_training_contract(args.training_contract)
    warm = load_structured_gpp_warm_start_contract(args.warm_start_contract)
    joint = load_structured_gpp_joint_training_contract(args.joint_training_contract)
    prepared = json.loads(args.prepared_manifest.read_text(encoding="utf-8"))
    prepared_sha256 = sha256_file(args.prepared_manifest)
    if (
        dataset_sha256 != training.payload["identity"]["dataset_manifest_sha256"]
        or prepared_sha256 != warm.payload["identity"]["prepared_manifest_sha256"]
        or prepared_sha256 != joint.payload["identity"]["prepared_manifest_sha256"]
        or prepared.get("identity", {}).get("training_contract_sha256") != training.sha256
        or prepared.get("identity", {}).get("dataset_manifest_sha256") != dataset_sha256
        or prepared.get("teacher_data_regenerated") is not False
        or prepared.get("historical_v5_used") is not False
        or prepared.get("sealed_test_read") is not False
    ):
        raise ValueError("structured GPP coverage audit data identity drift")

    contract = daily_markov_contract_from_metadata(dataset_manifest["markov_contract"])
    index = load_coherent_typed_sidecar_index(args.dataset_manifest, verify_hashes=False)
    references = tuple(index.select(spatial_split="train", temporal_split="train"))
    schedule = build_target_oriented_training_schedule(references, training, updates=1000)
    if schedule["schedule_sha256"] != prepared["identity"]["schedule_sha256"]:
        raise ValueError("structured GPP coverage audit schedule drift")
    entries = {int(entry["update"]): entry for entry in schedule["entries"]}
    warm_updates, joint_updates = _contract_updates(warm, joint)
    warm_keys = _merged_day_keys(entries[value] for value in warm_updates)
    joint_keys = _merged_day_keys(entries[value] for value in joint_updates)
    union = warm_keys["all"] | joint_keys["all"]
    overlap = warm_keys["all"] & joint_keys["all"]

    joint_schedule = build_joint_training_schedule()
    joint_presentations = tuple(value for group in joint_schedule.flat_groups for value in group)
    if set(joint_presentations) != set(joint_updates) or len(joint_presentations) != 992:
        raise ValueError("structured GPP coverage audit materialized joint schedule drift")
    direct_days_per_artifact = int(warm.payload["data"]["direct_gpp_teacher_days_per_artifact"])
    warm_presentations = len(warm_updates) * direct_days_per_artifact
    joint_day_presentations = len(joint_presentations) * direct_days_per_artifact

    accepted_train_days = sum(int(reference.day_count) for reference in references)
    eligible_train_days = sum(max(int(reference.day_count) - 1, 0) for reference in references)
    exposed_point_years = {key[:4] for key in union}
    monitor = _select_monitor_entries(
        references,
        exposed_point_years=exposed_point_years,
    )
    monitor_point_years = {
        (
            value["landpoint_id"],
            int(value["year"]),
            value["parent_shard_sha256"],
            value["typed_shard_sha256"],
        )
        for value in monitor
    }
    monitor_keys = {
        (
            value["landpoint_id"],
            int(value["year"]),
            value["parent_shard_sha256"],
            value["typed_shard_sha256"],
            int(row),
        )
        for value in monitor
        for row in value["monitor_rows"]
    }
    if monitor_point_years & exposed_point_years or monitor_keys & union:
        raise ValueError("structured GPP coverage audit monitor overlap")
    reserved_eligible_days = sum(int(value["day_count"]) - 1 for value in monitor)

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "identity": {
            "dataset_manifest_sha256": dataset_sha256,
            "markov_contract_sha256": contract.sha256,
            "training_contract_sha256": training.sha256,
            "warm_start_contract_sha256": warm.sha256,
            "joint_training_contract_sha256": joint.sha256,
            "prepared_manifest_sha256": prepared_sha256,
            "schedule_sha256": schedule["schedule_sha256"],
            "materialized_joint_schedule_sha256": joint_schedule.sha256,
        },
        "available_train_train": {
            "accepted_transitions": accepted_train_days,
            "eligible_after_first_row_exclusion": eligible_train_days,
            "point_years": len(references),
            "landpoints": len({str(value.parent.landpoint_id) for value in references}),
            "years": len({int(value.parent.year) for value in references}),
        },
        "exposure": {
            "warm_start": _summary(warm_keys["all"])
            | {"artifact_count": len(warm_updates), "day_presentations": warm_presentations},
            "joint_unique_artifacts": _summary(joint_keys["all"])
            | {
                "artifact_count": len(joint_updates),
                "artifact_presentations": len(joint_presentations),
                "day_presentations": joint_day_presentations,
            },
            "combined": _summary(union)
            | {
                "cross_stage_duplicate_days": len(overlap),
                "day_presentations": warm_presentations + joint_day_presentations,
                "fraction_of_accepted_train_train": len(union) / accepted_train_days,
                "fraction_of_eligible_train_train": len(union) / eligible_train_days,
            },
            "by_component_combined": {
                component: _summary(warm_keys[component] | joint_keys[component])
                for component in HORIZONS
            },
        },
        "proposed_monitor": {
            "selection_uses_target_values": False,
            "split": "train/train",
            "seed": MONITOR_SEED,
            "point_years": len(monitor),
            "distinct_landpoints": len({value["landpoint_id"] for value in monitor}),
            "sampled_days": len(monitor_keys),
            "days_per_point_year": MONITOR_DAYS_PER_POINT_YEAR,
            "reserved_eligible_days": reserved_eligible_days,
            "disjoint_from_prior_exposure": True,
            "entries": monitor,
        },
        "policy": {
            "teacher_target_values_read": False,
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "validation_or_sealed_test_read": False,
            "optimizer_updates": 0,
            "checkpoint_writes": 0,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": "measure_broad_direct_gpp_streaming_batch_throughput",
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--training-contract", type=Path, required=True)
    parser.add_argument("--warm-start-contract", type=Path, required=True)
    parser.add_argument("--joint-training-contract", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
