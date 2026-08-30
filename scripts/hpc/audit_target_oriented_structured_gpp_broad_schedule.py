"""Materialize the target-blind broad structured-GPP schedule without targets."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_broad_preflight import (
    build_broad_training_schedule,
    load_structured_gpp_broad_preflight_contract,
    reference_identity,
)
from scripts.hpc.audit_target_oriented_structured_gpp_data_coverage import (
    _contract_updates,
    _merged_day_keys,
)

SCHEMA_VERSION = "target_oriented_structured_gpp_broad_schedule_audit_v1"
COVERAGE_REPORT_SHA256 = "a53330a621cd63cd863798caea012a1517ef8552205839cb6f7f93aafff0528b"
SAMPLE_UPDATES = (1, 16, 250, 500, 1000, 2000, 3500, 5000)


def _canonical_selection(selection: Any) -> Mapping[str, Any]:
    return {
        "identity": list(reference_identity(selection.reference)),
        "rows": list(selection.rows),
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    started = time.perf_counter()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    if sha256_file(args.coverage_report) != COVERAGE_REPORT_SHA256:
        raise ValueError("structured GPP broad schedule coverage-report hash drift")

    from research.daily_coarse_graining.daily_markov_contract import (
        daily_markov_contract_from_metadata,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_joint_training import (
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

    coverage = json.loads(args.coverage_report.read_text(encoding="utf-8"))
    broad = load_structured_gpp_broad_preflight_contract(args.broad_preflight_contract)
    dataset = json.loads(args.dataset_manifest.read_text(encoding="utf-8"))
    dataset_sha256 = sha256_file(args.dataset_manifest)
    training = load_target_oriented_training_contract(args.training_contract)
    warm = load_structured_gpp_warm_start_contract(args.warm_start_contract)
    joint = load_structured_gpp_joint_training_contract(args.joint_training_contract)
    prepared = json.loads(args.prepared_manifest.read_text(encoding="utf-8"))
    prepared_sha256 = sha256_file(args.prepared_manifest)
    if (
        coverage.get("status") != "passed"
        or coverage["identity"]["dataset_manifest_sha256"] != dataset_sha256
        or broad.payload["identity"]["dataset_manifest_sha256"] != dataset_sha256
        or broad.payload["identity"]["coverage_report_sha256"] != COVERAGE_REPORT_SHA256
        or prepared_sha256 != warm.payload["identity"]["prepared_manifest_sha256"]
        or prepared_sha256 != joint.payload["identity"]["prepared_manifest_sha256"]
        or prepared["identity"]["training_contract_sha256"] != training.sha256
    ):
        raise ValueError("structured GPP broad schedule parent identity drift")

    markov = daily_markov_contract_from_metadata(dataset["markov_contract"])
    index = load_coherent_typed_sidecar_index(args.dataset_manifest, verify_hashes=False)
    references = tuple(index.select(spatial_split="train", temporal_split="train"))
    old_schedule = build_target_oriented_training_schedule(references, training, updates=1000)
    if old_schedule["schedule_sha256"] != prepared["identity"]["schedule_sha256"]:
        raise ValueError("structured GPP broad schedule prior schedule drift")
    old_entries = {int(value["update"]): value for value in old_schedule["entries"]}
    warm_updates, joint_updates = _contract_updates(warm, joint)
    prior = _merged_day_keys(
        old_entries[value] for value in (*warm_updates, *joint_updates)
    )["all"]
    if len(prior) != int(coverage["exposure"]["combined"]["unique_days"]):
        raise ValueError("structured GPP broad schedule prior exposure drift")

    monitor_entries = tuple(coverage["proposed_monitor"]["entries"])
    monitor_point_years = {
        (
            str(value["landpoint_id"]),
            int(value["year"]),
            str(value["parent_shard_sha256"]),
            str(value["typed_shard_sha256"]),
        )
        for value in monitor_entries
    }
    data = broad.payload["data"]
    schedule = build_broad_training_schedule(
        references,
        prior_exposed_days=prior,
        monitor_point_years=monitor_point_years,
        seed=int(data["selection_seed"]),
        optimizer_updates=int(data["optimizer_updates_maximum"]),
        point_years_per_update=int(data["point_years_per_update"]),
        days_per_point_year=int(data["days_per_point_year_per_update"]),
    )

    used_days = set()
    point_year_presentations: Counter[tuple[str, int, str, str]] = Counter()
    landpoint_day_counts: Counter[str] = Counter()
    year_day_counts: Counter[int] = Counter()
    sampled_batches = {}
    for batch in schedule.batches:
        if len(batch.selections) != int(data["point_years_per_update"]):
            raise ValueError("structured GPP broad schedule batch width drift")
        identities = tuple(reference_identity(value.reference) for value in batch.selections)
        if len({value[0] for value in identities}) != len(identities):
            raise ValueError("structured GPP broad schedule within-batch landpoint duplication")
        for selection, identity in zip(batch.selections, identities, strict=True):
            point_year_presentations[identity] += 1
            for row in selection.rows:
                key = (*identity, int(row))
                if key in used_days:
                    raise ValueError("structured GPP broad schedule repeated a training day")
                used_days.add(key)
                landpoint_day_counts[identity[0]] += 1
                year_day_counts[identity[1]] += 1
        if batch.update in SAMPLE_UPDATES:
            sampled_batches[str(batch.update)] = [
                _canonical_selection(value) for value in batch.selections
            ]
    if (
        len(schedule.batches) != int(data["optimizer_updates_maximum"])
        or len(used_days) != int(data["unique_training_days"])
        or schedule.unique_days != len(used_days)
        or used_days & prior
        or {value[:4] for value in used_days} & monitor_point_years
        or set(sampled_batches) != {str(value) for value in SAMPLE_UPDATES}
    ):
        raise ValueError("structured GPP broad schedule acceptance gate failed")

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "identity": {
            "broad_preflight_contract_sha256": broad.sha256,
            "coverage_report_sha256": COVERAGE_REPORT_SHA256,
            "dataset_manifest_sha256": dataset_sha256,
            "markov_contract_sha256": markov.sha256,
            "prepared_manifest_sha256": prepared_sha256,
            "prior_schedule_sha256": old_schedule["schedule_sha256"],
            "broad_schedule_sha256": schedule.sha256,
        },
        "schedule": {
            "optimizer_updates": len(schedule.batches),
            "point_years_per_update": int(data["point_years_per_update"]),
            "days_per_point_year_per_update": int(data["days_per_point_year_per_update"]),
            "days_per_update": int(data["effective_day_batch_size"]),
            "unique_days": len(used_days),
            "unique_point_years": len(point_year_presentations),
            "unique_landpoints": len(landpoint_day_counts),
            "unique_years": len(year_day_counts),
            "point_year_presentation_count": {
                "minimum": min(point_year_presentations.values()),
                "maximum": max(point_year_presentations.values()),
            },
            "landpoint_day_count": {
                "minimum": min(landpoint_day_counts.values()),
                "maximum": max(landpoint_day_counts.values()),
            },
            "year_day_count": {
                "minimum": min(year_day_counts.values()),
                "maximum": max(year_day_counts.values()),
            },
            "sample_batches": sampled_batches,
        },
        "exclusions": {
            "prior_exposed_days": len(prior),
            "monitor_point_years": len(monitor_point_years),
            "monitor_sampled_days": int(coverage["proposed_monitor"]["sampled_days"]),
            "overlap_with_prior_exposure": len(used_days & prior),
            "overlap_with_monitor_point_years": len(
                {value[:4] for value in used_days} & monitor_point_years
            ),
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
        "next_gate": "run_broad_direct_gpp_free_gpu_preflight",
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--training-contract", type=Path, required=True)
    parser.add_argument("--warm-start-contract", type=Path, required=True)
    parser.add_argument("--joint-training-contract", type=Path, required=True)
    parser.add_argument("--broad-preflight-contract", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--coverage-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
