"""Audit the frozen one-point DOC targets without training a neural model."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.hpc.prepare_successor_small_fit_batches import _initialize_cpu_backend
from scripts.hpc.run_strong_baseline_one_point_qualification import _prepare_point_year
from scripts.hpc.run_successor_fast_owner_gpu_probe import (
    FAST_STATISTICS_JSON_SHA256,
    _load_extended_runtime,
)
from scripts.hpc.run_successor_process_head_canonical_gate import (
    EXPECTED,
    FROZEN_LANDPOINT,
    _atomic_json,
)

AUDITED_FIELDS = (
    "inventory.carbon.DOC.incoming_amount",
    "inventory.carbon.DOC.outgoing_fraction",
    "typed.ok_leak.doc_free_to_adsorbed",
    "typed.ok_leak.doc_advective_interface",
    "typed.ok_leak.doc_diffusive_interface",
    "typed.ok_leak.doc_gross_decomposition",
)
PAIR_NAMES = (
    (
        "incoming_vs_free_adsorbed",
        "inventory.carbon.DOC.incoming_amount",
        "typed.ok_leak.doc_free_to_adsorbed",
    ),
    (
        "incoming_vs_advective",
        "inventory.carbon.DOC.incoming_amount",
        "typed.ok_leak.doc_advective_interface",
    ),
    (
        "free_adsorbed_vs_advective",
        "typed.ok_leak.doc_free_to_adsorbed",
        "typed.ok_leak.doc_advective_interface",
    ),
    (
        "advective_vs_diffusive",
        "typed.ok_leak.doc_advective_interface",
        "typed.ok_leak.doc_diffusive_interface",
    ),
)


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _target(dataset: Mapping[str, Any], name: str) -> Any:
    try:
        return dataset["targets"][name]
    except KeyError as error:
        raise ValueError(f"DOC audit target {name!r} is absent") from error


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    from research.daily_coarse_graining.daily_operator.doc_target_diagnostics import (
        inventory_partition_statistics,
        masked_field_statistics,
        paired_daily_aggregate_statistics,
    )

    runtime = _load_extended_runtime(args, backend_initializer=_initialize_cpu_backend)
    dataset = _prepare_point_year(runtime)
    if dataset["sample_count"] != 364:
        raise ValueError("DOC target audit requires the frozen 364-day point-year")

    missing_scales = sorted(set(AUDITED_FIELDS) - set(runtime["loss_scales"]))
    if missing_scales:
        raise ValueError(f"DOC target audit lacks frozen loss scales: {missing_scales}")
    field_reports = {}
    for name in AUDITED_FIELDS:
        target = _target(dataset, name)
        field_reports[name] = masked_field_statistics(
            target.value,
            target.defined,
            runtime["loss_scales"][name],
        )

    pair_reports = {}
    for report_name, left_name, right_name in PAIR_NAMES:
        left = _target(dataset, left_name)
        right = _target(dataset, right_name)
        pair_reports[report_name] = paired_daily_aggregate_statistics(
            left.value,
            left.defined,
            runtime["loss_scales"][left_name],
            right.value,
            right.defined,
            runtime["loss_scales"][right_name],
        )

    incoming = _target(dataset, "inventory.carbon.DOC.incoming_amount")
    outgoing = _target(dataset, "inventory.carbon.DOC.outgoing_fraction")
    partition = inventory_partition_statistics(
        incoming.value,
        incoming.defined,
        outgoing.value,
        outgoing.defined,
    )
    passed = (
        partition["mutually_exclusive"]
        and partition["incoming_negative_count"] == 0
        and partition["outgoing_out_of_unit_interval_count"] == 0
        and all(report["nonfinite_or_invalid_scale_count"] == 0 for report in field_reports.values())
    )
    report = {
        "schema_version": "hierarchical_coordinate_query_attention_v3_doc_target_audit_v1",
        "status": "passed" if passed else "failed",
        "training_updates": 0,
        "checkpoint_written": False,
        "teacher_data_regenerated": False,
        "historical_v5_used": False,
        "identity": {
            "git_head": _git_head(),
            "dataset_manifest_sha256": EXPECTED["dataset_manifest"],
            "fast_owner_statistics_json_sha256": FAST_STATISTICS_JSON_SHA256,
        },
        "data": {
            "landpoint_id": FROZEN_LANDPOINT,
            "year": 1961,
            "spatial_split": "train",
            "temporal_split": "train",
            "day_count": int(dataset["sample_count"]),
            "day_index_minimum": int(np.min(dataset["day_indices"])),
            "day_index_maximum": int(np.max(dataset["day_indices"])),
        },
        "inventory_partition": partition,
        "fields": field_reports,
        "pairwise": pair_reports,
        "next_gate": "review_doc_target_representation_and_coordinate_coupling",
    }
    _atomic_json(args.output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path, required=True)
    parser.add_argument("--base-statistics", type=Path, required=True)
    parser.add_argument("--process-statistics", type=Path, required=True)
    parser.add_argument("--fast-statistics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
