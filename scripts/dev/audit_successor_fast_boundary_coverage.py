#!/usr/bin/env python3
"""Audit ownership of every v6 fast-day leaf at the successor boundary.

This is a contract audit, not a training entry point.  It distinguishes a
same-owner residual baseline from the process that actually produces the
day-end value and flags daily-integral labels currently written into
instantaneous fast-state fields.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.daily_coarse_graining.canonical_training import (  # noqa: E402
    fast_day_target_representation_from_contract,
)
from research.daily_coarse_graining.constrained_daily_replay import (  # noqa: E402
    CARBON_INVENTORY_FIELDS,
    THERMAL_FIELDS,
    WATER_INVENTORY_FIELDS,
)
from research.daily_coarse_graining.markov_dataset import (  # noqa: E402
    defined_numeric_mask,
    load_markov_shard,
)

_DRIVER_FORMULAS = frozenset(
    {
        "driver_previous_step_state.old_eair",
        "driver_previous_step_state.old_qair",
        "driver_previous_step_state.old_zlev",
        "driver_previous_step_state.rau_old",
    }
)

_TYPED_FAST_STATE_COLLISIONS = {
    **{
        f"hydrol_previous_step_state.{name}": f"typed.water.{name}"
        for name in (
            "canopy2ground",
            "drainage",
            "drainage_per_soil",
            "precip2canopy",
            "precip2ground",
            "runoff",
            "runoff2peat",
            "runoff_per_soil",
            "subsnownobio",
            "wat_flux",
            "water2infilt",
        )
    },
    **{
        f"{component}.{name}": f"typed.energy.{name}"
        for component in (
            "driver_previous_step_state",
            "enerbil_previous_step_state",
        )
        for name in ("fluxsens", "soilflx", "soilflx_pft")
    },
}

_DAILY_NAMED_OUTPUTS = {
    "daily_interface.gpp_daily": ("daily.fast_gpp",),
    "daily_interface.resp_maint_part": ("daily.fast_maintenance_respiration",),
    "daily_interface.tsoil_daily": ("daily.temperature_context.tsoil",),
    "daily_interface.tsurf_daily": ("daily.temperature_context.tsurf",),
    "final_diagnostics.temp_sol": ("daily.temperature_context.tsurf",),
}

_DAILY_NAMED_DERIVATIONS = {
    "daily_interface.resp_maint_radia": ("daily.fast_maintenance_respiration",),
}

_FORCING_FORMULAS = frozenset(
    {
        "daily_interface.precip_daily",
        "daily_interface.snowfall_daily",
        "daily_interface.t2m_daily",
        "daily_interface.t2m_max_daily",
        "daily_interface.t2m_min_daily",
        "daily_interface.wspeed_daily",
        "final_diagnostics.t2mdiag",
    }
)

_START_DERIVATIONS = frozenset(
    {
        "daily_interface.humrel_daily",
        "daily_interface.litterhum_daily",
        "daily_interface.snowmass_daily",
        "daily_interface.soilhum_daily",
        "daily_interface.tmc_topgrass_daily",
    }
)

_CONSTANT_DERIVATIONS = frozenset({"daily_interface.flood_root_radia"})


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _state_owner_by_target_column(
    contract: Mapping[str, Any],
) -> dict[int, str]:
    owners: dict[int, str] = {}
    for leaf in contract["state_leaves"]:
        start = leaf.get("start")
        stop = leaf.get("stop")
        if start is None or stop is None:
            continue
        for index in range(int(start), int(stop)):
            owners[index] = str(leaf["key"])
    return owners


def _source_for_leaf(leaf: Mapping[str, Any]) -> dict[str, Any]:
    key = str(leaf["key"])
    component = leaf.get("component")
    field = str(leaf["path"][0])
    thermal = {f"{owner}.{name}" for owner, name, _bound in THERMAL_FIELDS}

    if component == "hydrol_previous_step_state" and field in WATER_INVENTORY_FIELDS:
        return {
            "source_kind": "conservative_water_endpoint",
            "source_outputs": (
                f"inventory.water.{field}.outgoing_fraction",
                f"inventory.water.{field}.incoming_amount",
            ),
            "coverage_status": "explicit_process_output",
        }
    if key in thermal:
        return {
            "source_kind": "bounded_thermal_endpoint",
            "source_outputs": (f"thermal.{key}.normalized_tendency",),
            "coverage_status": "explicit_process_output",
        }
    if component is None and str(leaf["family"]) == "ok_leak":
        if field in CARBON_INVENTORY_FIELDS:
            return {
                "source_kind": "conservative_carbon_endpoint",
                "source_outputs": (
                    f"inventory.carbon.{field}.outgoing_fraction",
                    f"inventory.carbon.{field}.incoming_amount",
                ),
                "coverage_status": "explicit_process_output",
            }
        return {
            "source_kind": "day_start_ok_leak_carry",
            "source_outputs": (),
            "coverage_status": "unverified_carry",
        }
    if key in _DRIVER_FORMULAS:
        return {
            "source_kind": "source_exact_final_forcing_formula",
            "source_outputs": (),
            "coverage_status": "exact_derivation",
        }
    if key in _TYPED_FAST_STATE_COLLISIONS:
        return {
            "source_kind": "daily_integral_written_to_instantaneous_state",
            "source_outputs": (_TYPED_FAST_STATE_COLLISIONS[key],),
            "coverage_status": "semantic_collision",
        }
    if key in _DAILY_NAMED_OUTPUTS:
        return {
            "source_kind": "named_daily_output",
            "source_outputs": _DAILY_NAMED_OUTPUTS[key],
            "coverage_status": "explicit_process_output",
        }
    if key in _DAILY_NAMED_DERIVATIONS:
        return {
            "source_kind": "derived_from_named_daily_output",
            "source_outputs": _DAILY_NAMED_DERIVATIONS[key],
            "coverage_status": "exact_derivation",
        }
    if key in _FORCING_FORMULAS:
        return {
            "source_kind": "native_forcing_formula",
            "source_outputs": (),
            "coverage_status": "exact_derivation",
        }
    if key in _START_DERIVATIONS:
        return {
            "source_kind": "derived_from_day_start_fast_state",
            "source_outputs": (),
            "coverage_status": "unverified_derivation",
        }
    if key in _CONSTANT_DERIVATIONS:
        return {
            "source_kind": "fixed_zero",
            "source_outputs": (),
            "coverage_status": "unverified_derivation",
        }
    return {
        "source_kind": "day_start_state_carry",
        "source_outputs": (),
        "coverage_status": "unverified_carry",
    }


def build_ledger(
    contract: Mapping[str, Any],
    *,
    attribution: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    representation = fast_day_target_representation_from_contract(contract)
    state_indices = np.asarray(representation.state_indices, dtype=np.int32)
    dynamic = set(
        int(index)
        for index in np.asarray(
            representation.dynamic_undefined_indices,
            dtype=np.int32,
        )
    )
    state_owner = _state_owner_by_target_column(contract)
    records = []
    for leaf in contract["fast_day_target_leaves"]:
        start = int(leaf["start"])
        stop = int(leaf["stop"])
        indices = state_indices[start:stop]
        mapped = indices >= 0
        baseline_owners = sorted({state_owner[int(index)] for index in indices[mapped]})
        source = _source_for_leaf(leaf)
        records.append(
            {
                "key": str(leaf["key"]),
                "family": str(leaf["family"]),
                "component": leaf.get("component"),
                "path": [str(item) for item in leaf["path"]],
                "shape": [int(item) for item in leaf["shape"]],
                "start": start,
                "stop": stop,
                "width": stop - start,
                "residual_baseline_mapped_columns": int(np.count_nonzero(mapped)),
                "mean_centered_columns": int(np.count_nonzero(~mapped)),
                "residual_baseline_state_owners": baseline_owners,
                "dynamic_undefined_columns": sum(index in dynamic for index in range(start, stop)),
                **source,
            }
        )

    status_counts = Counter(record["coverage_status"] for record in records)
    source_counts = Counter(record["source_kind"] for record in records)
    status_columns = Counter()
    for record in records:
        status_columns[record["coverage_status"]] += record["width"]

    all_teacher = None
    if attribution is not None:
        all_teacher_case = attribution.get("cases", {}).get("all_teacher", {})
        report = all_teacher_case.get("case", {}).get("canonical_internal_fast_target_vs_teacher")
        if report is not None:
            all_teacher = {
                "teacher_field_count": int(all_teacher_case.get("teacher_field_count", 0)),
                **report,
            }

    return {
        "schema_version": "successor_fast_boundary_coverage_v1",
        "target_width": int(contract["fast_day_target_width"]),
        "leaf_count": len(records),
        "residual_representation": {
            "same_owner_baseline_columns": int(np.count_nonzero(state_indices >= 0)),
            "mean_centered_columns": int(np.count_nonzero(state_indices < 0)),
            "note": (
                "A same-owner baseline is a residual representation; it does "
                "not imply numeric persistence across the day."
            ),
        },
        "coverage_status_leaf_counts": dict(sorted(status_counts.items())),
        "coverage_status_column_counts": dict(sorted(status_columns.items())),
        "source_kind_leaf_counts": dict(sorted(source_counts.items())),
        "all_teacher_observation": all_teacher,
        "blocking_leaves": [
            record
            for record in records
            if record["coverage_status"]
            in {
                "semantic_collision",
                "unverified_carry",
                "unverified_derivation",
            }
        ],
        "leaves": records,
    }


def _selected_shards(
    manifest: Mapping[str, Any],
    *,
    per_split: int,
) -> list[Mapping[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for item in manifest.get("shards", ()):  # type: ignore[union-attr]
        key = (str(item["spatial_split"]), str(item["temporal_split"]))
        grouped.setdefault(key, []).append(item)
    selected = []
    for key in sorted(grouped):
        items = sorted(
            grouped[key],
            key=lambda item: (int(item["year"]), str(item["landpoint_id"])),
        )
        selected.extend(items[:per_split])
    return selected


def scan_numeric_persistence(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    ledger: dict[str, Any],
    *,
    shards_per_split: int,
) -> dict[str, Any]:
    if shards_per_split <= 0:
        raise ValueError("shards_per_split must be positive")
    contract = manifest["markov_contract"]
    representation = fast_day_target_representation_from_contract(contract)
    state_indices = np.asarray(representation.state_indices, dtype=np.int32)
    selected = _selected_shards(manifest, per_split=shards_per_split)
    if not selected:
        raise ValueError("dataset manifest has no shards to scan")

    accumulators = {
        str(leaf["key"]): {
            "compared_values": 0,
            "changed_values": 0,
            "undefined_mismatches": 0,
            "maximum_absolute_delta": 0.0,
        }
        for leaf in contract["fast_day_target_leaves"]
    }
    transitions = 0
    references = []
    for item in selected:
        shard_path = manifest_path.parent / str(item["shard"])
        shard = load_markov_shard(shard_path)
        state = np.asarray(shard.state_trajectory[:-1], dtype=np.float64)
        target = np.asarray(shard.fast_day_target, dtype=np.float64)
        transitions += int(target.shape[0])
        references.append(
            {
                "landpoint_id": str(item["landpoint_id"]),
                "year": int(item["year"]),
                "spatial_split": str(item["spatial_split"]),
                "temporal_split": str(item["temporal_split"]),
                "transitions": int(target.shape[0]),
                "shard": str(item["shard"]),
            }
        )
        for leaf in contract["fast_day_target_leaves"]:
            start = int(leaf["start"])
            stop = int(leaf["stop"])
            indices = state_indices[start:stop]
            mapped = indices >= 0
            if not np.any(mapped):
                continue
            if not np.all(mapped):
                raise ValueError(f"partially mapped fast-day leaf: {leaf['key']}")
            observed = target[:, start:stop]
            baseline = state[:, indices]
            observed_defined = defined_numeric_mask(observed)
            baseline_defined = defined_numeric_mask(baseline)
            common = observed_defined & baseline_defined
            difference = np.abs(observed - baseline)
            accumulator = accumulators[str(leaf["key"])]
            accumulator["compared_values"] += int(np.count_nonzero(common))
            accumulator["changed_values"] += int(
                np.count_nonzero(
                    common
                    & ~np.isclose(
                        observed,
                        baseline,
                        atol=1.0e-12,
                        rtol=1.0e-12,
                    )
                )
            )
            accumulator["undefined_mismatches"] += int(np.count_nonzero(observed_defined != baseline_defined))
            if np.any(common):
                accumulator["maximum_absolute_delta"] = max(
                    float(accumulator["maximum_absolute_delta"]),
                    float(np.max(difference[common])),
                )

    by_key = {record["key"]: record for record in ledger["leaves"]}
    for key, accumulator in accumulators.items():
        compared = int(accumulator["compared_values"])
        changed = int(accumulator["changed_values"])
        accumulator["changed_fraction"] = changed / compared if compared else None
        accumulator["status"] = (
            "not_same_owner_mapped"
            if compared == 0
            else "changed"
            if changed or accumulator["undefined_mismatches"]
            else "sampled_numeric_carry"
        )
        by_key[key]["sampled_numeric_persistence"] = accumulator

    blockers = [
        record
        for record in ledger["leaves"]
        if record["coverage_status"] == "unverified_carry"
        and record["sampled_numeric_persistence"]["status"] == "changed"
    ]
    result = {
        "shards_per_split": shards_per_split,
        "shard_count": len(selected),
        "transitions": transitions,
        "references": references,
        "unverified_carry_leaves_observed_changing": len(blockers),
        "unverified_carry_columns_observed_changing": sum(int(record["width"]) for record in blockers),
        "changing_unverified_carry_leaves": [
            {
                "key": record["key"],
                "width": record["width"],
                **record["sampled_numeric_persistence"],
            }
            for record in blockers
        ],
    }
    ledger["numeric_persistence_scan"] = result
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        required=True,
        help="Accepted v6 dataset manifest containing markov_contract.",
    )
    parser.add_argument(
        "--attribution-report",
        type=Path,
        help="Optional successor attribution report with an all_teacher case.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSON report under outputs/.",
    )
    parser.add_argument(
        "--scan-shards-per-split",
        type=int,
        default=0,
        help=(
            "Read this many earliest shards from each spatial/temporal split "
            "and measure numeric day-start-to-fast-target changes."
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    manifest = _load_json(args.dataset_manifest)
    contract = manifest.get("markov_contract", manifest)
    attribution = _load_json(args.attribution_report) if args.attribution_report is not None else None
    report = build_ledger(contract, attribution=attribution)
    if args.scan_shards_per_split:
        scan_numeric_persistence(
            manifest,
            args.dataset_manifest,
            report,
            shards_per_split=args.scan_shards_per_split,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    summary_keys = [
        "target_width",
        "leaf_count",
        "residual_representation",
        "coverage_status_leaf_counts",
        "coverage_status_column_counts",
        "all_teacher_observation",
    ]
    if "numeric_persistence_scan" in report:
        summary_keys.append("numeric_persistence_scan")
    print(
        json.dumps(
            {key: report[key] for key in summary_keys},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
