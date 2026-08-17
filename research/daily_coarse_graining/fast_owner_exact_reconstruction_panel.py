"""Evaluate source-exact fast-owner reconstructions on a non-test v6 panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from research.daily_coarse_graining.daily_markov_contract import (
    daily_markov_contract_from_metadata,
    load_markov_shard,
)

try:
    from research.daily_coarse_graining.fast_owner_delta_representation_audit import (
        _source_reconstruction_report,
    )
except ModuleNotFoundError as error:
    if error.name != "research.daily_coarse_graining.fast_owner_delta_representation_audit":
        raise
    from fast_owner_endpoint_reachability_audit import _source_reconstruction_report
from research.daily_coarse_graining.typed_sidecar import (
    load_coherent_typed_sidecar_index,
)

SCHEMA_VERSION = "fast_owner_exact_reconstruction_panel_v1"
EXACT_CANDIDATES = (
    "condveg_roughness.z0m",
    "condveg_roughness.roughheight",
    "condveg_roughness.roughheight_pft",
    "hydrol_drysoil_fraction",
    "thermosoil_final_diagnostics.gtemp",
    "thermosoil_final_diagnostics.stempdiag",
    "thermosoil_final_diagnostics.tdeep",
    "thermosoil_final_diagnostics.hsdeep",
)
UNRESOLVED_CANDIDATES = ("condveg_roughness.z0h",)


def parse_selection(value: str) -> tuple[str, int]:
    landpoint, separator, year = value.rpartition(":")
    if not separator or not landpoint:
        raise ValueError(f"selection must be LANDPOINT:YEAR, got {value!r}")
    parsed_year = int(year)
    if parsed_year < 1900 or parsed_year > 2200:
        raise ValueError(f"selection year is outside the supported range: {parsed_year}")
    return landpoint, parsed_year


def _flatten_source_fields(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    condveg = report["condveg_roughness"]["fields"]
    thermosoil = report["thermosoil_final_diagnostics"]["fields"]
    return {
        **{f"condveg_roughness.{name}": value for name, value in condveg.items()},
        "hydrol_drysoil_fraction": report["hydrol_drysoil_fraction"],
        **{f"thermosoil_final_diagnostics.{name}": value for name, value in thermosoil.items()},
    }


def summarize_panel_fields(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    if not records:
        raise ValueError("exact-reconstruction panel cannot be empty")
    flattened = [_flatten_source_fields(record["source_reconstruction"]) for record in records]
    names = set(flattened[0])
    if any(set(fields) != names for fields in flattened[1:]):
        raise ValueError("exact-reconstruction panel field registries drift")
    return {
        name: {
            "point_year_count": len(records),
            "passed_point_year_count": sum(bool(fields[name]["passed"]) for fields in flattened),
            "failed_point_year_count": sum(not bool(fields[name]["passed"]) for fields in flattened),
            "value_count": sum(int(fields[name]["value_count"]) for fields in flattened),
            "maximum_absolute_error": max(float(fields[name]["maximum_absolute_error"]) for fields in flattened),
            "maximum_root_mean_squared_error": max(
                float(fields[name]["root_mean_squared_error"]) for fields in flattened
            ),
            "nonfinite_reconstructed_count": sum(
                int(fields[name]["nonfinite_reconstructed_count"]) for fields in flattened
            ),
        }
        for name in sorted(names)
    }


def build_panel_report(
    *,
    dataset_manifest: Path,
    selections: Sequence[tuple[str, int]],
) -> dict[str, Any]:
    manifest = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    contract = daily_markov_contract_from_metadata(manifest["markov_contract"])
    index = load_coherent_typed_sidecar_index(dataset_manifest, verify_hashes=False)
    references = {(reference.parent.landpoint_id, int(reference.parent.year)): reference for reference in index.shards}
    if len(selections) != len(set(selections)):
        raise ValueError("exact-reconstruction panel selections must be unique")

    records = []
    for selection in selections:
        reference = references.get(selection)
        if reference is None:
            raise ValueError(f"v6 panel selection is missing: {selection}")
        parent = reference.parent
        if parent.spatial_split == "test" or parent.temporal_split == "test":
            raise ValueError(f"sealed test selection is forbidden: {selection}")
        shard = load_markov_shard(parent.path, contract=contract)
        records.append(
            {
                "landpoint_id": parent.landpoint_id,
                "year": int(parent.year),
                "spatial_split": parent.spatial_split,
                "temporal_split": parent.temporal_split,
                "day_count": shard.days,
                "source_reconstruction": _source_reconstruction_report(shard, contract),
            }
        )

    fields = summarize_panel_fields(records)
    missing = sorted((set(EXACT_CANDIDATES) | set(UNRESOLVED_CANDIDATES)) - set(fields))
    if missing:
        raise ValueError(f"exact-reconstruction panel is missing fields: {missing}")
    exact_failures = [name for name in EXACT_CANDIDATES if fields[name]["failed_point_year_count"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": {
            "dataset_id": index.parent_index.dataset_id,
            "teacher_git_head": index.parent_index.teacher_git_head,
            "point_year_count": len(records),
        },
        "policy": {
            "teacher_data_regenerated": False,
            "historical_v5_read": False,
            "typed_sidecar_values_read": False,
            "sealed_test_allowed": False,
            "forcing_records_materialized": 0,
            "subdaily_state_scan_steps": 0,
        },
        "summary": {
            "status": "exact_candidates_passed" if not exact_failures else "exact_candidates_failed",
            "exact_candidate_count": len(EXACT_CANDIDATES),
            "exact_failure_count": len(exact_failures),
            "exact_failures": exact_failures,
            "unresolved_candidates": list(UNRESOLVED_CANDIDATES),
        },
        "fields": fields,
        "records": records,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--selection", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = build_panel_report(
        dataset_manifest=args.dataset_manifest,
        selections=tuple(parse_selection(value) for value in args.selection),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
