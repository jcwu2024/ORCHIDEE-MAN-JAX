"""Admit a reusable 669-point Daily Teacher data product before generation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from research.daily_coarse_graining.daily_markov_contract import (
    daily_markov_contract_from_metadata,
)
from research.daily_coarse_graining.teacher_production import (
    ROOT,
    ProductionSpec,
    load_production_spec,
)

POLICY_SCHEMA_VERSION = "daily_teacher_data_product_policy_v1"
REPORT_SCHEMA_VERSION = "daily_teacher_data_product_admission_v1"
PAPER_DAYS_PER_YEAR = 365


def _canonical_json_sha256(value: Mapping[str, Any] | Path) -> str:
    payload = (
        json.loads(value.read_text(encoding="utf-8"))
        if isinstance(value, Path)
        else value
    )
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _require_equal(name: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ValueError(f"{name} mismatch: observed {observed!r}, expected {expected!r}")


def _require_subset(name: str, observed: set[str], required: set[str]) -> None:
    missing = sorted(required - observed)
    if missing:
        raise ValueError(f"{name} is missing required values: {missing}")


def _state_discrete_width(contract) -> int:
    return sum(
        math.prod(leaf.shape)
        for leaf in contract.state_leaves
        if leaf.discrete
    )


def _target_key(leaf) -> str:
    if leaf.component is not None:
        return f"{leaf.component}.{'.'.join(leaf.path)}"
    return f"{leaf.family}.{'.'.join(leaf.path)}"


def _split_counts(spec: ProductionSpec) -> dict[str, int]:
    return {
        split: sum(item.spatial_split == split for item in spec.landpoints)
        for split in ("train", "validation", "test")
    }


def _transition_inventory(spec: ProductionSpec) -> dict[str, Any]:
    years = range(spec.first_year, spec.last_year + 1)
    cells: dict[str, dict[str, int]] = {}
    for spatial_split in ("train", "validation", "test"):
        point_count = sum(
            item.spatial_split == spatial_split for item in spec.landpoints
        )
        cells[spatial_split] = {}
        for temporal_split in ("train", "validation", "test"):
            transition_days = sum(
                PAPER_DAYS_PER_YEAR - (year == spec.first_year)
                for year in years
                if spec.temporal_split(year) == temporal_split
            )
            cells[spatial_split][temporal_split] = point_count * transition_days
    point_years = len(spec.landpoints) * (spec.last_year - spec.first_year + 1)
    total_transitions = sum(sum(row.values()) for row in cells.values())
    return {
        "point_years": point_years,
        "total_transitions": total_transitions,
        "split_transition_counts": cells,
        "train_train_transitions": cells["train"]["train"],
    }


def _resource_estimate(
    point_years: int, resource_basis: Mapping[str, Any]
) -> dict[str, Any]:
    hot_seconds = point_years * float(
        resource_basis["hot_capture_seconds_per_point_year"]
    )
    upper_seconds = point_years * float(
        resource_basis["complete_point_year_seconds_upper"]
    )
    workers = int(resource_basis["accepted_concurrent_workers"])
    cost_rate = float(resource_basis["cpu_cost_cny_per_core_hour"])
    return {
        "hot_capture_core_hours": hot_seconds / 3600.0,
        "complete_upper_core_hours": upper_seconds / 3600.0,
        "estimated_cpu_cost_cny": {
            "hot_capture_lower": hot_seconds / 3600.0 * cost_rate,
            "complete_upper": upper_seconds / 3600.0 * cost_rate,
        },
        "estimated_compressed_gib": {
            "lower": point_years
            * int(resource_basis["compressed_bytes_per_point_year_lower"])
            / (1024**3),
            "upper": point_years
            * int(resource_basis["compressed_bytes_per_point_year_upper"])
            / (1024**3),
        },
        "conservative_wall_days_at_accepted_concurrency": upper_seconds
        / workers
        / 86400.0,
        "accepted_concurrent_workers": workers,
        "basis": dict(resource_basis),
    }


def _validate_complete_production_manifest(
    *,
    manifest: Mapping[str, Any],
    spec: ProductionSpec,
    contract_sha256: str,
) -> None:
    _require_equal("production dataset status", manifest.get("status"), "complete")
    expected_years = range(spec.first_year, spec.last_year + 1)
    expected_shards = {
        (item.landpoint_id, year): (
            item.spatial_split,
            spec.temporal_split(year),
        )
        for item in spec.landpoints
        for year in expected_years
    }
    _require_equal("production dataset ID", manifest["dataset_id"], spec.dataset_id)
    _require_equal(
        "production landpoint count",
        manifest["landpoint_count"],
        len(spec.landpoints),
    )
    _require_equal(
        "production year count",
        manifest["year_count"],
        spec.last_year - spec.first_year + 1,
    )
    _require_equal(
        "production shard count",
        manifest["shard_count"],
        len(expected_shards),
    )
    observed_shards: dict[tuple[str, int], tuple[str, str]] = {}
    for shard in manifest["shards"]:
        key = (str(shard["landpoint_id"]), int(shard["year"]))
        if key in observed_shards:
            raise ValueError(f"duplicate production shard: {key}")
        _require_equal(
            f"shard contract hash for {key}",
            shard["markov_contract_sha256"],
            contract_sha256,
        )
        observed_shards[key] = (
            str(shard["spatial_split"]),
            str(shard["temporal_split"]),
        )
    missing = sorted(set(expected_shards) - set(observed_shards))
    extra = sorted(set(observed_shards) - set(expected_shards))
    if missing or extra:
        raise ValueError(
            "production shard inventory mismatch: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    wrong_splits = sorted(
        key
        for key, expected_split in expected_shards.items()
        if observed_shards[key] != expected_split
    )
    if wrong_splits:
        raise ValueError(f"production shard split mismatch: {wrong_splits[:5]}")


def admit_data_product(
    *,
    policy_path: str | Path,
    contract_manifest_path: str | Path,
    output_path: str | Path,
    require_production_dataset: bool = False,
) -> Path:
    policy_path = Path(policy_path).resolve()
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    _require_equal(
        "policy schema",
        policy.get("schema_version"),
        POLICY_SCHEMA_VERSION,
    )
    spec_path = _resolve_path(policy["production_spec"])
    _require_equal(
        "production spec hash",
        _canonical_json_sha256(spec_path),
        policy["production_spec_sha256"],
    )
    spec = load_production_spec(spec_path)
    _require_equal(
        "population manifest hash",
        spec.population_manifest_sha256,
        policy["population_manifest_sha256"],
    )

    population = policy["expected_population"]
    _require_equal("landpoint count", len(spec.landpoints), population["landpoints"])
    _require_equal("first year", spec.first_year, population["first_year"])
    _require_equal("last year", spec.last_year, population["last_year"])
    _require_equal("block size", spec.block_size, population["block_size"])
    _require_equal(
        "spatial split counts",
        _split_counts(spec),
        population["spatial_split_counts"],
    )
    _require_equal(
        "temporal splits",
        {name: list(bounds) for name, bounds in spec.temporal_splits.items()},
        population["temporal_splits"],
    )

    contract_manifest_path = Path(contract_manifest_path).resolve()
    manifest = json.loads(contract_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("contract evidence dataset manifest is not complete")
    metadata = manifest["markov_contract"]
    metadata_sha256 = _canonical_json_sha256(metadata)
    _require_equal(
        "dataset contract hash",
        manifest["markov_contract_sha256"],
        metadata_sha256,
    )
    expected = policy["expected_contract"]
    _require_equal(
        "dataset manifest schema",
        manifest.get("schema_version"),
        expected["dataset_manifest_schema"],
    )
    _require_equal("accepted contract hash", metadata_sha256, expected["sha256"])
    contract = daily_markov_contract_from_metadata(metadata)
    _require_equal("contract schema", contract.schema_version, expected["schema_version"])
    _require_equal(
        "continuous state width",
        contract.continuous_state_width,
        expected["continuous_state_width"],
    )
    _require_equal(
        "discrete state width",
        _state_discrete_width(contract),
        expected["discrete_state_width"],
    )
    _require_equal(
        "fast-day target width",
        contract.fast_day_target_width,
        expected["fast_day_target_width"],
    )
    _require_equal(
        "diagnostic width",
        contract.diagnostic_width,
        expected["diagnostic_width"],
    )
    _require_equal(
        "active PFT indices",
        list(contract.active_pft_indices),
        expected["active_pft_indices"],
    )
    _require_equal(
        "native forcing fields",
        list(contract.native_forcing.fields),
        expected["native_forcing_fields"],
    )
    _require_equal(
        "native forcing records per day",
        contract.native_forcing.source_records_per_day,
        expected["native_forcing_records_per_day"],
    )
    condition_widths = {
        "parameters": contract.static_conditions.parameter_width,
        "landpoint_static": contract.static_conditions.landpoint_static_width,
        "annual": contract.static_conditions.annual_condition_width,
    }
    _require_equal(
        "condition widths", condition_widths, expected["condition_widths"]
    )
    _require_subset(
        "state components",
        {leaf.component for leaf in contract.state_leaves},
        set(expected["required_state_components"]),
    )
    _require_subset(
        "state inventory",
        {leaf.key for leaf in contract.state_leaves},
        set(expected["required_state_keys"]),
    )
    _require_subset(
        "target families",
        {leaf.family for leaf in contract.fast_day_target_leaves},
        set(expected["required_target_families"]),
    )
    _require_subset(
        "target inventory",
        {_target_key(leaf) for leaf in contract.fast_day_target_leaves},
        set(expected["required_target_keys"]),
    )
    _require_subset(
        "diagnostic inventory",
        {leaf.name for leaf in contract.diagnostic_leaves},
        set(expected["required_diagnostics"]),
    )
    if require_production_dataset:
        expected_plan_sha256 = policy.get("expected_generation_plan_sha256")
        if expected_plan_sha256 is not None:
            _require_equal(
                "production generation plan hash",
                manifest.get("plan_sha256"),
                expected_plan_sha256,
            )
        expected_teacher_git_head = policy.get("expected_teacher_git_head")
        if expected_teacher_git_head is not None:
            _require_equal(
                "production Teacher git head",
                manifest.get("teacher_git_head"),
                expected_teacher_git_head,
            )
        _validate_complete_production_manifest(
            manifest=manifest,
            spec=spec,
            contract_sha256=metadata_sha256,
        )

    transitions = _transition_inventory(spec)
    cold = policy["cold_start_policy"]
    _require_equal("cold-start first year", spec.first_year, 1961)
    _require_equal(
        "cold-start transition count",
        PAPER_DAYS_PER_YEAR - 1,
        cold["first_year_transition_count"],
    )
    _require_equal(
        "ordinary-year transition count",
        PAPER_DAYS_PER_YEAR,
        cold["ordinary_year_transition_count"],
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "passed",
        "admission_phase": (
            "complete_669_dataset"
            if require_production_dataset
            else "pre_generation_contract"
        ),
        "policy_id": policy["policy_id"],
        "policy_sha256": _canonical_json_sha256(policy),
        "production_spec": str(spec_path),
        "production_spec_sha256": policy["production_spec_sha256"],
        "contract_evidence_manifest": str(contract_manifest_path),
        "contract_evidence_dataset_id": manifest["dataset_id"],
        "contract_sha256": metadata_sha256,
        "contract_summary": {
            "continuous_state_width": contract.continuous_state_width,
            "discrete_state_width": _state_discrete_width(contract),
            "fast_day_target_width": contract.fast_day_target_width,
            "diagnostic_width": contract.diagnostic_width,
            "condition_widths": condition_widths,
            "active_pft_indices": list(contract.active_pft_indices),
        },
        "production_inventory": {
            "landpoints": len(spec.landpoints),
            "years": spec.last_year - spec.first_year + 1,
            "spatial_split_counts": _split_counts(spec),
            "temporal_splits": {
                name: list(bounds) for name, bounds in spec.temporal_splits.items()
            },
            **transitions,
        },
        "cold_start_scope": dict(cold),
        "parameter_extension_scope": dict(policy["parameter_extension_policy"]),
        "resource_estimate": _resource_estimate(
            transitions["point_years"], policy["resource_basis"]
        ),
        "reuse_capabilities": [
            "one-step fast-day supervision",
            "multi-day and free-rollout state supervision",
            "alternative neural architectures and objectives",
            "streamed train-only normalization and balanced window sampling",
            "spatial, temporal, joint, and complete-chain validation",
            "separate parent-hash-bound parameter or forcing extension datasets",
        ],
        "excluded_scope": [
            "48 half-hour internal-state supervision",
            "fully neural Day 1 cold-start initialization",
            "independent parameter-response identification from baseline data alone",
            "generalization claims for unseen forcing products",
        ],
    }
    return _write_json(Path(output_path).resolve(), report)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--contract-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-production-dataset", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = admit_data_product(
        policy_path=args.policy,
        contract_manifest_path=args.contract_manifest,
        output_path=args.output,
        require_production_dataset=args.require_production_dataset,
    )
    print(json.dumps({"admission_report": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
