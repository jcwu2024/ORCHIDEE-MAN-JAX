"""Freeze, stage, and plan production-scale Daily Teacher datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from jax_orchidee.driver.reference_layout import resolve_paper_landpoint_reference
from jax_orchidee.driver.run_def_materialization import (
    materialize_run_def_values,
    read_run_def_values,
    write_materialized_run_def,
)
from research.daily_coarse_graining import teacher_shards

ROOT = Path(__file__).resolve().parents[2]
SPEC_SCHEMA_VERSION = "daily_teacher_production_spec_v1"
ASSET_SCHEMA_VERSION = "daily_teacher_production_assets_v1"
SPLIT_STRATEGY = "normalized_exogenous_maximin_holdout_v1"
DEFAULT_SPLIT_FEATURES = (
    "grid_x",
    "grid_y",
    "vcmax25",
    "maint_resp_slope_c",
    "alloc_min",
    "residence_time",
)
REQUIRED_REFERENCE_FILES = (
    "driver_start.nc",
    "sechiba_start.nc",
    "stomate_start.nc",
    "stomate_restart.nc",
    "stomate_history_1961.nc",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _portable_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


@dataclass(frozen=True)
class ProductionLandpoint:
    landpoint_id: str
    spatial_split: str


@dataclass(frozen=True)
class ProductionSpec:
    path: Path
    dataset_id: str
    population_manifest: Path
    population_manifest_sha256: str
    first_year: int
    last_year: int
    block_size: int
    temporal_splits: Mapping[str, tuple[int, int]]
    split_strategy: str
    split_features: tuple[str, ...]
    landpoints: tuple[ProductionLandpoint, ...]

    def temporal_split(self, year: int) -> str:
        matches = [
            name
            for name, (first, last) in self.temporal_splits.items()
            if first <= year <= last
        ]
        if len(matches) != 1:
            raise ValueError(f"year {year} maps to {len(matches)} temporal splits")
        return matches[0]


def _normalized_features(
    rows: Sequence[Mapping[str, Any]], feature_names: Sequence[str]
) -> np.ndarray:
    values = np.asarray(
        [[float(row[name]) for name in feature_names] for row in rows],
        dtype=np.float64,
    )
    lower = np.min(values, axis=0)
    span = np.max(values, axis=0) - lower
    span = np.where(span > 0.0, span, 1.0)
    return (values - lower) / span


def _maximin_order(
    rows: Sequence[Mapping[str, Any]],
    feature_names: Sequence[str],
    count: int,
) -> tuple[int, ...]:
    if count < 0 or count > len(rows):
        raise ValueError("maximin count must be within the population")
    if count == 0:
        return ()
    features = _normalized_features(rows, feature_names)
    center_distance = np.sum((features - 0.5) ** 2, axis=1)
    first = max(
        range(len(rows)),
        key=lambda index: (center_distance[index], str(rows[index]["landpoint_id"])),
    )
    selected = [first]
    minimum_distance = np.sum((features - features[first]) ** 2, axis=1)
    minimum_distance[first] = -1.0
    while len(selected) < count:
        index = max(
            range(len(rows)),
            key=lambda candidate: (
                minimum_distance[candidate],
                str(rows[candidate]["landpoint_id"]),
            ),
        )
        selected.append(index)
        distance = np.sum((features - features[index]) ** 2, axis=1)
        minimum_distance = np.minimum(minimum_distance, distance)
        minimum_distance[selected] = -1.0
    return tuple(selected)


def freeze_spec(
    population_manifest: str | Path,
    output: str | Path,
    *,
    dataset_id: str,
    first_year: int = 1961,
    last_year: int = 2010,
    block_size: int = 7,
    validation_landpoints: int = 67,
    test_landpoints: int = 67,
    temporal_splits: Mapping[str, tuple[int, int]] | None = None,
    split_features: Sequence[str] = DEFAULT_SPLIT_FEATURES,
    landpoint_ids: Sequence[str] | None = None,
) -> Path:
    population_manifest = Path(population_manifest).resolve()
    raw = json.loads(population_manifest.read_text(encoding="utf-8"))
    rows_by_id = {str(row["landpoint_id"]): row for row in raw["selected"]}
    population_ids = tuple(str(value) for value in raw["selected_landpoint_ids"])
    if len(population_ids) != len(set(population_ids)) or set(population_ids) != set(rows_by_id):
        raise ValueError("population manifest landpoint inventory is inconsistent")
    ids = population_ids if landpoint_ids is None else tuple(str(value) for value in landpoint_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("requested production landpoint IDs must be unique")
    unknown = sorted(set(ids) - set(population_ids))
    if unknown:
        raise ValueError(f"requested landpoints are absent from the population: {unknown}")
    if first_year != 1961:
        raise ValueError("production chains currently require a canonical 1961 cold start")
    if first_year > last_year or block_size < 2:
        raise ValueError("invalid production year range or block size")
    if validation_landpoints < 0 or test_landpoints < 0:
        raise ValueError("validation and test landpoint counts must be non-negative")
    if validation_landpoints + test_landpoints >= len(ids):
        raise ValueError("spatial holdouts leave no training landpoints")
    rows = [rows_by_id[landpoint_id] for landpoint_id in ids]
    holdout_order = _maximin_order(
        rows,
        split_features,
        validation_landpoints + test_landpoints,
    )
    validation_indices: set[int] = set()
    test_indices: set[int] = set()
    for index in holdout_order:
        validation_fraction = (
            len(validation_indices) / validation_landpoints
            if validation_landpoints
            else float("inf")
        )
        test_fraction = (
            len(test_indices) / test_landpoints
            if test_landpoints
            else float("inf")
        )
        if validation_fraction <= test_fraction:
            validation_indices.add(index)
        else:
            test_indices.add(index)
    if len(validation_indices) != validation_landpoints or len(test_indices) != test_landpoints:
        raise AssertionError("maximin holdout assignment did not satisfy requested counts")
    assignments = []
    for index, landpoint_id in enumerate(ids):
        split = (
            "validation"
            if index in validation_indices
            else "test"
            if index in test_indices
            else "train"
        )
        assignments.append({"id": landpoint_id, "spatial_split": split})
    if temporal_splits is None:
        if last_year - first_year + 1 < 7:
            temporal_splits = {
                "train": (first_year, last_year),
                "validation": (first_year, first_year - 1),
                "test": (first_year, first_year - 1),
            }
        else:
            temporal_splits = {
                "train": (first_year, last_year - 6),
                "validation": (last_year - 5, last_year - 3),
                "test": (last_year - 2, last_year),
            }
    payload = {
        "schema_version": SPEC_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "population_manifest": _portable_path(population_manifest),
        "population_manifest_sha256": _sha256_file(population_manifest),
        "population_count": len(population_ids),
        "first_year": first_year,
        "last_year": last_year,
        "block_size": block_size,
        "temporal_splits": {
            name: [int(bounds[0]), int(bounds[1])]
            for name, bounds in temporal_splits.items()
        },
        "spatial_split_strategy": SPLIT_STRATEGY,
        "spatial_split_features": list(split_features),
        "spatial_split_counts": {
            name: sum(item["spatial_split"] == name for item in assignments)
            for name in sorted(teacher_shards.SPLITS)
        },
        "landpoints": assignments,
        "required_reference_files": list(REQUIRED_REFERENCE_FILES),
    }
    path = _atomic_json(Path(output).resolve(), payload)
    load_production_spec(path)
    return path


def load_production_spec(path: str | Path) -> ProductionSpec:
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != SPEC_SCHEMA_VERSION:
        raise ValueError(f"production spec schema must be {SPEC_SCHEMA_VERSION!r}")
    population_manifest = _resolve_path(raw["population_manifest"])
    if not population_manifest.is_file():
        raise FileNotFoundError(population_manifest)
    if _sha256_file(population_manifest) != raw["population_manifest_sha256"]:
        raise ValueError("production population manifest hash drift")
    temporal_splits = {
        str(name): (int(bounds[0]), int(bounds[1]))
        for name, bounds in raw["temporal_splits"].items()
    }
    if set(temporal_splits) != teacher_shards.SPLITS:
        raise ValueError("production spec must define train, validation, and test years")
    landpoints = tuple(
        ProductionLandpoint(
            landpoint_id=str(item["id"]),
            spatial_split=str(item["spatial_split"]),
        )
        for item in raw["landpoints"]
    )
    ids = [item.landpoint_id for item in landpoints]
    if len(ids) != len(set(ids)):
        raise ValueError("production landpoint IDs must be unique")
    if any(item.spatial_split not in teacher_shards.SPLITS for item in landpoints):
        raise ValueError("production spec contains an unknown spatial split")
    spec = ProductionSpec(
        path=path,
        dataset_id=str(raw["dataset_id"]),
        population_manifest=population_manifest,
        population_manifest_sha256=str(raw["population_manifest_sha256"]),
        first_year=int(raw["first_year"]),
        last_year=int(raw["last_year"]),
        block_size=int(raw["block_size"]),
        temporal_splits=temporal_splits,
        split_strategy=str(raw["spatial_split_strategy"]),
        split_features=tuple(raw["spatial_split_features"]),
        landpoints=landpoints,
    )
    if spec.first_year != 1961 or spec.first_year > spec.last_year or spec.block_size < 2:
        raise ValueError("production spec requires a valid 1961-starting chain")
    for year in range(spec.first_year, spec.last_year + 1):
        spec.temporal_split(year)
    return spec


def source_inventory(spec: ProductionSpec, reference_root: str | Path) -> dict[str, Any]:
    reference_root = Path(reference_root).resolve()
    entries = []
    total_bytes = 0
    for item in spec.landpoints:
        reference = resolve_paper_landpoint_reference(reference_root, item.landpoint_id)
        if reference.output_dir is None or reference.used_run_def is None:
            raise FileNotFoundError(f"unresolved reference assets for {item.landpoint_id}")
        sources = {
            name: reference.output_dir / name for name in REQUIRED_REFERENCE_FILES
        }
        sources["used_run.def"] = reference.used_run_def
        missing = [name for name, source in sources.items() if not source.is_file()]
        if missing:
            raise FileNotFoundError(f"{item.landpoint_id} is missing production assets: {missing}")
        source_bytes = sum(path.stat().st_size for path in sources.values())
        total_bytes += source_bytes
        entries.append(
            {
                "landpoint_id": item.landpoint_id,
                "spatial_split": item.spatial_split,
                "source_bytes": source_bytes,
                "sources": {name: str(path) for name, path in sources.items()},
            }
        )
    return {
        "dataset_id": spec.dataset_id,
        "landpoint_count": len(entries),
        "year_count": spec.last_year - spec.first_year + 1,
        "point_year_count": len(entries) * (spec.last_year - spec.first_year + 1),
        "source_asset_bytes": total_bytes,
        "landpoints": entries,
    }


def stage_assets(
    spec: ProductionSpec,
    *,
    reference_root: str | Path,
    destination: str | Path,
) -> Path:
    inventory = source_inventory(spec, reference_root)
    destination = Path(destination).resolve()
    records = []
    for item in inventory["landpoints"]:
        landpoint_root = destination / "landpoints" / item["landpoint_id"]
        reference_dir = landpoint_root / "reference"
        reference_dir.mkdir(parents=True, exist_ok=True)
        files = {}
        for name, source_name in item["sources"].items():
            source = Path(source_name)
            target = (
                landpoint_root / "used_run.def"
                if name == "used_run.def"
                else reference_dir / name
            )
            shutil.copy2(source, target)
            files[name] = {
                "path": target.relative_to(destination).as_posix(),
                "bytes": target.stat().st_size,
                "sha256": _sha256_file(target),
            }
        records.append(
            {
                "landpoint_id": item["landpoint_id"],
                "spatial_split": item["spatial_split"],
                "files": files,
            }
        )
    payload = {
        "schema_version": ASSET_SCHEMA_VERSION,
        "dataset_id": spec.dataset_id,
        "production_spec_sha256": _sha256_file(spec.path),
        "landpoint_count": len(records),
        "landpoints": records,
    }
    return _atomic_json(destination / "asset_manifest.json", payload)


def verify_staged_assets(spec: ProductionSpec, asset_root: str | Path) -> dict[str, Any]:
    asset_root = Path(asset_root).resolve()
    raw = json.loads((asset_root / "asset_manifest.json").read_text(encoding="utf-8"))
    if raw.get("schema_version") != ASSET_SCHEMA_VERSION:
        raise ValueError(f"asset schema must be {ASSET_SCHEMA_VERSION!r}")
    if raw.get("dataset_id") != spec.dataset_id:
        raise ValueError("production asset dataset identity mismatch")
    if raw.get("production_spec_sha256") != _sha256_file(spec.path):
        raise ValueError("production asset spec hash drift")
    expected = {item.landpoint_id: item.spatial_split for item in spec.landpoints}
    observed = {item["landpoint_id"]: item["spatial_split"] for item in raw["landpoints"]}
    if observed != expected:
        raise ValueError("production asset landpoint inventory mismatch")
    total_bytes = 0
    file_count = 0
    for item in raw["landpoints"]:
        for metadata in item["files"].values():
            path = asset_root / metadata["path"]
            if not path.is_file() or _sha256_file(path) != metadata["sha256"]:
                raise ValueError(f"production asset hash mismatch: {path}")
            if path.stat().st_size != metadata["bytes"]:
                raise ValueError(f"production asset size mismatch: {path}")
            total_bytes += path.stat().st_size
            file_count += 1
    return {
        "dataset_id": spec.dataset_id,
        "landpoint_count": len(observed),
        "verified_files": file_count,
        "verified_bytes": total_bytes,
    }


def build_generation_plan(
    spec: ProductionSpec,
    *,
    asset_root: str | Path,
    teacher_config: str | Path,
    output_root: str | Path,
    plan_path: str | Path,
    days: int | None = None,
) -> Path:
    asset_root = Path(asset_root).resolve()
    verify_staged_assets(spec, asset_root)
    if days is not None and spec.first_year != spec.last_year:
        raise ValueError("bounded --days plans must contain exactly one year")
    if days is not None and days < 2:
        raise ValueError("bounded cold-start production plans require at least two days")
    entries = []
    for item in spec.landpoints:
        landpoint_root = asset_root / "landpoints" / item.landpoint_id
        archived_run_def = landpoint_root / "used_run.def"
        runtime_run_def = write_materialized_run_def(
            materialize_run_def_values(read_run_def_values(archived_run_def)),
            landpoint_root / "runtime_used_run.def",
            header_lines=(
                "# Runtime materialization of archived Fortran getin truth.",
                f"# Source: {archived_run_def}",
            ),
        )
        for year in range(spec.first_year, spec.last_year + 1):
            calendar_days = 366 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 365
            entry_days = calendar_days if days is None else min(int(days), calendar_days)
            entries.append(
                {
                    "landpoint_id": item.landpoint_id,
                    "year": year,
                    "days": entry_days,
                    "initialization_mode": (
                        teacher_shards.COLD_START_BOOTSTRAP
                        if year == spec.first_year
                        else teacher_shards.YEAR_START_CHECKPOINT
                    ),
                    "spatial_split": item.spatial_split,
                    "temporal_split": spec.temporal_split(year),
                    "run_def": str(runtime_run_def),
                    "reference_run_dir": str(landpoint_root / "reference"),
                }
            )
    payload = {
        "schema_version": teacher_shards.SCHEMA_VERSION,
        "dataset_id": spec.dataset_id,
        "teacher_config": str(Path(teacher_config).resolve()),
        "output_root": str(Path(output_root).resolve()),
        "block_size": spec.block_size,
        "entries": entries,
    }
    plan_path = _atomic_json(Path(plan_path).resolve(), payload)
    teacher_shards.load_plan(plan_path, require_inputs=True)
    return plan_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--population-manifest", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--dataset-id", required=True)
    freeze.add_argument("--first-year", type=int, default=1961)
    freeze.add_argument("--last-year", type=int, default=2010)
    freeze.add_argument("--block-size", type=int, default=7)
    freeze.add_argument("--validation-landpoints", type=int, default=67)
    freeze.add_argument("--test-landpoints", type=int, default=67)
    freeze.add_argument("--landpoint-id", action="append", dest="landpoint_ids")
    for name in ("inventory", "stage", "verify", "plan"):
        command = subparsers.add_parser(name)
        command.add_argument("--spec", type=Path, required=True)
        if name in {"inventory", "stage"}:
            command.add_argument("--reference-root", type=Path, required=True)
        if name == "stage":
            command.add_argument("--destination", type=Path, required=True)
        if name in {"verify", "plan"}:
            command.add_argument("--asset-root", type=Path, required=True)
        if name == "plan":
            command.add_argument("--teacher-config", type=Path, required=True)
            command.add_argument("--output-root", type=Path, required=True)
            command.add_argument("--plan-path", type=Path, required=True)
            command.add_argument("--days", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze":
        result = {
            "production_spec": str(
                freeze_spec(
                    args.population_manifest,
                    args.output,
                    dataset_id=args.dataset_id,
                    first_year=args.first_year,
                    last_year=args.last_year,
                    block_size=args.block_size,
                    validation_landpoints=args.validation_landpoints,
                    test_landpoints=args.test_landpoints,
                    landpoint_ids=args.landpoint_ids,
                )
            )
        }
    else:
        spec = load_production_spec(args.spec)
        if args.command == "inventory":
            result = source_inventory(spec, args.reference_root)
        elif args.command == "stage":
            result = {
                "asset_manifest": str(
                    stage_assets(
                        spec,
                        reference_root=args.reference_root,
                        destination=args.destination,
                    )
                )
            }
        elif args.command == "verify":
            result = verify_staged_assets(spec, args.asset_root)
        else:
            result = {
                "generation_plan": str(
                    build_generation_plan(
                        spec,
                        asset_root=args.asset_root,
                        teacher_config=args.teacher_config,
                        output_root=args.output_root,
                        plan_path=args.plan_path,
                        days=args.days,
                    )
                )
            }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
