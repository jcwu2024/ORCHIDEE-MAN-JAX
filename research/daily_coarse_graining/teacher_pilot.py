"""Freeze and stage a bounded multi-landpoint Daily Teacher pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from jax_orchidee.driver.reference_layout import resolve_paper_landpoint_reference
from research.daily_coarse_graining import teacher_shards

SPEC_SCHEMA_VERSION = "daily_teacher_pilot_spec_v1"
ASSET_SCHEMA_VERSION = "daily_teacher_pilot_assets_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class PilotLandpoint:
    landpoint_id: str
    spatial_split: str
    role: str


@dataclass(frozen=True)
class PilotSpec:
    pilot_id: str
    first_year: int
    last_year: int
    block_size: int
    temporal_splits: Mapping[str, tuple[int, int]]
    landpoints: tuple[PilotLandpoint, ...]
    required_reference_files: tuple[str, ...]
    checkpoint_name: str

    def temporal_split(self, year: int) -> str:
        matches = [name for name, (first, last) in self.temporal_splits.items() if first <= year <= last]
        if len(matches) != 1:
            raise ValueError(f"year {year} maps to {len(matches)} temporal splits")
        return matches[0]


def load_pilot_spec(path: str | Path) -> PilotSpec:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != SPEC_SCHEMA_VERSION:
        raise ValueError(f"pilot schema must be {SPEC_SCHEMA_VERSION!r}")
    landpoints = tuple(
        PilotLandpoint(
            landpoint_id=str(item["id"]),
            spatial_split=str(item["spatial_split"]),
            role=str(item["role"]),
        )
        for item in raw["landpoints"]
    )
    ids = [item.landpoint_id for item in landpoints]
    if len(ids) != len(set(ids)):
        raise ValueError("pilot landpoint IDs must be unique")
    if any(item.spatial_split not in teacher_shards.SPLITS for item in landpoints):
        raise ValueError("pilot contains an unknown spatial split")
    temporal = {str(name): (int(bounds[0]), int(bounds[1])) for name, bounds in raw["temporal_splits"].items()}
    if set(temporal) != teacher_shards.SPLITS:
        raise ValueError("pilot must define train, validation, and test years")
    spec = PilotSpec(
        pilot_id=str(raw["pilot_id"]),
        first_year=int(raw["first_year"]),
        last_year=int(raw["last_year"]),
        block_size=int(raw["block_size"]),
        temporal_splits=temporal,
        landpoints=landpoints,
        required_reference_files=tuple(raw["required_reference_files"]),
        checkpoint_name=str(raw["checkpoint_name"]),
    )
    if spec.first_year > spec.last_year or spec.block_size < 2:
        raise ValueError("pilot year range or block size is invalid")
    for year in range(spec.first_year, spec.last_year + 1):
        spec.temporal_split(year)
    return spec


def _checkpoint_source(root: Path, landpoint_id: str, name: str) -> Path:
    candidates = (
        root / landpoint_id / "compiled_checkpoints" / name,
        root / landpoint_id / name,
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"missing year-start checkpoint for {landpoint_id}")


def source_inventory(
    spec: PilotSpec,
    *,
    reference_root: str | Path,
    checkpoint_root: str | Path,
) -> dict[str, Any]:
    reference_root = Path(reference_root).resolve()
    checkpoint_root = Path(checkpoint_root).resolve()
    entries = []
    total_bytes = 0
    for item in spec.landpoints:
        reference = resolve_paper_landpoint_reference(reference_root, item.landpoint_id)
        if reference.output_dir is None or reference.used_run_def is None:
            raise FileNotFoundError(f"unresolved reference assets for {item.landpoint_id}")
        sources = {name: reference.output_dir / name for name in spec.required_reference_files}
        sources["used_run.def"] = reference.used_run_def
        sources["checkpoint"] = _checkpoint_source(checkpoint_root, item.landpoint_id, spec.checkpoint_name)
        missing = [name for name, path in sources.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{item.landpoint_id} is missing pilot assets: {missing}")
        size = sum(path.stat().st_size for path in sources.values())
        total_bytes += size
        entries.append(
            {
                "landpoint_id": item.landpoint_id,
                "spatial_split": item.spatial_split,
                "role": item.role,
                "source_bytes": size,
                "sources": {name: str(path) for name, path in sources.items()},
            }
        )
    return {
        "pilot_id": spec.pilot_id,
        "landpoint_count": len(entries),
        "year_count": spec.last_year - spec.first_year + 1,
        "point_year_count": len(entries) * (spec.last_year - spec.first_year + 1),
        "source_asset_bytes": total_bytes,
        "landpoints": entries,
    }


def stage_assets(
    spec: PilotSpec,
    *,
    reference_root: str | Path,
    checkpoint_root: str | Path,
    destination: str | Path,
) -> Path:
    inventory = source_inventory(
        spec,
        reference_root=reference_root,
        checkpoint_root=checkpoint_root,
    )
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
                landpoint_root / "checkpoint.pkl"
                if name == "checkpoint"
                else landpoint_root / "used_run.def"
                if name == "used_run.def"
                else reference_dir / name
            )
            target.parent.mkdir(parents=True, exist_ok=True)
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
                "role": item["role"],
                "files": files,
            }
        )
    payload = {
        "schema_version": ASSET_SCHEMA_VERSION,
        "pilot_id": spec.pilot_id,
        "landpoint_count": len(records),
        "landpoints": records,
    }
    manifest = destination / "asset_manifest.json"
    manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify_staged_assets(spec: PilotSpec, asset_root: str | Path) -> dict[str, Any]:
    asset_root = Path(asset_root).resolve()
    manifest_path = asset_root / "asset_manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != ASSET_SCHEMA_VERSION:
        raise ValueError(f"asset schema must be {ASSET_SCHEMA_VERSION!r}")
    if raw.get("pilot_id") != spec.pilot_id:
        raise ValueError("pilot asset identity mismatch")
    expected_ids = {item.landpoint_id for item in spec.landpoints}
    observed_ids = {item["landpoint_id"] for item in raw["landpoints"]}
    if observed_ids != expected_ids:
        raise ValueError("pilot asset landpoint set mismatch")
    total_bytes = 0
    for item in raw["landpoints"]:
        for metadata in item["files"].values():
            path = asset_root / metadata["path"]
            if not path.is_file() or _sha256_file(path) != metadata["sha256"]:
                raise ValueError(f"pilot asset hash mismatch: {path}")
            total_bytes += path.stat().st_size
    return {
        "pilot_id": spec.pilot_id,
        "landpoint_count": len(observed_ids),
        "verified_files": sum(len(item["files"]) for item in raw["landpoints"]),
        "verified_bytes": total_bytes,
    }


def build_generation_plan(
    spec: PilotSpec,
    *,
    asset_root: str | Path,
    teacher_config: str | Path,
    output_root: str | Path,
    plan_path: str | Path,
) -> Path:
    asset_root = Path(asset_root).resolve()
    verify_staged_assets(spec, asset_root)
    entries = []
    for item in spec.landpoints:
        landpoint_root = asset_root / "landpoints" / item.landpoint_id
        for year in range(spec.first_year, spec.last_year + 1):
            entry = {
                "landpoint_id": item.landpoint_id,
                "year": year,
                "days": 366 if year % 4 == 0 else 365,
                "spatial_split": item.spatial_split,
                "temporal_split": spec.temporal_split(year),
                "run_def": str(landpoint_root / "used_run.def"),
                "reference_run_dir": str(landpoint_root / "reference"),
            }
            if year == spec.first_year:
                entry["state_cache"] = str(landpoint_root / "checkpoint.pkl")
            entries.append(entry)
    payload = {
        "schema_version": teacher_shards.SCHEMA_VERSION,
        "dataset_id": spec.pilot_id,
        "teacher_config": str(Path(teacher_config).resolve()),
        "output_root": str(Path(output_root).resolve()),
        "block_size": spec.block_size,
        "entries": entries,
    }
    plan_path = Path(plan_path).resolve()
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    teacher_shards.load_plan(plan_path, require_inputs=True)
    return plan_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("--reference-root", type=Path, required=True)
    inventory.add_argument("--checkpoint-root", type=Path, required=True)
    stage = subparsers.add_parser("stage")
    stage.add_argument("--reference-root", type=Path, required=True)
    stage.add_argument("--checkpoint-root", type=Path, required=True)
    stage.add_argument("--destination", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--asset-root", type=Path, required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--asset-root", type=Path, required=True)
    plan.add_argument("--teacher-config", type=Path, required=True)
    plan.add_argument("--output-root", type=Path, required=True)
    plan.add_argument("--plan-path", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    spec = load_pilot_spec(args.spec)
    if args.command == "inventory":
        result = source_inventory(
            spec,
            reference_root=args.reference_root,
            checkpoint_root=args.checkpoint_root,
        )
    elif args.command == "stage":
        result = {
            "asset_manifest": str(
                stage_assets(
                    spec,
                    reference_root=args.reference_root,
                    checkpoint_root=args.checkpoint_root,
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
                )
            )
        }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
