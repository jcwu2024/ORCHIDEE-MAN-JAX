"""Losslessly migrate canonical Teacher datasets from contract v4 to v5."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from research.daily_coarse_graining.daily_markov_contract import (
    daily_markov_contract_from_metadata,
    load_markov_shard,
    upgrade_v4_contract_to_v5,
    upgrade_v4_fast_day_target_to_v5,
)
from research.daily_coarse_graining.markov_dataset import (
    DATASET_SCHEMA_VERSION,
    load_dataset_index,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: MappingLike) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


MappingLike = dict[str, Any]


def _write_migrated_shard(
    source: Path,
    destination: Path,
    *,
    v4_contract,
    v5_contract,
) -> str:
    with np.load(source, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    upgraded, observed_contract = upgrade_v4_fast_day_target_to_v5(
        arrays["fast_day_target"],
        arrays["state_trajectory"],
        v4_contract,
    )
    if observed_contract != v5_contract:
        raise ValueError("v5 contract drift while migrating shard")
    arrays["fast_day_target"] = upgraded
    if destination.is_file():
        with np.load(destination, allow_pickle=False) as payload:
            existing = {name: payload[name] for name in payload.files}
        if existing.keys() != arrays.keys() or any(
            existing[name].dtype != expected.dtype
            or existing[name].shape != expected.shape
            or not np.array_equal(existing[name], expected, equal_nan=True)
            for name, expected in arrays.items()
        ):
            raise ValueError(
                f"existing migrated shard is not derived from current source: {destination}"
            )
        load_markov_shard(destination, contract=v5_contract)
        return _sha256_file(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, destination)
    load_markov_shard(destination, contract=v5_contract)
    return _sha256_file(destination)


def migrate_dataset(
    source_manifest: Path,
    output_root: Path,
    *,
    dataset_id: str | None = None,
) -> Path:
    source_manifest = source_manifest.resolve()
    output_root = output_root.resolve()
    source_raw = json.loads(source_manifest.read_text(encoding="utf-8"))
    source_index = load_dataset_index(source_manifest, verify_hashes=True)
    if "markov_contract" not in source_raw:
        raise ValueError("source dataset manifest does not embed its contract")
    v4_contract = daily_markov_contract_from_metadata(source_raw["markov_contract"])
    if v4_contract.schema_version != "daily_markov_contract_v4":
        raise ValueError("source dataset is not a v4 Markov contract")
    if source_index.contract_sha256 != v4_contract.sha256:
        raise ValueError("source dataset contract hash does not match metadata")
    v5_contract = upgrade_v4_contract_to_v5(v4_contract)

    shards = []
    for reference in source_index.shards:
        relative = Path("shards") / reference.landpoint_id / f"{reference.year}.npz"
        destination = output_root / relative
        shard_hash = _write_migrated_shard(
            reference.path,
            destination,
            v4_contract=v4_contract,
            v5_contract=v5_contract,
        )
        shards.append(
            {
                "landpoint_id": reference.landpoint_id,
                "year": reference.year,
                "spatial_split": reference.spatial_split,
                "temporal_split": reference.temporal_split,
                "markov_contract_sha256": v5_contract.sha256,
                "source_shard_sha256": reference.sha256,
                "shard": relative.as_posix(),
                "shard_sha256": shard_hash,
            }
        )

    target_id = dataset_id or f"{source_index.dataset_id}-leaf-ci-v5"
    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_id": target_id,
        "status": "complete",
        "provisional_teacher": bool(source_raw.get("provisional_teacher", True)),
        "teacher_git_head": source_index.teacher_git_head,
        "landpoint_count": len({item["landpoint_id"] for item in shards}),
        "year_count": len({item["year"] for item in shards}),
        "shard_count": len(shards),
        "markov_contract_sha256": v5_contract.sha256,
        "markov_contract": v5_contract.metadata(),
        "derived_migration": {
            "source_dataset_id": source_index.dataset_id,
            "source_manifest": str(source_manifest),
            "source_contract_sha256": v4_contract.sha256,
            "policy": (
                "preserve every v4 B_fast column and append "
                "S[d+1].sechiba_finalize_state.leaf_ci"
            ),
            "teacher_rerun": False,
        },
        "shards": shards,
    }
    output_manifest = output_root / "dataset_manifest.json"
    _atomic_json(output_manifest, manifest)
    migrated_index = load_dataset_index(output_manifest, verify_hashes=True)
    if len(migrated_index.shards) != len(source_index.shards):
        raise RuntimeError("migrated dataset shard count changed")
    return output_manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = migrate_dataset(
        args.source_manifest,
        args.output_root,
        dataset_id=args.dataset_id,
    )
    print(json.dumps({"status": "passed", "manifest": str(manifest)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
