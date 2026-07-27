"""Frozen sampling, validation, and I/O protocol for the 669-point dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

from research.daily_coarse_graining.daily_markov_contract import load_markov_shard
from research.daily_coarse_graining.markov_dataset import (
    MarkovDatasetIndex,
    MarkovShardRef,
    collate_samples,
    load_dataset_index,
)
from research.daily_coarse_graining.teacher_production import ROOT

PROTOCOL_SCHEMA_VERSION = "daily_teacher_669_training_protocol_v1"
PREFLIGHT_SCHEMA_VERSION = "daily_teacher_streaming_preflight_v1"


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


@dataclass(frozen=True)
class TrainingProtocol:
    path: Path
    raw: Mapping[str, Any]
    sha256: str
    max_open_shards: int


def load_training_protocol(path: str | Path) -> TrainingProtocol:
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError(f"training protocol schema must be {PROTOCOL_SCHEMA_VERSION!r}")
    parent = raw["parent_data_product"]
    policy_path = _resolve(parent["policy"])
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    if _canonical_json_sha256(policy) != parent["policy_sha256"]:
        raise ValueError("training protocol parent policy hash drift")
    if parent["contract_sha256"] != policy["expected_contract"]["sha256"]:
        raise ValueError("training protocol contract hash drift")

    normalization = raw["normalization"]
    if normalization.get("fit_on") != {
        "spatial_split": "train",
        "temporal_split": "train",
    }:
        raise ValueError("normalization must use only train-point/train-year samples")
    sampling = raw["sampling"]
    if sampling.get("strategy") != "exact_once_hierarchical_shard_pool_v1":
        raise ValueError("unsupported production sampling strategy")
    if sampling.get("balance_units") != ["landpoint", "year", "day"]:
        raise ValueError("production sampling must balance landpoint, year, and day")
    max_open_shards = int(sampling["max_open_shards"])
    if max_open_shards < 1:
        raise ValueError("max_open_shards must be positive")

    validation = raw["validation"]
    required_slices = {"temporal", "spatial", "joint", "complete_chain", "sealed_final"}
    if set(validation) != required_slices:
        raise ValueError("training protocol validation slice inventory mismatch")
    for name in ("temporal", "spatial", "joint", "complete_chain"):
        if validation[name].get("model_selection") is not True:
            raise ValueError(f"{name} must be available for model selection")
        if any(value == "test" for value in validation[name].values()):
            raise ValueError(f"{name} must not consume a sealed test split")
    if validation["sealed_final"].get("model_selection") is not False:
        raise ValueError("sealed_final must not be used for model selection")

    gates = raw["promotion_gates"]
    if [gate["id"] for gate in gates] != [
        "one_step",
        "multiday",
        "annual_stability",
        "complete_chain",
    ]:
        raise ValueError("training promotion gate order mismatch")
    return TrainingProtocol(
        path=path,
        raw=raw,
        sha256=_canonical_json_sha256(raw),
        max_open_shards=max_open_shards,
    )


def balanced_reference_order(
    index: MarkovDatasetIndex,
    *,
    seed: int,
    spatial_split: str = "train",
    temporal_split: str = "train",
) -> tuple[MarkovShardRef, ...]:
    """Order every selected shard once while round-robining landpoints."""

    selected = index.select(
        spatial_split=spatial_split,
        temporal_split=temporal_split,
    )
    if not selected:
        raise ValueError("balanced sampling selection contains no shards")
    by_landpoint: dict[str, list[MarkovShardRef]] = {}
    for reference in selected:
        by_landpoint.setdefault(reference.landpoint_id, []).append(reference)
    rng = np.random.default_rng(seed)
    pending = {
        landpoint: [
            values[index]
            for index in rng.permutation(len(values))
        ]
        for landpoint, values in by_landpoint.items()
    }
    ordered = []
    while pending:
        active = sorted(pending)
        for offset in rng.permutation(len(active)):
            landpoint = active[int(offset)]
            ordered.append(pending[landpoint].pop())
            if not pending[landpoint]:
                del pending[landpoint]
    return tuple(ordered)


def balanced_samples(
    index: MarkovDatasetIndex,
    *,
    seed: int,
    max_open_shards: int,
    spatial_split: str = "train",
    temporal_split: str = "train",
    max_samples_per_shard: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield exact-once samples from a bounded, interleaved shard pool."""

    if max_open_shards < 1:
        raise ValueError("max_open_shards must be positive")
    if max_samples_per_shard is not None and max_samples_per_shard < 1:
        raise ValueError("max_samples_per_shard must be positive")
    references = balanced_reference_order(
        index,
        seed=seed,
        spatial_split=spatial_split,
        temporal_split=temporal_split,
    )
    rng = np.random.default_rng(seed)
    for pool_start in range(0, len(references), max_open_shards):
        pool_references = references[pool_start : pool_start + max_open_shards]
        pool = []
        for reference in pool_references:
            shard = load_markov_shard(reference.path)
            day_order = rng.permutation(shard.days)
            if max_samples_per_shard is not None:
                day_order = day_order[:max_samples_per_shard]
            pool.append((reference, shard, day_order))
        longest = max(len(day_order) for _, _, day_order in pool)
        for position in range(longest):
            for reference, shard, day_order in pool:
                if position >= len(day_order):
                    continue
                yield shard.sample(int(day_order[position])) | {
                    "sample_landpoint_id": reference.landpoint_id,
                    "sample_year": reference.year,
                    "sample_spatial_split": reference.spatial_split,
                    "sample_temporal_split": reference.temporal_split,
                }


def _process_peak_rss_bytes() -> int | None:
    if Path("/proc/self/status").is_file():
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    try:
        import resource
    except ImportError:
        return None
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if platform.system() == "Darwin" else peak * 1024


def _npz_uncompressed_bytes(path: Path) -> int:
    with zipfile.ZipFile(path) as archive:
        return sum(item.file_size for item in archive.infolist())


def _tree_nbytes(value: Any) -> int:
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, Mapping):
        return sum(_tree_nbytes(child) for child in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tree_nbytes(child) for child in value)
    return 0


def streaming_io_preflight(
    *,
    manifest_path: str | Path,
    protocol_path: str | Path,
    output_path: str | Path,
    max_shards: int = 8,
    batch_size: int = 32,
    samples_per_shard: int = 32,
    seed: int = 20260727,
) -> Path:
    """Measure bounded reader throughput on admitted shards without training."""

    if max_shards < 1 or batch_size < 1 or samples_per_shard < 1:
        raise ValueError("preflight limits must be positive")
    protocol = load_training_protocol(protocol_path)
    if max_shards > protocol.max_open_shards:
        raise ValueError("preflight max_shards exceeds the frozen shard-pool bound")
    index = load_dataset_index(manifest_path, verify_hashes=True)
    references = balanced_reference_order(index, seed=seed)[:max_shards]
    subset = MarkovDatasetIndex(
        dataset_id=index.dataset_id,
        teacher_git_head=index.teacher_git_head,
        contract_sha256=index.contract_sha256,
        shards=references,
    )
    compressed_bytes = sum(reference.path.stat().st_size for reference in references)
    uncompressed_bytes = sum(_npz_uncompressed_bytes(reference.path) for reference in references)

    started = time.perf_counter()
    peak_before = _process_peak_rss_bytes()
    sample_count = 0
    batch_count = 0
    max_batch_bytes = 0
    pending = []
    for sample in balanced_samples(
        subset,
        seed=seed,
        max_open_shards=max_shards,
        max_samples_per_shard=samples_per_shard,
    ):
        pending.append(sample)
        sample_count += 1
        if len(pending) == batch_size:
            batch = collate_samples(pending)
            max_batch_bytes = max(max_batch_bytes, _tree_nbytes(batch))
            batch_count += 1
            pending.clear()
    if pending:
        batch = collate_samples(pending)
        max_batch_bytes = max(max_batch_bytes, _tree_nbytes(batch))
        batch_count += 1
    elapsed = time.perf_counter() - started
    peak_after = _process_peak_rss_bytes()
    report = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "passed",
        "dataset": {
            "manifest": str(Path(manifest_path).resolve()),
            "manifest_sha256": _sha256_file(Path(manifest_path).resolve()),
            "dataset_id": index.dataset_id,
            "teacher_git_head": index.teacher_git_head,
            "contract_sha256": index.contract_sha256,
        },
        "protocol": {
            "path": str(protocol.path),
            "sha256": protocol.sha256,
        },
        "reader": {
            "strategy": "exact_once_hierarchical_shard_pool_v1",
            "selected_shards": len(references),
            "max_open_shards": max_shards,
            "samples_per_shard": samples_per_shard,
            "sample_count": sample_count,
            "batch_size": batch_size,
            "batch_count": batch_count,
            "compressed_bytes": compressed_bytes,
            "estimated_uncompressed_pool_bytes": uncompressed_bytes,
            "max_collated_batch_bytes": max_batch_bytes,
            "elapsed_seconds": elapsed,
            "samples_per_second": sample_count / elapsed,
            "peak_rss_before_bytes": peak_before,
            "peak_rss_after_bytes": peak_after,
        },
    }
    return _atomic_json(Path(output_path).resolve(), report)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--protocol", type=Path, required=True)
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--dataset", type=Path, required=True)
    preflight.add_argument("--protocol", type=Path, required=True)
    preflight.add_argument("--output", type=Path, required=True)
    preflight.add_argument("--max-shards", type=int, default=8)
    preflight.add_argument("--batch-size", type=int, default=32)
    preflight.add_argument("--samples-per-shard", type=int, default=32)
    preflight.add_argument("--seed", type=int, default=20260727)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        protocol = load_training_protocol(args.protocol)
        print(json.dumps({"status": "passed", "protocol_sha256": protocol.sha256}, indent=2))
        return 0
    output = streaming_io_preflight(
        manifest_path=args.dataset,
        protocol_path=args.protocol,
        output_path=args.output,
        max_shards=args.max_shards,
        batch_size=args.batch_size,
        samples_per_shard=args.samples_per_shard,
        seed=args.seed,
    )
    print(json.dumps({"status": "passed", "report": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
