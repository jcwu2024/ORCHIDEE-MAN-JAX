"""Fit or merge train/train statistics for the 79 added fast-owner values."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path, required=True)
    parser.add_argument("--base-statistics", type=Path, required=True)
    parser.add_argument("--process-statistics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)


def _load_context(args):
    from research.daily_coarse_graining.daily_markov_contract import (
        daily_markov_contract_from_metadata,
    )
    from research.daily_coarse_graining.daily_operator.fast_owner_reconstruction import (
        load_fast_owner_reconstruction_contract,
    )
    from research.daily_coarse_graining.daily_operator.successor_contract import (
        load_successor_candidate_contract,
    )
    from research.daily_coarse_graining.daily_operator.successor_fast_owner_decoder import (
        build_successor_fast_owner_schema,
    )
    from research.daily_coarse_graining.daily_operator.training_contract import (
        load_first_candidate_contract,
    )
    from research.daily_coarse_graining.daily_operator.training_data import (
        DailyProcessTrainingAdapter,
    )
    from research.daily_coarse_graining.daily_operator.training_statistics import (
        load_process_training_statistics,
    )
    from research.daily_coarse_graining.markov_dataset import load_training_statistics
    from research.daily_coarse_graining.typed_sidecar import (
        load_coherent_typed_sidecar_index,
    )

    successor = load_successor_candidate_contract()
    first = load_first_candidate_contract()
    dataset_path = args.dataset_manifest.resolve()
    acceptance_path = args.acceptance_report.resolve()
    base_path = args.base_statistics.resolve()
    process_path = args.process_statistics.resolve()
    process_metadata = json.loads(process_path.read_text(encoding="utf-8"))
    process_npz = process_path.parent / str(process_metadata["statistics_npz"])
    observed = {
        "dataset_manifest_sha256": _sha256(dataset_path),
        "acceptance_report_sha256": _sha256(acceptance_path),
        "training_statistics_json_sha256": _sha256(process_path),
        "training_statistics_npz_sha256": _sha256(process_npz),
    }
    expected = successor.payload["data"]
    drift = {
        name: (value, expected[name])
        for name, value in observed.items()
        if value != expected[name]
    }
    if drift:
        raise ValueError(f"successor accepted artifact hash drift: {drift}")
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if acceptance.get("status") != "passed":
        raise ValueError("v6 acceptance report is not passed")
    manifest = json.loads(dataset_path.read_text(encoding="utf-8"))
    contract = daily_markov_contract_from_metadata(manifest["markov_contract"])
    adapter = DailyProcessTrainingAdapter.from_contract(contract)
    index = load_coherent_typed_sidecar_index(dataset_path, verify_hashes=False)
    base = load_training_statistics(base_path, index=index.parent_index)
    process = load_process_training_statistics(
        process_path,
        index=index,
        adapter=adapter,
        candidate_contract=first,
        base_statistics=base,
    )
    owner = load_fast_owner_reconstruction_contract(markov_contract=contract)
    schema = build_successor_fast_owner_schema(
        process.schema,
        markov_contract=contract,
        owner_contract=owner,
    )
    return successor, owner, schema, adapter, index, process


def _fit(args) -> int:
    from research.daily_coarse_graining.daily_operator.successor_fast_owner_statistics import (
        fit_successor_fast_owner_statistics,
        write_successor_fast_owner_statistics,
    )

    successor, owner, schema, adapter, index, process = _load_context(args)
    started = time.perf_counter()
    statistics = fit_successor_fast_owner_statistics(
        index,
        adapter,
        process,
        successor,
        owner,
        schema,
        partition_index=args.partition_index,
        partition_count=args.partition_count,
        chunk_rows=args.chunk_rows,
    )
    output = write_successor_fast_owner_statistics(statistics, args.output)
    print(
        json.dumps(
            {
                "status": "passed",
                "mode": "fit",
                "partition_index": statistics.partition_index,
                "partition_count": statistics.partition_count,
                "sample_count": statistics.sample_count,
                "source_shard_count": statistics.source_shard_count,
                "rveget_defined_count": statistics.rveget_defined_count,
                "rveget_eligible_count": statistics.rveget_eligible_count,
                "output": str(output.resolve()),
                "output_sha256": _sha256(output),
                "npz_sha256": _sha256(output.with_suffix(".npz")),
                "elapsed_seconds": time.perf_counter() - started,
                "teacher_regenerated": False,
                "historical_v5_used": False,
            },
            indent=2,
        )
    )
    return 0


def _merge(args) -> int:
    from research.daily_coarse_graining.daily_operator.successor_fast_owner_statistics import (
        load_successor_fast_owner_statistics,
        merge_successor_fast_owner_statistics,
        write_successor_fast_owner_statistics,
    )

    successor, owner, schema, adapter, index, process = _load_context(args)
    paths = tuple(sorted(args.partials_dir.resolve().glob("part-*.json")))
    if not paths:
        raise ValueError("partials directory contains no part-*.json artifacts")
    partials = tuple(
        load_successor_fast_owner_statistics(
            path,
            index=index,
            adapter=adapter,
            base=process,
            successor_contract=successor,
            owner_contract=owner,
            schema=schema,
        )
        for path in paths
    )
    statistics = merge_successor_fast_owner_statistics(
        partials,
        index=index,
        adapter=adapter,
        base=process,
        successor_contract=successor,
        owner_contract=owner,
        schema=schema,
    )
    output = write_successor_fast_owner_statistics(statistics, args.output)
    load_successor_fast_owner_statistics(
        output,
        index=index,
        adapter=adapter,
        base=process,
        successor_contract=successor,
        owner_contract=owner,
        schema=schema,
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "mode": "merge",
                "partial_count": len(partials),
                "sample_count": statistics.sample_count,
                "source_shard_count": statistics.source_shard_count,
                "rveget_defined_count": statistics.rveget_defined_count,
                "rveget_eligible_count": statistics.rveget_eligible_count,
                "output": str(output.resolve()),
                "output_sha256": _sha256(output),
                "npz_sha256": _sha256(output.with_suffix(".npz")),
                "teacher_regenerated": False,
                "historical_v5_used": False,
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit = subparsers.add_parser("fit")
    _common(fit)
    fit.add_argument("--partition-index", type=int, required=True)
    fit.add_argument("--partition-count", type=int, required=True)
    fit.add_argument("--chunk-rows", type=int, default=64)
    fit.set_defaults(handler=_fit)
    merge = subparsers.add_parser("merge")
    _common(merge)
    merge.add_argument("--partials-dir", type=Path, required=True)
    merge.set_defaults(handler=_merge)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
