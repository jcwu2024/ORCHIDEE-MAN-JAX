"""Fit or merge the first process-head train/train statistics artifact."""

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
    parser.add_argument("--output", type=Path, required=True)


def _load_context(args):
    from research.daily_coarse_graining.daily_markov_contract import (
        daily_markov_contract_from_metadata,
    )
    from research.daily_coarse_graining.daily_operator.training_contract import (
        load_first_candidate_contract,
    )
    from research.daily_coarse_graining.daily_operator.training_data import (
        DailyProcessTrainingAdapter,
    )
    from research.daily_coarse_graining.markov_dataset import load_training_statistics
    from research.daily_coarse_graining.typed_sidecar import (
        load_coherent_typed_sidecar_index,
    )

    candidate = load_first_candidate_contract()
    data = candidate.payload["data"]
    dataset_path = args.dataset_manifest.resolve()
    acceptance_path = args.acceptance_report.resolve()
    base_path = args.base_statistics.resolve()
    base_metadata = json.loads(base_path.read_text(encoding="utf-8"))
    base_npz = base_path.parent / str(base_metadata["statistics_npz"])
    observed_hashes = {
        "dataset_manifest_sha256": _sha256(dataset_path),
        "acceptance_report_sha256": _sha256(acceptance_path),
        "training_statistics_json_sha256": _sha256(base_path),
        "training_statistics_npz_sha256": _sha256(base_npz),
    }
    drift = {
        name: (value, data[name])
        for name, value in observed_hashes.items()
        if value != data[name]
    }
    if drift:
        raise ValueError(f"first-candidate accepted artifact hash drift: {drift}")
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if acceptance.get("status") != "passed":
        raise ValueError("v6 acceptance report is not passed")
    manifest = json.loads(dataset_path.read_text(encoding="utf-8"))
    contract = daily_markov_contract_from_metadata(manifest["markov_contract"])
    adapter = DailyProcessTrainingAdapter.from_contract(contract)
    index = load_coherent_typed_sidecar_index(dataset_path, verify_hashes=False)
    base_statistics = load_training_statistics(base_path, index=index.parent_index)
    return (
        candidate,
        adapter,
        index,
        base_statistics,
        observed_hashes["training_statistics_json_sha256"],
        observed_hashes["training_statistics_npz_sha256"],
    )


def _fit(args) -> int:
    from research.daily_coarse_graining.daily_operator.training_statistics import (
        fit_process_training_statistics,
        write_process_training_statistics,
    )

    candidate, adapter, index, base, json_hash, npz_hash = _load_context(args)
    started = time.perf_counter()
    statistics = fit_process_training_statistics(
        index,
        adapter,
        base,
        candidate,
        base_statistics_json_sha256=json_hash,
        base_statistics_npz_sha256=npz_hash,
        partition_index=args.partition_index,
        partition_count=args.partition_count,
        chunk_rows=args.chunk_rows,
    )
    output = write_process_training_statistics(statistics, args.output)
    print(
        json.dumps(
            {
                "status": "passed",
                "mode": "fit",
                "partition_index": statistics.partition_index,
                "partition_count": statistics.partition_count,
                "sample_count": statistics.sample_count,
                "source_shard_count": statistics.source_shard_count,
                "output": str(output.resolve()),
                "output_sha256": _sha256(output),
                "npz_sha256": _sha256(output.with_suffix(".npz")),
                "elapsed_seconds": time.perf_counter() - started,
            },
            indent=2,
        )
    )
    return 0


def _merge(args) -> int:
    from research.daily_coarse_graining.daily_operator.training_statistics import (
        load_process_training_statistics,
        merge_process_training_statistics,
        write_process_training_statistics,
    )

    candidate, adapter, index, _base, _json_hash, _npz_hash = _load_context(args)
    paths = tuple(sorted(args.partials_dir.resolve().glob("part-*.json")))
    if not paths:
        raise ValueError("partials directory contains no part-*.json artifacts")
    partials = tuple(
        load_process_training_statistics(
            path,
            index=index,
            adapter=adapter,
            candidate_contract=candidate,
            base_statistics=_base,
        )
        for path in paths
    )
    statistics = merge_process_training_statistics(
        partials,
        index=index,
        adapter=adapter,
        candidate_contract=candidate,
    )
    output = write_process_training_statistics(statistics, args.output)
    load_process_training_statistics(
        output,
        index=index,
        adapter=adapter,
        candidate_contract=candidate,
        base_statistics=_base,
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "mode": "merge",
                "partial_count": len(partials),
                "sample_count": statistics.sample_count,
                "source_shard_count": statistics.source_shard_count,
                "output": str(output.resolve()),
                "output_sha256": _sha256(output),
                "npz_sha256": _sha256(output.with_suffix(".npz")),
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
