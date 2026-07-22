"""Benchmark bounded batching on verified Daily Markov Teacher shards."""

from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from research.daily_coarse_graining.markov_dataset import (
    MarkovDatasetIndex,
    load_dataset_index,
    prefetched_batches,
)


def _repeated_training_samples(index: MarkovDatasetIndex, repeats: int) -> Iterator[dict[str, Any]]:
    for _ in range(repeats):
        yield from index.samples(spatial_split="train", temporal_split="train")


def _array_bytes(value: Any) -> int:
    if isinstance(value, Mapping):
        return sum(_array_bytes(item) for item in value.values())
    return int(value.nbytes) if isinstance(value, np.ndarray) else 0


def benchmark(
    manifest: str | Path,
    *,
    repeats: int,
    batch_size: int,
    prefetch: int,
) -> dict[str, Any]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    index = load_dataset_index(manifest)
    if not index.select(spatial_split="train", temporal_split="train"):
        raise ValueError("dataset has no train/train shards")

    tracemalloc.start()
    started = time.perf_counter()
    batches = 0
    samples = 0
    largest_batch_bytes = 0
    checksum = 0.0
    for batch in prefetched_batches(
        _repeated_training_samples(index, repeats),
        batch_size,
        prefetch=prefetch,
    ):
        batches += 1
        samples += int(batch["state"].shape[0])
        largest_batch_bytes = max(largest_batch_bytes, _array_bytes(batch))
        checksum += float(np.sum(batch["forcing_native"], dtype=np.float64))
    elapsed = time.perf_counter() - started
    current_bytes, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "schema_version": "daily_markov_dataset_memory_benchmark_v1",
        "dataset_id": index.dataset_id,
        "contract_sha256": index.contract_sha256,
        "selection": {"spatial_split": "train", "temporal_split": "train"},
        "synthetic_repeats": repeats,
        "batch_size": batch_size,
        "prefetch_batches": prefetch,
        "samples": samples,
        "batches": batches,
        "elapsed_seconds": elapsed,
        "samples_per_second": samples / elapsed,
        "largest_collated_batch_bytes": largest_batch_bytes,
        "tracemalloc_current_bytes": current_bytes,
        "tracemalloc_peak_bytes": peak_bytes,
        "checksum": checksum,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--repeats", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    result = benchmark(
        arguments.manifest,
        repeats=arguments.repeats,
        batch_size=arguments.batch_size,
        prefetch=arguments.prefetch,
    )
    rendered = json.dumps(result, indent=2)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
