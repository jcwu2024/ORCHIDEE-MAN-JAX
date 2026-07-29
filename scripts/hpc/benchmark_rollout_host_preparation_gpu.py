"""Measure multi-landpoint rollout preparation time and host-memory growth."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import jax

from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend


def _process_memory_bytes() -> dict[str, int]:
    values = {}
    with Path("/proc/self/status").open(encoding="utf-8") as handle:
        for line in handle:
            name, _, raw = line.partition(":")
            if name in {"VmRSS", "VmHWM"}:
                values[name] = int(raw.strip().split()[0]) * 1024
    return {
        "rss_bytes": values["VmRSS"],
        "peak_rss_bytes": values["VmHWM"],
    }


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--landpoints", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--anchor-batch-size", type=int, default=256)
    parser.add_argument("--rollout-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.landpoints < 2 or args.horizon < 1:
        raise ValueError("benchmark requires at least two landpoints and one day")
    initialize_gpu_backend()

    from research.daily_coarse_graining.rollout_stability_run import (
        _load_rollout_resources,
        prepare_rollout_update,
    )

    resources = _load_rollout_resources(
        dataset_path=args.dataset.resolve(),
        statistics_path=args.statistics.resolve(),
        plan_path=None if args.plan is None else args.plan.resolve(),
    )
    references = sorted(
        resources.index.select(
            spatial_split="train",
            temporal_split="train",
        ),
        key=lambda item: (item.landpoint_id, item.year, str(item.path)),
    )
    selected = []
    seen = set()
    for reference in references:
        if reference.landpoint_id not in seen:
            selected.append(reference)
            seen.add(reference.landpoint_id)
        if len(selected) == args.landpoints:
            break
    if len(selected) != args.landpoints:
        raise ValueError("dataset has fewer train landpoints than requested")

    runtimes = {}
    baseline = _process_memory_bytes()
    records = []
    for update, reference in enumerate(selected):
        profile = {}
        started = time.perf_counter()
        prepared = prepare_rollout_update(
            reference=reference,
            update=update,
            horizon=args.horizon,
            anchor_batch_size=args.anchor_batch_size,
            rollout_batch_size=args.rollout_batch_size,
            seed=args.seed,
            resources=resources,
            runtime_cache=runtimes,
            timing=profile,
        )
        jax.block_until_ready(
            (
                prepared.anchor_batch,
                prepared.initial_states,
                prepared.initial_discrete_states,
                prepared.sequence,
                prepared.teacher_next_discrete_states,
            )
        )
        records.append(
            {
                "landpoint_id": reference.landpoint_id,
                "year": reference.year,
                "seconds": time.perf_counter() - started,
                "profile": profile,
                "memory": _process_memory_bytes(),
                "runtime_cache_size": len(runtimes),
            }
        )

    warm_started = time.perf_counter()
    warm = prepare_rollout_update(
        reference=selected[0],
        update=len(selected),
        horizon=args.horizon,
        anchor_batch_size=args.anchor_batch_size,
        rollout_batch_size=args.rollout_batch_size,
        seed=args.seed,
        resources=resources,
        runtime_cache=runtimes,
    )
    jax.block_until_ready(
        (
            warm.anchor_batch,
            warm.initial_states,
            warm.initial_discrete_states,
            warm.sequence,
            warm.teacher_next_discrete_states,
        )
    )
    final_memory = _process_memory_bytes()
    output = {
        "schema_version": "rollout_host_preparation_benchmark_v1",
        "status": "passed",
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "sealed_test_used": False,
        "landpoint_count": len(selected),
        "horizon": args.horizon,
        "anchor_batch_size": args.anchor_batch_size,
        "rollout_batch_size": args.rollout_batch_size,
        "baseline_memory": baseline,
        "records": records,
        "warm_cache_seconds": time.perf_counter() - warm_started,
        "final_memory": final_memory,
        "rss_growth_bytes": final_memory["rss_bytes"] - baseline["rss_bytes"],
        "runtime_cache_size": len(runtimes),
    }
    _atomic_json(args.output.resolve(), output)
    print(json.dumps({"status": "passed", "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
