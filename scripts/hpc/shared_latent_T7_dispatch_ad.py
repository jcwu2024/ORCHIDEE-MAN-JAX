"""Synchronization-only scheduling of four unchanged saved B1 AD calls.

No model/process logic is defined here. Mathematical reduction remains in
shared_latent_T7_batch_ad.merge_serial_results and public checkpointed loss.
"""
from __future__ import annotations

import time

MODES = ("block_each", "dispatch_four")


def execute_four(executable, singles, merge, *, mode, block, clock=time.monotonic):
    """Time complete calls and merge, draining both sides of the boundary.

dispatch_four keeps each returned full parameter/state tree alive until the
merge dependency completes. Dispatch timings are not per-sample GPU latency.
"""
    if mode not in MODES or len(singles) != 4:
        raise ValueError("one declared mode and exactly four independent cases required")
    # Both modes begin with the same completed input boundary, outside timing.
    block(tuple(singles))
    began = clock()
    rows, sample_seconds = [], []
    for single in singles:
        sample_start = clock()
        result = executable(*single)
        if mode == "block_each":
            block(result)
        sample_seconds.append(clock() - sample_start)
        rows.append(result)
    merged = merge(*rows)
    block(merged)
    elapsed = clock() - began
    return merged, {"mode": mode, "four_window_seconds": elapsed,
        "sample_seconds": sample_seconds,
        "sample_time_kind": "synchronized_full_AD" if mode == "block_each" else "host_dispatch_only",
        "trajectories_per_second": 4 / elapsed, "days_per_second": 28 / elapsed,
        "timing_includes_merge_and_final_completion": True}


def alternating_orders(pairs=3):
    if type(pairs) is not int or pairs < 3:
        raise ValueError("at least three paired measurements required")
    return [MODES if index % 2 == 0 else MODES[::-1] for index in range(pairs)]


def validate_scope(scope, *, output, wall_seconds):
    """Require a new independent grant; no old B4 or paid balance is accepted."""
    expected = {"schema_version": "latent_T7_dispatch_scope_v1", "approved": True,
        "funding_scope": "latent_T7_dispatch_20261009", "gpu_count": 1,
        "limit_gpu_seconds": 900, "session_limit_seconds": 900,
        "model_compile_cap": 0, "optimizer_update_cap": 0, "pairs": 3}
    for key, value in expected.items():
        if type(scope.get(key)) is not type(value) or scope.get(key) != value:
            raise ValueError("independent dispatch authorization differs: " + key)
    if scope.get("output") != str(output) or not 1 <= wall_seconds <= 840:
        raise ValueError("scope output binding or protected exit window differs")
    return scope
