"""Free-GPU serialization probe for the successor checkpoint handoff."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import (
    load_successor_checkpoint_handoff,
    load_successor_checkpoint_set,
    write_successor_checkpoint_set,
)


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"checkpoint handoff probe output already exists: {args.output}")
    started = time.perf_counter()

    import jax
    import jax.numpy as jnp

    handoff = load_successor_checkpoint_handoff(args.checkpoint_handoff)
    arms = tuple(handoff.checkpoint["arms"])
    device_parameters = {
        arm: {
            "weight": jnp.arange(12, dtype=jnp.float64).reshape(3, 4) + index,
            "bias": jnp.asarray([index, index + 0.5], dtype=jnp.float64),
        }
        for index, arm in enumerate(arms)
    }
    device_optimizers = {
        arm: {
            "step": jnp.asarray(500, dtype=jnp.int32),
            "first": jax.tree_util.tree_map(jnp.zeros_like, device_parameters[arm]),
            "second": jax.tree_util.tree_map(jnp.ones_like, device_parameters[arm]),
        }
        for arm in arms
    }
    jax.block_until_ready((device_parameters, device_optimizers))
    platforms = sorted(
        {
            leaf.devices().pop().platform
            for tree in (device_parameters, device_optimizers)
            for arm in arms
            for leaf in jax.tree_util.tree_leaves(tree[arm])
        }
    )
    if jax.default_backend() != "gpu" or platforms != ["gpu"]:
        raise ValueError(f"checkpoint handoff GPU probe ran on {jax.default_backend()}: {platforms}")

    parameters = jax.device_get(device_parameters)
    optimizers = jax.device_get(device_optimizers)
    parameter_counts = {
        arm: sum(int(np.asarray(leaf).size) for leaf in jax.tree_util.tree_leaves(parameters[arm]))
        for arm in arms
    }
    identity = {
        "run_kind": "free_gpu_serialization_probe",
        "consumer_git_head": _git_head(),
        "checkpoint_handoff_contract_sha256": handoff.sha256,
        "production_checkpoint": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="successor-checkpoint-probe-", dir=args.output.parent) as temporary:
        checkpoint_dir = Path(temporary) / "checkpoints"
        written = write_successor_checkpoint_set(
            checkpoint_dir,
            identity=identity,
            parameters=parameters,
            optimizers=optimizers,
            completed_updates={arm: 500 for arm in arms},
            parameter_counts=parameter_counts,
            final_metrics={arm: {"serialization_probe_only": True} for arm in arms},
            loss_reductions={arm: {"serialization_probe_only": True} for arm in arms},
            expected_arms=arms,
            required_updates=500,
            promotion_passed=True,
        )
        loaded = load_successor_checkpoint_set(
            checkpoint_dir,
            expected_manifest_sha256=str(written["manifest_sha256"]),
            identity=identity,
            expected_arms=arms,
            required_updates=500,
        )
        exact = all(
            all(
                np.array_equal(actual, expected)
                for actual, expected in zip(
                    jax.tree_util.tree_leaves(loaded[arm]["parameters"]),
                    jax.tree_util.tree_leaves(parameters[arm]),
                    strict=True,
                )
            )
            and all(
                np.array_equal(actual, expected)
                for actual, expected in zip(
                    jax.tree_util.tree_leaves(loaded[arm]["optimizer"]),
                    jax.tree_util.tree_leaves(optimizers[arm]),
                    strict=True,
                )
            )
            for arm in arms
        )
        if not exact:
            raise ValueError("successor checkpoint GPU serialization round-trip drift")
        checkpoint_manifest_sha256 = str(written["manifest_sha256"])
        checkpoint_sizes = {
            arm: int(written["checkpoints"][arm]["size_bytes"])
            for arm in arms
        }

    report = {
        "schema_version": "successor_checkpoint_handoff_gpu_probe_v1",
        "status": "passed",
        "run_kind": "free_gpu_serialization_probe",
        "identity": identity,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "parameter_platforms_before_device_get": platforms,
        "arms": list(arms),
        "parameter_counts": parameter_counts,
        "checkpoint_sizes": checkpoint_sizes,
        "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
        "round_trip_exact": exact,
        "temporary_checkpoint_removed": True,
        "training_updates": 0,
        "teacher_data_read": False,
        "teacher_data_regenerated": False,
        "historical_v5_used": False,
        "paid_gpu_allocation_used": False,
        "wall_seconds": time.perf_counter() - started,
    }
    _atomic_json(args.output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-handoff", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
