"""GPU-only timing probe for the accepted first process-head training graph."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from scripts.hpc.orcjax_gpu_runtime import initialize_gpu_backend


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path, required=True)
    parser.add_argument("--base-statistics", type=Path, required=True)
    parser.add_argument("--process-statistics", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    jax, devices = initialize_gpu_backend()
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from research.daily_coarse_graining.daily_operator.neural_architecture import (
        parameter_count,
    )
    from research.daily_coarse_graining.first_neural_process_training_run import (
        build_process_batch_loss,
        initialize_adamw,
        initialize_process_parameters,
        iter_process_training_batches,
        learning_rate_at_update,
        make_adamw_train_step,
        prepare_process_training,
    )

    prepared = prepare_process_training(
        dataset_manifest=args.dataset_manifest,
        acceptance_report=args.acceptance_report,
        base_statistics=args.base_statistics,
        process_statistics=args.process_statistics,
        protocol_path=args.protocol,
    )
    preflight = prepared.protocol.preflight
    batch_size = int(preflight["batch_size"])
    seed = int(preflight["seed"])
    batch_inputs, batch_targets = next(
        iter_process_training_batches(
            prepared,
            seed=seed,
            batch_size=batch_size,
            maximum_batches=1,
        )
    )
    initial_parameters = initialize_process_parameters(
        prepared,
        batch_inputs,
        seed=seed,
    )
    count = parameter_count(initial_parameters)
    arms = {}
    for arm in preflight["arms"]:
        parameters = initial_parameters
        optimizer = initialize_adamw(parameters)
        loss_function = build_process_batch_loss(prepared, arm=arm)
        step = make_adamw_train_step(loss_function, prepared.protocol)
        warmup_started = time.perf_counter()
        warmup_losses = []
        for update in range(1, int(preflight["warmup_updates"]) + 1):
            learning_rate = jnp.asarray(
                learning_rate_at_update(prepared.protocol, update),
                dtype=jnp.float64,
            )
            parameters, optimizer, loss, gradient_norm, clip_scale = step(
                parameters,
                optimizer,
                batch_inputs,
                batch_targets,
                learning_rate,
            )
            jax.block_until_ready((parameters, loss, gradient_norm, clip_scale))
            warmup_losses.append(float(loss))
        warmup_seconds = time.perf_counter() - warmup_started
        measured_started = time.perf_counter()
        measured_losses = []
        gradient_norms = []
        clip_scales = []
        first = int(preflight["warmup_updates"]) + 1
        stop = first + int(preflight["measured_updates"])
        for update in range(first, stop):
            learning_rate = jnp.asarray(
                learning_rate_at_update(prepared.protocol, update),
                dtype=jnp.float64,
            )
            parameters, optimizer, loss, gradient_norm, clip_scale = step(
                parameters,
                optimizer,
                batch_inputs,
                batch_targets,
                learning_rate,
            )
            jax.block_until_ready((parameters, loss, gradient_norm, clip_scale))
            measured_losses.append(float(loss))
            gradient_norms.append(float(gradient_norm))
            clip_scales.append(float(clip_scale))
        measured_seconds = time.perf_counter() - measured_started
        finite = all(np.all(np.isfinite(values)) for values in (measured_losses, gradient_norms, clip_scales))
        arms[arm] = {
            "status": "passed" if finite else "failed",
            "warmup_seconds_including_compile": warmup_seconds,
            "warmup_losses": warmup_losses,
            "measured_updates": len(measured_losses),
            "measured_seconds": measured_seconds,
            "seconds_per_update": measured_seconds / len(measured_losses),
            "updates_per_second": len(measured_losses) / measured_seconds,
            "first_measured_loss": measured_losses[0],
            "last_measured_loss": measured_losses[-1],
            "maximum_gradient_norm": max(gradient_norms),
            "minimum_clip_scale": min(clip_scales),
            "finite": finite,
        }
    report = {
        "schema_version": "first_neural_process_training_probe_v1",
        "status": "passed" if all(value["status"] == "passed" for value in arms.values()) else "failed",
        "checkpoint_written": False,
        "devices": [str(device) for device in devices],
        "parameter_count": count,
        "candidate_contract_sha256": prepared.candidate.sha256,
        "training_protocol_sha256": prepared.protocol.sha256,
        "artifact_hashes": dict(prepared.artifact_hashes),
        "batch": {
            "spatial_split": "train",
            "temporal_split": "train",
            "seed": seed,
            "batch_size": batch_size,
            "reused_for_both_arms": True,
        },
        "arms": arms,
    }
    _atomic_json(args.output.resolve(), report)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
