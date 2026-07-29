"""Real-shard gate for the frozen rollout-stability training implementation."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.daily_model_architecture import (
    verify_checkpoint_architecture,
)
from research.daily_coarse_graining.rollout_stability_run import (
    ARM_IDS,
    HARD_CONSTRAINT_FIELDS,
    _atomic_json,
    _bind_dynamic_transition,
    _load_pickle,
    _load_rollout_resources,
    _objective_kwargs,
    _trees_equal,
    build_arm_checkpoint,
    make_candidate_update_step,
    make_coefficient_calibration_step,
    prepare_rollout_update,
    verify_arm_checkpoint,
)

SCHEMA_VERSION = "canonical_rollout_stability_real_shard_smoke_v1"


def _tree_exact(left: Any, right: Any) -> bool:
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    return left_tree == right_tree and all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(left_leaves, right_leaves, strict=True)
    )


def _hard_counts(components) -> Mapping[str, int]:
    host = jax.device_get(components)
    return {name: int(getattr(host, name)) for name in HARD_CONSTRAINT_FIELDS}


def _compiled_args(parameters, optimizer, prepared, learning_rate):
    return (
        parameters,
        optimizer,
        prepared.anchor_batch,
        prepared.initial_states,
        prepared.initial_discrete_states,
        prepared.sequence,
        prepared.teacher_next_discrete_states,
        learning_rate,
    )


def _smoke_schedule(horizon: int) -> Mapping[str, Any]:
    horizon_key = str(horizon)
    payload = {
        "canonical_sha256": "real-shard-smoke-local-schedule",
        "horizon_counts": {horizon_key: 2},
        "entries": [
            {"update": 0, "reference_index": 0, "horizon": horizon},
            {"update": 1, "reference_index": 1, "horizon": horizon},
        ],
    }
    return payload


def run_real_shard_smoke(
    *,
    dataset_path: str | Path,
    statistics_path: str | Path,
    parent_best_checkpoint_path: str | Path,
    parent_optimizer_checkpoint_path: str | Path,
    output_path: str | Path,
    plan_path: str | Path | None = None,
    horizon: int = 1,
    anchor_batch_size: int = 2,
    rollout_batch_size: int = 1,
    seed: int = 20260728,
) -> Path:
    """Compile once and update two train landpoints through the dynamic tail."""

    if horizon not in {1, 3, 7, 30}:
        raise ValueError("rollout-stability smoke horizon must be 1, 3, 7, or 30")
    resources = _load_rollout_resources(
        dataset_path=Path(dataset_path).resolve(),
        statistics_path=Path(statistics_path).resolve(),
        plan_path=None if plan_path is None else Path(plan_path),
    )
    references = sorted(
        resources.index.select(spatial_split="train", temporal_split="train"),
        key=lambda item: (item.landpoint_id, item.year, str(item.path)),
    )
    selected = []
    seen_landpoints = set()
    for reference in references:
        if reference.landpoint_id not in seen_landpoints:
            selected.append(reference)
            seen_landpoints.add(reference.landpoint_id)
        if len(selected) == 2:
            break
    if len(selected) != 2:
        raise ValueError("real-shard smoke requires two train landpoints")

    best = _load_pickle(Path(parent_best_checkpoint_path).resolve())
    optimizer_checkpoint = _load_pickle(
        Path(parent_optimizer_checkpoint_path).resolve()
    )
    verify_checkpoint_architecture(best["identity"], resources.model_definition)
    verify_checkpoint_architecture(
        optimizer_checkpoint["identity"],
        resources.model_definition,
    )
    if not _trees_equal(best["parameters"], optimizer_checkpoint["parameters"]):
        raise ValueError("real-shard smoke parent parameter checkpoints differ")
    parameters = jax.tree_util.tree_map(jnp.asarray, best["parameters"])
    optimizer = jax.tree_util.tree_map(
        jnp.asarray,
        optimizer_checkpoint["optimizer"],
    )
    prepared = [
        prepare_rollout_update(
            reference=reference,
            update=index,
            horizon=horizon,
            anchor_batch_size=anchor_batch_size,
            rollout_batch_size=rollout_batch_size,
            seed=seed,
            resources=resources,
        )
        for index, reference in enumerate(selected)
    ]
    signatures = [item.trace_signature for item in prepared]
    if signatures[0] != signatures[1]:
        raise ValueError(
            "selected smoke landpoints require different static dispatch signatures"
        )

    transition = _bind_dynamic_transition(resources, prepared[0].runtime)
    coefficients = {
        "L_next": 1.0,
        "L_rollout": 1.0,
        "L_bias": 0.25,
        "L_science": 0.5,
    }
    rematerialize = horizon == 30
    step = make_candidate_update_step(
        coefficients=coefficients,
        fast_day_weights=resources.fast_day_weights,
        undefined_loss_weight=0.1,
        rematerialize=rematerialize,
        model_apply=resources.model_definition.apply,
        **_objective_kwargs(resources, transition),
    )
    rate = jnp.asarray(3.0e-5, dtype=jnp.float32)
    first_args = _compiled_args(
        parameters,
        optimizer,
        prepared[0],
        rate,
    )
    compile_started = time.perf_counter()
    executable = step.lower(*first_args).compile()
    compile_seconds = time.perf_counter() - compile_started

    first_started = time.perf_counter()
    first = executable(*first_args)
    jax.block_until_ready(first.parameters)
    first_seconds = time.perf_counter() - first_started
    if not bool(jax.device_get(first.update_applied)):
        raise ValueError(f"first smoke update failed closed: {_hard_counts(first.components)}")

    second_args = _compiled_args(
        first.parameters,
        first.optimizer,
        prepared[1],
        rate,
    )
    second_started = time.perf_counter()
    uninterrupted = executable(*second_args)
    jax.block_until_ready(uninterrupted.parameters)
    second_seconds = time.perf_counter() - second_started
    if not bool(jax.device_get(uninterrupted.update_applied)):
        raise ValueError(
            f"second smoke update failed closed: {_hard_counts(uninterrupted.components)}"
        )

    schedule = _smoke_schedule(horizon)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": resources.index.dataset_id,
        "model_architecture": resources.model_definition.identity(),
        "horizon": horizon,
    }
    checkpoint = build_arm_checkpoint(
        arm_id=ARM_IDS[1],
        identity=identity,
        parameters=first.parameters,
        optimizer=first.optimizer,
        next_update=1,
        schedule=schedule,
        history=[],
    )
    reloaded = pickle.loads(
        pickle.dumps(checkpoint, protocol=pickle.HIGHEST_PROTOCOL)
    )
    verify_arm_checkpoint(
        reloaded,
        arm_id=ARM_IDS[1],
        identity=identity,
        schedule=schedule,
    )
    resumed_args = _compiled_args(
        jax.tree_util.tree_map(jnp.asarray, reloaded["parameters"]),
        jax.tree_util.tree_map(jnp.asarray, reloaded["optimizer"]),
        prepared[1],
        rate,
    )
    resumed = executable(*resumed_args)
    jax.block_until_ready(resumed.parameters)
    restart_exact = _tree_exact(
        (uninterrupted.parameters, uninterrupted.optimizer),
        (resumed.parameters, resumed.optimizer),
    )
    if not restart_exact:
        raise ValueError("real-shard smoke checkpoint resume is not exact")

    calibration = make_coefficient_calibration_step(
        fast_day_weights=resources.fast_day_weights,
        undefined_loss_weight=0.1,
        rematerialize=rematerialize,
        model_apply=resources.model_definition.apply,
        **_objective_kwargs(resources, transition),
    )
    calibration_values, calibration_norms, calibration_components = calibration(
        parameters,
        prepared[0].anchor_batch,
        prepared[0].initial_states,
        prepared[0].initial_discrete_states,
        prepared[0].sequence,
        prepared[0].teacher_next_discrete_states,
    )
    calibration_values, calibration_norms = jax.device_get(
        (calibration_values, calibration_norms)
    )
    if (
        not np.all(np.isfinite(calibration_values))
        or not np.all(np.isfinite(calibration_norms))
        or any(_hard_counts(calibration_components).values())
    ):
        raise ValueError("real-shard smoke coefficient calibration is invalid")

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "dataset_id": resources.index.dataset_id,
        "markov_contract_sha256": resources.index.contract_sha256,
        "model_architecture": resources.model_definition.identity(),
        "parent_best_checkpoint": str(
            Path(parent_best_checkpoint_path).resolve()
        ),
        "parent_optimizer_checkpoint": str(
            Path(parent_optimizer_checkpoint_path).resolve()
        ),
        "landpoints": [reference.landpoint_id for reference in selected],
        "years": [int(reference.year) for reference in selected],
        "horizon": horizon,
        "anchor_batch_size": anchor_batch_size,
        "rollout_batch_size": rollout_batch_size,
        "trace_static_signature": signatures[0],
        "single_executable_reused_across_landpoints": True,
        "compile_seconds": compile_seconds,
        "first_update_seconds": first_seconds,
        "second_update_seconds": second_seconds,
        "first_loss": float(jax.device_get(first.loss)),
        "second_loss": float(jax.device_get(uninterrupted.loss)),
        "first_gradient_norm": float(jax.device_get(first.gradient_norm)),
        "second_gradient_norm": float(jax.device_get(uninterrupted.gradient_norm)),
        "first_hard_counts": _hard_counts(first.components),
        "second_hard_counts": _hard_counts(uninterrupted.components),
        "checkpoint_resume_exact": restart_exact,
        "calibration_component_values": [
            float(value) for value in calibration_values
        ],
        "calibration_gradient_norms": [
            float(value) for value in calibration_norms
        ],
        "state_delta_scale": resources.state_delta_scale_audit,
        "science_layout_sha256": resources.science_layout.sha256,
        "sealed_test_used": False,
    }
    return _atomic_json(Path(output_path).resolve(), report)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--parent-best-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-optimizer-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--anchor-batch-size", type=int, default=2)
    parser.add_argument("--rollout-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260728)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = run_real_shard_smoke(
        dataset_path=args.dataset,
        statistics_path=args.statistics,
        parent_best_checkpoint_path=args.parent_best_checkpoint,
        parent_optimizer_checkpoint_path=args.parent_optimizer_checkpoint,
        output_path=args.output,
        plan_path=args.plan,
        horizon=args.horizon,
        anchor_batch_size=args.anchor_batch_size,
        rollout_batch_size=args.rollout_batch_size,
        seed=args.seed,
    )
    print(json.dumps({"status": "passed", "report": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
