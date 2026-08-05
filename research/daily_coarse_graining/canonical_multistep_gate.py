"""Numerical and gradient gate for the differentiable retained daily tail."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from jax_orchidee.driver import orchestration as teacher
from research.daily_coarse_graining.canonical_retained_tail import (
    bind_canonical_retained_tail_transition,
)
from research.daily_coarse_graining.daily_markov_contract import (
    reconstruct_compiled_forcing_window,
    reconstruct_state_packet,
)
from research.daily_coarse_graining.markov_dataset import defined_numeric_mask
from research.daily_coarse_graining.replay_ceiling import (
    DEFAULT_CONFIG,
    DEFAULT_RUN_DEF,
    _block_until_ready,
    _capture_sequence,
    _load_state_cache,
)
from research.daily_coarse_graining.synthetic_operator_cost import (
    _tail_static_inputs,
)
from research.daily_coarse_graining.teacher_shards import build_shard_arrays

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "reference_mode"
    / "daily_coarse_graining"
    / "canonical_multistep_gradient_gate.json"
)


def _max_error(actual, expected, mask) -> tuple[float, float]:
    difference = np.abs(actual[mask] - expected[mask])
    if not difference.size:
        return 0.0, 0.0
    denominator = np.maximum(np.abs(expected[mask]), np.finfo(np.float64).tiny)
    return float(np.max(difference)), float(np.max(difference / denominator))


def _largest_leaf_errors(actual, expected, contract, *, limit: int = 12):
    rows = []
    for leaf in contract.state_leaves:
        if leaf.discrete:
            continue
        selected = slice(leaf.start, leaf.stop)
        actual_leaf = actual[..., selected]
        expected_leaf = expected[..., selected]
        mask = defined_numeric_mask(actual_leaf) & defined_numeric_mask(expected_leaf)
        absolute, relative = _max_error(actual_leaf, expected_leaf, mask)
        rows.append(
            {
                "key": leaf.key,
                "max_absolute_error": absolute,
                "max_relative_error": relative,
                "defined_status_mismatches": int(
                    np.count_nonzero(
                        defined_numeric_mask(actual_leaf)
                        != defined_numeric_mask(expected_leaf)
                    )
                ),
            }
        )
    return sorted(
        rows,
        key=lambda item: item["max_absolute_error"],
        reverse=True,
    )[:limit]


def _nonfinite_gradient_leaves(gradient, target, contract):
    rows = []
    for leaf in contract.fast_day_target_leaves:
        selected = slice(leaf.start, leaf.stop)
        numeric = defined_numeric_mask(target[..., selected])
        nonfinite = numeric & ~np.isfinite(gradient[..., selected])
        if np.any(nonfinite):
            rows.append(
                {
                    "key": leaf.key,
                    "defined_values": int(np.count_nonzero(numeric)),
                    "nonfinite_defined_gradients": int(
                        np.count_nonzero(nonfinite)
                    ),
                    "compact_indices": (
                        np.flatnonzero(nonfinite).astype(int).tolist()
                    ),
                }
            )
    return rows


def run_gate(args: argparse.Namespace) -> dict[str, Any]:
    cache = _load_state_cache(args.state_cache.resolve())
    initial_year_end_state = cache["state"]
    context = teacher.prepare_paper_1961_driver_context(
        args.config.resolve(),
        used_run_def_path=args.run_def.resolve(),
    )

    capture_started = time.perf_counter()
    records = _capture_sequence(
        config_path=args.config.resolve(),
        initial_year_end_state=initial_year_end_state,
        year=args.year,
        days=args.horizon + 1,
        context=context,
    )
    capture_seconds = time.perf_counter() - capture_started
    arrays, contract = build_shard_arrays(
        tuple(record.expected_result.day_end_state for record in records[:-1]),
        (None,) * args.horizon,
        records[1:],
        context,
        final_state=records[-1].expected_result.day_end_state,
    )
    state = arrays["state_trajectory"][0]
    expected_state = arrays["state_trajectory"][1:]
    discrete = {
        name.removeprefix("state_discrete__"): value[0]
        for name, value in arrays.items()
        if name.startswith("state_discrete__")
    }
    expected_discrete = {
        name.removeprefix("state_discrete__"): value[1:]
        for name, value in arrays.items()
        if name.startswith("state_discrete__")
    }
    fast_day_target = arrays["fast_day_target"]
    compiled_forcing = reconstruct_compiled_forcing_window(
        arrays["forcing_native"],
        contract.native_forcing,
        context,
        years=np.full((args.horizon,), args.year, dtype=np.int32),
        day_indices=arrays["day_index"],
    )
    canonical_packet = reconstruct_state_packet(
        state,
        discrete,
        contract,
        tstep=47,
    )
    forcing_batch = jax.tree_util.tree_map(
        np.asarray,
        compiled_forcing,
    )
    static = _tail_static_inputs(
        context,
        canonical_packet,
        forcing_batch,
        first_day=False,
    )
    transition = bind_canonical_retained_tail_transition(
        config_path=args.config.resolve(),
        context=context,
        contract=contract,
        static=static,
        runtime_year=args.year,
    )

    def rollout(initial_state, initial_discrete, targets, forcing_days, day_indices):
        def output_body(carry, inputs):
            target, forcing, science_day = inputs
            next_carry = transition(
                carry[0],
                carry[1],
                target,
                forcing,
                np.asarray(args.year, dtype=np.int32),
                science_day,
            )
            return next_carry, next_carry

        return jax.lax.scan(
            output_body,
            (initial_state, initial_discrete),
            (targets, forcing_days, day_indices),
        )

    forward = jax.jit(rollout)
    forward_started = time.perf_counter()
    (_, _), (actual_state, actual_discrete) = forward(
        state,
        discrete,
        fast_day_target,
        compiled_forcing,
        arrays["day_index"],
    )
    _block_until_ready((actual_state, actual_discrete))
    forward_compile_seconds = time.perf_counter() - forward_started
    actual_state = np.asarray(actual_state)
    expected_mask = defined_numeric_mask(expected_state)
    actual_mask = defined_numeric_mask(actual_state)
    max_absolute_error, max_relative_error = _max_error(
        actual_state,
        expected_state,
        expected_mask & actual_mask,
    )
    discrete_mismatches = sum(
        int(
            np.count_nonzero(
                np.asarray(actual_discrete[name]) != expected_discrete[name]
            )
        )
        for name in expected_discrete
    )

    probe = np.random.default_rng(0).normal(size=expected_state.shape)
    probe = np.where(expected_mask, probe, 0.0)

    def objective(target):
        (_, _), (next_state, _) = rollout(
            state,
            discrete,
            target,
            compiled_forcing,
            arrays["day_index"],
        )
        return jnp.vdot(jnp.where(expected_mask, next_state, 0.0), probe)

    gradient_started = time.perf_counter()
    objective_value, gradient = jax.jit(jax.value_and_grad(objective))(
        fast_day_target
    )
    _block_until_ready((objective_value, gradient))
    gradient_compile_seconds = time.perf_counter() - gradient_started
    gradient = np.asarray(gradient)
    learned_numeric = defined_numeric_mask(fast_day_target)
    finite_gradient = np.isfinite(gradient)
    gradient_norm = float(
        np.linalg.norm(np.where(finite_gradient & learned_numeric, gradient, 0.0))
    )

    forward_passed = (
        np.array_equal(actual_mask, expected_mask)
        and discrete_mismatches == 0
        and max_absolute_error <= args.atol + args.rtol * float(
            np.max(np.abs(expected_state[expected_mask]))
        )
    )
    gradient_passed = bool(
        np.all(finite_gradient[learned_numeric]) and gradient_norm > 0.0
    )
    return {
        "schema_version": "canonical_multistep_gradient_gate_v1",
        "status": "passed" if forward_passed and gradient_passed else "failed",
        "teacher_label": "provisional_teacher",
        "state_cache": str(args.state_cache.resolve()),
        "year": int(args.year),
        "teacher_day_start": 2,
        "horizon": int(args.horizon),
        "contract_sha256": contract.sha256,
        "capture_seconds": capture_seconds,
        "forward": {
            "passed": forward_passed,
            "compile_and_first_run_seconds": forward_compile_seconds,
            "max_absolute_error": max_absolute_error,
            "max_relative_error": max_relative_error,
            "defined_status_mismatches": int(
                np.count_nonzero(actual_mask != expected_mask)
            ),
            "discrete_mismatches": discrete_mismatches,
            "largest_leaf_errors": _largest_leaf_errors(
                actual_state,
                expected_state,
                contract,
            ),
        },
        "gradient": {
            "passed": gradient_passed,
            "compile_and_first_run_seconds": gradient_compile_seconds,
            "objective": float(objective_value),
            "width": int(gradient.size),
            "defined_target_values": int(np.count_nonzero(learned_numeric)),
            "finite_defined_values": int(
                np.count_nonzero(finite_gradient & learned_numeric)
            ),
            "undefined_target_values": int(np.count_nonzero(~learned_numeric)),
            "nonzero_defined_values": int(
                np.count_nonzero(gradient[learned_numeric])
            ),
            "l2_norm": gradient_norm,
            "max_absolute_value": float(
                np.max(np.abs(gradient[finite_gradient & learned_numeric]))
            ),
            "nonfinite_leaves": _nonfinite_gradient_leaves(
                gradient,
                fast_day_target,
                contract,
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-def", type=Path, default=DEFAULT_RUN_DEF)
    parser.add_argument("--year", type=int, default=1962)
    parser.add_argument("--horizon", type=int, choices=(1, 3, 7), default=1)
    parser.add_argument("--atol", type=float, default=1.0e-10)
    parser.add_argument("--rtol", type=float, default=1.0e-12)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = run_gate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
