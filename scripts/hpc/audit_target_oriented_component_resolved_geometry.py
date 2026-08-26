"""Audit component-resolved flux geometry at the frozen update-1000 state."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file
from scripts.hpc.audit_target_oriented_flux_gradient_variance import (
    AUDIT_UPDATES,
    CHECKPOINT_UPDATE,
    _add_gradient,
    _audit_family_trees,
    _cosine_matrix,
    _family_gram,
    _hard_host,
    _load_context,
    _mean_gradient,
    _tree_size,
    _zero_gradient,
)
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_attention_gpu_admission import PARAMETER_COUNT
from scripts.hpc.run_target_oriented_full_step_preflight import (
    _add_hard,
    _hard_failure_count,
    _slice,
    _zero_hard,
)
from scripts.hpc.target_oriented_training_prepared import load_prepared_target_artifact

SCHEMA_VERSION = "target_oriented_component_resolved_geometry_audit_v1"
SOURCE_NAMES = (
    "one_day_state_and_process",
    "seven_day_component_resolved_flux",
    "seven_day_independent_stock",
    "thirty_day_component_resolved_flux",
    "thirty_day_independent_stock",
)
PROTECTION_SOURCE_NAMES = (
    "one_day_state_and_process",
    "seven_day_independent_stock",
    "thirty_day_independent_stock",
)


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _merge(add: Any, values: Sequence[Any]) -> Any:
    if not values:
        raise ValueError("cannot merge empty component-resolved statistics")
    result = values[0]
    for value in values[1:]:
        result = add(result, value)
    return result


def _gradient_report(jax: Any, gradient: Any) -> Mapping[str, Any]:
    leaves = [np.asarray(value) for value in jax.device_get(jax.tree_util.tree_leaves(gradient))]
    count = sum(int(value.size) for value in leaves)
    finite = sum(int(np.count_nonzero(np.isfinite(value))) for value in leaves)
    norm = float(
        np.sqrt(
            sum(
                (float(np.sum(np.square(value.astype(np.float64)))) for value in leaves),
                0.0,
            )
        )
    )
    return {
        "value_count": count,
        "finite_value_count": finite,
        "nonfinite_value_count": count - finite,
        "l2_norm": norm,
    }


def _all_parameter_gram(jax: Any, jnp: Any, gradients: Sequence[Any]) -> np.ndarray:
    from jax.flatten_util import ravel_pytree

    vectors = [ravel_pytree(gradient)[0].astype(jnp.float64) for gradient in gradients]
    matrix = jnp.stack(vectors, axis=0)
    gram = jnp.matmul(matrix, matrix.T, precision=jax.lax.Precision.HIGHEST)
    jax.block_until_ready(gram)
    result = np.asarray(jax.device_get(gram), dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("component-resolved all-parameter Gram matrix is nonfinite")
    return result


def _cosine_report(gram: np.ndarray, names: Sequence[str]) -> Mapping[str, Any]:
    values = np.asarray(gram, dtype=np.float64)
    names = tuple(names)
    if values.shape != (len(names), len(names)):
        raise ValueError("component-resolved cosine registry drift")
    cosine = _cosine_matrix(values)
    return {
        "names": list(names),
        "gradient_norms": [float(value) for value in np.sqrt(np.maximum(np.diag(values), 0.0))],
        "cosine_matrix": [
            [float(value) if np.isfinite(value) else None for value in row]
            for row in cosine
        ],
    }


def _group_variance_report(gram: np.ndarray) -> Mapping[str, Any]:
    values = np.asarray(gram, dtype=np.float64)
    if values.shape != (len(AUDIT_UPDATES), len(AUDIT_UPDATES)):
        raise ValueError("component-resolved group Gram coverage drift")
    cosine = _cosine_matrix(values)
    upper = cosine[np.triu_indices(len(AUDIT_UPDATES), 1)]
    finite = upper[np.isfinite(upper)]
    norms = np.sqrt(np.maximum(np.diag(values), 0.0))
    return {
        "gradient_norm_minimum": float(np.min(norms)),
        "gradient_norm_median": float(np.median(norms)),
        "gradient_norm_maximum": float(np.max(norms)),
        "pairwise_cosine_minimum": float(np.min(finite)) if finite.size else None,
        "pairwise_cosine_median": float(np.median(finite)) if finite.size else None,
        "pairwise_cosine_negative_fraction": (
            float(np.mean(finite < 0.0)) if finite.size else None
        ),
    }


def _accumulate_source_gradient(
    jax: Any,
    jnp: Any,
    parameters: Any,
    step: Any,
    microbatches: Sequence[Any],
    total: Any,
) -> tuple[Any, float]:
    gradient = _zero_gradient(jax, jnp, parameters)
    loss = jnp.asarray(0.0, dtype=jnp.float64)
    for microbatch in microbatches:
        value, current = step(parameters, microbatch, total)
        loss = loss + value
        gradient = _add_gradient(jax, jnp, gradient, current)
    jax.block_until_ready((loss, gradient))
    return gradient, float(np.asarray(loss))


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)

    from research.daily_coarse_graining.daily_operator.target_oriented_flux_basis import (
        load_target_oriented_flux_basis_amendment,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_training import (
        add_error_sufficient_statistics,
        add_science_window_sufficient_statistics,
    )

    started = time.perf_counter()
    amendment = load_target_oriented_flux_basis_amendment(args.flux_basis_amendment)
    context = _load_context(args, science_flux_basis=amendment.flux_basis)
    runtime = context["runtime"]
    jax = runtime["jax"]
    jnp = runtime["jnp"]
    parameters = context["parameters"]
    steps = context["steps"]
    schedule = context["schedule"]
    prepared_records = context["prepared_records"]
    training_contract = context["training_contract"]

    family_counts = {
        name: _tree_size(jax, tree)
        for name, tree in _audit_family_trees(parameters).items()
    }
    if sum(family_counts.values()) != PARAMETER_COUNT:
        raise ValueError("component-resolved parameter coverage drift")

    memory_before = _memory_report(runtime["devices"][0])
    hard = _zero_hard(jnp)
    source_gradients: dict[str, list[Any]] = {name: [] for name in SOURCE_NAMES}
    group_records = []
    forward_seconds = 0.0
    gradient_seconds = 0.0

    for update in AUDIT_UPDATES:
        artifact = load_prepared_target_artifact(
            args.prepared_manifest,
            prepared_records[update],
            training_contract_sha256=training_contract.sha256,
            schedule_entry=schedule["entries"][update - 1],
        )
        if artifact.trace_signature != context["trace_signature"]:
            raise ValueError("component-resolved prepared trace drift")
        one_batches = [_slice(artifact.batch.one_day, index, index + 4) for index in range(0, 32, 4)]
        seven_batches = [_slice(artifact.batch.seven_day, index, index + 1) for index in range(8)]
        thirty_batches = [_slice(artifact.batch.thirty_day, index, index + 1) for index in range(4)]

        forward_started = time.perf_counter()
        one_values = [steps["one_forward"](parameters, batch) for batch in one_batches]
        seven_values = [steps["seven_forward"](parameters, batch) for batch in seven_batches]
        thirty_values = [steps["thirty_forward"](parameters, batch) for batch in thirty_batches]
        jax.block_until_ready((one_values, seven_values, thirty_values))
        total_state = _merge(
            lambda left, right: {
                name: add_error_sufficient_statistics(left[name], right[name])
                for name in left
            },
            [value[0] for value in one_values],
        )
        total_process = _merge(
            lambda left, right: {
                name: add_error_sufficient_statistics(left[name], right[name])
                for name in left
            },
            [value[1] for value in one_values],
        )
        total_seven = _merge(
            add_science_window_sufficient_statistics,
            [value[0] for value in seven_values],
        )
        total_thirty = _merge(
            add_science_window_sufficient_statistics,
            [value[0] for value in thirty_values],
        )
        for values in (one_values, seven_values, thirty_values):
            for value in values:
                hard = _add_hard(jax, hard, value[-1])
        forward_seconds += time.perf_counter() - forward_started

        gradient_started = time.perf_counter()
        group_sources = {}
        group_losses = {}

        one_gradient = _zero_gradient(jax, jnp, parameters)
        one_loss = jnp.asarray(0.0, dtype=jnp.float64)
        for microbatch in one_batches:
            value, current = steps["one_gradient"](
                parameters,
                microbatch,
                total_state,
                total_process,
            )
            one_loss = one_loss + value
            one_gradient = _add_gradient(jax, jnp, one_gradient, current)
        jax.block_until_ready((one_loss, one_gradient))
        group_sources[SOURCE_NAMES[0]] = one_gradient
        group_losses[SOURCE_NAMES[0]] = float(np.asarray(one_loss))

        for name, step_name, batches, total in (
            (SOURCE_NAMES[1], "seven_flux_gradient", seven_batches, total_seven),
            (SOURCE_NAMES[2], "seven_stock_gradient", seven_batches, total_seven),
            (SOURCE_NAMES[3], "thirty_flux_gradient", thirty_batches, total_thirty),
            (SOURCE_NAMES[4], "thirty_stock_gradient", thirty_batches, total_thirty),
        ):
            gradient, loss = _accumulate_source_gradient(
                jax,
                jnp,
                parameters,
                steps[step_name],
                batches,
                total,
            )
            group_sources[name] = gradient
            group_losses[name] = loss

        gradient_seconds += time.perf_counter() - gradient_started
        source_gram = _all_parameter_gram(
            jax,
            jnp,
            [group_sources[name] for name in SOURCE_NAMES],
        )
        combined = _zero_gradient(jax, jnp, parameters)
        for name in SOURCE_NAMES:
            combined = _add_gradient(jax, jnp, combined, group_sources[name])
        group_records.append(
            {
                "update": update,
                "runtime_year": int(schedule["entries"][update - 1]["runtime_year"]),
                "source_losses": group_losses,
                "source_gradients": {
                    name: _gradient_report(jax, group_sources[name])
                    for name in SOURCE_NAMES
                },
                "source_geometry": _cosine_report(source_gram, SOURCE_NAMES),
                "combined_gradient": _gradient_report(jax, combined),
            }
        )
        for name in SOURCE_NAMES:
            source_gradients[name].append(jax.device_get(group_sources[name]))
        del combined, group_sources
        print(f"component_resolved_geometry update={update} complete", flush=True)

    jax.block_until_ready(hard)
    hard_host = _hard_host(jax.device_get(hard))
    if _hard_failure_count(hard_host) != 0:
        raise RuntimeError(f"component-resolved geometry failed a physical hard gate: {hard_host}")

    common_sources = {
        name: _mean_gradient(jax, jnp, source_gradients[name])
        for name in SOURCE_NAMES
    }
    common_grams = _family_gram(
        jax,
        jnp,
        [common_sources[name] for name in SOURCE_NAMES],
    )
    source_geometry = {
        family: _cosine_report(gram, SOURCE_NAMES)
        for family, gram in common_grams.items()
    }
    batch_variance = {}
    for name in SOURCE_NAMES:
        gram = _all_parameter_gram(jax, jnp, source_gradients[name])
        batch_variance[name] = _group_variance_report(gram)

    amended_flux = _add_gradient(
        jax,
        jnp,
        common_sources["seven_day_component_resolved_flux"],
        common_sources["thirty_day_component_resolved_flux"],
    )
    protection_names = ("amended_flux", *PROTECTION_SOURCE_NAMES)
    protection_gradients = [amended_flux] + [common_sources[name] for name in PROTECTION_SOURCE_NAMES]
    protection_grams = _family_gram(jax, jnp, protection_gradients)
    protection_geometry = {
        family: _cosine_report(gram, protection_names)
        for family, gram in protection_grams.items()
    }

    all_reports = [
        record["source_gradients"][name]
        for record in group_records
        for name in SOURCE_NAMES
    ]
    finite = all(
        report["value_count"] == PARAMETER_COUNT
        and report["finite_value_count"] == PARAMETER_COUNT
        and report["l2_norm"] > 0.0
        for report in all_reports
    )
    if not finite:
        raise FloatingPointError("component-resolved geometry has an invalid gradient source")

    memory_after = _memory_report(runtime["devices"][0])
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "action": "geometry_measured_bounded_virtual_step_required",
        "identity": {
            "git_head": _git_head(),
            "flux_basis_amendment_sha256": amendment.sha256,
            "stage_checkpoint": str(args.stage_checkpoint.resolve()),
            "stage_checkpoint_sha256": args.stage_checkpoint_sha256,
            "completed_updates": CHECKPOINT_UPDATE,
            "prepared_manifest": str(args.prepared_manifest.resolve()),
            "prepared_manifest_sha256": sha256_file(args.prepared_manifest),
            "training_contract_sha256": training_contract.sha256,
            "target_contract_sha256": context["target_contract"].sha256,
            "state_anchor_policy_sha256": context["state_anchor_policy"].sha256,
        },
        "policy": {
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "sealed_test_read": False,
            "optimizer_invoked": False,
            "optimizer_updates_retained": 0,
            "checkpoint_written": False,
            "audit_updates": list(AUDIT_UPDATES),
            "flux_basis": list(amendment.flux_basis),
            "stock_basis": list(amendment.stock_basis),
            "gradient_sources": list(SOURCE_NAMES),
        },
        "family_parameter_counts": family_counts,
        "hard_counts": hard_host,
        "hard_failure_count": _hard_failure_count(hard_host),
        "groups": group_records,
        "common_source_geometry": source_geometry,
        "source_batch_variance": batch_variance,
        "amended_flux_vs_protection_geometry": protection_geometry,
        "timing_seconds": {
            "forward": forward_seconds,
            "gradient": gradient_seconds,
            "total": time.perf_counter() - started,
        },
        "memory_before": memory_before,
        "memory_after": memory_after,
    }
    _atomic_json(output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path, required=True)
    parser.add_argument("--base-statistics", type=Path, required=True)
    parser.add_argument("--process-statistics", type=Path, required=True)
    parser.add_argument("--fast-statistics", type=Path, required=True)
    parser.add_argument("--training-contract", type=Path, required=True)
    parser.add_argument("--flux-basis-amendment", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--prepared-updates", type=int, default=CHECKPOINT_UPDATE)
    parser.add_argument("--validation-prepared-manifest", type=Path, required=True)
    parser.add_argument("--stage-checkpoint", type=Path, required=True)
    parser.add_argument("--stage-checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
