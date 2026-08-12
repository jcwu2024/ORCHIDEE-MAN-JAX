"""Measure shared long-horizon loss coefficients over multiple real-v6 batches."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.hpc.run_successor_fast_owner_gpu_probe import (
    EXPECTED_PARAMETER_COUNT,
    _load_extended_runtime,
)
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_successor_small_fit import (
    _device_memory_report,
    _make_arm_objectives,
    _prepare_adapter,
    _retained_tail_trace_signature,
)
from scripts.hpc.successor_small_fit_prepared import (
    load_prepared_artifact,
    load_prepared_manifest,
    sha256_file,
)


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _aggregate_coefficient_records(
    records: Sequence[Mapping[str, Any]],
    *,
    component_names: Sequence[str],
    targets: Mapping[str, float],
    coefficient_minimum: float,
    coefficient_maximum: float,
    coefficient_aggregation: str = "shared_median_gradient_ratio",
    coefficient_quantile: float | None = None,
) -> Mapping[str, Any]:
    if not records:
        raise ValueError("multibatch coefficient records are empty")
    components = tuple(component_names)
    if set(targets) != set(components):
        raise ValueError("multibatch coefficient target registry drift")
    if coefficient_aggregation == "shared_median_gradient_ratio":
        if coefficient_quantile is not None:
            raise ValueError("median coefficient aggregation does not take a quantile")
    elif coefficient_aggregation == "shared_lower_quantile_gradient_ratio":
        if coefficient_quantile is None or not 0.0 < coefficient_quantile <= 0.5:
            raise ValueError("multibatch coefficient quantile must lie in (0, 0.5]")
    else:
        raise ValueError("unsupported multibatch coefficient aggregation")

    coefficients = {}
    distributions = {}
    for name in components:
        candidates = np.asarray(
            [
                float(targets[name])
                * float(record["anchor_gradient_norm"])
                / float(record["component_gradient_norms"][name])
                for record in records
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(candidates)) or np.any(candidates <= 0.0):
            raise ValueError(f"nonfinite multibatch coefficient candidate for {name}")
        coefficient = (
            float(np.median(candidates))
            if coefficient_aggregation == "shared_median_gradient_ratio"
            else float(np.quantile(candidates, coefficient_quantile, method="lower"))
        )
        if not coefficient_minimum <= coefficient <= coefficient_maximum:
            raise ValueError(f"shared multibatch coefficient for {name} is outside the interval")
        achieved = np.asarray(
            [
                coefficient * float(record["component_gradient_norms"][name]) / float(record["anchor_gradient_norm"])
                for record in records
            ],
            dtype=np.float64,
        )
        coefficients[name] = coefficient
        distributions[name] = {
            "candidate_minimum": float(np.min(candidates)),
            "candidate_p10": float(np.quantile(candidates, 0.10)),
            "candidate_median": float(np.median(candidates)),
            "candidate_p90": float(np.quantile(candidates, 0.90)),
            "candidate_maximum": float(np.max(candidates)),
            "achieved_ratio_minimum": float(np.min(achieved)),
            "achieved_ratio_median": float(np.median(achieved)),
            "achieved_ratio_maximum": float(np.max(achieved)),
        }
    result = {
        "resolved_coefficients": coefficients,
        "component_distributions": distributions,
        "record_count": len(records),
    }
    if coefficient_aggregation == "shared_lower_quantile_gradient_ratio":
        result.update(
            {
                "coefficient_quantile": coefficient_quantile,
                "quantile_method": "lower",
                "coefficient_aggregation": coefficient_aggregation,
            }
        )
    return result


def _make_measurement(runtime: Mapping[str, Any], arm: str, adapter: Any, long_scales: Any) -> Any:
    import jax.numpy as jnp

    from research.daily_coarse_graining.daily_operator.successor_small_fit import (
        successor_rollout_hard_gate_counts,
    )
    from research.daily_coarse_graining.daily_operator.successor_training import (
        LONG_HORIZON_COMPONENTS,
    )

    jax = runtime["jax"]
    anchor_objective, rollout_objective = _make_arm_objectives(
        runtime,
        arm,
        adapter,
        long_scales,
    )

    def tree_norm(tree: Any) -> Any:
        return jnp.sqrt(
            sum(
                (jnp.sum(jnp.asarray(leaf, dtype=jnp.float64) ** 2) for leaf in jax.tree_util.tree_leaves(tree)),
                jnp.asarray(0.0, dtype=jnp.float64),
            )
        )

    def tree_nonfinite(tree: Any) -> Any:
        return sum(
            (jnp.sum(~jnp.isfinite(jnp.asarray(leaf))) for leaf in jax.tree_util.tree_leaves(tree)),
            jnp.asarray(0, dtype=jnp.int64),
        )

    def measurement(parameters: Any, batch: Any, day_inputs: Any) -> Any:
        anchor_value, anchor_gradient = jax.value_and_grad(lambda values: anchor_objective(values, batch))(parameters)

        def component_vector(values: Any) -> Any:
            components, trace = rollout_objective(
                values,
                batch,
                day_inputs,
                rematerialize=True,
            )
            return components, trace

        component_values, pullback, trace = jax.vjp(
            component_vector,
            parameters,
            has_aux=True,
        )
        basis = jnp.eye(len(LONG_HORIZON_COMPONENTS), dtype=jnp.float64)
        component_gradients = jax.lax.map(lambda seed: pullback(seed)[0], basis)
        component_norms = jax.lax.map(
            lambda index: tree_norm(jax.tree_util.tree_map(lambda leaf: leaf[index], component_gradients)),
            jnp.arange(len(LONG_HORIZON_COMPONENTS)),
        )
        component_nonfinite = jax.lax.map(
            lambda index: tree_nonfinite(jax.tree_util.tree_map(lambda leaf: leaf[index], component_gradients)),
            jnp.arange(len(LONG_HORIZON_COMPONENTS)),
        )
        hard_counts = successor_rollout_hard_gate_counts(
            batch,
            trace,
            markov_contract=runtime["contract"],
        )
        return (
            anchor_value,
            component_values,
            tree_norm(anchor_gradient),
            component_norms,
            tree_nonfinite(anchor_gradient),
            component_nonfinite,
            hard_counts,
        )

    return jax.jit(measurement)


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"multibatch coefficient output already exists: {args.output}")
    runtime = _load_extended_runtime(args)
    jax = runtime["jax"]

    from research.daily_coarse_graining.daily_operator.successor_architecture import (
        SOURCE_ROUTED_ATTENTION,
        SOURCE_ROUTED_GRAPH_CONTROL,
        initialize_successor_parameters,
        successor_parameter_count,
    )
    from research.daily_coarse_graining.daily_operator.successor_small_fit import (
        SUFFICIENCY_SCHEMA_VERSION,
        build_successor_small_fit_schedule,
        load_successor_small_fit_contract,
        verify_successor_small_fit_schedule,
    )
    from research.daily_coarse_graining.daily_operator.successor_training import (
        LONG_HORIZON_COMPONENTS,
        derive_long_horizon_scales,
    )

    contract = load_successor_small_fit_contract(args.runner_contract)
    if contract.payload["schema_version"] != SUFFICIENCY_SCHEMA_VERSION:
        raise ValueError("multibatch coefficient probe requires the sufficiency-v2 contract")
    required_batches = int(contract.payload["objective"]["coefficient_preflight_batches"])
    references = runtime["index"].select(spatial_split="train", temporal_split="train")
    schedule = build_successor_small_fit_schedule(
        references,
        contract,
        updates=required_batches,
    )
    verify_successor_small_fit_schedule(schedule, references, contract)
    raw_manifest = json.loads(args.prepared_manifest.read_text(encoding="utf-8"))
    manifest = load_prepared_manifest(
        args.prepared_manifest,
        runner_contract_sha256=contract.sha256,
        schedule_sha256=schedule["schedule_sha256"],
        updates=required_batches,
        source_git_head=str(raw_manifest["identity"]["source_git_head"]),
    )

    first_entry = schedule["entries"][0]
    first_artifact = load_prepared_artifact(
        args.prepared_manifest,
        manifest["entries"][0],
        schedule_entry=first_entry,
    )
    _reference, adapter, adapter_timing = _prepare_adapter(runtime, first_entry)
    first_signature = _retained_tail_trace_signature(adapter)
    if first_artifact.trace_signature != first_signature:
        raise ValueError("multibatch coefficient trace identity drift")
    adapter = replace(adapter, day_inputs=first_artifact.day_inputs)

    first_input = jax.tree_util.tree_map(lambda value: value[0], first_artifact.batch.anchor_inputs)
    assembled = runtime["adapter"].input_assembler.assemble(first_input)
    arms = (SOURCE_ROUTED_ATTENTION, SOURCE_ROUTED_GRAPH_CONTROL)
    parameters = {
        arm: initialize_successor_parameters(
            jax.random.PRNGKey(int(contract.sampling["seed"])),
            schema=runtime["schema"],
            layout=runtime["layout"],
            inputs=assembled,
            arm=arm,
        )
        for arm in arms
    }
    counts = {arm: successor_parameter_count(parameters[arm]) for arm in arms}
    if set(counts.values()) != {EXPECTED_PARAMETER_COUNT}:
        raise ValueError(f"multibatch coefficient parameter count drift: {counts}")
    long_scales = derive_long_horizon_scales(
        runtime["base_statistics"].arrays["next_state"].scale,
        runtime["process_statistics"].outputs["daily.fast_gpp"].scale,
        runtime["contract"],
    )
    measurements = {arm: _make_measurement(runtime, arm, adapter, long_scales) for arm in arms}

    memory_before = _device_memory_report(runtime["devices"])
    records = []
    total_started = time.perf_counter()
    for index, entry in enumerate(schedule["entries"]):
        load_started = time.perf_counter()
        artifact = (
            first_artifact
            if index == 0
            else load_prepared_artifact(
                args.prepared_manifest,
                manifest["entries"][index],
                schedule_entry=entry,
            )
        )
        load_seconds = time.perf_counter() - load_started
        if artifact.trace_signature != first_signature:
            raise ValueError("multibatch coefficient artifact trace drift")
        for arm in arms:
            started = time.perf_counter()
            measured = measurements[arm](
                parameters[arm],
                artifact.batch,
                artifact.day_inputs,
            )
            jax.block_until_ready(measured)
            elapsed = time.perf_counter() - started
            (
                anchor_value,
                component_values,
                anchor_norm,
                component_norms,
                anchor_bad,
                component_bad,
                hard_counts,
            ) = measured
            anchor_value = np.asarray(anchor_value)
            component_values = np.asarray(component_values)
            anchor_norm = np.asarray(anchor_norm)
            component_norms = np.asarray(component_norms)
            anchor_bad = np.asarray(anchor_bad)
            component_bad = np.asarray(component_bad)
            hard_counts_host = {
                name: int(np.asarray(value))
                for name, value in hard_counts._asdict().items()
            }
            if (
                not np.isfinite(anchor_value)
                or not np.all(np.isfinite(component_values))
                or not np.isfinite(anchor_norm)
                or not np.all(np.isfinite(component_norms))
                or float(anchor_norm) <= 0.0
                or np.any(component_norms <= 0.0)
                or int(anchor_bad) != 0
                or np.any(component_bad != 0)
            ):
                raise ValueError(f"nonfinite multibatch coefficient measurement at {index + 1}/{arm}")
            records.append(
                {
                    "update": index + 1,
                    "landpoint_id": str(entry["landpoint_id"]),
                    "year": int(entry["year"]),
                    "arm": arm,
                    "anchor_loss": float(anchor_value),
                    "component_losses": {
                        name: float(component_values[position]) for position, name in enumerate(LONG_HORIZON_COMPONENTS)
                    },
                    "anchor_gradient_norm": float(anchor_norm),
                    "component_gradient_norms": {
                        name: float(component_norms[position]) for position, name in enumerate(LONG_HORIZON_COMPONENTS)
                    },
                    "hard_gate_counts": hard_counts_host,
                    "artifact_load_seconds": load_seconds,
                    "measurement_seconds": elapsed,
                }
            )
            print(
                json.dumps(
                    {
                        "event": "successor_multibatch_coefficient_measurement",
                        "update": index + 1,
                        "arm": arm,
                        "seconds": elapsed,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    objective = contract.payload["objective"]
    aggregation = _aggregate_coefficient_records(
        records,
        component_names=LONG_HORIZON_COMPONENTS,
        targets=objective["gradient_ratio_targets"],
        coefficient_minimum=float(objective["coefficient_minimum"]),
        coefficient_maximum=float(objective["coefficient_maximum"]),
        coefficient_aggregation=objective["coefficient_aggregation"],
        coefficient_quantile=(
            float(objective["coefficient_quantile"])
            if "coefficient_quantile" in objective
            else None
        ),
    )
    hard_gate_counts = {
        name: sum(int(record["hard_gate_counts"][name]) for record in records)
        for name in records[0]["hard_gate_counts"]
    }
    hard_failure_count = sum(hard_gate_counts.values())
    status = "passed" if hard_failure_count == 0 else "failed_hard_gates"
    report = {
        "schema_version": "successor_multibatch_coefficient_probe_v1",
        "status": status,
        "git_head": _git_head(),
        "runner_contract_sha256": contract.sha256,
        "prepared_manifest_sha256": sha256_file(args.prepared_manifest),
        "prepared_artifact_compression": manifest.get("artifact_compression", "none"),
        "parameter_count_per_arm": counts,
        "batch_count": required_batches,
        "arm_count": len(arms),
        "measurement_record_count": len(records),
        "same_coefficients_for_both_arms": True,
        "coefficient_aggregation": objective["coefficient_aggregation"],
        "aggregation": aggregation,
        "hard_gate_counts": hard_gate_counts,
        "hard_failure_count": hard_failure_count,
        "measurements": records,
        "adapter_preparation": adapter_timing,
        "device_memory_before": memory_before,
        "device_memory_after": _device_memory_report(runtime["devices"]),
        "elapsed_seconds": time.perf_counter() - total_started,
        "training_updates": 0,
        "checkpoint_written": False,
        "teacher_data_regenerated": False,
        "historical_v5_used": False,
        "sealed_test_used": False,
        "next_gate": (
            "review_shared_multibatch_coefficients_and_freeze_validation_panel"
            if status == "passed"
            else "diagnose_multibatch_state_or_budget_hard_gate_failure"
        ),
    }
    _atomic_json(args.output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path, required=True)
    parser.add_argument("--base-statistics", type=Path, required=True)
    parser.add_argument("--process-statistics", type=Path, required=True)
    parser.add_argument("--fast-statistics", type=Path, required=True)
    parser.add_argument("--runner-contract", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
