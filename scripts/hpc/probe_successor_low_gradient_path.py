"""Locate the first broken derivative on low-gradient successor batches."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.hpc.run_successor_fast_owner_gpu_probe import (
    EXPECTED_PARAMETER_COUNT,
    _head,
    _load_extended_runtime,
)
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_successor_small_fit import (
    _prepare_adapter,
    _retained_tail_trace_signature,
    _slice_rollout_batch,
)
from scripts.hpc.successor_small_fit_prepared import (
    load_prepared_artifact,
    load_prepared_manifest,
)


class _FrozenPredictionHead:
    def assemble_context(self, _prediction: Any, _inputs: Any) -> None:
        return None

    def __call__(self, prediction: Any, _context: Any, _inputs: Any) -> Any:
        return prediction


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _make_parameter_measurement(
    runtime: Mapping[str, Any],
    arm: str,
    adapter_template: Any,
    long_scales: Any,
) -> Any:
    import jax.numpy as jnp

    from scripts.hpc.run_successor_small_fit import _make_arm_objectives

    jax = runtime["jax"]
    _anchor, rollout = _make_arm_objectives(
        runtime,
        arm,
        adapter_template,
        long_scales,
    )

    def tree_norm(tree: Any) -> Any:
        return jnp.sqrt(
            sum(
                (jnp.sum(jnp.asarray(leaf, dtype=jnp.float64) ** 2) for leaf in jax.tree_util.tree_leaves(tree)),
                jnp.asarray(0.0, dtype=jnp.float64),
            )
        )

    def measurement(parameters: Any, batch: Any, day_inputs: Any) -> Any:
        def component_vector(values: Any) -> Any:
            return rollout(
                values,
                batch,
                day_inputs,
                rematerialize=True,
                horizons=(1,),
            )[0]

        losses, pullback = jax.vjp(component_vector, parameters)
        gradients = jax.lax.map(
            lambda seed: pullback(seed)[0],
            jnp.eye(4, dtype=jnp.float64),
        )
        norms = jax.lax.map(
            lambda index: tree_norm(jax.tree_util.tree_map(lambda leaf: leaf[index], gradients)),
            jnp.arange(4),
        )
        return losses, norms

    return jax.jit(measurement)


def _make_named_output_measurement(
    runtime: Mapping[str, Any],
    arm: str,
    adapter_template: Any,
    long_scales: Any,
) -> Any:
    import jax.numpy as jnp

    from research.daily_coarse_graining.daily_operator.model import bind_daily_operator
    from research.daily_coarse_graining.daily_operator.successor_small_fit import (
        successor_rollout_component_losses,
    )

    jax = runtime["jax"]
    head = _head(runtime, arm)

    def tree_norm(tree: Any) -> Any:
        return jnp.sqrt(
            sum(
                (jnp.sum(jnp.asarray(leaf, dtype=jnp.float64) ** 2) for leaf in jax.tree_util.tree_leaves(tree)),
                jnp.asarray(0.0, dtype=jnp.float64),
            )
        )

    def measurement(
        parameters: Any,
        item: Any,
        batch: Any,
        day_inputs: Any,
    ) -> Any:
        assembled = runtime["adapter"].input_assembler.assemble(item)
        context = head.assemble_context(parameters, assembled)
        named_outputs = head.named_outputs(parameters, context, assembled)
        retained_tail = replace(adapter_template, day_inputs=day_inputs)
        transition = bind_daily_operator(
            input_assembler=runtime["adapter"].input_assembler,
            process_head=_FrozenPredictionHead(),
            retained_tail=retained_tail,
        )

        def component_vector(candidate_outputs: Any) -> Any:
            prediction = head.prediction_from_named_outputs(
                candidate_outputs,
                assembled,
            )
            return successor_rollout_component_losses(
                prediction,
                batch,
                transition=transition,
                markov_contract=runtime["contract"],
                flux_scales=long_scales.fluxes,
                stock_scales=long_scales.stocks,
                rematerialize=False,
                horizons=(1,),
            )[0]

        losses, pullback = jax.vjp(component_vector, named_outputs)
        gradients = jax.lax.map(
            lambda seed: pullback(seed)[0],
            jnp.eye(4, dtype=jnp.float64),
        )
        total_norms = jax.lax.map(
            lambda index: tree_norm(jax.tree_util.tree_map(lambda leaf: leaf[index], gradients)),
            jnp.arange(4),
        )
        gpp_norms = jnp.sqrt(
            jnp.sum(
                jnp.asarray(gradients["daily.fast_gpp"], dtype=jnp.float64) ** 2,
                axis=tuple(range(1, gradients["daily.fast_gpp"].ndim)),
            )
        )
        maintenance_norms = jnp.sqrt(
            jnp.sum(
                jnp.asarray(
                    gradients["daily.fast_maintenance_respiration"],
                    dtype=jnp.float64,
                )
                ** 2,
                axis=tuple(
                    range(
                        1,
                        gradients["daily.fast_maintenance_respiration"].ndim,
                    )
                ),
            )
        )
        return (
            losses,
            total_norms,
            gpp_norms,
            maintenance_norms,
            named_outputs["daily.fast_gpp"],
            named_outputs["daily.fast_maintenance_respiration"],
        )

    return jax.jit(measurement)


def _component_record(values: Any, names: tuple[str, ...]) -> Mapping[str, float]:
    host = np.asarray(values, dtype=np.float64)
    return {name: float(host[index]) for index, name in enumerate(names)}


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    runtime = _load_extended_runtime(args)
    jax = runtime["jax"]
    from research.daily_coarse_graining.daily_operator.successor_architecture import (
        SOURCE_ROUTED_ATTENTION,
        SOURCE_ROUTED_GRAPH_CONTROL,
        initialize_successor_parameters,
        successor_parameter_count,
    )
    from research.daily_coarse_graining.daily_operator.successor_small_fit import (
        build_successor_small_fit_schedule,
        load_successor_small_fit_contract,
        verify_successor_small_fit_schedule,
    )
    from research.daily_coarse_graining.daily_operator.successor_training import (
        LONG_HORIZON_COMPONENTS,
        derive_long_horizon_scales,
    )

    contract = load_successor_small_fit_contract(args.runner_contract)
    references = runtime["index"].select(
        spatial_split="train",
        temporal_split="train",
    )
    required_batches = int(contract.payload["objective"]["coefficient_preflight_batches"])
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
    requested = tuple(int(value) for value in args.updates)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("low-gradient probe updates must be unique and nonempty")
    if any(value < 1 or value > required_batches for value in requested):
        raise ValueError("low-gradient probe update is outside the prepared manifest")

    first_artifact = load_prepared_artifact(
        args.prepared_manifest,
        manifest["entries"][0],
        schedule_entry=schedule["entries"][0],
    )
    _reference, adapter_template, adapter_timing = _prepare_adapter(
        runtime,
        schedule["entries"][0],
    )
    signature = _retained_tail_trace_signature(adapter_template)
    if first_artifact.trace_signature != signature:
        raise ValueError("low-gradient probe retained-tail identity drift")
    assembled = runtime["adapter"].input_assembler.assemble(
        jax.tree_util.tree_map(lambda value: value[0], first_artifact.batch.anchor_inputs)
    )
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
        raise ValueError("low-gradient probe parameter count drift")
    scales = derive_long_horizon_scales(
        runtime["base_statistics"].arrays["next_state"].scale,
        runtime["process_statistics"].outputs["daily.fast_gpp"].scale,
        runtime["contract"],
    )
    parameter_measurements = {arm: _make_parameter_measurement(runtime, arm, adapter_template, scales) for arm in arms}
    named_measurements = {arm: _make_named_output_measurement(runtime, arm, adapter_template, scales) for arm in arms}

    started = time.perf_counter()
    records = []
    for update in requested:
        entry = schedule["entries"][update - 1]
        artifact = load_prepared_artifact(
            args.prepared_manifest,
            manifest["entries"][update - 1],
            schedule_entry=entry,
        )
        if artifact.trace_signature != signature:
            raise ValueError("low-gradient probe artifact trace identity drift")
        one_day_batch = _slice_rollout_batch(
            jax,
            artifact.batch,
            1,
            None,
        )
        arm_records = {}
        for arm in arms:
            arm_started = time.perf_counter()
            parameter_losses, parameter_norms = parameter_measurements[arm](
                parameters[arm],
                one_day_batch,
                artifact.day_inputs,
            )
            jax.block_until_ready((parameter_losses, parameter_norms))
            windows = []
            for window_index in range(4):
                window_batch = _slice_rollout_batch(
                    jax,
                    artifact.batch,
                    1,
                    window_index,
                )
                item = jax.tree_util.tree_map(
                    lambda value: value[window_index, 0],
                    artifact.batch.rollout_inputs,
                )
                measured = named_measurements[arm](
                    parameters[arm],
                    item,
                    window_batch,
                    artifact.day_inputs,
                )
                jax.block_until_ready(measured)
                (
                    losses,
                    total_norms,
                    gpp_norms,
                    maintenance_norms,
                    predicted_gpp,
                    predicted_maintenance,
                ) = (np.asarray(value) for value in measured)
                target_gpp = artifact.batch.rollout_targets["daily.fast_gpp"]
                target_maintenance = artifact.batch.rollout_targets["daily.fast_maintenance_respiration"]
                windows.append(
                    {
                        "window_index": window_index,
                        "start": int(entry["rollout_starts"][window_index]),
                        "component_losses": _component_record(
                            losses,
                            LONG_HORIZON_COMPONENTS,
                        ),
                        "named_output_gradient_norms": {
                            "all_outputs": _component_record(
                                total_norms,
                                LONG_HORIZON_COMPONENTS,
                            ),
                            "daily.fast_gpp": _component_record(
                                gpp_norms,
                                LONG_HORIZON_COMPONENTS,
                            ),
                            "daily.fast_maintenance_respiration": _component_record(
                                maintenance_norms,
                                LONG_HORIZON_COMPONENTS,
                            ),
                        },
                        "predicted": {
                            "daily.fast_gpp": predicted_gpp.tolist(),
                            "daily.fast_maintenance_respiration": predicted_maintenance.tolist(),
                        },
                        "teacher": {
                            "daily.fast_gpp": np.asarray(target_gpp.value[window_index, 0]).tolist(),
                            "daily.fast_maintenance_respiration": np.asarray(
                                target_maintenance.value[window_index, 0]
                            ).tolist(),
                        },
                    }
                )
            arm_records[arm] = {
                "one_day_four_window_component_losses": _component_record(
                    parameter_losses,
                    LONG_HORIZON_COMPONENTS,
                ),
                "one_day_four_window_parameter_gradient_norms": _component_record(
                    parameter_norms,
                    LONG_HORIZON_COMPONENTS,
                ),
                "windows": windows,
                "seconds": time.perf_counter() - arm_started,
            }
        records.append(
            {
                "update": update,
                "landpoint_id": str(entry["landpoint_id"]),
                "year": int(entry["year"]),
                "arms": arm_records,
            }
        )

    report = {
        "schema_version": "successor_low_gradient_path_probe_v1",
        "status": "passed",
        "git_head": _git_head(),
        "updates": list(requested),
        "parameter_count_per_arm": counts,
        "adapter_preparation": adapter_timing,
        "records": records,
        "elapsed_seconds": time.perf_counter() - started,
        "training_updates": 0,
        "checkpoint_written": False,
        "teacher_data_regenerated": False,
        "historical_v5_used": False,
        "sealed_test_used": False,
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
    parser.add_argument("--updates", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
