"""Run one real-v6 no-checkpoint successor small-fit update on a GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.hpc.run_successor_fast_owner_gpu_probe import (
    EXPECTED_PARAMETER_COUNT,
    _head,
    _load_extended_runtime,
    _tree_numeric_report,
)
from scripts.hpc.run_successor_process_head_canonical_gate import (
    _atomic_json,
    _retained_adapter,
)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _plan_context(
    manifest: Mapping[str, Any],
    *,
    landpoint_id: str,
    year: int,
) -> Mapping[str, Any]:
    plan_path = Path(str(manifest["plan"])).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if _canonical_sha256(plan) != str(manifest["plan_sha256"]):
        raise ValueError("v6 generation plan hash drift")
    entries = tuple(
        item for item in plan["entries"] if str(item["landpoint_id"]) == landpoint_id and int(item["year"]) == year
    )
    if len(entries) != 1:
        raise ValueError("small-fit reference has no unique generation-plan entry")
    entry = entries[0]
    result = {
        "config": Path(str(plan["teacher_config"])).resolve(),
        "run_def": Path(str(entry["run_def"])).resolve(),
        "reference_run_dir": Path(str(entry["reference_run_dir"])).resolve(),
        "plan": plan_path,
    }
    if any(not path.exists() for path in result.values()):
        raise FileNotFoundError("small-fit generation-plan context is incomplete")
    return result


def _reference_for_entry(index: Any, entry: Mapping[str, Any]) -> Any:
    matches = tuple(
        reference
        for reference in index.select(spatial_split="train", temporal_split="train")
        if str(reference.parent.landpoint_id) == str(entry["landpoint_id"])
        and int(reference.parent.year) == int(entry["year"])
    )
    if len(matches) != 1:
        raise ValueError("small-fit schedule entry does not resolve to one coherent shard")
    return matches[0]


def _host_preflight(preflight: Any) -> Mapping[str, Any]:
    return {
        "split_label": preflight.split_label,
        "anchor_loss": float(np.asarray(preflight.anchor_loss)),
        "component_losses": {name: float(np.asarray(value)) for name, value in preflight.component_losses.items()},
        "anchor_gradient_norm": float(np.asarray(preflight.anchor_gradient_norm)),
        "component_gradient_norms": {
            name: float(np.asarray(value)) for name, value in preflight.component_gradient_norms.items()
        },
        "target_gradient_ratios": dict(preflight.target_gradient_ratios),
        "resolved_coefficients": dict(preflight.resolved_coefficients),
        "achieved_gradient_ratios": dict(preflight.achieved_gradient_ratios),
    }


def _leaf_failures(
    leaves: Any,
    actual: np.ndarray,
    expected: np.ndarray,
    actual_defined: np.ndarray,
    expected_defined: np.ndarray,
    tolerance: np.ndarray,
) -> list[Mapping[str, Any]]:
    records = []
    for leaf in leaves:
        if leaf.start is None or leaf.stop is None:
            continue
        selected = slice(int(leaf.start), int(leaf.stop))
        left_mask = actual_defined[selected]
        right_mask = expected_defined[selected]
        common = left_mask & right_mask
        difference = np.abs(actual[selected] - expected[selected])
        admitted = tolerance[selected]
        mask_mismatches = int(np.count_nonzero(left_mask != right_mask))
        numerical_failures = int(np.count_nonzero(common & (difference > admitted)))
        expected_undefined = ~right_mask
        undefined_equal = (actual[selected] == expected[selected]) | (
            np.isnan(actual[selected]) & np.isnan(expected[selected])
        )
        undefined_representation_mismatches = int(np.count_nonzero(expected_undefined & ~undefined_equal))
        if mask_mismatches or numerical_failures or undefined_representation_mismatches:
            records.append(
                {
                    "key": str(leaf.key),
                    "mask_mismatch_count": mask_mismatches,
                    "numerical_failure_count": numerical_failures,
                    "undefined_representation_mismatch_count": (undefined_representation_mismatches),
                    "actual_defined_count": int(np.count_nonzero(left_mask)),
                    "expected_defined_count": int(np.count_nonzero(right_mask)),
                    "maximum_common_absolute_error": (float(np.max(difference[common])) if np.any(common) else 0.0),
                }
            )
    return records


class _TeacherOutputHead:
    def __init__(self, base: Any):
        self.base = base

    def assemble_context(self, parameters: Any, inputs: Any) -> Any:
        del parameters
        return inputs.day_index

    def __call__(self, parameters: Any, context: Any, inputs: Any) -> Any:
        del context
        return self.base.prediction_from_named_outputs(parameters, inputs)


def _teacher_replay_gate(
    runtime: Mapping[str, Any],
    batch: Any,
    retained_tail: Any,
) -> Mapping[str, Any]:
    from research.daily_coarse_graining.daily_operator.model import bind_daily_operator
    from research.daily_coarse_graining.daily_operator.retained_tail import (
        retained_tail_input,
    )
    from research.daily_coarse_graining.daily_operator.successor_architecture import (
        SOURCE_ROUTED_ATTENTION,
    )
    from research.daily_coarse_graining.daily_operator.successor_small_fit import (
        successor_teacher_physical_outputs,
    )

    jax = runtime["jax"]
    head = _head(runtime, SOURCE_ROUTED_ATTENTION)
    replay_head = _TeacherOutputHead(head)
    transition = jax.jit(
        bind_daily_operator(
            input_assembler=runtime["adapter"].input_assembler,
            process_head=replay_head,
            retained_tail=retained_tail,
        )
    )
    cases = []
    for batch_index in range(int(np.shape(batch.rollout_inputs.year)[0])):
        inputs = jax.tree_util.tree_map(
            lambda value: value[batch_index, 0],
            batch.rollout_inputs,
        )
        targets = jax.tree_util.tree_map(
            lambda value: value[batch_index, 0],
            batch.rollout_targets,
        )
        assembled = runtime["adapter"].input_assembler.assemble(inputs)
        outputs = successor_teacher_physical_outputs(
            targets,
            assembled,
            runtime["schema"],
        )
        result = transition(outputs, inputs)
        jax.block_until_ready(result)
        internal_target = np.asarray(
            retained_tail._typed_fast_target(
                retained_tail_input(
                    assembled,
                    result.prediction,
                    result.fast_update,
                )
            )
        )
        expected_target = np.asarray(batch.expected_fast_day_target[batch_index, 0])
        internal_defined = np.isfinite(internal_target) & (np.abs(internal_target) < 0.5e20)
        target_defined = np.isfinite(expected_target) & (np.abs(expected_target) < 0.5e20)
        target_common = internal_defined & target_defined
        target_difference = np.abs(internal_target - expected_target)
        target_tolerance = 1.0e-8 + 1.0e-10 * np.maximum(
            np.abs(internal_target),
            np.abs(expected_target),
        )
        fast_target_passed = np.array_equal(internal_defined, target_defined) and bool(
            np.all(target_difference[target_common] <= target_tolerance[target_common])
        )
        fast_target_failures = _leaf_failures(
            runtime["contract"].fast_day_target_leaves,
            internal_target,
            expected_target,
            internal_defined,
            target_defined,
            target_tolerance,
        )
        actual = np.asarray(result.next_state.other)
        actual_defined = np.asarray(result.next_state_defined.other, dtype=np.bool_)
        expected = np.asarray(batch.expected_next_state[batch_index, 0])
        expected_defined = np.asarray(
            batch.expected_next_state_defined[batch_index, 0],
            dtype=np.bool_,
        )
        common = actual_defined & expected_defined
        undefined_exact = np.array_equal(
            actual[~expected_defined],
            expected[~expected_defined],
            equal_nan=True,
        )
        difference = np.abs(actual - expected)
        tolerance = 1.0e-8 + 1.0e-10 * np.maximum(np.abs(actual), np.abs(expected))
        state_failures = _leaf_failures(
            runtime["contract"].state_leaves,
            actual,
            expected,
            actual_defined,
            expected_defined,
            tolerance,
        )
        discrete_equal = all(
            np.array_equal(np.asarray(value), np.asarray(batch.expected_next_discrete_state[name][batch_index, 0]))
            for name, value in result.next_discrete_state.items()
        )
        water = float(np.asarray(result.diagnostics["water_budget_residual"]))
        carbon = float(np.asarray(result.diagnostics["carbon_budget_residual"]))
        passed = (
            fast_target_passed
            and np.array_equal(actual_defined, expected_defined)
            and undefined_exact
            and bool(np.all(difference[common] <= tolerance[common]))
            and discrete_equal
            and abs(water) <= 2.0e-12
            and abs(carbon) <= 2.0e-12
        )
        cases.append(
            {
                "status": "passed" if passed else "failed",
                "batch_index": batch_index,
                "day_index": int(np.asarray(inputs.day_index)),
                "fast_target": {
                    "status": "passed" if fast_target_passed else "failed",
                    "defined_mask_exact": bool(np.array_equal(internal_defined, target_defined)),
                    "maximum_defined_absolute_error": (
                        float(np.max(target_difference[target_common])) if np.any(target_common) else 0.0
                    ),
                    "maximum_defined_tolerance_ratio": (
                        float(np.max(target_difference[target_common] / target_tolerance[target_common]))
                        if np.any(target_common)
                        else 0.0
                    ),
                    "leaf_failures": fast_target_failures,
                },
                "defined_mask_exact": bool(np.array_equal(actual_defined, expected_defined)),
                "undefined_representation_exact": bool(undefined_exact),
                "maximum_defined_absolute_error": (float(np.max(difference[common])) if np.any(common) else 0.0),
                "maximum_defined_tolerance_ratio": (
                    float(np.max(difference[common] / tolerance[common])) if np.any(common) else 0.0
                ),
                "state_leaf_failures": state_failures,
                "discrete_state_exact": discrete_equal,
                "water_budget_residual": water,
                "carbon_budget_residual": carbon,
            }
        )
    return {
        "status": ("passed" if all(case["status"] == "passed" for case in cases) else "failed"),
        "cases": cases,
    }


def _probe_arm(
    runtime: Mapping[str, Any],
    batch: Any,
    arm: str,
    parameters: Any,
    coefficients: Mapping[str, float],
    retained_tail: Any,
    long_scales: Any,
    runner_contract: Any,
) -> Mapping[str, Any]:
    import jax.numpy as jnp

    from research.daily_coarse_graining.daily_operator.model import bind_daily_operator
    from research.daily_coarse_graining.daily_operator.successor_small_fit import (
        initialize_successor_adamw,
        successor_adamw_update,
        successor_anchor_batch_loss,
        successor_learning_rate,
        successor_rollout_component_losses,
        successor_rollout_hard_gate_counts,
        successor_small_fit_loss,
    )

    jax = runtime["jax"]
    head = _head(runtime, arm)
    transition = bind_daily_operator(
        input_assembler=runtime["adapter"].input_assembler,
        process_head=head,
        retained_tail=retained_tail,
    )

    def objective(values):
        return successor_small_fit_loss(
            values,
            batch,
            head=head,
            schema=runtime["schema"],
            loss_scales=runtime["loss_scales"],
            transition=transition,
            markov_contract=runtime["contract"],
            flux_scales=long_scales.fluxes,
            stock_scales=long_scales.stocks,
            coefficients=coefficients,
            rematerialize=True,
        )

    optimizer = initialize_successor_adamw(parameters)
    rate = successor_learning_rate(runner_contract, 1)

    @jax.jit
    def train_step(values, state):
        (loss, auxiliary), gradients = jax.value_and_grad(
            objective,
            has_aux=True,
        )(values)
        updated, next_state, norm = successor_adamw_update(
            values,
            gradients,
            state,
            jnp.asarray(rate, dtype=jnp.float64),
            runner_contract,
        )
        return updated, next_state, loss, auxiliary, gradients, norm

    started = time.perf_counter()
    updated, optimizer, before, before_aux, gradients, gradient_norm = train_step(
        parameters,
        optimizer,
    )
    jax.block_until_ready((updated, before, gradients))
    train_seconds = time.perf_counter() - started

    @jax.jit
    def post_update(values):
        anchor = successor_anchor_batch_loss(
            values,
            batch.anchor_inputs,
            batch.anchor_targets,
            head=head,
            schema=runtime["schema"],
            scales=runtime["loss_scales"],
        )
        components, trace = successor_rollout_component_losses(
            values,
            batch,
            transition=transition,
            markov_contract=runtime["contract"],
            flux_scales=long_scales.fluxes,
            stock_scales=long_scales.stocks,
            rematerialize=False,
        )
        coefficient_vector = jnp.asarray(
            tuple(coefficients[name] for name in coefficients),
            dtype=jnp.float64,
        )
        total = anchor + jnp.sum(coefficient_vector * components)
        hard_counts = successor_rollout_hard_gate_counts(
            batch,
            trace,
            markov_contract=runtime["contract"],
            budget_atol=float(runner_contract.payload["gates"]["budget_absolute_tolerance"]),
            stock_atol=float(runner_contract.payload["gates"]["stock_negative_absolute_tolerance"]),
        )
        return total, anchor, components, hard_counts

    started = time.perf_counter()
    after, after_anchor, after_components, hard_counts = post_update(updated)
    jax.block_until_ready((after, hard_counts))
    post_seconds = time.perf_counter() - started

    before_anchor, before_components = before_aux
    gradient_report = _tree_numeric_report(jax, gradients)
    parameter_report = _tree_numeric_report(jax, updated)
    hard_host = {name: int(np.asarray(value)) for name, value in hard_counts._asdict().items()}
    values = (
        float(np.asarray(before)),
        float(np.asarray(before_anchor)),
        *np.asarray(before_components).tolist(),
        float(np.asarray(after)),
        float(np.asarray(after_anchor)),
        *np.asarray(after_components).tolist(),
    )
    passed = (
        all(np.isfinite(value) for value in values)
        and gradient_report["all_finite"]
        and gradient_report["l2_norm"] > 0.0
        and parameter_report["all_finite"]
        and all(value == 0 for value in hard_host.values())
        and int(np.asarray(optimizer.step)) == 1
    )
    return {
        "status": "passed" if passed else "failed",
        "arm": arm,
        "learning_rate": rate,
        "train_compile_and_update_seconds": train_seconds,
        "post_update_compile_and_forward_seconds": post_seconds,
        "before": {
            "total": float(np.asarray(before)),
            "one_day_anchor": float(np.asarray(before_anchor)),
            "long_horizon_components": {
                name: float(value)
                for name, value in zip(
                    coefficients,
                    np.asarray(before_components),
                    strict=True,
                )
            },
        },
        "after": {
            "total": float(np.asarray(after)),
            "one_day_anchor": float(np.asarray(after_anchor)),
            "long_horizon_components": {
                name: float(value)
                for name, value in zip(
                    coefficients,
                    np.asarray(after_components),
                    strict=True,
                )
            },
        },
        "gradient": gradient_report | {"preclip_l2_norm": float(gradient_norm)},
        "updated_parameters": parameter_report,
        "hard_gate_counts": hard_host,
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"probe output already exists: {args.output}")
    runtime = _load_extended_runtime(args)
    jax = runtime["jax"]

    from jax_orchidee.driver import orchestration as teacher
    from research.daily_coarse_graining.daily_markov_contract import (
        load_markov_shard,
    )
    from research.daily_coarse_graining.daily_operator.model import bind_daily_operator
    from research.daily_coarse_graining.daily_operator.successor_architecture import (
        SOURCE_ROUTED_ATTENTION,
        SOURCE_ROUTED_GRAPH_CONTROL,
        initialize_successor_parameters,
        successor_parameter_count,
    )
    from research.daily_coarse_graining.daily_operator.successor_small_fit import (
        build_successor_small_fit_schedule,
        load_successor_small_fit_contract,
        prepare_successor_small_fit_batch,
        successor_anchor_batch_loss,
        successor_rollout_component_losses,
        verify_successor_small_fit_schedule,
    )
    from research.daily_coarse_graining.daily_operator.successor_training import (
        LONG_HORIZON_COMPONENTS,
        GradientRatioPreflightError,
        calibrate_gradient_ratio_coefficients,
        derive_long_horizon_scales,
    )

    runner_contract = load_successor_small_fit_contract(args.runner_contract)
    identity = runner_contract.payload["identity"]
    if (
        identity["successor_candidate_contract_sha256"] != runtime["successor_contract"].sha256
        or identity["fast_owner_contract_sha256"] != runtime["owner_contract"].contract_sha256
    ):
        raise ValueError("small-fit runner and model contracts differ")

    references = runtime["index"].select(
        spatial_split="train",
        temporal_split="train",
    )
    schedule = build_successor_small_fit_schedule(references, runner_contract)
    verify_successor_small_fit_schedule(schedule, references, runner_contract)
    entry = schedule["entries"][0]
    reference = _reference_for_entry(runtime["index"], entry)

    preparation_started = time.perf_counter()
    batch = prepare_successor_small_fit_batch(
        reference,
        entry,
        adapter=runtime["adapter"],
        schema=runtime["schema"],
        owner_contract=runtime["owner_contract"],
    )
    preparation_seconds = time.perf_counter() - preparation_started

    context_paths = _plan_context(
        runtime["manifest"],
        landpoint_id=str(entry["landpoint_id"]),
        year=int(entry["year"]),
    )
    context_started = time.perf_counter()
    teacher_context = teacher.prepare_paper_1961_driver_context(
        context_paths["config"],
        used_run_def_path=context_paths["run_def"],
        reference_run_dir=context_paths["reference_run_dir"],
    )
    parent = load_markov_shard(reference.parent.path, contract=runtime["contract"])
    sample = parent.sample(int(entry["rollout_starts"][0]))
    scoped_runtime = runtime | {"context_paths": context_paths}
    retained_tail = _retained_adapter(
        scoped_runtime,
        teacher_context,
        sample,
        lifecycle_start_tstep=48,
        result_day_index=2,
        fast_owner_materializer=runtime["materializer"],
    )
    context_seconds = time.perf_counter() - context_started
    teacher_replay = _teacher_replay_gate(runtime, batch, retained_tail)
    if teacher_replay["status"] != "passed":
        report = {
            "schema_version": "successor_small_fit_real_v6_single_update_probe_v1",
            "status": "failed",
            "failed_stage": "annual_shard_teacher_replay",
            "checkpoint_written": False,
            "training_updates_per_arm": 0,
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in runtime["devices"]],
            "identity": dict(identity)
            | {
                "runner_contract_sha256": runner_contract.sha256,
                "schedule_sha256": schedule["schedule_sha256"],
            },
            "schedule_entry": entry,
            "host_preparation_seconds": preparation_seconds,
            "teacher_context_and_retained_adapter_seconds": context_seconds,
            "annual_shard_teacher_replay": teacher_replay,
            "next_gate": "repair_annual_shard_retained_tail_runtime",
        }
        _atomic_json(args.output, report)
        return report

    first_input = jax.tree_util.tree_map(lambda value: value[0], batch.anchor_inputs)
    assembled = runtime["adapter"].input_assembler.assemble(first_input)
    arms = (SOURCE_ROUTED_ATTENTION, SOURCE_ROUTED_GRAPH_CONTROL)
    parameters = {
        arm: initialize_successor_parameters(
            jax.random.PRNGKey(int(runner_contract.sampling["seed"])),
            schema=runtime["schema"],
            layout=runtime["layout"],
            inputs=assembled,
            arm=arm,
        )
        for arm in arms
    }
    counts = {arm: successor_parameter_count(value) for arm, value in parameters.items()}
    if set(counts.values()) != {EXPECTED_PARAMETER_COUNT}:
        raise ValueError(f"small-fit parameter count drift: {counts}")

    long_scales = derive_long_horizon_scales(
        runtime["base_statistics"].arrays["next_state"].scale,
        runtime["process_statistics"].outputs["daily.fast_gpp"].scale,
        runtime["contract"],
    )
    candidate_head = _head(runtime, SOURCE_ROUTED_ATTENTION)
    candidate_transition = bind_daily_operator(
        input_assembler=runtime["adapter"].input_assembler,
        process_head=candidate_head,
        retained_tail=retained_tail,
    )

    def anchor_loss(values):
        return successor_anchor_batch_loss(
            values,
            batch.anchor_inputs,
            batch.anchor_targets,
            head=candidate_head,
            schema=runtime["schema"],
            scales=runtime["loss_scales"],
        )

    def component_vector(values):
        return successor_rollout_component_losses(
            values,
            batch,
            transition=candidate_transition,
            markov_contract=runtime["contract"],
            flux_scales=long_scales.fluxes,
            stock_scales=long_scales.stocks,
            rematerialize=True,
        )[0]

    component_losses = {
        name: (lambda values, index=index: component_vector(values)[index])
        for index, name in enumerate(LONG_HORIZON_COMPONENTS)
    }
    objective_contract = runner_contract.payload["objective"]
    preflight_started = time.perf_counter()
    try:
        preflight = calibrate_gradient_ratio_coefficients(
            parameters[SOURCE_ROUTED_ATTENTION],
            anchor_loss,
            component_losses,
            target_gradient_ratios=objective_contract["gradient_ratio_targets"],
            coefficient_minimum=float(objective_contract["coefficient_minimum"]),
            coefficient_maximum=float(objective_contract["coefficient_maximum"]),
            split_label="train/train",
            compile_gradients=True,
            combined_component_loss=component_vector,
        )
    except GradientRatioPreflightError as error:
        report = {
            "schema_version": "successor_small_fit_real_v6_single_update_probe_v1",
            "status": "failed",
            "failed_stage": "coefficient_gradient_preflight",
            "checkpoint_written": False,
            "training_updates_per_arm": 0,
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "backend": jax.default_backend(),
            "devices": [str(device) for device in runtime["devices"]],
            "identity": dict(identity)
            | {
                "runner_contract_sha256": runner_contract.sha256,
                "schedule_sha256": schedule["schedule_sha256"],
            },
            "schedule_entry": entry,
            "host_preparation_seconds": preparation_seconds,
            "teacher_context_and_retained_adapter_seconds": context_seconds,
            "annual_shard_teacher_replay": teacher_replay,
            "coefficient_preflight_seconds": time.perf_counter() - preflight_started,
            "coefficient_preflight_failure": error.diagnostic,
            "parameter_count_per_arm": counts,
            "next_gate": "repair_nonfinite_long_horizon_gradient",
        }
        _atomic_json(args.output, report)
        return report
    jax.block_until_ready(
        (
            preflight.anchor_loss,
            preflight.component_losses,
            preflight.anchor_gradient_norm,
            preflight.component_gradient_norms,
        )
    )
    preflight_seconds = time.perf_counter() - preflight_started
    coefficients = {name: float(preflight.resolved_coefficients[name]) for name in LONG_HORIZON_COMPONENTS}

    arm_reports = {
        arm: _probe_arm(
            runtime,
            batch,
            arm,
            parameters[arm],
            coefficients,
            retained_tail,
            long_scales,
            runner_contract,
        )
        for arm in arms
    }
    passed = all(value["status"] == "passed" for value in arm_reports.values())
    report = {
        "schema_version": "successor_small_fit_real_v6_single_update_probe_v1",
        "status": "passed" if passed else "failed",
        "checkpoint_written": False,
        "training_updates_per_arm": 1,
        "teacher_data_regenerated": False,
        "historical_v5_used": False,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in runtime["devices"]],
        "identity": dict(identity)
        | {
            "runner_contract_sha256": runner_contract.sha256,
            "schedule_sha256": schedule["schedule_sha256"],
        },
        "schedule_entry": entry,
        "host_preparation_seconds": preparation_seconds,
        "teacher_context_and_retained_adapter_seconds": context_seconds,
        "annual_shard_teacher_replay": teacher_replay,
        "coefficient_preflight_seconds": preflight_seconds,
        "coefficient_preflight": _host_preflight(preflight),
        "parameter_count_per_arm": counts,
        "matched_coefficients_shared_by_both_arms": True,
        "arms": arm_reports,
        "boundary": {
            "binding": runner_contract.payload["retained_tail"]["binding"],
            "native_forcing_records_per_day": 5,
            "reconstructs_48_forcing_records": False,
            "executes_48_step_state_scan": False,
        },
        "next_gate": "freeze_paid_resource_envelope" if passed else "repair_small_fit_runner",
    }
    _atomic_json(args.output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path, required=True)
    parser.add_argument("--base-statistics", type=Path, required=True)
    parser.add_argument("--process-statistics", type=Path, required=True)
    parser.add_argument("--fast-statistics", type=Path, required=True)
    parser.add_argument("--runner-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
