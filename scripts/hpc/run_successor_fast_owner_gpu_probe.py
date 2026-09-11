"""Run the bounded real-v6 GPU probe for the complete successor decoder."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.hpc.run_successor_process_head_canonical_gate import (
    ANCHOR_ROWS,
    EXPECTED,
    FROZEN_LANDPOINT,
    _atomic_json,
    _load_runtime,
    _retained_adapter,
    _verify,
)

FAST_STATISTICS_JSON_SHA256 = "b2e61a171156071d404d12b833130f2d0e98064564d1c3e007e16f0210734719"
FAST_STATISTICS_NPZ_SHA256 = "890d1fd6110912a835c760b4fe58e9320481a929797858cad5b70b1bf03eb99f"
EXPECTED_PARAMETER_COUNT = 8_722_689
PROBE_SEED = 20260815
FORWARD_ATOL = 1.0e-8
FORWARD_RTOL = 1.0e-10


def _tree_signature(jax: Any, value: Any) -> tuple[tuple[tuple[int, ...], str], ...]:
    return tuple((tuple(np.shape(leaf)), str(np.dtype(leaf.dtype))) for leaf in jax.tree_util.tree_leaves(value))


def _tree_numeric_report(jax: Any, value: Any) -> dict[str, Any]:
    squared_norm = 0.0
    maximum = 0.0
    element_count = 0
    finite = True
    nonzero_leaf_count = 0
    leaves = jax.tree_util.tree_leaves(value)
    for leaf in leaves:
        array = np.asarray(leaf, dtype=np.float64)
        finite = finite and bool(np.all(np.isfinite(array)))
        squared_norm += float(np.sum(array * array, dtype=np.float64))
        maximum = max(maximum, float(np.max(np.abs(array))) if array.size else 0.0)
        element_count += int(array.size)
        nonzero_leaf_count += int(bool(np.any(array != 0.0)))
    return {
        "all_finite": finite,
        "element_count": element_count,
        "leaf_count": len(leaves),
        "nonzero_leaf_count": nonzero_leaf_count,
        "l2_norm": float(np.sqrt(squared_norm)),
        "maximum_absolute_value": maximum,
    }


def _output_report(
    outputs: Mapping[str, Any],
    schema: Any,
) -> dict[str, Any]:
    expected = {field.name for field in schema.fields}
    if set(outputs) != expected:
        raise ValueError("successor GPU probe output registry drift")
    finite = True
    constraint_failures = []
    scalar_count = 0
    for field in schema.fields:
        value = np.asarray(outputs[field.name], dtype=np.float64)
        scalar_count += int(value.size)
        finite = finite and bool(np.all(np.isfinite(value)))
        if field.constraint == "nonnegative" and np.any(value < 0.0):
            constraint_failures.append(field.name)
        elif field.constraint == "unit_interval" and np.any((value < 0.0) | (value > 1.0)):
            constraint_failures.append(field.name)
    return {
        "field_count": len(outputs),
        "scalar_count": scalar_count,
        "all_finite": finite,
        "constraint_failures": constraint_failures,
    }


def _inactive_report(outputs: Mapping[str, Any], schema: Any) -> dict[str, Any]:
    failures = []
    checked = 0
    for field in schema.fields:
        if field.pft_axis is None:
            continue
        checked += 1
        inactive = np.take(
            np.asarray(outputs[field.name]),
            1,
            axis=field.pft_axis,
        )
        if np.any(inactive != 0.0):
            failures.append(field.name)
    return {
        "checked_field_count": checked,
        "failure_count": len(failures),
        "failures": failures,
    }


def _forward_comparison_report(
    eager: Mapping[str, Any],
    compiled: Mapping[str, Any],
) -> dict[str, Any]:
    passed = True
    maximum_absolute_error = 0.0
    maximum_tolerance_ratio = 0.0
    maximum_field = None
    maximum_reference_magnitude = 0.0
    global_maximum_absolute_error = 0.0
    for name in eager:
        left = np.asarray(eager[name], dtype=np.float64)
        right = np.asarray(compiled[name], dtype=np.float64)
        difference = np.abs(left - right)
        tolerance = FORWARD_ATOL + FORWARD_RTOL * np.maximum(
            np.abs(left),
            np.abs(right),
        )
        ratio = difference / tolerance
        index = np.unravel_index(int(np.argmax(ratio)), ratio.shape)
        field_ratio = float(ratio[index])
        global_maximum_absolute_error = max(
            global_maximum_absolute_error,
            float(np.max(difference)),
        )
        if field_ratio > maximum_tolerance_ratio:
            maximum_tolerance_ratio = field_ratio
            maximum_absolute_error = float(difference[index])
            maximum_field = name
            maximum_reference_magnitude = float(max(abs(left[index]), abs(right[index])))
        passed = passed and bool(np.all(difference <= tolerance))
    return {
        "passed": passed,
        "atol": FORWARD_ATOL,
        "rtol": FORWARD_RTOL,
        "maximum_tolerance_ratio": maximum_tolerance_ratio,
        "maximum_absolute_error_at_ratio": maximum_absolute_error,
        "maximum_ratio_field": maximum_field,
        "maximum_ratio_reference_magnitude": maximum_reference_magnitude,
        "global_maximum_absolute_error": global_maximum_absolute_error,
    }


def _load_extended_runtime(
    args: argparse.Namespace,
    *,
    backend_initializer: Any | None = None,
    initialize_probe_parameters: bool = True,
) -> dict[str, Any]:
    extra = {} if initialize_probe_parameters else {'initialize_probe_parameters': False}
    runtime = (
        _load_runtime(args, **extra)
        if backend_initializer is None
        else _load_runtime(args, backend_initializer=backend_initializer, **extra)
    )
    payload = json.loads(args.fast_statistics.read_text(encoding="utf-8"))
    arrays_path = args.fast_statistics.parent / str(payload["statistics_npz"])
    _verify(
        args.fast_statistics.resolve(),
        FAST_STATISTICS_JSON_SHA256,
        "successor_fast_owner_statistics_json",
    )
    _verify(
        arrays_path.resolve(),
        FAST_STATISTICS_NPZ_SHA256,
        "successor_fast_owner_statistics_npz",
    )

    from research.daily_coarse_graining.daily_operator.fast_owner_reconstruction import (
        build_successor_fast_owner_materializer,
        load_fast_owner_reconstruction_contract,
    )
    from research.daily_coarse_graining.daily_operator.successor_architecture import (
        build_successor_token_layout,
    )
    from research.daily_coarse_graining.daily_operator.successor_contract import (
        load_successor_candidate_contract,
    )
    from research.daily_coarse_graining.daily_operator.successor_fast_owner_decoder import (
        build_successor_fast_owner_schema,
    )
    from research.daily_coarse_graining.daily_operator.successor_fast_owner_statistics import (
        load_successor_fast_owner_statistics,
    )
    from research.daily_coarse_graining.daily_operator.successor_fast_owner_training import (
        build_successor_fast_owner_targets,
    )

    owner = load_fast_owner_reconstruction_contract(
        markov_contract=runtime["contract"],
    )
    if owner.contract_sha256 != EXPECTED["fast_owner_contract"]:
        raise ValueError("fast-owner contract identity drift")
    if len(owner.leaves) != 160:
        raise ValueError("fast-owner leaf count drift")
    schema = build_successor_fast_owner_schema(
        runtime["process_statistics"].schema,
        markov_contract=runtime["contract"],
        owner_contract=owner,
    )
    successor_contract = load_successor_candidate_contract()
    if successor_contract.sha256 != EXPECTED["successor_contract"]:
        raise ValueError("successor contract identity drift")
    statistics = load_successor_fast_owner_statistics(
        args.fast_statistics,
        index=runtime["index"],
        adapter=runtime["adapter"],
        base=runtime["process_statistics"],
        successor_contract=successor_contract,
        owner_contract=owner,
        schema=schema,
    )
    if statistics.sample_count != 8_591_565 or statistics.source_shard_count != 23_540:
        raise ValueError("fast-owner statistics coverage drift")
    if len(statistics.outputs) != 79:
        raise ValueError("fast-owner statistics field count drift")
    normalization = statistics.output_normalization(runtime["process_statistics"])
    scales = statistics.combined_loss_scales(runtime["process_statistics"])
    if len(schema.fields) != 181 or len(schema.labels) != 48 or len(normalization.mean) != 181 or len(scales) != 180:
        raise ValueError("complete successor decoder registry drift")
    layout = build_successor_token_layout(
        runtime["adapter"].input_assembler,
        schema,
        successor_contract,
    )
    example = runtime["examples"][ANCHOR_ROWS[0]]
    assembled = runtime["adapter"].input_assembler.assemble(example.inputs)
    targets = build_successor_fast_owner_targets(
        example,
        schema=schema,
        markov_contract=runtime["contract"],
        owner_contract=owner,
    )
    materializer = build_successor_fast_owner_materializer(
        runtime["contract"],
        forcing_layout=runtime["forcing_layout"],
    )
    return runtime | {
        "fast_statistics": statistics,
        "successor_contract": successor_contract,
        "owner_contract": owner,
        "schema": schema,
        "layout": layout,
        "output_normalization": normalization,
        "loss_scales": scales,
        "example": example,
        "probe_assembled": assembled,
        "targets": targets,
        "materializer": materializer,
    }


def _head(runtime: Mapping[str, Any], arm: str) -> Any:
    from research.daily_coarse_graining.daily_operator.successor_process_head import (
        SuccessorProcessHead,
    )

    return SuccessorProcessHead(
        arm=arm,
        contract=runtime["contract"],
        input_assembler=runtime["adapter"].input_assembler,
        schema=runtime["schema"],
        token_layout=runtime["layout"],
        input_normalization=runtime["input_normalization"],
        output_normalization=runtime["output_normalization"],
        forcing_layout=runtime["forcing_layout"],
        forcing_indices=runtime["forcing_indices"],
    )


def _probe_arm(
    runtime: Mapping[str, Any],
    arm: str,
    parameters: Any,
    retained_tail: Any,
) -> dict[str, Any]:
    from research.daily_coarse_graining.daily_operator.conservative_update import (
        validate_conservative_result,
    )
    from research.daily_coarse_graining.daily_operator.model import bind_daily_operator
    from research.daily_coarse_graining.daily_operator.successor_fast_owner_training import (
        successor_fast_owner_loss,
    )

    jax = runtime["jax"]
    head = _head(runtime, arm)
    inputs = runtime["probe_assembled"]

    def forward(values: Any, item: Any) -> Mapping[str, Any]:
        context = head.assemble_context(values, item)
        return head.named_outputs(values, context, item)

    eager_started = time.perf_counter()
    eager = forward(parameters, inputs)
    jax.block_until_ready(eager)
    eager_seconds = time.perf_counter() - eager_started
    eager_report = _output_report(eager, runtime["schema"])

    compiled_forward = jax.jit(forward)
    compile_started = time.perf_counter()
    compiled = compiled_forward(parameters, inputs)
    jax.block_until_ready(compiled)
    compile_seconds = time.perf_counter() - compile_started
    replay_started = time.perf_counter()
    replay = compiled_forward(parameters, inputs)
    jax.block_until_ready(replay)
    replay_seconds = time.perf_counter() - replay_started
    compiled_report = _output_report(compiled, runtime["schema"])
    comparison_report = _forward_comparison_report(eager, compiled)

    def loss_function(values: Any) -> tuple[Any, Any]:
        outputs = forward(values, inputs)
        loss = successor_fast_owner_loss(
            outputs,
            runtime["targets"],
            inputs=inputs,
            schema=runtime["schema"],
            scales=runtime["loss_scales"],
        )
        return loss.total, (
            loss.label_loss,
            loss.label_count,
            loss.defined_bce,
            loss.defined_count,
        )

    gradient_started = time.perf_counter()
    (loss_value, loss_aux), gradient = jax.jit(jax.value_and_grad(loss_function, has_aux=True))(parameters)
    jax.block_until_ready((loss_value, gradient))
    gradient_seconds = time.perf_counter() - gradient_started
    gradient_report = _tree_numeric_report(jax, gradient)
    label_loss, label_count, defined_bce, defined_count = loss_aux

    inactive_inputs = inputs._replace(
        pft=inputs.pft._replace(
            active_pft_mask=np.asarray([True, False], dtype=np.bool_),
        )
    )
    inactive_outputs = compiled_forward(parameters, inactive_inputs)
    jax.block_until_ready(inactive_outputs)
    inactive_report = _inactive_report(inactive_outputs, runtime["schema"])

    transition = bind_daily_operator(
        input_assembler=runtime["adapter"].input_assembler,
        process_head=head,
        retained_tail=retained_tail,
    )
    transition_started = time.perf_counter()
    result = jax.jit(transition)(parameters, runtime["example"].inputs)
    jax.block_until_ready(result)
    transition_seconds = time.perf_counter() - transition_started
    validate_conservative_result(result.fast_update)
    water_residual = float(result.fast_update.water.budget_residual)
    carbon_residual = float(result.fast_update.carbon.budget_residual)
    state_report = _tree_numeric_report(jax, result.next_state)

    loss_number = float(loss_value)
    passed = (
        eager_report["all_finite"]
        and not eager_report["constraint_failures"]
        and compiled_report["all_finite"]
        and not compiled_report["constraint_failures"]
        and np.isfinite(loss_number)
        and loss_number > 0.0
        and gradient_report["all_finite"]
        and np.isfinite(gradient_report["l2_norm"])
        and gradient_report["l2_norm"] > 0.0
        and inactive_report["failure_count"] == 0
        and abs(water_residual) <= 2.0e-12
        and abs(carbon_residual) <= 2.0e-12
        and state_report["all_finite"]
        and comparison_report["passed"]
    )
    return {
        "status": "passed" if passed else "failed",
        "arm": arm,
        "forward": {
            "eager_seconds": eager_seconds,
            "jit_compile_and_first_seconds": compile_seconds,
            "jit_replay_seconds": replay_seconds,
            "eager_jit_comparison": comparison_report,
            "eager": eager_report,
            "compiled": compiled_report,
        },
        "loss": {
            "total": loss_number,
            "label_loss": label_loss,
            "label_count": label_count,
            "defined_bce": defined_bce,
            "defined_count": defined_count,
        },
        "gradient": gradient_report | {"compile_and_run_seconds": gradient_seconds},
        "inactive_pft": inactive_report,
        "canonical_transition": {
            "compile_and_run_seconds": transition_seconds,
            "water_budget_residual": water_residual,
            "carbon_budget_residual": carbon_residual,
            "next_state": state_report,
            "named_output_count": len(result.prediction.named_outputs),
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"probe output already exists: {args.output}")
    runtime = _load_extended_runtime(args)
    jax = runtime["jax"]
    from jax_orchidee.driver import orchestration as teacher
    from research.daily_coarse_graining.daily_operator.successor_architecture import (
        SOURCE_ROUTED_ATTENTION,
        SOURCE_ROUTED_GRAPH_CONTROL,
        initialize_successor_parameters,
        successor_parameter_count,
    )

    parameters = {
        arm: initialize_successor_parameters(
            jax.random.PRNGKey(PROBE_SEED),
            schema=runtime["schema"],
            layout=runtime["layout"],
            inputs=runtime["probe_assembled"],
            arm=arm,
        )
        for arm in (SOURCE_ROUTED_ATTENTION, SOURCE_ROUTED_GRAPH_CONTROL)
    }
    counts = {arm: successor_parameter_count(value) for arm, value in parameters.items()}
    if set(counts.values()) != {EXPECTED_PARAMETER_COUNT}:
        raise ValueError(f"successor parameter count drift: {counts}")
    signatures = {arm: _tree_signature(jax, value) for arm, value in parameters.items()}
    if len(set(signatures.values())) != 1:
        raise ValueError("candidate/control parameter-tree drift")

    paths = runtime["context_paths"]
    teacher_context = teacher.prepare_paper_1961_driver_context(
        paths["config"],
        used_run_def_path=paths["run_def"],
        reference_run_dir=paths["reference_run_dir"],
    )
    sample = runtime["ordinary_parent"].sample(ANCHOR_ROWS[0])
    retained_tail = _retained_adapter(
        runtime,
        teacher_context,
        sample,
        lifecycle_start_tstep=48,
        result_day_index=2,
        fast_owner_materializer=runtime["materializer"],
    )
    arms = {
        arm: _probe_arm(runtime, arm, parameters[arm], retained_tail)
        for arm in (SOURCE_ROUTED_ATTENTION, SOURCE_ROUTED_GRAPH_CONTROL)
    }
    passed = all(value["status"] == "passed" for value in arms.values())
    report = {
        "schema_version": "successor_fast_owner_real_v6_gpu_probe_v1",
        "status": "passed" if passed else "failed",
        "checkpoint_written": False,
        "training_updates": 0,
        "rollout_days": 0,
        "promotion_authorized": False,
        "teacher_data_regenerated": False,
        "historical_v5_used": False,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in runtime["devices"]],
        "seed": PROBE_SEED,
        "identity": {
            "dataset_manifest_sha256": EXPECTED["dataset_manifest"],
            "process_statistics_json_sha256": EXPECTED["process_statistics_json"],
            "fast_owner_statistics_json_sha256": FAST_STATISTICS_JSON_SHA256,
            "fast_owner_statistics_npz_sha256": FAST_STATISTICS_NPZ_SHA256,
            "successor_contract_sha256": EXPECTED["successor_contract"],
            "fast_owner_contract_sha256": EXPECTED["fast_owner_contract"],
        },
        "data": {
            "landpoint_id": FROZEN_LANDPOINT,
            "year": 1961,
            "row": ANCHOR_ROWS[0],
            "day_index": int(np.asarray(sample["day_index"])),
            "spatial_split": "train",
            "temporal_split": "train",
        },
        "matched_boundary": {
            "field_count": len(runtime["schema"].fields),
            "loss_label_count": len(runtime["schema"].labels),
            "value_loss_scale_count": len(runtime["loss_scales"]),
            "parameter_count_per_arm": counts,
            "parameter_tree_identical": True,
            "binding": "bind_daily_operator(canonical_assembler, candidate_or_control, canonical_retained_tail_adapter)",
            "native_forcing_only": True,
            "reconstructs_48_forcing_records": False,
            "executes_48_step_state_scan": False,
        },
        "arms": arms,
        "next_gate": "bounded_small_fit_requires_separate_approval" if passed else "repair_gpu_probe",
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
