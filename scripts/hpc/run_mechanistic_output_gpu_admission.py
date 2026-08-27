"""Run a zero-update real-v6 GPU admission for the mechanistic-output head."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.hpc.run_strong_baseline_gpu_admission import _masked_state_report, _memory_report
from scripts.hpc.run_successor_fast_owner_gpu_probe import (
    PROBE_SEED,
    _load_extended_runtime,
    _tree_numeric_report,
)
from scripts.hpc.run_successor_process_head_canonical_gate import (
    ANCHOR_ROWS,
    FROZEN_LANDPOINT,
    _atomic_json,
    _retained_adapter,
)

FORWARD_ATOL = 1.0e-5
FORWARD_RTOL = 2.0e-6


def _masked_output_report(
    outputs: Mapping[str, Any],
    targets: Mapping[str, Any],
) -> dict[str, Any]:
    field_count = 0
    scalar_count = 0
    defined_count = 0
    undefined_count = 0
    nonfinite_defined_count = 0
    for name, value in outputs.items():
        array = np.asarray(value, dtype=np.float64)
        defined = np.asarray(targets[name].defined, dtype=np.bool_)
        if array.shape != defined.shape:
            raise ValueError(f"mechanistic output shape drift for {name}")
        field_count += 1
        scalar_count += int(array.size)
        defined_count += int(np.count_nonzero(defined))
        undefined_count += int(array.size - np.count_nonzero(defined))
        nonfinite_defined_count += int(np.count_nonzero(~np.isfinite(array[defined])))
    return {
        "field_count": field_count,
        "scalar_count": scalar_count,
        "defined_count": defined_count,
        "undefined_count": undefined_count,
        "nonfinite_defined_count": nonfinite_defined_count,
        "all_declared_defined_finite": nonfinite_defined_count == 0,
    }


def _masked_forward_comparison(
    eager: Mapping[str, Any],
    compiled: Mapping[str, Any],
    targets: Mapping[str, Any],
    *,
    atol: float = FORWARD_ATOL,
    rtol: float = FORWARD_RTOL,
) -> dict[str, Any]:
    maximum_ratio = 0.0
    maximum_error = 0.0
    maximum_field = None
    passed = True
    for name in eager:
        left = np.asarray(eager[name], dtype=np.float64)
        right = np.asarray(compiled[name], dtype=np.float64)
        selected = np.asarray(targets[name].defined, dtype=np.bool_)
        left = left[selected]
        right = right[selected]
        if not left.size:
            continue
        difference = np.abs(left - right)
        tolerance = atol + rtol * np.maximum(np.abs(left), np.abs(right))
        ratio = difference / tolerance
        field_ratio = float(np.max(ratio))
        if field_ratio > maximum_ratio:
            maximum_ratio = field_ratio
            maximum_error = float(np.max(difference))
            maximum_field = name
        passed = passed and bool(np.all(difference <= tolerance))
    return {
        "passed": passed,
        "atol": atol,
        "rtol": rtol,
        "maximum_tolerance_ratio": maximum_ratio,
        "maximum_absolute_error": maximum_error,
        "maximum_ratio_field": maximum_field,
    }


def _float_mapping(values: Mapping[str, Any]) -> dict[str, float]:
    return {name: float(np.asarray(value)) for name, value in values.items()}


def run(args: argparse.Namespace) -> dict[str, Any]:
    runtime = _load_extended_runtime(args)
    jax = runtime["jax"]
    device = runtime["devices"][0]

    from jax_orchidee.driver import orchestration as teacher
    from research.daily_coarse_graining.daily_operator.conservative_update import (
        apply_conservative_update,
        validate_conservative_result,
    )
    from research.daily_coarse_graining.daily_operator.mechanistic_output_contract import (
        build_mechanistic_learned_schema,
        load_mechanistic_output_contract,
    )
    from research.daily_coarse_graining.daily_operator.mechanistic_output_loss import (
        masked_mechanistic_output_loss,
    )
    from research.daily_coarse_graining.daily_operator.mechanistic_process_head import (
        MechanisticStrongBaselineProcessHead,
        build_mechanistic_output_layout,
    )
    from research.daily_coarse_graining.daily_operator.model import bind_daily_operator
    from research.daily_coarse_graining.daily_operator.strong_baseline_architecture import (
        initialize_strong_baseline_parameters,
        strong_baseline_parameter_count,
    )
    from research.daily_coarse_graining.daily_operator.strong_baseline_contract import (
        load_strong_baseline_contract,
    )

    architecture = load_strong_baseline_contract()
    mechanism = load_mechanistic_output_contract()
    learned_schema = build_mechanistic_learned_schema(runtime["schema"], mechanism)
    assembled = runtime["probe_assembled"]
    layout = build_mechanistic_output_layout(
        adapter=runtime["adapter"],
        complete_schema=runtime["schema"],
        learned_schema=learned_schema,
        owner_contract=runtime["owner_contract"],
        architecture_contract=architecture,
        mechanism_contract=mechanism,
        inputs=assembled,
    )
    memory_before_parameters = _memory_report(device)
    parameters = initialize_strong_baseline_parameters(
        jax.random.PRNGKey(PROBE_SEED),
        schema=learned_schema,
        layout=layout,
        inputs=assembled,
        contract=architecture,
    )
    parameter_count = strong_baseline_parameter_count(parameters)
    if parameter_count != 49_743_001:
        raise ValueError(f"mechanistic parameter count drift: {parameter_count}")
    head = MechanisticStrongBaselineProcessHead(
        contract=runtime["contract"],
        architecture_contract=architecture,
        mechanism_contract=mechanism,
        input_assembler=runtime["adapter"].input_assembler,
        complete_schema=runtime["schema"],
        learned_schema=learned_schema,
        layout=layout,
        input_normalization=runtime["input_normalization"],
        output_normalization=runtime["output_normalization"],
        forcing_layout=runtime["forcing_layout"],
        forcing_indices=runtime["forcing_indices"],
        message_gate_scale=0.0,
    )

    def forward(values: Any, inputs: Any) -> Mapping[str, Any]:
        context = head.assemble_context(values, inputs)
        primitive = head(values, context, inputs)
        update = apply_conservative_update(inputs, primitive)
        return head.materialize_post_update(
            values,
            context,
            inputs,
            primitive,
            update,
        ).named_outputs

    eager_started = time.perf_counter()
    eager = forward(parameters, assembled)
    jax.block_until_ready(eager)
    eager_seconds = time.perf_counter() - eager_started
    eager_report = _masked_output_report(eager, runtime["targets"])

    compiled_forward = jax.jit(forward)
    compile_started = time.perf_counter()
    compiled = compiled_forward(parameters, assembled)
    jax.block_until_ready(compiled)
    compile_seconds = time.perf_counter() - compile_started
    replay_started = time.perf_counter()
    replay = compiled_forward(parameters, assembled)
    jax.block_until_ready(replay)
    replay_seconds = time.perf_counter() - replay_started
    compiled_report = _masked_output_report(compiled, runtime["targets"])
    comparison = _masked_forward_comparison(
        eager,
        compiled,
        runtime["targets"],
    )

    def loss_function(values: Any) -> tuple[Any, Any]:
        context = head.assemble_context(values, assembled)
        primitive = head(values, context, assembled)
        update = apply_conservative_update(assembled, primitive)
        complete = head.materialize_post_update(
            values,
            context,
            assembled,
            primitive,
            update,
        )
        loss = masked_mechanistic_output_loss(
            complete.named_outputs,
            runtime["targets"],
            inputs=assembled,
            complete_schema=runtime["schema"],
            learned_schema=learned_schema,
            scales=runtime["loss_scales"],
            architecture_contract=architecture,
            mechanism_contract=mechanism,
            auxiliary_outputs=complete.auxiliary_outputs,
        )
        return loss.total, (
            loss.expert_mse,
            loss.expert_label_count,
            loss.exact_diagnostic_mse,
            loss.exact_diagnostic_count,
        )

    gradient_started = time.perf_counter()
    (loss_value, loss_aux), gradient = jax.jit(jax.value_and_grad(loss_function, has_aux=True))(parameters)
    jax.block_until_ready((loss_value, gradient))
    gradient_seconds = time.perf_counter() - gradient_started
    gradient_report = _tree_numeric_report(jax, gradient)
    expert_mse, expert_label_count, exact_mse, exact_count = loss_aux

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
    state_report = _masked_state_report(result.next_state, result.next_state_defined)

    loss_number = float(loss_value)
    passed = (
        eager_report["all_declared_defined_finite"]
        and compiled_report["all_declared_defined_finite"]
        and comparison["passed"]
        and np.isfinite(loss_number)
        and loss_number > 0.0
        and gradient_report["all_finite"]
        and gradient_report["l2_norm"] > 0.0
        and state_report["all_declared_defined_finite"]
        and abs(water_residual) <= 2.0e-12
        and abs(carbon_residual) <= 2.0e-12
        and len(result.prediction.named_outputs) == 181
    )
    report = {
        "schema_version": "mechanistic_output_real_v6_gpu_admission_v1",
        "status": "passed" if passed else "failed",
        "policy": {
            "training_updates": 0,
            "checkpoint_written": False,
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "sealed_test_read": False,
            "forcing_records_reconstructed": 0,
            "subdaily_state_scan_steps": 0,
        },
        "backend": jax.default_backend(),
        "devices": [str(value) for value in runtime["devices"]],
        "data": {
            "landpoint_id": FROZEN_LANDPOINT,
            "year": 1961,
            "row": ANCHOR_ROWS[0],
            "day_index": int(np.asarray(sample["day_index"])),
            "spatial_split": "train",
            "temporal_split": "train",
        },
        "model": {
            "parameter_count": parameter_count,
            "learned_field_count": len(learned_schema.fields),
            "complete_field_count": len(runtime["schema"].fields),
            "learned_scalar_count": sum(field.scalar_count for field in layout.ledger),
            "complete_scalar_count": sum(int(np.prod(field.shape)) for field in runtime["schema"].fields),
            "message_gate_scale": 0.0,
        },
        "forward": {
            "eager_seconds": eager_seconds,
            "jit_compile_and_first_seconds": compile_seconds,
            "jit_replay_seconds": replay_seconds,
            "eager": eager_report,
            "compiled": compiled_report,
            "comparison": comparison,
        },
        "loss": {
            "total": loss_number,
            "expert_mse": _float_mapping(expert_mse),
            "expert_label_count": _float_mapping(expert_label_count),
            "exact_diagnostic_mse": _float_mapping(exact_mse),
            "exact_diagnostic_count": _float_mapping(exact_count),
        },
        "gradient": gradient_report | {"compile_and_run_seconds": gradient_seconds},
        "canonical_transition": {
            "compile_and_run_seconds": transition_seconds,
            "water_budget_residual": water_residual,
            "carbon_budget_residual": carbon_residual,
            "next_state": state_report,
            "named_output_count": len(result.prediction.named_outputs),
        },
        "device_memory": {
            "before_parameters": memory_before_parameters,
            "after_all_gates": _memory_report(device),
        },
        "next_gate": "one_point_year_memorization" if passed else "repair_real_v6_admission",
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
