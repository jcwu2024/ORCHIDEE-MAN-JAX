"""Run the zero-update real-v6 GPU admission for the approved strong baseline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.hpc.run_successor_fast_owner_gpu_probe import (
    PROBE_SEED,
    _load_extended_runtime,
    _output_report,
    _tree_numeric_report,
)
from scripts.hpc.run_successor_process_head_canonical_gate import (
    ANCHOR_ROWS,
    EXPECTED,
    FROZEN_LANDPOINT,
    _atomic_json,
    _retained_adapter,
)

FORWARD_ATOL = 1.0e-5
FORWARD_RTOL = 2.0e-6


def _forward_comparison_report(
    eager: Mapping[str, Any],
    compiled: Mapping[str, Any],
) -> dict[str, Any]:
    maximum_ratio = 0.0
    maximum_error = 0.0
    maximum_field = None
    passed = True
    for name in eager:
        left = np.asarray(eager[name], dtype=np.float64)
        right = np.asarray(compiled[name], dtype=np.float64)
        difference = np.abs(left - right)
        tolerance = FORWARD_ATOL + FORWARD_RTOL * np.maximum(np.abs(left), np.abs(right))
        ratio = difference / tolerance
        field_ratio = float(np.max(ratio))
        if field_ratio > maximum_ratio:
            maximum_ratio = field_ratio
            maximum_error = float(np.max(difference))
            maximum_field = name
        passed = passed and bool(np.all(difference <= tolerance))
    return {
        "passed": passed,
        "atol": FORWARD_ATOL,
        "rtol": FORWARD_RTOL,
        "maximum_tolerance_ratio": maximum_ratio,
        "maximum_absolute_error": maximum_error,
        "maximum_ratio_field": maximum_field,
    }


def _masked_state_report(state: Any, defined: Any) -> dict[str, Any]:
    element_count = 0
    defined_count = 0
    undefined_count = 0
    finite = True
    squared_norm = 0.0
    maximum = 0.0
    for name in state._fields:
        values = np.asarray(getattr(state, name), dtype=np.float64)
        mask = np.asarray(getattr(defined, name), dtype=np.bool_)
        if values.shape != mask.shape:
            raise ValueError(f"next-state defined-mask shape drift for {name}")
        selected = values[mask]
        element_count += int(values.size)
        defined_count += int(selected.size)
        undefined_count += int(values.size - selected.size)
        finite = finite and bool(np.all(np.isfinite(selected)))
        squared_norm += float(np.sum(selected * selected, dtype=np.float64))
        maximum = max(maximum, float(np.max(np.abs(selected))) if selected.size else 0.0)
    return {
        "all_declared_defined_finite": finite,
        "element_count": element_count,
        "defined_count": defined_count,
        "undefined_count": undefined_count,
        "defined_l2_norm": float(np.sqrt(squared_norm)),
        "maximum_absolute_defined_value": maximum,
    }


def _memory_report(device: Any) -> Mapping[str, Any]:
    stats = device.memory_stats() or {}
    return {
        str(name): int(value) if isinstance(value, (int, np.integer)) else value
        for name, value in stats.items()
        if name in {"bytes_in_use", "peak_bytes_in_use", "bytes_limit", "num_allocs"}
    }


def _head(runtime: Mapping[str, Any]) -> Any:
    from research.daily_coarse_graining.daily_operator.strong_baseline_process_head import (
        StrongBaselineProcessHead,
    )

    return StrongBaselineProcessHead(
        contract=runtime["contract"],
        architecture_contract=runtime["architecture_contract"],
        input_assembler=runtime["adapter"].input_assembler,
        schema=runtime["schema"],
        layout=runtime["strong_layout"],
        input_normalization=runtime["input_normalization"],
        output_normalization=runtime["output_normalization"],
        forcing_layout=runtime["forcing_layout"],
        forcing_indices=runtime["forcing_indices"],
        message_gate_scale=0.0,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    runtime = _load_extended_runtime(args)
    jax = runtime["jax"]
    device = runtime["devices"][0]

    from jax_orchidee.driver import orchestration as teacher
    from research.daily_coarse_graining.daily_operator.conservative_update import (
        validate_conservative_result,
    )
    from research.daily_coarse_graining.daily_operator.model import bind_daily_operator
    from research.daily_coarse_graining.daily_operator.strong_baseline_architecture import (
        build_strong_baseline_layout,
        initialize_strong_baseline_parameters,
        strong_baseline_parameter_count,
    )
    from research.daily_coarse_graining.daily_operator.strong_baseline_contract import (
        load_strong_baseline_contract,
    )
    from research.daily_coarse_graining.daily_operator.strong_baseline_loss import (
        masked_equal_expert_label_macro_mse,
    )

    architecture_contract = load_strong_baseline_contract()
    assembled = runtime["probe_assembled"]
    strong_layout = build_strong_baseline_layout(
        adapter=runtime["adapter"],
        schema=runtime["schema"],
        owner_contract=runtime["owner_contract"],
        contract=architecture_contract,
        inputs=assembled,
    )
    runtime = dict(runtime) | {
        "architecture_contract": architecture_contract,
        "strong_layout": strong_layout,
    }
    memory_before_parameters = _memory_report(device)
    parameters = initialize_strong_baseline_parameters(
        jax.random.PRNGKey(PROBE_SEED),
        schema=runtime["schema"],
        layout=strong_layout,
        inputs=assembled,
        contract=architecture_contract,
    )
    parameter_count = strong_baseline_parameter_count(parameters)
    if parameter_count != 49_750_681:
        raise ValueError(f"strong-baseline parameter count drift: {parameter_count}")
    head = _head(runtime)

    def forward(values: Any, inputs: Any) -> Mapping[str, Any]:
        context = head.assemble_context(values, inputs)
        return head.named_outputs(values, context, inputs)

    eager_started = time.perf_counter()
    eager = forward(parameters, assembled)
    jax.block_until_ready(eager)
    eager_seconds = time.perf_counter() - eager_started
    eager_report = _output_report(eager, runtime["schema"])

    compiled_forward = jax.jit(forward)
    compiled_started = time.perf_counter()
    compiled = compiled_forward(parameters, assembled)
    jax.block_until_ready(compiled)
    compile_seconds = time.perf_counter() - compiled_started
    replay_started = time.perf_counter()
    replay = compiled_forward(parameters, assembled)
    jax.block_until_ready(replay)
    replay_seconds = time.perf_counter() - replay_started

    def loss_function(values: Any) -> tuple[Any, Any]:
        outputs = forward(values, assembled)
        loss = masked_equal_expert_label_macro_mse(
            outputs,
            runtime["targets"],
            inputs=assembled,
            schema=runtime["schema"],
            scales=runtime["loss_scales"],
            contract=architecture_contract,
        )
        return loss.total, (
            loss.expert_mse,
            loss.expert_label_count,
            loss.label_mse,
            loss.label_count,
        )

    gradient_started = time.perf_counter()
    (loss_value, loss_aux), gradient = jax.jit(
        jax.value_and_grad(loss_function, has_aux=True)
    )(parameters)
    jax.block_until_ready((loss_value, gradient))
    gradient_seconds = time.perf_counter() - gradient_started
    gradient_report = _tree_numeric_report(jax, gradient)
    expert_mse, expert_label_count, label_mse, label_count = loss_aux

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

    comparison = _forward_comparison_report(eager, compiled)
    compiled_report = _output_report(compiled, runtime["schema"])
    loss_number = float(loss_value)
    passed = (
        eager_report["all_finite"]
        and not eager_report["constraint_failures"]
        and compiled_report["all_finite"]
        and not compiled_report["constraint_failures"]
        and comparison["passed"]
        and np.isfinite(loss_number)
        and loss_number > 0.0
        and gradient_report["all_finite"]
        and gradient_report["l2_norm"] > 0.0
        and state_report["all_declared_defined_finite"]
        and abs(water_residual) <= 2.0e-12
        and abs(carbon_residual) <= 2.0e-12
    )
    report = {
        "schema_version": "strong_baseline_real_v6_gpu_admission_v1",
        "status": "passed" if passed else "failed",
        "training_updates": 0,
        "checkpoint_written": False,
        "teacher_data_regenerated": False,
        "historical_v5_used": False,
        "message_gate_scale": 0.0,
        "backend": jax.default_backend(),
        "devices": [str(value) for value in runtime["devices"]],
        "identity": {
            "dataset_manifest_sha256": EXPECTED["dataset_manifest"],
            "architecture_contract_sha256": architecture_contract.sha256,
        },
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
            "field_count": len(runtime["schema"].fields),
            "label_count": len(runtime["schema"].labels),
            "scalar_output_count": sum(field.scalar_count for field in strong_layout.ledger),
        },
        "forward": {
            "eager_seconds": eager_seconds,
            "jit_compile_and_first_seconds": compile_seconds,
            "jit_replay_seconds": replay_seconds,
            "eager": eager_report,
            "compiled": compiled_report,
            "eager_jit_comparison": comparison,
        },
        "loss": {
            "total": loss_number,
            "expert_mse": expert_mse,
            "expert_label_count": expert_label_count,
            "label_mse": label_mse,
            "label_count": label_count,
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
            "before_strong_parameters": memory_before_parameters,
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
