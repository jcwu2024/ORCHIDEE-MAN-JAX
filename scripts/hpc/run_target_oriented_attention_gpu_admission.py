"""Run a zero-update real-v6 GPU admission for target-oriented Attention."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.hpc.run_mechanistic_output_gpu_admission import (
    _float_mapping,
    _masked_forward_comparison,
    _masked_output_report,
)
from scripts.hpc.run_strong_baseline_gpu_admission import (
    _masked_state_report,
    _memory_report,
)
from scripts.hpc.run_successor_fast_owner_gpu_probe import (
    PROBE_SEED,
    _load_extended_runtime,
    _tree_numeric_report,
)
from scripts.hpc.run_successor_process_head_canonical_gate import (
    ANCHOR_ROWS,
    EXPECTED,
    FROZEN_LANDPOINT,
    _atomic_json,
    _retained_adapter,
)

PARAMETER_COUNT = 83_755_173
ARCHITECTURE_MANIFEST = (
    Path(__file__).resolve().parents[2] / "manifests" / "coarse_graining" / "target_oriented_attention_v1.json"
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _head(runtime: Mapping[str, Any], target_contract: Any) -> Any:
    from research.daily_coarse_graining.daily_operator.target_oriented_process_head import (
        MechanisticTargetOrientedAttentionProcessHead,
    )

    return MechanisticTargetOrientedAttentionProcessHead(
        contract=runtime["contract"],
        architecture_contract=runtime["architecture_contract"],
        mechanism_contract=runtime["mechanism_contract"],
        input_assembler=runtime["adapter"].input_assembler,
        complete_schema=runtime["schema"],
        learned_schema=runtime["learned_schema"],
        layout=runtime["target_layout"],
        input_normalization=runtime["input_normalization"],
        output_normalization=runtime["output_normalization"],
        forcing_layout=runtime["forcing_layout"],
        forcing_indices=runtime["forcing_indices"],
        message_gate_scale=1.0,
        target_contract=target_contract,
    )


def _gradient_family_report(jax: Any, gradient: Mapping[str, Any]) -> dict[str, Any]:
    target = gradient["target_oriented"]
    return {
        "base_source_encoder": _tree_numeric_report(
            jax,
            {name: value for name, value in gradient.items() if name != "target_oriented"},
        ),
        "domain_refinement": _tree_numeric_report(jax, target["domain_blocks"]),
        "science_cross_attention": _tree_numeric_report(
            jax,
            target["science_cross_blocks"],
        ),
        "science_token_mixer": _tree_numeric_report(
            jax,
            target["science_mixer_blocks"],
        ),
        "primary_heads": _tree_numeric_report(
            jax,
            (target["primary_decoder"], target["primary_heads"]),
        ),
        "state_heads": _tree_numeric_report(
            jax,
            (target["state_decoder"], target["state_heads"]),
        ),
        "auxiliary_heads": _tree_numeric_report(
            jax,
            (
                target["auxiliary_science_projection"],
                target["auxiliary_bottleneck"],
                target["auxiliary_label_heads"],
            ),
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    runtime = _load_extended_runtime(args)
    jax = runtime["jax"]
    device = runtime["devices"][0]

    import jax.numpy as jnp

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
        build_mechanistic_output_layout,
    )
    from research.daily_coarse_graining.daily_operator.model import bind_daily_operator
    from research.daily_coarse_graining.daily_operator.strong_baseline_contract import (
        load_strong_baseline_contract,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_attention import (
        CANDIDATE_NAME,
        initialize_target_oriented_attention_parameters,
        target_oriented_attention_parameter_count,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_contract import (
        AUXILIARY_SUPERVISION,
        PRIMARY_SCIENCE_DIRECT,
        STATE_BUDGET_ESSENTIAL,
        load_target_oriented_contract,
        necessary_process_supervision_loss,
    )

    architecture = load_strong_baseline_contract()
    mechanism = load_mechanistic_output_contract()
    target_contract = load_target_oriented_contract()
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
    runtime = dict(runtime) | {
        "architecture_contract": architecture,
        "mechanism_contract": mechanism,
        "learned_schema": learned_schema,
        "target_layout": layout,
    }
    memory_before_parameters = _memory_report(device)
    parameters = initialize_target_oriented_attention_parameters(
        jax.random.PRNGKey(PROBE_SEED),
        schema=learned_schema,
        layout=layout,
        inputs=assembled,
        architecture_contract=architecture,
        target_contract=target_contract,
    )
    parameter_count = target_oriented_attention_parameter_count(parameters)
    if parameter_count != PARAMETER_COUNT:
        raise ValueError(f"target-oriented parameter count drift: {parameter_count}")
    head = _head(runtime, target_contract)

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
    comparison = _masked_forward_comparison(eager, compiled, runtime["targets"])

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
        process = masked_mechanistic_output_loss(
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
        necessary = necessary_process_supervision_loss(process, target_contract)
        primary_labels = target_contract.labels_by_category[PRIMARY_SCIENCE_DIRECT]
        primary_active = jnp.stack(tuple(process.label_count[label] > 0.0 for label in primary_labels))
        primary_values = jnp.stack(tuple(process.label_mse[label] for label in primary_labels))
        primary_count = jnp.sum(primary_active, dtype=jnp.float64)
        primary = jnp.where(
            primary_count > 0.0,
            jnp.sum(jnp.where(primary_active, primary_values, 0.0)) / primary_count,
            0.0,
        )
        total = primary + necessary.total
        return total, (
            primary,
            necessary.category_loss,
            necessary.category_active_label_count,
            process.exact_diagnostic_mse,
            process.exact_diagnostic_count,
        )

    gradient_started = time.perf_counter()
    (loss_value, loss_aux), gradient = jax.jit(jax.value_and_grad(loss_function, has_aux=True))(parameters)
    jax.block_until_ready((loss_value, gradient))
    gradient_seconds = time.perf_counter() - gradient_started
    gradient_report = _tree_numeric_report(jax, gradient)
    gradient_families = _gradient_family_report(jax, gradient)
    primary_loss, category_loss, category_count, exact_mse, exact_count = loss_aux

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

    required_gradient_families = (
        "base_source_encoder",
        "domain_refinement",
        "science_cross_attention",
        "science_token_mixer",
        "primary_heads",
        "state_heads",
        "auxiliary_heads",
    )
    family_gradients_pass = all(
        gradient_families[name]["all_finite"] and gradient_families[name]["l2_norm"] > 0.0
        for name in required_gradient_families
    )
    category_counts_pass = all(
        float(category_count[name]) > 0.0
        for name in (
            PRIMARY_SCIENCE_DIRECT,
            STATE_BUDGET_ESSENTIAL,
            AUXILIARY_SUPERVISION,
        )
    )
    loss_number = float(loss_value)
    passed = (
        eager_report["all_declared_defined_finite"]
        and compiled_report["all_declared_defined_finite"]
        and comparison["passed"]
        and np.isfinite(loss_number)
        and loss_number > 0.0
        and gradient_report["all_finite"]
        and gradient_report["l2_norm"] > 0.0
        and family_gradients_pass
        and category_counts_pass
        and state_report["all_declared_defined_finite"]
        and abs(water_residual) <= 2.0e-12
        and abs(carbon_residual) <= 2.0e-12
        and len(result.prediction.named_outputs) == 181
    )
    report = {
        "schema_version": "target_oriented_attention_real_v6_gpu_admission_v1",
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
        "identity": {
            "dataset_manifest_sha256": EXPECTED["dataset_manifest"],
            "strong_baseline_contract_sha256": architecture.sha256,
            "target_oriented_contract_sha256": target_contract.sha256,
            "architecture_manifest_sha256": _file_sha256(ARCHITECTURE_MANIFEST),
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
            "candidate": CANDIDATE_NAME,
            "parameter_count": parameter_count,
            "learned_field_count": len(learned_schema.fields),
            "complete_field_count": len(runtime["schema"].fields),
            "learned_scalar_count": sum(field.scalar_count for field in layout.ledger),
            "complete_scalar_count": sum(int(np.prod(field.shape)) for field in runtime["schema"].fields),
            "message_gate_scale": 1.0,
        },
        "forward": {
            "eager_seconds": eager_seconds,
            "jit_compile_and_first_seconds": compile_seconds,
            "jit_replay_seconds": replay_seconds,
            "eager": eager_report,
            "compiled": compiled_report,
            "comparison": comparison,
        },
        "admission_proxy_loss": {
            "total": loss_number,
            "primary_science_direct": float(primary_loss),
            "necessary_category_loss": _float_mapping(category_loss),
            "category_active_label_count": _float_mapping(category_count),
            "exact_diagnostic_mse": _float_mapping(exact_mse),
            "exact_diagnostic_count": _float_mapping(exact_count),
            "note": "one-day gradient admission proxy, not the 7/30-day training objective",
        },
        "gradient": gradient_report
        | {
            "compile_and_run_seconds": gradient_seconds,
            "families": gradient_families,
        },
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
        "next_gate": ("bounded_training_design" if passed else "repair_real_v6_gpu_admission"),
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
