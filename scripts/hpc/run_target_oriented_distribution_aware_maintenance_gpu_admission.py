"""Zero-update real-v6 admission for distribution-aware maintenance."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.daily_coarse_graining.daily_operator.target_oriented_checkpoint import (
    load_target_oriented_stage_checkpoint,
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
from scripts.hpc.run_target_oriented_attention_gpu_admission import (
    PARAMETER_COUNT as PARENT_PARAMETER_COUNT,
)
from scripts.hpc.run_target_oriented_attention_gpu_admission import (
    _head as _parent_head,
)
from scripts.hpc.run_target_oriented_attention_gpu_admission import (
    _masked_forward_comparison,
    _masked_output_report,
)
from scripts.hpc.run_target_oriented_source_shaped_maintenance_gpu_admission import (
    _nonmaintenance_parent_comparison,
)
from scripts.hpc.run_target_oriented_training import FROZEN_CLIP_THRESHOLD
from scripts.hpc.target_oriented_validation_prepared import (
    load_target_oriented_validation_prepared,
)

CANDIDATE_PARAMETER_COUNT = 93_504_933
ADAPTER_PARAMETER_COUNT = 9_750_529
CHECKPOINT_UPDATE = 1000
CHECKPOINT_FORWARD_ATOL = 1.0e-5
CHECKPOINT_FORWARD_RTOL = 5.0e-6


def _with_adapter(parameters: Mapping[str, Any], adapter: Mapping[str, Any]) -> Mapping[str, Any]:
    target = dict(parameters["target_oriented"])
    target["distribution_aware_maintenance"] = adapter
    return dict(parameters, target_oriented=target)


def run(args: argparse.Namespace) -> Mapping[str, Any]:
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
    from research.daily_coarse_graining.daily_operator.target_oriented_contract import (
        AUXILIARY_SUPERVISION,
        PRIMARY_SCIENCE_DIRECT,
        STATE_BUDGET_ESSENTIAL,
        load_target_oriented_contract,
        necessary_process_supervision_loss,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_distribution_aware_maintenance import (
        distribution_aware_maintenance_parameter_count,
        initialize_distribution_aware_maintenance_parameters,
        load_distribution_aware_maintenance_contract,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_process_head import (
        build_target_oriented_distribution_aware_maintenance_head,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_source_shaped_maintenance import (
        MAINTENANCE_FIELD,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_training import (
        load_target_oriented_state_anchor_policy,
        load_target_oriented_training_contract,
    )

    architecture = load_strong_baseline_contract()
    mechanism = load_mechanistic_output_contract()
    target_contract = load_target_oriented_contract()
    distribution_contract = load_distribution_aware_maintenance_contract(args.distribution_contract)
    training_contract = load_target_oriented_training_contract(args.training_contract)
    state_anchor_policy = load_target_oriented_state_anchor_policy()
    validation_prepared, panel = load_target_oriented_validation_prepared(
        args.validation_prepared_manifest,
        training_contract_sha256=training_contract.sha256,
        target_contract_sha256=target_contract.sha256,
        verify_artifacts=False,
    )
    checkpoint = load_target_oriented_stage_checkpoint(
        args.stage_checkpoint,
        expected_sha256=args.stage_checkpoint_sha256,
        completed_updates=CHECKPOINT_UPDATE,
        parameter_count=PARENT_PARAMETER_COUNT,
        clip_threshold=FROZEN_CLIP_THRESHOLD,
        training_contract_sha256=training_contract.sha256,
        target_contract_sha256=target_contract.sha256,
        state_anchor_policy_sha256=state_anchor_policy.sha256,
        validation_panel_sha256=panel["panel_sha256"],
    )
    parent_parameters = jax.tree_util.tree_map(jnp.asarray, checkpoint["parameters"])

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
    parent_head = _parent_head(runtime, target_contract)

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
    head = build_target_oriented_distribution_aware_maintenance_head(
        parent_head,
        retained_tail,
        distribution_contract=distribution_contract,
    )

    memory_before_parameters = _memory_report(device)
    parameters = initialize_distribution_aware_maintenance_parameters(
        parent_parameters,
        jax.random.PRNGKey(PROBE_SEED + 20260828),
        schema=learned_schema,
        target_contract=target_contract,
        distribution_contract=distribution_contract,
    )
    parameter_count = distribution_aware_maintenance_parameter_count(parameters)
    if parameter_count != CANDIDATE_PARAMETER_COUNT:
        raise ValueError(f"distribution-aware maintenance parameter count drift: {parameter_count}")
    adapter_parameters = parameters["target_oriented"]["distribution_aware_maintenance"]
    adapter_count = sum(int(np.asarray(leaf).size) for leaf in jax.tree_util.tree_leaves(adapter_parameters))
    if adapter_count != ADAPTER_PARAMETER_COUNT:
        raise ValueError(f"distribution-aware adapter parameter count drift: {adapter_count}")

    parent_context = parent_head.assemble_context(parent_parameters, assembled)
    parent_learned = parent_head.learned_outputs(parent_parameters, parent_context, assembled)
    candidate_context = head.assemble_context(parameters, assembled)
    candidate_learned = head.learned_outputs(parameters, candidate_context, assembled)
    jax.block_until_ready((parent_learned, candidate_learned))
    parent_comparison = _nonmaintenance_parent_comparison(
        parent_learned,
        candidate_learned,
        MAINTENANCE_FIELD,
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
        atol=CHECKPOINT_FORWARD_ATOL,
        rtol=CHECKPOINT_FORWARD_RTOL,
    )

    def adapter_loss_function(
        fixed_values: Mapping[str, Any],
        adapter_values: Mapping[str, Any],
    ) -> tuple[Any, Any]:
        values = _with_adapter(fixed_values, adapter_values)
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
        labels = target_contract.labels_by_category[PRIMARY_SCIENCE_DIRECT]
        active = jnp.stack(tuple(process.label_count[label] > 0.0 for label in labels))
        values_by_label = jnp.stack(tuple(process.label_mse[label] for label in labels))
        count = jnp.sum(active, dtype=jnp.float64)
        primary = jnp.where(
            count > 0.0,
            jnp.sum(jnp.where(active, values_by_label, 0.0)) / count,
            0.0,
        )
        return primary + necessary.total, (
            primary,
            necessary.category_loss,
            necessary.category_active_label_count,
        )

    gradient_started = time.perf_counter()
    gradient_call = jax.jit(jax.value_and_grad(adapter_loss_function, argnums=1, has_aux=True))
    (loss_value, loss_aux), adapter_gradient = gradient_call(parameters, adapter_parameters)
    jax.block_until_ready((loss_value, adapter_gradient))
    gradient_seconds = time.perf_counter() - gradient_started
    gradient_report = _tree_numeric_report(jax, adapter_gradient)
    component_gradients = {
        "context_projection": _tree_numeric_report(jax, adapter_gradient["context_projection"]),
        "attention_blocks": tuple(
            _tree_numeric_report(jax, block["attention"]) for block in adapter_gradient["blocks"]
        ),
        "ffn_blocks": tuple(_tree_numeric_report(jax, block["ffn"]) for block in adapter_gradient["blocks"]),
        "final_projection": _tree_numeric_report(jax, adapter_gradient["output"]),
    }
    primary_loss, category_loss, category_count = loss_aux

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

    category_counts_pass = all(
        float(category_count[name]) > 0.0
        for name in (
            PRIMARY_SCIENCE_DIRECT,
            STATE_BUDGET_ESSENTIAL,
            AUXILIARY_SUPERVISION,
        )
    )
    component_gradients_pass = (
        component_gradients["context_projection"]["all_finite"]
        and component_gradients["context_projection"]["l2_norm"] > 0.0
        and all(value["all_finite"] and value["l2_norm"] > 0.0 for value in component_gradients["attention_blocks"])
        and all(value["all_finite"] and value["l2_norm"] > 0.0 for value in component_gradients["ffn_blocks"])
        and component_gradients["final_projection"]["all_finite"]
        and component_gradients["final_projection"]["l2_norm"] > 0.0
    )
    output_parameters = adapter_parameters["output"]
    initialization_pass = bool(
        np.any(np.asarray(output_parameters["kernel"]) != 0.0) and np.all(np.asarray(output_parameters["bias"]) == 0.0)
    )
    loss_number = float(loss_value)
    passed = (
        parent_comparison["exactly_equal"]
        and initialization_pass
        and eager_report["all_declared_defined_finite"]
        and compiled_report["all_declared_defined_finite"]
        and comparison["passed"]
        and np.isfinite(loss_number)
        and loss_number > 0.0
        and gradient_report["all_finite"]
        and gradient_report["element_count"] == ADAPTER_PARAMETER_COUNT
        and gradient_report["l2_norm"] > 0.0
        and component_gradients_pass
        and category_counts_pass
        and state_report["all_declared_defined_finite"]
        and abs(water_residual) <= 2.0e-12
        and abs(carbon_residual) <= 2.0e-12
        and len(result.prediction.named_outputs) == 181
    )
    report = {
        "schema_version": "target_oriented_distribution_aware_maintenance_real_v6_gpu_admission_v1",
        "status": "passed" if passed else "failed",
        "policy": {
            "training_updates": 0,
            "optimizer_created": False,
            "checkpoint_written": False,
            "parent_parameters_trainable": False,
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "validation_values_evaluated": False,
            "sealed_test_read": False,
            "forcing_records_reconstructed": 0,
            "subdaily_state_scan_steps": 0,
        },
        "backend": jax.default_backend(),
        "devices": [str(value) for value in runtime["devices"]],
        "identity": {
            "dataset_manifest_sha256": EXPECTED["dataset_manifest"],
            "parent_checkpoint_sha256": args.stage_checkpoint_sha256,
            "parent_checkpoint_update": CHECKPOINT_UPDATE,
            "strong_baseline_contract_sha256": architecture.sha256,
            "target_oriented_contract_sha256": target_contract.sha256,
            "distribution_aware_contract_sha256": distribution_contract.sha256,
            "training_contract_sha256": training_contract.sha256,
            "state_anchor_policy_sha256": state_anchor_policy.sha256,
            "validation_panel_sha256_for_checkpoint_identity_only": panel["panel_sha256"],
            "validation_prepared_manifest_sha256_for_checkpoint_identity_only": validation_prepared["manifest_sha256"],
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
            "candidate": distribution_contract.payload["identity"]["candidate"],
            "parameter_count": parameter_count,
            "parent_parameter_count": PARENT_PARAMETER_COUNT,
            "removed_parent_maintenance_parameters": 769,
            "new_adapter_parameters": adapter_count,
            "initialization_pass": initialization_pass,
            "nonmaintenance_parent_comparison": parent_comparison,
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
            "necessary_category_loss": {name: float(value) for name, value in category_loss.items()},
            "category_active_label_count": {name: float(value) for name, value in category_count.items()},
            "note": "one-day zero-update admission proxy, not a training objective",
        },
        "adapter_gradient": gradient_report
        | {
            "compile_and_run_seconds": gradient_seconds,
            "components": component_gradients,
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
        "next_gate": (
            "separately_authorized_post_admission_training_contract" if passed else "repair_real_v6_gpu_admission"
        ),
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
    parser.add_argument("--training-contract", type=Path, required=True)
    parser.add_argument("--validation-prepared-manifest", type=Path, required=True)
    parser.add_argument("--stage-checkpoint", type=Path, required=True)
    parser.add_argument("--stage-checkpoint-sha256", required=True)
    parser.add_argument("--distribution-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
