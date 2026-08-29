"""Zero-update real-v6 admission for the structured daily GPP specialist."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import (
    sha256_file,
)
from research.daily_coarse_graining.daily_operator.target_oriented_checkpoint import (
    load_target_oriented_stage_checkpoint,
)
from research.daily_coarse_graining.daily_operator.target_oriented_distribution_aware_checkpoint import (
    load_distribution_aware_checkpoint,
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
    PARAMETER_COUNT as ATTENTION_PARAMETER_COUNT,
)
from scripts.hpc.run_target_oriented_attention_gpu_admission import (
    _head as _attention_head,
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

PARENT_PARAMETER_COUNT = 93_504_933
SPECIALIST_PARAMETER_COUNT = 12_879_363
CANDIDATE_PARAMETER_COUNT = 106_383_527
ADAPTER_PARAMETER_COUNT = 9_750_529
ATTENTION_CHECKPOINT_UPDATE = 1000
DISTRIBUTION_CHECKPOINT_UPDATE = 250
CHECKPOINT_FORWARD_ATOL = 1.0e-5
CHECKPOINT_FORWARD_RTOL = 5.0e-6


def _replace_adapter(
    parameters: Mapping[str, Any],
    adapter: Mapping[str, Any],
) -> Mapping[str, Any]:
    target = dict(parameters["target_oriented"])
    target["distribution_aware_maintenance"] = adapter
    return dict(parameters, target_oriented=target)


def _replace_specialist(
    parameters: Mapping[str, Any],
    specialist: Mapping[str, Any],
) -> Mapping[str, Any]:
    target = dict(parameters["target_oriented"])
    target["structured_gpp"] = specialist
    return dict(parameters, target_oriented=target)


def _tree_size(jax: Any, value: Any) -> int:
    return sum(int(np.prod(leaf.shape, dtype=np.int64)) for leaf in jax.tree_util.tree_leaves(value))


def _parent_parameter_report(
    jax: Any,
    parent: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    gpp_index: int,
) -> Mapping[str, Any]:
    parent_target = parent["target_oriented"]
    candidate_target = candidate["target_oriented"]
    target_keys_match = set(candidate_target) == set(parent_target) | {"structured_gpp"}
    top_level_same = all(
        candidate[name] is value
        for name, value in parent.items()
        if name != "target_oriented"
    )
    target_same = all(
        candidate_target[name] is value
        for name, value in parent_target.items()
        if name != "primary_heads"
    )
    parent_heads = parent_target["primary_heads"]
    candidate_heads = candidate_target["primary_heads"]
    heads_same = len(parent_heads) == len(candidate_heads) and all(
        candidate_heads[index] is value
        for index, value in enumerate(parent_heads)
        if index != gpp_index
    )
    removed_count = _tree_size(jax, parent_heads[gpp_index])
    removed = candidate_heads[gpp_index] == {}
    exactly_unchanged = bool(
        target_keys_match
        and top_level_same
        and target_same
        and heads_same
        and removed
        and removed_count == 769
    )
    return {
        "parent_element_count": _tree_size(jax, parent),
        "target_registry_extended_only": target_keys_match,
        "top_level_array_objects_preserved": top_level_same,
        "target_array_objects_preserved": target_same,
        "non_gpp_primary_head_objects_preserved": heads_same,
        "removed_gpp_projection_parameters": removed_count,
        "old_gpp_projection_empty": removed,
        "exactly_unchanged": exactly_unchanged,
    }


def _component_gradient_reports(jax: Any, gradient: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "forcing_encoder": _tree_numeric_report(
            jax,
            (
                gradient["forcing_input"],
                gradient["forcing_field_embedding"],
                gradient["forcing_position_embedding"],
                gradient["forcing_blocks"],
            ),
        ),
        "pft_and_parent_context": _tree_numeric_report(
            jax,
            (
                gradient["branch_embedding"],
                gradient["parent_hidden_projection"],
                gradient["local_input"],
                gradient["pft_query_projection"],
            ),
        ),
        "potential_branch": _tree_numeric_report(jax, gradient["potential_blocks"]),
        "limitation_branch": _tree_numeric_report(jax, gradient["limitation_blocks"]),
        "correction_branch": _tree_numeric_report(jax, gradient["correction_blocks"]),
        "potential_output": _tree_numeric_report(jax, gradient["potential_output"]),
        "limitation_output": _tree_numeric_report(jax, gradient["limitation_output"]),
        "correction_output": _tree_numeric_report(jax, gradient["correction_output"]),
    }


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
        initialize_distribution_aware_maintenance_parameters,
        load_distribution_aware_maintenance_contract,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_process_head import (
        build_target_oriented_distribution_aware_maintenance_head,
        build_target_oriented_structured_gpp_head,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp import (
        GPP_FIELD,
        initialize_structured_gpp_parameters,
        load_structured_gpp_contract,
        resolve_gpp_field_index,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_training import (
        load_target_oriented_state_anchor_policy,
        load_target_oriented_training_contract,
    )

    architecture = load_strong_baseline_contract()
    mechanism = load_mechanistic_output_contract()
    target_contract = load_target_oriented_contract()
    distribution_contract = load_distribution_aware_maintenance_contract(args.distribution_contract)
    structured_contract = load_structured_gpp_contract(args.structured_contract)
    training_contract = load_target_oriented_training_contract(args.training_contract)
    state_anchor_policy = load_target_oriented_state_anchor_policy()
    validation_prepared, panel = load_target_oriented_validation_prepared(
        args.validation_prepared_manifest,
        training_contract_sha256=training_contract.sha256,
        target_contract_sha256=target_contract.sha256,
        verify_artifacts=False,
    )
    attention_checkpoint = load_target_oriented_stage_checkpoint(
        args.stage_checkpoint,
        expected_sha256=args.stage_checkpoint_sha256,
        completed_updates=ATTENTION_CHECKPOINT_UPDATE,
        parameter_count=ATTENTION_PARAMETER_COUNT,
        clip_threshold=FROZEN_CLIP_THRESHOLD,
        training_contract_sha256=training_contract.sha256,
        target_contract_sha256=target_contract.sha256,
        state_anchor_policy_sha256=state_anchor_policy.sha256,
        validation_panel_sha256=panel["panel_sha256"],
    )
    attention_parameters = jax.tree_util.tree_map(jnp.asarray, attention_checkpoint["parameters"])

    training_report = json.loads(args.distribution_training_report.read_text(encoding="utf-8"))
    if (
        training_report.get("schema_version") != "target_oriented_distribution_aware_training_v1"
        or training_report.get("status") != "passed"
        or int(training_report["training"]["optimizer_update_count"])
        != DISTRIBUTION_CHECKPOINT_UPDATE
        or training_report["checkpoint"]["sha256"] != args.distribution_checkpoint_sha256
        or structured_contract.payload["identity"]["parent_checkpoint_sha256"]
        != args.distribution_checkpoint_sha256
    ):
        raise ValueError("structured GPP distribution-aware parent identity drift")
    distribution_checkpoint = load_distribution_aware_checkpoint(
        args.distribution_checkpoint,
        expected_sha256=args.distribution_checkpoint_sha256,
        identity=training_report["identity"],
        completed_updates=DISTRIBUTION_CHECKPOINT_UPDATE,
        parameter_count=ADAPTER_PARAMETER_COUNT,
        clip_threshold=float(training_report["training"]["clip_threshold"]),
    )

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
    attention_head = _attention_head(runtime, target_contract)

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

    initialized_parent = initialize_distribution_aware_maintenance_parameters(
        attention_parameters,
        jax.random.PRNGKey(PROBE_SEED + 20260828),
        schema=learned_schema,
        target_contract=target_contract,
        distribution_contract=distribution_contract,
    )
    parent_parameters = _replace_adapter(
        initialized_parent,
        jax.device_put(distribution_checkpoint["adapter"]),
    )
    if _tree_size(jax, parent_parameters) != PARENT_PARAMETER_COUNT:
        raise ValueError("structured GPP parent parameter count drift")
    distribution_head = build_target_oriented_distribution_aware_maintenance_head(
        attention_head,
        retained_tail,
        distribution_contract=distribution_contract,
    )
    head = build_target_oriented_structured_gpp_head(
        distribution_head,
        structured_contract=structured_contract,
    )

    memory_before_parameters = _memory_report(device)
    parameters = initialize_structured_gpp_parameters(
        parent_parameters,
        jax.random.PRNGKey(PROBE_SEED + 20260829),
        schema=learned_schema,
        target_contract=target_contract,
        structured_contract=structured_contract,
    )
    specialist = parameters["target_oriented"]["structured_gpp"]
    specialist_count = _tree_size(jax, specialist)
    parameter_count = _tree_size(jax, parameters)
    if specialist_count != SPECIALIST_PARAMETER_COUNT or parameter_count != CANDIDATE_PARAMETER_COUNT:
        raise ValueError("structured GPP candidate parameter count drift")
    gpp_index = resolve_gpp_field_index(learned_schema, target_contract)
    parent_parameter_report = _parent_parameter_report(
        jax,
        parent_parameters,
        parameters,
        gpp_index=gpp_index,
    )

    parent_context = distribution_head.assemble_context(parent_parameters, assembled)
    parent_learned = distribution_head.learned_outputs(
        parent_parameters,
        parent_context,
        assembled,
    )
    candidate_context = head.assemble_context(parameters, assembled)
    candidate_learned = head.learned_outputs(parameters, candidate_context, assembled)
    jax.block_until_ready((parent_learned, candidate_learned))
    parent_output_comparison = _nonmaintenance_parent_comparison(
        parent_learned,
        candidate_learned,
        GPP_FIELD,
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

    def specialist_loss_function(
        fixed_values: Mapping[str, Any],
        specialist_values: Mapping[str, Any],
    ) -> tuple[Any, Any]:
        values = _replace_specialist(fixed_values, specialist_values)
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
    gradient_call = jax.jit(
        jax.value_and_grad(specialist_loss_function, argnums=1, has_aux=True)
    )
    (loss_value, loss_aux), specialist_gradient = gradient_call(parameters, specialist)
    jax.block_until_ready((loss_value, specialist_gradient))
    gradient_seconds = time.perf_counter() - gradient_started
    gradient_report = _tree_numeric_report(jax, specialist_gradient)
    component_gradients = _component_gradient_reports(jax, specialist_gradient)
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
    component_gradients_pass = all(
        value["all_finite"] and value["l2_norm"] > 0.0
        for value in component_gradients.values()
    )
    loss_number = float(loss_value)
    passed = (
        parent_parameter_report["exactly_unchanged"]
        and parent_output_comparison["exactly_equal"]
        and eager_report["all_declared_defined_finite"]
        and compiled_report["all_declared_defined_finite"]
        and comparison["passed"]
        and np.isfinite(loss_number)
        and loss_number > 0.0
        and gradient_report["all_finite"]
        and gradient_report["element_count"] == SPECIALIST_PARAMETER_COUNT
        and gradient_report["l2_norm"] > 0.0
        and component_gradients_pass
        and category_counts_pass
        and state_report["all_declared_defined_finite"]
        and abs(water_residual) <= 2.0e-12
        and abs(carbon_residual) <= 2.0e-12
        and len(result.prediction.named_outputs) == 181
    )
    report = {
        "schema_version": "target_oriented_structured_gpp_real_v6_gpu_admission_v1",
        "status": "passed" if passed else "failed",
        "policy": {
            "training_updates": 0,
            "optimizer_created": False,
            "checkpoint_written": False,
            "parent_parameters_trainable": False,
            "gradient_target": "target_oriented.structured_gpp_only",
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
            "attention_checkpoint_sha256": args.stage_checkpoint_sha256,
            "attention_checkpoint_update": ATTENTION_CHECKPOINT_UPDATE,
            "distribution_checkpoint_sha256": args.distribution_checkpoint_sha256,
            "distribution_checkpoint_update": DISTRIBUTION_CHECKPOINT_UPDATE,
            "distribution_training_report_sha256": sha256_file(
                args.distribution_training_report
            ),
            "strong_baseline_contract_sha256": architecture.sha256,
            "target_oriented_contract_sha256": target_contract.sha256,
            "distribution_aware_contract_sha256": distribution_contract.sha256,
            "structured_gpp_contract_sha256": structured_contract.sha256,
            "training_contract_sha256": training_contract.sha256,
            "state_anchor_policy_sha256": state_anchor_policy.sha256,
            "validation_panel_sha256_for_checkpoint_identity_only": panel["panel_sha256"],
            "validation_prepared_manifest_sha256_for_checkpoint_identity_only": validation_prepared[
                "manifest_sha256"
            ],
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
            "candidate": structured_contract.payload["identity"]["candidate"],
            "parameter_count": parameter_count,
            "parent_parameter_count": PARENT_PARAMETER_COUNT,
            "removed_parent_gpp_parameters": 769,
            "new_specialist_parameters": specialist_count,
            "parent_parameters": parent_parameter_report,
            "non_gpp_parent_output_comparison": parent_output_comparison,
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
            "necessary_category_loss": {
                name: float(value) for name, value in category_loss.items()
            },
            "category_active_label_count": {
                name: float(value) for name, value in category_count.items()
            },
            "note": "one-day zero-update admission proxy, not a training objective",
        },
        "specialist_gradient": gradient_report
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
            "freeze_separate_training_contract_before_optimizer_update"
            if passed
            else "repair_real_v6_gpu_admission"
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
    parser.add_argument("--distribution-training-report", type=Path, required=True)
    parser.add_argument("--distribution-checkpoint", type=Path, required=True)
    parser.add_argument("--distribution-checkpoint-sha256", required=True)
    parser.add_argument("--structured-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    report = run(parser.parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
