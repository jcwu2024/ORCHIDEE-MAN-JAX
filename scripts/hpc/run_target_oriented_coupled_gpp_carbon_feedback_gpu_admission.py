"""Zero-update real-v6 admission for coupled GPP/carbon-state feedback."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file
from research.daily_coarse_graining.daily_operator.target_oriented_checkpoint import (
    load_target_oriented_stage_checkpoint,
)
from research.daily_coarse_graining.daily_operator.target_oriented_distribution_aware_checkpoint import (
    load_distribution_aware_checkpoint,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_broad_checkpoint import (
    load_broad_checkpoint,
)
from scripts.hpc.run_strong_baseline_gpu_admission import _masked_state_report, _memory_report
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
from scripts.hpc.run_target_oriented_attention_gpu_admission import _head as _attention_head
from scripts.hpc.run_target_oriented_attention_gpu_admission import (
    _masked_forward_comparison,
    _masked_output_report,
)
from scripts.hpc.run_target_oriented_source_shaped_maintenance_gpu_admission import (
    _nonmaintenance_parent_comparison,
)
from scripts.hpc.run_target_oriented_structured_gpp_gpu_admission import (
    ADAPTER_PARAMETER_COUNT,
    ATTENTION_CHECKPOINT_UPDATE,
    CHECKPOINT_FORWARD_ATOL,
    CHECKPOINT_FORWARD_RTOL,
    DISTRIBUTION_CHECKPOINT_UPDATE,
    PARENT_PARAMETER_COUNT,
    _component_gradient_reports,
    _replace_adapter,
    _replace_specialist,
    _tree_size,
)
from scripts.hpc.run_target_oriented_training import FROZEN_CLIP_THRESHOLD
from scripts.hpc.target_oriented_validation_prepared import (
    load_target_oriented_validation_prepared,
)

SCHEMA_VERSION = "target_oriented_coupled_gpp_carbon_feedback_real_v6_gpu_admission_v1"
STRUCTURED_PARAMETER_COUNT = 12_879_363
ADDED_ADAPTER_PARAMETER_COUNT = 3_080_832
TRAINABLE_PARAMETER_COUNT = 15_960_195
PARENT_CANDIDATE_PARAMETER_COUNT = 106_383_527
CANDIDATE_PARAMETER_COUNT = 109_464_359


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _replace_trainable_subtrees(
    parameters: Mapping[str, Any],
    trainable: Mapping[str, Any],
) -> Mapping[str, Any]:
    target = dict(parameters["target_oriented"])
    target["structured_gpp"] = trainable["structured_gpp"]
    target["coupled_gpp_carbon_feedback"] = trainable["coupled_gpp_carbon_feedback"]
    return dict(parameters, target_oriented=target)


def _parent_parameter_report(
    jax: Any,
    parent: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> Mapping[str, Any]:
    parent_target = parent["target_oriented"]
    candidate_target = candidate["target_oriented"]
    target_keys_match = set(candidate_target) == set(parent_target) | {
        "coupled_gpp_carbon_feedback"
    }
    top_level_same = all(
        candidate[name] is value
        for name, value in parent.items()
        if name != "target_oriented"
    )
    target_same = all(candidate_target[name] is value for name, value in parent_target.items())
    return {
        "parent_element_count": _tree_size(jax, parent),
        "target_registry_extended_only": target_keys_match,
        "top_level_array_objects_preserved": top_level_same,
        "target_array_objects_preserved": target_same,
        "exactly_unchanged": bool(target_keys_match and top_level_same and target_same),
    }


def _adapter_gradient_reports(jax: Any, gradient: Mapping[str, Any]) -> Mapping[str, Any]:
    reports = {
        "coordinate_tokenization": _tree_numeric_report(
            jax,
            (
                gradient["value_input"],
                gradient["field_embedding"],
                gradient["axis_role_embedding"],
                gradient["axis_position_embedding"],
            ),
        ),
        "query_projection": _tree_numeric_report(jax, gradient["query_projection"]),
    }
    reports.update(
        {
            f"cross_attention_block_{index + 1}": _tree_numeric_report(jax, block)
            for index, block in enumerate(gradient["blocks"])
        }
    )
    reports.update(
        {
            f"residual_injection_{name}": _tree_numeric_report(jax, projection)
            for name, projection in zip(
                ("potential", "limitation", "correction"),
                gradient["residual_injections"],
                strict=True,
            )
        }
    )
    return reports


def _load_broad_report(path: Path, coupled_contract: Any) -> Mapping[str, Any]:
    identity = coupled_contract.payload["identity"]
    if sha256_file(path) != identity["parent_broad_training_report_sha256"]:
        raise ValueError("coupled GPP/carbon-feedback broad report hash drift")
    report = json.loads(path.read_text(encoding="utf-8"))
    selected = report.get("selected_checkpoint", {})
    if (
        report.get("schema_version") != "target_oriented_structured_gpp_broad_training_result_v1"
        or report.get("status") != "passed"
        or selected.get("sha256") != identity["parent_checkpoint_sha256"]
        or int(selected.get("completed_updates", -1)) != int(identity["parent_checkpoint_update"])
        or int(selected.get("parameter_count", -1)) != STRUCTURED_PARAMETER_COUNT
    ):
        raise ValueError("coupled GPP/carbon-feedback broad report identity drift")
    return report


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
    from research.daily_coarse_graining.daily_operator.parameter_routing import (
        DEFAULT_PFT_PARAMETER_ROUTING,
    )
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
    from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_carbon_feedback import (
        initialize_coupled_gpp_carbon_feedback_parameters,
        load_coupled_gpp_carbon_feedback_contract,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_distribution_aware_maintenance import (
        initialize_distribution_aware_maintenance_parameters,
        load_distribution_aware_maintenance_contract,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_process_head import (
        build_target_oriented_coupled_gpp_carbon_feedback_head,
        build_target_oriented_distribution_aware_maintenance_head,
        build_target_oriented_structured_gpp_head,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp import (
        GPP_FIELD,
        initialize_structured_gpp_parameters,
        load_structured_gpp_contract,
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
    coupled_contract = load_coupled_gpp_carbon_feedback_contract(args.coupled_contract)
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

    distribution_report = json.loads(args.distribution_training_report.read_text(encoding="utf-8"))
    if (
        distribution_report.get("schema_version") != "target_oriented_distribution_aware_training_v1"
        or distribution_report.get("status") != "passed"
        or int(distribution_report["training"]["optimizer_update_count"])
        != DISTRIBUTION_CHECKPOINT_UPDATE
        or distribution_report["checkpoint"]["sha256"] != args.distribution_checkpoint_sha256
    ):
        raise ValueError("coupled GPP/carbon-feedback distribution parent identity drift")
    distribution_checkpoint = load_distribution_aware_checkpoint(
        args.distribution_checkpoint,
        expected_sha256=args.distribution_checkpoint_sha256,
        identity=distribution_report["identity"],
        completed_updates=DISTRIBUTION_CHECKPOINT_UPDATE,
        parameter_count=ADAPTER_PARAMETER_COUNT,
        clip_threshold=float(distribution_report["training"]["clip_threshold"]),
    )
    broad_report = _load_broad_report(args.broad_training_report, coupled_contract)
    identity = coupled_contract.payload["identity"]
    if (
        sha256_file(args.selected_checkpoint) != identity["parent_checkpoint_sha256"]
        or structured_contract.sha256 != identity["parent_candidate_contract_sha256"]
    ):
        raise ValueError("coupled GPP/carbon-feedback structured parent identity drift")

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

    initialized_distribution = initialize_distribution_aware_maintenance_parameters(
        attention_parameters,
        jax.random.PRNGKey(PROBE_SEED + 20260828),
        schema=learned_schema,
        target_contract=target_contract,
        distribution_contract=distribution_contract,
    )
    distribution_parameters = _replace_adapter(
        initialized_distribution,
        jax.device_put(distribution_checkpoint["adapter"]),
    )
    if _tree_size(jax, distribution_parameters) != PARENT_PARAMETER_COUNT:
        raise ValueError("coupled GPP/carbon-feedback distribution parameter count drift")
    distribution_head = build_target_oriented_distribution_aware_maintenance_head(
        attention_head,
        retained_tail,
        distribution_contract=distribution_contract,
    )
    structured_head = build_target_oriented_structured_gpp_head(
        distribution_head,
        structured_contract=structured_contract,
    )
    initialized_structured = initialize_structured_gpp_parameters(
        distribution_parameters,
        jax.random.PRNGKey(PROBE_SEED + 20260829),
        schema=learned_schema,
        target_contract=target_contract,
        structured_contract=structured_contract,
    )
    selected_checkpoint = load_broad_checkpoint(
        args.selected_checkpoint,
        expected_sha256=identity["parent_checkpoint_sha256"],
        identity=broad_report["identity"],
        completed_updates=int(identity["parent_checkpoint_update"]),
        parameter_count=STRUCTURED_PARAMETER_COUNT,
        clip_threshold=float(broad_report["training"]["clip_threshold"]),
    )
    structured_parameters = _replace_specialist(
        initialized_structured,
        jax.device_put(selected_checkpoint["specialist"]),
    )
    if _tree_size(jax, structured_parameters) != PARENT_CANDIDATE_PARAMETER_COUNT:
        raise ValueError("coupled GPP/carbon-feedback structured parameter count drift")

    head = build_target_oriented_coupled_gpp_carbon_feedback_head(
        structured_head,
        coupled_contract=coupled_contract,
    )
    memory_before_parameters = _memory_report(device)
    parameters = initialize_coupled_gpp_carbon_feedback_parameters(
        structured_parameters,
        jax.random.PRNGKey(PROBE_SEED + 20260831),
        state_layout=head.carbon_state_layout,
        coupled_contract=coupled_contract,
        structured_contract=structured_contract,
    )
    trainable = {
        "structured_gpp": parameters["target_oriented"]["structured_gpp"],
        "coupled_gpp_carbon_feedback": parameters["target_oriented"][
            "coupled_gpp_carbon_feedback"
        ],
    }
    if (
        _tree_size(jax, trainable["coupled_gpp_carbon_feedback"])
        != ADDED_ADAPTER_PARAMETER_COUNT
        or _tree_size(jax, trainable) != TRAINABLE_PARAMETER_COUNT
        or _tree_size(jax, parameters) != CANDIDATE_PARAMETER_COUNT
    ):
        raise ValueError("coupled GPP/carbon-feedback candidate parameter count drift")
    parent_parameter_report = _parent_parameter_report(jax, structured_parameters, parameters)

    parent_context = structured_head.assemble_context(structured_parameters, assembled)
    parent_learned = structured_head.learned_outputs(
        structured_parameters,
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

    def trainable_loss_function(
        fixed_values: Mapping[str, Any],
        trainable_values: Mapping[str, Any],
    ) -> tuple[Any, Any]:
        values = _replace_trainable_subtrees(fixed_values, trainable_values)
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
        jax.value_and_grad(trainable_loss_function, argnums=1, has_aux=True)
    )
    (loss_value, loss_aux), trainable_gradient = gradient_call(parameters, trainable)
    jax.block_until_ready((loss_value, trainable_gradient))
    gradient_seconds = time.perf_counter() - gradient_started
    gradient_report = _tree_numeric_report(jax, trainable_gradient)
    structured_components = _component_gradient_reports(
        jax,
        trainable_gradient["structured_gpp"],
    )
    adapter_components = _adapter_gradient_reports(
        jax,
        trainable_gradient["coupled_gpp_carbon_feedback"],
    )
    primary_loss, category_loss, category_count = loss_aux

    def gpp_parameter_sensitivity(pft_parameters: Any) -> Any:
        pft = assembled.pft._replace(pft_parameters=pft_parameters)
        changed = assembled._replace(
            pft=pft,
            canonical_input=assembled.canonical_input._replace(pft=pft),
        )
        context = head.assemble_context(parameters, changed)
        return jnp.sum(head.learned_outputs(parameters, context, changed)[GPP_FIELD])

    route_started = time.perf_counter()
    route_gradient = jax.jit(jax.grad(gpp_parameter_sensitivity))(
        jnp.asarray(assembled.pft.pft_parameters)
    )
    jax.block_until_ready(route_gradient)
    route_seconds = time.perf_counter() - route_started
    vcmax_index = DEFAULT_PFT_PARAMETER_ROUTING.index("vcmax25")
    forbidden_names = (
        "maint_resp_slope_a",
        "maint_resp_slope_b",
        "maint_resp_slope_c",
        "alloc_min",
        "residence_time",
    )
    route_report = {
        "compile_and_run_seconds": route_seconds,
        "vcmax25_active_pft_gradient": float(route_gradient[1, vcmax_index]),
        "forbidden_channel_maximum_absolute_gradient": {
            name: float(
                jnp.max(
                    jnp.abs(route_gradient[:, DEFAULT_PFT_PARAMETER_ROUTING.index(name)])
                )
            )
            for name in forbidden_names
        },
    }
    route_pass = bool(
        np.isfinite(route_report["vcmax25_active_pft_gradient"])
        and abs(route_report["vcmax25_active_pft_gradient"]) > 0.0
        and all(value == 0.0 for value in route_report["forbidden_channel_maximum_absolute_gradient"].values())
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

    category_counts_pass = all(
        float(category_count[name]) > 0.0
        for name in (
            PRIMARY_SCIENCE_DIRECT,
            STATE_BUDGET_ESSENTIAL,
            AUXILIARY_SUPERVISION,
        )
    )
    components = (*structured_components.values(), *adapter_components.values())
    component_gradients_pass = all(
        value["all_finite"] and value["l2_norm"] > 0.0 for value in components
    )
    loss_number = float(loss_value)
    passed = bool(
        parent_parameter_report["exactly_unchanged"]
        and parent_output_comparison["exactly_equal"]
        and eager_report["all_declared_defined_finite"]
        and compiled_report["all_declared_defined_finite"]
        and comparison["passed"]
        and np.isfinite(loss_number)
        and loss_number > 0.0
        and gradient_report["all_finite"]
        and gradient_report["element_count"] == TRAINABLE_PARAMETER_COUNT
        and gradient_report["l2_norm"] > 0.0
        and component_gradients_pass
        and category_counts_pass
        and route_pass
        and state_report["all_declared_defined_finite"]
        and abs(water_residual) <= 2.0e-12
        and abs(carbon_residual) <= 2.0e-12
        and len(result.prediction.named_outputs) == 181
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "policy": {
            "training_updates": 0,
            "optimizer_created": False,
            "checkpoint_written": False,
            "parent_parameters_trainable": False,
            "gradient_target": "structured_gpp_plus_coupled_gpp_carbon_feedback",
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
            "git_head": _git_head(),
            "dataset_manifest_sha256": EXPECTED["dataset_manifest"],
            "attention_checkpoint_sha256": args.stage_checkpoint_sha256,
            "distribution_checkpoint_sha256": args.distribution_checkpoint_sha256,
            "broad_training_report_sha256": sha256_file(args.broad_training_report),
            "structured_checkpoint_sha256": sha256_file(args.selected_checkpoint),
            "structured_checkpoint_update": int(identity["parent_checkpoint_update"]),
            "strong_baseline_contract_sha256": architecture.sha256,
            "target_oriented_contract_sha256": target_contract.sha256,
            "distribution_aware_contract_sha256": distribution_contract.sha256,
            "structured_gpp_contract_sha256": structured_contract.sha256,
            "coupled_gpp_carbon_feedback_contract_sha256": coupled_contract.sha256,
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
            "candidate": coupled_contract.payload["identity"]["candidate"],
            "parameter_count": _tree_size(jax, parameters),
            "parent_parameter_count": _tree_size(jax, structured_parameters),
            "new_adapter_parameter_count": _tree_size(
                jax,
                trainable["coupled_gpp_carbon_feedback"],
            ),
            "trainable_parameter_count": _tree_size(jax, trainable),
            "carbon_coordinate_count": int(head.carbon_state_layout.state_indices.size),
            "carbon_field_count": len(head.carbon_state_layout.field_names),
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
        "trainable_gradient": gradient_report
        | {
            "compile_and_run_seconds": gradient_seconds,
            "structured_gpp_components": structured_components,
            "carbon_feedback_components": adapter_components,
        },
        "parameter_routes": route_report | {"passed": route_pass},
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
            "freeze_gpp_reachability_and_objective_calibration_contracts"
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
    parser.add_argument("--broad-training-report", type=Path, required=True)
    parser.add_argument("--selected-checkpoint", type=Path, required=True)
    parser.add_argument("--coupled-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    report = run(parser.parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
