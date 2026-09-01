"""Zero-training GPP-to-next-state reachability audit on real v6 days."""

from __future__ import annotations

import argparse
import json
import subprocess
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
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_broad_checkpoint import (
    load_broad_checkpoint,
)
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_fast_owner_gpu_probe import PROBE_SEED, _load_extended_runtime
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
from scripts.hpc.run_target_oriented_coupled_gpp_carbon_feedback_gpu_admission import (
    ADDED_ADAPTER_PARAMETER_COUNT,
    CANDIDATE_PARAMETER_COUNT,
    PARENT_CANDIDATE_PARAMETER_COUNT,
    STRUCTURED_PARAMETER_COUNT,
    TRAINABLE_PARAMETER_COUNT,
    _load_broad_report,
)
from scripts.hpc.run_target_oriented_structured_gpp_gpu_admission import (
    ADAPTER_PARAMETER_COUNT,
    ATTENTION_CHECKPOINT_UPDATE,
    DISTRIBUTION_CHECKPOINT_UPDATE,
    PARENT_PARAMETER_COUNT,
    _replace_adapter,
    _replace_specialist,
    _tree_size,
)
from scripts.hpc.run_target_oriented_training import FROZEN_CLIP_THRESHOLD
from scripts.hpc.target_oriented_validation_prepared import (
    load_target_oriented_validation_prepared,
)

SCHEMA_VERSION = "target_oriented_gpp_state_reachability_result_v1"


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _build_candidate(
    runtime: Mapping[str, Any],
    args: argparse.Namespace,
    reference_retained_tail: Any,
) -> Mapping[str, Any]:
    jax = runtime["jax"]
    import jax.numpy as jnp

    from research.daily_coarse_graining.daily_operator.mechanistic_output_contract import (
        build_mechanistic_learned_schema,
        load_mechanistic_output_contract,
    )
    from research.daily_coarse_graining.daily_operator.mechanistic_process_head import (
        build_mechanistic_output_layout,
    )
    from research.daily_coarse_graining.daily_operator.strong_baseline_contract import (
        load_strong_baseline_contract,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_contract import (
        load_target_oriented_contract,
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
    _validation_prepared, panel = load_target_oriented_validation_prepared(
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
        or int(distribution_report["training"]["optimizer_update_count"]) != DISTRIBUTION_CHECKPOINT_UPDATE
        or distribution_report["checkpoint"]["sha256"] != args.distribution_checkpoint_sha256
    ):
        raise ValueError("GPP reachability distribution parent identity drift")
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
        raise ValueError("GPP reachability structured parent identity drift")

    learned_schema = build_mechanistic_learned_schema(runtime["schema"], mechanism)
    layout = build_mechanistic_output_layout(
        adapter=runtime["adapter"],
        complete_schema=runtime["schema"],
        learned_schema=learned_schema,
        owner_contract=runtime["owner_contract"],
        architecture_contract=architecture,
        mechanism_contract=mechanism,
        inputs=runtime["probe_assembled"],
    )
    head_runtime = dict(runtime) | {
        "architecture_contract": architecture,
        "mechanism_contract": mechanism,
        "learned_schema": learned_schema,
        "target_layout": layout,
    }
    attention_head = _attention_head(head_runtime, target_contract)
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
        raise ValueError("GPP reachability distribution parameter count drift")
    distribution_head = build_target_oriented_distribution_aware_maintenance_head(
        attention_head,
        reference_retained_tail,
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
        raise ValueError("GPP reachability structured parameter count drift")
    head = build_target_oriented_coupled_gpp_carbon_feedback_head(
        structured_head,
        coupled_contract=coupled_contract,
    )
    parameters = initialize_coupled_gpp_carbon_feedback_parameters(
        structured_parameters,
        jax.random.PRNGKey(PROBE_SEED + 20260831),
        state_layout=head.carbon_state_layout,
        coupled_contract=coupled_contract,
        structured_contract=structured_contract,
    )
    trainable = {
        "structured_gpp": parameters["target_oriented"]["structured_gpp"],
        "coupled_gpp_carbon_feedback": parameters["target_oriented"]["coupled_gpp_carbon_feedback"],
    }
    if (
        _tree_size(jax, trainable["coupled_gpp_carbon_feedback"]) != ADDED_ADAPTER_PARAMETER_COUNT
        or _tree_size(jax, trainable) != TRAINABLE_PARAMETER_COUNT
        or _tree_size(jax, parameters) != CANDIDATE_PARAMETER_COUNT
    ):
        raise ValueError("GPP reachability candidate parameter count drift")
    return {
        "head": head,
        "parameters": parameters,
        "coupled_contract": coupled_contract,
        "state_anchor_policy": state_anchor_policy,
        "training_contract": training_contract,
        "target_contract": target_contract,
        "broad_report": broad_report,
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    runtime = _load_extended_runtime(args)
    jax = runtime["jax"]
    device = runtime["devices"][0]
    import jax.numpy as jnp

    from jax_orchidee.driver import orchestration as teacher
    from research.daily_coarse_graining.daily_operator.conservative_update import (
        apply_conservative_update,
    )
    from research.daily_coarse_graining.daily_operator.retained_tail import (
        retained_tail_input,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_gpp_state_reachability import (
        build_gpp_state_reachability_layout,
        load_gpp_state_reachability_contract,
        summarize_gpp_state_jacobians,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp import (
        GPP_FIELD,
    )

    reachability_contract = load_gpp_state_reachability_contract(args.reachability_contract)
    prior_report_sha256 = sha256_file(args.gpu_admission_report)
    prior_report = json.loads(args.gpu_admission_report.read_text(encoding="utf-8"))
    predecessor = reachability_contract.payload["predecessor"]
    if (
        prior_report_sha256 != predecessor["real_v6_admission_report_sha256"]
        or prior_report.get("status") != "passed"
        or prior_report.get("schema_version") != "target_oriented_coupled_gpp_carbon_feedback_real_v6_gpu_admission_v1"
    ):
        raise ValueError("GPP reachability real-v6 admission predecessor drift")

    paths = runtime["context_paths"]
    teacher_context = teacher.prepare_paper_1961_driver_context(
        paths["config"],
        used_run_def_path=paths["run_def"],
        reference_run_dir=paths["reference_run_dir"],
    )
    reference_sample = runtime["ordinary_parent"].sample(ANCHOR_ROWS[0])
    reference_tail = _retained_adapter(
        runtime,
        teacher_context,
        reference_sample,
        lifecycle_start_tstep=48,
        result_day_index=2,
        fast_owner_materializer=runtime["materializer"],
    )
    candidate = _build_candidate(runtime, args, reference_tail)
    head = candidate["head"]
    parameters = candidate["parameters"]
    if candidate["coupled_contract"].sha256 != predecessor["coupled_gpp_carbon_feedback_contract_sha256"]:
        raise ValueError("GPP reachability coupled architecture drift")
    if tuple(runtime["contract"].active_pft_indices) != tuple(
        reachability_contract.payload["differentiated_boundary"]["compact_pft_indices"]
    ):
        raise ValueError("GPP reachability compact PFT identity drift")
    layout = build_gpp_state_reachability_layout(runtime["contract"], head.carbon_state_layout)
    if layout.state_indices.size != 2_386 or len(layout.field_names) != 107:
        raise ValueError("GPP reachability selected-state layout drift")

    def learned_forward(assembled: Any) -> Mapping[str, Any]:
        context = head.assemble_context(parameters, assembled)
        return head.learned_outputs(parameters, context, assembled)

    compiled_learned = jax.jit(learned_forward)
    data_contract = reachability_contract.payload["data"]
    case_specs = [
        (
            f"ordinary-1961-row-{int(row):03d}",
            runtime["ordinary_parent"],
            int(row),
            48 * (int(row) + 1),
            int(day_index),
        )
        for row, day_index in zip(
            data_contract["ordinary_rows"],
            data_contract["ordinary_day_indices"],
            strict=True,
        )
    ]
    case_specs.extend(
        (
            f"restart-1962-row-{int(row):03d}",
            runtime["restart_parent"],
            int(row),
            0,
            int(day_index),
        )
        for row, day_index in zip(
            data_contract["restart_rows"],
            data_contract["restart_day_indices"],
            strict=True,
        )
    )

    jacobians = []
    case_reports = []
    for case_id, parent, row, lifecycle_start_tstep, expected_day_index in case_specs:
        sample = parent.sample(row)
        observed_day_index = int(np.asarray(sample["day_index"]))
        if observed_day_index != expected_day_index:
            raise ValueError(f"GPP reachability day-index drift for {case_id}")
        inputs = runtime["adapter"].assemble_input(sample)
        assembled = runtime["adapter"].input_assembler.assemble(inputs)
        retained_tail = _retained_adapter(
            runtime,
            teacher_context,
            sample,
            lifecycle_start_tstep=lifecycle_start_tstep,
            result_day_index=expected_day_index,
            fast_owner_materializer=runtime["materializer"],
        )
        learned_started = time.perf_counter()
        learned = compiled_learned(assembled)
        jax.block_until_ready(learned)
        learned_seconds = time.perf_counter() - learned_started
        gpp = jnp.asarray(learned[GPP_FIELD], dtype=jnp.float64)
        if gpp.shape != (1, 2):
            raise ValueError(f"GPP reachability input shape drift for {case_id}")

        def next_carbon_with_aux(value: Any) -> tuple[Any, Any]:
            replaced = dict(learned)
            replaced[GPP_FIELD] = value
            prediction = head.prediction_from_learned_outputs(replaced, assembled)
            update = apply_conservative_update(assembled, prediction)
            complete = head.materialize_post_update(
                parameters,
                None,
                assembled,
                prediction,
                update,
            )
            tail = retained_tail(retained_tail_input(assembled, complete, update))
            selected = jnp.asarray(tail.state.other)[jnp.asarray(layout.state_indices, dtype=jnp.int32)]
            return selected, (
                tail.state.other,
                tail.state_defined.other,
                update.water.budget_residual,
                update.carbon.budget_residual,
            )

        jacobian_started = time.perf_counter()
        jacobian_call = jax.jit(jax.jacfwd(next_carbon_with_aux, has_aux=True))
        jacobian, auxiliary = jacobian_call(gpp)
        jax.block_until_ready((jacobian, auxiliary))
        jacobian_seconds = time.perf_counter() - jacobian_started
        jacobian_host = np.asarray(jacobian, dtype=np.float64).reshape((2_386, 2))
        next_state, next_defined, water_residual, carbon_residual = auxiliary
        next_state_host = np.asarray(next_state, dtype=np.float64)
        next_defined_host = np.asarray(next_defined, dtype=np.bool_)
        water_number = float(np.asarray(water_residual))
        carbon_number = float(np.asarray(carbon_residual))
        state_finite = bool(np.all(np.isfinite(next_state_host[next_defined_host])))
        jacobian_finite = bool(np.all(np.isfinite(jacobian_host)))
        bare_maximum = float(np.max(np.abs(jacobian_host[:, 0])))
        pft14_count = int(np.count_nonzero(jacobian_host[:, 1]))
        jacobians.append(jacobian_host)
        case_reports.append(
            {
                "case_id": case_id,
                "year": int(np.asarray(sample["year"])),
                "row": row,
                "day_index": observed_day_index,
                "lifecycle_start_tstep": lifecycle_start_tstep,
                "learned_forward_seconds": learned_seconds,
                "jacobian_compile_and_run_seconds": jacobian_seconds,
                "candidate_gpp": np.asarray(gpp, dtype=np.float64).reshape(-1).tolist(),
                "jacobian_all_finite": jacobian_finite,
                "bare_soil_maximum_absolute_derivative": bare_maximum,
                "pft14_nonzero_coordinate_count": pft14_count,
                "next_state_declared_defined_values_finite": state_finite,
                "water_budget_residual": water_number,
                "carbon_budget_residual": carbon_number,
            }
        )

    anchor_policy = candidate["state_anchor_policy"].state_anchor
    raw_state_scale = np.asarray(
        runtime["base_statistics"].arrays["next_state"].scale,
        dtype=np.float64,
    )
    resolved_state_scale = np.where(
        raw_state_scale > float(anchor_policy["minimum_resolved_empirical_scale"]),
        raw_state_scale,
        float(anchor_policy["unresolved_scale_fallback"]),
    )[layout.state_indices]
    gpp_scale = np.asarray(
        runtime["process_statistics"].outputs[GPP_FIELD].scale,
        dtype=np.float64,
    ).reshape(-1)
    summary = summarize_gpp_state_jacobians(
        np.stack(jacobians),
        layout=layout,
        case_ids=[case[0] for case in case_specs],
        state_scale=resolved_state_scale,
        gpp_scale=gpp_scale,
        normalized_report_threshold=float(reachability_contract.payload["derivative"]["normalized_report_threshold"]),
    )
    acceptance = reachability_contract.payload["acceptance"]
    passed = bool(
        summary["all_finite"]
        and summary["bare_soil_maximum_absolute_derivative"]
        == float(acceptance["bare_soil_maximum_absolute_derivative"])
        and summary["union_reachable_coordinate_count"] >= int(acceptance["union_nonzero_coordinate_count_minimum"])
        and all(
            case["jacobian_all_finite"]
            and case["bare_soil_maximum_absolute_derivative"] == 0.0
            and case["pft14_nonzero_coordinate_count"]
            >= int(acceptance["pft14_nonzero_coordinate_count_minimum_per_case"])
            and case["next_state_declared_defined_values_finite"]
            and abs(case["water_budget_residual"]) <= float(acceptance["water_budget_absolute_tolerance"])
            and abs(case["carbon_budget_residual"]) <= float(acceptance["carbon_budget_absolute_tolerance"])
            for case in case_reports
        )
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "identity": {
            "git_head": _git_head(),
            "dataset_manifest_sha256": EXPECTED["dataset_manifest"],
            "reachability_contract_sha256": reachability_contract.sha256,
            "coupled_gpp_carbon_feedback_contract_sha256": candidate["coupled_contract"].sha256,
            "real_v6_admission_report_sha256": prior_report_sha256,
            "structured_checkpoint_sha256": sha256_file(args.selected_checkpoint),
            "structured_checkpoint_update": 1000,
            "state_anchor_policy_sha256": candidate["state_anchor_policy"].sha256,
        },
        "policy": {
            "training_updates": 0,
            "optimizer_created": False,
            "checkpoint_written": False,
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "fresh_validation_read": False,
            "sealed_test_read": False,
            "forcing_records_reconstructed": 0,
            "subdaily_state_scan_steps": 0,
        },
        "backend": jax.default_backend(),
        "devices": [str(value) for value in runtime["devices"]],
        "data": {
            "landpoint_id": FROZEN_LANDPOINT,
            "case_count": len(case_reports),
            "cases": case_reports,
        },
        "model": {
            "parameter_count": _tree_size(jax, parameters),
            "trainable_parameter_count": TRAINABLE_PARAMETER_COUNT,
            "compact_pft_indices": list(runtime["contract"].active_pft_indices),
            "carbon_coordinate_count": int(layout.state_indices.size),
            "carbon_field_count": len(layout.field_names),
        },
        "reachability": summary,
        "device_memory_after_all_cases": _memory_report(device),
        "next_gate": (
            "freeze_field_grouped_one_day_state_anchor_and_objective_calibration_contract"
            if passed
            else "repair_gpp_state_reachability_audit"
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
    parser.add_argument("--reachability-contract", type=Path, required=True)
    parser.add_argument("--gpu-admission-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
