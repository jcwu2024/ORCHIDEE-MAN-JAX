"""Localize the accepted Day-5 GPP jump inside the potential residual blocks."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_objective import (
    coupled_gpp_trainable_tree,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_potential_internal_attribution import (
    BLOCK_PARAMETER_GROUPS,
    COMPONENT_PARAMETER_GROUPS,
    EXPECTED_MAXIMUM_INTERVAL,
    INTERPOLATION_FRACTIONS,
    LOWER_FRACTION,
    PARENT_REPORT_SHA256,
    SCHEMA_VERSION,
    UPPER_FRACTION,
    classify_activation_precision_paths,
    classify_parameter_localization,
    replace_parameter_paths,
    summarize_trace_pair,
    traced_potential_branch,
    validate_parameter_group_coverage,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_snowheat_causal import (
    TARGET_ARTIFACT_UPDATE,
    TARGET_DAY_INDEX,
    TARGET_WINDOW_INDEX,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_structured_precision_attribution import (
    STRUCTURED_GPP_PARAMETER_COUNT,
    summarize_gpp_path,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_day5_crossed_attribution import (
    _runtime_year_for_update,
    _tree_day_batch,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_localized_jump import (
    FORWARD_REPRODUCTION_ATOL,
    FORWARD_REPRODUCTION_RTOL,
    _make_trace_evaluator,
    reconstruct_localized_jump_parameters,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_snowheat_causal import (
    FACTORIZATION_ATOL,
    FACTORIZATION_RTOL,
    _make_factor_evaluator,
    _scenario_day_starts,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_structured_precision_attribution import (
    _direction_integrity,
    _evaluate_record,
    _interpolate_tree,
    _record_is_valid,
    _replace_structured_gpp,
    _validate_initialized_structured_tree,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_structured_precision_attribution import (
    _parser as _precision_parser,
)
from scripts.hpc.calibrate_target_oriented_distribution_aware_guard import _load_artifact
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_coupled_gpp_artifact_aware_persistent_training import (
    _initialization_only,
)
from scripts.hpc.run_target_oriented_coupled_gpp_direct_recalibration import _git_head
from scripts.hpc.run_target_oriented_full_step_preflight import _slice

PARENT_SCHEMA_VERSION = "target_oriented_coupled_gpp_structured_precision_attribution_result_v1"
PARENT_CLASSIFICATION = "structured_gpp_conditioning_persists_with_float64_weights"
EXPECTED_RUNTIME_YEAR = 1976
EXPECTED_VEGETATION_PFT_INDEX = 1


def _load_parent_report(path: Path) -> tuple[Mapping[str, Any], str]:
    digest = sha256_file(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if digest != PARENT_REPORT_SHA256:
        raise ValueError("potential-internal parent report SHA256 drift")
    if (
        report.get("schema_version") != PARENT_SCHEMA_VERSION
        or report.get("status") != "passed"
        or report.get("classification") != PARENT_CLASSIFICATION
        or report.get("integrity", {}).get("engineering_gate_passed") is not True
    ):
        raise ValueError("potential-internal parent report status drift")
    return report, digest


def _make_potential_input_evaluator(reconstruction: Any) -> Any:
    runtime = reconstruction.runtime
    jax = reconstruction.jax
    jnp = reconstruction.jnp
    head = reconstruction.bootstrap["head"]
    assembler = runtime["adapter"].input_assembler
    contract = runtime["contract"]

    from research.daily_coarse_graining.daily_operator.successor_training import (
        rollout_input_from_template,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_attention import (
        _target_oriented_attention_decoded_outputs,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_carbon_feedback import (
        carbon_feedback_query_residuals,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_source_shaped_maintenance import (
        MAINTENANCE_FIELD,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp import (
        GPP_FIELD,
        POTENTIAL_FORCING_MASK,
        _structured_forcing_tokens,
        structured_gpp_query_state,
    )

    @jax.jit
    def evaluate(parameters: Any, templates: Any, continuous: Any, discrete: Any) -> Any:
        def one(template: Any, state: Any, state_discrete: Any) -> Any:
            current = rollout_input_from_template(template, state, state_discrete, contract)
            assembled = assembler.assemble(current)
            context = head.assemble_context(parameters, assembled)
            _outputs, hidden = _target_oriented_attention_decoded_outputs(
                parameters,
                context,
                assembled,
                schema=head.learned_schema,
                layout=head.layout,
                input_normalization=head.input_normalization,
                output_normalization=head.output_normalization,
                forcing_layout=head.forcing_layout,
                target_contract=head.target_contract,
                message_gate_scale=head.message_gate_scale,
                baseline_config=head.config,
                config=head.target_config,
                deferred_primary_fields=frozenset((GPP_FIELD, MAINTENANCE_FIELD)),
            )
            query_state = structured_gpp_query_state(
                parameters,
                context,
                hidden[GPP_FIELD],
                assembled,
                state_layout=head.structured_state_layout,
                input_normalization=head.input_normalization,
                config=head.structured_config,
            )
            residuals = carbon_feedback_query_residuals(
                parameters,
                query_state.base_query,
                query_state.local_active,
                assembled,
                state_layout=head.carbon_state_layout,
                input_normalization=head.input_normalization,
                config=head.coupled_config,
            )
            specialist = parameters["target_oriented"]["structured_gpp"]
            forcing_tokens, forcing_valid, forcing_field = _structured_forcing_tokens(
                specialist,
                assembled,
                head.input_normalization,
                head.structured_config,
            )
            forcing_count = forcing_tokens.shape[0]
            parent_source = jnp.asarray(context.source_tokens[forcing_count:], dtype=jnp.float32)
            parent_source_mask = jnp.asarray(context.source_mask[forcing_count:], dtype=jnp.bool_)
            parent_source_family = jnp.asarray(
                context.source_family_index[forcing_count:],
                dtype=jnp.int32,
            )
            canopy_index = head.layout.expert_names.index("canopy_gpp")
            allowed_family = jnp.asarray(head.layout.source_allowed[canopy_index], dtype=jnp.bool_)
            parent_source_mask &= allowed_family[parent_source_family]
            axis_tokens = jnp.asarray(context.expert_axis_tokens[canopy_index], dtype=jnp.float32)
            axis_mask = jnp.asarray(context.expert_axis_mask[canopy_index], dtype=jnp.bool_)
            shared_bank = jnp.concatenate((forcing_tokens, parent_source, axis_tokens), axis=0)
            n_pft = int(query_state.base_query.shape[0])
            shared_bank = jnp.broadcast_to(
                shared_bank[None, :, :],
                (n_pft, shared_bank.shape[0], head.structured_config.width),
            )
            local_bank = jnp.stack(
                (query_state.pft_tokens, query_state.local_tokens, query_state.parent_tokens),
                axis=1,
            )
            bank = jnp.concatenate((shared_bank, local_bank), axis=1)
            potential_forcing = jnp.asarray(POTENTIAL_FORCING_MASK, dtype=jnp.bool_)
            common_tail_mask = jnp.concatenate((parent_source_mask, axis_mask, jnp.ones((3,), dtype=jnp.bool_)))
            potential_mask = jnp.broadcast_to(
                jnp.concatenate((forcing_valid & potential_forcing, common_tail_mask))[None, :],
                (n_pft, bank.shape[1]),
            )
            potential_mask &= query_state.local_active[:, None]
            return {
                "query": query_state.base_query + specialist["branch_embedding"][0] + residuals[0],
                "bank": bank,
                "bank_mask": potential_mask,
                "active": query_state.local_active,
            }

        return jax.vmap(one)(templates, continuous, discrete)

    return evaluate


def _make_shadow_evaluators(reconstruction: Any, dtype: Any) -> tuple[Any, Any]:
    jax = reconstruction.jax
    jnp = reconstruction.jnp
    heads = int(reconstruction.bootstrap["head"].structured_config.attention_heads)

    @jax.jit
    def potential(
        blocks: Any,
        output: Any,
        branch_inputs: Mapping[str, Any],
    ) -> Any:
        return traced_potential_branch(
            jax,
            jnp,
            blocks,
            output,
            branch_inputs["query"],
            branch_inputs["bank"],
            branch_inputs["bank_mask"],
            branch_inputs["active"],
            heads=heads,
            dtype=dtype,
        )["potential"]

    @jax.jit
    def trace(
        blocks: Any,
        output: Any,
        branch_inputs: Mapping[str, Any],
    ) -> Any:
        return traced_potential_branch(
            jax,
            jnp,
            blocks,
            output,
            branch_inputs["query"],
            branch_inputs["bank"],
            branch_inputs["bank_mask"],
            branch_inputs["active"],
            heads=heads,
            dtype=dtype,
        )

    return potential, trace


def _selected_potential(record: Mapping[str, Any], pft_index: int) -> float:
    return float(np.asarray(record["potential"], dtype=np.float64).reshape(-1)[pft_index])


def _evaluate_blocks(
    blocks: Any,
    *,
    control_structured: Mapping[str, Any],
    reconstruction: Any,
    factor_evaluator: Any,
    template_batch: Any,
    control_state: Any,
    discrete_batch: Any,
    pft_index: int,
) -> Mapping[str, Any]:
    structured = dict(control_structured, potential_blocks=blocks)
    return _evaluate_record(
        factor_evaluator,
        reconstruction,
        _replace_structured_gpp(reconstruction.control_parameters, structured),
        template_batch,
        control_state,
        discrete_batch,
        pft_index,
    )


def _causal_controls(
    registry: Mapping[str, tuple[tuple[int | str, ...], ...]],
    lower_blocks: Any,
    upper_blocks: Any,
    **evaluate_kwargs: Any,
) -> tuple[dict[str, float], dict[str, float], dict[str, Any], list[Mapping[str, Any]]]:
    upper_only_gpp = {}
    upper_without_gpp = {}
    records = {}
    evaluated = []
    for index, (name, paths) in enumerate(registry.items(), start=1):
        upper_only = _evaluate_blocks(
            replace_parameter_paths(lower_blocks, upper_blocks, paths),
            **evaluate_kwargs,
        )
        upper_without = _evaluate_blocks(
            replace_parameter_paths(upper_blocks, lower_blocks, paths),
            **evaluate_kwargs,
        )
        upper_only_gpp[name] = float(upper_only["gpp"])
        upper_without_gpp[name] = float(upper_without["gpp"])
        records[name] = {
            "upper_group_only": upper_only,
            "upper_without_group": upper_without,
        }
        evaluated.extend((upper_only, upper_without))
        print(f"coupled_gpp potential_internal control={index}/{len(registry)} name={name}", flush=True)
    return upper_only_gpp, upper_without_gpp, records, evaluated


def _policy_record() -> Mapping[str, Any]:
    return {
        "logical_updates_applied": 0,
        "historical_parameter_displacements_reconstructed_in_memory": 1,
        "optimizer_invoked": False,
        "trainable_parameters_written": False,
        "checkpoint_written": False,
        "recovery_written": False,
        "architecture_changed": False,
        "teacher_data_regenerated": False,
        "historical_v5_used": False,
        "fresh_validation_values_evaluated": False,
        "sealed_test_read": False,
        "future_teacher_state_used": False,
        "forcing_records_reconstructed": 0,
        "subdaily_state_scan_steps": 0,
    }


def run(args: Any) -> Mapping[str, Any]:
    started = time.perf_counter()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)

    parent, parent_digest = _load_parent_report(args.structured_precision_report.resolve())
    bootstrap = _initialization_only(args)
    runtime = bootstrap["context"]["runtime"]
    jax = runtime["jax"]
    jnp = runtime["jnp"]
    if not bool(jax.config.jax_enable_x64):
        raise RuntimeError("potential-internal attribution requires JAX float64 support")
    runtime_year = _runtime_year_for_update(
        bootstrap["context"]["schedule"],
        TARGET_ARTIFACT_UPDATE,
    )
    if runtime_year != EXPECTED_RUNTIME_YEAR:
        raise ValueError("potential-internal runtime-year drift")
    initialized_structured = coupled_gpp_trainable_tree(bootstrap["initialized_parameters"])["structured_gpp"]
    _validate_initialized_structured_tree(initialized_structured)

    reconstruction = reconstruct_localized_jump_parameters(args, bootstrap=bootstrap)
    direction = _direction_integrity(reconstruction, parent)
    control_structured = coupled_gpp_trainable_tree(reconstruction.control_parameters)["structured_gpp"]
    treatment_structured = coupled_gpp_trainable_tree(reconstruction.treatment_parameters)["structured_gpp"]
    if any(
        int(sum(np.asarray(value).size for value in jax.tree_util.tree_leaves(tree))) != STRUCTURED_GPP_PARAMETER_COUNT
        for tree in (control_structured, treatment_structured)
    ):
        raise ValueError("potential-internal structured parameter count drift")
    control_blocks = control_structured["potential_blocks"]
    treatment_blocks = treatment_structured["potential_blocks"]
    coverage = validate_parameter_group_coverage(jax.device_get(control_blocks))

    artifact = _load_artifact(args, reconstruction.context, TARGET_ARTIFACT_UPDATE)
    microbatch = _slice(
        artifact.batch.seven_day,
        TARGET_WINDOW_INDEX,
        TARGET_WINDOW_INDEX + 1,
    )
    trace_evaluator = _make_trace_evaluator(bootstrap)
    control_trace, control_trajectories = trace_evaluator(reconstruction.control_parameters, microbatch)
    jax.block_until_ready((control_trace, control_trajectories))
    control_gpp_trace = np.asarray(
        control_trajectories.fluxes["gpp"].value,
        dtype=np.float64,
    ).reshape(-1)
    control_trace_reproduced = bool(
        np.allclose(
            control_gpp_trace,
            np.asarray(parent["control_gpp_trace"], dtype=np.float64),
            rtol=FORWARD_REPRODUCTION_RTOL,
            atol=FORWARD_REPRODUCTION_ATOL,
        )
    )
    templates, control_starts, control_discrete = _scenario_day_starts(
        reconstruction,
        microbatch,
        control_trace,
    )
    template_batch = _tree_day_batch(jax, templates, TARGET_DAY_INDEX)
    discrete_batch = _tree_day_batch(jax, control_discrete, TARGET_DAY_INDEX)
    control_state = jnp.asarray(control_starts[TARGET_DAY_INDEX])
    pft_index = EXPECTED_VEGETATION_PFT_INDEX
    factor_evaluator = _make_factor_evaluator(reconstruction)

    input_evaluator = _make_potential_input_evaluator(reconstruction)
    batched_inputs = input_evaluator(
        reconstruction.control_parameters,
        template_batch,
        control_state[None, :],
        discrete_batch,
    )
    jax.block_until_ready(batched_inputs)
    branch_inputs = jax.tree_util.tree_map(lambda value: value[0], batched_inputs)
    float32_potential, float32_trace = _make_shadow_evaluators(reconstruction, jnp.float32)
    float64_potential, float64_trace = _make_shadow_evaluators(reconstruction, jnp.float64)

    production_path = []
    shadow_float32_path = []
    full_float64_path = []
    lower_blocks = None
    upper_blocks = None
    lower_blocks_float64 = None
    upper_blocks_float64 = None
    all_factor_records = []
    float32_reproduced = True
    full_float64_finite = True
    for index, fraction in enumerate(INTERPOLATION_FRACTIONS, start=1):
        blocks = _interpolate_tree(
            jax,
            jnp,
            control_blocks,
            treatment_blocks,
            fraction,
            dtype=jnp.float32,
        )
        blocks_float64 = _interpolate_tree(
            jax,
            jnp,
            control_blocks,
            treatment_blocks,
            fraction,
            dtype=jnp.float64,
        )
        record = _evaluate_blocks(
            blocks,
            control_structured=control_structured,
            reconstruction=reconstruction,
            factor_evaluator=factor_evaluator,
            template_batch=template_batch,
            control_state=control_state,
            discrete_batch=discrete_batch,
            pft_index=pft_index,
        )
        shadow_potential = float(
            np.asarray(float32_potential(blocks, control_structured["potential_output"], branch_inputs))[pft_index]
        )
        diagnostic_potential = float(
            np.asarray(
                float64_potential(
                    blocks_float64,
                    control_structured["potential_output"],
                    branch_inputs,
                )
            )[pft_index]
        )
        production_potential = _selected_potential(record, pft_index)
        potential_to_gpp = float(record["gpp"]) / production_potential
        shadow_gpp = potential_to_gpp * shadow_potential
        diagnostic_gpp = potential_to_gpp * diagnostic_potential
        production_path.append({"fraction": fraction, **record})
        shadow_float32_path.append({"fraction": fraction, "potential": shadow_potential, "gpp": shadow_gpp})
        full_float64_path.append({"fraction": fraction, "potential": diagnostic_potential, "gpp": diagnostic_gpp})
        all_factor_records.append(record)
        float32_reproduced &= bool(
            np.isclose(
                shadow_potential,
                production_potential,
                rtol=FACTORIZATION_RTOL,
                atol=FACTORIZATION_ATOL,
            )
            and np.isclose(
                shadow_gpp,
                float(record["gpp"]),
                rtol=FACTORIZATION_RTOL,
                atol=FACTORIZATION_ATOL,
            )
        )
        full_float64_finite &= bool(
            np.isfinite(diagnostic_potential) and np.isfinite(diagnostic_gpp) and diagnostic_gpp >= 0.0
        )
        if fraction == LOWER_FRACTION:
            lower_blocks = blocks
            lower_blocks_float64 = blocks_float64
        if fraction == UPPER_FRACTION:
            upper_blocks = blocks
            upper_blocks_float64 = blocks_float64
        print(
            f"coupled_gpp potential_internal interpolation={index}/{len(INTERPOLATION_FRACTIONS)} "
            f"fraction={fraction:g}",
            flush=True,
        )

    if any(value is None for value in (lower_blocks, upper_blocks, lower_blocks_float64, upper_blocks_float64)):
        raise RuntimeError("potential-internal adjacent parameter trees were not retained")
    production_summary = summarize_gpp_path(
        INTERPOLATION_FRACTIONS,
        tuple(record["gpp"] for record in production_path),
    )
    expected_control_gpp = float(parent["parent_cells"]["CC"])
    expected_potential_only = float(
        parent["subtree_controls"]["groups"]["potential_blocks"]["factorizations"]["treatment_group_only"]["gpp"]
    )
    isolated_path_reproduced = bool(
        np.isclose(
            production_path[0]["gpp"],
            expected_control_gpp,
            rtol=FACTORIZATION_RTOL,
            atol=FACTORIZATION_ATOL,
        )
        and np.isclose(
            production_path[-1]["gpp"],
            expected_potential_only,
            rtol=FACTORIZATION_RTOL,
            atol=FACTORIZATION_ATOL,
        )
        and tuple(production_summary["maximum_adjacent_interval"]) == EXPECTED_MAXIMUM_INTERVAL
    )

    lower_index = INTERPOLATION_FRACTIONS.index(LOWER_FRACTION)
    upper_index = INTERPOLATION_FRACTIONS.index(UPPER_FRACTION)
    lower_gpp = float(production_path[lower_index]["gpp"])
    upper_gpp = float(production_path[upper_index]["gpp"])
    evaluate_kwargs = {
        "control_structured": control_structured,
        "reconstruction": reconstruction,
        "factor_evaluator": factor_evaluator,
        "template_batch": template_batch,
        "control_state": control_state,
        "discrete_batch": discrete_batch,
        "pft_index": pft_index,
    }
    block_only, block_without, block_records, evaluated = _causal_controls(
        BLOCK_PARAMETER_GROUPS,
        lower_blocks,
        upper_blocks,
        **evaluate_kwargs,
    )
    all_factor_records.extend(evaluated)
    component_only, component_without, component_records, evaluated = _causal_controls(
        COMPONENT_PARAMETER_GROUPS,
        lower_blocks,
        upper_blocks,
        **evaluate_kwargs,
    )
    all_factor_records.extend(evaluated)
    parameter_attribution = classify_parameter_localization(
        lower_gpp=lower_gpp,
        upper_gpp=upper_gpp,
        block_upper_only_gpp=block_only,
        block_upper_without_gpp=block_without,
        component_upper_only_gpp=component_only,
        component_upper_without_gpp=component_without,
    )
    parameter_attribution["blocks"]["factorizations"] = block_records
    parameter_attribution["components"]["factorizations"] = component_records

    activation_precision = classify_activation_precision_paths(
        fractions=INTERPOLATION_FRACTIONS,
        production_gpp=tuple(record["gpp"] for record in production_path),
        full_float64_gpp=tuple(record["gpp"] for record in full_float64_path),
    )
    trace_pairs = {}
    trace_finite = True
    trace_dtypes = {}
    for name, trace_function, lower, upper in (
        ("production_equivalent_float32", float32_trace, lower_blocks, upper_blocks),
        (
            "full_float64_potential_branch",
            float64_trace,
            lower_blocks_float64,
            upper_blocks_float64,
        ),
    ):
        first = trace_function(lower, control_structured["potential_output"], branch_inputs)
        second = trace_function(upper, control_structured["potential_output"], branch_inputs)
        jax.block_until_ready((first, second))
        first_host = jax.device_get(first)
        second_host = jax.device_get(second)
        pair = summarize_trace_pair(first_host, second_host, pft_index=pft_index)
        trace_pairs[name] = pair
        trace_finite &= bool(pair["all_stages_finite"])
        trace_dtypes[name] = str(np.asarray(first_host["potential"]).dtype)

    direction_passed = bool(
        direction["combined_gradient_norm_reproduced"] and direction["treatment_parameter_displacement_reproduced"]
    )
    factors_passed = all(_record_is_valid(record) for record in all_factor_records)
    dtype_passed = bool(
        trace_dtypes["production_equivalent_float32"] == "float32"
        and trace_dtypes["full_float64_potential_branch"] == "float64"
    )
    engineering_passed = bool(
        direction_passed
        and control_trace_reproduced
        and isolated_path_reproduced
        and coverage["exact_nonoverlapping_coverage"]
        and factors_passed
        and float32_reproduced
        and full_float64_finite
        and trace_finite
        and dtype_passed
    )
    precision_classification = str(activation_precision["classification"])
    parameter_classification = str(parameter_attribution["classification"])
    if not engineering_passed:
        classification = "engineering_gate_failed"
        next_gate = "inspect_written_engineering_report_before_any_rerun"
    else:
        classification = precision_classification
        if precision_classification == "float32_internal_activation_precision_dominant_for_sampled_jump":
            next_gate = "discuss_numerical_rescaling_or_internal_precision_control"
        elif parameter_classification == "potential_jump_localized_to_internal_parameter_components":
            next_gate = "discuss_bounded_conditioning_control_for_selected_components"
        elif parameter_classification == "potential_jump_requires_distributed_within_block_interaction":
            next_gate = "discuss_block_reconditioning_or_gpp_potential_parameterization"
        else:
            next_gate = "discuss_gpp_potential_parameterization_without_selecting_a_leaf_winner"

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if engineering_passed else "engineering_gate_failed",
        "classification": classification,
        "identity": {
            "git_head": _git_head(),
            "structured_precision_report_sha256": parent_digest,
        },
        "target": {
            "artifact_update": TARGET_ARTIFACT_UPDATE,
            "runtime_year": runtime_year,
            "window_index": TARGET_WINDOW_INDEX,
            "day_index": TARGET_DAY_INDEX,
            "vegetation_pft_index": pft_index,
            "lower_fraction": LOWER_FRACTION,
            "upper_fraction": UPPER_FRACTION,
        },
        "direction_reconstruction": direction,
        "control_gpp_trace": control_gpp_trace.tolist(),
        "parameter_group_coverage": coverage,
        "isolated_potential_path": {
            "production_float32": production_path,
            "production_equivalent_shadow_float32": shadow_float32_path,
            "full_float64_potential_branch": full_float64_path,
            "production_summary": production_summary,
            "expected_control_gpp": expected_control_gpp,
            "expected_potential_block_only_gpp": expected_potential_only,
        },
        "parameter_attribution": parameter_attribution,
        "activation_precision_attribution": activation_precision,
        "adjacent_activation_traces": trace_pairs,
        "integrity": {
            "engineering_gate_passed": engineering_passed,
            "direction_reproduced": direction_passed,
            "control_trace_reproduced": control_trace_reproduced,
            "isolated_potential_path_reproduced": isolated_path_reproduced,
            "parameter_group_coverage_exact": coverage["exact_nonoverlapping_coverage"],
            "all_factor_records_finite_and_gpp_nonnegative": factors_passed,
            "float32_shadow_reproduced": float32_reproduced,
            "full_float64_path_finite_and_nonnegative": full_float64_finite,
            "all_adjacent_trace_stages_finite": trace_finite,
            "shadow_dtypes_exact": dtype_passed,
            "shadow_dtypes": trace_dtypes,
            "factorization_tolerance": {
                "rtol": FACTORIZATION_RTOL,
                "atol": FACTORIZATION_ATOL,
            },
        },
        "policy": _policy_record(),
        "backend": jax.default_backend(),
        "devices": [str(value) for value in runtime["devices"]],
        "memory_after": _memory_report(bootstrap["device"]),
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": next_gate,
    }
    _atomic_json(output, report)
    return report


def _parser() -> Any:
    parser = _precision_parser()
    parser.add_argument("--structured-precision-report", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
