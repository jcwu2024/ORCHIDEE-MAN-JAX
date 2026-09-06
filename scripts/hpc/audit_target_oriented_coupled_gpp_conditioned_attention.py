"""Zero-update real-v6 and pathological-continuity admission for conditioned GPP attention."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.daily_coarse_graining.daily_operator.mechanistic_output_loss import (
    masked_mechanistic_output_loss,
)
from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_conditioned_attention import (
    conditioned_continuity_gate,
    conditioned_coupled_gpp_trainable_tree,
    fixed_reference_attention_temperatures,
    initialize_coupled_gpp_conditioned_attention_parameters,
    load_coupled_gpp_conditioned_attention_contract,
    replace_conditioned_coupled_gpp_trainable,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_objective import (
    coupled_gpp_trainable_tree,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_potential_internal_attribution import (
    LOWER_FRACTION,
    UPPER_FRACTION,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_snowheat_causal import (
    TARGET_ARTIFACT_UPDATE,
    TARGET_DAY_INDEX,
    TARGET_WINDOW_INDEX,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp import (
    GPP_FIELD,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_day5_crossed_attribution import (
    _factor_record,
    _tree_day,
    _tree_day_batch,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_localized_jump import (
    FORWARD_REPRODUCTION_ATOL,
    FORWARD_REPRODUCTION_RTOL,
    _make_trace_evaluator,
    reconstruct_localized_jump_parameters,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_potential_internal_attribution import (
    EXPECTED_RUNTIME_YEAR,
    EXPECTED_VEGETATION_PFT_INDEX,
    _make_potential_input_evaluator,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_snowheat_causal import (
    FACTORIZATION_ATOL,
    FACTORIZATION_RTOL,
    _scenario_day_starts,
)
from scripts.hpc.audit_target_oriented_coupled_gpp_structured_precision_attribution import (
    _interpolate_tree,
    _replace_structured_gpp,
)
from scripts.hpc.calibrate_target_oriented_distribution_aware_guard import _load_artifact
from scripts.hpc.run_strong_baseline_gpu_admission import _masked_state_report, _memory_report
from scripts.hpc.run_successor_fast_owner_gpu_probe import _tree_numeric_report
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_attention_gpu_admission import _masked_output_report
from scripts.hpc.run_target_oriented_coupled_gpp_artifact_aware_persistent_training import (
    _initialization_only,
)
from scripts.hpc.run_target_oriented_coupled_gpp_artifact_aware_persistent_training import (
    _parser as _persistent_parser,
)
from scripts.hpc.run_target_oriented_coupled_gpp_carbon_feedback_gpu_admission import (
    _adapter_gradient_reports,
)
from scripts.hpc.run_target_oriented_coupled_gpp_direct_recalibration import _git_head
from scripts.hpc.run_target_oriented_full_step_preflight import _slice
from scripts.hpc.run_target_oriented_structured_gpp_gpu_admission import (
    _component_gradient_reports,
    _tree_size,
)

SCHEMA_VERSION = "target_oriented_coupled_gpp_conditioned_attention_admission_result_v1"
PARENT_REPORT_SHA256 = "2c8a28d82eec1824163a3796a2777716cca8455a42241591bfa9d21478d94836"
PARENT_SCHEMA_VERSION = "target_oriented_coupled_gpp_attention_head_token_attribution_result_v1"
PARENT_CLASSIFICATION = "head7_qk_pair_and_doc_token_mediate_sampled_jump"
CANDIDATE_PARAMETER_COUNT = 109_464_375
TRAINABLE_PARAMETER_COUNT = 15_960_211
RETAINED_PARAMETER_COUNT = 93_504_164


def _load_parent_report(path: Path) -> tuple[Mapping[str, Any], str]:
    digest = sha256_file(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        digest != PARENT_REPORT_SHA256
        or report.get("schema_version") != PARENT_SCHEMA_VERSION
        or report.get("status") != "passed"
        or report.get("classification") != PARENT_CLASSIFICATION
        or report.get("integrity", {}).get("engineering_gate_passed") is not True
    ):
        raise ValueError("conditioned-attention parent attribution drift")
    return report, digest


def _make_factor_evaluator(reconstruction: Any, head: Any) -> Any:
    jax = reconstruction.jax
    assembler = reconstruction.runtime["adapter"].input_assembler
    contract = reconstruction.runtime["contract"]

    from research.daily_coarse_graining.daily_operator.successor_training import (
        rollout_input_from_template,
    )

    @jax.jit
    def evaluate(parameters: Any, templates: Any, continuous: Any, discrete: Any) -> Any:
        def one(template: Any, state: Any, state_discrete: Any) -> Any:
            current = rollout_input_from_template(template, state, state_discrete, contract)
            assembled = assembler.assemble(current)
            context = head.assemble_context(parameters, assembled)
            return head.gpp_factorization(parameters, context, assembled)

        return jax.vmap(one)(templates, continuous, discrete)

    return evaluate


def _evaluate_factor_record(
    evaluator: Any,
    reconstruction: Any,
    parameters: Any,
    template_batch: Any,
    state: Any,
    discrete_batch: Any,
    pft_index: int,
) -> Mapping[str, Any]:
    factors = evaluator(parameters, template_batch, state[None, :], discrete_batch)
    reconstruction.jax.block_until_ready(factors)
    selected = reconstruction.jax.tree_util.tree_map(lambda value: value[0], factors)
    return _factor_record(reconstruction.jax.device_get(selected), pft_index)


def _make_conditioned_potential_trace(reconstruction: Any, normalization_epsilon: float) -> Any:
    jax = reconstruction.jax
    jnp = reconstruction.jnp

    from research.daily_coarse_graining.daily_operator.strong_baseline_architecture import (
        _ffn,
        _layer_norm,
        _linear,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp import (
        bounded_cosine_attention,
        bounded_cosine_attention_logits,
    )

    heads = int(reconstruction.bootstrap["head"].structured_config.attention_heads)

    @jax.jit
    def evaluate(blocks: Any, output: Any, branch_inputs: Any, temperatures: Any) -> Any:
        encoded = branch_inputs["query"][:, None, :]
        logits = []
        for block_index, block in enumerate(blocks):
            logits.append(
                bounded_cosine_attention_logits(
                    block["attention"],
                    encoded,
                    branch_inputs["bank"],
                    heads=heads,
                    temperatures=temperatures[block_index],
                    normalization_epsilon=normalization_epsilon,
                )
            )
            attention = bounded_cosine_attention(
                block["attention"],
                encoded,
                branch_inputs["bank"],
                heads=heads,
                mask=branch_inputs["bank_mask"][:, None, :],
                temperatures=temperatures[block_index],
                normalization_epsilon=normalization_epsilon,
            )
            encoded = _layer_norm(
                encoded + attention,
                block["attention_norm_scale"],
                block["attention_norm_bias"],
            )
            encoded = _layer_norm(
                encoded + _ffn(block["ffn"], encoded),
                block["ffn_norm_scale"],
                block["ffn_norm_bias"],
            )
        potential = jax.nn.softplus(_linear(output, encoded[:, 0, :])[:, 0])
        potential = jnp.where(branch_inputs["active"], potential, 0.0)
        return {"potential": potential, "logits": tuple(logits)}

    return evaluate


def _potential_gradient_report(jax: Any, gradient: Mapping[str, Any]) -> Mapping[str, Any]:
    blocks = []
    passed = True
    for index, block in enumerate(gradient["potential_blocks"]):
        projections = {
            name: _tree_numeric_report(jax, block["attention"][name]) for name in ("query", "key", "value", "output")
        }
        block_passed = all(value["all_finite"] and value["l2_norm"] > 0.0 for value in projections.values())
        passed &= block_passed
        blocks.append({"block": index, "projections": projections, "passed": block_passed})
    return {"blocks": blocks, "passed": bool(passed)}


def _one_day_admission(
    reconstruction: Any,
    head: Any,
    reference_head: Any,
    parameters: Any,
    artifact: Any,
) -> Mapping[str, Any]:
    jax = reconstruction.jax
    runtime = reconstruction.runtime
    context = reconstruction.context

    from research.daily_coarse_graining.daily_operator.model import (
        bind_daily_operator_dynamic_retained_inputs,
    )

    one = artifact.batch.one_day
    inputs = _tree_day(jax, one.inputs, 0)
    retained_inputs = _tree_day(jax, one.day_inputs, 0)
    targets = _tree_day(jax, one.process_targets, 0)
    assembled = runtime["adapter"].input_assembler.assemble(inputs)
    candidate_context = head.assemble_context(parameters, assembled)
    candidate_outputs = head.learned_outputs(parameters, candidate_context, assembled)
    reference_outputs = reference_head.learned_outputs(
        parameters,
        reference_head.assemble_context(parameters, assembled),
        assembled,
    )
    parent_parameters = reconstruction.bootstrap["initialized_parameters"]
    parent_head = reconstruction.bootstrap["head"]
    parent_outputs = parent_head.learned_outputs(
        parent_parameters,
        parent_head.assemble_context(parent_parameters, assembled),
        assembled,
    )
    jax.block_until_ready((candidate_outputs, reference_outputs, parent_outputs))
    non_gpp_equal = bool(
        all(
            np.array_equal(np.asarray(value), np.asarray(parent_outputs[name]))
            for name, value in candidate_outputs.items()
            if name != GPP_FIELD
        )
    )
    reference_match = bool(
        np.allclose(
            np.asarray(candidate_outputs[GPP_FIELD]),
            np.asarray(reference_outputs[GPP_FIELD]),
            rtol=1.0e-6,
            atol=1.0e-8,
        )
    )

    transition = bind_daily_operator_dynamic_retained_inputs(
        input_assembler=runtime["adapter"].input_assembler,
        process_head=head,
        retained_tail_template=context["template"],
    )
    compiled_transition = jax.jit(transition)
    forward_started = time.perf_counter()
    result = compiled_transition(parameters, inputs, retained_inputs)
    jax.block_until_ready(result)
    forward_seconds = time.perf_counter() - forward_started
    output_report = _masked_output_report(result.prediction.named_outputs, targets)
    state_report = _masked_state_report(result.next_state, result.next_state_defined)
    water_residual = float(result.fast_update.water.budget_residual)
    carbon_residual = float(result.fast_update.carbon.budget_residual)

    trainable = conditioned_coupled_gpp_trainable_tree(parameters)
    fixed = replace_conditioned_coupled_gpp_trainable(parameters, {})

    def loss_function(trainable_values: Any) -> Any:
        values = replace_conditioned_coupled_gpp_trainable(fixed, trainable_values)
        current = transition(values, inputs, retained_inputs)
        loss = masked_mechanistic_output_loss(
            current.prediction.named_outputs,
            targets,
            inputs=runtime["adapter"].input_assembler.assemble(inputs),
            complete_schema=runtime["schema"],
            learned_schema=runtime["learned_schema"],
            scales=runtime["loss_scales"],
            architecture_contract=runtime["architecture_contract"],
            mechanism_contract=runtime["mechanism_contract"],
            auxiliary_outputs=current.prediction.auxiliary_outputs,
        )
        return loss.total

    gradient_started = time.perf_counter()
    loss_value, gradient = jax.jit(jax.value_and_grad(loss_function))(trainable)
    jax.block_until_ready((loss_value, gradient))
    gradient_seconds = time.perf_counter() - gradient_started
    gradient_report = _tree_numeric_report(jax, gradient)
    structured_report = _component_gradient_reports(jax, gradient["structured_gpp"])
    adapter_report = _adapter_gradient_reports(jax, gradient["coupled_gpp_carbon_feedback"])
    temperature_report = _tree_numeric_report(jax, gradient["coupled_gpp_conditioning"])
    temperature_values = np.asarray(
        jax.device_get(gradient["coupled_gpp_conditioning"]["potential_temperature_logits"]),
        dtype=np.float64,
    )
    temperature_blocks_nonzero = bool(np.all(np.linalg.norm(temperature_values, axis=1) > 0.0))
    potential_report = _potential_gradient_report(jax, gradient["structured_gpp"])
    components_finite = bool(
        all(value["all_finite"] for value in (*structured_report.values(), *adapter_report.values()))
    )
    passed = bool(
        non_gpp_equal
        and reference_match
        and output_report["all_declared_defined_finite"]
        and state_report["all_declared_defined_finite"]
        and np.isfinite(float(loss_value))
        and gradient_report["all_finite"]
        and gradient_report["element_count"] == TRAINABLE_PARAMETER_COUNT
        and gradient_report["l2_norm"] > 0.0
        and components_finite
        and temperature_report["all_finite"]
        and temperature_blocks_nonzero
        and potential_report["passed"]
        and abs(water_residual) <= 2.0e-12
        and abs(carbon_residual) <= 2.0e-12
    )
    return {
        "passed": passed,
        "artifact_update": TARGET_ARTIFACT_UPDATE,
        "sample_ordinal": 0,
        "forward_compile_and_run_seconds": forward_seconds,
        "loss_and_gradient_compile_and_run_seconds": gradient_seconds,
        "complete_loss": float(loss_value),
        "candidate_and_fixed_reference_match": reference_match,
        "non_gpp_parent_outputs_bit_identical": non_gpp_equal,
        "output": output_report,
        "next_state": state_report,
        "water_budget_residual": water_residual,
        "carbon_budget_residual": carbon_residual,
        "gradient": gradient_report
        | {
            "structured_components": structured_report,
            "carbon_feedback_components": adapter_report,
            "temperature": temperature_report,
            "temperature_block_l2": np.linalg.norm(temperature_values, axis=1).tolist(),
            "potential_attention": potential_report,
        },
    }


def run(args: Any) -> Mapping[str, Any]:
    started = time.perf_counter()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)

    parent, parent_digest = _load_parent_report(args.head_token_report.resolve())
    conditioning_contract = load_coupled_gpp_conditioned_attention_contract(args.conditioning_contract)
    bootstrap = _initialization_only(args)
    reconstruction = reconstruct_localized_jump_parameters(args, bootstrap=bootstrap)
    jax = reconstruction.jax
    jnp = reconstruction.jnp
    parent_head = bootstrap["head"]
    runtime_year = int(reconstruction.context["schedule"]["entries"][TARGET_ARTIFACT_UPDATE - 1]["runtime_year"])
    if (
        parent_head.coupled_contract.sha256
        != conditioning_contract.payload["identity"]["parent_candidate_contract_sha256"]
        or runtime_year != EXPECTED_RUNTIME_YEAR
    ):
        raise ValueError("conditioned-attention coupled parent contract drift")

    from research.daily_coarse_graining.daily_operator.target_oriented_process_head import (
        build_target_oriented_coupled_gpp_conditioned_attention_head,
    )

    head = build_target_oriented_coupled_gpp_conditioned_attention_head(
        parent_head,
        conditioning_contract=conditioning_contract,
    )
    reference_head = build_target_oriented_coupled_gpp_conditioned_attention_head(
        parent_head,
        conditioning_contract=conditioning_contract,
        fixed_temperature_reference=True,
    )
    initialized_parameters = initialize_coupled_gpp_conditioned_attention_parameters(
        bootstrap["initialized_parameters"],
        conditioning_contract=conditioning_contract,
        structured_config=head.structured_config,
        config=head.conditioning_config,
    )
    trainable = conditioned_coupled_gpp_trainable_tree(initialized_parameters)
    fixed = replace_conditioned_coupled_gpp_trainable(initialized_parameters, {})
    if (
        _tree_size(jax, initialized_parameters) != CANDIDATE_PARAMETER_COUNT
        or _tree_size(jax, trainable) != TRAINABLE_PARAMETER_COUNT
        or _tree_size(jax, fixed) != RETAINED_PARAMETER_COUNT
    ):
        raise ValueError("conditioned-attention parameter boundary drift")

    control_structured = coupled_gpp_trainable_tree(reconstruction.control_parameters)["structured_gpp"]
    treatment_structured = coupled_gpp_trainable_tree(reconstruction.treatment_parameters)["structured_gpp"]
    lower_blocks = _interpolate_tree(
        jax,
        jnp,
        control_structured["potential_blocks"],
        treatment_structured["potential_blocks"],
        LOWER_FRACTION,
        dtype=jnp.float32,
    )
    upper_blocks = _interpolate_tree(
        jax,
        jnp,
        control_structured["potential_blocks"],
        treatment_structured["potential_blocks"],
        UPPER_FRACTION,
        dtype=jnp.float32,
    )
    lower_parent_parameters = _replace_structured_gpp(
        reconstruction.control_parameters,
        dict(control_structured, potential_blocks=lower_blocks),
    )
    upper_parent_parameters = _replace_structured_gpp(
        reconstruction.control_parameters,
        dict(control_structured, potential_blocks=upper_blocks),
    )
    lower_parameters = initialize_coupled_gpp_conditioned_attention_parameters(
        lower_parent_parameters,
        conditioning_contract=conditioning_contract,
        structured_config=head.structured_config,
        config=head.conditioning_config,
    )
    upper_parameters = initialize_coupled_gpp_conditioned_attention_parameters(
        upper_parent_parameters,
        conditioning_contract=conditioning_contract,
        structured_config=head.structured_config,
        config=head.conditioning_config,
    )

    artifact = _load_artifact(args, reconstruction.context, TARGET_ARTIFACT_UPDATE)
    microbatch = _slice(artifact.batch.seven_day, TARGET_WINDOW_INDEX, TARGET_WINDOW_INDEX + 1)
    trace_evaluator = _make_trace_evaluator(bootstrap)
    control_trace, control_trajectories = trace_evaluator(reconstruction.control_parameters, microbatch)
    jax.block_until_ready((control_trace, control_trajectories))
    control_gpp_trace = np.asarray(control_trajectories.fluxes["gpp"].value, dtype=np.float64).reshape(-1)
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

    parent_evaluator = _make_factor_evaluator(reconstruction, parent_head)
    candidate_evaluator = _make_factor_evaluator(reconstruction, head)
    reference_evaluator = _make_factor_evaluator(reconstruction, reference_head)
    lower_parent = _evaluate_factor_record(
        parent_evaluator,
        reconstruction,
        lower_parent_parameters,
        template_batch,
        control_state,
        discrete_batch,
        pft_index,
    )
    upper_parent = _evaluate_factor_record(
        parent_evaluator,
        reconstruction,
        upper_parent_parameters,
        template_batch,
        control_state,
        discrete_batch,
        pft_index,
    )
    lower_candidate = _evaluate_factor_record(
        candidate_evaluator,
        reconstruction,
        lower_parameters,
        template_batch,
        control_state,
        discrete_batch,
        pft_index,
    )
    upper_candidate = _evaluate_factor_record(
        candidate_evaluator,
        reconstruction,
        upper_parameters,
        template_batch,
        control_state,
        discrete_batch,
        pft_index,
    )
    lower_reference = _evaluate_factor_record(
        reference_evaluator,
        reconstruction,
        lower_parameters,
        template_batch,
        control_state,
        discrete_batch,
        pft_index,
    )
    upper_reference = _evaluate_factor_record(
        reference_evaluator,
        reconstruction,
        upper_parameters,
        template_batch,
        control_state,
        discrete_batch,
        pft_index,
    )
    expected_lower = float(parent["endpoints"]["lower_factorization"]["gpp"])
    expected_upper = float(parent["endpoints"]["upper_factorization"]["gpp"])
    parent_endpoints_reproduced = bool(
        np.isclose(lower_parent["gpp"], expected_lower, rtol=FACTORIZATION_RTOL, atol=FACTORIZATION_ATOL)
        and np.isclose(upper_parent["gpp"], expected_upper, rtol=FACTORIZATION_RTOL, atol=FACTORIZATION_ATOL)
    )

    input_evaluator = _make_potential_input_evaluator(reconstruction)
    batched_inputs = input_evaluator(
        reconstruction.control_parameters,
        template_batch,
        control_state[None, :],
        discrete_batch,
    )
    jax.block_until_ready(batched_inputs)
    branch_inputs = jax.tree_util.tree_map(lambda value: value[0], batched_inputs)
    temperatures = head.potential_attention_temperatures(lower_parameters)
    conditioned_trace = _make_conditioned_potential_trace(
        reconstruction,
        head.conditioning_config.normalization_epsilon,
    )
    lower_trace = conditioned_trace(
        lower_blocks,
        control_structured["potential_output"],
        branch_inputs,
        temperatures,
    )
    upper_trace = conditioned_trace(
        upper_blocks,
        control_structured["potential_output"],
        branch_inputs,
        temperatures,
    )
    jax.block_until_ready((lower_trace, upper_trace))
    trace_host = jax.device_get((lower_trace, upper_trace))
    maximum_absolute_logit = float(
        max(np.max(np.abs(np.asarray(value, dtype=np.float64))) for trace in trace_host for value in trace["logits"])
    )
    trace_matches = bool(
        np.isclose(
            np.asarray(trace_host[0]["potential"])[pft_index],
            np.asarray(lower_candidate["potential"])[pft_index],
            rtol=FACTORIZATION_RTOL,
            atol=FACTORIZATION_ATOL,
        )
        and np.isclose(
            np.asarray(trace_host[1]["potential"])[pft_index],
            np.asarray(upper_candidate["potential"])[pft_index],
            rtol=FACTORIZATION_RTOL,
            atol=FACTORIZATION_ATOL,
        )
    )
    continuity_spec = conditioning_contract.payload["pathological_continuity_gate"]
    continuity = conditioned_continuity_gate(
        parent_lower_gpp=lower_parent["gpp"],
        parent_upper_gpp=upper_parent["gpp"],
        candidate_lower_gpp=lower_candidate["gpp"],
        candidate_upper_gpp=upper_candidate["gpp"],
        reference_lower_gpp=lower_reference["gpp"],
        reference_upper_gpp=upper_reference["gpp"],
        maximum_absolute_logit=maximum_absolute_logit,
        jump_fraction_maximum=float(continuity_spec["conditioned_jump_fraction_of_parent_maximum"]),
        logit_absolute_maximum=float(
            conditioning_contract.payload["local_acceptance"]["conditioned_logits_absolute_maximum"]
        ),
        reference_rtol=float(continuity_spec["candidate_and_fixed_reference_endpoint_match_rtol"]),
        reference_atol=float(continuity_spec["candidate_and_fixed_reference_endpoint_match_atol"]),
    )
    continuity_passed = bool(
        control_trace_reproduced and parent_endpoints_reproduced and trace_matches and continuity["passed"]
    )

    one_day = _one_day_admission(
        reconstruction,
        head,
        reference_head,
        initialized_parameters,
        artifact,
    )
    engineering_passed = bool(continuity_passed and one_day["passed"])
    classification = (
        "conditioned_attention_removes_known_qk_scale_jump"
        if engineering_passed
        else "conditioned_attention_zero_update_gate_failed"
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if engineering_passed else "failed",
        "classification": classification,
        "identity": {
            "git_head": _git_head(),
            "conditioning_contract_sha256": conditioning_contract.sha256,
            "parent_head_token_report_sha256": parent_digest,
            "coupled_parent_contract_sha256": parent_head.coupled_contract.sha256,
        },
        "policy": {
            "training_updates": 0,
            "optimizer_created": False,
            "checkpoint_written": False,
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "fresh_validation_values_evaluated": False,
            "sealed_test_read": False,
            "forcing_records_reconstructed": 0,
            "subdaily_state_scan_steps": 0,
        },
        "model": {
            "candidate": conditioning_contract.payload["identity"]["candidate"],
            "parameter_count": _tree_size(jax, initialized_parameters),
            "trainable_parameter_count": _tree_size(jax, trainable),
            "retained_parameter_count": _tree_size(jax, fixed),
            "temperature_parameter_count": _tree_size(
                jax,
                trainable["coupled_gpp_conditioning"],
            ),
            "initial_temperatures": np.asarray(jax.device_get(temperatures), dtype=np.float64).tolist(),
            "fixed_reference_temperatures": np.asarray(
                fixed_reference_attention_temperatures(config=head.conditioning_config),
                dtype=np.float64,
            ).tolist(),
        },
        "pathological_continuity": {
            "target": dict(parent["target"]),
            "control_gpp_trace": control_gpp_trace.tolist(),
            "parent_lower": lower_parent,
            "parent_upper": upper_parent,
            "candidate_lower": lower_candidate,
            "candidate_upper": upper_candidate,
            "fixed_reference_lower": lower_reference,
            "fixed_reference_upper": upper_reference,
            "gate": continuity,
            "conditioned_trace_matches_production": trace_matches,
        },
        "real_v6_one_day": one_day,
        "integrity": {
            "engineering_gate_passed": engineering_passed,
            "control_trace_reproduced": control_trace_reproduced,
            "parent_endpoints_reproduced": parent_endpoints_reproduced,
            "conditioned_trace_matches_production": trace_matches,
            "pathological_continuity_passed": continuity_passed,
            "real_v6_one_day_passed": one_day["passed"],
        },
        "backend": jax.default_backend(),
        "devices": [str(value) for value in reconstruction.runtime["devices"]],
        "memory_after": _memory_report(bootstrap["device"]),
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": (
            "freeze_discarded_one_update_conditioned_attention_preflight"
            if engineering_passed
            else "inspect_conditioned_attention_zero_update_failure_before_training"
        ),
    }
    _atomic_json(output, report)
    return report


def _parser() -> Any:
    parser = _persistent_parser()
    parser.add_argument("--parent-persistent-training-report", type=Path, required=True)
    parser.add_argument("--parent-rejected-proposal", type=Path, required=True)
    parser.add_argument("--parent-artifact-aware-recovery", type=Path, required=True)
    parser.add_argument("--head-token-report", type=Path, required=True)
    parser.add_argument("--conditioning-contract", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
