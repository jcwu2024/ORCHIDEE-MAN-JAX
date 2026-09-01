"""Zero-update four-source objective calibration for the coupled GPP head."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import (
    sha256_file,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_objective import (
    DIRECT_SOURCE,
    GRADIENT_SOURCES,
    RETAINED_PARAMETER_COUNT,
    SEVEN_SOURCE,
    STATE_SOURCE,
    THIRTY_SOURCE,
    TRAINABLE_PARAMETER_COUNT,
    add_reachable_state_anchor_statistics,
    build_reachable_state_anchor_layout,
    combined_gradient_p90_clip,
    coupled_gpp_trainable_tree,
    field_grouped_state_anchor_from_sufficient_statistics,
    linearized_reachable_state_anchor_loss,
    load_coupled_gpp_objective_calibration_contract,
    reachable_state_anchor_sufficient_statistics,
    replace_coupled_gpp_trainable,
    resolve_coupled_gpp_objective_coefficients,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp import (
    GPP_FIELD,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_training import (
    linearized_structured_gpp_direct_loss,
    structured_gpp_direct_loss,
    structured_gpp_direct_statistics,
)
from scripts.hpc.audit_target_oriented_flux_gradient_variance import (
    _add_gradient,
    _load_context,
    _tree_size,
    _zero_gradient,
)
from scripts.hpc.audit_target_oriented_gpp_state_reachability import (
    _build_candidate,
)
from scripts.hpc.calibrate_target_oriented_distribution_aware_guard import (
    _load_artifact,
)
from scripts.hpc.calibrate_target_oriented_structured_gpp_joint_gradients import (
    _horizon_gradient,
    _science_record,
    _tree_cosine,
)
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_fast_owner_gpu_probe import _tree_numeric_report
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_full_step_preflight import (
    _hard_failure_count,
    _make_steps,
    _slice,
)
from scripts.hpc.run_target_oriented_structured_gpp_training_preflight import (
    _direct_gradient,
    _hard_host,
    _one_day_hard,
    _retained_parameter_report,
)

SCHEMA_VERSION = "target_oriented_coupled_gpp_objective_calibration_result_v1"


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _make_direct_steps(
    runtime: Mapping[str, Any],
    head: Any,
    fixed_parameters: Mapping[str, Any],
) -> Mapping[str, Any]:
    jax = runtime["jax"]
    jnp = runtime["jnp"]
    fixed_parameters = replace_coupled_gpp_trainable(fixed_parameters, {})

    from research.daily_coarse_graining.daily_operator.parameter_routing import (
        PFT_TRAIT_BARE_SOIL,
    )

    scale = jnp.asarray(
        runtime["process_statistics"].outputs[GPP_FIELD].scale,
        dtype=jnp.float64,
    )

    def predicted_gpp(parameters: Any, inputs: Any) -> Any:
        assembled = runtime["adapter"].input_assembler.assemble(inputs)
        context = head.assemble_context(parameters, assembled)
        return head.learned_outputs(parameters, context, assembled)[GPP_FIELD]

    def active_vegetation(inputs: Any) -> Any:
        active = jnp.asarray(inputs.pft.active_pft_mask, dtype=jnp.bool_)
        fraction = jnp.asarray(inputs.pft.pft_fraction, dtype=jnp.float64)
        traits = jnp.asarray(inputs.pft.pft_traits, dtype=jnp.float64)
        return active & (fraction > 0.0) & (traits[..., PFT_TRAIT_BARE_SOIL] <= 0.5)

    one_predict = jax.vmap(predicted_gpp, in_axes=(None, 0))
    window_predict = jax.vmap(
        jax.vmap(predicted_gpp, in_axes=(None, 0)),
        in_axes=(None, 0),
    )

    def one_statistics(trainable: Any, fixed: Any, batch: Any) -> Any:
        parameters = replace_coupled_gpp_trainable(fixed, trainable)
        predicted = one_predict(parameters, batch.inputs)
        target = batch.process_targets[GPP_FIELD]
        active = active_vegetation(batch.inputs)[..., None, :]
        return structured_gpp_direct_statistics(
            predicted,
            target.value,
            target.defined,
            scale,
            active,
        )

    def window_statistics(trainable: Any, fixed: Any, batch: Any) -> Any:
        parameters = replace_coupled_gpp_trainable(fixed, trainable)
        predicted = window_predict(parameters, batch.inputs)
        active = active_vegetation(batch.inputs)[..., None, :]
        return structured_gpp_direct_statistics(
            predicted,
            batch.expected_fast_gpp,
            batch.expected_fast_gpp_defined,
            scale,
            active,
        )

    one_forward = jax.jit(one_statistics)
    window_forward = jax.jit(window_statistics)
    one_gradient = jax.jit(
        jax.value_and_grad(
            lambda trainable, fixed, batch, total: linearized_structured_gpp_direct_loss(
                one_statistics(trainable, fixed, batch),
                total,
            ),
            argnums=0,
        )
    )
    window_gradient = jax.jit(
        jax.value_and_grad(
            lambda trainable, fixed, batch, total: linearized_structured_gpp_direct_loss(
                window_statistics(trainable, fixed, batch),
                total,
            ),
            argnums=0,
        )
    )
    return {
        "one_forward": lambda trainable, batch: one_forward(
            trainable,
            fixed_parameters,
            batch,
        ),
        "window_forward": lambda trainable, batch: window_forward(
            trainable,
            fixed_parameters,
            batch,
        ),
        "one_gradient": lambda trainable, batch, total: one_gradient(
            trainable,
            fixed_parameters,
            batch,
            total,
        ),
        "window_gradient": lambda trainable, batch, total: window_gradient(
            trainable,
            fixed_parameters,
            batch,
            total,
        ),
    }


def _make_state_steps(
    runtime: Mapping[str, Any],
    transition: Any,
    fixed_parameters: Mapping[str, Any],
    state_scale: Any,
    layout: Any,
) -> Mapping[str, Any]:
    jax = runtime["jax"]
    fixed_parameters = replace_coupled_gpp_trainable(fixed_parameters, {})

    def statistics(trainable: Any, fixed: Any, batch: Any) -> Any:
        parameters = replace_coupled_gpp_trainable(fixed, trainable)
        result = jax.vmap(transition, in_axes=(None, 0, 0))(
            parameters,
            batch.inputs,
            batch.day_inputs,
        )
        return reachable_state_anchor_sufficient_statistics(
            result.next_state.other,
            batch.expected_next_state.other,
            result.next_state_defined.other,
            batch.expected_next_state_defined.other,
            state_scale,
            layout=layout,
        )

    forward = jax.jit(statistics)
    gradient = jax.jit(
        jax.value_and_grad(
            lambda trainable, fixed, batch, total: linearized_reachable_state_anchor_loss(
                statistics(trainable, fixed, batch),
                total,
                field_names=layout.field_names,
            ),
            argnums=0,
        )
    )
    return {
        "forward": lambda trainable, batch: forward(
            trainable,
            fixed_parameters,
            batch,
        ),
        "gradient": lambda trainable, batch, total: gradient(
            trainable,
            fixed_parameters,
            batch,
            total,
        ),
    }


def _state_gradient(
    jax: Any,
    jnp: Any,
    trainable: Any,
    batch: Any,
    state_steps: Mapping[str, Any],
    layout: Any,
) -> tuple[Any, Any, float, float]:
    microbatches = tuple(_slice(batch.one_day, start, start + 4) for start in range(0, 32, 4))
    started = time.perf_counter()
    values = [state_steps["forward"](trainable, item) for item in microbatches]
    jax.block_until_ready(values)
    total = values[0]
    for value in values[1:]:
        total = add_reachable_state_anchor_statistics(total, value)
    result = field_grouped_state_anchor_from_sufficient_statistics(
        total,
        field_names=layout.field_names,
    )
    jax.block_until_ready(result)
    forward_seconds = time.perf_counter() - started

    started = time.perf_counter()
    gradient = _zero_gradient(jax, jnp, trainable)
    for item in microbatches:
        _value, current = state_steps["gradient"](trainable, item, total)
        gradient = _add_gradient(jax, jnp, gradient, current)
    jax.block_until_ready(gradient)
    return result, gradient, forward_seconds, time.perf_counter() - started


def _combine_host_gradients(
    jax: Any,
    gradients: Mapping[str, Any],
    coefficients: Mapping[str, float],
) -> Any:
    return jax.tree_util.tree_map(
        lambda *values: sum(
            (
                np.float32(coefficients[name]) * np.asarray(value, dtype=np.float32)
                for name, value in zip(GRADIENT_SOURCES, values, strict=True)
            ),
            np.float32(0.0),
        ),
        *(gradients[name] for name in GRADIENT_SOURCES),
    )


def _state_record(result: Any) -> Mapping[str, Any]:
    return {
        "total": float(np.asarray(result.total)),
        "field_loss": {name: float(np.asarray(value)) for name, value in result.field_loss.items()},
        "field_count": {name: int(np.asarray(value)) for name, value in result.field_count.items()},
    }


def _validate_reachability_report(
    path: Path,
    contract: Any,
    layout: Any,
) -> Mapping[str, Any]:
    expected_hash = contract.payload["identity"]["reachability_report_sha256"]
    report = json.loads(path.read_text(encoding="utf-8"))
    reachable = report.get("reachability", {})
    observed = tuple(
        (
            str(record["field"]),
            tuple(int(value) for value in record["reachable_global_state_indices"]),
        )
        for record in reachable.get("reachable_fields", ())
    )
    expected = tuple(
        (
            name,
            tuple(int(value) for value in layout.global_state_indices[layout.field_index == field]),
        )
        for field, name in enumerate(layout.field_names)
    )
    if (
        sha256_file(path) != expected_hash
        or report.get("status") != "passed"
        or report.get("schema_version") != "target_oriented_gpp_state_reachability_result_v1"
        or observed != expected
    ):
        raise ValueError("coupled GPP objective calibration reachability identity drift")
    return report


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()

    objective_contract = load_coupled_gpp_objective_calibration_contract(args.objective_calibration_contract)
    layout = build_reachable_state_anchor_layout(objective_contract)
    reachability_report = _validate_reachability_report(
        args.reachability_report,
        objective_contract,
        layout,
    )

    from research.daily_coarse_graining.daily_operator.model import (
        bind_daily_operator_dynamic_retained_inputs,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_flux_basis import (
        load_target_oriented_flux_basis_amendment,
    )

    amendment = load_target_oriented_flux_basis_amendment(args.flux_basis_amendment)
    identity = objective_contract.payload["identity"]
    if amendment.sha256 != identity["flux_basis_amendment_sha256"]:
        raise ValueError("coupled GPP objective calibration flux-basis drift")
    context = dict(
        _load_context(
            args,
            science_flux_basis=amendment.flux_basis,
            verify_validation_artifacts=False,
        )
    )
    runtime = context["runtime"]
    jax = runtime["jax"]
    jnp = runtime["jnp"]
    if (
        sha256_file(args.prepared_manifest) != identity["prepared_manifest_sha256"]
        or context["training_contract"].sha256 != identity["base_training_contract_sha256"]
        or context["target_contract"].sha256 != identity["target_contract_sha256"]
    ):
        raise ValueError("coupled GPP objective calibration data identity drift")

    candidate = _build_candidate(runtime, args, context["template"])
    parameters = candidate["parameters"]
    head = candidate["head"]
    if (
        candidate["coupled_contract"].sha256 != identity["coupled_candidate_contract_sha256"]
        or sha256_file(args.selected_checkpoint) != identity["structured_checkpoint_sha256"]
        or sha256_file(args.gpu_admission_report) != identity["real_v6_admission_report_sha256"]
        or _tree_size(jax, parameters) != int(identity["candidate_parameter_count"])
    ):
        raise ValueError("coupled GPP objective calibration candidate identity drift")

    trainable = coupled_gpp_trainable_tree(parameters)
    fixed_parameters = replace_coupled_gpp_trainable(parameters, {})
    if (
        _tree_size(jax, trainable) != TRAINABLE_PARAMETER_COUNT
        or _tree_size(jax, fixed_parameters) != RETAINED_PARAMETER_COUNT
    ):
        raise ValueError("coupled GPP objective calibration parameter boundary drift")

    steps = _make_steps(
        runtime,
        head,
        context["template"],
        context["scales"],
        context["target_contract"],
        science_flux_basis=amendment.flux_basis,
    )
    gradient_steps = steps["make_trainable_gradient_steps"](
        fixed_parameters,
        replace_coupled_gpp_trainable,
    )
    direct_steps = _make_direct_steps(runtime, head, parameters)
    transition = bind_daily_operator_dynamic_retained_inputs(
        input_assembler=runtime["adapter"].input_assembler,
        process_head=head,
        retained_tail_template=context["template"],
    )
    state_steps = _make_state_steps(
        runtime,
        transition,
        parameters,
        context["scales"].state.other,
        layout,
    )
    memory_before = _memory_report(runtime["devices"][0])

    records: list[dict[str, Any]] = []
    stored_gradients: list[Mapping[str, Any]] = []
    for update in objective_contract.calibration_updates:
        artifact = _load_artifact(args, context, update)
        record_started = time.perf_counter()
        direct, direct_gradient, direct_forward, direct_backward = _direct_gradient(
            jax,
            jnp,
            trainable,
            artifact.batch,
            direct_steps,
        )
        state, state_gradient, state_forward, state_backward = _state_gradient(
            jax,
            jnp,
            trainable,
            artifact.batch,
            state_steps,
            layout,
        )
        one_hard = _one_day_hard(jax, jnp, parameters, artifact.batch, steps)
        seven, seven_hard, seven_gradient, seven_timing = _horizon_gradient(
            jax,
            jnp,
            parameters,
            trainable,
            artifact.batch,
            steps,
            gradient_steps,
            horizon=7,
            flux_basis=amendment.flux_basis,
        )
        thirty, thirty_hard, thirty_gradient, thirty_timing = _horizon_gradient(
            jax,
            jnp,
            parameters,
            trainable,
            artifact.batch,
            steps,
            gradient_steps,
            horizon=30,
            flux_basis=amendment.flux_basis,
        )
        host_gradients = jax.device_get(
            {
                DIRECT_SOURCE: direct_gradient,
                STATE_SOURCE: state_gradient,
                SEVEN_SOURCE: seven_gradient,
                THIRTY_SOURCE: thirty_gradient,
            }
        )
        source_reports = {name: _tree_numeric_report(jax, host_gradients[name]) for name in GRADIENT_SOURCES}
        pairwise = {
            f"{left}_vs_{right}": _tree_cosine(
                jax,
                host_gradients[left],
                host_gradients[right],
            )
            for left, right in combinations(GRADIENT_SOURCES, 2)
        }
        records.append(
            {
                "artifact_update": update,
                "runtime_year": int(context["schedule"]["entries"][update - 1]["runtime_year"]),
                "objectives": {
                    DIRECT_SOURCE: {
                        "total": float(np.asarray(structured_gpp_direct_loss(direct))),
                        "evaluated_values": int(np.asarray(direct.count)),
                    },
                    STATE_SOURCE: _state_record(state),
                    SEVEN_SOURCE: _science_record(seven),
                    THIRTY_SOURCE: _science_record(thirty),
                },
                "source_gradients": source_reports,
                "pairwise_gradient_cosines": pairwise,
                "hard_counts": {
                    "one_day": _hard_host(jax.device_get(one_hard)),
                    "seven_day": _hard_host(jax.device_get(seven_hard)),
                    "thirty_day": _hard_host(jax.device_get(thirty_hard)),
                },
                "timing_seconds": {
                    "direct_forward": direct_forward,
                    "direct_gradient": direct_backward,
                    "state_forward": state_forward,
                    "state_gradient": state_backward,
                    "seven_day": seven_timing,
                    "thirty_day": thirty_timing,
                    "total": time.perf_counter() - record_started,
                },
            }
        )
        stored_gradients.append(host_gradients)
        print(
            "coupled_gpp objective_calibration "
            f"artifact={update} "
            + " ".join(f"{name}={source_reports[name]['l2_norm']:.9g}" for name in GRADIENT_SOURCES)
            + f" seconds={records[-1]['timing_seconds']['total']:.3f}",
            flush=True,
        )
        del direct_gradient, state_gradient, seven_gradient, thirty_gradient

    coefficients = resolve_coupled_gpp_objective_coefficients(
        records,
        objective_contract,
    )
    combined_norms = []
    for record, gradients in zip(records, stored_gradients, strict=True):
        combined = _combine_host_gradients(jax, gradients, coefficients.resolved)
        report = _tree_numeric_report(jax, combined)
        record["combined_gradient"] = report
        record["source_to_combined_cosines"] = {
            name: _tree_cosine(jax, gradients[name], combined) for name in GRADIENT_SOURCES
        }
        combined_norms.append(float(report["l2_norm"]))
        del combined
    stored_gradients.clear()

    calibration = objective_contract.payload["coefficient_calibration"]
    clip_threshold = combined_gradient_p90_clip(
        combined_norms,
        minimum=float(calibration["combined_gradient_clip_minimum"]),
    )
    source_passed = all(
        report["all_finite"]
        and report["element_count"] == TRAINABLE_PARAMETER_COUNT
        and report["l2_norm"] > 0.0
        and report["nonzero_leaf_count"] == report["leaf_count"]
        for record in records
        for report in record["source_gradients"].values()
    )
    combined_passed = all(
        record["combined_gradient"]["all_finite"]
        and record["combined_gradient"]["element_count"] == TRAINABLE_PARAMETER_COUNT
        and record["combined_gradient"]["l2_norm"] > 0.0
        and record["combined_gradient"]["nonzero_leaf_count"] == record["combined_gradient"]["leaf_count"]
        for record in records
    )
    objective_passed = all(
        np.isfinite(float(objective["total"])) and float(objective["total"]) > 0.0
        for record in records
        for objective in record["objectives"].values()
    ) and all(record["objectives"][DIRECT_SOURCE]["evaluated_values"] == 208 for record in records)
    fields_passed = all(
        set(record["objectives"][STATE_SOURCE]["field_count"]) == set(layout.field_names)
        and all(int(value) > 0 for value in record["objectives"][STATE_SOURCE]["field_count"].values())
        for record in records
    )
    hard_passed = all(_hard_failure_count(value) == 0 for record in records for value in record["hard_counts"].values())
    retained = _retained_parameter_report(
        jax,
        fixed_parameters,
        fixed_parameters,
    )
    checkpoint_unchanged = sha256_file(args.selected_checkpoint) == identity["structured_checkpoint_sha256"]
    passed = bool(
        source_passed
        and combined_passed
        and objective_passed
        and fields_passed
        and hard_passed
        and coefficients.inside_interval
        and np.isfinite(clip_threshold)
        and clip_threshold > 0.0
        and retained["exactly_unchanged"]
        and checkpoint_unchanged
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "identity": {
            "git_head": _git_head(),
            "objective_calibration_contract_sha256": objective_contract.sha256,
            "coupled_candidate_contract_sha256": candidate["coupled_contract"].sha256,
            "reachability_report_sha256": sha256_file(args.reachability_report),
            "real_v6_admission_report_sha256": sha256_file(args.gpu_admission_report),
            "structured_checkpoint_sha256": sha256_file(args.selected_checkpoint),
            "prepared_manifest_sha256": sha256_file(args.prepared_manifest),
            "flux_basis_amendment_sha256": amendment.sha256,
        },
        "policy": {
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "fresh_validation_values_evaluated": False,
            "sealed_test_read": False,
            "optimizer_created": False,
            "optimizer_updates_executed": 0,
            "checkpoint_written": False,
            "forcing_records_reconstructed": 0,
            "subdaily_state_scan_steps": 0,
        },
        "backend": jax.default_backend(),
        "devices": [str(value) for value in runtime["devices"]],
        "model": {
            "candidate_parameter_count": _tree_size(jax, parameters),
            "trainable_parameter_count": _tree_size(jax, trainable),
            "retained_parameters": retained,
        },
        "reachable_state_anchor": {
            "coordinate_count": int(layout.global_state_indices.size),
            "field_count": len(layout.field_names),
            "field_names": list(layout.field_names),
            "reachability_union_coordinate_count": reachability_report["reachability"][
                "union_reachable_coordinate_count"
            ],
        },
        "records": records,
        "calibration": {
            "target_gradient_norm_ratios": dict(calibration["target_gradient_norm_ratios"]),
            "coefficient_candidates": {name: list(values) for name, values in coefficients.candidates.items()},
            "resolved_coefficients": dict(coefficients.resolved),
            "coefficients_inside_interval": coefficients.inside_interval,
            "combined_gradient_norms": combined_norms,
            "combined_gradient_p90_clip_threshold": clip_threshold,
        },
        "acceptance": {
            "source_gradients_passed": source_passed,
            "combined_gradients_passed": combined_passed,
            "objectives_passed": objective_passed,
            "all_reachable_fields_active": fields_passed,
            "physical_hard_gates_passed": hard_passed,
            "retained_parameters_passed": retained["exactly_unchanged"],
            "structured_checkpoint_unchanged": checkpoint_unchanged,
        },
        "memory_before": memory_before,
        "memory_after": _memory_report(runtime["devices"][0]),
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": objective_contract.payload["next_gate"]["on_pass" if passed else "on_failure"],
    }
    _atomic_json(output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path, required=True)
    parser.add_argument("--base-statistics", type=Path, required=True)
    parser.add_argument("--process-statistics", type=Path, required=True)
    parser.add_argument("--fast-statistics", type=Path, required=True)
    parser.add_argument("--training-contract", type=Path, required=True)
    parser.add_argument("--flux-basis-amendment", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--prepared-updates", type=int, default=1000)
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
    parser.add_argument("--reachability-report", type=Path, required=True)
    parser.add_argument(
        "--objective-calibration-contract",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
