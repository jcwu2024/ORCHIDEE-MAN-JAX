"""Calibrate structured-GPP gradients and execute one discarded update."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_training import (
    CANDIDATE_PARAMETER_COUNT,
    SPECIALIST_PARAMETER_COUNT,
    calibrate_structured_gpp_gradient_norms,
    linearized_structured_gpp_direct_loss,
    load_structured_gpp_training_preflight_contract,
    replace_specialist,
    specialist_element_count,
    specialist_tree,
    structured_gpp_direct_loss,
    structured_gpp_direct_statistics,
)
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    add_error_sufficient_statistics,
    add_science_window_sufficient_statistics,
    initialize_target_oriented_adamw,
    science_window_from_sufficient_statistics,
    target_oriented_adamw_update,
)
from scripts.hpc.audit_target_oriented_flux_gradient_variance import (
    _add_gradient,
    _load_context,
    _tree_size,
    _zero_gradient,
)
from scripts.hpc.calibrate_target_oriented_distribution_aware_guard import (
    _load_artifact,
    _replace_adapter,
)
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_fast_owner_gpu_probe import PROBE_SEED, _tree_numeric_report
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_attention_gpu_admission import _head as _parent_head
from scripts.hpc.run_target_oriented_full_step_preflight import (
    _add_hard,
    _hard_failure_count,
    _make_steps,
    _merge_statistics,
    _slice,
    _zero_hard,
)
from scripts.hpc.run_target_oriented_structured_gpp_gpu_admission import (
    _component_gradient_reports,
)

SCHEMA_VERSION = "target_oriented_structured_gpp_training_preflight_result_v1"
GPP_FIELD = "daily.fast_gpp"
PARENT_PARAMETER_COUNT = 93_504_933
ADAPTER_PARAMETER_COUNT = 9_750_529


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _retained_parameter_report(
    jax: Any,
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> Mapping[str, Any]:
    def retained(parameters: Mapping[str, Any]) -> Mapping[str, Any]:
        target = {name: value for name, value in parameters["target_oriented"].items() if name != "structured_gpp"}
        return dict(parameters, target_oriented=target)

    before_leaves, before_tree = jax.tree_util.tree_flatten(retained(before))
    after_leaves, after_tree = jax.tree_util.tree_flatten(retained(after))
    same_tree = before_tree == after_tree
    same_objects = same_tree and all(left is right for left, right in zip(before_leaves, after_leaves, strict=True))
    return {
        "same_tree_structure": same_tree,
        "same_array_objects": same_objects,
        "leaf_count": len(before_leaves),
        "element_count": sum(int(np.prod(value.shape, dtype=np.int64)) for value in before_leaves),
        "exactly_unchanged": same_objects,
    }


def _hard_host(value: Any) -> Mapping[str, int]:
    return {name: int(np.asarray(item)) for name, item in value._asdict().items()}


def _make_direct_steps(
    runtime: Mapping[str, Any],
    head: Any,
    fixed_parameters: Mapping[str, Any],
) -> Mapping[str, Any]:
    jax = runtime["jax"]
    jnp = runtime["jnp"]
    fixed_parameters = replace_specialist(fixed_parameters, {})

    from research.daily_coarse_graining.daily_operator.parameter_routing import (
        PFT_TRAIT_BARE_SOIL,
    )

    scale = jnp.asarray(runtime["process_statistics"].outputs[GPP_FIELD].scale, dtype=jnp.float64)

    def predicted_gpp(parameters, inputs):
        assembled = runtime["adapter"].input_assembler.assemble(inputs)
        context = head.assemble_context(parameters, assembled)
        return head.learned_outputs(parameters, context, assembled)[GPP_FIELD]

    def active_vegetation(inputs):
        active = jnp.asarray(inputs.pft.active_pft_mask, dtype=jnp.bool_)
        fraction = jnp.asarray(inputs.pft.pft_fraction, dtype=jnp.float64)
        traits = jnp.asarray(inputs.pft.pft_traits, dtype=jnp.float64)
        return active & (fraction > 0.0) & (traits[..., PFT_TRAIT_BARE_SOIL] <= 0.5)

    one_predict = jax.vmap(predicted_gpp, in_axes=(None, 0))
    window_predict = jax.vmap(jax.vmap(predicted_gpp, in_axes=(None, 0)), in_axes=(None, 0))

    def one_statistics(specialist, fixed, batch):
        parameters = replace_specialist(fixed, specialist)
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

    def window_statistics(specialist, fixed, batch):
        parameters = replace_specialist(fixed, specialist)
        predicted = window_predict(parameters, batch.inputs)
        active = active_vegetation(batch.inputs)[..., None, :]
        return structured_gpp_direct_statistics(
            predicted,
            batch.expected_fast_gpp,
            batch.expected_fast_gpp_defined,
            scale,
            active,
        )

    one_forward_impl = jax.jit(one_statistics)
    window_forward_impl = jax.jit(window_statistics)
    one_gradient_impl = jax.jit(
        jax.value_and_grad(
            lambda specialist, fixed, batch, total: linearized_structured_gpp_direct_loss(
                one_statistics(specialist, fixed, batch), total
            ),
            argnums=0,
        )
    )
    window_gradient_impl = jax.jit(
        jax.value_and_grad(
            lambda specialist, fixed, batch, total: linearized_structured_gpp_direct_loss(
                window_statistics(specialist, fixed, batch), total
            ),
            argnums=0,
        )
    )
    return {
        "one_forward": lambda specialist, batch: one_forward_impl(specialist, fixed_parameters, batch),
        "window_forward": lambda specialist, batch: window_forward_impl(specialist, fixed_parameters, batch),
        "one_gradient": lambda specialist, batch, total: one_gradient_impl(specialist, fixed_parameters, batch, total),
        "window_gradient": lambda specialist, batch, total: window_gradient_impl(
            specialist, fixed_parameters, batch, total
        ),
    }


def _direct_microbatches(batch: Any) -> tuple[tuple[str, Any], ...]:
    return tuple(
        [("one_day", _slice(batch.one_day, index, index + 4)) for index in range(0, 32, 4)]
        + [("seven_day", _slice(batch.seven_day, index, index + 1)) for index in range(8)]
        + [("thirty_day", _slice(batch.thirty_day, index, index + 1)) for index in range(4)]
    )


def _direct_forward(
    jax: Any,
    specialist: Any,
    batch: Any,
    direct_steps: Mapping[str, Any],
) -> tuple[Any, float]:
    started = time.perf_counter()
    statistics = []
    for index, (source, microbatch) in enumerate(_direct_microbatches(batch)):
        step = direct_steps["one_forward"] if source == "one_day" else direct_steps["window_forward"]
        statistics.append(step(specialist, microbatch))
        if index in (0, 8, 16):
            print(f"structured_gpp direct_forward source={source}", flush=True)
    jax.block_until_ready(statistics)
    total = statistics[0]
    for value in statistics[1:]:
        total = add_error_sufficient_statistics(total, value)
    jax.block_until_ready(total)
    return total, time.perf_counter() - started


def _direct_gradient(
    jax: Any,
    jnp: Any,
    specialist: Any,
    batch: Any,
    direct_steps: Mapping[str, Any],
) -> tuple[Any, Any, float, float]:
    total, forward_seconds = _direct_forward(jax, specialist, batch, direct_steps)
    started = time.perf_counter()
    gradient = _zero_gradient(jax, jnp, specialist)
    for index, (source, microbatch) in enumerate(_direct_microbatches(batch)):
        step = direct_steps["one_gradient"] if source == "one_day" else direct_steps["window_gradient"]
        _value, current = step(specialist, microbatch, total)
        gradient = _add_gradient(jax, jnp, gradient, current)
        if index in (0, 8, 16):
            print(f"structured_gpp direct_gradient source={source}", flush=True)
    jax.block_until_ready(gradient)
    return total, gradient, forward_seconds, time.perf_counter() - started


def _horizon_gradient(
    jax: Any,
    jnp: Any,
    parameters: Mapping[str, Any],
    specialist: Any,
    batch: Any,
    steps: Mapping[str, Any],
    gradient_steps: Mapping[str, Any],
    *,
    horizon: int,
    flux_basis: tuple[str, ...],
) -> tuple[Any, Any, Any, float, float]:
    source = batch.seven_day if horizon == 7 else batch.thirty_day
    count = 8 if horizon == 7 else 4
    name = "seven" if horizon == 7 else "thirty"
    microbatches = [_slice(source, index, index + 1) for index in range(count)]
    started = time.perf_counter()
    forward = [steps[f"{name}_forward"](parameters, value) for value in microbatches]
    jax.block_until_ready(forward)
    print(f"structured_gpp horizon_forward days={horizon}", flush=True)
    total = _merge_statistics(
        add_science_window_sufficient_statistics,
        [value[0] for value in forward],
    )
    hard = _zero_hard(jnp)
    for value in forward:
        hard = _add_hard(jax, hard, value[1])
    forward_seconds = time.perf_counter() - started

    started = time.perf_counter()
    gradient = _zero_gradient(jax, jnp, specialist)
    for microbatch in microbatches:
        _value, current = gradient_steps[f"{name}_gradient"](specialist, microbatch, total)
        gradient = _add_gradient(jax, jnp, gradient, current)
    jax.block_until_ready((gradient, hard))
    print(f"structured_gpp horizon_gradient days={horizon}", flush=True)
    result = science_window_from_sufficient_statistics(total, flux_basis=flux_basis)
    return result, gradient, hard, forward_seconds, time.perf_counter() - started


def _one_day_hard(
    jax: Any,
    jnp: Any,
    parameters: Mapping[str, Any],
    batch: Any,
    steps: Mapping[str, Any],
) -> Any:
    hard = _zero_hard(jnp)
    for start in range(0, 32, 4):
        _state, _process, current = steps["one_forward"](parameters, _slice(batch.one_day, start, start + 4))
        hard = _add_hard(jax, hard, current)
    jax.block_until_ready(hard)
    return hard


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()

    contract = load_structured_gpp_training_preflight_contract(args.preflight_contract)
    identity = contract.payload["identity"]
    if sha256_file(args.gpu_admission_report) != identity["gpu_admission_report_sha256"]:
        raise ValueError("structured GPP admission report identity drift")
    admission = json.loads(args.gpu_admission_report.read_text(encoding="utf-8"))
    if admission.get("status") != "passed" or admission["policy"]["training_updates"] != 0:
        raise ValueError("structured GPP real-v6 admission is not accepted")
    if sha256_file(args.prepared_manifest) != identity["prepared_manifest_sha256"]:
        raise ValueError("structured GPP prepared-manifest identity drift")

    from research.daily_coarse_graining.daily_operator.target_oriented_distribution_aware_checkpoint import (
        load_distribution_aware_checkpoint,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_distribution_aware_maintenance import (
        initialize_distribution_aware_maintenance_parameters,
        load_distribution_aware_maintenance_contract,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_flux_basis import (
        load_target_oriented_flux_basis_amendment,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_process_head import (
        build_target_oriented_distribution_aware_maintenance_head,
        build_target_oriented_structured_gpp_head,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp import (
        initialize_structured_gpp_parameters,
        load_structured_gpp_contract,
    )

    amendment = load_target_oriented_flux_basis_amendment(args.flux_basis_amendment)
    if amendment.sha256 != identity["flux_basis_amendment_sha256"]:
        raise ValueError("structured GPP flux-basis identity drift")
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
    target_contract = context["target_contract"]
    if (
        context["training_contract"].sha256 != identity["base_training_contract_sha256"]
        or target_contract.sha256 != identity["target_contract_sha256"]
    ):
        raise ValueError("structured GPP parent contract identity drift")

    distribution_contract = load_distribution_aware_maintenance_contract(args.distribution_contract)
    structured_contract = load_structured_gpp_contract(args.structured_contract)
    if structured_contract.sha256 != identity["candidate_contract_sha256"]:
        raise ValueError("structured GPP candidate contract identity drift")
    training_report = json.loads(args.distribution_training_report.read_text(encoding="utf-8"))
    if (
        training_report.get("status") != "passed"
        or training_report["checkpoint"]["sha256"] != args.distribution_checkpoint_sha256
        or args.distribution_checkpoint_sha256 != identity["parent_distribution_checkpoint_sha256"]
    ):
        raise ValueError("structured GPP distribution checkpoint identity drift")
    checkpoint = load_distribution_aware_checkpoint(
        args.distribution_checkpoint,
        expected_sha256=args.distribution_checkpoint_sha256,
        identity=training_report["identity"],
        completed_updates=250,
        parameter_count=ADAPTER_PARAMETER_COUNT,
        clip_threshold=float(training_report["training"]["clip_threshold"]),
    )

    initialized_parent = initialize_distribution_aware_maintenance_parameters(
        context["parameters"],
        jax.random.PRNGKey(PROBE_SEED + 20260828),
        schema=runtime["learned_schema"],
        target_contract=target_contract,
        distribution_contract=distribution_contract,
    )
    parent_parameters = _replace_adapter(
        initialized_parent,
        jax.device_put(checkpoint["adapter"]),
    )
    if _tree_size(jax, parent_parameters) != PARENT_PARAMETER_COUNT:
        raise ValueError("structured GPP preflight parent parameter count drift")
    distribution_head = build_target_oriented_distribution_aware_maintenance_head(
        _parent_head(runtime, target_contract),
        context["template"],
        distribution_contract=distribution_contract,
    )
    head = build_target_oriented_structured_gpp_head(
        distribution_head,
        structured_contract=structured_contract,
    )
    parameters = initialize_structured_gpp_parameters(
        parent_parameters,
        jax.random.PRNGKey(PROBE_SEED + 20260829),
        schema=runtime["learned_schema"],
        target_contract=target_contract,
        structured_contract=structured_contract,
    )
    specialist = specialist_tree(parameters)
    if (
        _tree_size(jax, parameters) != CANDIDATE_PARAMETER_COUNT
        or specialist_element_count(specialist) != SPECIALIST_PARAMETER_COUNT
    ):
        raise ValueError("structured GPP preflight candidate parameter count drift")

    steps = _make_steps(
        runtime,
        head,
        context["template"],
        context["scales"],
        target_contract,
        science_flux_basis=amendment.flux_basis,
    )
    horizon_gradient_steps = steps["make_trainable_gradient_steps"](
        parameters,
        replace_specialist,
    )
    direct_steps = _make_direct_steps(runtime, head, parameters)
    memory_before = _memory_report(runtime["devices"][0])

    records = []
    direct_norms = []
    seven_norms = []
    thirty_norms = []
    for update in contract.calibration_updates:
        artifact = _load_artifact(args, context, update)
        record_started = time.perf_counter()
        direct_total, direct_gradient, direct_forward_seconds, direct_gradient_seconds = _direct_gradient(
            jax, jnp, specialist, artifact.batch, direct_steps
        )
        seven, seven_gradient, seven_hard, seven_forward_seconds, seven_gradient_seconds = _horizon_gradient(
            jax,
            jnp,
            parameters,
            specialist,
            artifact.batch,
            steps,
            horizon_gradient_steps,
            horizon=7,
            flux_basis=amendment.flux_basis,
        )
        thirty, thirty_gradient, thirty_hard, thirty_forward_seconds, thirty_gradient_seconds = _horizon_gradient(
            jax,
            jnp,
            parameters,
            specialist,
            artifact.batch,
            steps,
            horizon_gradient_steps,
            horizon=30,
            flux_basis=amendment.flux_basis,
        )
        reports = {
            "direct_gpp": _tree_numeric_report(jax, jax.device_get(direct_gradient)),
            "seven_day": _tree_numeric_report(jax, jax.device_get(seven_gradient)),
            "thirty_day": _tree_numeric_report(jax, jax.device_get(thirty_gradient)),
        }
        direct_norms.append(float(reports["direct_gpp"]["l2_norm"]))
        seven_norms.append(float(reports["seven_day"]["l2_norm"]))
        thirty_norms.append(float(reports["thirty_day"]["l2_norm"]))
        hard = {
            "seven_day": _hard_host(jax.device_get(seven_hard)),
            "thirty_day": _hard_host(jax.device_get(thirty_hard)),
        }
        records.append(
            {
                "artifact_update": update,
                "runtime_year": int(context["schedule"]["entries"][update - 1]["runtime_year"]),
                "objectives": {
                    "direct_gpp": float(np.asarray(structured_gpp_direct_loss(direct_total))),
                    "seven_day_science": float(np.asarray(seven.total)),
                    "thirty_day_science": float(np.asarray(thirty.total)),
                    "direct_gpp_evaluated_values": int(np.asarray(direct_total.count)),
                },
                "gradients": reports,
                "hard_counts": hard,
                "timing_seconds": {
                    "direct_forward": direct_forward_seconds,
                    "direct_gradient": direct_gradient_seconds,
                    "seven_forward": seven_forward_seconds,
                    "seven_gradient": seven_gradient_seconds,
                    "thirty_forward": thirty_forward_seconds,
                    "thirty_gradient": thirty_gradient_seconds,
                    "total": time.perf_counter() - record_started,
                },
            }
        )
        print(
            f"structured_gpp calibration={update} direct={direct_norms[-1]:.9g} "
            f"seven={seven_norms[-1]:.9g} thirty={thirty_norms[-1]:.9g} "
            f"seconds={records[-1]['timing_seconds']['total']:.3f}",
            flush=True,
        )
        del direct_gradient, seven_gradient, thirty_gradient

    calibration = calibrate_structured_gpp_gradient_norms(
        direct_norms,
        seven_norms,
        thirty_norms,
        minimum_clip_threshold=float(contract.payload["gradient_calibration"]["direct_gpp_clip_threshold_minimum"]),
    )
    calibration_passed = (
        all(
            report["all_finite"]
            and report["element_count"] == SPECIALIST_PARAMETER_COUNT
            and report["l2_norm"] > 0.0
            and (name != "direct_gpp" or report["nonzero_leaf_count"] == report["leaf_count"])
            for record in records
            for name, report in record["gradients"].items()
        )
        and all(np.isfinite(value) for record in records for value in record["objectives"].values())
        and all(_hard_failure_count(value) == 0 for record in records for value in record["hard_counts"].values())
    )

    artifact_update = int(contract.ephemeral_update["artifact_update"])
    artifact = _load_artifact(args, context, artifact_update)
    proposal_started = time.perf_counter()
    direct_before, gradient, direct_forward_seconds, direct_gradient_seconds = _direct_gradient(
        jax, jnp, specialist, artifact.batch, direct_steps
    )
    hard_before = _one_day_hard(jax, jnp, parameters, artifact.batch, steps)
    optimizer = initialize_target_oriented_adamw(specialist)
    learning_rate = float(contract.ephemeral_update["learning_rate"])
    optimizer_step = jax.jit(
        lambda values, derivatives, state: target_oriented_adamw_update(
            values,
            derivatives,
            state,
            learning_rate=jnp.asarray(learning_rate, dtype=jnp.float32),
            clip_norm=jnp.asarray(calibration.direct_p90_clip_threshold, dtype=jnp.float32),
            contract=context["training_contract"],
        )
    )
    update_started = time.perf_counter()
    updated_specialist, optimizer_after, raw_norm, clip_scale = optimizer_step(specialist, gradient, optimizer)
    jax.block_until_ready((updated_specialist, optimizer_after, raw_norm, clip_scale))
    optimizer_seconds = time.perf_counter() - update_started
    proposed_parameters = replace_specialist(parameters, updated_specialist)
    direct_after, post_forward_seconds = _direct_forward(jax, updated_specialist, artifact.batch, direct_steps)
    hard_after = _one_day_hard(jax, jnp, proposed_parameters, artifact.batch, steps)
    step = jax.tree_util.tree_map(lambda after, before: after - before, updated_specialist, specialist)
    gradient_report = _tree_numeric_report(jax, jax.device_get(gradient))
    step_report = _tree_numeric_report(jax, jax.device_get(step))
    gradient_components = _component_gradient_reports(jax, jax.device_get(gradient))
    step_components = _component_gradient_reports(jax, jax.device_get(step))
    retained = _retained_parameter_report(jax, parameters, proposed_parameters)
    optimizer_numeric = _tree_numeric_report(jax, jax.device_get(optimizer_after))
    before_loss = float(np.asarray(structured_gpp_direct_loss(direct_before)))
    after_loss = float(np.asarray(structured_gpp_direct_loss(direct_after)))
    hard_before_host = _hard_host(jax.device_get(hard_before))
    hard_after_host = _hard_host(jax.device_get(hard_after))
    ephemeral_passed = (
        np.isfinite(before_loss)
        and np.isfinite(after_loss)
        and after_loss < before_loss
        and gradient_report["all_finite"]
        and gradient_report["element_count"] == SPECIALIST_PARAMETER_COUNT
        and gradient_report["nonzero_leaf_count"] == gradient_report["leaf_count"]
        and step_report["all_finite"]
        and step_report["l2_norm"] > 0.0
        and all(value["all_finite"] and value["l2_norm"] > 0.0 for value in step_components.values())
        and optimizer_numeric["all_finite"]
        and retained["exactly_unchanged"]
        and _hard_failure_count(hard_before_host) == 0
        and _hard_failure_count(hard_after_host) == 0
        and int(np.asarray(optimizer_after.step)) == 1
        and float(np.asarray(raw_norm)) > 0.0
        and 0.0 < float(np.asarray(clip_scale)) <= 1.0
    )
    passed = calibration_passed and ephemeral_passed
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if passed else "failed",
        "identity": {
            "git_head": _git_head(),
            "preflight_contract_sha256": contract.sha256,
            "candidate_contract_sha256": structured_contract.sha256,
            "gpu_admission_report_sha256": sha256_file(args.gpu_admission_report),
            "prepared_manifest_sha256": sha256_file(args.prepared_manifest),
            "base_training_contract_sha256": context["training_contract"].sha256,
            "target_contract_sha256": target_contract.sha256,
            "flux_basis_amendment_sha256": amendment.sha256,
            "distribution_checkpoint_sha256": args.distribution_checkpoint_sha256,
        },
        "policy": {
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "validation_values_evaluated": False,
            "sealed_test_read": False,
            "calibration_optimizer_created": False,
            "ephemeral_optimizer_created": True,
            "training_updates_executed": 1,
            "training_updates_persisted": 0,
            "checkpoint_written": False,
            "recovery_written": False,
            "proposal_retained_after_process_exit": False,
            "forcing_records_reconstructed": 0,
            "subdaily_state_scan_steps": 0,
        },
        "backend": jax.default_backend(),
        "devices": [str(value) for value in runtime["devices"]],
        "model": {
            "candidate_parameter_count": _tree_size(jax, parameters),
            "trainable_specialist_parameter_count": specialist_element_count(specialist),
            "retained_parent_parameters": retained,
        },
        "calibration": {
            "passed": calibration_passed,
            "updates": list(contract.calibration_updates),
            "direct_gpp_p90_clip_threshold": calibration.direct_p90_clip_threshold,
            "median_gradient_norms": dict(calibration.median_norms),
            "initial_horizon_to_direct_norm_ratios": dict(calibration.initial_horizon_to_direct_norm_ratios),
            "records": records,
        },
        "ephemeral_update": {
            "passed": ephemeral_passed,
            "artifact_update": artifact_update,
            "runtime_year": int(context["schedule"]["entries"][artifact_update - 1]["runtime_year"]),
            "learning_rate": learning_rate,
            "clip_threshold": calibration.direct_p90_clip_threshold,
            "raw_gradient_norm": float(np.asarray(raw_norm)),
            "clip_scale": float(np.asarray(clip_scale)),
            "direct_gpp_loss_before": before_loss,
            "direct_gpp_loss_after": after_loss,
            "direct_gpp_relative_change": (after_loss - before_loss) / before_loss,
            "direct_gpp_evaluated_values": int(np.asarray(direct_before.count)),
            "gradient": gradient_report | {"components": gradient_components},
            "parameter_step": step_report | {"components": step_components},
            "optimizer_state": optimizer_numeric,
            "hard_counts_before": hard_before_host,
            "hard_counts_after": hard_after_host,
            "timing_seconds": {
                "direct_forward": direct_forward_seconds,
                "direct_gradient": direct_gradient_seconds,
                "optimizer_compile_and_step": optimizer_seconds,
                "post_update_direct_forward": post_forward_seconds,
                "total": time.perf_counter() - proposal_started,
            },
        },
        "device_memory": {
            "before_calibration": memory_before,
            "after_preflight": _memory_report(runtime["devices"][0]),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": (
            "freeze_bounded_250_update_direct_gpp_warm_start_contract"
            if passed
            else "stop_and_repair_structured_gpp_training_preflight"
        ),
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
    parser.add_argument("--preflight-contract", type=Path, required=True)
    parser.add_argument("--gpu-admission-report", type=Path, required=True)
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
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
