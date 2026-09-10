"""Evaluate the broad structured-GPP checkpoint on a fresh 7/30-day panel."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import (
    sha256_file,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_broad_checkpoint import (
    load_broad_checkpoint,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_fresh_screen import (
    load_structured_gpp_fresh_screen_contract,
    verify_fresh_validation_panel,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_training import (
    CANDIDATE_PARAMETER_COUNT,
    SPECIALIST_PARAMETER_COUNT,
    replace_specialist,
    specialist_element_count,
)
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    add_error_sufficient_statistics,
    add_target_oriented_acceptance_statistics,
    derive_target_oriented_acceptance_scales,
    finalize_target_oriented_acceptance_statistics,
    normalized_error_from_sufficient_statistics,
    normalized_error_sufficient_statistics,
    target_oriented_acceptance_sufficient_statistics,
)
from scripts.hpc.audit_target_oriented_flux_gradient_variance import (
    _hard_host,
    _load_context,
    _tree_size,
)
from scripts.hpc.calibrate_target_oriented_distribution_aware_guard import (
    _replace_adapter,
)
from scripts.hpc.run_successor_fast_owner_gpu_probe import PROBE_SEED
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_attention_gpu_admission import _head as _parent_head
from scripts.hpc.run_target_oriented_full_step_preflight import HardCounts
from scripts.hpc.run_target_oriented_structured_gpp_training_preflight import (
    ADAPTER_PARAMETER_COUNT,
    PARENT_PARAMETER_COUNT,
)
from scripts.hpc.run_target_oriented_training import (
    _add_hard_host,
    _hard_failure_count,
    _validation_records,
)
from scripts.hpc.successor_rollout_screen_prepared import (
    load_prepared_rollout_artifact,
)
from scripts.hpc.target_oriented_validation_prepared import (
    load_target_oriented_validation_prepared,
)

SCHEMA_VERSION = "target_oriented_structured_gpp_fresh_screen_result_v1"


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _load_broad_training_report(
    path: Path,
    screen: Any,
) -> Mapping[str, Any]:
    identity = screen.payload["identity"]
    if sha256_file(path) != identity["broad_training_report_sha256"]:
        raise ValueError("structured GPP broad-training report hash drift")
    report = json.loads(path.read_text(encoding="utf-8"))
    selected = report.get("selected_checkpoint", {})
    if (
        report.get("schema_version")
        != "target_oriented_structured_gpp_broad_training_result_v1"
        or report.get("status") != "passed"
        or report.get("next_gate")
        != "freeze_fresh_independent_seven_thirty_day_science_screen"
        or report.get("acceptance", {}).get("broad_stage_accepted") is not True
        or report.get("policy", {}).get("consumed_96_window_panel_read") is not False
        or int(selected.get("completed_updates", -1))
        != int(identity["selected_broad_checkpoint_update"])
        or selected.get("sha256") != identity["selected_broad_checkpoint_sha256"]
        or int(selected.get("parameter_count", -1))
        != int(identity["trainable_specialist_parameter_count"])
    ):
        raise ValueError("structured GPP broad-training result is not screen eligible")
    return report


def _make_fresh_evaluator(
    runtime: Mapping[str, Any],
    head: Any,
    template: Any,
    acceptance_scales: Any,
    state_scales: Any,
    *,
    include_trajectories: bool = False,
) -> Any:
    import jax
    import jax.numpy as jnp

    from research.daily_coarse_graining.daily_operator.long_horizon_objective import (
        rollout_hard_gate_counts,
    )
    from research.daily_coarse_graining.daily_operator.model import bind_daily_operator
    from research.daily_coarse_graining.daily_operator.successor_training import (
        TERMINAL_STOCK_FIELDS,
        extract_biomass_productivity_trajectories,
        rollout_daily_operator,
        trajectories_from_rollout,
    )

    learned_names = tuple(field.name for field in runtime["learned_schema"].fields)

    @partial(jax.jit, static_argnames=("horizon",))
    def evaluate(parameters: Any, batch: Any, day_inputs: Any, *, horizon: int) -> Any:
        retained = replace(template, day_inputs=day_inputs)
        transition = bind_daily_operator(
            input_assembler=runtime["adapter"].input_assembler,
            process_head=head,
            retained_tail=retained,
        )

        def rollout_one(inputs: Any) -> Any:
            return rollout_daily_operator(
                parameters,
                inputs,
                transition=transition,
                contract=runtime["contract"],
                rematerialize=False,
            )

        trace = jax.vmap(rollout_one)(batch.rollout_inputs)
        predicted = jax.vmap(
            lambda value: trajectories_from_rollout(value, runtime["contract"])
        )(trace)

        def expected_one(
            state: Any,
            defined: Any,
            gpp: Any,
            gpp_defined: Any,
            inputs: Any,
        ) -> Any:
            return extract_biomass_productivity_trajectories(
                state,
                defined,
                daily_fast_gpp=gpp,
                daily_fast_gpp_defined=gpp_defined,
                pft_fraction=inputs.pft.pft_fraction,
                active_pft_mask=inputs.pft.active_pft_mask,
                contract=runtime["contract"],
            )

        expected = jax.vmap(expected_one)(
            batch.expected_next_state,
            batch.expected_next_state_defined,
            batch.expected_fast_gpp,
            batch.expected_fast_gpp_defined,
            batch.rollout_inputs,
        )
        one_predicted = jax.tree_util.tree_map(lambda value: value[0], predicted)
        one_expected = jax.tree_util.tree_map(lambda value: value[0], expected)
        rollout_statistics = target_oriented_acceptance_sufficient_statistics(
            one_predicted.fluxes,
            one_expected.fluxes,
            one_predicted.stocks,
            one_expected.stocks,
            scales=acceptance_scales,
            horizon=horizon,
        )

        teacher_inputs = jax.tree_util.tree_map(
            lambda value: value[0],
            batch.rollout_inputs,
        )
        def direct_gpp_step(carry: Any, inputs: Any) -> tuple[Any, Any]:
            assembled = runtime["adapter"].input_assembler.assemble(inputs)
            context = head.assemble_context(parameters, assembled)
            prediction = head(parameters, context, assembled)
            return carry, prediction.named_outputs["daily.fast_gpp"]

        _, direct_values = jax.lax.scan(direct_gpp_step, None, teacher_inputs)
        direct_predicted = jnp.asarray(direct_values, dtype=jnp.float64)
        direct_expected = jnp.asarray(batch.expected_fast_gpp[0], dtype=jnp.float64)
        direct_defined = jnp.asarray(
            batch.expected_fast_gpp_defined[0],
            dtype=jnp.bool_,
        )
        fraction = jnp.asarray(teacher_inputs.pft.pft_fraction, dtype=jnp.float64)
        active = jnp.asarray(teacher_inputs.pft.active_pft_mask, dtype=jnp.bool_)
        if fraction.ndim == 2:
            fraction = fraction[:, None, :]
        if active.ndim == 2:
            active = active[:, None, :]
        predicted_defined = jnp.broadcast_to(active, direct_predicted.shape)
        admitted = active & predicted_defined & direct_defined
        complete = jnp.all((~active) | admitted, axis=-1)
        predicted_area = jnp.sum(
            jnp.where(admitted, direct_predicted * fraction, 0.0),
            axis=-1,
        )
        expected_area = jnp.sum(
            jnp.where(admitted, direct_expected * fraction, 0.0),
            axis=-1,
        )
        direct_statistics = normalized_error_sufficient_statistics(
            predicted_area,
            expected_area,
            complete,
            acceptance_scales.fluxes["gpp"],
        )

        rollout_hard = rollout_hard_gate_counts(
            one_predicted.stocks,
            one_expected.stocks,
            nonnegative_stock_fields=TERMINAL_STOCK_FIELDS,
            budget_residuals={
                "water": trace.result.diagnostics["water_budget_residual"][0],
                "carbon": trace.result.diagnostics["carbon_budget_residual"][0],
            },
            predicted_discrete_state=jax.tree_util.tree_map(
                lambda value: value[0],
                trace.result.next_discrete_state,
            ),
            expected_discrete_state=jax.tree_util.tree_map(
                lambda value: value[0],
                batch.expected_next_discrete_state,
            ),
            budget_atol=2.0e-12,
            stock_atol=2.0e-12,
        )
        actual_state = jnp.asarray(trace.result.next_state.other[0])
        actual_defined = jnp.asarray(
            trace.result.next_state_defined.other[0],
            dtype=jnp.bool_,
        )
        wanted_defined = jnp.asarray(
            batch.expected_next_state_defined[0],
            dtype=jnp.bool_,
        )
        dynamic = jnp.broadcast_to(
            jnp.asarray(state_scales.state_declared_dynamic.other, dtype=jnp.bool_),
            actual_defined.shape,
        )
        mismatch = actual_defined != wanted_defined
        prediction_nonfinite = sum(
            (
                jnp.sum(~jnp.isfinite(jnp.asarray(value)), dtype=jnp.int32)
                for name, value in trace.result.prediction.named_outputs.items()
                if name in learned_names
            ),
            jnp.asarray(0, dtype=jnp.int32),
        )
        hard = HardCounts(
            prediction_nonfinite + rollout_hard.nonfinite_defined_prediction,
            jnp.sum(wanted_defined & ~jnp.isfinite(actual_state), dtype=jnp.int32),
            jnp.sum(mismatch & ~dynamic, dtype=jnp.int32)
            + rollout_hard.defined_mask_mismatch,
            jnp.sum(mismatch & dynamic, dtype=jnp.int32),
            rollout_hard.discrete_state_mismatch,
            rollout_hard.budget_residual_failure,
            rollout_hard.negative_declared_stock,
        )
        if include_trajectories:
            return rollout_statistics, direct_statistics, hard, one_predicted, one_expected
        return rollout_statistics, direct_statistics, hard

    return evaluate


def _direct_report(statistics: Any) -> Mapping[str, Any]:
    error = normalized_error_from_sufficient_statistics(statistics)
    mse = float(np.asarray(error.mse))
    bias = float(np.asarray(error.mean_bias))
    count = float(np.asarray(error.count))
    rmse = float(np.sqrt(max(mse, 0.0)))
    return {
        "normalized_rmse": rmse,
        "normalized_signed_bias": bias,
        "evaluated_days": count,
        "finite_and_nonempty": bool(
            count > 0.0 and np.isfinite(rmse) and np.isfinite(bias)
        ),
        "promotion_gate": "report_only",
    }


def _evaluate_panel(
    runtime: Mapping[str, Any],
    *,
    parameters: Any,
    evaluator: Any,
    prepared_path: Path,
    prepared: Mapping[str, Any],
    panel: Mapping[str, Any],
    target_contract: Any,
    mode: str,
) -> Mapping[str, Any]:
    jax = runtime["jax"]
    rollout_cells: dict[tuple[str, int], Any] = {}
    direct_cells: dict[tuple[str, int], Any] = {}
    direct_total = None
    hard = {name: 0 for name in HardCounts._fields}
    timings = []
    selected = _validation_records(panel, prepared, mode=mode)
    for ordinal, (record, (slice_id, horizon, entry)) in enumerate(
        selected,
        start=1,
    ):
        artifact = load_prepared_rollout_artifact(
            prepared_path,
            record,
            slice_id=slice_id,
            window_entry=entry,
        )
        started = time.perf_counter()
        rollout_statistics, direct_statistics, local_hard = evaluator(
            parameters,
            artifact.batch,
            artifact.day_inputs,
            horizon=horizon,
        )
        jax.block_until_ready((rollout_statistics, direct_statistics, local_hard))
        timings.append(time.perf_counter() - started)
        rollout_host, direct_host, hard_host = jax.device_get(
            (rollout_statistics, direct_statistics, local_hard)
        )
        key = (slice_id, horizon)
        rollout_cells[key] = (
            rollout_host
            if key not in rollout_cells
            else add_target_oriented_acceptance_statistics(
                rollout_cells[key],
                rollout_host,
            )
        )
        direct_cells[key] = (
            direct_host
            if key not in direct_cells
            else add_error_sufficient_statistics(direct_cells[key], direct_host)
        )
        direct_total = (
            direct_host
            if direct_total is None
            else add_error_sufficient_statistics(direct_total, direct_host)
        )
        hard = _add_hard_host(hard, _hard_host(hard_host))
        print(
            f"fresh_screen ordinal={ordinal}/{len(selected)} "
            f"slice={slice_id} horizon={horizon}",
            flush=True,
        )

    reports = {}
    direct_reports = {}
    macros = []
    for slice_id in ("temporal", "spatial", "joint"):
        reports[slice_id] = {}
        direct_reports[slice_id] = {}
        for horizon in (7, 30):
            key = (slice_id, horizon)
            report = finalize_target_oriented_acceptance_statistics(
                rollout_cells[key],
                horizon=horizon,
                target_contract=target_contract,
            )
            reports[slice_id][str(horizon)] = report
            direct_reports[slice_id][str(horizon)] = _direct_report(
                direct_cells[key]
            )
            macros.append(report["science_macro_normalized_rmse"])
    hard_failure_count = _hard_failure_count(hard)
    full_panel = mode == "formal"
    direct_all_valid = all(
        value["finite_and_nonempty"]
        for by_horizon in direct_reports.values()
        for value in by_horizon.values()
    )
    all_absolute = full_panel and hard_failure_count == 0 and all(
        reports[slice_id][str(horizon)]["all_key_gates_passed"]
        for slice_id in reports
        for horizon in (7, 30)
    )
    return {
        "full_frozen_panel": full_panel,
        "evaluated_windows": len(selected),
        "science_macro_normalized_rmse": float(np.mean(macros)),
        "hard_counts": hard,
        "hard_failure_count": hard_failure_count,
        "direct_teacher_state_gpp": {
            "aggregate": _direct_report(direct_total),
            "all_cells_finite_and_nonempty": direct_all_valid,
            "slices": direct_reports,
        },
        "absolute_short_horizon_passed": all_absolute,
        "slices": reports,
        "timing_seconds": {
            "total": float(sum(timings)),
            "minimum": float(min(timings)),
            "median": float(np.median(timings)),
            "maximum": float(max(timings)),
        },
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    started = time.perf_counter()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    if report_path.exists():
        raise FileExistsError(report_path)

    screen = load_structured_gpp_fresh_screen_contract(args.screen_contract)
    screen_identity = screen.payload["identity"]
    broad_report = _load_broad_training_report(args.broad_training_report, screen)
    if sha256_file(args.selected_checkpoint) != screen_identity[
        "selected_broad_checkpoint_sha256"
    ]:
        raise ValueError("structured GPP broad checkpoint hash drift")
    if sha256_file(args.distribution_training_report) != screen_identity[
        "parent_distribution_report_sha256"
    ]:
        raise ValueError("structured GPP parent training report hash drift")
    if args.stage_checkpoint_sha256 != screen_identity[
        "base_attention_checkpoint_sha256"
    ]:
        raise ValueError("structured GPP base checkpoint identity drift")
    if args.distribution_checkpoint_sha256 != screen_identity[
        "parent_distribution_checkpoint_sha256"
    ]:
        raise ValueError("structured GPP distribution checkpoint identity drift")

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
    from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_broad_training import (
        load_structured_gpp_broad_training_contract,
    )

    amendment = load_target_oriented_flux_basis_amendment(args.flux_basis_amendment)
    context = dict(
        _load_context(
            args,
            science_flux_basis=amendment.flux_basis,
            verify_validation_artifacts=False,
        )
    )
    runtime = context["runtime"]
    jax = runtime["jax"]
    target_contract = context["target_contract"]
    broad_contract = load_structured_gpp_broad_training_contract(
        args.broad_training_contract
    )
    if (
        amendment.sha256 != screen_identity["flux_basis_amendment_sha256"]
        or context["training_contract"].sha256
        != screen_identity["base_training_contract_sha256"]
        or target_contract.sha256 != screen_identity["target_contract_sha256"]
        or broad_contract.sha256 != screen_identity["broad_training_contract_sha256"]
    ):
        raise ValueError("structured GPP fresh-screen parent contract drift")

    structured_contract = load_structured_gpp_contract(args.structured_contract)
    distribution_contract = load_distribution_aware_maintenance_contract(
        args.distribution_contract
    )
    if structured_contract.sha256 != screen_identity["candidate_contract_sha256"]:
        raise ValueError("structured GPP fresh-screen candidate contract drift")

    distribution_report = json.loads(
        args.distribution_training_report.read_text(encoding="utf-8")
    )
    distribution_checkpoint = load_distribution_aware_checkpoint(
        args.distribution_checkpoint,
        expected_sha256=args.distribution_checkpoint_sha256,
        identity=distribution_report["identity"],
        completed_updates=250,
        parameter_count=ADAPTER_PARAMETER_COUNT,
        clip_threshold=float(distribution_report["training"]["clip_threshold"]),
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
        jax.device_put(distribution_checkpoint["adapter"]),
    )
    if _tree_size(jax, parent_parameters) != PARENT_PARAMETER_COUNT:
        raise ValueError("structured GPP fresh-screen parent parameter count drift")

    distribution_head = build_target_oriented_distribution_aware_maintenance_head(
        _parent_head(runtime, target_contract),
        context["template"],
        distribution_contract=distribution_contract,
    )
    head = build_target_oriented_structured_gpp_head(
        distribution_head,
        structured_contract=structured_contract,
    )
    initialized = initialize_structured_gpp_parameters(
        parent_parameters,
        jax.random.PRNGKey(PROBE_SEED + 20260829),
        schema=runtime["learned_schema"],
        target_contract=target_contract,
        structured_contract=structured_contract,
    )
    selected_checkpoint = load_broad_checkpoint(
        args.selected_checkpoint,
        expected_sha256=screen_identity["selected_broad_checkpoint_sha256"],
        identity=broad_report["identity"],
        completed_updates=int(screen_identity["selected_broad_checkpoint_update"]),
        parameter_count=SPECIALIST_PARAMETER_COUNT,
        clip_threshold=float(broad_report["training"]["clip_threshold"]),
    )
    specialist = jax.device_put(selected_checkpoint["specialist"])
    parameters = replace_specialist(initialized, specialist)
    if (
        specialist_element_count(specialist) != SPECIALIST_PARAMETER_COUNT
        or _tree_size(jax, parameters) != CANDIDATE_PARAMETER_COUNT
    ):
        raise ValueError("structured GPP fresh-screen candidate parameter count drift")

    validation_prepared, panel = load_target_oriented_validation_prepared(
        args.fresh_validation_prepared_manifest,
        training_contract_sha256=context["training_contract"].sha256,
        target_contract_sha256=target_contract.sha256,
        verify_artifacts=False,
    )
    consumed_panel = json.loads(
        args.consumed_validation_panel.read_text(encoding="utf-8")
    )
    verify_fresh_validation_panel(
        panel,
        runtime["index"],
        screen,
        target_contract,
        consumed_panel,
    )
    if validation_prepared.get("screen_contract_sha256") != screen.sha256:
        raise ValueError("fresh validation-prepared screen identity drift")

    acceptance_scales = derive_target_oriented_acceptance_scales(
        runtime["base_statistics"].arrays["next_state"].scale,
        runtime["process_statistics"].outputs["daily.fast_gpp"].scale,
        runtime["contract"],
    )
    evaluator = _make_fresh_evaluator(
        runtime,
        head,
        context["template"],
        acceptance_scales,
        context["scales"],
    )
    panel_report = _evaluate_panel(
        runtime,
        parameters=parameters,
        evaluator=evaluator,
        prepared_path=args.fresh_validation_prepared_manifest,
        prepared=validation_prepared,
        panel=panel,
        target_contract=target_contract,
        mode=args.mode,
    )
    direct_valid = panel_report["direct_teacher_state_gpp"][
        "all_cells_finite_and_nonempty"
    ]
    admission_passed = (
        args.mode == "admission"
        and panel_report["hard_failure_count"] == 0
        and direct_valid
    )
    formal_passed = (
        args.mode == "formal"
        and panel_report["absolute_short_horizon_passed"]
        and direct_valid
    )
    next_gate = screen.payload["next_gate"][
        "on_admission_pass"
        if admission_passed
        else "on_admission_failure"
        if args.mode == "admission"
        else "on_formal_pass"
        if formal_passed
        else "on_formal_failure"
    ]
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "mode": args.mode,
        "identity": {
            "git_head": _git_head(),
            "screen_contract_sha256": screen.sha256,
            "broad_training_report_sha256": sha256_file(
                args.broad_training_report
            ),
            "selected_checkpoint_sha256": sha256_file(args.selected_checkpoint),
            "selected_checkpoint_optimizer_step": int(
                screen_identity["selected_broad_checkpoint_update"]
            ),
            "validation_prepared_manifest_sha256": validation_prepared[
                "manifest_sha256"
            ],
            "validation_panel_sha256": panel["panel_sha256"],
            "consumed_validation_panel_sha256": consumed_panel["panel_sha256"],
        },
        "policy": {
            "optimizer_updates_executed": 0,
            "checkpoint_writes": 0,
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "sealed_test_read": False,
            "forcing_records_reconstructed": 0,
            "subdaily_state_scan_steps": 0,
        },
        "model": {
            "candidate_parameter_count": CANDIDATE_PARAMETER_COUNT,
            "trainable_specialist_parameter_count": SPECIALIST_PARAMETER_COUNT,
            "retained_parent_parameter_count": int(
                screen_identity["retained_parent_parameter_count"]
            ),
        },
        "panel": panel_report,
        "acceptance": {
            "admission_execution_and_hard_gates_passed": admission_passed,
            "formal_absolute_short_horizon_passed": formal_passed,
            "scientific_promotion_decided": args.mode == "formal",
        },
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": next_gate,
    }
    _atomic_json(report_path, report)
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
    parser.add_argument("--fresh-validation-prepared-manifest", type=Path, required=True)
    parser.add_argument("--consumed-validation-panel", type=Path, required=True)
    parser.add_argument("--stage-checkpoint", type=Path, required=True)
    parser.add_argument("--stage-checkpoint-sha256", required=True)
    parser.add_argument("--distribution-contract", type=Path, required=True)
    parser.add_argument("--distribution-training-report", type=Path, required=True)
    parser.add_argument("--distribution-checkpoint", type=Path, required=True)
    parser.add_argument("--distribution-checkpoint-sha256", required=True)
    parser.add_argument("--structured-contract", type=Path, required=True)
    parser.add_argument("--broad-training-contract", type=Path, required=True)
    parser.add_argument("--broad-training-report", type=Path, required=True)
    parser.add_argument("--selected-checkpoint", type=Path, required=True)
    parser.add_argument("--screen-contract", type=Path, required=True)
    parser.add_argument("--mode", choices=("admission", "formal"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
