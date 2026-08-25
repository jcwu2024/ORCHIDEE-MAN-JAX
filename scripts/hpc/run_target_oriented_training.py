"""Run the retained-update target-oriented warmup stage and fixed validation."""

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

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file
from research.daily_coarse_graining.daily_operator.target_oriented_checkpoint import (
    load_target_oriented_recovery,
    load_target_oriented_stage_checkpoint,
    write_target_oriented_best_checkpoint,
    write_target_oriented_recovery,
    write_target_oriented_stage_checkpoint,
)
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    add_target_oriented_acceptance_statistics,
    derive_target_oriented_acceptance_scales,
    finalize_target_oriented_acceptance_statistics,
    ordered_target_oriented_validation_entries,
    target_oriented_acceptance_sufficient_statistics,
    verify_target_oriented_validation_panel,
)
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_fast_owner_gpu_probe import PROBE_SEED, _load_extended_runtime
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_successor_small_fit import _retained_tail_trace_signature
from scripts.hpc.run_target_oriented_attention_gpu_admission import PARAMETER_COUNT, _head
from scripts.hpc.run_target_oriented_full_step_preflight import (
    HardCounts,
    _effective_gradient,
    _hard_failure_count,
    _make_steps,
    _prepare_template,
    _validate_prepared_objective_shapes,
)
from scripts.hpc.successor_rollout_screen_prepared import load_prepared_rollout_artifact
from scripts.hpc.target_oriented_training_prepared import (
    load_prepared_target_artifact,
    load_prepared_target_manifest,
    load_prepared_target_manifest_range,
)
from scripts.hpc.target_oriented_validation_prepared import (
    load_target_oriented_validation_prepared,
)

FROZEN_CLIP_THRESHOLD = 5_323_655.8449
FORMAL_STAGE_END = 250
FORMAL_CONTINUATION_END = 1000
RECOVERY_INTERVAL = 50
CONDITIONING_UPDATES = frozenset((1, 50, 100, 150, 200, 250, 500, 750, 1000))


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _host_tree(jax: Any, value: Any) -> Any:
    return jax.tree_util.tree_map(np.asarray, jax.device_get(value))


def _hard_host(value: HardCounts) -> Mapping[str, int]:
    return {name: int(np.asarray(item)) for name, item in value._asdict().items()}


def _add_hard_host(left: Mapping[str, int], right: Mapping[str, int]) -> Mapping[str, int]:
    return {name: int(left.get(name, 0)) + int(right.get(name, 0)) for name in HardCounts._fields}


def _family_trees(gradient: Mapping[str, Any]) -> Mapping[str, Any]:
    target = gradient["target_oriented"]
    return {
        "base_source_encoder": {name: value for name, value in gradient.items() if name != "target_oriented"},
        "domain_refinement": target["domain_blocks"],
        "science_cross_attention": target["science_cross_blocks"],
        "science_token_mixer": target["science_mixer_blocks"],
        "primary_heads": (target["primary_decoder"], target["primary_heads"]),
        "state_heads": (target["state_decoder"], target["state_heads"]),
        "auxiliary_heads": (
            target["auxiliary_science_projection"],
            target["auxiliary_bottleneck"],
            target["auxiliary_label_heads"],
        ),
    }


def _make_gradient_conditioner(jax: Any, jnp: Any) -> Any:
    @jax.jit
    def report(gradient: Mapping[str, Any]) -> Mapping[str, Any]:
        result = {}
        for name, tree in _family_trees(gradient).items():
            leaves = jax.tree_util.tree_leaves(tree)
            squared = sum(
                (
                    jnp.sum(jnp.square(jnp.asarray(leaf, dtype=jnp.float32)), dtype=jnp.float32)
                    for leaf in leaves
                ),
                jnp.asarray(0.0, dtype=jnp.float32),
            )
            nonfinite = sum(
                (jnp.sum(~jnp.isfinite(jnp.asarray(leaf)), dtype=jnp.int32) for leaf in leaves),
                jnp.asarray(0, dtype=jnp.int32),
            )
            nonzero = sum(
                (jnp.sum(jnp.asarray(leaf) != 0, dtype=jnp.int32) for leaf in leaves),
                jnp.asarray(0, dtype=jnp.int32),
            )
            result[name] = {
                "l2_norm": jnp.sqrt(squared),
                "nonfinite_count": nonfinite,
                "nonzero_count": nonzero,
            }
        return result

    return report


def _condition_sources(jax: Any, conditioner: Any, sources: Mapping[str, Any]) -> Mapping[str, Any]:
    result = {}
    for source, gradient in sources.items():
        report = conditioner(gradient)
        jax.block_until_ready(report)
        result[source] = jax.tree_util.tree_map(lambda value: float(np.asarray(value)), report)
        for family in result[source].values():
            family["nonfinite_count"] = int(family["nonfinite_count"])
            family["nonzero_count"] = int(family["nonzero_count"])
    return result


def _make_optimizer_step(jax: Any, jnp: Any, training_contract: Any) -> Any:
    from research.daily_coarse_graining.daily_operator.target_oriented_training import (
        target_oriented_adamw_update,
    )

    @jax.jit
    def step(parameters: Any, gradients: Any, optimizer: Any, learning_rate: Any) -> Any:
        updated, next_optimizer, raw_norm, clip_scale = target_oriented_adamw_update(
            parameters,
            gradients,
            optimizer,
            learning_rate=learning_rate,
            clip_norm=FROZEN_CLIP_THRESHOLD,
            contract=training_contract,
        )
        nonfinite = sum(
            (
                jnp.sum(~jnp.isfinite(jnp.asarray(leaf)), dtype=jnp.int32)
                for leaf in jax.tree_util.tree_leaves((updated, next_optimizer))
            ),
            jnp.asarray(0, dtype=jnp.int32),
        )
        step_squared = sum(
            (
                jnp.sum(
                    jnp.square(jnp.asarray(after) - jnp.asarray(before)),
                    dtype=jnp.float32,
                )
                for before, after in zip(
                    jax.tree_util.tree_leaves(parameters),
                    jax.tree_util.tree_leaves(updated),
                    strict=True,
                )
            ),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        return updated, next_optimizer, raw_norm, clip_scale, nonfinite, jnp.sqrt(step_squared)

    return step


def _make_validation_evaluator(
    runtime: Mapping[str, Any],
    head: Any,
    template: Any,
    acceptance_scales: Any,
    state_scales: Any,
) -> Any:
    import jax
    import jax.numpy as jnp

    from research.daily_coarse_graining.daily_operator.long_horizon_objective import rollout_hard_gate_counts
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
        predicted = jax.vmap(lambda value: trajectories_from_rollout(value, runtime["contract"]))(trace)

        def expected_one(state: Any, defined: Any, gpp: Any, gpp_defined: Any, inputs: Any) -> Any:
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
        statistics = target_oriented_acceptance_sufficient_statistics(
            one_predicted.fluxes,
            one_expected.fluxes,
            one_predicted.stocks,
            one_expected.stocks,
            scales=acceptance_scales,
            horizon=horizon,
        )
        rollout_hard = rollout_hard_gate_counts(
            one_predicted.stocks,
            one_expected.stocks,
            nonnegative_stock_fields=TERMINAL_STOCK_FIELDS,
            budget_residuals={
                "water": trace.result.diagnostics["water_budget_residual"][0],
                "carbon": trace.result.diagnostics["carbon_budget_residual"][0],
            },
            predicted_discrete_state=jax.tree_util.tree_map(lambda value: value[0], trace.result.next_discrete_state),
            expected_discrete_state=jax.tree_util.tree_map(
                lambda value: value[0],
                batch.expected_next_discrete_state,
            ),
            budget_atol=2.0e-12,
            stock_atol=2.0e-12,
        )
        actual_state = jnp.asarray(trace.result.next_state.other[0])
        actual_defined = jnp.asarray(trace.result.next_state_defined.other[0], dtype=jnp.bool_)
        wanted_defined = jnp.asarray(batch.expected_next_state_defined[0], dtype=jnp.bool_)
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
            jnp.sum(mismatch & ~dynamic, dtype=jnp.int32) + rollout_hard.defined_mask_mismatch,
            jnp.sum(mismatch & dynamic, dtype=jnp.int32),
            rollout_hard.discrete_state_mismatch,
            rollout_hard.budget_residual_failure,
            rollout_hard.negative_declared_stock,
        )
        return statistics, hard

    return evaluate


def _validation_records(
    panel: Mapping[str, Any],
    prepared: Mapping[str, Any],
    *,
    mode: str,
) -> list[tuple[Mapping[str, Any], tuple[str, int, Mapping[str, Any]]]]:
    ordered = ordered_target_oriented_validation_entries(panel)
    paired = list(zip(prepared["entries"], ordered, strict=True))
    if mode == "formal":
        return paired
    selected = []
    seen = set()
    for value in paired:
        cell = (value[1][0], value[1][1])
        if cell not in seen:
            seen.add(cell)
            selected.append(value)
    return selected


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
    label: str,
) -> Mapping[str, Any]:
    jax = runtime["jax"]
    cells: dict[tuple[str, int], Any] = {}
    hard = {name: 0 for name in HardCounts._fields}
    timings = []
    selected = _validation_records(panel, prepared, mode=mode)
    for ordinal, (record, (slice_id, horizon, entry)) in enumerate(selected, start=1):
        artifact = load_prepared_rollout_artifact(
            prepared_path,
            record,
            slice_id=slice_id,
            window_entry=entry,
        )
        started = time.perf_counter()
        statistics, local_hard = evaluator(
            parameters,
            artifact.batch,
            artifact.day_inputs,
            horizon=horizon,
        )
        jax.block_until_ready((statistics, local_hard))
        timings.append(time.perf_counter() - started)
        host_statistics = jax.device_get(statistics)
        key = (slice_id, horizon)
        cells[key] = (
            host_statistics
            if key not in cells
            else add_target_oriented_acceptance_statistics(cells[key], host_statistics)
        )
        hard = _add_hard_host(hard, _hard_host(jax.device_get(local_hard)))
        print(
            f"target_training validation={label} ordinal={ordinal}/{len(selected)} "
            f"slice={slice_id} horizon={horizon}",
            flush=True,
        )
    reports = {}
    macros = []
    for slice_id in ("temporal", "spatial", "joint"):
        reports[slice_id] = {}
        for horizon in (7, 30):
            report = finalize_target_oriented_acceptance_statistics(
                cells[(slice_id, horizon)],
                horizon=horizon,
                target_contract=target_contract,
            )
            reports[slice_id][str(horizon)] = report
            macros.append(report["science_macro_normalized_rmse"])
    hard_failure_count = _hard_failure_count(hard)
    full_panel = mode == "formal"
    all_absolute = full_panel and hard_failure_count == 0 and all(
        reports[slice_id][str(horizon)]["all_key_gates_passed"]
        for slice_id in reports
        for horizon in (7, 30)
    )
    return {
        "label": label,
        "full_frozen_panel": full_panel,
        "evaluated_windows": len(selected),
        "science_macro_normalized_rmse": float(np.mean(macros)),
        "hard_counts": hard,
        "hard_failure_count": hard_failure_count,
        "absolute_short_horizon_passed": all_absolute,
        "slices": reports,
        "timing_seconds": {
            "total": float(sum(timings)),
            "minimum": float(min(timings)),
            "median": float(np.median(timings)),
            "maximum": float(max(timings)),
        },
    }


def _load_latest_recovery(
    paths: tuple[Path, Path],
    *,
    identity: Mapping[str, Any],
    maximum_updates: int,
) -> tuple[Mapping[str, Any] | None, Path | None]:
    loaded = []
    for path in paths:
        value = load_target_oriented_recovery(
            path,
            identity=identity,
            parameter_count=PARAMETER_COUNT,
            clip_threshold=FROZEN_CLIP_THRESHOLD,
            maximum_updates=maximum_updates,
        )
        if value is not None:
            loaded.append((value, path))
    return max(loaded, key=lambda item: int(item[0]["completed_updates"])) if loaded else (None, None)


def _stage_bounds(args: argparse.Namespace) -> tuple[int, int, int]:
    mode = str(args.mode)
    start = int(args.stage_start)
    end = int(args.stage_end)
    if mode == "admission":
        if (start, end) == (0, FORMAL_STAGE_END):
            run_end = 1
        elif (start, end) == (FORMAL_STAGE_END, FORMAL_CONTINUATION_END):
            run_end = FORMAL_STAGE_END + 1
        else:
            raise ValueError("target-oriented admission stage boundary is not frozen")
    elif (start, end) in (
        (0, FORMAL_STAGE_END),
        (FORMAL_STAGE_END, FORMAL_CONTINUATION_END),
    ):
        run_end = end
    else:
        raise ValueError("target-oriented runner stage boundary is not frozen")
    if int(args.prepared_updates) != end:
        raise ValueError("target-oriented prepared schedule does not match the stage end")
    has_checkpoint = args.stage_checkpoint is not None
    has_checkpoint_sha256 = bool(args.stage_checkpoint_sha256)
    if start == 0 and (has_checkpoint or has_checkpoint_sha256):
        raise ValueError("target-oriented first stage cannot load a parent stage checkpoint")
    if start > 0 and not (has_checkpoint and has_checkpoint_sha256):
        raise ValueError("target-oriented continuation requires the exact parent stage checkpoint")
    return start, end, run_end


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    mode = str(args.mode)
    stage_start, stage_end, run_updates = _stage_bounds(args)
    if float(args.clip_threshold) != FROZEN_CLIP_THRESHOLD:
        raise ValueError("target-oriented clip threshold drift")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    if report_path.exists():
        raise FileExistsError(report_path)

    started = time.perf_counter()
    runtime = _load_extended_runtime(args)
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
    from research.daily_coarse_graining.daily_operator.target_oriented_attention import (
        initialize_target_oriented_attention_parameters,
        target_oriented_attention_parameter_count,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_contract import (
        compose_target_oriented_loss,
        load_target_oriented_contract,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_training import (
        build_target_oriented_training_schedule,
        derive_target_oriented_scales,
        initialize_target_oriented_adamw,
        load_target_oriented_state_anchor_policy,
        load_target_oriented_training_contract,
        target_oriented_learning_rate,
    )

    architecture = load_strong_baseline_contract()
    mechanism = load_mechanistic_output_contract()
    target_contract = load_target_oriented_contract()
    training_contract = load_target_oriented_training_contract(args.training_contract)
    state_anchor_policy = load_target_oriented_state_anchor_policy()
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
    runtime = dict(runtime) | {
        "architecture_contract": architecture,
        "mechanism_contract": mechanism,
        "learned_schema": learned_schema,
        "target_layout": layout,
    }
    head = _head(runtime, target_contract)
    parameters = initialize_target_oriented_attention_parameters(
        jax.random.PRNGKey(PROBE_SEED),
        schema=learned_schema,
        layout=layout,
        inputs=runtime["probe_assembled"],
        architecture_contract=architecture,
        target_contract=target_contract,
    )
    if target_oriented_attention_parameter_count(parameters) != PARAMETER_COUNT:
        raise ValueError("target-oriented formal runner parameter count drift")

    references = runtime["index"].select(spatial_split="train", temporal_split="train")
    schedule = build_target_oriented_training_schedule(
        references,
        training_contract,
        updates=int(args.prepared_updates),
    )
    raw_prepared = json.loads(args.prepared_manifest.read_text(encoding="utf-8"))
    prepared_arguments = {
        "training_contract_sha256": training_contract.sha256,
        "schedule_sha256": schedule["schedule_sha256"],
        "updates": int(args.prepared_updates),
        "source_git_head": str(raw_prepared.get("identity", {}).get("source_git_head", "")),
    }
    if stage_start == 0:
        prepared = load_prepared_target_manifest(
            args.prepared_manifest,
            **prepared_arguments,
        )
    else:
        prepared = load_prepared_target_manifest_range(
            args.prepared_manifest,
            start_update=stage_start + 1,
            end_update=run_updates,
            **prepared_arguments,
        )
    prepared_records = {int(record["update"]): record for record in prepared["entries"]}
    validation_prepared, panel = load_target_oriented_validation_prepared(
        args.validation_prepared_manifest,
        training_contract_sha256=training_contract.sha256,
        target_contract_sha256=target_contract.sha256,
    )
    verify_target_oriented_validation_panel(
        panel,
        runtime["index"],
        training_contract,
        target_contract,
    )

    first_update = stage_start + 1
    template = _prepare_template(runtime, schedule["entries"][first_update - 1])
    if template.static_owner.include_modelout_diagnostics:
        template = replace(
            template,
            static_owner=replace(template.static_owner, include_modelout_diagnostics=False),
        )
    scales = derive_target_oriented_scales(
        runtime["base_statistics"].arrays["next_state"].scale,
        runtime["process_statistics"].outputs["daily.fast_gpp"].scale,
        runtime["contract"],
        state_anchor_policy=state_anchor_policy,
    )
    acceptance_scales = derive_target_oriented_acceptance_scales(
        runtime["base_statistics"].arrays["next_state"].scale,
        runtime["process_statistics"].outputs["daily.fast_gpp"].scale,
        runtime["contract"],
    )
    first_artifact = load_prepared_target_artifact(
        args.prepared_manifest,
        prepared_records[first_update],
        training_contract_sha256=training_contract.sha256,
        schedule_entry=schedule["entries"][first_update - 1],
    )
    _validate_prepared_objective_shapes(first_artifact, complete_schema=runtime["schema"], scales=scales)
    if first_artifact.trace_signature != _retained_tail_trace_signature(template):
        raise ValueError("target-oriented training retained executable structure drift")
    if any(record["trace_signature"] != first_artifact.trace_signature for record in validation_prepared["entries"]):
        raise ValueError("target-oriented validation retained executable structure drift")

    steps = _make_steps(runtime, head, template, scales, target_contract)
    evaluator = _make_validation_evaluator(runtime, head, template, acceptance_scales, scales)
    optimizer_step = _make_optimizer_step(jax, jnp, training_contract)
    conditioner = _make_gradient_conditioner(jax, jnp)
    optimizer = initialize_target_oriented_adamw(parameters)
    jax.block_until_ready((parameters, optimizer))

    identity = {
        "git_head": _git_head(),
        "mode": mode,
        "stage_start": stage_start,
        "stage_end": stage_end,
        "training_contract_sha256": training_contract.sha256,
        "target_contract_sha256": target_contract.sha256,
        "state_anchor_policy_sha256": state_anchor_policy.sha256,
        "schedule_sha256": schedule["schedule_sha256"],
        "prepared_manifest_sha256": sha256_file(args.prepared_manifest),
        "validation_panel_sha256": panel["panel_sha256"],
        "validation_prepared_manifest_sha256": validation_prepared["manifest_sha256"],
        "clip_threshold": FROZEN_CLIP_THRESHOLD,
    }
    parent_stage = None
    if stage_start > 0:
        parent_stage = load_target_oriented_stage_checkpoint(
            args.stage_checkpoint,
            expected_sha256=args.stage_checkpoint_sha256,
            completed_updates=stage_start,
            parameter_count=PARAMETER_COUNT,
            clip_threshold=FROZEN_CLIP_THRESHOLD,
            training_contract_sha256=training_contract.sha256,
            target_contract_sha256=target_contract.sha256,
            state_anchor_policy_sha256=state_anchor_policy.sha256,
            validation_panel_sha256=panel["panel_sha256"],
        )
        identity["parent_stage_checkpoint_sha256"] = args.stage_checkpoint_sha256
        identity["parent_stage_git_head"] = parent_stage["identity"]["git_head"]
    recovery_paths = (output_dir / "recovery-0.pkl", output_dir / "recovery-1.pkl")
    recovery, recovery_path = _load_latest_recovery(
        recovery_paths,
        identity=identity,
        maximum_updates=run_updates,
    )
    if recovery is None:
        if parent_stage is None:
            completed = 0
            records = []
            initial_validation = _evaluate_panel(
                runtime,
                parameters=parameters,
                evaluator=evaluator,
                prepared_path=args.validation_prepared_manifest,
                prepared=validation_prepared,
                panel=panel,
                target_contract=target_contract,
                mode=mode,
                label="initialization",
            )
            if mode == "formal" and initial_validation["hard_failure_count"] == 0:
                write_target_oriented_best_checkpoint(
                    output_dir / "best.pkl",
                    identity=identity,
                    parameters=_host_tree(jax, parameters),
                    completed_updates=0,
                    parameter_count=PARAMETER_COUNT,
                    validation=initial_validation,
                )
            print("target_training initialized and validation completed", flush=True)
        else:
            completed = stage_start
            parameters = parent_stage["parameters"]
            optimizer = parent_stage["optimizer"]
            records = list(parent_stage["update_records"])
            jax.block_until_ready((parameters, optimizer))
            if mode == "admission":
                initial_validation = _evaluate_panel(
                    runtime,
                    parameters=parameters,
                    evaluator=evaluator,
                    prepared_path=args.validation_prepared_manifest,
                    prepared=validation_prepared,
                    panel=panel,
                    target_contract=target_contract,
                    mode=mode,
                    label=f"update_{stage_start}_admission_baseline",
                )
            else:
                initial_validation = dict(parent_stage["validation"])
            print(
                f"target_training continued stage_checkpoint={args.stage_checkpoint} "
                f"completed={completed}",
                flush=True,
            )
    else:
        completed = int(recovery["completed_updates"])
        parameters = recovery["parameters"]
        optimizer = recovery["optimizer"]
        records = list(recovery["update_records"])
        initial_validation = dict(recovery["initial_validation"])
        print(f"target_training resumed recovery={recovery_path} completed={completed}", flush=True)

    progress_path = output_dir / "progress.json"
    _atomic_json(
        progress_path,
        {
            "schema_version": "target_oriented_training_progress_v1",
            "identity": identity,
            "mode": mode,
            "completed_updates": completed,
            "initial_validation": initial_validation,
            "updates": records,
        },
    )
    memory_before = _memory_report(runtime["devices"][0])
    for update in range(completed + 1, run_updates + 1):
        update_started = time.perf_counter()
        entry = schedule["entries"][update - 1]
        artifact = load_prepared_target_artifact(
            args.prepared_manifest,
            prepared_records[update],
            training_contract_sha256=training_contract.sha256,
            schedule_entry=entry,
        )
        if artifact.trace_signature != first_artifact.trace_signature:
            raise ValueError("target-oriented training prepared trace drift")
        (
            gradient,
            gradient_sources,
            components,
            component_details,
            hard,
            hard_sources,
            forward_seconds,
            gradient_seconds,
        ) = _effective_gradient(
            jax,
            jnp,
            parameters,
            artifact.batch,
            steps,
            target_contract,
        )
        jax.block_until_ready((gradient, components, hard))
        hard_host = _hard_host(jax.device_get(hard))
        if _hard_failure_count(hard_host) != 0:
            raise RuntimeError(f"target-oriented update {update} failed a physical hard gate: {hard_host}")
        component_values = {
            name: value.total if hasattr(value, "total") else value
            for name, value in components.items()
        }
        total = compose_target_oriented_loss(
            one_day_state_anchor_value=component_values["one_day_state_anchor"],
            seven_day_science=component_values["seven_day_science"],
            thirty_day_science=component_values["thirty_day_science"],
            necessary_process_supervision=component_values["necessary_process_supervision"],
            contract=target_contract,
        )
        learning_rate = target_oriented_learning_rate(training_contract, update)
        proposal = optimizer_step(
            parameters,
            gradient,
            optimizer,
            jnp.asarray(learning_rate, dtype=jnp.float32),
        )
        jax.block_until_ready(proposal)
        parameters, optimizer = proposal[0], proposal[1]
        raw_norm = float(np.asarray(proposal[2]))
        clip_scale = float(np.asarray(proposal[3]))
        nonfinite = int(np.asarray(proposal[4]))
        step_norm = float(np.asarray(proposal[5]))
        if nonfinite != 0 or not all(np.isfinite(value) for value in (raw_norm, clip_scale, step_norm)):
            raise RuntimeError(f"target-oriented optimizer proposal is nonfinite at update {update}")
        conditioning = (
            _condition_sources(jax, conditioner, gradient_sources)
            if update in CONDITIONING_UPDATES
            else None
        )
        record = {
            "update": update,
            "runtime_year": int(entry["runtime_year"]),
            "learning_rate": learning_rate,
            "loss": float(np.asarray(total.total)),
            "components": {name: float(np.asarray(value)) for name, value in component_values.items()},
            "component_details": jax.tree_util.tree_map(
                lambda value: float(np.asarray(value)),
                jax.device_get(component_details),
            ),
            "unclipped_gradient_norm": raw_norm,
            "clip_threshold": FROZEN_CLIP_THRESHOLD,
            "clip_scale": clip_scale,
            "clipped": clip_scale < 1.0,
            "parameter_step_norm": step_norm,
            "hard_counts": hard_host,
            "hard_count_sources": {
                source: _hard_host(jax.device_get(value))
                for source, value in hard_sources.items()
            },
            "gradient_conditioning": conditioning,
            "statistics_forward_seconds": forward_seconds,
            "microbatch_gradient_seconds": gradient_seconds,
            "total_seconds": time.perf_counter() - update_started,
        }
        records.append(record)
        _atomic_json(
            progress_path,
            {
                "schema_version": "target_oriented_training_progress_v1",
                "identity": identity,
                "mode": mode,
                "completed_updates": update,
                "initial_validation": initial_validation,
                "updates": records,
            },
        )
        print(
            f"target_training update={update}/{run_updates} loss={record['loss']:.9g} "
            f"norm={raw_norm:.9g} clip={clip_scale:.6g} seconds={record['total_seconds']:.3f}",
            flush=True,
        )
        if update % RECOVERY_INTERVAL == 0 or update == run_updates:
            host_parameters, host_optimizer = _host_tree(jax, (parameters, optimizer))
            recovery_record = write_target_oriented_recovery(
                recovery_paths[(update // RECOVERY_INTERVAL) % 2],
                identity=identity,
                parameters=host_parameters,
                optimizer=host_optimizer,
                completed_updates=update,
                parameter_count=PARAMETER_COUNT,
                clip_threshold=FROZEN_CLIP_THRESHOLD,
                initial_validation=initial_validation,
                update_records=records,
            )
            loaded = load_target_oriented_recovery(
                recovery_record["path"],
                identity=identity,
                parameter_count=PARAMETER_COUNT,
                clip_threshold=FROZEN_CLIP_THRESHOLD,
                maximum_updates=run_updates,
            )
            if loaded is None or int(loaded["completed_updates"]) != update:
                raise RuntimeError("target-oriented recovery verification failed")
            print(f"target_training recovery_written update={update}", flush=True)

    final_validation = _evaluate_panel(
        runtime,
        parameters=parameters,
        evaluator=evaluator,
        prepared_path=args.validation_prepared_manifest,
        prepared=validation_prepared,
        panel=panel,
        target_contract=target_contract,
        mode=mode,
        label=f"update_{run_updates}",
    )
    initial_macro = float(initial_validation["science_macro_normalized_rmse"])
    final_macro = float(final_validation["science_macro_normalized_rmse"])
    relative_improvement = (initial_macro - final_macro) / max(initial_macro, 1.0e-30)
    checkpoint_records = {}
    if mode == "formal":
        host_parameters, host_optimizer = _host_tree(jax, (parameters, optimizer))
        checkpoint_records["stage"] = write_target_oriented_stage_checkpoint(
            output_dir / f"stage-update-{run_updates:04d}.pkl",
            identity=identity,
            parameters=host_parameters,
            optimizer=host_optimizer,
            completed_updates=run_updates,
            parameter_count=PARAMETER_COUNT,
            clip_threshold=FROZEN_CLIP_THRESHOLD,
            validation=final_validation,
            update_records=records,
        )
        if (
            final_validation["hard_failure_count"] == 0
            and final_macro < initial_macro
        ):
            checkpoint_records["best"] = write_target_oriented_best_checkpoint(
                output_dir / "best.pkl",
                identity=identity,
                parameters=host_parameters,
                completed_updates=run_updates,
                parameter_count=PARAMETER_COUNT,
                validation=final_validation,
            )
    memory_after = _memory_report(runtime["devices"][0])
    stage_records = [record for record in records if int(record["update"]) > stage_start]
    clip_count = sum(int(record["clipped"]) for record in stage_records)
    classification = {
        "status": "passed" if final_validation["absolute_short_horizon_passed"] else "rejected",
        "absolute_short_horizon_passed": final_validation["absolute_short_horizon_passed"],
        "relative_macro_improvement": relative_improvement,
        "minimum_pacing_improvement": 0.05,
        "pacing_improvement_passed": relative_improvement >= 0.05,
        "relative_improvement_is_not_promotion": True,
    }
    report = {
        "schema_version": "target_oriented_training_stage_report_v1",
        "status": "completed",
        "mode": mode,
        "identity": identity,
        "policy": {
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "sealed_test_read": False,
            "stage_start": stage_start,
            "stage_end": stage_end,
            "updates_retained": run_updates,
            "updates_retained_this_stage": len(stage_records),
            "completed_updates": run_updates,
            "recovery_interval": RECOVERY_INTERVAL,
            "maximum_recovery_generations": 2,
        },
        "backend": jax.default_backend(),
        "devices": [str(value) for value in runtime["devices"]],
        "memory_before": memory_before,
        "memory_after": memory_after,
        "initial_validation": initial_validation,
        "final_validation": final_validation,
        "updates": records,
        "clip_fraction": clip_count / len(stage_records),
        "cumulative_clip_fraction": sum(int(record["clipped"]) for record in records) / len(records),
        "classification": classification,
        "checkpoints": checkpoint_records,
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": (
            "review_free_admission_then_run_formal_stage"
            if mode == "admission"
            else (
                "enter_separately_authorized_365_day_gate"
                if final_validation["absolute_short_horizon_passed"]
                else (
                    (
                        "review_update_251_to_1000_continuation"
                        if run_updates == FORMAL_STAGE_END
                        else "review_update_1001_to_3000_continuation"
                    )
                    if relative_improvement >= 0.05
                    else "stop_and_attribute_warmup_failure"
                )
            )
        ),
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
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--validation-prepared-manifest", type=Path, required=True)
    parser.add_argument("--prepared-updates", type=int, default=FORMAL_STAGE_END)
    parser.add_argument("--stage-start", type=int, default=0)
    parser.add_argument("--stage-end", type=int, default=FORMAL_STAGE_END)
    parser.add_argument("--stage-checkpoint", type=Path)
    parser.add_argument("--stage-checkpoint-sha256")
    parser.add_argument("--clip-threshold", type=float, default=FROZEN_CLIP_THRESHOLD)
    parser.add_argument("--mode", choices=("admission", "formal"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
