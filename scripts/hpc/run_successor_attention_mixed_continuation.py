"""Train the Attention candidate with mixed point-years and compressed cosine decay."""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.daily_coarse_graining.daily_operator.attention_mixed_continuation import (
    classify_attention_continuation,
    continuation_learning_rate,
    load_attention_continuation_state,
    load_attention_mixed_continuation_contract,
    mixed_update_groups,
    sha256_file,
    write_attention_continuation_state,
)
from research.daily_coarse_graining.daily_operator.successor_architecture import (
    SOURCE_ROUTED_ATTENTION,
)
from research.daily_coarse_graining.daily_operator.successor_staged_checkpoint import (
    load_staged_recovery,
)
from scripts.hpc.prepare_successor_rollout_screen import (
    _prepare_adapter as _prepare_validation_adapter,
)
from scripts.hpc.prepare_successor_staged_validation import (
    _reference as _validation_reference,
)
from scripts.hpc.probe_successor_batch_diversity import (
    _join_context_anchors,
    _slice_context_batch,
    _stack_contexts,
)
from scripts.hpc.run_successor_batch_diversity_short_fit import (
    _make_mixed_train_step,
)
from scripts.hpc.run_successor_fast_owner_gpu_probe import (
    EXPECTED_PARAMETER_COUNT,
    _load_extended_runtime,
)
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_successor_small_fit import (
    _component_record,
    _device_memory_report,
    _hard_counts_host,
    _prepare_adapter,
    _retained_tail_trace_signature,
)
from scripts.hpc.run_successor_staged_training import (
    _evaluate_panel,
    _training_identity,
)
from scripts.hpc.successor_small_fit_prepared import load_prepared_artifact
from scripts.hpc.successor_staged_prepared import (
    load_staged_prepared_manifest,
    stage_update_range,
)
from scripts.hpc.successor_staged_validation_prepared import (
    load_staged_validation_prepared,
    ordered_validation_entries,
)

PARENT_UPDATE = 1000
TARGET_STAGED_MILESTONE = 2000
PROGRESS_SCHEMA = "successor_attention_mixed_continuation_progress_v1"


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _load_baseline(path: Path, *, arms: tuple[str, ...]) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("status") != "passed"
        or int(payload.get("milestone", -1)) != PARENT_UPDATE
        or set(payload.get("validation", {})) != set(arms)
    ):
        raise ValueError("attention continuation baseline is not update 1000")
    return payload["validation"][SOURCE_ROUTED_ATTENTION]


def _load_progress(
    path: Path,
    *,
    identity: Mapping[str, Any],
    completed_updates: int,
) -> list[Mapping[str, Any]]:
    if not path.exists():
        if completed_updates:
            raise ValueError("attention continuation state has no matching progress")
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("updates", [])
    if (
        payload.get("schema_version") != PROGRESS_SCHEMA
        or payload.get("identity") != dict(identity)
        or int(payload.get("completed_updates", -1)) != int(completed_updates)
        or len(records) != int(completed_updates)
    ):
        raise ValueError("attention continuation progress drift")
    return list(records)


def _write_progress(
    path: Path,
    *,
    identity: Mapping[str, Any],
    records: list[Mapping[str, Any]],
) -> None:
    _atomic_json(
        path,
        {
            "schema_version": PROGRESS_SCHEMA,
            "identity": dict(identity),
            "completed_updates": len(records),
            "updates": records,
        },
    )


def _training_summary(
    records: list[Mapping[str, Any]],
    *,
    clip_threshold: float,
) -> Mapping[str, Any]:
    def summarize(values: list[Mapping[str, Any]]) -> Mapping[str, Any]:
        components = tuple(values[0]["long_horizon_components"])
        return {
            "updates": len(values),
            "median_loss": float(np.median([item["loss"] for item in values])),
            "median_one_day_anchor": float(
                np.median([item["one_day_anchor"] for item in values])
            ),
            "median_gradient_norm": float(
                np.median([item["gradient_norm"] for item in values])
            ),
            "gradient_clip_count": sum(
                float(item["gradient_norm"]) > clip_threshold for item in values
            ),
            "median_long_horizon_components": {
                name: float(
                    np.median(
                        [item["long_horizon_components"][name] for item in values]
                    )
                )
                for name in components
            },
        }

    window = min(50, len(records))
    return {
        "first_window": summarize(records[:window]),
        "last_window": summarize(records[-window:]),
        "all_updates_gradient_clip_count": sum(
            float(item["gradient_norm"]) > clip_threshold for item in records
        ),
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    continuation = load_attention_mixed_continuation_contract(
        args.continuation_contract
    )
    if (
        sha256_file(args.prepared_manifest)
        != continuation.payload["identity"]["prepared_manifest_sha256"]
        or sha256_file(args.baseline_report)
        != continuation.payload["identity"]["baseline_stage_report_sha256"]
    ):
        raise ValueError("attention continuation input report identity drift")

    runtime = _load_extended_runtime(args)
    jax = runtime["jax"]
    from research.daily_coarse_graining.daily_operator.successor_architecture import (
        initialize_successor_parameters,
        successor_parameter_count,
    )
    from research.daily_coarse_graining.daily_operator.successor_small_fit import (
        LONG_HORIZON_COMPONENTS,
        STAGED_SCHEMA_VERSION,
        build_successor_small_fit_schedule,
        load_successor_small_fit_contract,
        successor_learning_rate,
        verify_successor_small_fit_schedule,
    )
    from research.daily_coarse_graining.daily_operator.successor_training import (
        derive_long_horizon_scales,
    )
    from research.daily_coarse_graining.daily_operator.successor_training_assets import (
        load_successor_training_assets,
    )

    staged = load_successor_small_fit_contract(args.runner_contract)
    if (
        staged.payload["schema_version"] != STAGED_SCHEMA_VERSION
        or continuation.payload["identity"]["staged_training_contract_sha256"]
        != staged.sha256
    ):
        raise ValueError("attention continuation staged-v3 identity drift")
    arms = tuple(staged.payload["matched_arms"])
    references = runtime["index"].select(
        spatial_split="train",
        temporal_split="train",
    )
    schedule = build_successor_small_fit_schedule(
        references,
        staged,
        updates=int(staged.sampling["maximum_updates_per_arm"]),
    )
    verify_successor_small_fit_schedule(schedule, references, staged)
    start_update, end_update = stage_update_range(
        staged.sampling["stage_milestones"],
        TARGET_STAGED_MILESTONE,
    )
    prepared = load_staged_prepared_manifest(
        args.prepared_manifest,
        staged_training_contract_sha256=staged.sha256,
        full_schedule_sha256=schedule["schedule_sha256"],
        start_update=start_update,
        end_update=end_update,
    )
    groups = mixed_update_groups(
        prepared["entries"],
        point_years_per_update=int(
            continuation.training["point_years_per_update"]
        ),
        expected_updates=int(continuation.training["continuation_updates"]),
    )
    validation_prepared, prepared_panel = load_staged_validation_prepared(
        args.validation_prepared_manifest,
        staged_training_contract_sha256=staged.sha256,
        validation_panel_sha256=continuation.payload["identity"][
            "validation_panel_sha256"
        ],
    )
    assets = load_successor_training_assets(
        args.coefficient_contract,
        args.validation_panel,
        coefficient_report_path=args.coefficient_report,
        prepared_manifest_path=args.coefficient_prepared_manifest,
        runner_contract=staged,
        components=LONG_HORIZON_COMPONENTS,
    )
    if prepared_panel != assets.validation_panel:
        raise ValueError("attention continuation validation panel drift")
    parent_identity = _training_identity(
        runtime,
        staged,
        schedule,
        assets,
        validation_prepared,
    )
    identity = {
        "continuation_contract_sha256": continuation.sha256,
        "parent_training_identity": parent_identity,
        "parent_update": PARENT_UPDATE,
        "architecture": SOURCE_ROUTED_ATTENTION,
    }

    first_record = prepared["entries"][0]
    first_entry = schedule["entries"][int(first_record["update"]) - 1]
    first_artifact = load_prepared_artifact(
        args.prepared_manifest,
        first_record,
        schedule_entry=first_entry,
    )
    _reference, adapter_template, adapter_timing = _prepare_adapter(
        runtime,
        first_entry,
    )
    trace_signature = _retained_tail_trace_signature(adapter_template)
    if first_artifact.trace_signature != trace_signature:
        raise ValueError("attention continuation retained-tail trace drift")
    assembled = runtime["adapter"].input_assembler.assemble(
        jax.tree_util.tree_map(
            lambda value: value[0],
            first_artifact.batch.anchor_inputs,
        )
    )
    initialized = {
        arm: initialize_successor_parameters(
            jax.random.PRNGKey(int(staged.sampling["seed"])),
            schema=runtime["schema"],
            layout=runtime["layout"],
            inputs=assembled,
            arm=arm,
        )
        for arm in arms
    }
    counts = {arm: successor_parameter_count(initialized[arm]) for arm in arms}
    if counts.get(SOURCE_ROUTED_ATTENTION) != EXPECTED_PARAMETER_COUNT:
        raise ValueError("attention continuation parameter count drift")
    parent = load_staged_recovery(
        args.parent_recovery,
        identity=parent_identity,
        expected_arms=arms,
        parameter_counts=counts,
        minimum_completed_updates=PARENT_UPDATE,
        maximum_completed_updates=PARENT_UPDATE,
    )
    if parent is None:
        raise ValueError("attention continuation requires update-1000 recovery")

    state = load_attention_continuation_state(
        args.state_path,
        identity=identity,
        parameter_count=EXPECTED_PARAMETER_COUNT,
        maximum_updates=int(continuation.training["continuation_updates"]),
    )
    completed = 0 if state is None else int(state["completed_updates"])
    parameters = (
        parent["parameters"][SOURCE_ROUTED_ATTENTION]
        if state is None
        else state["parameters"]
    )
    optimizer = (
        parent["optimizers"][SOURCE_ROUTED_ATTENTION]
        if state is None
        else state["optimizer"]
    )
    progress_path = args.output.with_name("progress.json")
    records = _load_progress(
        progress_path,
        identity=identity,
        completed_updates=completed,
    )

    long_scales = derive_long_horizon_scales(
        runtime["base_statistics"].arrays["next_state"].scale,
        runtime["process_statistics"].outputs["daily.fast_gpp"].scale,
        runtime["contract"],
    )
    train_step = _make_mixed_train_step(
        runtime,
        SOURCE_ROUTED_ATTENTION,
        adapter_template,
        assets.coefficients,
        long_scales,
        staged,
    )
    initial_learning_rate = successor_learning_rate(staged, PARENT_UPDATE + 1)
    memory_before = _device_memory_report(runtime["devices"])
    recovery_interval = int(continuation.training["recovery_interval"])
    for continuation_update in range(completed + 1, len(groups) + 1):
        group_records = groups[continuation_update - 1]
        group = []
        for record in group_records:
            entry = schedule["entries"][int(record["update"]) - 1]
            artifact = load_prepared_artifact(
                args.prepared_manifest,
                record,
                schedule_entry=entry,
            )
            if artifact.trace_signature != trace_signature:
                raise ValueError("attention continuation prepared trace drift")
            group.append((entry, artifact))
        contexts = tuple(
            _slice_context_batch(jax, artifact.batch, context_index=index)
            for index, (_entry, artifact) in enumerate(group)
        )
        anchor_batch = _join_context_anchors(jax, contexts)
        context_batch = _stack_contexts(jax, contexts)
        context_inputs = _stack_contexts(
            jax,
            tuple(artifact.day_inputs for _entry, artifact in group),
        )
        rate = np.asarray(
            continuation_learning_rate(
                continuation,
                continuation_update,
                initial_learning_rate=initial_learning_rate,
            ),
            dtype=np.float64,
        )
        result = train_step(
            parameters,
            optimizer,
            anchor_batch,
            context_batch,
            context_inputs,
            rate,
        )
        jax.block_until_ready(result)
        hard_counts = _hard_counts_host(result.hard_counts)
        if not bool(np.asarray(result.update_applied)):
            raise RuntimeError(
                f"attention continuation update {continuation_update} failed: "
                f"{hard_counts}"
            )
        parameters = result.parameters
        optimizer = result.optimizer
        records.append(
            {
                "continuation_update": continuation_update,
                "optimizer_state_step": PARENT_UPDATE + continuation_update,
                "learning_rate": float(rate),
                "point_years": [
                    {
                        "prepared_update": int(entry["update"]),
                        "landpoint_id": str(entry["landpoint_id"]),
                        "year": int(entry["year"]),
                    }
                    for entry, _artifact in group
                ],
                "loss": float(np.asarray(result.loss)),
                "one_day_anchor": float(np.asarray(result.anchor)),
                "long_horizon_components": _component_record(
                    LONG_HORIZON_COMPONENTS,
                    result.components,
                ),
                "gradient_norm": float(np.asarray(result.gradient_norm)),
                "hard_gate_counts": hard_counts,
            }
        )
        if continuation_update % recovery_interval == 0:
            write_attention_continuation_state(
                args.state_path,
                identity=identity,
                parameters=jax.device_get(parameters),
                optimizer=jax.device_get(optimizer),
                completed_updates=continuation_update,
                parameter_count=EXPECTED_PARAMETER_COUNT,
                compute_sha256=False,
            )
            _write_progress(
                progress_path,
                identity=identity,
                records=records,
            )
        del group, contexts, anchor_batch, context_batch, context_inputs
        gc.collect()
        print(
            json.dumps(
                {
                    "event": "successor_attention_mixed_continuation_update",
                    "update": continuation_update,
                    "total": len(groups),
                    "learning_rate": float(rate),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    first_slice, first_validation_entry = ordered_validation_entries(
        assets.validation_panel
    )[0]
    validation_reference = _validation_reference(
        runtime["index"],
        first_slice,
        first_validation_entry,
    )
    validation_adapter = _prepare_validation_adapter(
        runtime,
        validation_reference,
        int(first_validation_entry["start_rows"][0]),
    )
    validation = _evaluate_panel(
        runtime,
        contract=staged,
        parameters={SOURCE_ROUTED_ATTENTION: parameters},
        arms=(SOURCE_ROUTED_ATTENTION,),
        long_scales=long_scales,
        adapter_template=validation_adapter,
        prepared_manifest_path=args.validation_prepared_manifest,
        prepared=validation_prepared,
        panel=assets.validation_panel,
    )[SOURCE_ROUTED_ATTENTION]
    baseline = _load_baseline(args.baseline_report, arms=arms)
    classification = classify_attention_continuation(
        baseline,
        validation,
        contract=continuation,
    )
    final_state = write_attention_continuation_state(
        args.state_path,
        identity=identity,
        parameters=jax.device_get(parameters),
        optimizer=jax.device_get(optimizer),
        completed_updates=len(groups),
        parameter_count=EXPECTED_PARAMETER_COUNT,
        classification=classification,
    )
    _write_progress(progress_path, identity=identity, records=records)
    clip_threshold = float(continuation.training["global_gradient_norm_clip"])
    report = {
        "schema_version": "successor_attention_mixed_annealed_continuation_report_v1",
        "status": "completed",
        "git_head": _git_head(),
        "architecture": SOURCE_ROUTED_ATTENTION,
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "parent_update": PARENT_UPDATE,
        "continuation_updates": len(groups),
        "source_point_years": len(groups)
        * int(continuation.training["point_years_per_update"]),
        "identity": identity,
        "learning_rate": {
            "schedule": continuation.training["schedule"],
            "initial": initial_learning_rate,
            "final": float(
                continuation_learning_rate(
                    continuation,
                    len(groups),
                    initial_learning_rate=initial_learning_rate,
                )
            ),
        },
        "training_summary": _training_summary(
            records,
            clip_threshold=clip_threshold,
        ),
        "updates": records,
        "validation": validation,
        "classification": classification,
        "state": final_state,
        "adapter_preparation": adapter_timing,
        "device_memory_before": memory_before,
        "device_memory_after": _device_memory_report(runtime["devices"]),
        "teacher_data_regenerated": False,
        "historical_v5_used": False,
        "sealed_test_used": False,
        "reconstructs_48_forcing_records": False,
        "executes_48_step_state_scan": False,
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": (
            continuation.payload["pass_action"]
            if classification["passed"]
            else continuation.payload["fail_action"]
        ),
    }
    _atomic_json(args.output, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--acceptance-report", type=Path, required=True)
    parser.add_argument("--base-statistics", type=Path, required=True)
    parser.add_argument("--process-statistics", type=Path, required=True)
    parser.add_argument("--fast-statistics", type=Path, required=True)
    parser.add_argument("--runner-contract", type=Path, required=True)
    parser.add_argument("--continuation-contract", type=Path, required=True)
    parser.add_argument("--coefficient-contract", type=Path, required=True)
    parser.add_argument("--coefficient-report", type=Path, required=True)
    parser.add_argument("--coefficient-prepared-manifest", type=Path, required=True)
    parser.add_argument("--validation-panel", type=Path, required=True)
    parser.add_argument("--validation-prepared-manifest", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--parent-recovery", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--state-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
