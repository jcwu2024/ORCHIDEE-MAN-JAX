"""Evaluate the frozen successor 7/30-day matched rollout screen."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file
from research.daily_coarse_graining.daily_operator.successor_rollout_screen import (
    ARM_REPORT_SCHEMA_VERSION,
    EXPECTED_ARMS,
    REQUIRED_SLICES,
    add_rollout_screen_statistics,
    admit_successor_screen_checkpoints,
    classify_successor_rollout_screen,
    empty_rollout_screen_statistics,
    finalize_rollout_screen_statistics,
    load_successor_rollout_screen_contract,
    reference_for_screen_entry,
    successor_rollout_screen_statistics,
    verify_successor_screen_window_plan,
)
from scripts.hpc.prepare_successor_rollout_screen import _prepare_adapter
from scripts.hpc.run_successor_fast_owner_gpu_probe import (
    _head,
    _load_extended_runtime,
)
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_successor_small_fit import _retained_tail_trace_signature
from scripts.hpc.successor_rollout_screen_prepared import (
    load_prepared_rollout_artifact,
    load_prepared_rollout_manifest,
)


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _make_evaluator(
    runtime: Mapping[str, Any],
    arm: str,
    adapter_template: Any,
    long_scales: Any,
    screen_contract: Any,
) -> Any:
    from research.daily_coarse_graining.daily_operator.model import bind_daily_operator

    jax = runtime["jax"]
    head = _head(runtime, arm)

    @jax.jit
    def evaluate(parameters: Any, batch: Any, day_inputs: Any) -> Any:
        retained_tail = replace(adapter_template, day_inputs=day_inputs)
        transition = bind_daily_operator(
            input_assembler=runtime["adapter"].input_assembler,
            process_head=head,
            retained_tail=retained_tail,
        )
        return successor_rollout_screen_statistics(
            parameters,
            batch,
            transition=transition,
            markov_contract=runtime["contract"],
            flux_scales=long_scales.fluxes,
            stock_scales=long_scales.stocks,
            budget_atol=float(screen_contract.gates["budget_absolute_tolerance"]),
            stock_atol=float(screen_contract.gates["stock_negative_absolute_tolerance"]),
        )

    return evaluate


def _ordered_plan_entries(plan: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    return [
        (slice_id, entry)
        for slice_id in REQUIRED_SLICES
        for entry in plan["slices"][slice_id]["entries"]
    ]


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"successor rollout-screen output already exists: {args.output}")
    started = time.perf_counter()
    contract = load_successor_rollout_screen_contract(args.screen_contract)
    runtime = _load_extended_runtime(args)
    jax = runtime["jax"]
    git_head = _git_head()

    if sha256_file(args.prepared_manifest) != args.prepared_manifest_sha256:
        raise ValueError("successor rollout prepared-manifest file hash drift")
    if sha256_file(args.window_plan) != args.window_plan_file_sha256:
        raise ValueError("successor rollout window-plan file hash drift")
    plan = json.loads(args.window_plan.read_text(encoding="utf-8"))
    verify_successor_screen_window_plan(plan, runtime["index"], contract)
    prepared = load_prepared_rollout_manifest(
        args.prepared_manifest,
        screen_contract_sha256=contract.sha256,
        window_plan_sha256=plan["plan_sha256"],
        source_git_head=git_head,
    )
    if Path(str(prepared["window_plan"])).resolve() != args.window_plan.resolve():
        raise ValueError("successor rollout prepared manifest and window-plan paths differ")
    ordered = _ordered_plan_entries(plan)
    if len(prepared["entries"]) != len(ordered):
        raise ValueError("successor rollout prepared inventory is incomplete")
    for record, (slice_id, entry) in zip(prepared["entries"], ordered, strict=True):
        if record.get("slice_id") != slice_id or record.get("window_entry") != entry:
            raise ValueError("successor rollout prepared order differs from the frozen plan")

    admitted = admit_successor_screen_checkpoints(
        contract,
        fit_report_path=args.fit_report,
        checkpoint_dir=args.checkpoint_dir,
        expected_fit_report_sha256=args.fit_report_sha256,
        expected_checkpoint_manifest_sha256=args.checkpoint_manifest_sha256,
    )
    from research.daily_coarse_graining.daily_operator.successor_training import (
        derive_long_horizon_scales,
    )

    long_scales = derive_long_horizon_scales(
        runtime["base_statistics"].arrays["next_state"].scale,
        runtime["process_statistics"].outputs["daily.fast_gpp"].scale,
        runtime["contract"],
    )
    first_slice, first_entry = ordered[0]
    first_reference = reference_for_screen_entry(
        runtime["index"],
        contract,
        slice_id=first_slice,
        entry=first_entry,
    )
    adapter_template = _prepare_adapter(
        runtime,
        first_reference,
        int(first_entry["start_rows"][0]),
    )
    template_signature = _retained_tail_trace_signature(adapter_template)
    if any(record["trace_signature"] != template_signature for record in prepared["entries"]):
        raise ValueError("successor rollout retained-tail trace signature changed across the panel")
    evaluators = {
        arm: _make_evaluator(runtime, arm, adapter_template, long_scales, contract)
        for arm in EXPECTED_ARMS
    }
    accumulators = {
        arm: {slice_id: empty_rollout_screen_statistics() for slice_id in REQUIRED_SLICES}
        for arm in EXPECTED_ARMS
    }
    timings = {arm: [] for arm in EXPECTED_ARMS}
    compile_cache = {arm: [] for arm in EXPECTED_ARMS}
    for ordinal, (record, (slice_id, entry)) in enumerate(
        zip(prepared["entries"], ordered, strict=True),
        start=1,
    ):
        artifact = load_prepared_rollout_artifact(
            args.prepared_manifest,
            record,
            slice_id=slice_id,
            window_entry=entry,
        )
        for arm in EXPECTED_ARMS:
            evaluator = evaluators[arm]
            cache_before = int(evaluator._cache_size())
            evaluation_started = time.perf_counter()
            statistics = evaluator(
                admitted.parameters[arm],
                artifact.batch,
                artifact.day_inputs,
            )
            jax.block_until_ready(statistics)
            timings[arm].append(time.perf_counter() - evaluation_started)
            cache_after = int(evaluator._cache_size())
            compile_cache[arm].append(
                {
                    "ordinal": ordinal,
                    "before": cache_before,
                    "after": cache_after,
                    "new_executables": cache_after - cache_before,
                }
            )
            accumulators[arm][slice_id] = add_rollout_screen_statistics(
                accumulators[arm][slice_id],
                jax.device_get(statistics),
            )
        print(
            json.dumps(
                {
                    "event": "successor_rollout_screen_evaluated",
                    "ordinal": ordinal,
                    "total": len(ordered),
                    "slice_id": slice_id,
                    "landpoint_id": entry["landpoint_id"],
                    "year": entry["year"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    arm_reports = {}
    total_windows = sum(
        len(entry["start_rows"])
        for _slice_id, entry in ordered
    )
    for arm in EXPECTED_ARMS:
        slices = {}
        for slice_id in REQUIRED_SLICES:
            finalized = finalize_rollout_screen_statistics(
                accumulators[arm][slice_id],
                contract=contract,
            )
            finalized["window_count"] = sum(
                len(entry["start_rows"])
                for entry in plan["slices"][slice_id]["entries"]
            )
            slices[slice_id] = finalized
        arm_reports[arm] = {
            "schema_version": ARM_REPORT_SCHEMA_VERSION,
            "status": "completed",
            "arm": arm,
            "mode": plan["mode"],
            "window_plan_sha256": plan["plan_sha256"],
            "window_count": total_windows,
            "slices": slices,
            "sealed_test_used": False,
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "timing_seconds": {
                "total": sum(timings[arm]),
                "per_artifact": timings[arm],
            },
            "jit_cache": compile_cache[arm],
            "model_architecture": {
                "cross_day_memory": "canonical_state_only",
                "retained_tail": "canonical_source_backed_dynamic_transition",
            },
        }
    if plan["mode"] == "formal_bounded_screen":
        classification = classify_successor_rollout_screen(
            candidate=arm_reports[EXPECTED_ARMS[0]],
            control=arm_reports[EXPECTED_ARMS[1]],
            contract=contract,
        )
        status = "completed"
        next_gate = (
            "request_separate_365_day_authorization"
            if classification["status"] == "passed"
            else "stop_and_attribute_bounded_7_30_day_failure"
        )
    else:
        hard_total = sum(
            value
            for arm in EXPECTED_ARMS
            for slice_id in REQUIRED_SLICES
            for value in arm_reports[arm]["slices"][slice_id]["hard_counts"].values()
        )
        status = "passed" if hard_total == 0 else "failed"
        classification = None
        next_gate = (
            "request_paid_formal_bounded_screen_authorization"
            if status == "passed"
            else "repair_free_smoke_hard_failure"
        )
    report = {
        "schema_version": "successor_7_30_rollout_runner_report_v1",
        "status": status,
        "mode": plan["mode"],
        "screen_contract_sha256": contract.sha256,
        "evaluation_git_head": git_head,
        "prepared_manifest": str(args.prepared_manifest.resolve()),
        "prepared_manifest_sha256": args.prepared_manifest_sha256,
        "window_plan": str(args.window_plan.resolve()),
        "window_plan_file_sha256": args.window_plan_file_sha256,
        "window_plan_sha256": plan["plan_sha256"],
        "checkpoint_admission": dict(admitted.evidence),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in runtime["devices"]],
        "arms": arm_reports,
        "classification": classification,
        "teacher_data_regenerated": False,
        "historical_v5_used": False,
        "sealed_test_used": False,
        "boundary": contract.payload["boundary"],
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": next_gate,
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
    parser.add_argument("--screen-contract", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--prepared-manifest-sha256", required=True)
    parser.add_argument("--window-plan", type=Path, required=True)
    parser.add_argument("--window-plan-file-sha256", required=True)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--fit-report-sha256", required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0 if report["status"] in ("passed", "completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
