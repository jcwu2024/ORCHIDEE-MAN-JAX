"""Evaluate the source-shaped initialization on the frozen train/train panel."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import (
    sha256_file,
)
from research.daily_coarse_graining.daily_operator.target_oriented_flux_basis import (
    load_target_oriented_flux_basis_amendment,
)
from research.daily_coarse_graining.daily_operator.target_oriented_source_shaped_screen import (
    compare_source_shaped_train_train_groups,
    load_source_shaped_train_train_screen_contract,
)
from scripts.hpc.audit_target_oriented_flux_gradient_variance import (
    AUDIT_FLUX_KEYS,
    _load_context,
    _tree_size,
)
from scripts.hpc.audit_target_oriented_flux_protected_virtual_step import (
    _evaluate_group,
)
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_fast_owner_gpu_probe import PROBE_SEED
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_attention_gpu_admission import _head as _parent_head
from scripts.hpc.run_target_oriented_full_step_preflight import _make_steps, _slice
from scripts.hpc.target_oriented_training_prepared import load_prepared_target_artifact

SCHEMA_VERSION = "target_oriented_source_shaped_train_train_zero_panel_v1"
CANDIDATE_PARAMETER_COUNT = 84_148_646


def _git_head() -> str:
    return subprocess.check_output(("git", "rev-parse", "HEAD"), text=True).strip()


def _groups(context: Mapping[str, Any], updates: tuple[int, ...]) -> list[Mapping[str, Any]]:
    groups = []
    for update in updates:
        artifact = load_prepared_target_artifact(
            context["prepared_manifest"],
            context["prepared_records"][update],
            training_contract_sha256=context["training_contract"].sha256,
            schedule_entry=context["schedule"]["entries"][update - 1],
        )
        if artifact.trace_signature != context["trace_signature"]:
            raise ValueError("source-shaped zero-panel prepared trace drift")
        groups.append(
            {
                "update": update,
                "runtime_year": int(context["schedule"]["entries"][update - 1]["runtime_year"]),
                "one_batches": [_slice(artifact.batch.one_day, index, index + 4) for index in range(0, 32, 4)],
                "seven_batches": [
                    _slice(artifact.batch.seven_day, index, index + 1)
                    for index in range(8)
                ],
                "thirty_batches": [
                    _slice(artifact.batch.thirty_day, index, index + 1)
                    for index in range(4)
                ],
            }
        )
    return groups


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    screen_contract = load_source_shaped_train_train_screen_contract(args.screen_contract)
    source_contract_sha256 = screen_contract.payload["identity"]["source_shaped_contract_sha256"]
    if sha256_file(args.parent_baseline_report) != screen_contract.payload["identity"][
        "parent_train_train_report_sha256"
    ]:
        raise ValueError("source-shaped parent train/train report SHA256 drift")
    parent_report = json.loads(args.parent_baseline_report.read_text(encoding="utf-8"))
    parent_groups = parent_report["baseline_groups"]

    amendment = load_target_oriented_flux_basis_amendment(args.flux_basis_amendment)
    context = dict(
        _load_context(
            args,
            science_flux_basis=amendment.flux_basis,
            verify_validation_artifacts=False,
        )
    )
    context["prepared_manifest"] = args.prepared_manifest
    runtime = context["runtime"]
    jax = runtime["jax"]
    target_contract = context["target_contract"]
    memory_before = _memory_report(runtime["devices"][0])

    from research.daily_coarse_graining.daily_operator.target_oriented_process_head import (
        build_target_oriented_source_shaped_maintenance_head,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_source_shaped_maintenance import (
        initialize_source_shaped_maintenance_parameters,
        load_source_shaped_maintenance_contract,
    )
    from research.daily_coarse_graining.daily_operator.target_oriented_training import (
        derive_target_oriented_acceptance_scales,
    )

    source_contract = load_source_shaped_maintenance_contract(args.source_contract)
    if source_contract.sha256 != source_contract_sha256:
        raise ValueError("source-shaped zero-panel mechanism contract drift")
    parameters = initialize_source_shaped_maintenance_parameters(
        context["parameters"],
        jax.random.PRNGKey(PROBE_SEED + 20260827),
        schema=runtime["learned_schema"],
        target_contract=target_contract,
        source_contract=source_contract,
    )
    if _tree_size(jax, parameters) != CANDIDATE_PARAMETER_COUNT:
        raise ValueError("source-shaped zero-panel parameter count drift")
    head = build_target_oriented_source_shaped_maintenance_head(
        _parent_head(runtime, target_contract),
        context["template"],
        source_contract=source_contract,
    )
    acceptance_scales = derive_target_oriented_acceptance_scales(
        runtime["base_statistics"].arrays["next_state"].scale,
        runtime["process_statistics"].outputs["daily.fast_gpp"].scale,
        runtime["contract"],
    )
    steps = _make_steps(
        runtime,
        head,
        context["template"],
        context["scales"],
        target_contract,
        acceptance_scales=acceptance_scales,
        acceptance_flux_names=AUDIT_FLUX_KEYS,
        science_flux_basis=amendment.flux_basis,
    )

    candidate_groups = []
    for group in _groups(context, screen_contract.evaluation_updates):
        record, _totals = _evaluate_group(
            jax,
            runtime["jnp"],
            parameters,
            steps,
            group,
            target_contract,
            flux_basis=amendment.flux_basis,
        )
        candidate_groups.append(
            {
                "update": int(group["update"]),
                "runtime_year": int(group["runtime_year"]),
            }
            | record
        )
        print(f"source_shaped zero_panel update={group['update']} complete", flush=True)

    comparison = compare_source_shaped_train_train_groups(
        parent_groups,
        candidate_groups,
        screen_contract,
    )
    memory_after = _memory_report(runtime["devices"][0])
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if comparison["hard_gate_passed"] and comparison["all_scores_finite"] else "failed",
        "identity": {
            "git_head": _git_head(),
            "screen_contract_sha256": screen_contract.sha256,
            "source_shaped_contract_sha256": source_contract.sha256,
            "parent_checkpoint_sha256": args.stage_checkpoint_sha256,
            "parent_train_train_report_sha256": sha256_file(args.parent_baseline_report),
            "prepared_manifest_sha256": sha256_file(args.prepared_manifest),
            "training_contract_sha256": context["training_contract"].sha256,
            "target_contract_sha256": target_contract.sha256,
        },
        "policy": {
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "validation_values_evaluated": False,
            "sealed_test_read": False,
            "optimizer_created": False,
            "training_updates": 0,
            "checkpoint_written": False,
            "forcing_records_reconstructed": 0,
            "subdaily_state_scan_steps": 0,
        },
        "backend": jax.default_backend(),
        "devices": [str(value) for value in runtime["devices"]],
        "parameter_count": _tree_size(jax, parameters),
        "candidate_groups": candidate_groups,
        "comparison_to_fixed_parent": comparison,
        "memory_before": memory_before,
        "memory_after": memory_after,
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": (
            "review_then_request_free_one_update_admission"
            if comparison["hard_gate_passed"] and comparison["all_scores_finite"]
            else "stop_and_repair_zero_update_panel"
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
    parser.add_argument("--source-contract", type=Path, required=True)
    parser.add_argument("--screen-contract", type=Path, required=True)
    parser.add_argument("--prepared-manifest", type=Path, required=True)
    parser.add_argument("--prepared-updates", type=int, default=1000)
    parser.add_argument("--validation-prepared-manifest", type=Path, required=True)
    parser.add_argument("--stage-checkpoint", type=Path, required=True)
    parser.add_argument("--stage-checkpoint-sha256", required=True)
    parser.add_argument("--parent-baseline-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
