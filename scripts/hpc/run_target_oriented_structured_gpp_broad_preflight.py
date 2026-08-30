"""Run broad direct-GPP gradient and independent finite-step preflight."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_broad_preflight import (
    load_structured_gpp_broad_preflight_contract,
    reference_identity,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_checkpoint import (
    load_structured_gpp_checkpoint,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_training import (
    CANDIDATE_PARAMETER_COUNT,
    SPECIALIST_PARAMETER_COUNT,
    replace_specialist,
    specialist_element_count,
    structured_gpp_direct_loss,
)
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    add_error_sufficient_statistics,
    initialize_target_oriented_adamw,
)
from scripts.hpc.audit_target_oriented_flux_gradient_variance import _load_context, _tree_size
from scripts.hpc.calibrate_target_oriented_distribution_aware_guard import (
    _load_artifact,
    _replace_adapter,
)
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_fast_owner_gpu_probe import PROBE_SEED
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_attention_gpu_admission import _head as _parent_head
from scripts.hpc.run_target_oriented_full_step_preflight import (
    _hard_failure_count,
    _make_steps,
    _slice,
)
from scripts.hpc.run_target_oriented_source_shaped_branch_training import (
    _make_optimizer_step,
)
from scripts.hpc.run_target_oriented_structured_gpp_joint_training import (
    _tree_scalar_report,
)
from scripts.hpc.run_target_oriented_structured_gpp_short_horizon_screen import (
    _load_screen_contract,
    _load_training_report,
)
from scripts.hpc.run_target_oriented_structured_gpp_streaming_throughput import (
    COVERAGE_REPORT_SHA256,
    DirectGppBatch,
    _direct_statistics_and_gradient,
    _round_robin,
    _stack,
)
from scripts.hpc.run_target_oriented_structured_gpp_training_preflight import (
    ADAPTER_PARAMETER_COUNT,
    PARENT_PARAMETER_COUNT,
    _make_direct_steps,
    _one_day_hard,
    _retained_parameter_report,
)

SCHEMA_VERSION = "target_oriented_structured_gpp_broad_preflight_result_v1"
SCHEDULE_SCHEMA_VERSION = "target_oriented_structured_gpp_broad_schedule_audit_v1"
GPP_FIELD = "daily.fast_gpp"
GRADIENT_UPDATES = tuple(range(1, 17))
PROPOSAL_UPDATES = tuple(range(1, 5))


def _resolve_schedule_batches(
    schedule_report: Mapping[str, Any],
    references: Sequence[Any],
) -> tuple[tuple[tuple[Any, tuple[int, ...]], ...], ...]:
    by_identity = {reference_identity(value): value for value in references}
    samples = schedule_report["schedule"]["sample_batches"]
    result = []
    for update in GRADIENT_UPDATES:
        records = samples.get(str(update), ())
        resolved = []
        for record in records:
            identity = (
                str(record["identity"][0]),
                int(record["identity"][1]),
                str(record["identity"][2]),
                str(record["identity"][3]),
            )
            rows = tuple(int(value) for value in record["rows"])
            if identity not in by_identity or len(rows) != 4:
                raise ValueError("structured GPP broad preflight schedule entry drift")
            resolved.append((by_identity[identity], rows))
        if len(resolved) != 52 or len({reference_identity(value[0])[0] for value in resolved}) != 52:
            raise ValueError("structured GPP broad preflight batch composition drift")
        result.append(tuple(resolved))
    return tuple(result)


def _prepare_batch(
    runtime: Mapping[str, Any],
    selections: Sequence[tuple[Any, tuple[int, ...]]],
) -> tuple[DirectGppBatch, Mapping[str, float]]:
    from research.daily_coarse_graining.daily_markov_contract import load_markov_shard

    adapter = runtime["adapter"]
    read_seconds = 0.0
    assemble_seconds = 0.0
    groups = []
    for reference, rows in selections:
        started = time.perf_counter()
        parent = load_markov_shard(reference.parent.path, contract=runtime["contract"])
        read_seconds += time.perf_counter() - started
        if int(reference.day_count) != parent.days or any(
            row < 1 or row >= parent.days for row in rows
        ):
            raise ValueError("structured GPP broad preflight source row drift")
        started = time.perf_counter()
        groups.append(tuple(adapter.assemble_direct_gpp(parent.sample(row)) for row in rows))
        assemble_seconds += time.perf_counter() - started
    examples = _round_robin(groups)
    started = time.perf_counter()
    batch = DirectGppBatch(
        inputs=_stack(runtime["jax"], [value.inputs for value in examples]),
        process_targets={
            GPP_FIELD: _stack(runtime["jax"], [value.target for value in examples])
        },
    )
    stack_seconds = time.perf_counter() - started
    if len(examples) != 208:
        raise ValueError("structured GPP broad preflight day count drift")
    return batch, {
        "parent_shard_read_seconds": read_seconds,
        "direct_gpp_assemble_seconds": assemble_seconds,
        "stack_seconds": stack_seconds,
        "total_seconds": read_seconds + assemble_seconds + stack_seconds,
    }


def _direct_forward(
    jax: Any,
    specialist: Any,
    batch: DirectGppBatch,
    direct_steps: Mapping[str, Any],
) -> Any:
    values = [
        direct_steps["one_forward"](specialist, _slice(batch, start, start + 4))
        for start in range(0, 208, 4)
    ]
    jax.block_until_ready(values)
    total = values[0]
    for value in values[1:]:
        total = add_error_sufficient_statistics(total, value)
    jax.block_until_ready(total)
    return total


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    started = time.perf_counter()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    broad = load_structured_gpp_broad_preflight_contract(args.broad_preflight_contract)
    if sha256_file(args.coverage_report) != COVERAGE_REPORT_SHA256:
        raise ValueError("structured GPP broad preflight coverage-report hash drift")
    schedule_report = json.loads(args.schedule_report.read_text(encoding="utf-8"))
    if (
        schedule_report.get("schema_version") != SCHEDULE_SCHEMA_VERSION
        or schedule_report.get("status") != "passed"
        or schedule_report.get("next_gate") != "run_broad_direct_gpp_free_gpu_preflight"
        or schedule_report["identity"]["broad_preflight_contract_sha256"] != broad.sha256
        or schedule_report["identity"]["coverage_report_sha256"] != COVERAGE_REPORT_SHA256
        or int(schedule_report["schedule"]["unique_days"]) != 1_040_000
    ):
        raise ValueError("structured GPP broad preflight schedule-report drift")

    screen = _load_screen_contract(args.screen_contract)
    screen_identity = screen["identity"]
    training_report = _load_training_report(args.joint_training_report, screen)

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
        amendment.sha256 != screen_identity["flux_basis_amendment_sha256"]
        or context["training_contract"].sha256 != screen_identity["base_training_contract_sha256"]
        or target_contract.sha256 != screen_identity["target_contract_sha256"]
    ):
        raise ValueError("structured GPP broad preflight parent contract drift")

    structured_contract = load_structured_gpp_contract(args.structured_contract)
    distribution_contract = load_distribution_aware_maintenance_contract(args.distribution_contract)
    distribution_report = json.loads(args.distribution_training_report.read_text(encoding="utf-8"))
    distribution_checkpoint = load_distribution_aware_checkpoint(
        args.distribution_checkpoint,
        expected_sha256=screen_identity["parent_distribution_checkpoint_sha256"],
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
        initialized_parent, jax.device_put(distribution_checkpoint["adapter"])
    )
    if _tree_size(jax, parent_parameters) != PARENT_PARAMETER_COUNT:
        raise ValueError("structured GPP broad preflight parent parameter count drift")
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
    checkpoint = load_structured_gpp_checkpoint(
        args.selected_checkpoint,
        expected_sha256=screen_identity["selected_checkpoint_sha256"],
        identity=training_report["identity"],
        completed_updates=int(screen_identity["selected_checkpoint_optimizer_step"]),
        parameter_count=SPECIALIST_PARAMETER_COUNT,
        clip_threshold=float(training_report["training"]["clip_threshold"]),
    )
    specialist = jax.device_put(checkpoint["specialist"])
    parameters = replace_specialist(initialized, specialist)
    if (
        specialist_element_count(specialist) != SPECIALIST_PARAMETER_COUNT
        or _tree_size(jax, parameters) != CANDIDATE_PARAMETER_COUNT
    ):
        raise ValueError("structured GPP broad preflight candidate parameter count drift")

    references = tuple(runtime["index"].select(spatial_split="train", temporal_split="train"))
    batches = _resolve_schedule_batches(schedule_report, references)
    direct_steps = _make_direct_steps(runtime, head, initialized)
    gradient_records = []
    raw_norms = []
    for update, selections in enumerate(batches, start=1):
        host_batch, preparation = _prepare_batch(runtime, selections)
        transfer_started = time.perf_counter()
        batch = jax.device_put(host_batch)
        jax.block_until_ready(batch)
        transfer_seconds = time.perf_counter() - transfer_started
        statistics, gradient, timing = _direct_statistics_and_gradient(
            jax, jnp, specialist, batch, direct_steps
        )
        loss = float(np.asarray(structured_gpp_direct_loss(statistics)))
        gradient_report = _tree_scalar_report(jax, jnp, gradient)
        if (
            not np.isfinite(loss)
            or loss <= 0.0
            or int(np.asarray(statistics.count)) != 208
            or not gradient_report["all_finite"]
            or gradient_report["element_count"] != SPECIALIST_PARAMETER_COUNT
            or gradient_report["nonzero_leaf_count"] != gradient_report["leaf_count"]
            or gradient_report["l2_norm"] <= 0.0
        ):
            raise RuntimeError(f"structured GPP broad preflight gradient batch {update} failed")
        raw_norms.append(float(gradient_report["l2_norm"]))
        gradient_records.append(
            {
                "schedule_update": update,
                "loss": loss,
                "evaluated_values": int(np.asarray(statistics.count)),
                "gradient": gradient_report,
                "preparation_seconds": preparation,
                "host_to_device_seconds": transfer_seconds,
                "compute_seconds": timing,
            }
        )
        print(
            f"structured_gpp broad_preflight gradient={update}/16 "
            f"loss={loss:.9g} norm={gradient_report['l2_norm']:.9g}",
            flush=True,
        )

    clip_threshold = float(np.quantile(np.asarray(raw_norms), 0.95))
    optimizer_step = _make_optimizer_step(jax, jnp, broad, clip_threshold)
    full_steps = _make_steps(
        runtime,
        head,
        context["template"],
        context["scales"],
        target_contract,
        science_flux_basis=amendment.flux_basis,
    )
    physical_artifact = _load_artifact(args, context, 251)
    proposal_records = []
    retained_reports = []
    for update in PROPOSAL_UPDATES:
        host_batch, _preparation = _prepare_batch(runtime, batches[update - 1])
        batch = jax.device_put(host_batch)
        statistics, gradient, _timing = _direct_statistics_and_gradient(
            jax, jnp, specialist, batch, direct_steps
        )
        before_loss = float(np.asarray(structured_gpp_direct_loss(statistics)))
        optimizer = initialize_target_oriented_adamw(specialist)
        proposal = optimizer_step(
            specialist,
            gradient,
            optimizer,
            jnp.asarray(float(broad.optimizer["learning_rate"]), dtype=jnp.float32),
        )
        jax.block_until_ready(proposal)
        updated, updated_optimizer, raw_norm, clip_scale, nonfinite, step_norm = proposal
        after_statistics = _direct_forward(jax, updated, batch, direct_steps)
        after_loss = float(np.asarray(structured_gpp_direct_loss(after_statistics)))
        proposed_parameters = replace_specialist(initialized, updated)
        hard = jax.device_get(
            _one_day_hard(
                jax,
                jnp,
                proposed_parameters,
                physical_artifact.batch,
                full_steps,
            )
        )
        hard_counts = {name: int(np.asarray(value)) for name, value in hard._asdict().items()}
        retained = _retained_parameter_report(jax, parameters, proposed_parameters)
        record = {
            "schedule_update": update,
            "before_loss": before_loss,
            "after_loss": after_loss,
            "relative_change": after_loss / before_loss - 1.0,
            "optimizer_step": int(np.asarray(updated_optimizer.step)),
            "raw_gradient_norm": float(np.asarray(raw_norm)),
            "clip_scale": float(np.asarray(clip_scale)),
            "nonfinite_count": int(np.asarray(nonfinite)),
            "parameter_step_l2": float(np.asarray(step_norm)),
            "physical_hard_counts": hard_counts,
            "retained_parent": retained,
        }
        if (
            not np.isfinite(after_loss)
            or after_loss >= before_loss
            or record["optimizer_step"] != 1
            or record["nonfinite_count"] != 0
            or not np.isfinite(record["parameter_step_l2"])
            or record["parameter_step_l2"] <= 0.0
            or _hard_failure_count(hard_counts) != 0
            or retained["exactly_unchanged"] is not True
        ):
            raise RuntimeError(f"structured GPP broad preflight proposal {update} failed")
        proposal_records.append(record)
        retained_reports.append(retained)
        print(
            f"structured_gpp broad_preflight proposal={update}/4 "
            f"change={record['relative_change']:.9g}",
            flush=True,
        )

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "identity": {
            "broad_preflight_contract_sha256": broad.sha256,
            "schedule_report_sha256": sha256_file(args.schedule_report),
            "broad_schedule_sha256": schedule_report["identity"]["broad_schedule_sha256"],
            "coverage_report_sha256": COVERAGE_REPORT_SHA256,
            "selected_checkpoint_sha256": screen_identity["selected_checkpoint_sha256"],
        },
        "policy": {
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "monitor_values_read": False,
            "validation_or_sealed_test_values_read": False,
            "chained_training_updates": 0,
            "optimizer_checkpoint_writes": 0,
            "typed_sidecar_payload_streamed": False,
            "seven_or_thirty_day_training_payload_streamed": False,
            "physical_probe_prepared_artifact": 251,
        },
        "gradient_preflight": {
            "batch_count": len(gradient_records),
            "days_per_batch": 208,
            "records": gradient_records,
            "raw_gradient_norm": {
                "minimum": float(np.min(raw_norms)),
                "median": float(np.median(raw_norms)),
                "maximum": float(np.max(raw_norms)),
                "p95_clip_threshold": clip_threshold,
            },
        },
        "independent_proposals": {
            "count": len(proposal_records),
            "learning_rate": float(broad.optimizer["learning_rate"]),
            "fresh_optimizer_for_every_proposal": True,
            "records": proposal_records,
        },
        "memory": _memory_report(runtime["devices"][0]),
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": "freeze_broad_direct_gpp_paid_stage_1_training_contract",
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
    parser.add_argument("--structured-contract", type=Path, required=True)
    parser.add_argument("--joint-training-report", type=Path, required=True)
    parser.add_argument("--selected-checkpoint", type=Path, required=True)
    parser.add_argument("--screen-contract", type=Path, required=True)
    parser.add_argument("--coverage-report", type=Path, required=True)
    parser.add_argument("--broad-preflight-contract", type=Path, required=True)
    parser.add_argument("--schedule-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
