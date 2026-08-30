"""Measure direct-only raw-v6 streaming and structured-GPP compute throughput."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

import numpy as np

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_checkpoint import (
    load_structured_gpp_checkpoint,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_training import (
    CANDIDATE_PARAMETER_COUNT,
    SPECIALIST_PARAMETER_COUNT,
    replace_specialist,
    specialist_element_count,
)
from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    add_error_sufficient_statistics,
)
from scripts.hpc.audit_target_oriented_flux_gradient_variance import _load_context, _tree_size
from scripts.hpc.calibrate_target_oriented_distribution_aware_guard import _replace_adapter
from scripts.hpc.run_strong_baseline_gpu_admission import _memory_report
from scripts.hpc.run_successor_fast_owner_gpu_probe import PROBE_SEED
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json
from scripts.hpc.run_target_oriented_attention_gpu_admission import _head as _parent_head
from scripts.hpc.run_target_oriented_full_step_preflight import _slice
from scripts.hpc.run_target_oriented_structured_gpp_short_horizon_screen import (
    _load_screen_contract,
    _load_training_report,
)
from scripts.hpc.run_target_oriented_structured_gpp_training_preflight import (
    ADAPTER_PARAMETER_COUNT,
    PARENT_PARAMETER_COUNT,
    _add_gradient,
    _make_direct_steps,
    _zero_gradient,
)

SCHEMA_VERSION = "target_oriented_structured_gpp_streaming_throughput_v1"
COVERAGE_REPORT_SHA256 = "a53330a621cd63cd863798caea012a1517ef8552205839cb6f7f93aafff0528b"
GPP_FIELD = "daily.fast_gpp"
SOURCE_POINT_YEARS = 4
DAYS_PER_POINT_YEAR = 52
MAXIMUM_BATCH_SIZE = SOURCE_POINT_YEARS * DAYS_PER_POINT_YEAR
MICROBATCH_SIZE = 4
BATCH_SIZES = (32, 64, 128, MAXIMUM_BATCH_SIZE)
HOT_REPEATS = 2


class DirectGppBatch(NamedTuple):
    inputs: Any
    process_targets: Mapping[str, Any]


def _reference_identity(reference: Any) -> tuple[str, int, str, str]:
    return (
        str(reference.parent.landpoint_id),
        int(reference.parent.year),
        str(reference.parent.sha256),
        str(reference.sha256),
    )


def _round_robin(values: Sequence[Sequence[Any]]) -> tuple[Any, ...]:
    if not values or len({len(group) for group in values}) != 1:
        raise ValueError("structured GPP streaming groups must have one common nonzero length")
    if not values[0]:
        raise ValueError("structured GPP streaming groups cannot be empty")
    return tuple(value for row in range(len(values[0])) for group in values for value in (group[row],))


def _stack(jax: Any, values: Sequence[Any]) -> Any:
    return jax.tree_util.tree_map(lambda *items: np.stack(items), *values)


def _prepare_direct_batch(
    runtime: Mapping[str, Any],
    references: Sequence[Any],
) -> tuple[DirectGppBatch, Mapping[str, Any]]:
    from research.daily_coarse_graining.daily_markov_contract import load_markov_shard
    from research.daily_coarse_graining.typed_sidecar import read_typed_sidecar_shard

    jax = runtime["jax"]
    adapter = runtime["adapter"]
    grouped_examples = []
    records = []
    started = time.perf_counter()
    for reference in references:
        local_started = time.perf_counter()
        parent = load_markov_shard(reference.parent.path, contract=runtime["contract"])
        values, defined, days = read_typed_sidecar_shard(
            reference.path,
            contract=adapter.typed_contract,
            expected_days=reference.day_count,
        )
        if not np.array_equal(days, parent.day_index) or int(reference.day_count) <= DAYS_PER_POINT_YEAR:
            raise ValueError("structured GPP streaming source shard is not eligible")
        examples = tuple(
            adapter.assemble(
                parent.sample(row),
                supplemental_values={name: value[row] for name, value in values.items()},
                supplemental_defined={name: value[row] for name, value in defined.items()},
            )
            for row in range(1, DAYS_PER_POINT_YEAR + 1)
        )
        grouped_examples.append(examples)
        records.append(
            {
                "identity": list(_reference_identity(reference)),
                "rows": [1, DAYS_PER_POINT_YEAR],
                "read_and_assemble_seconds": time.perf_counter() - local_started,
            }
        )
    examples = _round_robin(grouped_examples)
    batch = DirectGppBatch(
        inputs=_stack(jax, [value.inputs for value in examples]),
        process_targets={
            GPP_FIELD: _stack(
                jax,
                [value.targets.supervision[GPP_FIELD] for value in examples],
            )
        },
    )
    if len(examples) != MAXIMUM_BATCH_SIZE:
        raise ValueError("structured GPP streaming batch size drift")
    return batch, {
        "point_years": records,
        "days": len(examples),
        "read_assemble_and_stack_seconds": time.perf_counter() - started,
    }


def _direct_statistics_and_gradient(
    jax: Any,
    jnp: Any,
    specialist: Any,
    batch: DirectGppBatch,
    direct_steps: Mapping[str, Any],
) -> tuple[Any, Any, Mapping[str, float]]:
    batch_size = int(batch.inputs.year.shape[0])
    if batch_size % MICROBATCH_SIZE:
        raise ValueError("structured GPP streaming batch must divide its microbatch size")
    microbatches = tuple(
        _slice(batch, start, start + MICROBATCH_SIZE)
        for start in range(0, batch_size, MICROBATCH_SIZE)
    )
    forward_started = time.perf_counter()
    statistics = [direct_steps["one_forward"](specialist, value) for value in microbatches]
    jax.block_until_ready(statistics)
    total = statistics[0]
    for value in statistics[1:]:
        total = add_error_sufficient_statistics(total, value)
    jax.block_until_ready(total)
    forward_seconds = time.perf_counter() - forward_started

    gradient_started = time.perf_counter()
    gradient = _zero_gradient(jax, jnp, specialist)
    for value in microbatches:
        _loss, current = direct_steps["one_gradient"](specialist, value, total)
        gradient = _add_gradient(jax, jnp, gradient, current)
    jax.block_until_ready(gradient)
    gradient_seconds = time.perf_counter() - gradient_started
    return total, gradient, {
        "forward_seconds": forward_seconds,
        "gradient_seconds": gradient_seconds,
        "total_seconds": forward_seconds + gradient_seconds,
    }


def _median_timings(values: Sequence[Mapping[str, float]]) -> Mapping[str, float]:
    return {
        name: float(np.median([value[name] for value in values]))
        for name in ("forward_seconds", "gradient_seconds", "total_seconds")
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    started = time.perf_counter()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    if sha256_file(args.coverage_report) != COVERAGE_REPORT_SHA256:
        raise ValueError("structured GPP streaming coverage report hash drift")
    coverage = json.loads(args.coverage_report.read_text(encoding="utf-8"))
    if coverage.get("status") != "passed" or coverage.get("next_gate") != "measure_broad_direct_gpp_streaming_batch_throughput":
        raise ValueError("structured GPP streaming coverage gate is not accepted")

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
        raise ValueError("structured GPP streaming parent contract drift")

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
    parent_parameters = _replace_adapter(initialized_parent, jax.device_put(distribution_checkpoint["adapter"]))
    if _tree_size(jax, parent_parameters) != PARENT_PARAMETER_COUNT:
        raise ValueError("structured GPP streaming parent parameter count drift")
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
    if specialist_element_count(specialist) != SPECIALIST_PARAMETER_COUNT or _tree_size(jax, parameters) != CANDIDATE_PARAMETER_COUNT:
        raise ValueError("structured GPP streaming candidate parameter count drift")

    monitor_identities = {
        (
            value["landpoint_id"],
            int(value["year"]),
            value["parent_shard_sha256"],
            value["typed_shard_sha256"],
        )
        for value in coverage["proposed_monitor"]["entries"]
    }
    references = sorted(
        (
            value
            for value in runtime["index"].select(spatial_split="train", temporal_split="train")
            if _reference_identity(value) not in monitor_identities and int(value.day_count) > DAYS_PER_POINT_YEAR
        ),
        key=_reference_identity,
    )
    selected = []
    seen_landpoints = set()
    for reference in references:
        landpoint = str(reference.parent.landpoint_id)
        if landpoint in seen_landpoints:
            continue
        selected.append(reference)
        seen_landpoints.add(landpoint)
        if len(selected) == SOURCE_POINT_YEARS:
            break
    if len(selected) != SOURCE_POINT_YEARS:
        raise ValueError("structured GPP streaming source selection drift")

    host_batch, preparation = _prepare_direct_batch(runtime, selected)
    transfer_started = time.perf_counter()
    device_batch = jax.device_put(host_batch)
    jax.block_until_ready(device_batch)
    transfer_seconds = time.perf_counter() - transfer_started
    direct_steps = _make_direct_steps(runtime, head, initialized)

    warm_batch = _slice(device_batch, 0, MICROBATCH_SIZE)
    warm_started = time.perf_counter()
    warm_total, warm_gradient, _warm_timing = _direct_statistics_and_gradient(
        jax,
        jnp,
        specialist,
        warm_batch,
        direct_steps,
    )
    jax.block_until_ready((warm_total, warm_gradient))
    warm_seconds = time.perf_counter() - warm_started

    benchmarks = {}
    for size in BATCH_SIZES:
        batch = _slice(device_batch, 0, size)
        timings = []
        totals = []
        gradients = []
        for _repeat in range(HOT_REPEATS):
            total, gradient, timing = _direct_statistics_and_gradient(
                jax,
                jnp,
                specialist,
                batch,
                direct_steps,
            )
            totals.append(total)
            gradients.append(gradient)
            timings.append(timing)
        median = _median_timings(timings)
        gradient_report = {
            "element_count": _tree_size(jax, gradients[-1]),
            "all_finite": all(np.all(np.isfinite(np.asarray(value))) for value in jax.tree_util.tree_leaves(jax.device_get(gradients[-1]))),
        }
        if gradient_report != {"element_count": SPECIALIST_PARAMETER_COUNT, "all_finite": True}:
            raise ValueError("structured GPP streaming gradient gate failed")
        loss_statistics_finite = all(
            np.all(np.isfinite(np.asarray(value)))
            for value in jax.tree_util.tree_leaves(jax.device_get(totals[-1]))
        )
        if not loss_statistics_finite:
            raise ValueError("structured GPP streaming loss-statistics gate failed")
        benchmarks[str(size)] = {
            "hot_repeats": HOT_REPEATS,
            "timings": timings,
            "median": median,
            "days_per_second": size / median["total_seconds"],
            "projected_hours_per_million_days": 1_000_000.0 / (size / median["total_seconds"]) / 3600.0,
            "gradient": gradient_report,
            "loss_statistics_finite": loss_statistics_finite,
        }

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "identity": {
            "coverage_report_sha256": COVERAGE_REPORT_SHA256,
            "screen_contract_sha256": screen["sha256"],
            "selected_checkpoint_sha256": screen_identity["selected_checkpoint_sha256"],
            "dataset_manifest_sha256": coverage["identity"]["dataset_manifest_sha256"],
        },
        "policy": {
            "split": "train/train",
            "monitor_point_years_read": False,
            "validation_or_sealed_test_values_read": False,
            "teacher_data_regenerated": False,
            "historical_v5_used": False,
            "optimizer_updates": 0,
            "checkpoint_writes": 0,
            "retained_tail_payload_streamed": False,
            "seven_or_thirty_day_payload_streamed": False,
        },
        "batch": {
            "source_point_years": SOURCE_POINT_YEARS,
            "days_per_point_year": DAYS_PER_POINT_YEAR,
            "maximum_batch_size": MAXIMUM_BATCH_SIZE,
            "microbatch_size": MICROBATCH_SIZE,
            "round_robin_across_point_years": True,
            "preparation": preparation,
            "host_to_device_seconds": transfer_seconds,
        },
        "compute": {
            "warm_compile_seconds": warm_seconds,
            "benchmarks": benchmarks,
            "memory": _memory_report(runtime["devices"][0]),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "next_gate": "freeze_broad_direct_gpp_training_and_monitor_contract",
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
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    report = run(_parser().parse_args())
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
