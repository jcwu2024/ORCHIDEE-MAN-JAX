"""Compare one shard row with the exact outer compiled Teacher boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import jax
import numpy as np

from jax_orchidee.driver import orchestration as teacher
from jax_orchidee.stomate import soilcarbon_kernels
from research.daily_coarse_graining.canonical_teacher_reentry import (
    teacher_reentry_packet,
    teacher_reentry_templates,
)
from research.daily_coarse_graining.carbon_budget_ownership import (
    extract_compiled_ok_leak_driver_steps,
    ok_leak_driver_capture_metadata,
)
from research.daily_coarse_graining.daily_markov_contract import (
    daily_markov_contract_from_metadata,
    extract_state,
    load_markov_shard,
    make_compiled_training_output_projector,
)
from research.daily_coarse_graining.replay_ceiling import (
    capture_pre_daily_stomate_record,
)
from scripts.dev.probe_ok_leak_driver_capture import (
    _array_comparison,
    _atomic_write_json,
    _ok_leak_endpoint_comparisons,
    _resolve,
    _sha256_array,
    _sha256_file,
    _state_leaf_comparisons,
    _target_leaf_comparisons,
    _within_tolerance,
)

_REENTRY_ONLY_DAILY_ACCUMULATORS = frozenset(
    {"flood_root_radia", "resp_maint_part", "resp_maint_radia"}
)


def _configure_doc_sqrt_mode(mode: str) -> None:
    """Select a diagnostic-only DOC sqrt graph before the first JAX trace."""

    if mode == "production":
        return
    if mode == "raw_sqrt":
        soilcarbon_kernels._sqrt_with_finite_zero_tangent = jax.numpy.sqrt
        return
    raise ValueError(f"unsupported DOC sqrt mode: {mode}")


def teacher_block_for_day(
    day_index: int, *, block_size: int = 7, days_in_year: int = 365
) -> tuple[int, int, int]:
    """Return the original Teacher block start, size, and target offset."""

    if day_index < 2 or day_index > days_in_year:
        raise ValueError("later-day Teacher block capture requires Day 2..year end")
    if block_size < 2:
        raise ValueError("Teacher block size must be at least two")
    block_start = 2 + ((day_index - 2) // block_size) * block_size
    observed_size = min(block_size, days_in_year - block_start + 1)
    return block_start, observed_size, day_index - block_start


def _ok_leak_endpoint_passed(
    comparisons: dict[str, dict[str, object]],
    *,
    tolerance: float = 1.0e-12,
) -> bool:
    return bool(
        comparisons
        and all(
            _within_tolerance(item, tolerance)
            for item in comparisons.values()
        )
    )


def _normalize_daily_accumulator_schema(packet, produced_packet):
    component = "slowproc_stomate_previous_step_state"
    field = "daily_accumulators"
    current = dict(packet.fields_by_component[component][field])
    produced = dict(produced_packet.fields_by_component[component][field])
    missing = tuple(sorted(set(produced) - set(current)))
    dropped = tuple(sorted(set(current) - set(produced)))
    reset_fields = frozenset(teacher.PAPER_DAY_ZERO_DAILY_RESET_FIELDS)
    if not set(missing) <= _REENTRY_ONLY_DAILY_ACCUMULATORS:
        raise ValueError(
            f"outer-block reentry has unexplained missing accumulators: {missing}"
        )
    if not set(dropped) <= reset_fields:
        raise ValueError(
            f"outer-block reentry has non-reset dropped accumulators: {dropped}"
        )
    normalized = {
        name: (
            current[name]
            if name in current
            else np.zeros_like(np.asarray(produced_value))
        )
        for name, produced_value in produced.items()
    }
    if any(np.any(np.asarray(value)) for value in normalized.values()):
        raise ValueError("outer-block reentry accumulators must remain zero")
    fields = {
        name: dict(values) for name, values in packet.fields_by_component.items()
    }
    fields[component][field] = normalized
    normalized_packet = teacher.DriverPreviousStepStatePacket(
        tstep=packet.tstep,
        fields_by_component=fields,
        provenance_by_component=dict(packet.provenance_by_component),
    )
    return normalized_packet, {"zero_filled": missing, "dropped": dropped}


def run_probe(args: argparse.Namespace):
    doc_sqrt_mode = getattr(args, "doc_sqrt_mode", "production")
    capture_npz = getattr(args, "capture_npz", None)
    _configure_doc_sqrt_mode(doc_sqrt_mode)
    manifest_path = args.dataset_manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = daily_markov_contract_from_metadata(manifest["markov_contract"])
    references = tuple(
        item
        for item in manifest["shards"]
        if item["landpoint_id"] == args.landpoint_id
        and int(item["year"]) == args.year
    )
    if len(references) != 1:
        raise ValueError("outer-block probe requires exactly one matching shard")
    reference = references[0]
    shard_path = (manifest_path.parent / reference["shard"]).resolve()
    shard = load_markov_shard(shard_path)
    target_matches = np.flatnonzero(
        np.asarray(shard.day_index) == args.day_index
    )
    if target_matches.size != 1:
        raise ValueError("outer-block probe day is absent or duplicated")
    target_row = int(target_matches[0])
    block_start_day = (
        args.day_index
        if getattr(args, "block_start_day", None) is None
        else int(args.block_start_day)
    )
    block_matches = np.flatnonzero(
        np.asarray(shard.day_index) == block_start_day
    )
    if block_matches.size != 1:
        raise ValueError("outer-block start day is absent or duplicated")
    block_row = int(block_matches[0])
    if args.block_days < 1:
        raise ValueError("block_days must be positive")
    if block_row + args.block_days > shard.fast_day_target.shape[0]:
        raise ValueError("requested block extends beyond the source shard")
    target_offset = args.day_index - block_start_day
    if target_offset < 0 or target_offset >= args.block_days:
        raise ValueError("target day must fall inside the replay block")
    day_start_state_sha256 = _sha256_array(shard.state_trajectory[target_row])
    fast_day_target_sha256 = _sha256_array(shard.fast_day_target[target_row])
    expected_state_hash = getattr(args, "expected_day_start_state_sha256", None)
    expected_target_hash = getattr(args, "expected_fast_day_target_sha256", None)
    if expected_state_hash is not None and day_start_state_sha256 != expected_state_hash:
        raise ValueError("capture plan day-start state hash mismatch")
    if expected_target_hash is not None and fast_day_target_sha256 != expected_target_hash:
        raise ValueError("capture plan fast-day target hash mismatch")

    plan_path = args.plan.resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    entries = tuple(
        item
        for item in plan["entries"]
        if item["landpoint_id"] == args.landpoint_id
        and int(item["year"]) == args.year
    )
    if len(entries) != 1:
        raise ValueError("outer-block probe requires exactly one matching plan entry")
    entry = entries[0]
    root = Path(__file__).resolve().parents[2]
    config_path = _resolve(root, plan["teacher_config"])
    context = teacher.prepare_paper_1961_driver_context(
        config_path,
        used_run_def_path=_resolve(root, entry["run_def"]),
        reference_run_dir=_resolve(root, entry["reference_run_dir"]),
    )
    templates = teacher_reentry_templates(context)
    discrete = {
        name: np.asarray(values[block_row])
        for name, values in shard.discrete_trajectories.items()
    }
    packet = teacher_reentry_packet(
        np.asarray(shard.state_trajectory[block_row]),
        discrete,
        contract,
        tstep=(block_start_day - 1) * templates.steps_per_day - 1,
        templates=templates,
    )

    steps_per_day = templates.steps_per_day
    day_numbers = tuple(range(block_start_day, block_start_day + args.block_days))
    forcing_days = tuple(
        teacher._paper_compiled_forcing_day(
            context,
            year=args.year,
            start_tstep=(day_index - 1) * steps_per_day,
            steps_per_stomate=steps_per_day,
        )
        for day_index in day_numbers
    )
    block_forcing = jax.tree_util.tree_map(
        lambda *values: np.stack(tuple(np.asarray(value) for value in values)),
        *forcing_days,
    )
    block_day_numbers = np.asarray(day_numbers, dtype=np.int32)
    schema_record = capture_pre_daily_stomate_record(
        config_path,
        previous_state=packet,
        year=args.year,
        day_index=block_start_day,
        start_tstep=(block_start_day - 1) * steps_per_day,
        used_run_def_path=context.run_def_path,
        prepared_context=context,
        module_jit=True,
        diffuco_local_jit=True,
        use_static_jit_daily_carbon=True,
        use_compiled_sechiba_day=True,
        retain_stomate_step_results=False,
        prebuild_day_payloads=True,
    )
    packet, schema_adjustments = _normalize_daily_accumulator_schema(
        packet, schema_record.expected_result.day_end_state
    )
    transition_inputs = teacher._paper_1961_later_day_transition_inputs(
        context=context,
        current_state=packet,
        year=args.year,
        start=(block_start_day - 1) * steps_per_day,
        steps_per_stomate=steps_per_day,
        fixed_format_trace_dir=None,
        static_trace_fields=None,
        prebuild_day_payloads=True,
    )
    initial = teacher.fast_state_from_previous_packet(packet)
    boundary_state_spec = teacher.fast_state_from_previous_packet(
        schema_record.half_hour_transition.current_state
    ).spec
    projector = make_compiled_training_output_projector(
        contract,
        day_start_state_spec=initial.spec,
        boundary_state_spec=boundary_state_spec,
    )

    def target_projector(current_values, boundary):
        _state, _discrete, target, _diagnostics = projector(
            current_values, boundary
        )
        if capture_npz is not None:
            return target, boundary.ok_leak_driver_steps
        return target

    executable, state_spec = teacher._paper_compiled_later_day_block_executable(
        config_path,
        context=context,
        initial_state=packet,
        year=args.year,
        block_forcing=block_forcing,
        block_day_numbers=block_day_numbers,
        prebound_hydrol_runtime_static_tables=(
            transition_inputs.hydrol_runtime_static_tables
        ),
        daily_carbon_dispatch=teacher._paper_daily_carbon_static_dispatch(
            context, packet
        ),
        stomate_parameter_values=teacher._compiled_stomate_parameter_values(
            context
        ),
        capture_pre_daily_training_boundaries=True,
        training_output_projector=target_projector,
        training_output_projector_key=(
            "ok_leak_outer_block_probe_v2:"
            f"{'drivers' if capture_npz is not None else 'target'}:"
            f"{contract.sha256}"
        ),
    )
    season_values = {
        name: value
        for name, value in teacher.read_stomate_restart_season_state(
            context.first_step_restart_state.stomate_input
        )._asdict().items()
        if name != "provenance"
    }
    final_values, stacked_outputs = executable(
        initial.values_by_component,
        block_forcing,
        block_day_numbers,
        teacher._compiled_stomate_parameter_values(context),
        teacher._compiled_hydrol_table_arrays(
            transition_inputs.hydrol_runtime_static_tables
        ),
        teacher._compiled_landpoint_payload(
            transition_inputs.compiled_base_payload_template
        ),
        context.first_step_restart_state.stomate,
        season_values,
        teacher._compiled_diffuco_parameter_values(context),
    )
    capture_arrays = None
    if capture_npz is None:
        stacked_targets = stacked_outputs
    else:
        stacked_targets, stacked_driver_steps = stacked_outputs
        selected_steps = jax.tree_util.tree_map(
            lambda value: np.asarray(jax.device_get(value))[target_offset],
            stacked_driver_steps,
        )
        capture_arrays = extract_compiled_ok_leak_driver_steps(selected_steps)
    actual_targets = np.asarray(jax.device_get(stacked_targets))
    expected_targets = np.asarray(
        shard.fast_day_target[block_row : block_row + args.block_days]
    )
    actual_target = actual_targets[target_offset]
    expected_target = expected_targets[target_offset]
    target_comparison = _array_comparison(
        actual_targets, expected_targets
    )
    target_leaf_comparisons = _target_leaf_comparisons(
        actual_target,
        expected_target,
        contract.fast_day_target_leaves,
    )
    ok_leak_comparisons_by_day = {}
    for offset, day_index in enumerate(day_numbers):
        comparisons = _target_leaf_comparisons(
            actual_targets[offset],
            expected_targets[offset],
            contract.fast_day_target_leaves,
        )
        ok_leak_comparisons_by_day[str(day_index)] = (
            _ok_leak_endpoint_comparisons(comparisons)
        )
    target_ok_leak_passed = _ok_leak_endpoint_passed(
        ok_leak_comparisons_by_day[str(args.day_index)]
    )
    all_days_ok_leak_passed = all(
        _ok_leak_endpoint_passed(comparisons)
        for comparisons in ok_leak_comparisons_by_day.values()
    )
    final_packet = teacher.previous_packet_from_fast_state(
        teacher.DriverFastStateBundle(
            tstep=(block_start_day + args.block_days - 1) * steps_per_day - 1,
            values_by_component=jax.tree_util.tree_map(
                lambda value: value, final_values
            ),
            spec=state_spec,
        )
    )
    actual_state, actual_discrete = extract_state(final_packet, contract)
    state_comparison = _array_comparison(
        actual_state,
        np.asarray(shard.state_trajectory[block_row + args.block_days]),
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    state_leaf_comparisons = _state_leaf_comparisons(
        actual_state,
        np.asarray(shard.state_trajectory[block_row + args.block_days]),
        contract.state_leaves,
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    discrete_comparisons = {
        name: _array_comparison(
            actual_discrete[name], values[block_row + args.block_days]
        )
        for name, values in shard.discrete_trajectories.items()
    }
    passed = bool(
        all_days_ok_leak_passed
        and state_comparison["within_tolerance"]
        and all(item["exact"] for item in discrete_comparisons.values())
    )
    capture = None
    if capture_npz is not None:
        assert capture_arrays is not None
        capture_npz.parent.mkdir(parents=True, exist_ok=True)
        temporary = capture_npz.with_suffix(capture_npz.suffix + ".tmp")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **capture_arrays)
        temporary.replace(capture_npz)
        capture = {
            **ok_leak_driver_capture_metadata(capture_arrays),
            "path": str(capture_npz),
            "sha256": _sha256_file(capture_npz),
        }
    capture_interface_passed = (
        None
        if capture_npz is None
        else bool(
            capture is not None
            and target_ok_leak_passed
            and all(item["exact"] for item in discrete_comparisons.values())
        )
    )
    report = {
        "schema_version": "ok_leak_outer_block_reentry_probe_v2",
        "passed": passed,
        "capture_interface_passed": capture_interface_passed,
        "dataset_manifest": str(manifest_path),
        "dataset_id": manifest["dataset_id"],
        "teacher_git_head": manifest.get("teacher_git_head"),
        "markov_contract_sha256": manifest.get("markov_contract_sha256"),
        "landpoint_id": args.landpoint_id,
        "year": args.year,
        "day_index": args.day_index,
        "target_offset": target_offset,
        "block_start_day": block_start_day,
        "block_days": args.block_days,
        "block_day_numbers": day_numbers,
        "doc_sqrt_mode": doc_sqrt_mode,
        "source_shard": str(shard_path),
        "source_shard_sha256": reference.get("shard_sha256"),
        "day_start_state_sha256": day_start_state_sha256,
        "fast_day_target_sha256": fast_day_target_sha256,
        "capture_plan_sha256": getattr(args, "capture_plan_sha256", None),
        "block_start_state_sha256": _sha256_array(
            shard.state_trajectory[block_row]
        ),
        "reentry_schema_adjustments": schema_adjustments,
        "fast_day_target": target_comparison,
        "fast_day_target_leaves": target_leaf_comparisons,
        "ok_leak_endpoint_comparisons_by_day": ok_leak_comparisons_by_day,
        "target_ok_leak_endpoint_within_1e-12": target_ok_leak_passed,
        "ok_leak_endpoint_within_1e-12": all_days_ok_leak_passed,
        "next_continuous_state": state_comparison,
        "next_state_diagnostic_passed": state_comparison["within_tolerance"],
        "next_continuous_state_leaves": state_leaf_comparisons,
        "next_discrete_state": discrete_comparisons,
        "capture": capture,
        "capture_npz": None if capture_npz is None else capture_npz.name,
        "capture_npz_sha256": None if capture is None else capture["sha256"],
        "sealed_test_used": False,
        "decision": (
            "outer_compiled_ok_leak_boundary_reproduces_shard"
            if passed
            else "mismatch_survives_outer_compiled_ok_leak_boundary"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(args.output, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--landpoint-id", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--day-index", type=int, required=True)
    parser.add_argument("--block-start-day", type=int)
    parser.add_argument("--block-days", type=int, default=1)
    parser.add_argument(
        "--doc-sqrt-mode",
        choices=("production", "raw_sqrt"),
        default="production",
        help="Diagnostic DOC sqrt graph; raw_sqrt reproduces the Teacher formula graph.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capture-npz", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    report = run_probe(build_parser().parse_args(argv))
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
