"""Compare one shard row with the exact outer compiled Teacher boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import jax
import numpy as np

from jax_orchidee.driver import orchestration as teacher
from research.daily_coarse_graining.canonical_teacher_reentry import (
    teacher_reentry_packet,
    teacher_reentry_templates,
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
    _resolve,
    _sha256_array,
)

_REENTRY_ONLY_DAILY_ACCUMULATORS = frozenset(
    {"flood_root_radia", "resp_maint_part", "resp_maint_radia"}
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
    matches = np.flatnonzero(np.asarray(shard.day_index) == args.day_index)
    if matches.size != 1:
        raise ValueError("outer-block probe day is absent or duplicated")
    row = int(matches[0])

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
        name: np.asarray(values[row])
        for name, values in shard.discrete_trajectories.items()
    }
    packet = teacher_reentry_packet(
        np.asarray(shard.state_trajectory[row]),
        discrete,
        contract,
        tstep=(args.day_index - 1) * templates.steps_per_day - 1,
        templates=templates,
    )

    steps_per_day = templates.steps_per_day
    forcing = teacher._paper_compiled_forcing_day(
        context,
        year=args.year,
        start_tstep=(args.day_index - 1) * steps_per_day,
        steps_per_stomate=steps_per_day,
    )
    block_forcing = jax.tree_util.tree_map(
        lambda value: np.expand_dims(np.asarray(value), axis=0), forcing
    )
    block_day_numbers = np.asarray([args.day_index], dtype=np.int32)
    schema_record = capture_pre_daily_stomate_record(
        config_path,
        previous_state=packet,
        year=args.year,
        day_index=args.day_index,
        start_tstep=(args.day_index - 1) * steps_per_day,
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
        start=(args.day_index - 1) * steps_per_day,
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
            f"ok_leak_outer_block_probe_v1:{contract.sha256}"
        ),
    )
    season_values = {
        name: value
        for name, value in teacher.read_stomate_restart_season_state(
            context.first_step_restart_state.stomate_input
        )._asdict().items()
        if name != "provenance"
    }
    final_values, stacked_targets = executable(
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
    actual_target = np.asarray(jax.device_get(stacked_targets))[0]
    target_comparison = _array_comparison(
        actual_target, np.asarray(shard.fast_day_target[row])
    )
    final_packet = teacher.previous_packet_from_fast_state(
        teacher.DriverFastStateBundle(
            tstep=args.day_index * steps_per_day - 1,
            values_by_component=jax.tree_util.tree_map(
                lambda value: value, final_values
            ),
            spec=state_spec,
        )
    )
    actual_state, actual_discrete = extract_state(final_packet, contract)
    state_comparison = _array_comparison(
        actual_state,
        np.asarray(shard.state_trajectory[row + 1]),
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    discrete_comparisons = {
        name: _array_comparison(actual_discrete[name], values[row + 1])
        for name, values in shard.discrete_trajectories.items()
    }
    passed = bool(
        target_comparison["exact"]
        and state_comparison["within_tolerance"]
        and all(item["exact"] for item in discrete_comparisons.values())
    )
    report = {
        "schema_version": "ok_leak_outer_block_reentry_probe_v1",
        "passed": passed,
        "dataset_id": manifest["dataset_id"],
        "landpoint_id": args.landpoint_id,
        "year": args.year,
        "day_index": args.day_index,
        "source_shard": str(shard_path),
        "day_start_state_sha256": _sha256_array(shard.state_trajectory[row]),
        "reentry_schema_adjustments": schema_adjustments,
        "fast_day_target": target_comparison,
        "next_continuous_state": state_comparison,
        "next_discrete_state": discrete_comparisons,
        "decision": (
            "outer_compiled_boundary_reproduces_shard"
            if passed
            else "mismatch_survives_outer_compiled_boundary"
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
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    report = run_probe(build_parser().parse_args(argv))
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
