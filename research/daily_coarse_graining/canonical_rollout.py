"""Validation-only free rollout for the canonical daily neural operator."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import time
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence
from unittest.mock import patch

import jax
import numpy as np

from jax_orchidee.driver import orchestration as teacher
from jax_orchidee.sechiba.restart_io import (
    SECHIBA_RESTART_COMPONENT_FIELDS,
    SECHIBA_RESTART_TO_SOURCE_NAMES,
)
from jax_orchidee.sechiba.restart_lifecycle import SECHIBA_FINALIZE_SOURCE_FIELDS
from research.daily_coarse_graining.canonical_daily_model import (
    CanonicalModelConfig,
    canonical_model_apply,
)
from research.daily_coarse_graining.canonical_training import (
    fast_day_target_representation_from_contract,
    prepare_canonical_inference_batch,
    restore_fast_day_inference_prediction,
)
from research.daily_coarse_graining.canonical_training_run import (
    CHECKPOINT_SCHEMA_VERSION,
    load_contract_metadata,
    verify_training_acceptance,
)
from research.daily_coarse_graining.daily_markov_contract import (
    DailyMarkovContract,
    ReconstructedFastDayTarget,
    daily_markov_contract_from_metadata,
    extract_state,
    load_markov_shard,
    reconstruct_compiled_forcing_day,
    reconstruct_fast_day_target,
    reconstruct_state_fields,
)
from research.daily_coarse_graining.markov_dataset import (
    MarkovShardRef,
    TrainingStatistics,
    defined_numeric_mask,
    load_dataset_index,
    load_training_statistics,
)
from research.daily_coarse_graining.replay_ceiling import (
    _minimal_daily_fold,
    _ok_leak_with_restart_shape,
)
from research.daily_coarse_graining.synthetic_operator_cost import (
    _tail_static_inputs,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _packet_from_canonical_state(
    continuous: np.ndarray,
    discrete: Mapping[str, np.ndarray],
    contract: DailyMarkovContract,
    *,
    tstep: int,
    require_complete_finalize: bool = False,
):
    fields = reconstruct_state_fields(
        continuous,
        discrete,
        contract,
        require_complete_finalize=require_complete_finalize,
    )
    provenance = {
        component: ("canonical daily Markov contract reconstruction",)
        for component in fields
    }
    return teacher.DriverPreviousStepStatePacket(
        tstep=int(tstep),
        fields_by_component=fields,
        provenance_by_component=provenance,
    )


_SLOWPROC_FINALIZE_FIELDS = frozenset(
    SECHIBA_RESTART_TO_SOURCE_NAMES.get(name, name)
    for name in SECHIBA_RESTART_COMPONENT_FIELDS["slowproc"]
)


def _contract_finalize_fields(contract: DailyMarkovContract) -> frozenset[str]:
    fields = frozenset(
        leaf.path[0]
        for leaf in contract.state_leaves
        if leaf.component == "sechiba_finalize_state"
    )
    if not fields:
        raise ValueError("canonical contract has no cross-day SECHIBA finalize fields")
    unknown = fields - SECHIBA_FINALIZE_SOURCE_FIELDS
    if unknown:
        raise ValueError(
            "canonical contract contains unknown SECHIBA finalize fields: "
            f"{sorted(unknown)}"
        )
    return fields


def _projected_finalize_after_slowproc(
    previous: Mapping[str, Any],
    slowproc: Mapping[str, Any],
    *,
    contract_fields: frozenset[str],
) -> dict[str, Any]:
    """Update the contract's cross-day finalize projection without inventing mirrors."""

    actual = frozenset(previous)
    if actual != contract_fields:
        raise ValueError(
            "projected SECHIBA finalize state does not match the canonical contract: "
            f"missing={sorted(contract_fields - actual)}, "
            f"extra={sorted(actual - contract_fields)}"
        )
    return {
        name: slowproc[name]
        if name in _SLOWPROC_FINALIZE_FIELDS and name in slowproc
        else value
        for name, value in previous.items()
    }


def _ok_leak_result(
    boundary: ReconstructedFastDayTarget,
    end_state,
):
    values = boundary.ok_leak_updates
    required = {
        "litter_above",
        "litter_below",
        "lignin_struc_above",
        "lignin_struc_below",
        "litterpart",
        "dead_leaves",
        "fuel_1hr",
        "fuel_10hr",
        "fuel_100hr",
        "fuel_1000hr",
        "carbon_32l",
        "DOC",
        "interception_storage",
        "deepC_peat",
    }
    missing = tuple(sorted(required - values.keys()))
    if missing:
        raise ValueError(f"fast-day boundary is missing OK_LEAK fields {missing}")
    minimal = SimpleNamespace(
        littercalc=SimpleNamespace(
            litter_above=values["litter_above"],
            litter_below=values["litter_below"],
            lignin_struc_above=values["lignin_struc_above"],
            lignin_struc_below=values["lignin_struc_below"],
            litterpart=values["litterpart"],
            dead_leaves=values["dead_leaves"],
            fuel=SimpleNamespace(
                fuel_1hr=values["fuel_1hr"],
                fuel_10hr=values["fuel_10hr"],
                fuel_100hr=values["fuel_100hr"],
                fuel_1000hr=values["fuel_1000hr"],
            ),
        ),
        soilcarbon=SimpleNamespace(
            carbon_32l=values["carbon_32l"],
            doc=values["DOC"],
            perma_peat=SimpleNamespace(deepc_peat=values["deepC_peat"]),
        ),
        interception_storage=values["interception_storage"],
    )
    return _ok_leak_with_restart_shape(minimal, end_state)


def _runtime_boundary(
    boundary: ReconstructedFastDayTarget,
    *,
    metadata,
    end_tstep: int,
):
    end_state = teacher.DriverPreviousStepStatePacket(
        tstep=int(end_tstep),
        fields_by_component={
            name: dict(values)
            for name, values in boundary.fields_by_component.items()
        },
        provenance_by_component={
            name: ("canonical neural fast-day boundary",)
            for name in boundary.fields_by_component
        },
    )
    final = boundary.final_diagnostics
    thermosoil = end_state.fields_by_component[
        "thermosoil_previous_step_state"
    ]
    transition = teacher.DriverRuntimeDayStepTransition(
        completed_entry_payloads=(
            {
                "t2m": final["t2mdiag"],
                "stempdiag": thermosoil["stempdiag"],
                "t2mdiag": final["t2mdiag"],
                "temp_sol": final["temp_sol"],
            },
        ),
        current_state=end_state,
        first_step_metadata=metadata,
        stopped_at_tstep=None,
        state_gaps=(),
        compiled_entry_stacks={
            "t2m": final["t2mdiag"],
            "stempdiag": thermosoil["stempdiag"],
        },
    )
    return transition, _minimal_daily_fold(boundary.daily_fields), _ok_leak_result(
        boundary, end_state
    )


def _run_retained_tail_day(
    *,
    config_path: Path,
    context,
    previous_state,
    boundary: ReconstructedFastDayTarget,
    compiled_forcing,
    static: Mapping[str, Any],
    contract_finalize_fields: frozenset[str],
    year: int,
    day_index: int,
):
    steps_per_day = int(round(context.runtime.dt_stomate / context.runtime.dt_sechiba))
    start_tstep = (int(day_index) - 1) * steps_per_day
    transition, daily_fold, ok_leak = _runtime_boundary(
        boundary,
        metadata=static["metadata"],
        end_tstep=start_tstep + steps_per_day - 1,
    )
    ok_leak_updates = teacher._paper_half_hour_ok_leak_state_updates(ok_leak)
    replay_values = {
        "_paper_1961_later_day_half_hour_transition": transition,
        "_paper_later_day_daily_process_from_completed_entries": daily_fold,
        "_paper_later_day_daily_fold_prerequisite_gaps": (),
        "stomate_maintenance_respiration_parts_from_stacks": np.asarray(0.0),
        "_paper_half_hour_ok_leak_fold_from_entries": (
            ok_leak,
            ok_leak_updates,
        ),
    }
    with ExitStack() as stack:
        for name, value in replay_values.items():
            stack.enter_context(
                patch.object(
                    teacher,
                    name,
                    lambda *args, _value=value, **kwargs: _value,
                )
            )
        stack.enter_context(
            patch.object(
                teacher,
                "_sechiba_finalize_state_after_slowproc",
                lambda previous, slowproc: _projected_finalize_after_slowproc(
                    previous,
                    slowproc,
                    contract_fields=contract_finalize_fields,
                ),
            )
        )
        result = teacher.paper_1961_driver_later_day_runtime_result(
            config_path,
            previous_state=previous_state,
            day_index=day_index,
            year=year,
            start_tstep=start_tstep,
            used_run_def_path=context.run_def_path,
            prepared_context=context,
            module_jit=True,
            diffuco_local_jit=True,
            use_static_jit_daily_carbon=False,
            prebuild_day_payloads=True,
            use_compiled_sechiba_day=True,
            prebound_hydrol_runtime_static_tables=teacher.HydrolRuntimeStaticTables(
                mineral=teacher.MineralCWRRTables(
                    **static["hydrol_table_arrays"]._asdict(),
                    imin=static["mineral_imin"],
                    imax=static["mineral_imax"],
                ),
                peat=None,
            ),
            materialize_compiled_entries=False,
            outer_compiled_daily_carbon_dispatch=static["daily_carbon_dispatch"],
            compiled_forcing_series=compiled_forcing,
            model_day_number=day_index,
            compiled_stomate_parameter_values=static["stomate_parameter_values"],
            compiled_landpoint_payload=static["landpoint_payload"],
            compiled_stomate_restart_template=static["stomate_restart_template"],
            compiled_stomate_season_template=teacher.StomateRestartSeasonState(
                **static["stomate_season_values"],
                provenance=static["season"].provenance,
            ),
            compiled_diffuco_parameter_values=static["diffuco_parameter_values"],
        )
    if not bool(getattr(result, "ready_for_day_end_state", True)):
        raise RuntimeError(
            f"retained daily tail failed: {getattr(result, 'missing_components', ())}"
        )
    return result


def _state_metrics(
    actual: np.ndarray,
    expected: np.ndarray,
    statistics: TrainingStatistics,
    contract: DailyMarkovContract,
) -> dict[str, Any]:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    actual_defined = defined_numeric_mask(actual)
    expected_defined = defined_numeric_mask(expected)
    common = actual_defined & expected_defined
    scale = np.asarray(statistics.arrays["state"].scale)
    normalized_error = np.where(common, (actual - expected) / scale, 0.0)
    result = {}
    for component in dict.fromkeys(leaf.component for leaf in contract.state_leaves):
        selected = np.zeros(actual.shape[-1], dtype=bool)
        for leaf in contract.state_leaves:
            if leaf.component == component and not leaf.discrete:
                selected[leaf.start : leaf.stop] = True
        mask = common & selected
        count = int(np.count_nonzero(mask))
        result[component] = {
            "normalized_rmse": (
                float(np.sqrt(np.sum(normalized_error[mask] ** 2) / count))
                if count
                else None
            ),
            "evaluated_values": count,
        }
    leaves = []
    for leaf in contract.state_leaves:
        if leaf.discrete:
            continue
        selected = slice(leaf.start, leaf.stop)
        leaf_common = common[selected]
        count = int(np.count_nonzero(leaf_common))
        leaf_actual = actual[selected]
        leaf_expected = expected[selected]
        leaf_error = normalized_error[selected]
        leaves.append(
            {
                "key": leaf.key,
                "normalized_rmse": (
                    float(
                        np.sqrt(
                            np.sum(leaf_error[leaf_common] ** 2) / count
                        )
                    )
                    if count
                    else None
                ),
                "max_absolute_error": (
                    float(
                        np.max(
                            np.abs(
                                leaf_actual[leaf_common]
                                - leaf_expected[leaf_common]
                            )
                        )
                    )
                    if count
                    else None
                ),
                "defined_status_mismatches": int(
                    np.count_nonzero(
                        actual_defined[selected] != expected_defined[selected]
                    )
                ),
                "evaluated_values": count,
            }
        )
    largest_leaves = sorted(
        leaves,
        key=lambda item: (
            -1.0
            if item["normalized_rmse"] is None
            else item["normalized_rmse"]
        ),
        reverse=True,
    )[:12]
    count = int(np.count_nonzero(common))
    return {
        "normalized_rmse": (
            float(np.sqrt(np.sum(normalized_error[common] ** 2) / count))
            if count
            else None
        ),
        "max_absolute_error": (
            float(np.max(np.abs(actual[common] - expected[common])))
            if count
            else None
        ),
        "defined_status_mismatches": int(
            np.count_nonzero(actual_defined != expected_defined)
        ),
        "evaluated_values": count,
        "components": result,
        "largest_leaves": largest_leaves,
    }


def _select_reference(
    dataset_path: Path,
    *,
    landpoint_id: str,
    year: int,
) -> MarkovShardRef:
    index = load_dataset_index(dataset_path, verify_hashes=True)
    matches = tuple(
        item
        for item in index.shards
        if item.landpoint_id == landpoint_id and item.year == year
    )
    if len(matches) != 1:
        raise ValueError(
            f"expected one dataset shard for {landpoint_id}:{year}, found {len(matches)}"
        )
    reference = matches[0]
    if "test" in {reference.spatial_split, reference.temporal_split}:
        raise ValueError("canonical rollout cannot evaluate a sealed test split")
    if "validation" not in {reference.spatial_split, reference.temporal_split}:
        raise ValueError("canonical rollout requires a validation split")
    return reference


def _plan_entry(plan_path: Path, *, landpoint_id: str, year: int):
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    matches = [
        item
        for item in plan["entries"]
        if str(item["landpoint_id"]) == landpoint_id and int(item["year"]) == year
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one generation-plan entry for {landpoint_id}:{year}"
        )
    return plan, matches[0]


def _load_neural_checkpoint(
    checkpoint_path: Path,
    *,
    dataset_path: Path,
    statistics_path: Path,
    acceptance_path: Path,
):
    with checkpoint_path.open("rb") as handle:
        checkpoint = pickle.load(handle)
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported canonical neural checkpoint schema")
    index = load_dataset_index(dataset_path)
    identity = checkpoint["identity"]
    expected = {
        "dataset_id": index.dataset_id,
        "teacher_git_head": index.teacher_git_head,
        "markov_contract_sha256": index.contract_sha256,
        "statistics_sha256": _sha256_file(statistics_path),
        "acceptance_sha256": _sha256_file(acceptance_path),
    }
    for name, value in expected.items():
        if identity.get(name) != value:
            raise ValueError(f"canonical checkpoint identity mismatch for {name}")
    return checkpoint, CanonicalModelConfig(**identity["model_config"])


def run_rollout(args) -> dict[str, Any]:
    dataset_path = args.dataset.resolve()
    statistics_path = args.statistics.resolve()
    acceptance_path = args.acceptance.resolve()
    verify_training_acceptance(
        acceptance_path,
        dataset_path,
        statistics_path,
    )
    reference = _select_reference(
        dataset_path,
        landpoint_id=args.landpoint_id,
        year=args.year,
    )
    shard = load_markov_shard(reference.path)
    if args.start_day < 1 or args.days < 1:
        raise ValueError("rollout start day and length must be positive")
    stop_day = args.start_day + args.days - 1
    if stop_day > shard.days:
        raise ValueError("rollout window extends beyond the selected shard")

    metadata = load_contract_metadata(dataset_path)
    contract = daily_markov_contract_from_metadata(metadata)
    representation = fast_day_target_representation_from_contract(metadata)
    contract_finalize_fields = _contract_finalize_fields(contract)
    statistics = load_training_statistics(statistics_path)
    plan, entry = _plan_entry(
        args.plan.resolve(),
        landpoint_id=args.landpoint_id,
        year=args.year,
    )
    config_path = Path(plan["teacher_config"])
    context = teacher.prepare_paper_1961_driver_context(
        config_path,
        used_run_def_path=entry["run_def"],
        reference_run_dir=entry["reference_run_dir"],
    )

    first_index = args.start_day - 1
    current_state = np.asarray(shard.state_trajectory[first_index]).copy()
    current_discrete = {
        name: np.asarray(values[first_index]).copy()
        for name, values in shard.discrete_trajectories.items()
    }
    steps_per_day = int(round(context.runtime.dt_stomate / context.runtime.dt_sechiba))
    current_packet = _packet_from_canonical_state(
        current_state,
        current_discrete,
        contract,
        tstep=(args.start_day - 1) * steps_per_day - 1,
    )
    first_forcing = reconstruct_compiled_forcing_day(
        shard.forcing_native[first_index],
        contract.native_forcing,
        context,
        year=args.year,
        day_index=args.start_day,
    )
    forcing_batch = jax.tree_util.tree_map(
        lambda value: np.expand_dims(np.asarray(value), axis=0),
        first_forcing,
    )
    static = _tail_static_inputs(
        context,
        current_packet,
        forcing_batch,
        first_day=False,
    )

    model = None
    model_config = None
    checkpoint_identity = None
    if args.mode == "neural":
        checkpoint, model_config = _load_neural_checkpoint(
            args.checkpoint.resolve(),
            dataset_path=dataset_path,
            statistics_path=statistics_path,
            acceptance_path=acceptance_path,
        )
        model = jax.jit(canonical_model_apply)
        parameters = jax.tree_util.tree_map(jax.numpy.asarray, checkpoint["parameters"])
        checkpoint_identity = checkpoint["identity"]

    records = []
    started = time.perf_counter()
    for day_index in range(args.start_day, stop_day + 1):
        offset = day_index - 1
        current_packet = _packet_from_canonical_state(
            current_state,
            current_discrete,
            contract,
            tstep=(day_index - 1) * steps_per_day - 1,
        )
        if args.mode == "teacher-replay":
            fast_day_target = np.asarray(shard.fast_day_target[offset])
        else:
            inference = prepare_canonical_inference_batch(
                {
                    "state": current_state[None, :],
                    "forcing_native": shard.forcing_native[offset][None, ...],
                    "parameters": shard.parameters[None, :],
                    "landpoint_static": shard.landpoint_static[None, :],
                    "annual_conditions": shard.annual_conditions[None, :],
                    "year": np.asarray([args.year]),
                    "day_index": np.asarray([day_index]),
                },
                statistics,
                representation,
            )
            if (
                inference.model_input.state.shape[-1] != model_config.state_width
                or inference.model_input.forcing_native.shape[-1]
                != model_config.forcing_width
            ):
                raise ValueError("rollout input width does not match checkpoint")
            prediction = model(parameters, inference.model_input)
            prediction = jax.device_get(prediction)
            fast_day_target = restore_fast_day_inference_prediction(
                prediction.normalized_fast_day_target,
                prediction.dynamic_undefined_flip_logits,
                inference,
                statistics,
                representation,
            )[0]
        boundary = reconstruct_fast_day_target(
            fast_day_target,
            contract.fast_day_target_leaves,
            template_fields=current_packet.fields_by_component,
        )
        compiled_forcing = reconstruct_compiled_forcing_day(
            shard.forcing_native[offset],
            contract.native_forcing,
            context,
            year=args.year,
            day_index=day_index,
        )
        tail = _run_retained_tail_day(
            config_path=config_path,
            context=context,
            previous_state=current_packet,
            boundary=boundary,
            compiled_forcing=compiled_forcing,
            static=static,
            contract_finalize_fields=contract_finalize_fields,
            year=args.year,
            day_index=day_index,
        )
        next_state, next_discrete = extract_state(tail.day_end_state, contract)
        expected = np.asarray(shard.state_trajectory[offset + 1])
        expected_discrete = {
            name: np.asarray(values[offset + 1])
            for name, values in shard.discrete_trajectories.items()
        }
        discrete_mismatches = sum(
            int(np.count_nonzero(next_discrete[name] != expected_discrete[name]))
            for name in expected_discrete
        )
        records.append(
            {
                "day_index": day_index,
                "state": _state_metrics(next_state, expected, statistics, contract),
                "discrete_mismatches": discrete_mismatches,
            }
        )
        if args.state_feedback == "teacher":
            current_state = expected.copy()
            current_discrete = {
                name: value.copy() for name, value in expected_discrete.items()
            }
        else:
            current_state = np.asarray(next_state)
            current_discrete = {
                name: np.asarray(value) for name, value in next_discrete.items()
            }

    elapsed = time.perf_counter() - started
    replay_gate = all(
        record["state"]["defined_status_mismatches"] == 0
        and record["discrete_mismatches"] == 0
        and record["state"]["normalized_rmse"] <= args.teacher_replay_tolerance
        for record in records
    )
    status = (
        ("passed" if replay_gate else "failed")
        if args.mode == "teacher-replay"
        else "completed"
    )
    return {
        "schema_version": "canonical_daily_validation_rollout_v1",
        "status": status,
        "mode": args.mode,
        "dataset_id": load_dataset_index(dataset_path).dataset_id,
        "dataset_manifest_sha256": _sha256_file(dataset_path),
        "contract_sha256": contract.sha256,
        "checkpoint_identity": checkpoint_identity,
        "checkpoint": (
            None
            if args.mode == "teacher-replay"
            else {
                "path": str(args.checkpoint.resolve()),
                "sha256": _sha256_file(args.checkpoint.resolve()),
            }
        ),
        "selection": {
            "landpoint_id": args.landpoint_id,
            "year": args.year,
            "spatial_split": reference.spatial_split,
            "temporal_split": reference.temporal_split,
            "start_day": args.start_day,
            "days": args.days,
            "state_feedback": args.state_feedback,
            "sealed_test_used": False,
        },
        "teacher_replay_tolerance": args.teacher_replay_tolerance,
        "teacher_replay_gate": (
            replay_gate if args.mode == "teacher-replay" else None
        ),
        "elapsed_seconds": elapsed,
        "days": records,
        "summary": {
            "last_day_normalized_state_rmse": records[-1]["state"][
                "normalized_rmse"
            ],
            "max_normalized_state_rmse": max(
                record["state"]["normalized_rmse"] for record in records
            ),
            "defined_status_mismatches": sum(
                record["state"]["defined_status_mismatches"] for record in records
            ),
            "discrete_mismatches": sum(
                record["discrete_mismatches"] for record in records
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--landpoint-id", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--start-day", type=int, default=100)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument(
        "--mode",
        choices=("teacher-replay", "neural"),
        default="neural",
    )
    parser.add_argument("--teacher-replay-tolerance", type=float, default=1.0e-10)
    parser.add_argument(
        "--state-feedback",
        choices=("free", "teacher"),
        default="free",
        help="Use predicted or Teacher next-day state as the following input.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.mode == "neural" and args.checkpoint is None:
        raise ValueError("neural rollout requires --checkpoint")
    if args.mode == "teacher-replay" and args.state_feedback != "free":
        raise ValueError("Teacher replay does not accept Teacher-forced feedback")
    result = run_rollout(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
