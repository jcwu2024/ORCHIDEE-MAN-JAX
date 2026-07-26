"""Diagnose neural rollout drift with Teacher queries at model-visited states."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import numpy as np

from jax_orchidee.driver import orchestration as teacher
from research.daily_coarse_graining.canonical_rollout import (
    _contract_finalize_fields,
    _load_neural_checkpoint,
    _packet_from_canonical_state,
    _plan_entry,
    _run_retained_tail_day,
    _select_reference,
)
from research.daily_coarse_graining.canonical_teacher_reentry import (
    query_teacher_day,
    teacher_reentry_templates,
)
from research.daily_coarse_graining.canonical_training import (
    fast_day_target_representation_from_contract,
    prepare_canonical_inference_batch,
    restore_fast_day_inference_prediction,
)
from research.daily_coarse_graining.canonical_training_run import (
    load_contract_metadata,
    verify_training_acceptance,
)
from research.daily_coarse_graining.daily_markov_contract import (
    DailyMarkovContract,
    daily_markov_contract_from_metadata,
    extract_state,
    load_markov_shard,
    reconstruct_compiled_forcing_day,
    reconstruct_fast_day_target,
)
from research.daily_coarse_graining.markov_dataset import (
    TrainingStatistics,
    defined_numeric_mask,
    load_dataset_index,
    load_training_statistics,
)
from research.daily_coarse_graining.synthetic_operator_cost import (
    _tail_static_inputs,
)

SCHEMA_VERSION = "counterfactual_teacher_reentry_v1"
DEFAULT_NAMED_STATES = (
    "biomass",
    "litterpart",
    "npp_daily",
    "resp_growth",
    "resp_maint",
    "resp_hetero",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _normalized_metrics(actual, expected, scale) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    expected = np.asarray(expected, dtype=np.float64).reshape(-1)
    scale = np.asarray(scale, dtype=np.float64).reshape(-1)
    if actual.shape != expected.shape or actual.shape != scale.shape:
        raise ValueError("metric arrays must have the same flattened shape")
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("metric scale must be finite and positive")
    actual_defined = defined_numeric_mask(actual)
    expected_defined = defined_numeric_mask(expected)
    common = actual_defined & expected_defined
    count = int(np.count_nonzero(common))
    if count:
        physical_error = actual[common] - expected[common]
        normalized_error = physical_error / scale[common]
        normalized_rmse = float(np.sqrt(np.mean(normalized_error**2)))
        normalized_mae = float(np.mean(np.abs(normalized_error)))
        max_absolute_error = float(np.max(np.abs(physical_error)))
    else:
        normalized_rmse = None
        normalized_mae = None
        max_absolute_error = None
    return {
        "normalized_rmse": normalized_rmse,
        "normalized_mae": normalized_mae,
        "max_absolute_error": max_absolute_error,
        "defined_status_mismatches": int(
            np.count_nonzero(actual_defined != expected_defined)
        ),
        "evaluated_values": count,
    }


def _fast_day_metrics(
    actual,
    expected,
    statistics: TrainingStatistics,
    contract: DailyMarkovContract,
) -> dict[str, Any]:
    scale = np.asarray(statistics.arrays["fast_day_target"].scale)
    result = _normalized_metrics(actual, expected, scale)
    families = {}
    for family in dict.fromkeys(leaf.family for leaf in contract.fast_day_target_leaves):
        selected = np.zeros((contract.fast_day_target_width,), dtype=bool)
        for leaf in contract.fast_day_target_leaves:
            if leaf.family == family:
                selected[leaf.start : leaf.stop] = True
        families[family] = _normalized_metrics(
            np.asarray(actual)[selected],
            np.asarray(expected)[selected],
            scale[selected],
        )
    result["families"] = families
    return result


def _state_metrics(
    actual,
    expected,
    statistics: TrainingStatistics,
    contract: DailyMarkovContract,
    *,
    named_states: Sequence[str] = DEFAULT_NAMED_STATES,
) -> dict[str, Any]:
    scale = np.asarray(statistics.arrays["state"].scale)
    result = _normalized_metrics(actual, expected, scale)
    components = {}
    for component in dict.fromkeys(leaf.component for leaf in contract.state_leaves):
        selected = np.zeros((contract.continuous_state_width,), dtype=bool)
        for leaf in contract.state_leaves:
            if leaf.component == component and not leaf.discrete:
                selected[leaf.start : leaf.stop] = True
        if np.any(selected):
            components[component] = _normalized_metrics(
                np.asarray(actual)[selected],
                np.asarray(expected)[selected],
                scale[selected],
            )
    named = {}
    for name in named_states:
        matches = tuple(
            leaf
            for leaf in contract.state_leaves
            if not leaf.discrete and leaf.key.endswith(f".{name}")
        )
        if len(matches) != 1:
            raise ValueError(f"expected one continuous state leaf named {name!r}")
        leaf = matches[0]
        selected = slice(leaf.start, leaf.stop)
        named[name] = _normalized_metrics(
            np.asarray(actual)[selected],
            np.asarray(expected)[selected],
            scale[selected],
        )
    result["components"] = components
    result["named_states"] = named
    return result


def _discrete_metrics(
    actual: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, Any]:
    if set(actual) != set(expected):
        raise ValueError("discrete state schemas do not match")
    leaves = {
        name: int(np.count_nonzero(np.asarray(actual[name]) != np.asarray(expected[name])))
        for name in sorted(expected)
    }
    return {"mismatches": int(sum(leaves.values())), "leaves": leaves}


def _validate_oracle_offsets(offsets: Sequence[int], days: int) -> tuple[int, ...]:
    values = tuple(int(value) for value in offsets)
    if days < 2:
        raise ValueError("counterfactual diagnosis requires at least two rollout days")
    if not values:
        raise ValueError("at least one counterfactual oracle offset is required")
    if tuple(sorted(set(values))) != values:
        raise ValueError("counterfactual oracle offsets must be unique and increasing")
    if values[0] < 1 or values[-1] >= days:
        raise ValueError("oracle offsets must select model-fed days after the first day")
    return values


def _diagnostic_classification(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("counterfactual classification requires records")
    reentry_valid = all(
        record["teacher_reentry"]["completed"]
        and record["teacher_next_vs_clean_next"]["defined_status_mismatches"] == 0
        and record["teacher_discrete_vs_clean_next"]["mismatches"] == 0
        for record in records
        if record["teacher_reentry"]["completed"]
    ) and all(record["teacher_reentry"]["completed"] for record in records)
    if not reentry_valid:
        return {
            "teacher_reentry_valid": False,
            "classification": "teacher_reentry_or_state_validity_failure",
            "mean_operator_normalized_rmse": None,
            "mean_state_sensitivity_normalized_rmse": None,
            "operator_to_state_sensitivity_ratio": None,
        }
    operator = np.asarray(
        [record["neural_next_vs_teacher_next"]["normalized_rmse"] for record in records],
        dtype=np.float64,
    )
    sensitivity = np.asarray(
        [record["teacher_next_vs_clean_next"]["normalized_rmse"] for record in records],
        dtype=np.float64,
    )
    finite = np.isfinite(operator) & np.isfinite(sensitivity)
    if not np.all(finite):
        classification = "teacher_reentry_or_state_validity_failure"
        ratio = None
    else:
        operator_mean = float(np.mean(operator))
        sensitivity_mean = float(np.mean(sensitivity))
        ratio = (
            float(operator_mean / sensitivity_mean)
            if sensitivity_mean > 0.0
            else (float("inf") if operator_mean > 0.0 else 0.0)
        )
        classification = (
            "matched_operator_error_at_least_as_large_as_state_sensitivity"
            if operator_mean >= sensitivity_mean
            else "state_sensitivity_larger_than_matched_operator_error"
        )
    return {
        "teacher_reentry_valid": reentry_valid,
        "classification": classification,
        "mean_operator_normalized_rmse": (
            float(np.mean(operator[finite])) if np.any(finite) else None
        ),
        "mean_state_sensitivity_normalized_rmse": (
            float(np.mean(sensitivity[finite])) if np.any(finite) else None
        ),
        "operator_to_state_sensitivity_ratio": ratio,
    }


def run_diagnostic(args: argparse.Namespace) -> Mapping[str, Any]:
    dataset_path = args.dataset.resolve()
    statistics_path = args.statistics.resolve()
    acceptance_path = args.acceptance.resolve()
    checkpoint_path = args.checkpoint.resolve()
    verify_training_acceptance(acceptance_path, dataset_path, statistics_path)
    reference = _select_reference(
        dataset_path,
        landpoint_id=args.landpoint_id,
        year=args.year,
        allow_training_split_diagnostic=args.allow_training_split_diagnostic,
    )
    shard = load_markov_shard(reference.path)
    if args.start_day < 1 or args.days < 2:
        raise ValueError("diagnostic start day must be positive and days at least two")
    stop_day = args.start_day + args.days - 1
    if stop_day > shard.days:
        raise ValueError("diagnostic window extends beyond the selected shard")
    oracle_offsets = _validate_oracle_offsets(args.oracle_offset, args.days)

    metadata = load_contract_metadata(dataset_path)
    contract = daily_markov_contract_from_metadata(metadata)
    representation = fast_day_target_representation_from_contract(metadata)
    statistics = load_training_statistics(statistics_path)
    contract_finalize_fields = _contract_finalize_fields(contract)
    plan, entry = _plan_entry(
        args.plan.resolve(), landpoint_id=args.landpoint_id, year=args.year
    )
    config_path = Path(plan["teacher_config"])
    context = teacher.prepare_paper_1961_driver_context(
        config_path,
        used_run_def_path=entry["run_def"],
        reference_run_dir=entry["reference_run_dir"],
    )
    reentry_templates = teacher_reentry_templates(context)
    checkpoint, model_definition = _load_neural_checkpoint(
        checkpoint_path,
        dataset_path=dataset_path,
        statistics_path=statistics_path,
        acceptance_path=acceptance_path,
        contract_metadata=metadata,
    )
    model = jax.jit(model_definition.apply)
    parameters = jax.tree_util.tree_map(jax.numpy.asarray, checkpoint["parameters"])

    first_index = args.start_day - 1
    current_state = np.asarray(shard.state_trajectory[first_index]).copy()
    current_discrete = {
        name: np.asarray(values[first_index]).copy()
        for name, values in shard.discrete_trajectories.items()
    }
    steps_per_day = int(round(context.runtime.dt_stomate / context.runtime.dt_sechiba))
    first_packet = _packet_from_canonical_state(
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
        lambda value: np.expand_dims(np.asarray(value), axis=0), first_forcing
    )
    static = _tail_static_inputs(
        context, first_packet, forcing_batch, first_day=False
    )

    trajectory = []
    counterfactual = []
    started = time.perf_counter()
    for relative_offset, day_index in enumerate(range(args.start_day, stop_day + 1)):
        offset = day_index - 1
        clean_state = np.asarray(shard.state_trajectory[offset])
        clean_next = np.asarray(shard.state_trajectory[offset + 1])
        clean_next_discrete = {
            name: np.asarray(values[offset + 1])
            for name, values in shard.discrete_trajectories.items()
        }
        current_packet = _packet_from_canonical_state(
            current_state,
            current_discrete,
            contract,
            tstep=(day_index - 1) * steps_per_day - 1,
        )
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
            inference.model_input.state.shape[-1]
            != model_definition.config.state_width
        ):
            raise ValueError("diagnostic state width does not match checkpoint")
        prediction = jax.device_get(model(parameters, inference.model_input))
        neural_fast = restore_fast_day_inference_prediction(
            prediction.normalized_fast_day_target,
            prediction.dynamic_undefined_flip_logits,
            inference,
            statistics,
            representation,
        )[0]
        neural_boundary = reconstruct_fast_day_target(
            neural_fast,
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
        neural_tail = _run_retained_tail_day(
            config_path=config_path,
            context=context,
            previous_state=current_packet,
            boundary=neural_boundary,
            compiled_forcing=compiled_forcing,
            static=static,
            contract_finalize_fields=contract_finalize_fields,
            year=args.year,
            day_index=day_index,
        )
        neural_next, neural_next_discrete = extract_state(
            neural_tail.day_end_state, contract
        )
        trajectory.append(
            {
                "relative_day": relative_offset + 1,
                "day_index": day_index,
                "model_start_vs_clean_start": _state_metrics(
                    current_state, clean_state, statistics, contract
                ),
                "neural_next_vs_clean_next": _state_metrics(
                    neural_next, clean_next, statistics, contract
                ),
                "neural_discrete_vs_clean_next": _discrete_metrics(
                    neural_next_discrete, clean_next_discrete
                ),
            }
        )

        if relative_offset in oracle_offsets:
            oracle_started = time.perf_counter()
            try:
                oracle = query_teacher_day(
                    config_path=config_path,
                    context=context,
                    templates=reentry_templates,
                    continuous=current_state,
                    discrete=current_discrete,
                    contract=contract,
                    year=args.year,
                    day_index=day_index,
                )
                oracle_fast = oracle.fast_day_target
                oracle_next = oracle.next_state
                oracle_next_discrete = oracle.next_discrete_state
            except Exception as error:  # noqa: BLE001 - failure is diagnostic evidence
                record = {
                    "relative_day": relative_offset + 1,
                    "day_index": day_index,
                    "teacher_reentry": {
                        "completed": False,
                        "elapsed_seconds": time.perf_counter() - oracle_started,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "traceback": traceback.format_exc(),
                    },
                    "model_start_vs_clean_start": _state_metrics(
                        current_state, clean_state, statistics, contract
                    ),
                    "neural_next_vs_clean_next": _state_metrics(
                        neural_next, clean_next, statistics, contract
                    ),
                }
            else:
                oracle_seconds = time.perf_counter() - oracle_started
                clean_fast = np.asarray(shard.fast_day_target[offset])
                record = {
                    "relative_day": relative_offset + 1,
                    "day_index": day_index,
                    "teacher_reentry": {
                        "completed": True,
                        "elapsed_seconds": oracle_seconds,
                    },
                    "model_start_vs_clean_start": _state_metrics(
                        current_state, clean_state, statistics, contract
                    ),
                    "neural_fast_vs_clean_fast": _fast_day_metrics(
                        neural_fast, clean_fast, statistics, contract
                    ),
                    "teacher_fast_vs_clean_fast": _fast_day_metrics(
                        oracle_fast, clean_fast, statistics, contract
                    ),
                    "neural_fast_vs_teacher_fast": _fast_day_metrics(
                        neural_fast, oracle_fast, statistics, contract
                    ),
                    "neural_next_vs_clean_next": _state_metrics(
                        neural_next, clean_next, statistics, contract
                    ),
                    "teacher_next_vs_clean_next": _state_metrics(
                        oracle_next, clean_next, statistics, contract
                    ),
                    "neural_next_vs_teacher_next": _state_metrics(
                        neural_next, oracle_next, statistics, contract
                    ),
                    "teacher_discrete_vs_clean_next": _discrete_metrics(
                        oracle_next_discrete, clean_next_discrete
                    ),
                    "neural_discrete_vs_teacher_next": _discrete_metrics(
                        neural_next_discrete, oracle_next_discrete
                    ),
                }
            counterfactual.append(record)
            print(
                json.dumps(
                    {
                        "relative_day": record["relative_day"],
                        "teacher_reentry": record["teacher_reentry"],
                        "operator_rmse": record.get(
                            "neural_next_vs_teacher_next", {}
                        ).get("normalized_rmse"),
                        "state_sensitivity_rmse": record.get(
                            "teacher_next_vs_clean_next", {}
                        ).get("normalized_rmse"),
                    },
                    indent=2,
                ),
                flush=True,
            )

        current_state = np.asarray(neural_next)
        current_discrete = {
            name: np.asarray(value) for name, value in neural_next_discrete.items()
        }

    classification = _diagnostic_classification(counterfactual)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "dataset_id": load_dataset_index(dataset_path).dataset_id,
        "dataset_manifest_sha256": _sha256_file(dataset_path),
        "contract_sha256": contract.sha256,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256_file(checkpoint_path),
            "training_objective": checkpoint["identity"].get(
                "training_objective"
            ),
            "model_architecture": model_definition.identity(),
        },
        "selection": {
            "landpoint_id": args.landpoint_id,
            "year": args.year,
            "spatial_split": reference.spatial_split,
            "temporal_split": reference.temporal_split,
            "start_day": args.start_day,
            "days": args.days,
            "oracle_relative_offsets_zero_based": list(oracle_offsets),
            "state_feedback": "free",
            "training_split_diagnostic": args.allow_training_split_diagnostic,
            "sealed_test_used": False,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "trajectory": trajectory,
        "counterfactual_teacher_queries": counterfactual,
        "summary": classification,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--landpoint-id", required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--start-day", type=int, default=100)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument(
        "--oracle-offset",
        action="append",
        type=int,
        default=None,
        help="Zero-based window offset to query after Day 1; repeat as needed.",
    )
    parser.add_argument("--allow-training-split-diagnostic", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.oracle_offset is None:
        args.oracle_offset = [1, 3, 6]
    result = run_diagnostic(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output, result)
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
