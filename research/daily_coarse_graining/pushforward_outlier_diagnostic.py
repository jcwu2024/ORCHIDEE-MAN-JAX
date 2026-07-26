"""Attribute a deterministic pushforward loss outlier without retraining."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import numpy as np

from research.daily_coarse_graining.canonical_multistep import (
    canonical_multistep_rollout,
)
from research.daily_coarse_graining.canonical_multistep_training_run import (
    _compiled_forcing_batch,
    _load_plan,
    _make_runtime,
    _sequence,
)
from research.daily_coarse_graining.canonical_pushforward_training_run import (
    _make_prefix_step,
    _matched_training_sequence,
    _slice_sequence,
    _teacher_targets,
    parse_prefix_curriculum,
)
from research.daily_coarse_graining.canonical_rollout import _load_neural_checkpoint
from research.daily_coarse_graining.canonical_teacher_reentry import (
    teacher_reentry_templates,
)
from research.daily_coarse_graining.canonical_training import (
    fast_day_target_representation_from_contract,
    loss_weights_from_contract,
)
from research.daily_coarse_graining.canonical_training_run import (
    load_contract_metadata,
    verify_training_acceptance,
)
from research.daily_coarse_graining.daily_markov_contract import (
    daily_markov_contract_from_metadata,
    load_markov_shard,
)
from research.daily_coarse_graining.markov_dataset import (
    MarkovShardRef,
    collate_windows,
    defined_numeric_mask,
    load_dataset_index,
    load_training_statistics,
)

SCHEMA_VERSION = "pushforward_outlier_diagnostic_v1"

_CONTROL_STATE_KEYS = frozenset(
    {
        "slowproc_stomate_previous_step_state.altmax",
        "slowproc_stomate_previous_step_state.everywhere",
        "slowproc_stomate_previous_step_state.fixed_cryoturbation_depth",
        "slowproc_stomate_previous_step_state.veget_max",
        "sechiba_finalize_state.veget_max",
    }
)
_CARBON_STATE_KEYS = frozenset(
    {
        "slowproc_stomate_previous_step_state.carbon_32l",
        "slowproc_stomate_previous_step_state.DOC",
        "slowproc_stomate_previous_step_state.deepC_peat",
    }
)
_CARBON_FAST_KEYS = frozenset(
    {
        "ok_leak.carbon_32l",
        "ok_leak.DOC",
        "ok_leak.deepC_peat",
    }
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def replay_selection(
    references: Sequence[MarkovShardRef],
    *,
    curriculum: str,
    batch_size: int,
    seed: int,
    target_prefix: int,
    target_step: int,
) -> tuple[MarkovShardRef, np.ndarray, Mapping[int, Mapping[str, int]]]:
    """Replay only the RNG calls made by the fixed pushforward trainer."""

    stages = parse_prefix_curriculum(curriculum)
    rng = np.random.default_rng(seed)
    history = {}
    selected = None
    for stage in stages:
        if target_step >= stage.steps and stage.prefix_days == target_prefix:
            raise ValueError("target step falls outside its prefix stage")
        order = rng.permutation(len(references))
        counts: Counter[str] = Counter()
        for step in range(stage.steps):
            if step and step % len(order) == 0:
                order = rng.permutation(len(references))
            reference = references[int(order[step % len(order)])]
            days = 364 if reference.year == 1961 else 365
            horizon = stage.prefix_days + 1
            choices = days - horizon + 1
            starts = rng.integers(0, choices, size=batch_size)
            counts[reference.landpoint_id] += 1
            if stage.prefix_days == target_prefix and step == target_step:
                selected = (reference, starts)
        history[stage.prefix_days] = dict(sorted(counts.items()))
    if selected is None:
        raise ValueError("target prefix is absent from the curriculum")
    return selected[0], selected[1], history


def _normalized_huber(actual, expected, scale):
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    scale = np.asarray(scale, dtype=np.float64)
    common = defined_numeric_mask(actual) & defined_numeric_mask(expected)
    error = np.where(common, (actual - expected) / scale, 0.0)
    absolute = np.abs(error)
    huber = np.where(absolute <= 1.0, 0.5 * error * error, absolute - 0.5)
    return common, error, huber


def _leaf_attribution(
    actual,
    expected,
    scale,
    weights,
    leaves,
    *,
    owner_name: str,
) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    common, error, huber = _normalized_huber(actual, expected, scale)
    weights = np.asarray(weights, dtype=np.float64)
    weighted = huber * weights[None, :]
    total = float(np.sum(weighted))
    records = []
    for leaf in leaves:
        start = int(leaf.start)
        stop = int(leaf.stop)
        selected_common = common[:, start:stop]
        count = int(np.count_nonzero(selected_common))
        selected_error = error[:, start:stop]
        selected_weighted = weighted[:, start:stop]
        contribution = float(np.sum(selected_weighted))
        sample_metrics = []
        for sample_index in range(actual.shape[0]):
            sample_common = selected_common[sample_index]
            sample_count = int(np.count_nonzero(sample_common))
            sample_error = selected_error[sample_index]
            sample_metrics.append(
                {
                    "sample_index": sample_index,
                    "evaluated_values": sample_count,
                    "normalized_rmse": (
                        float(np.sqrt(np.sum(sample_error**2) / sample_count))
                        if sample_count
                        else None
                    ),
                    "maximum_absolute_normalized_error": (
                        float(np.max(np.abs(sample_error))) if sample_count else None
                    ),
                    "defined_status_mismatches": int(
                        np.count_nonzero(
                            defined_numeric_mask(actual[sample_index, start:stop])
                            != defined_numeric_mask(
                                expected[sample_index, start:stop]
                            )
                        )
                    ),
                    "maximum_absolute_teacher_value": (
                        float(
                            np.max(
                                np.abs(
                                    expected[sample_index, start:stop][sample_common]
                                )
                            )
                        )
                        if sample_count
                        else None
                    ),
                    "maximum_absolute_prediction_value": (
                        float(
                            np.max(
                                np.abs(actual[sample_index, start:stop][sample_common])
                            )
                        )
                        if sample_count
                        else None
                    ),
                }
            )
        records.append(
            {
                "key": leaf.key,
                "owner": getattr(leaf, owner_name),
                "width": stop - start,
                "evaluated_values": count,
                "normalized_rmse": (
                    float(np.sqrt(np.sum(selected_error**2) / count)) if count else None
                ),
                "maximum_absolute_normalized_error": (
                    float(np.max(np.abs(selected_error))) if count else None
                ),
                "weighted_huber_contribution": contribution,
                "weighted_huber_share": contribution / total if total > 0.0 else None,
                "maximum_absolute_teacher_value": (
                    float(np.max(np.abs(expected[:, start:stop][selected_common])))
                    if count
                    else None
                ),
                "maximum_absolute_prediction_value": (
                    float(np.max(np.abs(actual[:, start:stop][selected_common])))
                    if count
                    else None
                ),
                "defined_status_mismatches": int(
                    np.count_nonzero(
                        defined_numeric_mask(actual[:, start:stop])
                        != defined_numeric_mask(expected[:, start:stop])
                    )
                ),
                "samples": sample_metrics,
            }
        )
    ranked = sorted(
        records,
        key=lambda item: item["weighted_huber_contribution"],
        reverse=True,
    )
    count = int(np.count_nonzero(common))
    return {
        "normalized_rmse": float(np.sqrt(np.sum(error**2) / count)) if count else None,
        "weighted_huber_total": total,
        "defined_status_mismatches": int(
            np.count_nonzero(defined_numeric_mask(actual) != defined_numeric_mask(expected))
        ),
        "top_leaves": ranked[:20],
    }


def _selected_leaf_values(values, leaves, keys: frozenset[str]) -> dict[str, Any]:
    """Keep exact values and masks for a small, predeclared diagnostic subset."""

    values = np.asarray(values, dtype=np.float64)
    selected = {}
    for leaf in leaves:
        if leaf.key not in keys:
            continue
        start = int(leaf.start)
        stop = int(leaf.stop)
        subset = values[:, start:stop]
        selected[leaf.key] = {
            "shape": list(getattr(leaf, "shape", (stop - start,))),
            "values": subset.tolist(),
            "defined": defined_numeric_mask(subset).tolist(),
        }
    missing = sorted(keys - selected.keys())
    if missing:
        raise ValueError(f"diagnostic contract is missing selected leaves {missing}")
    return selected


def run_diagnostic(args: argparse.Namespace) -> Mapping[str, Any]:
    dataset = args.dataset.resolve()
    statistics_path = args.statistics.resolve()
    acceptance = args.acceptance.resolve()
    checkpoint_path = args.checkpoint.resolve()
    verify_training_acceptance(acceptance, dataset, statistics_path)
    index = load_dataset_index(dataset, verify_hashes=True)
    references = list(index.select(spatial_split="train", temporal_split="train"))
    reference, starts, replay_counts = replay_selection(
        references,
        curriculum=args.curriculum,
        batch_size=args.batch_size,
        seed=args.seed,
        target_prefix=args.target_prefix,
        target_step=args.target_step,
    )
    shard = load_markov_shard(reference.path)
    horizon = args.target_prefix + 1
    if np.any(starts + horizon > shard.days):
        raise ValueError("replayed sample extends beyond the selected shard")
    batch = collate_windows(
        tuple(shard.window(int(start), horizon) for start in starts)
    )

    raw_manifest = json.loads(dataset.read_text(encoding="utf-8"))
    plan_path = (args.plan or Path(raw_manifest["plan"])).resolve()
    plan, entries = _load_plan(plan_path, str(raw_manifest["plan_sha256"]))
    metadata = load_contract_metadata(dataset)
    contract = daily_markov_contract_from_metadata(metadata)
    representation = fast_day_target_representation_from_contract(metadata)
    statistics = load_training_statistics(statistics_path, index=index)
    checkpoint, _model_config = _load_neural_checkpoint(
        checkpoint_path,
        dataset_path=dataset,
        statistics_path=statistics_path,
        acceptance_path=acceptance,
    )
    parameters = jax.tree_util.tree_map(jax.numpy.asarray, checkpoint["parameters"])
    config_path = Path(plan["teacher_config"])
    runtime = _make_runtime(
        landpoint_id=reference.landpoint_id,
        entry=entries[reference.landpoint_id],
        config_path=config_path,
        contract=contract,
        batch=batch,
    )
    compiled_forcing = _compiled_forcing_batch(batch, contract, runtime.context)
    full_sequence = _sequence(batch, compiled_forcing)
    prefix_sequence = _slice_sequence(full_sequence, 0, args.target_prefix)
    prefix = _make_prefix_step(
        statistics=statistics,
        representation=representation,
        transition=runtime.transition,
    )
    terminal = jax.device_get(
        prefix(
            parameters,
            jax.numpy.asarray(batch["initial_state"]),
            jax.tree_util.tree_map(jax.numpy.asarray, batch["initial_discrete_state"]),
            prefix_sequence,
        )
    )
    terminal_states = np.asarray(terminal.continuous_state)
    terminal_discrete = {
        name: np.asarray(value) for name, value in terminal.discrete_state.items()
    }
    teacher_fast, teacher_next, _teacher_next_discrete = _teacher_targets(
        batch=batch,
        offset=args.target_prefix,
        terminal_states=terminal_states,
        terminal_discrete_states=terminal_discrete,
        config_path=config_path,
        runtime=runtime,
        templates=teacher_reentry_templates(runtime.context),
        contract=contract,
    )
    matched = _matched_training_sequence(
        batch,
        compiled_forcing,
        offset=args.target_prefix,
        terminal_states=terminal_states,
        teacher_fast_targets=teacher_fast,
        teacher_next_states=teacher_next,
    )
    fast_weights = jax.numpy.asarray(loss_weights_from_contract(metadata))

    def one(initial_state, initial_discrete, sequence):
        return canonical_multistep_rollout(
            parameters,
            initial_state,
            initial_discrete,
            sequence,
            statistics=statistics,
            representation=representation,
            fast_day_weights=fast_weights,
            retained_tail_transition=runtime.transition,
            state_loss_weight=1.0,
            undefined_loss_weight=0.1,
        )

    result = jax.device_get(
        jax.jit(jax.vmap(one))(terminal_states, terminal_discrete, matched)
    )
    predicted_fast = np.asarray(result.steps.physical_fast_day_target[:, 0])
    predicted_next = np.asarray(result.final_carry.continuous_state)
    fast_attribution = _leaf_attribution(
        predicted_fast,
        teacher_fast,
        statistics.arrays["fast_day_target"].scale,
        loss_weights_from_contract(metadata),
        contract.fast_day_target_leaves,
        owner_name="family",
    )
    state_leaves = tuple(leaf for leaf in contract.state_leaves if not leaf.discrete)
    clean_terminal_states = np.asarray(
        batch["state_trajectory"][:, args.target_prefix]
    )
    terminal_state_attribution = _leaf_attribution(
        terminal_states,
        clean_terminal_states,
        statistics.arrays["state"].scale,
        np.full(contract.continuous_state_width, 1.0 / contract.continuous_state_width),
        state_leaves,
        owner_name="component",
    )
    state_attribution = _leaf_attribution(
        predicted_next,
        teacher_next,
        statistics.arrays["state"].scale,
        np.full(contract.continuous_state_width, 1.0 / contract.continuous_state_width),
        state_leaves,
        owner_name="component",
    )
    losses = np.asarray(result.loss, dtype=np.float64)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "dataset": {
            "id": index.dataset_id,
            "manifest_sha256": _sha256_file(dataset),
            "sealed_test_used": False,
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": _sha256_file(checkpoint_path),
            "training_objective": checkpoint["identity"].get("training_objective"),
        },
        "selection": {
            "seed": args.seed,
            "curriculum": args.curriculum,
            "prefix_days": args.target_prefix,
            "stage_step_zero_based": args.target_step,
            "landpoint_id": reference.landpoint_id,
            "year": reference.year,
            "window_start_indices_zero_based": starts.tolist(),
            "terminal_day_indices": np.asarray(batch["day_index"])[
                :, args.target_prefix
            ].tolist(),
            "replayed_landpoint_batch_counts": replay_counts,
        },
        "loss": {
            "per_sample": losses.tolist(),
            "mean": float(np.mean(losses)),
            "maximum": float(np.max(losses)),
            "fast_day": np.asarray(result.steps.fast_day_loss[:, 0]).tolist(),
            "state": np.asarray(result.steps.state_loss[:, 0]).tolist(),
            "undefined": np.asarray(result.steps.undefined_loss[:, 0]).tolist(),
        },
        "fast_day_attribution": fast_attribution,
        "terminal_state_vs_clean_attribution": terminal_state_attribution,
        "next_state_attribution": state_attribution,
        "control_state_values": {
            "model_terminal": _selected_leaf_values(
                terminal_states,
                state_leaves,
                _CONTROL_STATE_KEYS,
            ),
            "clean_terminal": _selected_leaf_values(
                clean_terminal_states,
                state_leaves,
                _CONTROL_STATE_KEYS,
            ),
            "model_terminal_discrete": {
                name: np.asarray(value).tolist()
                for name, value in terminal_discrete.items()
            },
            "clean_terminal_discrete": {
                name: np.asarray(value)[:, args.target_prefix].tolist()
                for name, value in batch["discrete_trajectory"].items()
            },
        },
        "carbon_status_values": {
            "model_terminal": _selected_leaf_values(
                terminal_states,
                state_leaves,
                _CARBON_STATE_KEYS,
            ),
            "clean_terminal": _selected_leaf_values(
                clean_terminal_states,
                state_leaves,
                _CARBON_STATE_KEYS,
            ),
            "teacher_fast": _selected_leaf_values(
                teacher_fast,
                contract.fast_day_target_leaves,
                _CARBON_FAST_KEYS,
            ),
            "model_fast": _selected_leaf_values(
                predicted_fast,
                contract.fast_day_target_leaves,
                _CARBON_FAST_KEYS,
            ),
        },
        "classification": {
            "extreme_loss_reproduced": bool(np.max(losses) > 1.0e3),
            "extreme_threshold": 1.0e3,
            "dominant_fast_day_leaf": fast_attribution["top_leaves"][0]["key"],
            "dominant_next_state_leaf": state_attribution["top_leaves"][0]["key"],
            "terminal_state_defined_status_mismatches": terminal_state_attribution[
                "defined_status_mismatches"
            ],
            "dominant_terminal_state_leaf": terminal_state_attribution["top_leaves"][
                0
            ]["key"],
            "defined_status_mismatches": (
                fast_attribution["defined_status_mismatches"]
                + state_attribution["defined_status_mismatches"]
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--curriculum", default="0:48,1:48,3:48,7:48")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--target-prefix", type=int, default=7)
    parser.add_argument("--target-step", type=int, default=47)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_diagnostic(args)
    _atomic_json(args.output.resolve(), result)
    print(json.dumps(result["classification"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
