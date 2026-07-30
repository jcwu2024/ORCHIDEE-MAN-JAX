"""Frozen post-training screening for the matched rollout-stability arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.canonical_multistep import (
    canonical_multistep_rollout,
)
from research.daily_coarse_graining.canonical_multistep_training_run import (
    _compiled_forcing_batch,
    _make_runtime_and_compiled_forcing,
    _sequence,
)
from research.daily_coarse_graining.canonical_training import (
    _defined_numeric_mask_compiled,
)
from research.daily_coarse_graining.canonical_training_run import (
    verify_training_acceptance,
)
from research.daily_coarse_graining.markov_dataset import (
    MarkovShardRef,
    load_markov_shard,
)
from research.daily_coarse_graining.production_training_protocol import (
    load_training_protocol,
)
from research.daily_coarse_graining.rollout_stability_objective import (
    SCIENCE_FIELDS,
)
from research.daily_coarse_graining.rollout_stability_protocol import (
    RolloutStabilityProtocol,
    load_rollout_stability_protocol,
)
from research.daily_coarse_graining.rollout_stability_run import (
    ARM_IDS,
    _arm_identity,
    _bind_dynamic_transition,
    _collate_shard_windows,
    _load_pickle,
    _load_rollout_resources,
    _load_verified_execution_manifest,
    _sha256_file,
    broadcast_retained_tail_day_inputs,
    retained_tail_trace_signature,
    verify_arm_checkpoint,
)

REPORT_SCHEMA_VERSION = "canonical_rollout_stability_screening_report_v1"
PROGRESS_SCHEMA_VERSION = "canonical_rollout_stability_screening_progress_v2"
ARM_REPORT_SCHEMA_VERSION = "canonical_rollout_stability_screening_arm_v1"
REQUIRED_SLICES = ("temporal", "spatial", "joint")
REQUIRED_HORIZONS = (1, 7, 30)
REQUIRED_FEEDBACK = ("teacher_forced", "free")
RELATIVE_DENOMINATOR_SCALE_RATIO = 1.0e-6
CHECKPOINT_EVERY_REFERENCES = 5


class WeightedErrorSums(NamedTuple):
    normalized_sum: Any
    normalized_squared_sum: Any
    normalized_huber_sum: Any
    physical_sum: Any
    physical_absolute_sum: Any
    relative_absolute_sum: Any
    evaluated_values: Any
    relative_evaluated_values: Any


class WeightedHardSums(NamedTuple):
    state_unexpected_defined_status_mismatches: Any
    state_declared_dynamic_status_mismatches: Any
    state_dynamic_evaluated_values: Any
    fast_unexpected_defined_status_mismatches: Any
    fast_declared_dynamic_status_mismatches: Any
    fast_dynamic_evaluated_values: Any
    discrete_state_mismatches: Any
    nonfinite_defined_state_values: Any
    nonfinite_defined_fast_values: Any
    negative_source_nonnegative_carbon_stocks: Any


class ScreeningBatchSums(NamedTuple):
    state_all: WeightedErrorSums
    state_terminal: WeightedErrorSums
    fast_all: WeightedErrorSums
    fast_terminal: WeightedErrorSums
    tendency_all: WeightedErrorSums
    hard_all: WeightedHardSums


def _current_git_head() -> str:
    root = Path(__file__).resolve().parents[2]
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
    ).strip()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _atomic_pickle(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _cell_id(horizon: int, feedback: str) -> str:
    if horizon not in REQUIRED_HORIZONS or feedback not in REQUIRED_FEEDBACK:
        raise ValueError("invalid rollout-stability screening cell")
    return f"horizon_{horizon:03d}_{feedback}"


def _weighted_error_sums(
    predicted,
    teacher,
    scale,
    weights,
    *,
    relative_floor,
) -> WeightedErrorSums:
    predicted = jnp.asarray(predicted)
    teacher = jnp.asarray(teacher)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    if predicted.ndim != 3 or teacher.shape != predicted.shape:
        raise ValueError("screening values require matching batch/horizon/column axes")
    if weights.ndim != 3 or weights.shape[1:] != predicted.shape[:2]:
        raise ValueError("screening weights do not match batch/horizon axes")
    predicted_defined = _defined_numeric_mask_compiled(predicted)
    teacher_defined = _defined_numeric_mask_compiled(teacher)
    common = predicted_defined & teacher_defined
    error = jnp.where(common, predicted - teacher, 0.0)
    normalized = error / jnp.asarray(scale)
    absolute_normalized = jnp.abs(normalized)
    huber = jnp.where(
        absolute_normalized <= 1.0,
        0.5 * normalized * normalized,
        absolute_normalized - 0.5,
    )
    relative_valid = common & (jnp.abs(teacher) >= jnp.asarray(relative_floor))
    relative = jnp.where(
        relative_valid,
        jnp.abs(error) / jnp.maximum(jnp.abs(teacher), relative_floor),
        0.0,
    )
    weighted_common = weights[..., None] * common[None, ...]
    weighted_relative = weights[..., None] * relative_valid[None, ...]
    axes = (1, 2)
    return WeightedErrorSums(
        normalized_sum=jnp.sum(
            weights[..., None] * normalized[None, ...],
            axis=axes,
        ),
        normalized_squared_sum=jnp.sum(
            weights[..., None] * normalized[None, ...] ** 2,
            axis=axes,
        ),
        normalized_huber_sum=jnp.sum(
            weights[..., None] * huber[None, ...],
            axis=axes,
        ),
        physical_sum=jnp.sum(
            weights[..., None] * error[None, ...],
            axis=axes,
        ),
        physical_absolute_sum=jnp.sum(
            weights[..., None] * jnp.abs(error)[None, ...],
            axis=axes,
        ),
        relative_absolute_sum=jnp.sum(
            weights[..., None] * relative[None, ...],
            axis=axes,
        ),
        evaluated_values=jnp.sum(weighted_common, axis=axes),
        relative_evaluated_values=jnp.sum(weighted_relative, axis=axes),
    )


def _per_sample_discrete_mismatches(predicted, teacher):
    predicted_leaves, predicted_tree = jax.tree_util.tree_flatten(predicted)
    teacher_leaves, teacher_tree = jax.tree_util.tree_flatten(teacher)
    if predicted_tree != teacher_tree:
        raise ValueError("screening discrete-state trees differ")
    total = None
    for left, right in zip(predicted_leaves, teacher_leaves, strict=True):
        mismatch = jnp.asarray(left) != jnp.asarray(right)
        if mismatch.ndim < 2:
            raise ValueError("screening discrete state lacks batch/horizon axes")
        if mismatch.ndim > 2:
            mismatch = jnp.sum(
                mismatch,
                axis=tuple(range(2, mismatch.ndim)),
            )
        else:
            mismatch = mismatch.astype(jnp.int64)
        total = mismatch if total is None else total + mismatch
    if total is None:
        raise ValueError("screening discrete state is empty")
    return total


def _weighted_count(mask, weights):
    mask = jnp.asarray(mask)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    if mask.ndim == 2:
        return jnp.sum(weights * mask[None, ...], axis=(1, 2))
    return jnp.sum(weights[..., None] * mask[None, ...], axis=(1, 2, 3))


def make_screening_batch_step(
    *,
    resources,
    transition,
):
    """Compile one forward-only summary kernel shared by both frozen arms."""

    state_scale = jnp.asarray(resources.statistics.arrays["state"].scale)
    fast_scale = jnp.asarray(resources.statistics.arrays["fast_day_target"].scale)
    delta_scale = jnp.asarray(resources.state_delta_scale)
    state_relative_floor = RELATIVE_DENOMINATOR_SCALE_RATIO * state_scale
    fast_relative_floor = RELATIVE_DENOMINATOR_SCALE_RATIO * fast_scale
    dynamic_fast_indices = np.asarray(
        resources.representation.dynamic_undefined_indices,
        dtype=np.int32,
    )
    dynamic_state_indices = np.asarray(
        resources.representation.state_indices,
        dtype=np.int32,
    )[dynamic_fast_indices]
    if np.any(dynamic_state_indices < 0):
        raise ValueError("screening dynamic output lacks a canonical state owner")
    state_dynamic_mask = (
        jnp.zeros(
            (resources.contract.continuous_state_width,),
            dtype=bool,
        )
        .at[jnp.asarray(dynamic_state_indices)]
        .set(True)
    )
    fast_dynamic_mask = (
        jnp.zeros(
            (resources.contract.fast_day_target_width,),
            dtype=bool,
        )
        .at[jnp.asarray(dynamic_fast_indices)]
        .set(True)
    )
    nonnegative_state_mask = jnp.asarray(
        resources.science_layout.nonnegative_state_mask,
        dtype=bool,
    )
    model_apply = resources.model_definition.apply

    def one(parameters, initial_state, initial_discrete_state, sequence):
        return canonical_multistep_rollout(
            parameters,
            initial_state,
            initial_discrete_state,
            sequence,
            statistics=resources.statistics,
            representation=resources.representation,
            fast_day_weights=resources.fast_day_weights,
            retained_tail_transition=transition,
            state_weights=resources.process_weighting.weights,
            state_loss_weight=1.0,
            undefined_loss_weight=0.1,
            rematerialize=False,
            model_apply=model_apply,
        )

    def summarize(
        parameters,
        initial_states,
        initial_discrete_states,
        sequences,
        teacher_next_discrete_states,
        all_weights,
        terminal_weights,
    ):
        results = jax.vmap(
            lambda initial_state, initial_discrete_state, sequence: one(
                parameters,
                initial_state,
                initial_discrete_state,
                sequence,
            )
        )(initial_states, initial_discrete_states, sequences)
        predicted_state = results.steps.continuous_state
        predicted_fast = results.steps.physical_fast_day_target
        teacher_state = sequences.teacher_next_state
        teacher_fast = sequences.teacher_fast_day_target
        predicted_previous = jnp.concatenate(
            (
                jnp.asarray(initial_states)[:, None, :],
                predicted_state[:, :-1, :],
            ),
            axis=1,
        )
        predicted_tendency = predicted_state - predicted_previous
        teacher_tendency = teacher_state - sequences.teacher_state

        state_all = _weighted_error_sums(
            predicted_state,
            teacher_state,
            state_scale,
            all_weights,
            relative_floor=state_relative_floor,
        )
        state_terminal = _weighted_error_sums(
            predicted_state,
            teacher_state,
            state_scale,
            terminal_weights,
            relative_floor=state_relative_floor,
        )
        fast_all = _weighted_error_sums(
            predicted_fast,
            teacher_fast,
            fast_scale,
            all_weights,
            relative_floor=fast_relative_floor,
        )
        fast_terminal = _weighted_error_sums(
            predicted_fast,
            teacher_fast,
            fast_scale,
            terminal_weights,
            relative_floor=fast_relative_floor,
        )
        tendency_all = _weighted_error_sums(
            predicted_tendency,
            teacher_tendency,
            delta_scale,
            all_weights,
            relative_floor=(RELATIVE_DENOMINATOR_SCALE_RATIO * delta_scale),
        )

        predicted_state_defined = _defined_numeric_mask_compiled(predicted_state)
        teacher_state_defined = _defined_numeric_mask_compiled(teacher_state)
        state_status = predicted_state_defined != teacher_state_defined
        predicted_fast_defined = _defined_numeric_mask_compiled(predicted_fast)
        teacher_fast_defined = _defined_numeric_mask_compiled(teacher_fast)
        fast_status = predicted_fast_defined != teacher_fast_defined
        discrete = _per_sample_discrete_mismatches(
            results.steps.discrete_state,
            teacher_next_discrete_states,
        )
        state_dynamic_values = jnp.broadcast_to(
            state_dynamic_mask,
            predicted_state.shape,
        )
        fast_dynamic_values = jnp.broadcast_to(
            fast_dynamic_mask,
            predicted_fast.shape,
        )
        hard = WeightedHardSums(
            state_unexpected_defined_status_mismatches=_weighted_count(
                state_status & ~state_dynamic_values,
                all_weights,
            ),
            state_declared_dynamic_status_mismatches=_weighted_count(
                state_status & state_dynamic_values,
                all_weights,
            ),
            state_dynamic_evaluated_values=_weighted_count(
                state_dynamic_values,
                all_weights,
            ),
            fast_unexpected_defined_status_mismatches=_weighted_count(
                fast_status & ~fast_dynamic_values,
                all_weights,
            ),
            fast_declared_dynamic_status_mismatches=_weighted_count(
                fast_status & fast_dynamic_values,
                all_weights,
            ),
            fast_dynamic_evaluated_values=_weighted_count(
                fast_dynamic_values,
                all_weights,
            ),
            discrete_state_mismatches=jnp.sum(
                all_weights * discrete[None, ...],
                axis=(1, 2),
            ),
            nonfinite_defined_state_values=_weighted_count(
                teacher_state_defined & ~jnp.isfinite(predicted_state),
                all_weights,
            ),
            nonfinite_defined_fast_values=_weighted_count(
                teacher_fast_defined & ~jnp.isfinite(predicted_fast),
                all_weights,
            ),
            negative_source_nonnegative_carbon_stocks=_weighted_count(
                predicted_state_defined & nonnegative_state_mask & (predicted_state < 0.0),
                all_weights,
            ),
        )
        return ScreeningBatchSums(
            state_all=state_all,
            state_terminal=state_terminal,
            fast_all=fast_all,
            fast_terminal=fast_terminal,
            tendency_all=tendency_all,
            hard_all=hard,
        )

    return jax.jit(summarize)


def _host_tree(value):
    return jax.tree_util.tree_map(lambda item: np.asarray(jax.device_get(item)), value)


def _select_weight_index(value, index: int):
    return jax.tree_util.tree_map(lambda item: np.asarray(item)[index], value)


def _add_trees(left, right):
    if left is None:
        return _host_tree(right)
    return jax.tree_util.tree_map(
        lambda first, second: np.asarray(first) + np.asarray(second),
        left,
        _host_tree(right),
    )


def _empty_accumulators() -> dict[str, Any]:
    return {
        arm: {
            slice_id: {
                _cell_id(horizon, feedback): {
                    "sums": None,
                    "windows": 0,
                    "predicted_days": 0,
                    "batches": 0,
                }
                for horizon in REQUIRED_HORIZONS
                for feedback in REQUIRED_FEEDBACK
            }
            for slice_id in REQUIRED_SLICES
        }
        for arm in ARM_IDS
    }


def _accumulate_cell(
    accumulators: dict[str, Any],
    *,
    arm_id: str,
    slice_id: str,
    horizon: int,
    feedback: str,
    sums,
    windows: int,
    predicted_days: int,
) -> None:
    cell = accumulators[arm_id][slice_id][_cell_id(horizon, feedback)]
    cell["sums"] = _add_trees(cell["sums"], sums)
    cell["windows"] += int(windows)
    cell["predicted_days"] += int(predicted_days)
    cell["batches"] += 1


def _padded_window_batches(shard, *, horizon: int, batch_size: int):
    choices = shard.days - horizon + 1
    if choices < 1:
        raise ValueError("screening shard is shorter than the requested horizon")
    starts = np.arange(choices, dtype=np.int64)
    for offset in range(0, choices, batch_size):
        selected = starts[offset : offset + batch_size]
        valid = selected.size
        if valid < batch_size:
            selected = np.pad(
                selected,
                (0, batch_size - valid),
                mode="edge",
            )
        yield (
            _collate_shard_windows(
                shard,
                starts=selected,
                horizon=horizon,
            ),
            np.arange(batch_size) < valid,
            selected,
        )


def _teacher_window_weights(days: int, day_indices: np.ndarray):
    all_weights = []
    terminal_weights = []
    for horizon in REQUIRED_HORIZONS:
        choices = days - horizon + 1
        lower = np.maximum(0, day_indices - horizon + 1)
        upper = np.minimum(day_indices, choices - 1)
        multiplicity = np.maximum(upper - lower + 1, 0)
        all_weights.append(multiplicity.astype(np.float64))
        terminal_weights.append(((day_indices >= horizon - 1) & (day_indices < days)).astype(np.float64))
    return (
        np.stack(all_weights, axis=0)[:, :, None],
        np.stack(terminal_weights, axis=0)[:, :, None],
    )


def _free_window_weights(valid: np.ndarray, horizon: int):
    all_weights = np.broadcast_to(
        valid[None, :, None],
        (1, valid.size, horizon),
    ).astype(np.float64)
    terminal = np.zeros_like(all_weights)
    terminal[:, :, -1] = valid
    return all_weights, terminal


def build_model_selection_inventory(index, training_protocol) -> dict[str, Any]:
    """Resolve all non-test model-selection shards from the frozen protocol."""

    result = {}
    for slice_id in REQUIRED_SLICES:
        definition = training_protocol.raw["validation"][slice_id]
        if definition.get("model_selection") is not True:
            raise ValueError(f"screening slice {slice_id} is not model-selection data")
        spatial = str(definition["spatial_split"])
        temporal = str(definition["temporal_split"])
        if "test" in {spatial, temporal}:
            raise ValueError("screening inventory attempted to use the sealed test")
        references = tuple(
            sorted(
                index.select(
                    spatial_split=spatial,
                    temporal_split=temporal,
                ),
                key=lambda item: (
                    item.landpoint_id,
                    item.year,
                    str(item.path),
                ),
            )
        )
        if not references:
            raise ValueError(f"screening slice {slice_id} is empty")
        if any(
            reference.spatial_split != spatial
            or reference.temporal_split != temporal
            or "test" in {reference.spatial_split, reference.temporal_split}
            for reference in references
        ):
            raise ValueError(f"screening slice {slice_id} inventory drift")
        result[slice_id] = {
            "spatial_split": spatial,
            "temporal_split": temporal,
            "references": references,
            "shards": len(references),
            "landpoints": len({item.landpoint_id for item in references}),
            "years": sorted({int(item.year) for item in references}),
        }
    return result


def _inventory_identity(inventory: Mapping[str, Any]) -> Mapping[str, Any]:
    payload = {
        slice_id: {
            "spatial_split": inventory[slice_id]["spatial_split"],
            "temporal_split": inventory[slice_id]["temporal_split"],
            "references": [
                {
                    "landpoint_id": item.landpoint_id,
                    "year": int(item.year),
                    "path": str(item.path),
                }
                for item in inventory[slice_id]["references"]
            ],
        }
        for slice_id in REQUIRED_SLICES
    }
    return {
        "schema_version": "rollout_stability_screening_inventory_v1",
        "slices": payload,
        "canonical_sha256": _canonical_digest(payload),
    }


def _selected_tasks(
    inventory: Mapping[str, Any],
    *,
    max_shards_per_slice: int | None,
) -> tuple[tuple[str, MarkovShardRef], ...]:
    tasks = []
    for slice_id in REQUIRED_SLICES:
        references = inventory[slice_id]["references"]
        if max_shards_per_slice is not None:
            references = references[:max_shards_per_slice]
        tasks.extend((slice_id, reference) for reference in references)
    return tuple(
        sorted(
            tasks,
            key=lambda item: (
                item[1].landpoint_id,
                item[0],
                item[1].year,
                str(item[1].path),
            ),
        )
    )


def _verify_execution_artifacts(
    *,
    execution: Mapping[str, Any],
    protocol: RolloutStabilityProtocol,
    verify_dataset_hashes: bool,
) -> None:
    artifacts = execution["artifacts"]
    expected_protocol_artifacts = protocol.raw["artifacts"]
    names = (
        ("dataset_manifest", "dataset_manifest_sha256"),
        ("training_statistics", "training_statistics_sha256"),
        ("acceptance_report", "acceptance_report_sha256"),
    )
    for execution_name, protocol_name in names:
        asset = artifacts[execution_name]
        path = Path(asset["path"]).resolve()
        observed = _sha256_file(path)
        if observed != asset["sha256"]:
            raise ValueError(f"screening execution artifact drift for {execution_name}")
        if observed != expected_protocol_artifacts[protocol_name]:
            raise ValueError(f"screening protocol artifact drift for {execution_name}")
    training_protocol = load_training_protocol(artifacts["training_protocol"]["path"])
    if (
        training_protocol.sha256 != artifacts["training_protocol"]["sha256"]
        or training_protocol.sha256 != expected_protocol_artifacts["training_protocol_sha256"]
    ):
        raise ValueError("screening protocol artifact drift for training_protocol")
    verify_training_acceptance(
        artifacts["acceptance_report"]["path"],
        artifacts["dataset_manifest"]["path"],
        artifacts["training_statistics"]["path"],
        verify_dataset_hashes=verify_dataset_hashes,
    )


def _load_verified_arms(
    *,
    experiment_root: Path,
    execution: Mapping[str, Any],
    protocol: RolloutStabilityProtocol,
    resources,
) -> tuple[dict[str, Any], dict[str, Any]]:
    schedule_path = Path(execution["artifacts"]["sampling_schedule"]["path"]).resolve()
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    parameters = {}
    identities = {}
    budget = int(protocol.raw["optimization"]["screening_updates"])
    for arm_id in ARM_IDS:
        report_path = experiment_root / arm_id / "training_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("schema_version") != "canonical_rollout_stability_arm_report_v1"
            or report.get("status") != "completed"
            or report.get("arm_id") != arm_id
            or int(report.get("updates", -1)) != budget
            or report.get("sealed_test_used") is not False
        ):
            raise ValueError(f"screening arm report is incomplete for {arm_id}")
        checkpoint_path = Path(report["checkpoint"]).resolve()
        if _sha256_file(checkpoint_path) != report["checkpoint_sha256"]:
            raise ValueError(f"screening checkpoint hash drift for {arm_id}")
        checkpoint = _load_pickle(checkpoint_path)
        identity = _arm_identity(
            execution=execution,
            arm_id=arm_id,
            resources=resources,
        )
        verify_arm_checkpoint(
            checkpoint,
            arm_id=arm_id,
            identity=identity,
            schedule=schedule,
        )
        if int(checkpoint["next_update"]) != budget:
            raise ValueError(f"screening checkpoint did not reach budget for {arm_id}")
        if report.get("identity") != identity:
            raise ValueError(f"screening arm-report identity drift for {arm_id}")
        parameters[arm_id] = jax.tree_util.tree_map(
            jnp.asarray,
            checkpoint["parameters"],
        )
        identities[arm_id] = {
            "training_report": {
                "path": str(report_path),
                "sha256": _sha256_file(report_path),
            },
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": report["checkpoint_sha256"],
            },
            "identity": identity,
        }
    return parameters, identities


def _group_masks(resources):
    state = {}
    width = resources.contract.continuous_state_width
    for group in resources.process_weighting.metadata["groups"]:
        mask = np.zeros((width,), dtype=bool)
        for leaf in group["leaves"]:
            mask[int(leaf["start"]) : int(leaf["stop"])] = True
        state[str(group["id"])] = mask
    fast = {}
    fast_width = resources.contract.fast_day_target_width
    for leaf in resources.contract.fast_day_target_leaves:
        mask = fast.setdefault(
            str(leaf.family),
            np.zeros((fast_width,), dtype=bool),
        )
        mask[int(leaf.start) : int(leaf.stop)] = True
    return state, fast


def _error_metrics(sums: WeightedErrorSums, mask: np.ndarray) -> Mapping[str, Any]:
    mask = np.asarray(mask, dtype=bool)
    counts = np.asarray(sums.evaluated_values)[mask]
    total_count = float(np.sum(counts))
    if total_count <= 0:
        raise ValueError("screening metric selection has no evaluated values")
    relative_counts = np.asarray(sums.relative_evaluated_values)[mask]
    relative_count = float(np.sum(relative_counts))
    return {
        "normalized_rmse": float(np.sqrt(np.sum(np.asarray(sums.normalized_squared_sum)[mask]) / total_count)),
        "normalized_mean_bias": float(np.sum(np.asarray(sums.normalized_sum)[mask]) / total_count),
        "normalized_huber": float(np.sum(np.asarray(sums.normalized_huber_sum)[mask]) / total_count),
        "physical_mean_bias": float(np.sum(np.asarray(sums.physical_sum)[mask]) / total_count),
        "physical_mean_absolute_error": float(np.sum(np.asarray(sums.physical_absolute_sum)[mask]) / total_count),
        "relative_mean_absolute_error": (
            float(np.sum(np.asarray(sums.relative_absolute_sum)[mask]) / relative_count) if relative_count > 0 else None
        ),
        "evaluated_values": int(round(total_count)),
        "relative_evaluated_values": int(round(relative_count)),
    }


def _grouped_error_metrics(
    sums: WeightedErrorSums,
    groups: Mapping[str, np.ndarray],
) -> Mapping[str, Any]:
    width = np.asarray(sums.evaluated_values).size
    return {
        "global": _error_metrics(sums, np.ones((width,), dtype=bool)),
        "families": {name: _error_metrics(sums, mask) for name, mask in sorted(groups.items())},
    }


def _tendency_bias_metrics(
    sums: WeightedErrorSums,
    *,
    groups: Mapping[str, np.ndarray],
    process_weights: np.ndarray,
) -> Mapping[str, Any]:
    count = np.asarray(sums.evaluated_values)
    valid = count > 0
    mean = np.divide(
        np.asarray(sums.normalized_sum),
        count,
        out=np.zeros_like(count, dtype=np.float64),
        where=valid,
    )
    absolute = np.abs(mean)
    huber = np.where(absolute <= 1.0, 0.5 * mean * mean, absolute - 0.5)

    def score(mask):
        selected = np.asarray(mask, dtype=bool) & valid
        weights = np.asarray(process_weights, dtype=np.float64) * selected
        denominator = float(np.sum(weights))
        if denominator <= 0:
            raise ValueError("screening tendency selection has no valid values")
        return {
            "weighted_huber_mean_bias": float(np.sum(weights * huber) / denominator),
            "weighted_absolute_normalized_mean_bias": float(np.sum(weights * absolute) / denominator),
            "evaluated_columns": int(np.count_nonzero(selected)),
        }

    return {
        "global": score(np.ones_like(valid, dtype=bool)),
        "families": {name: score(mask) for name, mask in sorted(groups.items())},
    }


def _science_metrics(
    *,
    state: WeightedErrorSums,
    tendency: WeightedErrorSums,
    resources,
) -> Mapping[str, Any]:
    result = {}
    for index, field in enumerate(SCIENCE_FIELDS):
        mask = np.asarray(resources.science_layout.field_masks[index], dtype=bool)
        state_metrics = _error_metrics(state, mask)
        tendency_metrics = _error_metrics(tendency, mask)
        state_score = float(state_metrics["normalized_huber"])
        has_tendency = bool(resources.science_layout.tendency_field_mask[index])
        tendency_score = float(tendency_metrics["normalized_huber"])
        result[field] = {
            "normalized_rmse": state_metrics["normalized_rmse"],
            "mean_bias": state_metrics["physical_mean_bias"],
            "absolute_error": state_metrics["physical_mean_absolute_error"],
            "relative_error_where_denominator_is_scientifically_valid": (state_metrics["relative_mean_absolute_error"]),
            "relative_denominator_policy": ("abs(Teacher) >= 1e-6 * train-only column scale"),
            "tendency_error": tendency_metrics["normalized_rmse"],
            "multiannual_drift_slope": {
                "status": "deferred",
                "reason": (
                    "the frozen screening horizon is at most 30 days; "
                    "multiannual drift is evaluated at the complete-chain gate"
                ),
            },
            "objective_score": (0.5 * (state_score + tendency_score) if has_tendency else state_score),
            "tendency_supervision": has_tendency,
            "evaluated_values": state_metrics["evaluated_values"],
            "relative_evaluated_values": state_metrics["relative_evaluated_values"],
        }
    return result


def _finalize_cell(cell: Mapping[str, Any], *, resources) -> Mapping[str, Any]:
    sums = cell["sums"]
    if sums is None:
        raise ValueError("screening cell has no accumulated evidence")
    state_groups, fast_groups = _group_masks(resources)
    hard = {name: int(round(float(np.asarray(getattr(sums.hard_all, name))))) for name in sums.hard_all._fields}
    return {
        "windows": int(cell["windows"]),
        "predicted_days": int(cell["predicted_days"]),
        "batches": int(cell["batches"]),
        "state_all_leads": _grouped_error_metrics(
            sums.state_all,
            state_groups,
        ),
        "state_terminal_lead": _grouped_error_metrics(
            sums.state_terminal,
            state_groups,
        ),
        "fast_boundary_all_leads": _grouped_error_metrics(
            sums.fast_all,
            fast_groups,
        ),
        "fast_boundary_terminal_lead": _grouped_error_metrics(
            sums.fast_terminal,
            fast_groups,
        ),
        "tendency_bias_all_leads": _tendency_bias_metrics(
            sums.tendency_all,
            groups=state_groups,
            process_weights=resources.process_weighting.weights,
        ),
        "science_terminal_state_and_all_lead_tendency": _science_metrics(
            state=sums.state_terminal,
            tendency=sums.tendency_all,
            resources=resources,
        ),
        "hard_counts": hard,
    }


def _reference_key(slice_id: str, reference: MarkovShardRef) -> str:
    return "|".join(
        (
            slice_id,
            reference.landpoint_id,
            str(int(reference.year)),
            str(reference.path),
        )
    )


def _expected_cell_counts(
    references: Sequence[tuple[str, MarkovShardRef]],
    horizon: int,
    *,
    reference_days: Mapping[str, int],
):
    windows = sum(
        int(reference_days[_reference_key(slice_id, reference)]) - horizon + 1 for slice_id, reference in references
    )
    return {
        "windows": int(windows),
        "predicted_days": int(windows * horizon),
    }


def _ratio_record(
    *,
    gate_id: str,
    candidate: float,
    control: float,
    threshold: float,
) -> Mapping[str, Any]:
    candidate = float(candidate)
    control = float(control)
    if not np.isfinite(candidate) or not np.isfinite(control):
        raise ValueError(f"nonfinite screening metric for {gate_id}")
    if candidate < 0.0 or control < 0.0:
        raise ValueError(f"screening ratio metric must be nonnegative for {gate_id}")
    if control == 0.0:
        ratio = 1.0 if candidate == 0.0 else None
        passed = candidate == 0.0
        policy = "both_zero_equal" if passed else "failed_nonzero_candidate_over_zero_control"
    else:
        ratio = candidate / control
        passed = ratio <= threshold
        policy = "ordinary_ratio"
    return {
        "id": gate_id,
        "candidate": candidate,
        "control": control,
        "ratio": ratio,
        "zero_denominator_policy": policy,
        "threshold_max": float(threshold),
        "passed": bool(passed),
    }


def _require_complete_arm_report(
    report: Mapping[str, Any],
    *,
    expected_counts: Mapping[str, Mapping[int, Mapping[str, int]]],
) -> None:
    if (
        report.get("schema_version") != ARM_REPORT_SCHEMA_VERSION
        or report.get("status") != "completed"
        or report.get("sealed_test_used") is not False
    ):
        raise ValueError("screening refuses an incomplete arm report")
    if set(report.get("slices", {})) != set(REQUIRED_SLICES):
        raise ValueError("screening arm slice inventory is incomplete")
    for slice_id in REQUIRED_SLICES:
        cells = report["slices"][slice_id]["cells"]
        required = {_cell_id(horizon, feedback) for horizon in REQUIRED_HORIZONS for feedback in REQUIRED_FEEDBACK}
        if set(cells) != required:
            raise ValueError(f"screening cell inventory is incomplete for {slice_id}")
        for horizon in REQUIRED_HORIZONS:
            expected = expected_counts[slice_id][horizon]
            for feedback in REQUIRED_FEEDBACK:
                cell = cells[_cell_id(horizon, feedback)]
                if (
                    int(cell["windows"]) != expected["windows"]
                    or int(cell["predicted_days"]) != expected["predicted_days"]
                ):
                    raise ValueError(f"screening evidence is incomplete for {slice_id}/{horizon}/{feedback}")


def classify_screening(
    *,
    control: Mapping[str, Any],
    candidate: Mapping[str, Any],
    protocol: RolloutStabilityProtocol,
    expected_counts: Mapping[str, Mapping[int, Mapping[str, int]]],
) -> Mapping[str, Any]:
    """Apply every frozen relative gate and exact hard constraint."""

    _require_complete_arm_report(control, expected_counts=expected_counts)
    _require_complete_arm_report(candidate, expected_counts=expected_counts)
    gates = protocol.raw["screening_validation"]["gates_relative_to_one_step_continuation_control"]
    ratio_records = []

    for slice_id in REQUIRED_SLICES:
        control_cell = control["slices"][slice_id]["cells"][_cell_id(1, "teacher_forced")]
        candidate_cell = candidate["slices"][slice_id]["cells"][_cell_id(1, "teacher_forced")]
        for boundary in (
            "fast_boundary_terminal_lead",
            "state_terminal_lead",
        ):
            ratio_records.append(
                _ratio_record(
                    gate_id=f"one_step_global/{slice_id}/{boundary}",
                    candidate=candidate_cell[boundary]["global"]["normalized_rmse"],
                    control=control_cell[boundary]["global"]["normalized_rmse"],
                    threshold=gates["one_step_global_ratio_max"],
                )
            )
            for family in sorted(control_cell[boundary]["families"]):
                ratio_records.append(
                    _ratio_record(
                        gate_id=(f"one_step_family/{slice_id}/{boundary}/{family}"),
                        candidate=candidate_cell[boundary]["families"][family]["normalized_rmse"],
                        control=control_cell[boundary]["families"][family]["normalized_rmse"],
                        threshold=gates["one_step_family_ratio_max"],
                    )
                )

        for horizon, threshold_name in (
            (7, "seven_day_free_rollout_global_ratio_max"),
            (30, "thirty_day_free_rollout_global_ratio_max"),
        ):
            control_free = control["slices"][slice_id]["cells"][_cell_id(horizon, "free")]
            candidate_free = candidate["slices"][slice_id]["cells"][_cell_id(horizon, "free")]
            ratio_records.append(
                _ratio_record(
                    gate_id=f"free_rollout_global/{slice_id}/horizon_{horizon}",
                    candidate=candidate_free["state_terminal_lead"]["global"]["normalized_rmse"],
                    control=control_free["state_terminal_lead"]["global"]["normalized_rmse"],
                    threshold=gates[threshold_name],
                )
            )
            for family in (
                "global",
                *sorted(control_free["tendency_bias_all_leads"]["families"]),
            ):
                control_bias = (
                    control_free["tendency_bias_all_leads"][family]
                    if family == "global"
                    else control_free["tendency_bias_all_leads"]["families"][family]
                )
                candidate_bias = (
                    candidate_free["tendency_bias_all_leads"][family]
                    if family == "global"
                    else candidate_free["tendency_bias_all_leads"]["families"][family]
                )
                ratio_records.append(
                    _ratio_record(
                        gate_id=(f"tendency_bias/{slice_id}/horizon_{horizon}/{family}"),
                        candidate=candidate_bias["weighted_huber_mean_bias"],
                        control=control_bias["weighted_huber_mean_bias"],
                        threshold=gates["tendency_bias_ratio_max"],
                    )
                )

        for horizon, feedback in (
            (1, "teacher_forced"),
            (7, "free"),
            (30, "free"),
        ):
            control_science = control["slices"][slice_id]["cells"][_cell_id(horizon, feedback)][
                "science_terminal_state_and_all_lead_tendency"
            ]
            candidate_science = candidate["slices"][slice_id]["cells"][_cell_id(horizon, feedback)][
                "science_terminal_state_and_all_lead_tendency"
            ]
            for field in SCIENCE_FIELDS:
                ratio_records.append(
                    _ratio_record(
                        gate_id=(f"science/{slice_id}/horizon_{horizon}/{feedback}/{field}"),
                        candidate=candidate_science[field]["objective_score"],
                        control=control_science[field]["objective_score"],
                        threshold=gates["science_metric_ratio_max"],
                    )
                )

    hard_records = []
    hard_zero_fields = (
        "state_unexpected_defined_status_mismatches",
        "fast_unexpected_defined_status_mismatches",
        "discrete_state_mismatches",
        "nonfinite_defined_state_values",
        "nonfinite_defined_fast_values",
        "negative_source_nonnegative_carbon_stocks",
    )
    dynamic_control = 0
    dynamic_candidate = 0
    dynamic_values_control = 0
    dynamic_values_candidate = 0
    unique_cells = (
        (1, "teacher_forced"),
        (7, "free"),
        (30, "free"),
    )
    for arm_name, arm in (("control", control), ("candidate", candidate)):
        for slice_id in REQUIRED_SLICES:
            for horizon, feedback in unique_cells:
                cell = arm["slices"][slice_id]["cells"][_cell_id(horizon, feedback)]
                counts = cell["hard_counts"]
                for field in hard_zero_fields:
                    hard_records.append(
                        {
                            "id": (f"{arm_name}/{slice_id}/horizon_{horizon}/{feedback}/{field}"),
                            "count": int(counts[field]),
                            "required": 0,
                            "passed": int(counts[field]) == 0,
                        }
                    )
                if arm_name == "control":
                    dynamic_control += int(counts["state_declared_dynamic_status_mismatches"])
                    dynamic_values_control += int(counts["state_dynamic_evaluated_values"])
                else:
                    dynamic_candidate += int(counts["state_declared_dynamic_status_mismatches"])
                    dynamic_values_candidate += int(counts["state_dynamic_evaluated_values"])
    if dynamic_values_control != dynamic_values_candidate:
        raise ValueError("matched screening dynamic-status denominators differ")
    dynamic_ratio = _ratio_record(
        gate_id="declared_dynamic_status_error_rate",
        candidate=(dynamic_candidate / dynamic_values_candidate if dynamic_values_candidate else 0.0),
        control=(dynamic_control / dynamic_values_control if dynamic_values_control else 0.0),
        threshold=gates["declared_dynamic_status_error_ratio_max"],
    )
    ratio_records.append(dynamic_ratio)

    structural = []
    for arm_name, arm in (("control", control), ("candidate", candidate)):
        structural.extend(
            [
                {
                    "id": f"{arm_name}/restart_split_prediction",
                    "passed": bool(arm["structural_gates"]["restart_split_exact"]),
                },
                {
                    "id": f"{arm_name}/hidden_cross_day_network_memory",
                    "passed": (arm["model_architecture"]["cross_day_memory"] == "canonical_state_only"),
                },
                {
                    "id": f"{arm_name}/source_backed_retained_tail",
                    "passed": (
                        arm["structural_gates"]["retained_tail"] == "canonical_source_backed_dynamic_transition"
                    ),
                },
            ]
        )
    passed = (
        all(record["passed"] for record in ratio_records)
        and all(record["passed"] for record in hard_records)
        and all(record["passed"] for record in structural)
    )
    return {
        "status": "passed" if passed else "rejected",
        "decision": (
            "advance_to_three_seed_confirmation" if passed else "stop_and_attribute_declared_screening_failure"
        ),
        "ratio_gates": ratio_records,
        "hard_constraints": hard_records,
        "structural_constraints": structural,
        "declared_dynamic_status": {
            "control_mismatches": dynamic_control,
            "candidate_mismatches": dynamic_candidate,
            "evaluated_values_per_arm": dynamic_values_control,
            "gate": dynamic_ratio,
        },
    }


def _slice_sequence(sequence, start: int, stop: int):
    return jax.tree_util.tree_map(lambda value: value[start:stop], sequence)


def _restart_split_probe(
    *,
    parameters,
    prepared,
    resources,
    transition,
    split_day: int = 15,
) -> Mapping[str, Any]:
    horizon = int(np.shape(prepared.sequence.day_index)[1])
    if not 0 < split_day < horizon:
        raise ValueError("restart-split probe split is outside the horizon")
    initial_state = prepared.initial_states[0]
    initial_discrete = jax.tree_util.tree_map(
        lambda value: value[0],
        prepared.initial_discrete_states,
    )
    sequence = jax.tree_util.tree_map(
        lambda value: value[0],
        prepared.sequence,
    )
    model_apply = resources.model_definition.apply

    def run(value, state, discrete, selected):
        return canonical_multistep_rollout(
            value,
            state,
            discrete,
            selected,
            statistics=resources.statistics,
            representation=resources.representation,
            fast_day_weights=resources.fast_day_weights,
            retained_tail_transition=transition,
            state_weights=resources.process_weighting.weights,
            state_loss_weight=1.0,
            undefined_loss_weight=0.1,
            rematerialize=False,
            model_apply=model_apply,
        )

    def probe(value, state, discrete, selected):
        complete = run(value, state, discrete, selected)
        first = run(
            value,
            state,
            discrete,
            _slice_sequence(selected, 0, split_day),
        )
        second = run(
            value,
            first.final_carry.continuous_state,
            first.final_carry.discrete_state,
            _slice_sequence(selected, split_day, horizon),
        )
        continuous_equal = jnp.array_equal(
            complete.final_carry.continuous_state,
            second.final_carry.continuous_state,
        )
        continuous_max = jnp.max(jnp.abs(complete.final_carry.continuous_state - second.final_carry.continuous_state))
        left, tree_left = jax.tree_util.tree_flatten(complete.final_carry.discrete_state)
        right, tree_right = jax.tree_util.tree_flatten(second.final_carry.discrete_state)
        if tree_left != tree_right:
            raise ValueError("restart-split discrete tree drift")
        discrete_equal = jnp.all(
            jnp.stack(
                [
                    jnp.array_equal(first_value, second_value)
                    for first_value, second_value in zip(left, right, strict=True)
                ]
            )
        )
        return continuous_equal, continuous_max, discrete_equal

    continuous_equal, continuous_max, discrete_equal = jax.jit(probe)(
        parameters,
        initial_state,
        initial_discrete,
        sequence,
    )
    return {
        "horizon_days": horizon,
        "split_after_days": split_day,
        "continuous_bit_exact": bool(jax.device_get(continuous_equal)),
        "continuous_max_absolute_difference": float(jax.device_get(continuous_max)),
        "discrete_exact": bool(jax.device_get(discrete_equal)),
        "passed": bool(jax.device_get(continuous_equal) and jax.device_get(discrete_equal)),
    }


class _PreparedBatch(NamedTuple):
    initial_states: Any
    initial_discrete_states: Any
    sequence: Any
    teacher_next_discrete_states: Any


def _prepare_batch(resources, raw, runtime, compiled_forcing=None):
    if compiled_forcing is None:
        compiled_forcing = _compiled_forcing_batch(
            raw,
            resources.contract,
            runtime.context,
        )
    batch_size, horizon = np.shape(raw["day_index"])
    retained_inputs = broadcast_retained_tail_day_inputs(
        compiled_forcing,
        runtime.static,
        batch_size=batch_size,
        horizon=horizon,
    )
    return _PreparedBatch(
        initial_states=jnp.asarray(raw["initial_state"]),
        initial_discrete_states=jax.tree_util.tree_map(
            jnp.asarray,
            raw["initial_discrete_state"],
        ),
        sequence=_sequence(raw, retained_inputs),
        teacher_next_discrete_states=jax.tree_util.tree_map(
            jnp.asarray,
            raw["teacher_next_discrete_state"],
        ),
    )


def _evaluation_identity(
    *,
    protocol: RolloutStabilityProtocol,
    execution: Mapping[str, Any],
    arms: Mapping[str, Any],
    inventory_identity: Mapping[str, Any],
    batch_sizes: Mapping[int, int],
    max_shards_per_slice: int | None,
    verify_dataset_hashes: bool,
) -> Mapping[str, Any]:
    payload = {
        "schema_version": "canonical_rollout_stability_screening_identity_v1",
        "evaluation_git_head": _current_git_head(),
        "training_git_head": execution["training_git_head"],
        "protocol_sha256": protocol.sha256,
        "execution_canonical_sha256": execution["canonical_sha256"],
        "artifacts": execution["artifacts"],
        "arms": arms,
        "inventory_canonical_sha256": inventory_identity["canonical_sha256"],
        "horizons": list(REQUIRED_HORIZONS),
        "feedback": list(REQUIRED_FEEDBACK),
        "batch_sizes": {str(key): int(value) for key, value in batch_sizes.items()},
        "relative_denominator_scale_ratio": RELATIVE_DENOMINATOR_SCALE_RATIO,
        "max_shards_per_slice": max_shards_per_slice,
        "dataset_content_hashes_reverified": bool(verify_dataset_hashes),
        "sealed_test_used": False,
    }
    return payload | {"canonical_sha256": _canonical_digest(payload)}


def run_screening(args: argparse.Namespace) -> Path:
    started = time.perf_counter()
    protocol = load_rollout_stability_protocol(args.protocol)
    experiment_root = args.experiment_root.resolve()
    execution_path = experiment_root / "rollout_stability_execution.json"
    execution = _load_verified_execution_manifest(
        execution_path,
        protocol,
        require_current_training_head=False,
    )
    _verify_execution_artifacts(
        execution=execution,
        protocol=protocol,
        verify_dataset_hashes=not args.skip_dataset_content_hash_verification,
    )
    dataset_path = Path(execution["artifacts"]["dataset_manifest"]["path"]).resolve()
    statistics_path = Path(execution["artifacts"]["training_statistics"]["path"]).resolve()
    resources = _load_rollout_resources(
        dataset_path=dataset_path,
        statistics_path=statistics_path,
    )
    training_protocol = load_training_protocol(execution["artifacts"]["training_protocol"]["path"])
    inventory = build_model_selection_inventory(
        resources.index,
        training_protocol,
    )
    inventory_identity = _inventory_identity(inventory)
    parameters, arm_identities = _load_verified_arms(
        experiment_root=experiment_root,
        execution=execution,
        protocol=protocol,
        resources=resources,
    )
    batch_sizes = {
        1: int(args.batch_size_1),
        7: int(args.batch_size_7),
        30: int(args.batch_size_30),
    }
    if any(value < 1 for value in batch_sizes.values()):
        raise ValueError("screening batch sizes must be positive")
    identity = _evaluation_identity(
        protocol=protocol,
        execution=execution,
        arms=arm_identities,
        inventory_identity=inventory_identity,
        batch_sizes=batch_sizes,
        max_shards_per_slice=args.max_shards_per_slice,
        verify_dataset_hashes=not args.skip_dataset_content_hash_verification,
    )
    output_root = args.output_root.resolve()
    progress_path = output_root / "screening_progress.pkl"
    tasks = _selected_tasks(
        inventory,
        max_shards_per_slice=args.max_shards_per_slice,
    )
    if progress_path.exists():
        progress = _load_pickle(progress_path)
        if (
            progress.get("schema_version") != PROGRESS_SCHEMA_VERSION
            or progress.get("identity") != identity
            or progress.get("task_count") != len(tasks)
        ):
            raise ValueError("existing screening progress identity drift")
        accumulators = progress["accumulators"]
        next_task = int(progress["next_task"])
        structural_gates = progress["structural_gates"]
        reference_days = progress["reference_days"]
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        accumulators = _empty_accumulators()
        next_task = 0
        structural_gates = {}
        reference_days = {}

    runtime = None
    runtime_landpoint = None
    transition = None
    trace_signature = None
    compiled_steps = {}
    interval_started = time.perf_counter()

    for task_index in range(next_task, len(tasks)):
        slice_id, reference = tasks[task_index]
        shard = load_markov_shard(reference.path)
        reference_key = _reference_key(slice_id, reference)
        previous_days = reference_days.setdefault(reference_key, int(shard.days))
        if int(previous_days) != int(shard.days):
            raise ValueError("screening shard transition count drift")
        first_batch_for_runtime = runtime_landpoint != reference.landpoint_id

        for raw, valid, starts in _padded_window_batches(
            shard,
            horizon=1,
            batch_size=batch_sizes[1],
        ):
            compiled_forcing = None
            if first_batch_for_runtime:
                runtime, compiled_forcing = _make_runtime_and_compiled_forcing(
                    landpoint_id=reference.landpoint_id,
                    entry=resources.plan_entries[reference.landpoint_id],
                    config_path=resources.config_path,
                    contract=resources.contract,
                    batch=raw,
                )
                runtime_landpoint = reference.landpoint_id
                first_batch_for_runtime = False
                observed_signature = retained_tail_trace_signature(runtime.static)
                if transition is None:
                    transition = _bind_dynamic_transition(resources, runtime)
                    trace_signature = observed_signature
                elif observed_signature != trace_signature:
                    raise ValueError("screening retained-tail trace signature drift")
            prepared = _prepare_batch(
                resources,
                raw,
                runtime,
                compiled_forcing=compiled_forcing,
            )
            if (1, 3) not in compiled_steps:
                compiled_steps[(1, 3)] = make_screening_batch_step(
                    resources=resources,
                    transition=transition,
                )
            step = compiled_steps[(1, 3)]
            all_weights, terminal_weights = _teacher_window_weights(
                shard.days,
                starts,
            )
            all_weights *= valid[None, :, None]
            terminal_weights *= valid[None, :, None]
            for arm_id in ARM_IDS:
                batch_sums = _host_tree(
                    step(
                        parameters[arm_id],
                        *prepared,
                        jnp.asarray(all_weights),
                        jnp.asarray(terminal_weights),
                    )
                )
                for weight_index, horizon in enumerate(REQUIRED_HORIZONS):
                    selected = _select_weight_index(batch_sums, weight_index)
                    windows = int(round(float(np.sum(terminal_weights[weight_index]))))
                    predicted_days = int(round(float(np.sum(all_weights[weight_index]))))
                    _accumulate_cell(
                        accumulators,
                        arm_id=arm_id,
                        slice_id=slice_id,
                        horizon=horizon,
                        feedback="teacher_forced",
                        sums=selected,
                        windows=windows,
                        predicted_days=predicted_days,
                    )
                    if horizon == 1:
                        _accumulate_cell(
                            accumulators,
                            arm_id=arm_id,
                            slice_id=slice_id,
                            horizon=1,
                            feedback="free",
                            sums=selected,
                            windows=windows,
                            predicted_days=predicted_days,
                        )

        for horizon in (7, 30):
            for raw, valid, _ in _padded_window_batches(
                shard,
                horizon=horizon,
                batch_size=batch_sizes[horizon],
            ):
                prepared = _prepare_batch(resources, raw, runtime)
                if (horizon, 1) not in compiled_steps:
                    compiled_steps[(horizon, 1)] = make_screening_batch_step(
                        resources=resources,
                        transition=transition,
                    )
                step = compiled_steps[(horizon, 1)]
                all_weights, terminal_weights = _free_window_weights(
                    valid,
                    horizon,
                )
                for arm_id in ARM_IDS:
                    batch_sums = _host_tree(
                        step(
                            parameters[arm_id],
                            *prepared,
                            jnp.asarray(all_weights),
                            jnp.asarray(terminal_weights),
                        )
                    )
                    _accumulate_cell(
                        accumulators,
                        arm_id=arm_id,
                        slice_id=slice_id,
                        horizon=horizon,
                        feedback="free",
                        sums=_select_weight_index(batch_sums, 0),
                        windows=int(np.count_nonzero(valid)),
                        predicted_days=int(np.count_nonzero(valid) * horizon),
                    )
                if not structural_gates and horizon == 30:
                    for arm_id in ARM_IDS:
                        structural_gates[arm_id] = _restart_split_probe(
                            parameters=parameters[arm_id],
                            prepared=prepared,
                            resources=resources,
                            transition=transition,
                        )

        next_task = task_index + 1
        if next_task % CHECKPOINT_EVERY_REFERENCES == 0 or next_task == len(tasks):
            _atomic_pickle(
                progress_path,
                {
                    "schema_version": PROGRESS_SCHEMA_VERSION,
                    "identity": identity,
                    "task_count": len(tasks),
                    "next_task": next_task,
                    "accumulators": accumulators,
                    "structural_gates": structural_gates,
                    "reference_days": reference_days,
                },
            )
            elapsed = time.perf_counter() - interval_started
            print(
                "rollout_screening "
                f"references={next_task}/{len(tasks)} "
                f"last={reference.landpoint_id}:{reference.year}:{slice_id} "
                f"interval_seconds={elapsed:.3f}",
                flush=True,
            )
            interval_started = time.perf_counter()

    full_evidence = args.max_shards_per_slice is None
    selected_by_slice = {
        slice_id: tuple((task_slice, reference) for task_slice, reference in tasks if task_slice == slice_id)
        for slice_id in REQUIRED_SLICES
    }
    expected_counts = {
        slice_id: {
            horizon: _expected_cell_counts(
                selected_by_slice[slice_id],
                horizon,
                reference_days=reference_days,
            )
            for horizon in REQUIRED_HORIZONS
        }
        for slice_id in REQUIRED_SLICES
    }
    arm_reports = {}
    for arm_id in ARM_IDS:
        slices = {}
        for slice_id in REQUIRED_SLICES:
            cells = {
                cell_id: _finalize_cell(cell, resources=resources)
                for cell_id, cell in accumulators[arm_id][slice_id].items()
            }
            slices[slice_id] = {
                "selection": {
                    "spatial_split": inventory[slice_id]["spatial_split"],
                    "temporal_split": inventory[slice_id]["temporal_split"],
                    "shards": len(selected_by_slice[slice_id]),
                    "complete_population_shards": inventory[slice_id]["shards"],
                },
                "cells": cells,
            }
        architecture = arm_identities[arm_id]["identity"]["model_architecture"]
        arm_report = {
            "schema_version": ARM_REPORT_SCHEMA_VERSION,
            "status": "completed" if full_evidence else "smoke_completed",
            "arm_id": arm_id,
            "identity": arm_identities[arm_id],
            "model_architecture": architecture,
            "slices": slices,
            "structural_gates": {
                "restart_split_exact": bool(structural_gates[arm_id]["passed"]),
                "restart_split_probe": structural_gates[arm_id],
                "retained_tail": "canonical_source_backed_dynamic_transition",
            },
            "sealed_test_used": False,
        }
        arm_reports[arm_id] = arm_report
        _atomic_json(
            output_root / arm_id / "screening_report.json",
            arm_report,
        )

    classification = None
    if full_evidence:
        classification = classify_screening(
            control=arm_reports["one_step_continuation_control"],
            candidate=arm_reports["mixed_horizon_stability_v1"],
            protocol=protocol,
            expected_counts=expected_counts,
        )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": (classification["status"] if classification is not None else "smoke_completed"),
        "promotion_eligible": classification is not None,
        "identity": identity,
        "inventory": inventory_identity,
        "selected_shards": {slice_id: len(selected_by_slice[slice_id]) for slice_id in REQUIRED_SLICES},
        "expected_counts": {
            slice_id: {str(horizon): counts for horizon, counts in horizons.items()}
            for slice_id, horizons in expected_counts.items()
        },
        "arm_reports": {
            arm_id: {
                "path": str(output_root / arm_id / "screening_report.json"),
                "sha256": _sha256_file(output_root / arm_id / "screening_report.json"),
            }
            for arm_id in ARM_IDS
        },
        "classification": classification,
        "elapsed_seconds": time.perf_counter() - started,
        "sealed_test_used": False,
    }
    report_path = output_root / ("matched_screening_report.json" if full_evidence else "screening_smoke_report.json")
    return _atomic_json(report_path, report)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size-1", type=int, default=256)
    parser.add_argument("--batch-size-7", type=int, default=256)
    parser.add_argument("--batch-size-30", type=int, default=128)
    parser.add_argument("--max-shards-per-slice", type=int)
    parser.add_argument(
        "--skip-dataset-content-hash-verification",
        action="store_true",
        help=(
            "Trust the hash-bound complete parent acceptance instead of "
            "rehashing every admitted shard in this evaluation process."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_shards_per_slice is not None and args.max_shards_per_slice < 1:
        raise ValueError("screening smoke shard limit must be positive")
    report = run_screening(args)
    payload = json.loads(report.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "status": payload["status"],
                "promotion_eligible": payload["promotion_eligible"],
                "report": str(report),
                "report_sha256": _sha256_file(report),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
