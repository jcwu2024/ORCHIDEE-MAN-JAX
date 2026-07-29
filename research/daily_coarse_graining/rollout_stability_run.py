"""Prepare and audit the frozen canonical rollout-stability experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.canonical_daily_model import parameter_count
from research.daily_coarse_graining.canonical_multistep import (
    CanonicalMultistepSequence,
)
from research.daily_coarse_graining.canonical_multistep_training_run import (
    LandpointRuntime,
    _compiled_forcing_batch,
    _load_plan,
    _make_runtime_and_compiled_forcing,
    _model_config,
    _sequence,
)
from research.daily_coarse_graining.canonical_retained_tail import (
    CanonicalRetainedTailDayInputs,
    bind_canonical_retained_tail_dynamic_transition,
    canonical_retained_tail_day_inputs,
)
from research.daily_coarse_graining.canonical_state_objective import (
    stabilized_state_delta_scale,
    state_process_weighting_from_contract,
)
from research.daily_coarse_graining.canonical_training import (
    CanonicalTrainingBatch,
    fast_day_target_representation_from_contract,
    loss_weights_from_contract,
    prepare_canonical_batch,
)
from research.daily_coarse_graining.canonical_training_run import (
    CHECKPOINT_SCHEMA_VERSION,
    _adam_update,
    verify_training_acceptance,
)
from research.daily_coarse_graining.daily_markov_contract import (
    daily_markov_contract_from_metadata,
)
from research.daily_coarse_graining.daily_model_architecture import (
    AXIS_PROCESS_COUPLED_V1,
    build_daily_model_definition,
    checkpoint_architecture_id,
    verify_checkpoint_architecture,
)
from research.daily_coarse_graining.markov_dataset import (
    MarkovDatasetIndex,
    MarkovShardRef,
    collate_windows,
    load_dataset_index,
    load_markov_shard,
    load_training_statistics,
)
from research.daily_coarse_graining.production_training_protocol import (
    load_training_protocol,
)
from research.daily_coarse_graining.rollout_stability_objective import (
    RolloutStabilityComponents,
    canonical_fast_day_anchor_loss,
    rollout_stability_candidate_loss,
    rollout_stability_components,
    science_objective_layout_from_contract,
)
from research.daily_coarse_graining.rollout_stability_protocol import (
    REQUIRED_PARENT_DECISION,
    RolloutStabilityProtocol,
    load_rollout_stability_protocol,
)

PREFLIGHT_SCHEMA_VERSION = "canonical_rollout_stability_preflight_v1"
SCHEDULE_SCHEMA_VERSION = "canonical_rollout_stability_sampling_schedule_v1"
CALIBRATION_SCHEMA_VERSION = "canonical_rollout_stability_coefficient_calibration_v1"
PARENT_REPORT_SCHEMA_VERSION = "canonical_architecture_ab_report_v1"
ARM_CHECKPOINT_SCHEMA_VERSION = "canonical_rollout_stability_arm_checkpoint_v1"
EXECUTION_MANIFEST_SCHEMA_VERSION = "canonical_rollout_stability_execution_v1"
ARM_IDS = ("one_step_continuation_control", "mixed_horizon_stability_v1")
HARD_CONSTRAINT_FIELDS = (
    "defined_status_mismatches",
    "discrete_state_mismatches",
    "nonfinite_defined_values",
    "negative_source_nonnegative_carbon_stocks",
)


class RolloutStabilityUpdateResult(NamedTuple):
    """Result of one fail-closed optimizer update."""

    parameters: Any
    optimizer: Any
    loss: Any
    components: Any
    gradient_norm: Any
    nonfinite_gradient_values: Any
    update_applied: Any


class RolloutStabilityResources(NamedTuple):
    raw_manifest: Mapping[str, Any]
    plan: Mapping[str, Any]
    plan_entries: Mapping[str, Any]
    index: MarkovDatasetIndex
    contract: Any
    representation: Any
    statistics: Any
    model_definition: Any
    fast_day_weights: Any
    process_weighting: Any
    state_delta_scale: Any
    state_delta_scale_audit: Mapping[str, Any]
    science_layout: Any
    config_path: Path


class PreparedRolloutUpdate(NamedTuple):
    anchor_batch: CanonicalTrainingBatch
    initial_states: Any
    initial_discrete_states: Any
    sequence: CanonicalMultistepSequence
    teacher_next_discrete_states: Any
    runtime: LandpointRuntime
    trace_signature: str
    anchor_starts: np.ndarray
    rollout_starts: np.ndarray


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _atomic_pickle(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _load_pickle(path: Path) -> Mapping[str, Any]:
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint {path} is not a mapping")
    return value


def _trees_equal(left: Any, right: Any) -> bool:
    left_leaves, left_tree = jax.tree_util.tree_flatten(left)
    right_leaves, right_tree = jax.tree_util.tree_flatten(right)
    return left_tree == right_tree and all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(left_leaves, right_leaves, strict=True)
    )


def _gradient_l2_norm(gradients: Any):
    return jnp.sqrt(
        sum(
            jnp.sum(jnp.asarray(value) * jnp.asarray(value))
            for value in jax.tree_util.tree_leaves(gradients)
        )
    )


def _tree_nonfinite_count(value: Any):
    leaves = jax.tree_util.tree_leaves(value)
    if not leaves:
        return jnp.asarray(0, dtype=jnp.int32)
    return sum(
        jnp.count_nonzero(~jnp.isfinite(jnp.asarray(leaf)))
        for leaf in leaves
    )


def _empty_rollout_components(anchor_loss) -> RolloutStabilityComponents:
    zero_float = jnp.zeros_like(anchor_loss)
    zero_count = jnp.asarray(0, dtype=jnp.int32)
    return RolloutStabilityComponents(
        L_fast=anchor_loss,
        L_next=zero_float,
        L_rollout=zero_float,
        L_bias=zero_float,
        L_science=zero_float,
        defined_status_mismatches=zero_count,
        discrete_state_mismatches=zero_count,
        nonfinite_defined_values=zero_count,
        negative_source_nonnegative_carbon_stocks=zero_count,
    )


def _hard_constraint_count(components: RolloutStabilityComponents):
    return sum(
        jnp.asarray(getattr(components, name), dtype=jnp.int32)
        for name in HARD_CONSTRAINT_FIELDS
    )


def _select_tree(predicate, accepted, rejected):
    return jax.tree_util.tree_map(
        lambda yes, no: jnp.where(predicate, yes, no),
        accepted,
        rejected,
    )


def make_control_update_step(
    *,
    fast_day_weights,
    undefined_loss_weight: float,
    model_apply,
):
    """Compile the one-step continuation arm used as the matched control."""

    def update(parameters, optimizer, anchor_batch, learning_rate):
        def objective(value):
            return canonical_fast_day_anchor_loss(
                value,
                anchor_batch,
                fast_day_weights=fast_day_weights,
                undefined_loss_weight=undefined_loss_weight,
                model_apply=model_apply,
            )

        loss, gradients = jax.value_and_grad(objective)(parameters)
        proposed_parameters, proposed_optimizer = _adam_update(
            parameters,
            gradients,
            optimizer,
            learning_rate=learning_rate,
        )
        nonfinite_gradient_values = _tree_nonfinite_count(gradients)
        applied = jnp.isfinite(loss) & (nonfinite_gradient_values == 0)
        next_parameters = _select_tree(applied, proposed_parameters, parameters)
        next_optimizer = _select_tree(applied, proposed_optimizer, optimizer)
        return RolloutStabilityUpdateResult(
            next_parameters,
            next_optimizer,
            loss,
            _empty_rollout_components(loss),
            _gradient_l2_norm(gradients),
            nonfinite_gradient_values,
            applied,
        )

    return jax.jit(update)


def make_candidate_update_step(
    *,
    coefficients: Mapping[str, Any],
    fast_day_weights,
    undefined_loss_weight: float,
    rematerialize: bool,
    model_apply,
    **objective_kwargs,
):
    """Compile one mixed-horizon candidate update with hard fail-closed gates."""

    def update(
        parameters,
        optimizer,
        anchor_batch,
        initial_states,
        initial_discrete_states,
        sequences,
        teacher_next_discrete_states,
        learning_rate,
    ):
        def objective(value):
            result = rollout_stability_candidate_loss(
                value,
                anchor_batch,
                initial_states,
                initial_discrete_states,
                sequences,
                coefficients=coefficients,
                fast_day_weights=fast_day_weights,
                undefined_loss_weight=undefined_loss_weight,
                teacher_next_discrete_states=teacher_next_discrete_states,
                rematerialize=rematerialize,
                model_apply=model_apply,
                **objective_kwargs,
            )
            return result.loss, result.components

        (loss, components), gradients = jax.value_and_grad(
            objective,
            has_aux=True,
        )(parameters)
        proposed_parameters, proposed_optimizer = _adam_update(
            parameters,
            gradients,
            optimizer,
            learning_rate=learning_rate,
        )
        nonfinite_gradient_values = _tree_nonfinite_count(gradients)
        applied = (
            (jnp.asarray(_hard_constraint_count(components)) == 0)
            & jnp.isfinite(loss)
            & (nonfinite_gradient_values == 0)
        )
        next_parameters = _select_tree(applied, proposed_parameters, parameters)
        next_optimizer = _select_tree(applied, proposed_optimizer, optimizer)
        return RolloutStabilityUpdateResult(
            next_parameters,
            next_optimizer,
            loss,
            components,
            _gradient_l2_norm(gradients),
            nonfinite_gradient_values,
            applied,
        )

    return jax.jit(update)


def make_coefficient_calibration_step(
    *,
    fast_day_weights,
    undefined_loss_weight: float,
    rematerialize: bool,
    model_apply,
    **objective_kwargs,
):
    """Compile paired component gradient norms for one fixed horizon."""

    def calibrate(
        parameters,
        anchor_batch,
        initial_states,
        initial_discrete_states,
        sequences,
        teacher_next_discrete_states,
    ):
        def component_vector(value):
            anchor = canonical_fast_day_anchor_loss(
                value,
                anchor_batch,
                fast_day_weights=fast_day_weights,
                undefined_loss_weight=undefined_loss_weight,
                model_apply=model_apply,
            )
            components = rollout_stability_components(
                value,
                initial_states,
                initial_discrete_states,
                sequences,
                fast_day_weights=fast_day_weights,
                undefined_loss_weight=undefined_loss_weight,
                teacher_next_discrete_states=teacher_next_discrete_states,
                rematerialize=rematerialize,
                model_apply=model_apply,
                **objective_kwargs,
            )
            return jnp.stack(
                (
                    anchor,
                    components.L_next,
                    components.L_rollout,
                    components.L_bias,
                    components.L_science,
                )
            ), components

        values, pullback, components = jax.vjp(
            component_vector,
            parameters,
            has_aux=True,
        )
        basis = jnp.eye(5, dtype=values.dtype)
        gradients = jax.vmap(lambda cotangent: pullback(cotangent)[0])(basis)
        squared_norms = sum(
            jnp.sum(
                jnp.asarray(leaf) * jnp.asarray(leaf),
                axis=tuple(range(1, jnp.asarray(leaf).ndim)),
            )
            for leaf in jax.tree_util.tree_leaves(gradients)
        )
        nonfinite_counts = sum(
            jnp.count_nonzero(
                ~jnp.isfinite(jnp.asarray(leaf)),
                axis=tuple(range(1, jnp.asarray(leaf).ndim)),
            )
            for leaf in jax.tree_util.tree_leaves(gradients)
        )
        return values, jnp.sqrt(squared_norms), nonfinite_counts, components

    return jax.jit(calibrate)


def broadcast_retained_tail_day_inputs(
    compiled_forcing,
    static: Mapping[str, Any],
    *,
    batch_size: int,
    horizon: int,
) -> CanonicalRetainedTailDayInputs:
    """Broadcast one landpoint's dynamic arrays over a rollout batch."""

    if batch_size < 1 or horizon < 1:
        raise ValueError("retained-tail batch and horizon must be positive")
    forcing_leaves = jax.tree_util.tree_leaves(compiled_forcing)
    if not forcing_leaves or any(
        np.shape(value)[:2] != (batch_size, horizon)
        for value in forcing_leaves
    ):
        raise ValueError("compiled forcing must have batch and horizon axes")
    single = canonical_retained_tail_day_inputs(None, static)

    def broadcast(value):
        array = jnp.asarray(value)
        return jnp.broadcast_to(array, (batch_size, horizon, *array.shape))

    dynamic = jax.tree_util.tree_map(broadcast, single)
    return dynamic._replace(compiled_forcing=compiled_forcing)


def _trace_static_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _trace_static_jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if hasattr(value, "_asdict"):
        return {
            "__type__": type(value).__qualname__,
            "fields": _trace_static_jsonable(value._asdict()),
        }
    if isinstance(value, (tuple, list)):
        return [_trace_static_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if callable(value):
        return {
            "__callable__": (
                f"{getattr(value, '__module__', '')}."
                f"{getattr(value, '__qualname__', type(value).__qualname__)}"
            )
        }
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    array = np.asarray(value)
    if array.dtype.kind == "O":
        raise TypeError(f"unsupported trace-static value {type(value)!r}")
    return {
        "__array__": True,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def retained_tail_trace_signature(static: Mapping[str, Any]) -> str:
    """Hash only controls intentionally bound into a retained-tail executable."""

    metadata = static["metadata"]
    payload = {
        "mineral_imin": int(static["mineral_imin"]),
        "mineral_imax": int(static["mineral_imax"]),
        "daily_carbon_dispatch": _trace_static_jsonable(
            static["daily_carbon_dispatch"]
        ),
        "metadata": _trace_static_jsonable(
            {
                "nstm": metadata.nstm,
                "is_tree": metadata.is_tree,
                "is_peat": metadata.is_peat,
                "npts": metadata.npts,
            }
        ),
        "season_provenance": _trace_static_jsonable(static["season"].provenance),
    }
    return _canonical_sha256(payload)


def completed_horizon_counts(
    schedule: Mapping[str, Any],
    *,
    next_update: int,
) -> Mapping[str, int]:
    entries = schedule["entries"]
    if not 0 <= next_update <= len(entries):
        raise ValueError("checkpoint next_update is outside the sampling schedule")
    counts = {str(key): 0 for key in schedule["horizon_counts"]}
    for entry in entries[:next_update]:
        key = str(int(entry["horizon"]))
        if key not in counts:
            raise ValueError(f"sampling schedule contains undeclared horizon {key}")
        counts[key] += 1
    return counts


def build_arm_checkpoint(
    *,
    arm_id: str,
    identity: Mapping[str, Any],
    parameters: Any,
    optimizer: Any,
    next_update: int,
    schedule: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    if arm_id not in ARM_IDS:
        raise ValueError(f"unknown rollout-stability arm {arm_id!r}")
    return {
        "schema_version": ARM_CHECKPOINT_SCHEMA_VERSION,
        "arm_id": arm_id,
        "identity": dict(identity),
        "parameters": jax.device_get(parameters),
        "optimizer": jax.device_get(optimizer),
        "next_update": int(next_update),
        "schedule_canonical_sha256": str(schedule["canonical_sha256"]),
        "completed_horizon_counts": completed_horizon_counts(
            schedule,
            next_update=next_update,
        ),
        "history": list(history),
    }


def verify_arm_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    arm_id: str,
    identity: Mapping[str, Any],
    schedule: Mapping[str, Any],
) -> None:
    if checkpoint.get("schema_version") != ARM_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported rollout-stability checkpoint schema")
    if checkpoint.get("arm_id") != arm_id:
        raise ValueError("rollout-stability checkpoint arm drift")
    if checkpoint.get("identity") != dict(identity):
        raise ValueError("rollout-stability checkpoint identity drift")
    if checkpoint.get("schedule_canonical_sha256") != schedule["canonical_sha256"]:
        raise ValueError("rollout-stability checkpoint schedule drift")
    next_update = int(checkpoint.get("next_update", -1))
    expected_counts = completed_horizon_counts(
        schedule,
        next_update=next_update,
    )
    if checkpoint.get("completed_horizon_counts") != expected_counts:
        raise ValueError("rollout-stability checkpoint horizon-count drift")
    if "parameters" not in checkpoint or "optimizer" not in checkpoint:
        raise ValueError("rollout-stability checkpoint state is incomplete")


def _load_rollout_resources(
    *,
    dataset_path: Path,
    statistics_path: Path,
    plan_path: Path | None = None,
) -> RolloutStabilityResources:
    raw_manifest = json.loads(dataset_path.read_text(encoding="utf-8"))
    resolved_plan_path = (
        Path(raw_manifest["plan"]).resolve()
        if plan_path is None
        else plan_path.resolve()
    )
    plan, entries = _load_plan(
        resolved_plan_path,
        str(raw_manifest["plan_sha256"]),
    )
    index = load_dataset_index(
        dataset_path,
        verify_hashes=False,
        verify_files=False,
    )
    metadata = raw_manifest["markov_contract"]
    contract = daily_markov_contract_from_metadata(metadata)
    representation = fast_day_target_representation_from_contract(metadata)
    statistics = load_training_statistics(statistics_path, index=index)
    config = _model_config(contract, representation)
    model_definition = build_daily_model_definition(
        AXIS_PROCESS_COUPLED_V1,
        config,
        metadata,
    )
    process_weighting = state_process_weighting_from_contract(metadata)
    state_delta_scale, scale_audit = stabilized_state_delta_scale(
        statistics,
        floor_ratio=0.01,
    )
    return RolloutStabilityResources(
        raw_manifest=raw_manifest,
        plan=plan,
        plan_entries=entries,
        index=index,
        contract=contract,
        representation=representation,
        statistics=statistics,
        model_definition=model_definition,
        fast_day_weights=jnp.asarray(loss_weights_from_contract(metadata)),
        process_weighting=process_weighting,
        state_delta_scale=jnp.asarray(state_delta_scale),
        state_delta_scale_audit=scale_audit,
        science_layout=science_objective_layout_from_contract(metadata),
        config_path=Path(plan["teacher_config"]).resolve(),
    )


def _collate_reference_windows(
    reference: MarkovShardRef,
    *,
    starts: Sequence[int],
    horizon: int,
) -> Mapping[str, Any]:
    shard = load_markov_shard(reference.path)
    return _collate_shard_windows(shard, starts=starts, horizon=horizon)


def _collate_shard_windows(
    shard,
    *,
    starts: Sequence[int],
    horizon: int,
) -> Mapping[str, Any]:
    choices = shard.days - horizon + 1
    indices = np.asarray(starts, dtype=np.int64)
    if indices.ndim != 1 or indices.size < 1:
        raise ValueError("rollout-stability starts must be a nonempty vector")
    if np.any(indices < 0) or np.any(indices >= choices):
        raise ValueError("rollout-stability start falls outside the shard")
    return collate_windows(
        tuple(shard.window(int(start), horizon) for start in indices)
    )


def _anchor_training_batch(
    batch: Mapping[str, Any],
    resources: RolloutStabilityResources,
) -> CanonicalTrainingBatch:
    if np.shape(batch["day_index"])[1] != 1:
        raise ValueError("matched anchor batch must have horizon one")
    return prepare_canonical_batch(
        {
            "state": batch["initial_state"],
            "forcing_native": batch["forcing_native"][:, 0],
            "parameters": batch["parameters"][:, 0],
            "landpoint_static": batch["landpoint_static"][:, 0],
            "annual_conditions": batch["annual_conditions"][:, 0],
            "year": batch["year"][:, 0],
            "day_index": batch["day_index"][:, 0],
            "fast_day_target": batch["teacher_fast_day_target"][:, 0],
        },
        resources.statistics,
        resources.representation,
    )


def prepare_rollout_update(
    *,
    reference: MarkovShardRef,
    update: int,
    horizon: int,
    anchor_batch_size: int,
    rollout_batch_size: int,
    seed: int,
    resources: RolloutStabilityResources,
    runtime: LandpointRuntime | None = None,
    runtime_cache: dict[str, LandpointRuntime] | None = None,
) -> PreparedRolloutUpdate:
    """Materialize one deterministic update without using validation/test data."""

    if reference.spatial_split != "train" or reference.temporal_split != "train":
        raise ValueError("rollout-stability training attempted to use non-train data")
    shard = load_markov_shard(reference.path)
    anchor_starts, rollout_starts = sample_start_indices(
        shard_days=shard.days,
        update=update,
        horizon=horizon,
        anchor_batch_size=anchor_batch_size,
        rollout_batch_size=rollout_batch_size,
        seed=seed,
    )
    anchor_raw = _collate_shard_windows(
        shard,
        starts=anchor_starts,
        horizon=1,
    )
    rollout_raw = _collate_shard_windows(
        shard,
        starts=rollout_starts,
        horizon=horizon,
    )
    compiled_forcing = None
    if runtime is None:
        if runtime_cache is not None:
            runtime = runtime_cache.get(reference.landpoint_id)
        if runtime is None:
            runtime, compiled_forcing = _make_runtime_and_compiled_forcing(
                landpoint_id=reference.landpoint_id,
                entry=resources.plan_entries[reference.landpoint_id],
                config_path=resources.config_path,
                contract=resources.contract,
                batch=rollout_raw,
            )
            if runtime_cache is not None:
                runtime_cache[reference.landpoint_id] = runtime
    if compiled_forcing is None:
        compiled_forcing = _compiled_forcing_batch(
            rollout_raw,
            resources.contract,
            runtime.context,
        )
    retained_inputs = broadcast_retained_tail_day_inputs(
        compiled_forcing,
        runtime.static,
        batch_size=rollout_batch_size,
        horizon=horizon,
    )
    return PreparedRolloutUpdate(
        anchor_batch=_anchor_training_batch(anchor_raw, resources),
        initial_states=jnp.asarray(rollout_raw["initial_state"]),
        initial_discrete_states=jax.tree_util.tree_map(
            jnp.asarray,
            rollout_raw["initial_discrete_state"],
        ),
        sequence=_sequence(rollout_raw, retained_inputs),
        teacher_next_discrete_states=jax.tree_util.tree_map(
            jnp.asarray,
            rollout_raw["teacher_next_discrete_state"],
        ),
        runtime=runtime,
        trace_signature=retained_tail_trace_signature(runtime.static),
        anchor_starts=np.asarray(anchor_starts, dtype=np.int64),
        rollout_starts=np.asarray(rollout_starts, dtype=np.int64),
    )


def prepare_anchor_update(
    *,
    reference: MarkovShardRef,
    update: int,
    horizon: int,
    anchor_batch_size: int,
    rollout_batch_size: int,
    seed: int,
    resources: RolloutStabilityResources,
) -> CanonicalTrainingBatch:
    """Materialize only the matched control anchor for one scheduled update."""

    if reference.spatial_split != "train" or reference.temporal_split != "train":
        raise ValueError("rollout-stability control attempted to use non-train data")
    shard = load_markov_shard(reference.path)
    anchor_starts, _ = sample_start_indices(
        shard_days=shard.days,
        update=update,
        horizon=horizon,
        anchor_batch_size=anchor_batch_size,
        rollout_batch_size=rollout_batch_size,
        seed=seed,
    )
    anchor_raw = _collate_shard_windows(
        shard,
        starts=anchor_starts,
        horizon=1,
    )
    return _anchor_training_batch(anchor_raw, resources)


def _objective_kwargs(
    resources: RolloutStabilityResources,
    transition,
) -> Mapping[str, Any]:
    return {
        "statistics": resources.statistics,
        "representation": resources.representation,
        "retained_tail_transition": transition,
        "process_weighting": resources.process_weighting,
        "state_delta_scale": resources.state_delta_scale,
        "science_layout": resources.science_layout,
    }


def _bind_dynamic_transition(
    resources: RolloutStabilityResources,
    runtime: LandpointRuntime,
):
    return bind_canonical_retained_tail_dynamic_transition(
        config_path=resources.config_path,
        context=runtime.context,
        contract=resources.contract,
        trace_static=runtime.static,
        runtime_year=1962,
    )


def verify_parent_architecture_assets(
    *,
    protocol: RolloutStabilityProtocol,
    report_path: str | Path,
    best_checkpoint_path: str | Path,
    optimizer_checkpoint_path: str | Path,
) -> Mapping[str, Any]:
    """Require the accepted axis/process parameters and matching Adam state."""

    report_path = Path(report_path).resolve()
    best_checkpoint_path = Path(best_checkpoint_path).resolve()
    optimizer_checkpoint_path = Path(optimizer_checkpoint_path).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != PARENT_REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported parent architecture report schema")
    if report.get("status") != "completed":
        raise ValueError("parent architecture report is not completed")
    parent = protocol.raw["parent_architecture_screen"]
    experiment = report.get("experiment", {})
    if experiment.get("sha256") != parent["experiment_canonical_sha256"]:
        raise ValueError("parent architecture report experiment hash drift")
    classification = report.get("classification", {})
    if classification.get("status") != parent["required_classification_status"]:
        raise ValueError("parent architecture classification did not pass")
    if classification.get("decision") != REQUIRED_PARENT_DECISION:
        raise ValueError("parent architecture decision does not admit Experiment B")

    arm = report.get("arms", {}).get("axis_process", {})
    if arm.get("model_architecture", {}).get("id") != AXIS_PROCESS_COUPLED_V1:
        raise ValueError("parent selected arm is not axis_process_coupled_v1")
    if arm.get("test_split_evaluated") is not False:
        raise ValueError("parent architecture report evaluated the sealed test split")
    best_sha256 = _sha256_file(best_checkpoint_path)
    if best_sha256 != arm.get("best_checkpoint_sha256"):
        raise ValueError("parent best checkpoint hash drift")

    best = _load_pickle(best_checkpoint_path)
    optimizer = _load_pickle(optimizer_checkpoint_path)
    for name, payload in (("best", best), ("optimizer", optimizer)):
        if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(f"unsupported parent {name} checkpoint schema")
        if checkpoint_architecture_id(payload.get("identity", {})) != (
            AXIS_PROCESS_COUPLED_V1
        ):
            raise ValueError(f"parent {name} checkpoint architecture drift")
    if best.get("identity") != optimizer.get("identity"):
        raise ValueError("parent best and optimizer checkpoint identities differ")
    if not _trees_equal(best.get("parameters"), optimizer.get("parameters")):
        raise ValueError("parent best parameters differ from optimizer checkpoint")
    if "optimizer" not in optimizer:
        raise ValueError("parent optimizer checkpoint has no optimizer state")
    best_epoch = int(arm["best_epoch"])
    if int(best.get("epoch", -1)) != best_epoch:
        raise ValueError("parent best checkpoint epoch drift")
    if int(optimizer.get("completed_epochs", -1)) != best_epoch:
        raise ValueError("parent optimizer checkpoint is not the selected best epoch")
    return {
        "report": {
            "path": str(report_path),
            "sha256": _sha256_file(report_path),
            "experiment_sha256": experiment["sha256"],
            "decision": classification["decision"],
        },
        "best_checkpoint": {
            "path": str(best_checkpoint_path),
            "sha256": best_sha256,
            "epoch": best_epoch,
        },
        "optimizer_checkpoint": {
            "path": str(optimizer_checkpoint_path),
            "sha256": _sha256_file(optimizer_checkpoint_path),
            "completed_epochs": int(optimizer["completed_epochs"]),
        },
        "model_architecture": arm["model_architecture"],
    }


def _balanced_reference_indices(
    references: Sequence[MarkovShardRef],
    *,
    updates: int,
    seed: int,
) -> list[int]:
    """Round-robin landpoints while visiting distinct years before reuse."""

    by_landpoint: dict[str, list[int]] = defaultdict(list)
    for index, reference in enumerate(references):
        by_landpoint[reference.landpoint_id].append(index)
    if not by_landpoint:
        raise ValueError("rollout schedule has no train/train references")
    landpoints = sorted(by_landpoint)
    rng = np.random.default_rng(seed)
    year_orders = {
        landpoint: list(rng.permutation(indices))
        for landpoint, indices in by_landpoint.items()
    }
    result = []
    cycle = 0
    while len(result) < updates:
        for offset in rng.permutation(len(landpoints)):
            landpoint = landpoints[int(offset)]
            candidates = year_orders[landpoint]
            result.append(int(candidates[cycle % len(candidates)]))
            if len(result) == updates:
                break
        cycle += 1
    return result


def _exact_horizon_schedule(
    horizon_specs: Sequence[Mapping[str, Any]],
    *,
    updates: int,
    seed: int,
) -> list[int]:
    probabilities = np.asarray(
        [float(spec["probability"]) for spec in horizon_specs],
        dtype=np.float64,
    )
    raw = probabilities * updates
    counts = np.floor(raw).astype(np.int64)
    remainder = updates - int(np.sum(counts))
    order = np.argsort(-(raw - counts), kind="stable")
    counts[order[:remainder]] += 1
    values = np.concatenate(
        [
            np.full((int(count),), int(spec["days"]), dtype=np.int16)
            for count, spec in zip(counts, horizon_specs, strict=True)
        ]
    )
    rng = np.random.default_rng(seed)
    rng.shuffle(values)
    return [int(value) for value in values]


def build_sampling_schedule(
    index: MarkovDatasetIndex,
    protocol: RolloutStabilityProtocol,
) -> Mapping[str, Any]:
    """Freeze every update's landpoint/year and mixed rollout horizon."""

    references = sorted(
        index.select(spatial_split="train", temporal_split="train"),
        key=lambda item: (item.landpoint_id, item.year, str(item.path)),
    )
    optimization = protocol.raw["optimization"]
    updates = int(optimization["screening_updates"])
    seed = int(optimization["screening_seed"])
    selected = _balanced_reference_indices(references, updates=updates, seed=seed)
    horizons = _exact_horizon_schedule(
        protocol.raw["mixed_horizon_sampling"]["horizons"],
        updates=updates,
        seed=seed,
    )
    reference_inventory = [
        {
            "landpoint_id": reference.landpoint_id,
            "year": int(reference.year),
            "spatial_split": reference.spatial_split,
            "temporal_split": reference.temporal_split,
            "path": str(reference.path),
            "sha256": reference.sha256,
        }
        for reference in references
    ]
    entries = [
        {
            "update": update,
            "reference_index": reference_index,
            "horizon": horizons[update],
        }
        for update, reference_index in enumerate(selected)
    ]
    horizon_counts = {
        str(spec["days"]): horizons.count(int(spec["days"]))
        for spec in protocol.raw["mixed_horizon_sampling"]["horizons"]
    }
    landpoint_counts: dict[str, int] = defaultdict(int)
    for reference_index in selected:
        landpoint_counts[references[reference_index].landpoint_id] += 1
    payload = {
        "schema_version": SCHEDULE_SCHEMA_VERSION,
        "protocol_sha256": protocol.sha256,
        "dataset_id": index.dataset_id,
        "dataset_contract_sha256": index.contract_sha256,
        "seed": seed,
        "updates": updates,
        "selection_policy": "round_robin_landpoint_distinct_year_then_seeded_start_v1",
        "start_sampling": {
            "anchor": "seed_sequence(seed,update,0)_without_replacement",
            "rollout": "seed_sequence(seed,update,1)_without_replacement",
        },
        "reference_inventory": reference_inventory,
        "entries": entries,
        "horizon_counts": horizon_counts,
        "landpoint_update_count_min": min(landpoint_counts.values()),
        "landpoint_update_count_max": max(landpoint_counts.values()),
    }
    return payload | {"canonical_sha256": _canonical_sha256(payload)}


def sample_start_indices(
    *,
    shard_days: int,
    update: int,
    horizon: int,
    anchor_batch_size: int,
    rollout_batch_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct exact anchor and rollout starts from a checkpointed update."""

    anchor_choices = int(shard_days)
    rollout_choices = int(shard_days) - int(horizon) + 1
    if anchor_choices < anchor_batch_size or rollout_choices < rollout_batch_size:
        raise ValueError("scheduled shard cannot provide unique batch start days")
    anchor_rng = np.random.default_rng(np.random.SeedSequence([seed, update, 0]))
    rollout_rng = np.random.default_rng(np.random.SeedSequence([seed, update, 1]))
    anchor = anchor_rng.choice(
        anchor_choices,
        size=anchor_batch_size,
        replace=False,
    )
    rollout = rollout_rng.choice(
        rollout_choices,
        size=rollout_batch_size,
        replace=False,
    )
    return anchor.astype(np.int32), rollout.astype(np.int32)


def calibrate_gradient_coefficients(
    records: Sequence[Mapping[str, Any]],
    protocol: RolloutStabilityProtocol,
) -> Mapping[str, Any]:
    """Resolve fixed coefficients from paired train-only gradient norms."""

    calibration = protocol.raw["loss"]["coefficient_calibration"]
    expected_per_horizon = int(calibration["calibration_batches_per_horizon"])
    horizons = [
        int(item["days"])
        for item in protocol.raw["mixed_horizon_sampling"]["horizons"]
    ]
    observed_horizons = [int(record["horizon"]) for record in records]
    for horizon in horizons:
        if observed_horizons.count(horizon) != expected_per_horizon:
            raise ValueError(f"gradient calibration horizon {horizon} count drift")
    components = ("L_next", "L_rollout", "L_bias", "L_science")
    target_ratios = calibration["target_gradient_norm_ratios"]
    low, high = (float(value) for value in calibration["coefficient_clip"])
    resolved = {}
    audits = {}
    for component in components:
        ratios = []
        measured = 0
        for record in records:
            anchor = float(record["gradient_norms"]["L_fast"])
            value = float(record["gradient_norms"][component])
            if not np.isfinite(anchor) or anchor <= 0.0:
                raise ValueError("gradient calibration has invalid L_fast norm")
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"gradient calibration has invalid {component} norm")
            measured += 1
            if value > 0.0:
                ratios.append(value / anchor)
        if not ratios:
            raise ValueError(f"gradient calibration has no positive {component} norms")
        median_ratio = float(np.median(np.asarray(ratios, dtype=np.float64)))
        coefficient = float(
            np.clip(float(target_ratios[component]) / median_ratio, low, high)
        )
        resolved[component] = coefficient
        audits[component] = {
            "records_measured": measured,
            "positive_ratio_records": len(ratios),
            "median_unweighted_to_anchor_gradient_norm_ratio": median_ratio,
            "target_weighted_to_anchor_gradient_norm_ratio": float(
                target_ratios[component]
            ),
            "resolved_coefficient": coefficient,
        }
    payload = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "status": "passed",
        "protocol_sha256": protocol.sha256,
        "method": calibration["method"],
        "seed": int(calibration["fixed_calibration_seed"]),
        "train_only": True,
        "record_count": len(records),
        "records_sha256": _canonical_sha256(list(records)),
        "resolved_coefficients": resolved,
        "component_audit": audits,
    }
    return payload | {"canonical_sha256": _canonical_sha256(payload)}


def _load_verified_preflight(
    preflight_path: str | Path,
    protocol: RolloutStabilityProtocol,
) -> Mapping[str, Any]:
    preflight_path = Path(preflight_path).resolve()
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
        raise ValueError("unsupported rollout-stability preflight schema")
    if preflight.get("status") != "awaiting_coefficient_calibration":
        raise ValueError("rollout-stability preflight has an invalid status")
    if preflight.get("protocol", {}).get("sha256") != protocol.sha256:
        raise ValueError("rollout-stability preflight protocol drift")
    if preflight.get("sealed_test_used") is not False:
        raise ValueError("rollout-stability preflight used the sealed test split")
    schedule_info = preflight["artifacts"]["sampling_schedule"]
    schedule_path = Path(schedule_info["path"]).resolve()
    if _sha256_file(schedule_path) != schedule_info["sha256"]:
        raise ValueError("rollout-stability preflight schedule file drift")
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    if schedule.get("canonical_sha256") != schedule_info["canonical_sha256"]:
        raise ValueError("rollout-stability preflight schedule identity drift")
    return preflight


def _horizon_spec(
    protocol: RolloutStabilityProtocol,
    horizon: int,
) -> Mapping[str, Any]:
    matches = [
        item
        for item in protocol.raw["mixed_horizon_sampling"]["horizons"]
        if int(item["days"]) == int(horizon)
    ]
    if len(matches) != 1:
        raise ValueError(f"rollout-stability horizon {horizon} is not declared")
    return matches[0]


def _reference_from_inventory(
    inventory: Sequence[Mapping[str, Any]],
    index: int,
) -> MarkovShardRef:
    item = inventory[index]
    return MarkovShardRef(
        landpoint_id=str(item["landpoint_id"]),
        year=int(item["year"]),
        spatial_split=str(item["spatial_split"]),
        temporal_split=str(item["temporal_split"]),
        path=Path(item["path"]).resolve(),
        sha256=str(item["sha256"]),
        contract_sha256="",
    )


def _parent_parameters(
    preflight: Mapping[str, Any],
    resources: RolloutStabilityResources,
):
    best = _load_pickle(
        Path(preflight["parent_architecture"]["best_checkpoint"]["path"])
    )
    verify_checkpoint_architecture(best["identity"], resources.model_definition)
    return jax.tree_util.tree_map(jnp.asarray, best["parameters"])


def _calibration_schedule(
    references: Sequence[MarkovShardRef],
    protocol: RolloutStabilityProtocol,
) -> Sequence[tuple[int, MarkovShardRef, int]]:
    calibration = protocol.raw["loss"]["coefficient_calibration"]
    count = int(calibration["calibration_batches_per_horizon"])
    seed = int(calibration["fixed_calibration_seed"])
    horizons = [
        int(item["days"])
        for item in protocol.raw["mixed_horizon_sampling"]["horizons"]
    ]
    selected = _balanced_reference_indices(
        references,
        updates=count * len(horizons),
        seed=seed,
    )
    return tuple(
        (
            ordinal,
            references[selected[ordinal]],
            horizons[ordinal // count],
        )
        for ordinal in range(len(selected))
    )


def run_coefficient_calibration(
    *,
    preflight_path: str | Path,
    protocol_path: str | Path,
    dataset_path: str | Path,
    statistics_path: str | Path,
    output_root: str | Path,
    plan_path: str | Path | None = None,
) -> Path:
    """Measure and freeze train-only paired gradient-norm coefficients."""

    protocol = load_rollout_stability_protocol(protocol_path)
    preflight = _load_verified_preflight(preflight_path, protocol)
    dataset_path = Path(dataset_path).resolve()
    statistics_path = Path(statistics_path).resolve()
    for name, path in (
        ("dataset_manifest", dataset_path),
        ("training_statistics", statistics_path),
    ):
        if _sha256_file(path) != preflight["artifacts"][name]["sha256"]:
            raise ValueError(f"rollout-stability calibration {name} drift")
    resources = _load_rollout_resources(
        dataset_path=dataset_path,
        statistics_path=statistics_path,
        plan_path=None if plan_path is None else Path(plan_path),
    )
    parameters = _parent_parameters(preflight, resources)
    references = sorted(
        resources.index.select(spatial_split="train", temporal_split="train"),
        key=lambda item: (item.landpoint_id, item.year, str(item.path)),
    )
    calibration_seed = int(
        protocol.raw["loss"]["coefficient_calibration"]["fixed_calibration_seed"]
    )
    anchor_batch_size = int(protocol.raw["optimization"]["anchor_batch_size"])
    compiled: dict[tuple[int, bool, str], Any] = {}
    runtimes: dict[str, LandpointRuntime] = {}
    records = []
    calibration_schedule = _calibration_schedule(references, protocol)
    for ordinal, reference, horizon in calibration_schedule:
        spec = _horizon_spec(protocol, horizon)
        prepared = prepare_rollout_update(
            reference=reference,
            update=ordinal,
            horizon=horizon,
            anchor_batch_size=anchor_batch_size,
            rollout_batch_size=int(spec["batch_size"]),
            seed=calibration_seed,
            resources=resources,
            runtime_cache=runtimes,
        )
        rematerialize = bool(spec["rematerialize"])
        key = (horizon, rematerialize, prepared.trace_signature)
        if key not in compiled:
            transition = _bind_dynamic_transition(resources, prepared.runtime)
            compiled[key] = make_coefficient_calibration_step(
                fast_day_weights=resources.fast_day_weights,
                undefined_loss_weight=float(
                    protocol.raw["loss"]["undefined_flip_binary_weight"]
                ),
                rematerialize=rematerialize,
                model_apply=resources.model_definition.apply,
                **_objective_kwargs(resources, transition),
            )
        values, norms, nonfinite_counts, components = compiled[key](
            parameters,
            prepared.anchor_batch,
            prepared.initial_states,
            prepared.initial_discrete_states,
            prepared.sequence,
            prepared.teacher_next_discrete_states,
        )
        values, norms, nonfinite_counts, components = jax.device_get(
            (values, norms, nonfinite_counts, components)
        )
        if int(_hard_constraint_count(components)) != 0:
            raise ValueError(
                "rollout-stability coefficient calibration violated a hard constraint"
            )
        if (
            not np.all(np.isfinite(values))
            or not np.all(np.isfinite(norms))
            or np.any(np.asarray(nonfinite_counts) != 0)
        ):
            raise ValueError(
                "rollout-stability coefficient calibration produced nonfinite "
                f"component gradients: {np.asarray(nonfinite_counts).tolist()}"
            )
        names = ("L_fast", "L_next", "L_rollout", "L_bias", "L_science")
        records.append(
            {
                "horizon": horizon,
                "batch": ordinal
                % int(
                    protocol.raw["loss"]["coefficient_calibration"][
                        "calibration_batches_per_horizon"
                    ]
                ),
                "landpoint_id": reference.landpoint_id,
                "year": int(reference.year),
                "gradient_norms": {
                    name: float(norms[index])
                    for index, name in enumerate(names)
                },
                "nonfinite_gradient_values": {
                    name: int(nonfinite_counts[index])
                    for index, name in enumerate(names)
                },
                "component_values": {
                    name: float(values[index])
                    for index, name in enumerate(names)
                },
            }
        )
        print(
            "rollout_calibration "
            f"batch={ordinal + 1}/{len(calibration_schedule)} "
            f"horizon={horizon} landpoint={reference.landpoint_id}",
            flush=True,
        )

    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    records_path = output_root / "coefficient_calibration_records.json"
    _atomic_json(
        records_path,
        {
            "schema_version": "canonical_rollout_stability_gradient_records_v1",
            "protocol_sha256": protocol.sha256,
            "train_only": True,
            "records": records,
        },
    )
    calibration = dict(calibrate_gradient_coefficients(records, protocol))
    calibration.pop("canonical_sha256")
    calibration["records"] = {
        "path": str(records_path),
        "sha256": _sha256_file(records_path),
    }
    calibration["parent_best_checkpoint_sha256"] = preflight[
        "parent_architecture"
    ]["best_checkpoint"]["sha256"]
    calibration["canonical_sha256"] = _canonical_sha256(calibration)
    calibration_path = output_root / "coefficient_calibration.json"
    _atomic_json(calibration_path, calibration)
    return calibration_path


def create_execution_manifest(
    *,
    preflight_path: str | Path,
    calibration_path: str | Path,
    protocol_path: str | Path,
    output_root: str | Path,
) -> Path:
    """Bind all resolved Experiment B inputs after coefficient calibration."""

    protocol = load_rollout_stability_protocol(protocol_path)
    preflight = _load_verified_preflight(preflight_path, protocol)
    calibration_path = Path(calibration_path).resolve()
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        raise ValueError("unsupported rollout-stability calibration schema")
    if calibration.get("status") != "passed":
        raise ValueError("rollout-stability coefficient calibration did not pass")
    if calibration.get("protocol_sha256") != protocol.sha256:
        raise ValueError("rollout-stability calibration protocol drift")
    if calibration.get("train_only") is not True:
        raise ValueError("rollout-stability calibration was not train-only")
    calibration_payload = dict(calibration)
    calibration_sha256 = calibration_payload.pop("canonical_sha256", None)
    if calibration_sha256 != _canonical_sha256(calibration_payload):
        raise ValueError("rollout-stability calibration canonical hash drift")
    records = calibration["records"]
    if _sha256_file(Path(records["path"])) != records["sha256"]:
        raise ValueError("rollout-stability calibration record file drift")
    parent = preflight["parent_architecture"]
    if calibration.get("parent_best_checkpoint_sha256") != parent[
        "best_checkpoint"
    ]["sha256"]:
        raise ValueError("rollout-stability calibration parent checkpoint drift")
    payload = {
        "schema_version": EXECUTION_MANIFEST_SCHEMA_VERSION,
        "status": "ready_for_matched_arm_training",
        "protocol": preflight["protocol"],
        "parent_architecture_ab_report": parent["report"],
        "selected_parent_checkpoint": parent["best_checkpoint"],
        "parent_optimizer_checkpoint": parent["optimizer_checkpoint"],
        "resolved_model_architecture_identity": parent["model_architecture"],
        "coefficient_calibration": {
            "path": str(calibration_path),
            "sha256": _sha256_file(calibration_path),
            "canonical_sha256": calibration["canonical_sha256"],
            "resolved_coefficients": calibration["resolved_coefficients"],
        },
        "artifacts": preflight["artifacts"],
        "optimizer": {
            "id": protocol.raw["optimization"]["optimizer"],
            "learning_rate": float(
                protocol.raw["optimization"]["learning_rate"]
            ),
            "update_budget": int(
                protocol.raw["optimization"]["screening_updates"]
            ),
            "seed": int(protocol.raw["optimization"]["screening_seed"]),
        },
        "arms": list(ARM_IDS),
        "same_anchor_batches": True,
        "sealed_test_used": False,
        "output_root": str(Path(output_root).resolve()),
    }
    payload["canonical_sha256"] = _canonical_sha256(payload)
    path = Path(output_root).resolve() / "rollout_stability_execution.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("existing rollout-stability execution manifest drift")
        return path
    return _atomic_json(path, payload)


def _load_verified_execution_manifest(
    execution_path: str | Path,
    protocol: RolloutStabilityProtocol,
) -> Mapping[str, Any]:
    execution_path = Path(execution_path).resolve()
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    if execution.get("schema_version") != EXECUTION_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported rollout-stability execution schema")
    if execution.get("status") != "ready_for_matched_arm_training":
        raise ValueError("rollout-stability execution manifest is not ready")
    if execution.get("protocol", {}).get("sha256") != protocol.sha256:
        raise ValueError("rollout-stability execution protocol drift")
    canonical_sha256 = execution.get("canonical_sha256")
    canonical_payload = dict(execution)
    canonical_payload.pop("canonical_sha256", None)
    if canonical_sha256 != _canonical_sha256(canonical_payload):
        raise ValueError("rollout-stability execution canonical hash drift")
    calibration = execution["coefficient_calibration"]
    calibration_path = Path(calibration["path"]).resolve()
    if _sha256_file(calibration_path) != calibration["sha256"]:
        raise ValueError("rollout-stability execution calibration file drift")
    schedule = execution["artifacts"]["sampling_schedule"]
    if _sha256_file(Path(schedule["path"])) != schedule["sha256"]:
        raise ValueError("rollout-stability execution schedule file drift")
    if execution.get("arms") != list(ARM_IDS):
        raise ValueError("rollout-stability execution arm inventory drift")
    if execution.get("same_anchor_batches") is not True:
        raise ValueError("rollout-stability execution lost matched anchors")
    if execution.get("sealed_test_used") is not False:
        raise ValueError("rollout-stability execution used the sealed test split")
    return execution


def _arm_identity(
    *,
    execution: Mapping[str, Any],
    arm_id: str,
    resources: RolloutStabilityResources,
) -> Mapping[str, Any]:
    return {
        "execution_canonical_sha256": execution["canonical_sha256"],
        "protocol_sha256": execution["protocol"]["sha256"],
        "arm_id": arm_id,
        "dataset_id": resources.index.dataset_id,
        "teacher_git_head": resources.index.teacher_git_head,
        "markov_contract_sha256": resources.index.contract_sha256,
        "model_architecture": resources.model_definition.identity(),
        "coefficient_calibration_sha256": execution["coefficient_calibration"][
            "sha256"
        ],
        "schedule_canonical_sha256": execution["artifacts"][
            "sampling_schedule"
        ]["canonical_sha256"],
    }


def _initial_arm_state(
    *,
    execution: Mapping[str, Any],
    resources: RolloutStabilityResources,
):
    best = _load_pickle(Path(execution["selected_parent_checkpoint"]["path"]))
    optimizer_checkpoint = _load_pickle(
        Path(execution["parent_optimizer_checkpoint"]["path"])
    )
    verify_checkpoint_architecture(best["identity"], resources.model_definition)
    verify_checkpoint_architecture(
        optimizer_checkpoint["identity"],
        resources.model_definition,
    )
    if not _trees_equal(best["parameters"], optimizer_checkpoint["parameters"]):
        raise ValueError("rollout-stability parent parameter checkpoints differ")
    return (
        jax.tree_util.tree_map(jnp.asarray, best["parameters"]),
        jax.tree_util.tree_map(jnp.asarray, optimizer_checkpoint["optimizer"]),
    )


def _component_record(components: RolloutStabilityComponents) -> Mapping[str, Any]:
    host = jax.device_get(components)
    return {
        name: (
            int(getattr(host, name))
            if name in HARD_CONSTRAINT_FIELDS
            else float(getattr(host, name))
        )
        for name in host._fields
    }


def run_rollout_stability_arm(
    *,
    arm_id: str,
    execution_path: str | Path,
    protocol_path: str | Path,
    dataset_path: str | Path,
    statistics_path: str | Path,
    output_root: str | Path,
    plan_path: str | Path | None = None,
) -> Path:
    """Run or exactly resume one arm of the matched Experiment B screen."""

    if arm_id not in ARM_IDS:
        raise ValueError(f"unknown rollout-stability arm {arm_id!r}")
    protocol = load_rollout_stability_protocol(protocol_path)
    execution = _load_verified_execution_manifest(execution_path, protocol)
    dataset_path = Path(dataset_path).resolve()
    statistics_path = Path(statistics_path).resolve()
    for name, path in (
        ("dataset_manifest", dataset_path),
        ("training_statistics", statistics_path),
    ):
        if _sha256_file(path) != execution["artifacts"][name]["sha256"]:
            raise ValueError(f"rollout-stability arm {name} drift")
    resources = _load_rollout_resources(
        dataset_path=dataset_path,
        statistics_path=statistics_path,
        plan_path=None if plan_path is None else Path(plan_path),
    )
    identity = _arm_identity(
        execution=execution,
        arm_id=arm_id,
        resources=resources,
    )
    schedule_path = Path(
        execution["artifacts"]["sampling_schedule"]["path"]
    ).resolve()
    schedule = json.loads(schedule_path.read_text(encoding="utf-8"))
    if schedule["canonical_sha256"] != identity["schedule_canonical_sha256"]:
        raise ValueError("rollout-stability arm schedule identity drift")

    arm_root = Path(output_root).resolve() / arm_id
    arm_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = arm_root / "checkpoint.pkl"
    if checkpoint_path.exists():
        checkpoint = _load_pickle(checkpoint_path)
        verify_arm_checkpoint(
            checkpoint,
            arm_id=arm_id,
            identity=identity,
            schedule=schedule,
        )
        parameters = jax.tree_util.tree_map(jnp.asarray, checkpoint["parameters"])
        optimizer = jax.tree_util.tree_map(jnp.asarray, checkpoint["optimizer"])
        next_update = int(checkpoint["next_update"])
        history = list(checkpoint["history"])
    else:
        parameters, optimizer = _initial_arm_state(
            execution=execution,
            resources=resources,
        )
        next_update = 0
        history = []

    optimization = protocol.raw["optimization"]
    budget = int(optimization["screening_updates"])
    if len(schedule["entries"]) != budget:
        raise ValueError("rollout-stability update budget and schedule differ")
    seed = int(optimization["screening_seed"])
    anchor_batch_size = int(optimization["anchor_batch_size"])
    learning_rate = jnp.asarray(
        optimization["learning_rate"],
        dtype=jnp.float32,
    )
    checkpoint_every = int(optimization["checkpoint_every_updates"])
    progress_every = int(optimization["progress_every_updates"])
    undefined_weight = float(protocol.raw["loss"]["undefined_flip_binary_weight"])
    coefficients = execution["coefficient_calibration"]["resolved_coefficients"]
    control_step = make_control_update_step(
        fast_day_weights=resources.fast_day_weights,
        undefined_loss_weight=undefined_weight,
        model_apply=resources.model_definition.apply,
    )
    candidate_steps: dict[tuple[int, bool, str], Any] = {}
    runtimes: dict[str, LandpointRuntime] = {}
    interval_started = time.perf_counter()
    interval_losses = []
    interval_gradients = []

    for update in range(next_update, budget):
        entry = schedule["entries"][update]
        if int(entry["update"]) != update:
            raise ValueError("rollout-stability schedule update order drift")
        reference = _reference_from_inventory(
            schedule["reference_inventory"],
            int(entry["reference_index"]),
        )
        horizon = int(entry["horizon"])
        spec = _horizon_spec(protocol, horizon)
        if arm_id == "one_step_continuation_control":
            anchor_batch = prepare_anchor_update(
                reference=reference,
                update=update,
                horizon=horizon,
                anchor_batch_size=anchor_batch_size,
                rollout_batch_size=int(spec["batch_size"]),
                seed=seed,
                resources=resources,
                runtime_cache=runtimes,
            )
            result = control_step(
                parameters,
                optimizer,
                anchor_batch,
                learning_rate,
            )
        else:
            prepared = prepare_rollout_update(
                reference=reference,
                update=update,
                horizon=horizon,
                anchor_batch_size=anchor_batch_size,
                rollout_batch_size=int(spec["batch_size"]),
                seed=seed,
                resources=resources,
            )
            rematerialize = bool(spec["rematerialize"])
            key = (horizon, rematerialize, prepared.trace_signature)
            if key not in candidate_steps:
                transition = _bind_dynamic_transition(resources, prepared.runtime)
                candidate_steps[key] = make_candidate_update_step(
                    coefficients=coefficients,
                    fast_day_weights=resources.fast_day_weights,
                    undefined_loss_weight=undefined_weight,
                    rematerialize=rematerialize,
                    model_apply=resources.model_definition.apply,
                    **_objective_kwargs(resources, transition),
                )
            result = candidate_steps[key](
                parameters,
                optimizer,
                prepared.anchor_batch,
                prepared.initial_states,
                prepared.initial_discrete_states,
                prepared.sequence,
                prepared.teacher_next_discrete_states,
                learning_rate,
            )
        if not bool(jax.device_get(result.update_applied)):
            component_record = _component_record(result.components)
            raise ValueError(
                "rollout-stability update failed closed at "
                f"update={update}: loss={float(jax.device_get(result.loss))}, "
                "gradient_norm="
                f"{float(jax.device_get(result.gradient_norm))}, "
                "nonfinite_gradient_values="
                f"{int(jax.device_get(result.nonfinite_gradient_values))}, "
                f"components={component_record}"
            )
        parameters = result.parameters
        optimizer = result.optimizer
        interval_losses.append(result.loss)
        interval_gradients.append(result.gradient_norm)
        completed = update + 1

        if completed % progress_every == 0 or completed == budget:
            loss_values = np.asarray(
                jax.device_get(jnp.stack(interval_losses)),
                dtype=np.float64,
            )
            gradient_values = np.asarray(
                jax.device_get(jnp.stack(interval_gradients)),
                dtype=np.float64,
            )
            record = {
                "next_update": completed,
                "last_horizon": horizon,
                "last_landpoint_id": reference.landpoint_id,
                "mean_loss_since_last_record": float(np.mean(loss_values)),
                "last_loss": float(loss_values[-1]),
                "mean_gradient_norm_since_last_record": float(
                    np.mean(gradient_values)
                ),
                "last_components": _component_record(result.components),
                "seconds_since_last_record": time.perf_counter()
                - interval_started,
            }
            history.append(record)
            interval_started = time.perf_counter()
            interval_losses.clear()
            interval_gradients.clear()
            print(
                "rollout_training "
                f"arm={arm_id} update={completed}/{budget} "
                f"horizon={horizon} loss={record['last_loss']:.8g}",
                flush=True,
            )
        if completed % checkpoint_every == 0 or completed == budget:
            checkpoint = build_arm_checkpoint(
                arm_id=arm_id,
                identity=identity,
                parameters=parameters,
                optimizer=optimizer,
                next_update=completed,
                schedule=schedule,
                history=history,
            )
            _atomic_pickle(checkpoint_path, checkpoint)

    final_checkpoint = _load_pickle(checkpoint_path)
    verify_arm_checkpoint(
        final_checkpoint,
        arm_id=arm_id,
        identity=identity,
        schedule=schedule,
    )
    if int(final_checkpoint["next_update"]) != budget:
        raise ValueError("rollout-stability arm stopped before its update budget")
    report = {
        "schema_version": "canonical_rollout_stability_arm_report_v1",
        "status": "completed",
        "arm_id": arm_id,
        "identity": identity,
        "updates": budget,
        "parameter_count": parameter_count(final_checkpoint["parameters"]),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "completed_horizon_counts": final_checkpoint[
            "completed_horizon_counts"
        ],
        "compiled_candidate_executables": len(candidate_steps),
        "history": final_checkpoint["history"],
        "sealed_test_used": False,
    }
    return _atomic_json(arm_root / "training_report.json", report)


def prepare_rollout_stability(
    *,
    protocol_path: str | Path,
    parent_report_path: str | Path,
    parent_best_checkpoint_path: str | Path,
    parent_optimizer_checkpoint_path: str | Path,
    dataset_path: str | Path,
    statistics_path: str | Path,
    acceptance_path: str | Path,
    training_protocol_path: str | Path,
    environment_lock_path: str | Path,
    output_root: str | Path,
    verify_dataset_hashes: bool = True,
) -> Path:
    """Create an immutable preflight and deterministic update schedule."""

    protocol = load_rollout_stability_protocol(protocol_path)
    parent = verify_parent_architecture_assets(
        protocol=protocol,
        report_path=parent_report_path,
        best_checkpoint_path=parent_best_checkpoint_path,
        optimizer_checkpoint_path=parent_optimizer_checkpoint_path,
    )
    paths = {
        "dataset_manifest": Path(dataset_path).resolve(),
        "training_statistics": Path(statistics_path).resolve(),
        "acceptance_report": Path(acceptance_path).resolve(),
    }
    for name, path in paths.items():
        expected = protocol.raw["artifacts"][f"{name}_sha256"]
        if _sha256_file(path) != expected:
            raise ValueError(f"rollout stability {name} hash drift")
    training_protocol = load_training_protocol(training_protocol_path)
    if training_protocol.sha256 != protocol.raw["artifacts"]["training_protocol_sha256"]:
        raise ValueError("rollout stability production training protocol hash drift")
    accepted = verify_training_acceptance(
        paths["acceptance_report"],
        paths["dataset_manifest"],
        paths["training_statistics"],
        verify_dataset_hashes=verify_dataset_hashes,
    )
    index = load_dataset_index(paths["dataset_manifest"], verify_hashes=False)
    schedule = build_sampling_schedule(index, protocol)
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    schedule_path = output_root / "sampling_schedule.json"
    schedule_payload = dict(schedule)
    if schedule_path.exists():
        existing = json.loads(schedule_path.read_text(encoding="utf-8"))
        if existing != schedule_payload:
            raise ValueError("existing rollout stability sampling schedule drift")
    else:
        _atomic_json(schedule_path, schedule_payload)

    environment_lock_path = Path(environment_lock_path).resolve()
    payload = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "status": "awaiting_coefficient_calibration",
        "protocol": {
            "path": str(protocol.path),
            "sha256": protocol.sha256,
        },
        "parent_architecture": parent,
        "artifacts": {
            name: {"path": str(path), "sha256": _sha256_file(path)}
            for name, path in paths.items()
        }
        | {
            "training_protocol": {
                "path": str(training_protocol.path),
                "sha256": training_protocol.sha256,
            },
            "environment_lock": {
                "path": str(environment_lock_path),
                "sha256": _sha256_file(environment_lock_path),
            },
            "sampling_schedule": {
                "path": str(schedule_path),
                "sha256": _sha256_file(schedule_path),
                "canonical_sha256": schedule["canonical_sha256"],
            },
        },
        "accepted_identity": dict(accepted["identity"]),
        "optimizer_update_budget": int(
            protocol.raw["optimization"]["screening_updates"]
        ),
        "screening_seed": int(protocol.raw["optimization"]["screening_seed"]),
        "output_root": str(output_root),
        "sealed_test_used": False,
    }
    preflight_path = output_root / "rollout_stability_preflight.json"
    if preflight_path.exists():
        existing = json.loads(preflight_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("existing rollout stability preflight identity drift")
        return preflight_path
    return _atomic_json(preflight_path, payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("prepare", "calibrate", "manifest", "arm"),
        default="prepare",
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--execution", type=Path)
    parser.add_argument("--arm", choices=ARM_IDS)
    parser.add_argument("--parent-report", type=Path)
    parser.add_argument("--parent-best-checkpoint", type=Path)
    parser.add_argument("--parent-optimizer-checkpoint", type=Path)
    parser.add_argument("--acceptance", type=Path)
    parser.add_argument("--training-protocol", type=Path)
    parser.add_argument("--environment-lock", type=Path)
    parser.add_argument("--skip-dataset-hash-verification", action="store_true")
    return parser


def _require_args(args: argparse.Namespace, names: Sequence[str]) -> None:
    missing = [name for name in names if getattr(args, name) is None]
    if missing:
        flags = ", ".join(f"--{name.replace('_', '-')}" for name in missing)
        raise ValueError(f"{args.phase} phase requires {flags}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.phase == "prepare":
        _require_args(
            args,
            (
                "parent_report",
                "parent_best_checkpoint",
                "parent_optimizer_checkpoint",
                "acceptance",
                "training_protocol",
                "environment_lock",
            ),
        )
        output = prepare_rollout_stability(
            protocol_path=args.protocol,
            parent_report_path=args.parent_report,
            parent_best_checkpoint_path=args.parent_best_checkpoint,
            parent_optimizer_checkpoint_path=args.parent_optimizer_checkpoint,
            dataset_path=args.dataset,
            statistics_path=args.statistics,
            acceptance_path=args.acceptance,
            training_protocol_path=args.training_protocol,
            environment_lock_path=args.environment_lock,
            output_root=args.output_root,
            verify_dataset_hashes=not args.skip_dataset_hash_verification,
        )
    elif args.phase == "calibrate":
        _require_args(args, ("preflight",))
        output = run_coefficient_calibration(
            preflight_path=args.preflight,
            protocol_path=args.protocol,
            dataset_path=args.dataset,
            statistics_path=args.statistics,
            output_root=args.output_root,
            plan_path=args.plan,
        )
    elif args.phase == "manifest":
        _require_args(args, ("preflight", "calibration"))
        output = create_execution_manifest(
            preflight_path=args.preflight,
            calibration_path=args.calibration,
            protocol_path=args.protocol,
            output_root=args.output_root,
        )
    else:
        _require_args(args, ("execution", "arm"))
        output = run_rollout_stability_arm(
            arm_id=args.arm,
            execution_path=args.execution,
            protocol_path=args.protocol,
            dataset_path=args.dataset,
            statistics_path=args.statistics,
            output_root=args.output_root,
            plan_path=args.plan,
        )
    print(json.dumps({"status": "completed", "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
