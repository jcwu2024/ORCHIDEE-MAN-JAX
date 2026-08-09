"""Contract-bound sampling and objectives for the successor bounded small fit."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.daily_markov_contract import load_markov_shard
from research.daily_coarse_graining.markov_dataset import defined_numeric_mask
from research.daily_coarse_graining.typed_sidecar import read_typed_sidecar_shard

from .long_horizon_objective import (
    long_horizon_biomass_productivity_objective,
    rollout_hard_gate_counts,
)
from .successor_fast_owner_training import (
    build_successor_fast_owner_targets,
    successor_fast_owner_loss,
)
from .successor_training import (
    LONG_HORIZON_COMPONENTS,
    TERMINAL_STOCK_FIELDS,
    extract_biomass_productivity_trajectories,
    rollout_daily_operator,
    trajectories_from_rollout,
)

SCHEMA_VERSION = "successor_small_fit_runner_v1"
FROZEN_CONTRACT_SHA256 = "8317fc74b6ba921b0ed60e774743884ddc81b233c75ce1823f3246211d1640de"
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3] / "manifests" / "coarse_graining" / "successor_small_fit_runner_v1.json"
)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("contract_sha256", None)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SuccessorSmallFitContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def sampling(self) -> Mapping[str, Any]:
        return self.payload["sampling"]

    @property
    def optimizer(self) -> Mapping[str, Any]:
        return self.payload["optimizer"]


def load_successor_small_fit_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> SuccessorSmallFitContract:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
    observed = _canonical_sha256(payload)
    if payload.get("contract_sha256") != observed:
        raise ValueError("successor small-fit contract hash mismatch")
    if observed != FROZEN_CONTRACT_SHA256:
        raise ValueError("successor small-fit frozen hash drift")

    sampling = payload.get("sampling", {})
    if (
        sampling.get("split") != "train/train"
        or sampling.get("test_split_sealed") is not True
        or sampling.get("same_schedule_for_both_arms") is not True
        or sampling.get("exclude_first_shard_row") is not True
    ):
        raise ValueError("successor small-fit sampling policy drift")
    if (
        int(sampling.get("maximum_updates_per_arm", 0)) != 500
        or int(sampling.get("one_day_batch_size", 0)) != 16
        or int(sampling.get("rollout_batch_size", 0)) != 2
        or int(sampling.get("rollout_days", 0)) != 30
        or tuple(sampling.get("objective_horizons_days", ())) != (1, 7, 30)
    ):
        raise ValueError("successor small-fit batch or horizon budget drift")

    optimizer = payload.get("optimizer", {})
    if (
        optimizer.get("name") != "adamw"
        or optimizer.get("schedule") != "linear_warmup_cosine_decay"
        or int(optimizer.get("warmup_updates", 0)) != 50
        or int(optimizer.get("maximum_updates", 0)) != 500
        or float(optimizer.get("global_gradient_norm_clip", 0.0)) != 1.0
    ):
        raise ValueError("successor small-fit optimizer drift")
    retained = payload.get("retained_tail", {})
    if (
        retained.get("runtime_scope") != "landpoint_year"
        or int(retained.get("native_forcing_records_per_day", 0)) != 5
        or retained.get("reconstruct_48_forcing_records") is not False
        or retained.get("execute_48_step_state_scan") is not False
    ):
        raise ValueError("successor small-fit retained-tail policy drift")
    return SuccessorSmallFitContract(payload, source, observed)


def _reference_key(reference: Any) -> tuple[str, int]:
    return (str(reference.parent.landpoint_id), int(reference.parent.year))


def _balanced_reference_order(references: Sequence[Any], seed: int) -> tuple[Any, ...]:
    grouped: dict[str, list[Any]] = {}
    for reference in sorted(references, key=_reference_key):
        parent = reference.parent
        if parent.spatial_split != "train" or parent.temporal_split != "train":
            raise ValueError("small-fit schedule attempted to admit a non-train shard")
        grouped.setdefault(str(parent.landpoint_id), []).append(reference)
    if not grouped:
        raise ValueError("small-fit schedule has no train/train references")

    rng = np.random.default_rng(seed)
    landpoints = np.asarray(sorted(grouped), dtype=object)
    landpoints = landpoints[rng.permutation(landpoints.size)]
    per_landpoint = {
        landpoint: [rows[index] for index in rng.permutation(len(rows))] for landpoint, rows in grouped.items()
    }
    ordered = []
    for round_index in range(max(len(rows) for rows in per_landpoint.values())):
        for landpoint in landpoints:
            rows = per_landpoint[str(landpoint)]
            if round_index < len(rows):
                ordered.append(rows[round_index])
    return tuple(ordered)


def _update_rng(seed: int, update: int, reference: Any) -> np.random.Generator:
    identity = (
        f"{seed}:{update}:{reference.parent.landpoint_id}:"
        f"{reference.parent.year}:{reference.parent.sha256}:{reference.sha256}"
    ).encode()
    entropy = np.frombuffer(hashlib.sha256(identity).digest(), dtype=np.uint32)
    return np.random.default_rng(np.random.SeedSequence(entropy.tolist()))


def build_successor_small_fit_schedule(
    references: Sequence[Any],
    contract: SuccessorSmallFitContract,
    *,
    updates: int | None = None,
) -> Mapping[str, Any]:
    """Freeze one portable matched schedule without exposing identity to the model."""

    sampling = contract.sampling
    maximum = int(sampling["maximum_updates_per_arm"])
    count = maximum if updates is None else int(updates)
    if not 1 <= count <= maximum:
        raise ValueError("small-fit schedule exceeds the frozen update budget")
    ordered = _balanced_reference_order(references, int(sampling["seed"]))
    if count > len(ordered):
        raise ValueError("small-fit schedule requests more unique annual shards than exist")

    anchor_size = int(sampling["one_day_batch_size"])
    rollout_size = int(sampling["rollout_batch_size"])
    horizon = int(sampling["rollout_days"])
    minimum_start = 1
    entries = []
    for update, reference in enumerate(ordered[:count], start=1):
        days = int(reference.day_count)
        if days - minimum_start < anchor_size or days - horizon < minimum_start:
            raise ValueError(f"annual shard {_reference_key(reference)} is too short")
        rng = _update_rng(int(sampling["seed"]), update, reference)
        anchor_starts = rng.choice(
            np.arange(minimum_start, days, dtype=np.int64),
            size=anchor_size,
            replace=False,
        )
        rollout_starts = rng.choice(
            np.arange(minimum_start, days - horizon + 1, dtype=np.int64),
            size=rollout_size,
            replace=False,
        )
        entries.append(
            {
                "update": update,
                "landpoint_id": str(reference.parent.landpoint_id),
                "year": int(reference.parent.year),
                "parent_shard_sha256": str(reference.parent.sha256),
                "typed_shard_sha256": str(reference.sha256),
                "day_count": days,
                "anchor_starts": [int(value) for value in anchor_starts],
                "rollout_starts": [int(value) for value in rollout_starts],
            }
        )
    payload = {
        "schema_version": "successor_small_fit_schedule_v1",
        "runner_contract_sha256": contract.sha256,
        "seed": int(sampling["seed"]),
        "updates": count,
        "entries": entries,
    }
    return payload | {"schedule_sha256": _canonical_sha256(payload)}


def verify_successor_small_fit_schedule(
    schedule: Mapping[str, Any],
    references: Sequence[Any],
    contract: SuccessorSmallFitContract,
) -> None:
    supplied = dict(schedule)
    observed = str(supplied.pop("schedule_sha256", ""))
    if observed != _canonical_sha256(supplied):
        raise ValueError("small-fit schedule hash mismatch")
    expected = build_successor_small_fit_schedule(
        references,
        contract,
        updates=int(schedule["updates"]),
    )
    if schedule != expected:
        raise ValueError("small-fit schedule does not match its frozen derivation")


class PreparedSuccessorSmallFitBatch(NamedTuple):
    anchor_inputs: Any
    anchor_targets: Any
    rollout_inputs: Any
    rollout_targets: Any
    expected_next_state: Any
    expected_next_state_defined: Any
    expected_fast_gpp: Any
    expected_fast_gpp_defined: Any
    expected_fast_day_target: Any
    expected_next_discrete_state: Any


def _stack(values: Sequence[Any]) -> Any:
    if not values:
        raise ValueError("cannot stack an empty small-fit tree")
    return jax.tree_util.tree_map(lambda *items: np.stack(items), *values)


def prepare_successor_small_fit_batch(
    reference: Any,
    entry: Mapping[str, Any],
    *,
    adapter: Any,
    schema: Any,
    owner_contract: Any,
) -> PreparedSuccessorSmallFitBatch:
    """Load one coherent annual shard and assemble its matched training update."""

    parent_ref = reference.parent
    if parent_ref.spatial_split != "train" or parent_ref.temporal_split != "train":
        raise ValueError("small-fit batch attempted to load a non-train shard")
    expected_identity = (
        str(entry["landpoint_id"]),
        int(entry["year"]),
        str(entry["parent_shard_sha256"]),
        str(entry["typed_shard_sha256"]),
    )
    observed_identity = (
        str(parent_ref.landpoint_id),
        int(parent_ref.year),
        str(parent_ref.sha256),
        str(reference.sha256),
    )
    if observed_identity != expected_identity:
        raise ValueError("small-fit schedule and coherent shard identity differ")

    parent = load_markov_shard(parent_ref.path, contract=adapter.contract)
    typed_values, typed_defined, typed_days = read_typed_sidecar_shard(
        reference.path,
        contract=adapter.typed_contract,
        expected_days=reference.day_count,
    )
    if not np.array_equal(typed_days, parent.day_index):
        raise ValueError("small-fit typed and parent day axes differ")

    def example(row: int) -> Any:
        return adapter.assemble(
            parent.sample(row),
            supplemental_values={name: value[row] for name, value in typed_values.items()},
            supplemental_defined={name: value[row] for name, value in typed_defined.items()},
        )

    anchor_examples = [example(int(row)) for row in entry["anchor_starts"]]
    anchor_inputs = _stack([value.inputs for value in anchor_examples])
    anchor_targets = _stack(
        [
            build_successor_fast_owner_targets(
                value,
                schema=schema,
                markov_contract=adapter.contract,
                owner_contract=owner_contract,
            )
            for value in anchor_examples
        ]
    )

    horizon = 30
    rollout_inputs = []
    rollout_targets = []
    expected_gpp = []
    expected_gpp_defined = []
    expected_fast_day_target = []
    expected_state = []
    expected_discrete = []
    for raw_start in entry["rollout_starts"]:
        start = int(raw_start)
        stop = start + horizon
        if int(parent.day_index[start]) < 2:
            raise ValueError("small-fit rollout entered a first-day lifecycle row")
        examples = [example(row) for row in range(start, stop)]
        rollout_inputs.append(_stack([value.inputs for value in examples]))
        rollout_targets.append(
            _stack(
                [
                    build_successor_fast_owner_targets(
                        value,
                        schema=schema,
                        markov_contract=adapter.contract,
                        owner_contract=owner_contract,
                    )
                    for value in examples
                ]
            )
        )
        expected_gpp.append(np.stack([value.targets.supervision["daily.fast_gpp"].value for value in examples]))
        expected_gpp_defined.append(
            np.stack([value.targets.supervision["daily.fast_gpp"].defined for value in examples])
        )
        expected_fast_day_target.append(np.asarray(parent.fast_day_target[start:stop]))
        states = np.asarray(parent.state_trajectory[start + 1 : stop + 1])
        expected_state.append(states)
        expected_discrete.append(
            {name: np.asarray(values[start + 1 : stop + 1]) for name, values in parent.discrete_trajectories.items()}
        )
    state = np.stack(expected_state)
    return PreparedSuccessorSmallFitBatch(
        anchor_inputs=anchor_inputs,
        anchor_targets=anchor_targets,
        rollout_inputs=_stack(rollout_inputs),
        rollout_targets=_stack(rollout_targets),
        expected_next_state=state,
        expected_next_state_defined=defined_numeric_mask(state),
        expected_fast_gpp=np.stack(expected_gpp),
        expected_fast_gpp_defined=np.stack(expected_gpp_defined),
        expected_fast_day_target=np.stack(expected_fast_day_target),
        expected_next_discrete_state=_stack(expected_discrete),
    )


def successor_teacher_physical_outputs(
    targets: Mapping[str, Any],
    inputs: Any,
    schema: Any,
) -> Mapping[str, Any]:
    """Invert the frozen loss representation for a same-day Teacher replay."""

    if set(targets) != {field.name for field in schema.fields}:
        raise ValueError("small-fit Teacher replay target registry drift")
    canonical = getattr(inputs, "canonical_input", inputs)
    state = jnp.asarray(canonical.continuous_state, dtype=jnp.float64)
    result = {}
    for field in schema.fields:
        target = targets[field.name]
        value = jnp.asarray(target.value, dtype=jnp.float64)
        if field.value_representation == "day_start_delta":
            start = int(field.day_start_state_start)
            stop = int(field.day_start_state_stop)
            base = state[start:stop].reshape(field.shape)
            defined = jnp.isfinite(base) & (jnp.abs(base) < 0.5e20)
            value = value + jnp.where(defined, base, 0.0)
        elif field.value_representation == "defined_logit":
            value = jnp.where(value >= 0.5, 20.0, -20.0)
        result[field.name] = value
    return result


def successor_anchor_batch_loss(
    parameters: Any,
    inputs: Any,
    targets: Any,
    *,
    head: Any,
    schema: Any,
    scales: Mapping[str, Any],
) -> Any:
    def one(item: Any, wanted: Any) -> Any:
        assembled = head.input_assembler.assemble(item)
        context = head.assemble_context(parameters, assembled)
        outputs = head.named_outputs(parameters, context, assembled)
        return successor_fast_owner_loss(
            outputs,
            wanted,
            inputs=assembled,
            schema=schema,
            scales=scales,
        ).total

    return jnp.mean(jax.vmap(one)(inputs, targets))


def successor_rollout_component_losses(
    parameters: Any,
    batch: PreparedSuccessorSmallFitBatch,
    *,
    transition: Any,
    markov_contract: Any,
    flux_scales: Mapping[str, Any],
    stock_scales: Mapping[str, Any],
    rematerialize: bool = True,
) -> tuple[Any, Any]:
    """Return four unweighted 1/7/30-day components and the rollout trace."""

    def rollout_one(templates: Any) -> Any:
        return rollout_daily_operator(
            parameters,
            templates,
            transition=transition,
            contract=markov_contract,
            rematerialize=rematerialize,
        )

    trace = jax.vmap(rollout_one)(batch.rollout_inputs)
    predicted = jax.vmap(lambda value: trajectories_from_rollout(value, markov_contract))(trace)

    def expected_one(state, defined, gpp, gpp_defined, templates):
        return extract_biomass_productivity_trajectories(
            state,
            defined,
            daily_fast_gpp=gpp,
            daily_fast_gpp_defined=gpp_defined,
            pft_fraction=templates.pft.pft_fraction,
            active_pft_mask=templates.pft.active_pft_mask,
            contract=markov_contract,
        )

    expected = jax.vmap(expected_one)(
        batch.expected_next_state,
        batch.expected_next_state_defined,
        batch.expected_fast_gpp,
        batch.expected_fast_gpp_defined,
        batch.rollout_inputs,
    )
    unit = {name: 1.0 for name in LONG_HORIZON_COMPONENTS}

    def components_one(actual, wanted):
        result = long_horizon_biomass_productivity_objective(
            actual.fluxes,
            wanted.fluxes,
            actual.stocks,
            wanted.stocks,
            flux_scales=flux_scales,
            stock_scales=stock_scales,
            coefficients=unit,
            horizons=(1, 7, 30),
        )
        return jnp.stack(tuple(getattr(result, name) for name in LONG_HORIZON_COMPONENTS))

    return jnp.mean(jax.vmap(components_one)(predicted, expected), axis=0), trace


def successor_small_fit_loss(
    parameters: Any,
    batch: PreparedSuccessorSmallFitBatch,
    *,
    head: Any,
    schema: Any,
    loss_scales: Mapping[str, Any],
    transition: Any,
    markov_contract: Any,
    flux_scales: Mapping[str, Any],
    stock_scales: Mapping[str, Any],
    coefficients: Mapping[str, Any],
    rematerialize: bool = True,
) -> tuple[Any, tuple[Any, Any]]:
    """Evaluate the unit anchor plus calibrated long-horizon components."""

    if tuple(coefficients) != LONG_HORIZON_COMPONENTS:
        raise ValueError("small-fit coefficient registry drift")
    anchor = successor_anchor_batch_loss(
        parameters,
        batch.anchor_inputs,
        batch.anchor_targets,
        head=head,
        schema=schema,
        scales=loss_scales,
    )
    components, _trace = successor_rollout_component_losses(
        parameters,
        batch,
        transition=transition,
        markov_contract=markov_contract,
        flux_scales=flux_scales,
        stock_scales=stock_scales,
        rematerialize=rematerialize,
    )
    coefficient_vector = jnp.asarray(
        tuple(coefficients[name] for name in LONG_HORIZON_COMPONENTS),
        dtype=jnp.float64,
    )
    total = anchor + jnp.sum(coefficient_vector * components)
    return total, (anchor, components)


def successor_rollout_hard_gate_counts(
    batch: PreparedSuccessorSmallFitBatch,
    trace: Any,
    *,
    markov_contract: Any,
    budget_atol: float = 2.0e-12,
    stock_atol: float = 2.0e-12,
) -> Any:
    """Aggregate the frozen state, budget and lifecycle hard gates over a batch."""

    predicted = jax.vmap(lambda value: trajectories_from_rollout(value, markov_contract))(trace)

    def expected_one(state, defined, gpp, gpp_defined, templates):
        return extract_biomass_productivity_trajectories(
            state,
            defined,
            daily_fast_gpp=gpp,
            daily_fast_gpp_defined=gpp_defined,
            pft_fraction=templates.pft.pft_fraction,
            active_pft_mask=templates.pft.active_pft_mask,
            contract=markov_contract,
        )

    expected = jax.vmap(expected_one)(
        batch.expected_next_state,
        batch.expected_next_state_defined,
        batch.expected_fast_gpp,
        batch.expected_fast_gpp_defined,
        batch.rollout_inputs,
    )

    def one(actual, wanted, result, wanted_discrete):
        return rollout_hard_gate_counts(
            actual.stocks,
            wanted.stocks,
            nonnegative_stock_fields=TERMINAL_STOCK_FIELDS,
            budget_residuals={
                "water": result.diagnostics["water_budget_residual"],
                "carbon": result.diagnostics["carbon_budget_residual"],
            },
            predicted_discrete_state=result.next_discrete_state,
            expected_discrete_state=wanted_discrete,
            budget_atol=budget_atol,
            stock_atol=stock_atol,
        )

    counts = jax.vmap(one)(
        predicted,
        expected,
        trace.result,
        batch.expected_next_discrete_state,
    )
    return jax.tree_util.tree_map(lambda value: jnp.sum(value, axis=0), counts)


class SuccessorAdamWState(NamedTuple):
    step: Any
    first: Any
    second: Any


def initialize_successor_adamw(parameters: Any) -> SuccessorAdamWState:
    zeros = jax.tree_util.tree_map(jnp.zeros_like, parameters)
    return SuccessorAdamWState(jnp.asarray(0, dtype=jnp.int32), zeros, zeros)


def successor_learning_rate(contract: SuccessorSmallFitContract, update: int) -> float:
    optimizer = contract.optimizer
    maximum = int(optimizer["maximum_updates"])
    warmup = int(optimizer["warmup_updates"])
    if not 1 <= update <= maximum:
        raise ValueError("successor update lies outside the frozen budget")
    peak = float(optimizer["peak_learning_rate"])
    final = float(optimizer["final_learning_rate"])
    if update <= warmup:
        return peak * update / warmup
    progress = (update - warmup) / (maximum - warmup)
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return float(final + (peak - final) * cosine)


def successor_adamw_update(
    parameters: Any,
    gradients: Any,
    state: SuccessorAdamWState,
    learning_rate: Any,
    contract: SuccessorSmallFitContract,
) -> tuple[Any, SuccessorAdamWState, Any]:
    optimizer = contract.optimizer
    beta1 = float(optimizer["beta1"])
    beta2 = float(optimizer["beta2"])
    epsilon = float(optimizer["epsilon"])
    decay = float(optimizer["weight_decay"])
    clip_norm = float(optimizer["global_gradient_norm_clip"])
    squared = sum(
        (jnp.sum(jnp.asarray(leaf) ** 2) for leaf in jax.tree_util.tree_leaves(gradients)),
        jnp.asarray(0.0, dtype=jnp.float64),
    )
    norm = jnp.sqrt(squared)
    scale = jnp.minimum(1.0, clip_norm / jnp.maximum(norm, 1.0e-30))
    clipped = jax.tree_util.tree_map(lambda value: value * scale, gradients)
    step = state.step + 1
    first = jax.tree_util.tree_map(
        lambda old, grad: beta1 * old + (1.0 - beta1) * grad,
        state.first,
        clipped,
    )
    second = jax.tree_util.tree_map(
        lambda old, grad: beta2 * old + (1.0 - beta2) * grad * grad,
        state.second,
        clipped,
    )
    correction1 = 1.0 - jnp.power(beta1, step)
    correction2 = 1.0 - jnp.power(beta2, step)
    updated = jax.tree_util.tree_map(
        lambda value, mean, variance: (
            value
            - learning_rate * ((mean / correction1) / (jnp.sqrt(variance / correction2) + epsilon) + decay * value)
        ),
        parameters,
        first,
        second,
    )
    return updated, SuccessorAdamWState(step, first, second), norm
