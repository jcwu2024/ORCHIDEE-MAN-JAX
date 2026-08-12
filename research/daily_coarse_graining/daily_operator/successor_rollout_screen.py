"""Frozen 7/30-day matched rollout screen for the successor process head."""

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

from .long_horizon_objective import RolloutHardGateCounts
from .successor_architecture import (
    SOURCE_ROUTED_ATTENTION,
    SOURCE_ROUTED_GRAPH_CONTROL,
)
from .successor_fit_checkpoint import (
    load_successor_checkpoint_set,
    sha256_file,
)
from .successor_training import (
    CUMULATIVE_FLUX_FIELDS,
    TERMINAL_STOCK_FIELDS,
    extract_biomass_productivity_trajectories,
    rollout_daily_operator,
    trajectories_from_rollout,
)

SCREEN_SCHEMA_VERSION = "successor_7_30_rollout_screen_v1"
ARM_REPORT_SCHEMA_VERSION = "successor_7_30_rollout_arm_report_v1"
MATCHED_REPORT_SCHEMA_VERSION = "successor_7_30_rollout_matched_report_v1"
FROZEN_SCREEN_SHA256 = "99ec6eb08fc33ca94364de604b262fa29139e2dcfd7c9176d1a5d772a0b6fc07"
DEFAULT_SCREEN_PATH = (
    Path(__file__).resolve().parents[3]
    / "manifests"
    / "coarse_graining"
    / "successor_7_30_rollout_screen_v1.json"
)
REQUIRED_SLICES = ("temporal", "spatial", "joint")
REQUIRED_HORIZONS = (7, 30)
EXPECTED_ARMS = (SOURCE_ROUTED_ATTENTION, SOURCE_ROUTED_GRAPH_CONTROL)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("contract_sha256", None)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class SuccessorRolloutScreenContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def slices(self) -> Mapping[str, Mapping[str, str]]:
        return self.payload["data_policy"]["allowed_model_selection_slices"]

    @property
    def selection(self) -> Mapping[str, Any]:
        return self.payload["window_selection"]

    @property
    def gates(self) -> Mapping[str, Any]:
        return self.payload["gates"]


def load_successor_rollout_screen_contract(
    path: str | Path = DEFAULT_SCREEN_PATH,
) -> SuccessorRolloutScreenContract:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    observed = _canonical_sha256(payload)
    if payload.get("schema_version") != SCREEN_SCHEMA_VERSION:
        raise ValueError("unsupported successor rollout-screen schema")
    if payload.get("status") != "frozen_before_fit_result":
        raise ValueError("successor rollout screen was not frozen before fit results")
    if payload.get("contract_sha256") != observed or observed != FROZEN_SCREEN_SHA256:
        raise ValueError("successor rollout-screen contract hash drift")
    if tuple(payload["checkpoint_admission"]["arms"]) != EXPECTED_ARMS:
        raise ValueError("successor rollout-screen arm registry drift")
    if tuple(payload["window_selection"]["reported_horizons_days"]) != REQUIRED_HORIZONS:
        raise ValueError("successor rollout-screen horizon registry drift")
    if tuple(payload["data_policy"]["allowed_model_selection_slices"]) != REQUIRED_SLICES:
        raise ValueError("successor rollout-screen slice registry drift")
    return SuccessorRolloutScreenContract(payload, source, observed)


def _rank(seed: int, *parts: Any) -> str:
    text = "|".join((str(seed), *(str(part) for part in parts)))
    return hashlib.sha256(text.encode()).hexdigest()


def _reference_identity(reference: Any) -> Mapping[str, Any]:
    parent = reference.parent
    return {
        "landpoint_id": str(parent.landpoint_id),
        "year": int(parent.year),
        "parent_shard_sha256": str(parent.sha256),
        "typed_shard_sha256": str(reference.sha256),
        "day_count": int(reference.day_count),
    }


def resolve_successor_screen_inventory(
    index: Any,
    contract: SuccessorRolloutScreenContract,
) -> Mapping[str, Mapping[str, Any]]:
    """Resolve only the three declared non-test model-selection slices."""

    inventory = {}
    for slice_id in REQUIRED_SLICES:
        split = contract.slices[slice_id]
        spatial = str(split["spatial_split"])
        temporal = str(split["temporal_split"])
        if "test" in (spatial, temporal):
            raise ValueError("successor rollout screen attempted to admit sealed test data")
        references = tuple(
            index.select(spatial_split=spatial, temporal_split=temporal)
        )
        if not references:
            raise ValueError(f"successor rollout-screen slice is empty: {slice_id}")
        if any(
            reference.parent.spatial_split != spatial
            or reference.parent.temporal_split != temporal
            or "test"
            in (
                reference.parent.spatial_split,
                reference.parent.temporal_split,
            )
            for reference in references
        ):
            raise ValueError(f"successor rollout-screen split drift: {slice_id}")
        inventory[slice_id] = {
            "spatial_split": spatial,
            "temporal_split": temporal,
            "references": references,
        }
    return inventory


def build_successor_screen_window_plan(
    index: Any,
    contract: SuccessorRolloutScreenContract,
    *,
    mode: str,
) -> Mapping[str, Any]:
    """Freeze a deterministic bounded panel before either arm is evaluated."""

    try:
        mode_policy = contract.selection["modes"][mode]
    except KeyError as error:
        raise ValueError(f"unknown successor rollout-screen mode: {mode}") from error
    inventory = resolve_successor_screen_inventory(index, contract)
    seed = int(contract.selection["seed"])
    horizon = int(contract.selection["rollout_days"])
    shard_limit = int(mode_policy["shards_per_slice"])
    starts_per_shard = int(mode_policy["windows_per_shard"])
    slices = {}
    for slice_id in REQUIRED_SLICES:
        references = inventory[slice_id]["references"]
        ranked = sorted(
            references,
            key=lambda reference: (
                _rank(
                    seed,
                    "shard",
                    slice_id,
                    reference.parent.landpoint_id,
                    reference.parent.year,
                    reference.parent.sha256,
                    reference.sha256,
                ),
                str(reference.parent.landpoint_id),
                int(reference.parent.year),
            ),
        )
        unique_landpoints = []
        seen_landpoints = set()
        for reference in ranked:
            landpoint_id = str(reference.parent.landpoint_id)
            if landpoint_id not in seen_landpoints:
                seen_landpoints.add(landpoint_id)
                unique_landpoints.append(reference)
        if len(unique_landpoints) < shard_limit:
            raise ValueError(f"successor rollout-screen slice has too few shards: {slice_id}")
        entries = []
        for reference in unique_landpoints[:shard_limit]:
            # Row zero is excluded even when its source day index is already Day 2.
            valid_starts = range(1, int(reference.day_count) - horizon + 1)
            ranked_starts = sorted(
                valid_starts,
                key=lambda start: (
                    _rank(
                        seed,
                        "start",
                        slice_id,
                        reference.parent.landpoint_id,
                        reference.parent.year,
                        start,
                    ),
                    start,
                ),
            )
            selected_starts = []
            for start in ranked_starts:
                if all(abs(start - previous) >= horizon for previous in selected_starts):
                    selected_starts.append(start)
                if len(selected_starts) == starts_per_shard:
                    break
            if len(selected_starts) < starts_per_shard:
                raise ValueError("successor rollout-screen shard has too few valid windows")
            entries.append(
                _reference_identity(reference)
                | {"start_rows": sorted(selected_starts)}
            )
        slices[slice_id] = {
            "spatial_split": inventory[slice_id]["spatial_split"],
            "temporal_split": inventory[slice_id]["temporal_split"],
            "entries": entries,
        }
    plan = {
        "schema_version": "successor_7_30_rollout_window_plan_v1",
        "screen_contract_sha256": contract.sha256,
        "mode": mode,
        "promotion_eligible": bool(mode_policy["promotion_eligible"]),
        "selection_seed": seed,
        "rollout_days": horizon,
        "slices": slices,
        "sealed_test_used": False,
    }
    plan["plan_sha256"] = _canonical_sha256(plan)
    verify_successor_screen_window_plan(plan, index, contract)
    return plan


def verify_successor_screen_window_plan(
    plan: Mapping[str, Any],
    index: Any,
    contract: SuccessorRolloutScreenContract,
) -> None:
    observed = dict(plan)
    claimed = observed.pop("plan_sha256", None)
    if claimed != _canonical_sha256(observed):
        raise ValueError("successor rollout-screen window-plan hash drift")
    mode = str(plan.get("mode"))
    mode_policy = contract.selection["modes"].get(mode)
    if (
        plan.get("schema_version") != "successor_7_30_rollout_window_plan_v1"
        or plan.get("screen_contract_sha256") != contract.sha256
        or plan.get("sealed_test_used") is not False
        or mode_policy is None
        or bool(plan.get("promotion_eligible"))
        is not bool(mode_policy["promotion_eligible"])
        or int(plan.get("rollout_days", -1)) != int(contract.selection["rollout_days"])
        or set(plan.get("slices", {})) != set(REQUIRED_SLICES)
    ):
        raise ValueError("successor rollout-screen window-plan identity drift")
    inventory = resolve_successor_screen_inventory(index, contract)
    for slice_id in REQUIRED_SLICES:
        references = {
            (
                str(item.parent.landpoint_id),
                int(item.parent.year),
                str(item.parent.sha256),
                str(item.sha256),
            ): item
            for item in inventory[slice_id]["references"]
        }
        cell = plan["slices"][slice_id]
        if (
            cell.get("spatial_split") != inventory[slice_id]["spatial_split"]
            or cell.get("temporal_split") != inventory[slice_id]["temporal_split"]
            or len(cell.get("entries", ())) != int(mode_policy["shards_per_slice"])
        ):
            raise ValueError(f"successor rollout-screen slice plan drift: {slice_id}")
        landpoints = [str(entry["landpoint_id"]) for entry in cell["entries"]]
        if len(set(landpoints)) != len(landpoints):
            raise ValueError("successor rollout-screen plan repeated a landpoint within one slice")
        for entry in cell["entries"]:
            key = (
                str(entry["landpoint_id"]),
                int(entry["year"]),
                str(entry["parent_shard_sha256"]),
                str(entry["typed_shard_sha256"]),
            )
            reference = references.get(key)
            starts = tuple(int(value) for value in entry.get("start_rows", ()))
            if (
                reference is None
                or int(entry["day_count"]) != int(reference.day_count)
                or len(starts) != int(mode_policy["windows_per_shard"])
                or len(set(starts)) != len(starts)
                or any(
                    abs(left - right) < int(contract.selection["rollout_days"])
                    for index, left in enumerate(starts)
                    for right in starts[index + 1 :]
                )
                or any(
                    start < 1
                    or start + int(contract.selection["rollout_days"])
                    > int(reference.day_count)
                    for start in starts
                )
            ):
                raise ValueError("successor rollout-screen window identity drift")


def reference_for_screen_entry(
    index: Any,
    contract: SuccessorRolloutScreenContract,
    *,
    slice_id: str,
    entry: Mapping[str, Any],
) -> Any:
    split = contract.slices[slice_id]
    matches = tuple(
        reference
        for reference in index.select(
            spatial_split=split["spatial_split"],
            temporal_split=split["temporal_split"],
        )
        if _reference_identity(reference)
        == {name: entry[name] for name in _reference_identity(reference)}
    )
    if len(matches) != 1:
        raise ValueError("successor rollout-screen entry did not resolve uniquely")
    return matches[0]


class SuccessorRolloutScreenBatch(NamedTuple):
    rollout_inputs: Any
    expected_next_state: Any
    expected_next_state_defined: Any
    expected_fast_gpp: Any
    expected_fast_gpp_defined: Any
    expected_next_discrete_state: Any


def _stack(values: Sequence[Any]) -> Any:
    if not values:
        raise ValueError("cannot stack an empty successor rollout-screen tree")
    return jax.tree_util.tree_map(lambda *items: np.stack(items), *values)


def prepare_successor_rollout_screen_batch(
    reference: Any,
    entry: Mapping[str, Any],
    *,
    adapter: Any,
    rollout_days: int = 30,
) -> SuccessorRolloutScreenBatch:
    """Load one admitted annual shard without weakening train-only fit admission."""

    parent_ref = reference.parent
    if "test" in (parent_ref.spatial_split, parent_ref.temporal_split):
        raise ValueError("successor rollout screen attempted to read sealed test data")
    if _reference_identity(reference) != {
        name: entry[name] for name in _reference_identity(reference)
    }:
        raise ValueError("successor rollout-screen entry and shard identity differ")
    starts = tuple(int(value) for value in entry["start_rows"])
    if any(start < 1 or start + rollout_days > int(reference.day_count) for start in starts):
        raise ValueError("successor rollout screen crossed an annual shard boundary")

    parent = load_markov_shard(parent_ref.path, contract=adapter.contract)
    typed_values, typed_defined, typed_days = read_typed_sidecar_shard(
        reference.path,
        contract=adapter.typed_contract,
        expected_days=reference.day_count,
    )
    if not np.array_equal(typed_days, parent.day_index):
        raise ValueError("successor rollout-screen typed and parent day axes differ")

    def example(row: int) -> Any:
        return adapter.assemble(
            parent.sample(row),
            supplemental_values={name: value[row] for name, value in typed_values.items()},
            supplemental_defined={name: value[row] for name, value in typed_defined.items()},
        )

    rollout_inputs = []
    expected_state = []
    expected_gpp = []
    expected_gpp_defined = []
    expected_discrete = []
    for start in starts:
        stop = start + rollout_days
        examples = [example(row) for row in range(start, stop)]
        rollout_inputs.append(_stack([value.inputs for value in examples]))
        expected_gpp.append(
            np.stack([value.targets.supervision["daily.fast_gpp"].value for value in examples])
        )
        expected_gpp_defined.append(
            np.stack([value.targets.supervision["daily.fast_gpp"].defined for value in examples])
        )
        expected_state.append(np.asarray(parent.state_trajectory[start + 1 : stop + 1]))
        expected_discrete.append(
            {
                name: np.asarray(values[start + 1 : stop + 1])
                for name, values in parent.discrete_trajectories.items()
            }
        )
    states = np.stack(expected_state)
    return SuccessorRolloutScreenBatch(
        rollout_inputs=_stack(rollout_inputs),
        expected_next_state=states,
        expected_next_state_defined=defined_numeric_mask(states),
        expected_fast_gpp=np.stack(expected_gpp),
        expected_fast_gpp_defined=np.stack(expected_gpp_defined),
        expected_next_discrete_state=_stack(expected_discrete),
    )


class RolloutScreenStatistics(NamedTuple):
    flux_squared_sum: Any
    flux_signed_sum: Any
    flux_absolute_sum: Any
    flux_count: Any
    stock_squared_sum: Any
    stock_signed_sum: Any
    stock_absolute_sum: Any
    stock_count: Any
    hard_counts: RolloutHardGateCounts


def _trajectory_statistics(
    predicted: Any,
    expected: Any,
    scale: Any,
    *,
    cumulative: bool,
    horizon: int,
) -> tuple[Any, Any, Any, Any]:
    predicted_value = jnp.asarray(predicted.value, dtype=jnp.float64)
    expected_value = jnp.asarray(expected.value, dtype=jnp.float64)
    defined = (
        jnp.asarray(predicted.defined, dtype=jnp.bool_)
        & jnp.asarray(expected.defined, dtype=jnp.bool_)
    )
    if cumulative:
        valid = jnp.all(defined[:, :horizon], axis=1)
        left = jnp.sum(jnp.where(defined[:, :horizon], predicted_value[:, :horizon], 0.0), axis=1)
        right = jnp.sum(jnp.where(defined[:, :horizon], expected_value[:, :horizon], 0.0), axis=1)
        denominator = horizon * jnp.asarray(scale, dtype=jnp.float64)
    else:
        valid = defined[:, horizon - 1]
        left = predicted_value[:, horizon - 1]
        right = expected_value[:, horizon - 1]
        denominator = jnp.asarray(scale, dtype=jnp.float64)
    residual = jnp.where(valid, (left - right) / denominator, 0.0)
    return (
        jnp.sum(residual * residual, dtype=jnp.float64),
        jnp.sum(residual, dtype=jnp.float64),
        jnp.sum(jnp.abs(residual), dtype=jnp.float64),
        jnp.sum(valid, dtype=jnp.int64),
    )


def successor_rollout_screen_statistics(
    parameters: Any,
    batch: SuccessorRolloutScreenBatch,
    *,
    transition: Any,
    markov_contract: Any,
    flux_scales: Mapping[str, Any],
    stock_scales: Mapping[str, Any],
    budget_atol: float = 2.0e-12,
    stock_atol: float = 2.0e-12,
) -> RolloutScreenStatistics:
    """Run matched free rollouts and return additive 7/30-day evidence."""

    def rollout_one(templates: Any) -> Any:
        return rollout_daily_operator(
            parameters,
            templates,
            transition=transition,
            contract=markov_contract,
            rematerialize=False,
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
    flux = [
        [
            _trajectory_statistics(
                predicted.fluxes[name],
                expected.fluxes[name],
                flux_scales[name],
                cumulative=True,
                horizon=horizon,
            )
            for name in CUMULATIVE_FLUX_FIELDS
        ]
        for horizon in REQUIRED_HORIZONS
    ]
    stock = [
        [
            _trajectory_statistics(
                predicted.stocks[name],
                expected.stocks[name],
                stock_scales[name],
                cumulative=False,
                horizon=horizon,
            )
            for name in TERMINAL_STOCK_FIELDS
        ]
        for horizon in REQUIRED_HORIZONS
    ]

    def component(values: Sequence[Sequence[tuple[Any, ...]]], index: int) -> Any:
        return jnp.stack(tuple(jnp.stack(tuple(item[index] for item in row)) for row in values))

    from .successor_small_fit import successor_rollout_hard_gate_counts

    hard = successor_rollout_hard_gate_counts(
        batch,
        trace,
        markov_contract=markov_contract,
        budget_atol=budget_atol,
        stock_atol=stock_atol,
    )
    return RolloutScreenStatistics(
        *(component(flux, index) for index in range(4)),
        *(component(stock, index) for index in range(4)),
        hard,
    )


def empty_rollout_screen_statistics() -> RolloutScreenStatistics:
    zeros_flux = np.zeros((len(REQUIRED_HORIZONS), len(CUMULATIVE_FLUX_FIELDS)), dtype=np.float64)
    zeros_stock = np.zeros((len(REQUIRED_HORIZONS), len(TERMINAL_STOCK_FIELDS)), dtype=np.float64)
    hard = RolloutHardGateCounts(*(np.asarray(0, dtype=np.int64) for _ in RolloutHardGateCounts._fields))
    return RolloutScreenStatistics(
        zeros_flux.copy(),
        zeros_flux.copy(),
        zeros_flux.copy(),
        zeros_flux.astype(np.int64),
        zeros_stock.copy(),
        zeros_stock.copy(),
        zeros_stock.copy(),
        zeros_stock.astype(np.int64),
        hard,
    )


def add_rollout_screen_statistics(
    left: RolloutScreenStatistics,
    right: RolloutScreenStatistics,
) -> RolloutScreenStatistics:
    return jax.tree_util.tree_map(
        lambda first, second: np.asarray(first) + np.asarray(second),
        left,
        right,
    )


def _metric_record(squared: Any, signed: Any, absolute: Any, count: Any) -> Mapping[str, Any]:
    count_value = int(np.asarray(count))
    if count_value <= 0:
        raise ValueError("successor rollout-screen metric has no evaluated values")
    return {
        "normalized_rmse": float(np.sqrt(float(np.asarray(squared)) / count_value)),
        "normalized_signed_bias": float(np.asarray(signed)) / count_value,
        "normalized_mean_absolute_error": float(np.asarray(absolute)) / count_value,
        "evaluated_values": count_value,
    }


def _finalize_rollout_statistics(
    statistics: RolloutScreenStatistics,
    *,
    primary_keys: Sequence[str],
) -> Mapping[str, Any]:
    primary = set(primary_keys)
    cells = {}
    for horizon_index, horizon in enumerate(REQUIRED_HORIZONS):
        keys = {}
        for field_index, name in enumerate(CUMULATIVE_FLUX_FIELDS):
            key = f"cumulative.{name}"
            keys[key] = _metric_record(
                statistics.flux_squared_sum[horizon_index, field_index],
                statistics.flux_signed_sum[horizon_index, field_index],
                statistics.flux_absolute_sum[horizon_index, field_index],
                statistics.flux_count[horizon_index, field_index],
            )
        for field_index, name in enumerate(TERMINAL_STOCK_FIELDS):
            key = f"terminal.{name}"
            keys[key] = _metric_record(
                statistics.stock_squared_sum[horizon_index, field_index],
                statistics.stock_signed_sum[horizon_index, field_index],
                statistics.stock_absolute_sum[horizon_index, field_index],
                statistics.stock_count[horizon_index, field_index],
            )
        primary_mse = [keys[key]["normalized_rmse"] ** 2 for key in sorted(primary)]
        cells[str(horizon)] = {
            "primary_composite_normalized_rmse": float(np.sqrt(np.mean(primary_mse))),
            "keys": keys,
        }
    return {
        "horizons": cells,
        "hard_counts": {
            name: int(np.asarray(getattr(statistics.hard_counts, name)))
            for name in RolloutHardGateCounts._fields
        },
    }


def finalize_rollout_screen_statistics(
    statistics: RolloutScreenStatistics,
    *,
    contract: SuccessorRolloutScreenContract,
) -> Mapping[str, Any]:
    return _finalize_rollout_statistics(
        statistics,
        primary_keys=contract.payload["metrics"]["primary_keys"],
    )


class AdmittedSuccessorCheckpoints(NamedTuple):
    parameters: Mapping[str, Any]
    evidence: Mapping[str, Any]


def admit_successor_screen_checkpoints(
    contract: SuccessorRolloutScreenContract,
    *,
    fit_report_path: str | Path,
    checkpoint_dir: str | Path,
    expected_fit_report_sha256: str,
    expected_checkpoint_manifest_sha256: str,
) -> AdmittedSuccessorCheckpoints:
    """Admit both successful fit arms and discard their optimizer trees."""

    report_path = Path(fit_report_path).resolve()
    root = Path(checkpoint_dir).resolve()
    if sha256_file(report_path) != expected_fit_report_sha256:
        raise ValueError("successor fit report hash drift")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    policy = contract.payload["checkpoint_admission"]
    required_updates = int(policy["required_updates_per_arm"])
    completed = report.get("completed_updates_per_arm", {})
    counts = report.get("parameter_count_per_arm", {})
    checkpoint_record = report.get("checkpoint_set") or {}
    if (
        report.get("status") != policy["required_fit_status"]
        or report.get("run_kind") != policy["required_run_kind"]
        or report.get("checkpoint_written") is not True
        or report.get("promotion_gates_passed") is not True
        or report.get("teacher_data_regenerated") is not False
        or report.get("historical_v5_used") is not False
        or report.get("test_split_used") is not False
        or set(completed) != set(EXPECTED_ARMS)
        or any(int(completed[arm]) != required_updates for arm in EXPECTED_ARMS)
        or any(int(counts.get(arm, -1)) != int(policy["parameter_count_per_arm"]) for arm in EXPECTED_ARMS)
        or checkpoint_record.get("manifest_sha256") != expected_checkpoint_manifest_sha256
        or Path(str(checkpoint_record.get("path", ""))).resolve() != root
    ):
        raise ValueError("successor fit report did not satisfy checkpoint admission")
    loaded = load_successor_checkpoint_set(
        root,
        expected_manifest_sha256=expected_checkpoint_manifest_sha256,
        identity=report["identity"],
        expected_arms=EXPECTED_ARMS,
        required_updates=required_updates,
    )
    return AdmittedSuccessorCheckpoints(
        parameters={arm: loaded[arm]["parameters"] for arm in EXPECTED_ARMS},
        evidence={
            "fit_report": str(report_path),
            "fit_report_sha256": expected_fit_report_sha256,
            "checkpoint_dir": str(root),
            "checkpoint_manifest_sha256": expected_checkpoint_manifest_sha256,
            "fit_git_head": report["identity"]["git_head"],
            "updates_per_arm": required_updates,
            "optimizer_loaded_for_evaluation": False,
        },
    )


def _ratio_gate(
    gate_id: str,
    candidate: float,
    control: float,
    threshold: float,
) -> Mapping[str, Any]:
    candidate = float(candidate)
    control = float(control)
    if not np.isfinite(candidate) or not np.isfinite(control) or candidate < 0.0 or control < 0.0:
        raise ValueError(f"invalid successor rollout-screen ratio metric: {gate_id}")
    if control == 0.0:
        ratio = 1.0 if candidate == 0.0 else None
        passed = candidate == 0.0
        policy = "both_zero_equal" if passed else "nonzero_candidate_over_zero_control"
    else:
        ratio = candidate / control
        passed = ratio <= threshold
        policy = "ordinary_ratio"
    return {
        "id": gate_id,
        "candidate": candidate,
        "control": control,
        "ratio": ratio,
        "threshold_max": float(threshold),
        "zero_denominator_policy": policy,
        "passed": bool(passed),
    }


def classify_successor_rollout_screen(
    *,
    candidate: Mapping[str, Any],
    control: Mapping[str, Any],
    contract: SuccessorRolloutScreenContract,
) -> Mapping[str, Any]:
    """Classify only complete, byte-matched formal evidence."""

    for name, report in (("candidate", candidate), ("control", control)):
        if (
            report.get("schema_version") != ARM_REPORT_SCHEMA_VERSION
            or report.get("status") != "completed"
            or report.get("arm")
            != (SOURCE_ROUTED_ATTENTION if name == "candidate" else SOURCE_ROUTED_GRAPH_CONTROL)
            or report.get("sealed_test_used") is not False
            or set(report.get("slices", {})) != set(REQUIRED_SLICES)
        ):
            raise ValueError(f"successor rollout-screen {name} evidence is incomplete")
    if (
        candidate.get("window_plan_sha256") != control.get("window_plan_sha256")
        or candidate.get("window_plan_sha256") is None
        or candidate.get("mode") != control.get("mode")
        or candidate.get("window_count") != control.get("window_count")
    ):
        raise ValueError("successor rollout-screen arms did not use byte-identical windows")
    if candidate.get("mode") != "formal_bounded_screen":
        raise ValueError("free successor rollout smoke is not promotion eligible")

    gates = contract.gates
    metric_policy = contract.payload["metrics"]
    primary = set(metric_policy["primary_keys"])
    other = set(metric_policy["other_keys"])
    if primary & other or primary | other != {
        *(f"cumulative.{name}" for name in CUMULATIVE_FLUX_FIELDS),
        *(f"terminal.{name}" for name in TERMINAL_STOCK_FIELDS),
    }:
        raise ValueError("successor rollout-screen key classification drift")

    ratio_gates = []
    bias_gates = []
    for slice_id in REQUIRED_SLICES:
        candidate_slice = candidate["slices"][slice_id]
        control_slice = control["slices"][slice_id]
        if candidate_slice.get("window_count") != control_slice.get("window_count"):
            raise ValueError("successor rollout-screen slice window counts differ")
        for horizon in REQUIRED_HORIZONS:
            cell_id = str(horizon)
            candidate_cell = candidate_slice["horizons"][cell_id]
            control_cell = control_slice["horizons"][cell_id]
            ratio_gates.append(
                _ratio_gate(
                    f"primary_composite/{slice_id}/horizon_{horizon}",
                    candidate_cell["primary_composite_normalized_rmse"],
                    control_cell["primary_composite_normalized_rmse"],
                    1.0 - float(gates["attention_primary_composite_improvement_minimum"]),
                )
            )
            if set(candidate_cell["keys"]) != primary | other or set(control_cell["keys"]) != primary | other:
                raise ValueError("successor rollout-screen metric key registry drift")
            for key in sorted(primary | other):
                candidate_metric = candidate_cell["keys"][key]
                control_metric = control_cell["keys"][key]
                if candidate_metric["evaluated_values"] != control_metric["evaluated_values"]:
                    raise ValueError("successor rollout-screen metric denominators differ")
                regression = (
                    gates["primary_key_regression_maximum"]
                    if key in primary
                    else gates["other_key_regression_maximum"]
                )
                ratio_gates.append(
                    _ratio_gate(
                        f"key_rmse/{slice_id}/horizon_{horizon}/{key}",
                        candidate_metric["normalized_rmse"],
                        control_metric["normalized_rmse"],
                        1.0 + float(regression),
                    )
                )
                bias = abs(float(candidate_metric["normalized_signed_bias"]))
                bias_gates.append(
                    {
                        "id": f"candidate_bias/{slice_id}/horizon_{horizon}/{key}",
                        "absolute_normalized_bias": bias,
                        "threshold_max": float(gates["absolute_normalized_key_bias_maximum"]),
                        "passed": bias <= float(gates["absolute_normalized_key_bias_maximum"]),
                    }
                )

    hard_gates = []
    for arm_name, report in (("candidate", candidate), ("control", control)):
        for slice_id in REQUIRED_SLICES:
            hard = report["slices"][slice_id]["hard_counts"]
            if set(hard) != set(contract.payload["hard_counts"]):
                raise ValueError("successor rollout-screen hard-count registry drift")
            for key, value in hard.items():
                hard_gates.append(
                    {
                        "id": f"{arm_name}/{slice_id}/{key}",
                        "count": int(value),
                        "required": int(gates["hard_failure_count"]),
                        "passed": int(value) == int(gates["hard_failure_count"]),
                    }
                )
    passed = (
        all(item["passed"] for item in ratio_gates)
        and all(item["passed"] for item in bias_gates)
        and all(item["passed"] for item in hard_gates)
    )
    return {
        "schema_version": MATCHED_REPORT_SCHEMA_VERSION,
        "status": "passed" if passed else "rejected",
        "decision": (
            "advance_to_separately_authorized_365_day_gate"
            if passed
            else "stop_and_attribute_bounded_7_30_day_failure"
        ),
        "window_plan_sha256": candidate["window_plan_sha256"],
        "ratio_gates": ratio_gates,
        "bias_gates": bias_gates,
        "hard_gates": hard_gates,
        "sealed_test_used": False,
    }
