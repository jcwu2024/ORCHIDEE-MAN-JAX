"""Frozen sampling and optimization contract for target-oriented training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, NamedTuple, Sequence

import jax.numpy as jnp

from .long_horizon_objective import MaskedTrajectory, NormalizedError
from .target_oriented_contract import (
    INDEPENDENT_FLUX_BASIS,
    INDEPENDENT_STOCK_BASIS,
    NecessaryProcessSupervisionResult,
    OneDayStateAnchorResult,
    ScienceWindowResult,
    TargetOrientedContract,
    necessary_process_supervision_loss,
)
from .training_loss import ProcessLossResult
from .types import StateDefinedMasks, StateFeatureGroups

SCHEMA_VERSION = "target_oriented_bounded_training_v1"
FROZEN_CONTRACT_SHA256 = "68d1079c7bcaffed11f880688dc465ce02285c526e5651b0a9ce33069657256d"
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3] / "manifests" / "coarse_graining" / "target_oriented_bounded_training_v1.json"
)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("contract_sha256", None)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TargetOrientedTrainingContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def data(self) -> Mapping[str, Any]:
        return self.payload["data"]

    @property
    def optimizer(self) -> Mapping[str, Any]:
        return self.payload["optimizer"]


class ErrorSufficientStatistics(NamedTuple):
    sum_squared: Any
    sum_signed: Any
    count: Any


class ScienceWindowSufficientStatistics(NamedTuple):
    fluxes: Mapping[str, ErrorSufficientStatistics]
    stocks: Mapping[str, ErrorSufficientStatistics]


def _validate_contract(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("target-oriented training schema drift")
    if payload.get("status") != "frozen_for_training_step_preflight":
        raise ValueError("target-oriented training contract is not frozen")

    identity = payload.get("identity", {})
    if (
        identity.get("candidate") != "target_oriented_hierarchical_coordinate_attention_v1"
        or int(identity.get("parameter_count", 0)) != 83_755_173
        or identity.get("target_oriented_objective_sha256")
        != "7c9857c0ca57ee65cfcf7514a216a217c3b1661f6cc869c6dfb85d3786bfce5c"
        or identity.get("dataset_manifest_sha256") != "5fc2c0691bfa53252f875f7646f3c1a5bea203c2759a6e947b26bb51dc13aa99"
    ):
        raise ValueError("target-oriented training identity drift")

    data = payload.get("data", {})
    if data.get("release_id") != "pft14-daily-teacher-v6" or data.get("split") != "train/train":
        raise ValueError("target-oriented training data boundary drift")
    if any(
        bool(data.get(name))
        for name in (
            "teacher_regeneration_allowed",
            "historical_v5_allowed",
            "sealed_test_allowed",
        )
    ):
        raise ValueError("target-oriented training opens a forbidden data path")
    one = data.get("one_day", {})
    seven = data.get("seven_day", {})
    thirty = data.get("thirty_day", {})
    if (
        int(one.get("sample_count", 0)),
        int(one.get("distinct_point_year_count", 0)),
        int(one.get("samples_per_point_year", 0)),
    ) != (32, 16, 2):
        raise ValueError("target-oriented one-day batch drift")
    if (
        int(seven.get("window_count", 0)),
        int(seven.get("distinct_point_year_count", 0)),
        int(seven.get("horizon_days", 0)),
    ) != (8, 8, 7):
        raise ValueError("target-oriented seven-day batch drift")
    if (
        int(thirty.get("window_count", 0)),
        int(thirty.get("distinct_point_year_count", 0)),
        int(thirty.get("horizon_days", 0)),
    ) != (4, 4, 30):
        raise ValueError("target-oriented thirty-day batch drift")
    if (
        int(data.get("distinct_point_years_per_update", 0)),
        int(data.get("distinct_landpoints_per_update", 0)),
        int(data.get("teacher_days_per_update", 0)),
    ) != (28, 28, 208):
        raise ValueError("target-oriented effective-batch exposure drift")

    micro = payload.get("microbatch", {})
    if (
        int(micro.get("one_day_size", 0)),
        int(micro.get("seven_day_window_size", 0)),
        int(micro.get("thirty_day_window_size", 0)),
        int(micro.get("optimizer_updates_per_effective_batch", 0)),
    ) != (4, 1, 1, 1):
        raise ValueError("target-oriented microbatch policy drift")
    if (
        micro.get("global_bias_reduction") != "exact_sufficient_statistics_across_the_full_effective_batch"
        or micro.get("rollout_rematerialization") is not True
        or micro.get("multi_gpu_required") is not False
    ):
        raise ValueError("target-oriented accumulation or rematerialization drift")

    objective = payload.get("objective", {})
    expected_components = (
        "one_day_state_anchor",
        "seven_day_science",
        "thirty_day_science",
        "necessary_process_supervision",
    )
    if tuple(objective.get("components", ())) != expected_components or objective.get("coefficients") != {
        "one_day_state_anchor": 1.0,
        "seven_day_science": 1.0,
        "thirty_day_science": 1.0,
        "necessary_process_supervision": 0.1,
    }:
        raise ValueError("target-oriented frozen objective drift")

    optimizer = payload.get("optimizer", {})
    if (
        optimizer.get("name"),
        float(optimizer.get("peak_learning_rate", 0.0)),
        float(optimizer.get("final_learning_rate", 0.0)),
        int(optimizer.get("warmup_updates", 0)),
        int(optimizer.get("maximum_updates", 0)),
        tuple(optimizer.get("stage_milestones", ())),
        optimizer.get("schedule"),
    ) != (
        "adamw",
        1.0e-4,
        1.0e-5,
        250,
        3000,
        (250, 1000, 3000),
        "linear_warmup_cosine_decay",
    ):
        raise ValueError("target-oriented optimizer schedule drift")
    clipping = optimizer.get("global_gradient_norm_clip", {})
    if (
        int(clipping.get("calibration_batches", 0)),
        clipping.get("calibration_split"),
        clipping.get("statistic"),
        clipping.get("freeze_before_first_update"),
        clipping.get("validation_tuning_allowed"),
    ) != (8, "train/train", "p90", True, False):
        raise ValueError("target-oriented gradient clipping calibration drift")

    validation = payload.get("validation", {})
    if (
        tuple(validation.get("slices", ())) != ("temporal", "spatial", "joint")
        or tuple(validation.get("horizons_days", ())) != (7, 30)
        or int(validation.get("windows_per_slice_per_horizon", 0)) != 16
        or tuple(validation.get("evaluate_at_stage_milestones", ())) != (250, 1000, 3000)
        or validation.get("promotion") != "absolute_target_oriented_7_30_day_gates_only"
        or validation.get("sealed_test_used") is not False
    ):
        raise ValueError("target-oriented validation policy drift")

    preflight = payload.get("preflight", {})
    if (
        int(preflight.get("updates_retained", -1)) != 0
        or preflight.get("checkpoint_written") is not False
        or int(preflight.get("required_gradient_calibration_batches", 0)) != 8
        or int(preflight.get("actual_peak_bytes_in_use_maximum", 0)) != 28 * 1024**3
    ):
        raise ValueError("target-oriented preflight policy drift")

    inference = payload.get("inference_constraints", {})
    if (
        int(inference.get("native_forcing_records_per_day", 0)) != 5
        or inference.get("reconstruct_48_forcing_records") is not False
        or inference.get("execute_48_step_state_scan") is not False
        or inference.get("cross_day_neural_hidden_state") is not False
    ):
        raise ValueError("target-oriented inference boundary drift")


def load_target_oriented_training_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> TargetOrientedTrainingContract:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    _validate_contract(payload)
    observed = _canonical_sha256(payload)
    if payload.get("contract_sha256") != observed:
        raise ValueError("target-oriented training contract SHA256 drift")
    if observed != FROZEN_CONTRACT_SHA256:
        raise ValueError("target-oriented training contract is not frozen v1")
    return TargetOrientedTrainingContract(payload, source, observed)


def normalized_error_sufficient_statistics(
    predicted: Any,
    expected: Any,
    defined: Any,
    scale: Any,
) -> ErrorSufficientStatistics:
    predicted = jnp.asarray(predicted, dtype=jnp.float64)
    expected = jnp.asarray(expected, dtype=jnp.float64)
    defined = jnp.asarray(defined, dtype=jnp.bool_)
    if predicted.shape != expected.shape or defined.shape != predicted.shape:
        raise ValueError("target-oriented sufficient-statistic shapes differ")
    scale = jnp.broadcast_to(jnp.asarray(scale, dtype=jnp.float64), predicted.shape)
    valid = defined & jnp.isfinite(predicted) & jnp.isfinite(expected) & jnp.isfinite(scale) & (scale > 0.0)
    residual = jnp.where(valid, (predicted - expected) / scale, 0.0)
    return ErrorSufficientStatistics(
        sum_squared=jnp.sum(residual * residual, dtype=jnp.float64),
        sum_signed=jnp.sum(residual, dtype=jnp.float64),
        count=jnp.sum(valid, dtype=jnp.float64),
    )


def add_error_sufficient_statistics(
    left: ErrorSufficientStatistics,
    right: ErrorSufficientStatistics,
) -> ErrorSufficientStatistics:
    return ErrorSufficientStatistics(
        sum_squared=left.sum_squared + right.sum_squared,
        sum_signed=left.sum_signed + right.sum_signed,
        count=left.count + right.count,
    )


def merge_error_sufficient_statistics(
    values: Sequence[Mapping[str, ErrorSufficientStatistics]],
) -> Mapping[str, ErrorSufficientStatistics]:
    if not values:
        raise ValueError("cannot merge an empty target-oriented statistic sequence")
    keys = tuple(values[0])
    if any(tuple(value) != keys for value in values[1:]):
        raise ValueError("target-oriented sufficient-statistic registry drift")
    result = dict(values[0])
    for value in values[1:]:
        result = {name: add_error_sufficient_statistics(result[name], value[name]) for name in keys}
    return result


def normalized_error_from_sufficient_statistics(
    statistics: ErrorSufficientStatistics,
) -> NormalizedError:
    count = jnp.asarray(statistics.count, dtype=jnp.float64)
    mean = jnp.where(count > 0.0, statistics.sum_signed / count, 0.0)
    mse = jnp.where(count > 0.0, statistics.sum_squared / count, 0.0)
    return NormalizedError(
        mse=mse,
        mean_bias=mean,
        bias_squared=mean * mean,
        count=count,
    )


def one_day_state_sufficient_statistics(
    predicted: StateFeatureGroups,
    expected: StateFeatureGroups,
    predicted_defined: StateDefinedMasks,
    expected_defined: StateDefinedMasks,
    scales: StateFeatureGroups,
) -> Mapping[str, ErrorSufficientStatistics]:
    return {
        name: normalized_error_sufficient_statistics(
            getattr(predicted, name),
            getattr(expected, name),
            jnp.asarray(getattr(predicted_defined, name), dtype=jnp.bool_)
            & jnp.asarray(getattr(expected_defined, name), dtype=jnp.bool_),
            getattr(scales, name),
        )
        for name in StateFeatureGroups._fields
    }


def one_day_state_anchor_from_sufficient_statistics(
    statistics: Mapping[str, ErrorSufficientStatistics],
) -> OneDayStateAnchorResult:
    if tuple(statistics) != tuple(StateFeatureGroups._fields):
        raise ValueError("target-oriented state statistic group drift")
    errors = {
        name: normalized_error_from_sufficient_statistics(statistics[name]) for name in StateFeatureGroups._fields
    }
    group_mse = {name: error.mse for name, error in errors.items()}
    group_bias = {name: error.bias_squared for name, error in errors.items()}
    group_count = {name: error.count for name, error in errors.items()}
    group_loss = {name: 0.5 * (error.mse + error.bias_squared) for name, error in errors.items()}
    active = jnp.stack(tuple(value > 0.0 for value in group_count.values()))
    values = jnp.stack(tuple(group_loss.values()))
    active_count = jnp.sum(active, dtype=jnp.float64)
    total = jnp.where(
        active_count > 0.0,
        jnp.sum(jnp.where(active, values, 0.0)) / active_count,
        0.0,
    )
    return OneDayStateAnchorResult(
        total=total,
        group_loss=group_loss,
        group_mse=group_mse,
        group_bias_squared=group_bias,
        group_count=group_count,
    )


def _masked_sum(left: MaskedTrajectory, right: MaskedTrajectory) -> MaskedTrajectory:
    return MaskedTrajectory(
        value=jnp.asarray(left.value) + jnp.asarray(right.value),
        defined=jnp.asarray(left.defined, dtype=jnp.bool_) & jnp.asarray(right.defined, dtype=jnp.bool_),
    )


def _masked_difference(
    left: MaskedTrajectory,
    right: MaskedTrajectory,
) -> MaskedTrajectory:
    return MaskedTrajectory(
        value=jnp.asarray(left.value) - jnp.asarray(right.value),
        defined=jnp.asarray(left.defined, dtype=jnp.bool_) & jnp.asarray(right.defined, dtype=jnp.bool_),
    )


def _independent_trajectories(
    fluxes: Mapping[str, MaskedTrajectory],
    stocks: Mapping[str, MaskedTrajectory],
) -> tuple[Mapping[str, MaskedTrajectory], Mapping[str, MaskedTrajectory]]:
    required_fluxes = {
        "gpp",
        "maintenance_respiration",
        "growth_respiration",
        "heterotrophic_respiration",
    }
    required_stocks = {
        "aboveground_biomass_carbon",
        "belowground_biomass_carbon",
        "total_live_biomass_carbon",
        "ecosystem_carbon",
    }
    if not required_fluxes <= set(fluxes) or not required_stocks <= set(stocks):
        raise ValueError("target-oriented science trajectory registry is incomplete")
    return (
        {
            "gpp": fluxes["gpp"],
            "autotrophic_respiration": _masked_sum(
                fluxes["maintenance_respiration"],
                fluxes["growth_respiration"],
            ),
            "heterotrophic_respiration": fluxes["heterotrophic_respiration"],
        },
        {
            "aboveground_biomass_carbon": stocks["aboveground_biomass_carbon"],
            "belowground_biomass_carbon": stocks["belowground_biomass_carbon"],
            "nonliving_ecosystem_carbon": _masked_difference(
                stocks["ecosystem_carbon"],
                stocks["total_live_biomass_carbon"],
            ),
        },
    )


def science_window_sufficient_statistics(
    predicted_fluxes: Mapping[str, MaskedTrajectory],
    expected_fluxes: Mapping[str, MaskedTrajectory],
    predicted_stocks: Mapping[str, MaskedTrajectory],
    expected_stocks: Mapping[str, MaskedTrajectory],
    *,
    flux_scales: Mapping[str, Any],
    stock_scales: Mapping[str, Any],
    horizon: int,
) -> ScienceWindowSufficientStatistics:
    if int(horizon) not in (7, 30):
        raise ValueError("target-oriented training horizon must be 7 or 30 days")
    if set(flux_scales) != set(INDEPENDENT_FLUX_BASIS):
        raise ValueError("target-oriented independent flux scale registry drift")
    if set(stock_scales) != set(INDEPENDENT_STOCK_BASIS):
        raise ValueError("target-oriented independent stock scale registry drift")
    predicted_flux, predicted_stock = _independent_trajectories(predicted_fluxes, predicted_stocks)
    expected_flux, expected_stock = _independent_trajectories(expected_fluxes, expected_stocks)

    flux_statistics = {}
    for name in INDEPENDENT_FLUX_BASIS:
        predicted = predicted_flux[name]
        expected = expected_flux[name]
        if jnp.asarray(predicted.value).shape[0] < horizon:
            raise ValueError("target-oriented flux trajectory is shorter than horizon")
        defined = jnp.asarray(predicted.defined[:horizon], dtype=jnp.bool_) & jnp.asarray(
            expected.defined[:horizon], dtype=jnp.bool_
        )
        complete = jnp.all(defined, axis=0)
        flux_statistics[name] = normalized_error_sufficient_statistics(
            jnp.sum(jnp.where(defined, predicted.value[:horizon], 0.0), axis=0),
            jnp.sum(jnp.where(defined, expected.value[:horizon], 0.0), axis=0),
            complete,
            horizon * jnp.asarray(flux_scales[name], dtype=jnp.float64),
        )

    stock_statistics = {}
    for name in INDEPENDENT_STOCK_BASIS:
        predicted = predicted_stock[name]
        expected = expected_stock[name]
        if jnp.asarray(predicted.value).shape[0] < horizon:
            raise ValueError("target-oriented stock trajectory is shorter than horizon")
        defined = jnp.asarray(predicted.defined[horizon - 1], dtype=jnp.bool_) & jnp.asarray(
            expected.defined[horizon - 1], dtype=jnp.bool_
        )
        stock_statistics[name] = normalized_error_sufficient_statistics(
            predicted.value[horizon - 1],
            expected.value[horizon - 1],
            defined,
            stock_scales[name],
        )
    return ScienceWindowSufficientStatistics(flux_statistics, stock_statistics)


def add_science_window_sufficient_statistics(
    left: ScienceWindowSufficientStatistics,
    right: ScienceWindowSufficientStatistics,
) -> ScienceWindowSufficientStatistics:
    return ScienceWindowSufficientStatistics(
        {
            name: add_error_sufficient_statistics(left.fluxes[name], right.fluxes[name])
            for name in INDEPENDENT_FLUX_BASIS
        },
        {
            name: add_error_sufficient_statistics(left.stocks[name], right.stocks[name])
            for name in INDEPENDENT_STOCK_BASIS
        },
    )


def science_window_from_sufficient_statistics(
    statistics: ScienceWindowSufficientStatistics,
) -> ScienceWindowResult:
    if set(statistics.fluxes) != set(INDEPENDENT_FLUX_BASIS) or set(statistics.stocks) != set(INDEPENDENT_STOCK_BASIS):
        raise ValueError("target-oriented science statistic registry drift")
    flux_errors = {
        name: normalized_error_from_sufficient_statistics(statistics.fluxes[name]) for name in INDEPENDENT_FLUX_BASIS
    }
    stock_errors = {
        name: normalized_error_from_sufficient_statistics(statistics.stocks[name]) for name in INDEPENDENT_STOCK_BASIS
    }

    def macro(errors: Mapping[str, NormalizedError], attribute: str) -> Any:
        active = jnp.stack(tuple(error.count > 0.0 for error in errors.values()))
        values = jnp.stack(tuple(getattr(error, attribute) for error in errors.values()))
        count = jnp.sum(active, dtype=jnp.float64)
        return jnp.where(
            count > 0.0,
            jnp.sum(jnp.where(active, values, 0.0)) / count,
            0.0,
        )

    flux_mse = macro(flux_errors, "mse")
    flux_bias = macro(flux_errors, "bias_squared")
    stock_mse = macro(stock_errors, "mse")
    stock_bias = macro(stock_errors, "bias_squared")
    return ScienceWindowResult(
        total=0.25 * (flux_mse + flux_bias + stock_mse + stock_bias),
        flux_mse=flux_mse,
        flux_bias_squared=flux_bias,
        stock_mse=stock_mse,
        stock_bias_squared=stock_bias,
        flux_errors=flux_errors,
        stock_errors=stock_errors,
    )


def process_supervision_sufficient_statistics(
    process_loss: ProcessLossResult,
) -> Mapping[str, ErrorSufficientStatistics]:
    if set(process_loss.label_mse) != set(process_loss.label_count):
        raise ValueError("target-oriented process statistic registry drift")
    return {
        name: ErrorSufficientStatistics(
            sum_squared=jnp.asarray(process_loss.label_mse[name], dtype=jnp.float64)
            * jnp.asarray(process_loss.label_count[name], dtype=jnp.float64),
            sum_signed=jnp.asarray(0.0, dtype=jnp.float64),
            count=jnp.asarray(process_loss.label_count[name], dtype=jnp.float64),
        )
        for name in process_loss.label_mse
    }


def necessary_process_supervision_from_sufficient_statistics(
    statistics: Mapping[str, ErrorSufficientStatistics],
    contract: TargetOrientedContract,
) -> NecessaryProcessSupervisionResult:
    label_mse = {name: normalized_error_from_sufficient_statistics(value).mse for name, value in statistics.items()}
    label_count = {name: value.count for name, value in statistics.items()}
    return necessary_process_supervision_loss(
        ProcessLossResult(
            total=jnp.asarray(0.0, dtype=jnp.float64),
            label_mse=label_mse,
            label_count=label_count,
            field_mse={},
            field_count={},
        ),
        contract,
    )


def _reference_identity(reference: Any) -> tuple[str, int, str, str]:
    parent = reference.parent
    return (
        str(parent.landpoint_id),
        int(parent.year),
        str(parent.sha256),
        str(reference.sha256),
    )


def _rank(seed: int, update: int, identity: Sequence[Any]) -> str:
    text = "|".join(str(value) for value in (seed, update, *identity))
    return hashlib.sha256(text.encode()).hexdigest()


def _start_rows(
    reference: Any,
    *,
    seed: int,
    update: int,
    component: str,
    count: int,
    horizon: int,
    exclude_first: bool,
) -> tuple[int, ...]:
    first = 1 if exclude_first else 0
    stop = int(reference.day_count) - int(horizon) + 1
    candidates = range(first, stop)
    identity = _reference_identity(reference)
    ranked = sorted(
        candidates,
        key=lambda row: (_rank(seed, update, (*identity, component, row)), row),
    )
    if len(ranked) < count:
        raise ValueError("target-oriented shard has too few valid start rows")
    return tuple(sorted(ranked[:count]))


def build_target_oriented_training_schedule(
    references: Sequence[Any],
    contract: TargetOrientedTrainingContract,
    *,
    updates: int | None = None,
) -> Mapping[str, Any]:
    data = contract.data
    maximum = int(contract.optimizer["maximum_updates"])
    requested = maximum if updates is None else int(updates)
    if not 1 <= requested <= maximum:
        raise ValueError("target-oriented schedule update count is outside the frozen budget")
    unique = {_reference_identity(reference): reference for reference in references}
    if len(unique) != len(references):
        raise ValueError("target-oriented training references are not unique")
    required = int(data["distinct_point_years_per_update"])
    landpoints = {identity[0] for identity in unique}
    if len(landpoints) < required:
        raise ValueError("target-oriented training requires 28 distinct landpoints")
    seed = int(data["selection_seed"])
    exclude_first = bool(data["exclude_first_shard_row"])
    entries = []
    identities = tuple(unique)
    for update in range(1, requested + 1):
        ranked = sorted(identities, key=lambda value: (_rank(seed, update, value), value))
        selected = []
        seen_landpoints = set()
        for identity in ranked:
            if identity[0] in seen_landpoints:
                continue
            selected.append(identity)
            seen_landpoints.add(identity[0])
            if len(selected) == required:
                break
        if len(selected) != required:
            raise ValueError("target-oriented update cannot satisfy landpoint diversity")

        one_count = int(data["one_day"]["distinct_point_year_count"])
        seven_count = int(data["seven_day"]["distinct_point_year_count"])
        groups = {
            "one_day": selected[:one_count],
            "seven_day": selected[one_count : one_count + seven_count],
            "thirty_day": selected[one_count + seven_count :],
        }
        records = {}
        for component, component_identities in groups.items():
            settings = data[component]
            horizon = int(settings.get("horizon_days", 1))
            count = int(
                settings.get(
                    "samples_per_point_year",
                    settings.get("windows_per_point_year", 1),
                )
            )
            records[component] = [
                {
                    "landpoint_id": identity[0],
                    "year": identity[1],
                    "parent_shard_sha256": identity[2],
                    "typed_shard_sha256": identity[3],
                    "start_rows": list(
                        _start_rows(
                            unique[identity],
                            seed=seed,
                            update=update,
                            component=component,
                            count=count,
                            horizon=horizon,
                            exclude_first=exclude_first,
                        )
                    ),
                }
                for identity in component_identities
            ]
        entries.append({"update": update, "components": records})
    schedule = {
        "schema_version": "target_oriented_training_schedule_v1",
        "training_contract_sha256": contract.sha256,
        "updates": requested,
        "entries": entries,
    }
    schedule["schedule_sha256"] = _canonical_sha256(schedule)
    return schedule


__all__ = [
    "DEFAULT_CONTRACT_PATH",
    "ErrorSufficientStatistics",
    "FROZEN_CONTRACT_SHA256",
    "SCHEMA_VERSION",
    "ScienceWindowSufficientStatistics",
    "TargetOrientedTrainingContract",
    "add_error_sufficient_statistics",
    "add_science_window_sufficient_statistics",
    "build_target_oriented_training_schedule",
    "load_target_oriented_training_contract",
    "merge_error_sufficient_statistics",
    "necessary_process_supervision_from_sufficient_statistics",
    "normalized_error_from_sufficient_statistics",
    "normalized_error_sufficient_statistics",
    "one_day_state_anchor_from_sufficient_statistics",
    "one_day_state_sufficient_statistics",
    "process_supervision_sufficient_statistics",
    "science_window_from_sufficient_statistics",
    "science_window_sufficient_statistics",
]
