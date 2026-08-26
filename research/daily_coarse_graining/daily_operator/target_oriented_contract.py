"""Target-oriented loss and acceptance contract for the daily operator."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, NamedTuple

import jax.numpy as jnp
import numpy as np

from .long_horizon_objective import MaskedTrajectory, NormalizedError
from .neural_architecture import NeuralOutputSchema
from .target_oriented_flux_basis import (
    COMPONENT_RESOLVED_FLUX_BASIS,
    LEGACY_AGGREGATE_FLUX_BASIS,
)
from .target_oriented_flux_basis import (
    INDEPENDENT_STOCK_BASIS as AMENDED_INDEPENDENT_STOCK_BASIS,
)
from .training_loss import ProcessLossResult
from .types import StateDefinedMasks, StateFeatureGroups

SCHEMA_VERSION = "target_oriented_daily_operator_v1"
FROZEN_CONTRACT_SHA256 = "7c9857c0ca57ee65cfcf7514a216a217c3b1661f6cc869c6dfb85d3786bfce5c"
DEFAULT_CONTRACT_PATH = (
    Path(__file__).resolve().parents[3] / "manifests" / "coarse_graining" / "target_oriented_daily_operator_v1.json"
)

PRIMARY_SCIENCE_DIRECT = "primary_science_direct"
STATE_BUDGET_ESSENTIAL = "state_budget_essential"
AUXILIARY_SUPERVISION = "auxiliary_supervision"
EXACT_OR_DIAGNOSTIC = "exact_or_diagnostic"
LEARNED_CATEGORIES = (
    PRIMARY_SCIENCE_DIRECT,
    STATE_BUDGET_ESSENTIAL,
    AUXILIARY_SUPERVISION,
)
# Retain the historical v1 name so its checkpoint and contract remain replayable.
INDEPENDENT_FLUX_BASIS = LEGACY_AGGREGATE_FLUX_BASIS
INDEPENDENT_STOCK_BASIS = AMENDED_INDEPENDENT_STOCK_BASIS
SUPPORTED_FLUX_BASES = (
    INDEPENDENT_FLUX_BASIS,
    COMPONENT_RESOLVED_FLUX_BASIS,
)
BASE_FLUX_FIELDS = (
    "gpp",
    "maintenance_respiration",
    "growth_respiration",
    "heterotrophic_respiration",
)
BASE_STOCK_FIELDS = (
    "aboveground_biomass_carbon",
    "belowground_biomass_carbon",
    "total_live_biomass_carbon",
    "ecosystem_carbon",
)
LOSS_COMPONENTS = (
    "one_day_state_anchor",
    "seven_day_science",
    "thirty_day_science",
    "necessary_process_supervision",
)


def contract_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("contract_sha256", None)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TargetOrientedContract:
    payload: Mapping[str, Any]
    source_path: Path
    sha256: str

    @property
    def exact_fields(self) -> tuple[str, ...]:
        return tuple(str(name) for name in self.payload["output_hierarchy"]["exact_or_diagnostic_fields"])

    @property
    def labels_by_category(self) -> Mapping[str, tuple[str, ...]]:
        hierarchy = self.payload["output_hierarchy"]
        return {
            PRIMARY_SCIENCE_DIRECT: tuple(str(name) for name in hierarchy["primary_science_direct_labels"]),
            STATE_BUDGET_ESSENTIAL: tuple(str(name) for name in hierarchy["state_budget_essential_labels"]),
            AUXILIARY_SUPERVISION: tuple(str(name) for name in hierarchy["auxiliary_supervision_labels"]),
        }

    @property
    def loss_coefficients(self) -> Mapping[str, float]:
        return {str(name): float(value) for name, value in self.payload["training_objective"]["coefficients"].items()}


class OutputHierarchy(NamedTuple):
    learned_fields_by_category: Mapping[str, tuple[str, ...]]
    exact_or_diagnostic_fields: tuple[str, ...]
    learned_field_count: int
    learned_scalar_count: int


class CarbonFluxIdentityResult(NamedTuple):
    autotrophic_respiration: Any
    npp_from_components: Any
    nep: Any
    nee: Any
    npp_identity_residual: Any


class CarbonStockIdentityResult(NamedTuple):
    total_live_biomass_carbon_from_parts: Any
    nonliving_ecosystem_carbon: Any
    total_live_identity_residual: Any


class OneDayStateAnchorResult(NamedTuple):
    total: Any
    group_loss: Mapping[str, Any]
    group_mse: Mapping[str, Any]
    group_bias_squared: Mapping[str, Any]
    group_count: Mapping[str, Any]


class ScienceWindowResult(NamedTuple):
    total: Any
    flux_mse: Any
    flux_bias_squared: Any
    stock_mse: Any
    stock_bias_squared: Any
    flux_errors: Mapping[str, NormalizedError]
    stock_errors: Mapping[str, NormalizedError]


class NecessaryProcessSupervisionResult(NamedTuple):
    total: Any
    category_loss: Mapping[str, Any]
    category_active_label_count: Mapping[str, Any]


class TargetOrientedLossResult(NamedTuple):
    total: Any
    components: Mapping[str, Any]
    weighted_components: Mapping[str, Any]


def _validate_contract_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("target-oriented schema version drift")
    if payload.get("status") != "frozen_before_successor_architecture":
        raise ValueError("target-oriented contract is not frozen")
    dataset = payload.get("dataset", {})
    if dataset.get("release_id") != "pft14-daily-teacher-v6":
        raise ValueError("target-oriented contract is not bound to Teacher v6")
    if any(
        bool(dataset.get(name))
        for name in (
            "teacher_regeneration_allowed",
            "historical_v5_allowed",
            "sealed_test_allowed_for_selection",
        )
    ):
        raise ValueError("target-oriented contract opens a forbidden data path")

    inference = payload.get("inference_constraints", {})
    if (
        inference.get("native_forcing_records_per_day") != 5
        or inference.get("reconstruct_48_forcing_records") is not False
        or inference.get("execute_48_step_state_scan") is not False
    ):
        raise ValueError("target-oriented inference boundary drift")

    hierarchy = payload.get("output_hierarchy", {})
    if (
        hierarchy.get("complete_field_count"),
        hierarchy.get("learned_field_count"),
        hierarchy.get("learned_scalar_count"),
    ) != (181, 161, 39_006):
        raise ValueError("target-oriented output inventory drift")
    category_sets = {
        name: set(values)
        for name, values in TargetOrientedContract(
            payload=payload,
            source_path=Path("."),
            sha256="",
        ).labels_by_category.items()
    }
    if any(not values for values in category_sets.values()):
        raise ValueError("target-oriented learned category is empty")
    union = set().union(*category_sets.values())
    if sum(len(values) for values in category_sets.values()) != len(union):
        raise ValueError("target-oriented label categories overlap")
    if len(union) != 48:
        raise ValueError("target-oriented label hierarchy must partition 48 labels")
    exact = tuple(str(name) for name in hierarchy.get("exact_or_diagnostic_fields", ()))
    if len(exact) != 20 or len(set(exact)) != 20:
        raise ValueError("target-oriented exact/diagnostic inventory drift")
    if hierarchy.get("expected_category_counts") != {
        PRIMARY_SCIENCE_DIRECT: {"field_count": 2, "scalar_count": 26},
        STATE_BUDGET_ESSENTIAL: {"field_count": 93, "scalar_count": 3_682},
        AUXILIARY_SUPERVISION: {"field_count": 66, "scalar_count": 35_298},
        EXACT_OR_DIAGNOSTIC: {"field_count": 20, "scalar_count": 365},
    }:
        raise ValueError("target-oriented category count contract drift")

    identities = payload.get("carbon_identities", {})
    if tuple(identities.get("optimization_flux_basis", ())) != INDEPENDENT_FLUX_BASIS:
        raise ValueError("target-oriented independent flux basis drift")
    if tuple(identities.get("optimization_stock_basis", ())) != INDEPENDENT_STOCK_BASIS:
        raise ValueError("target-oriented independent stock basis drift")
    if identities.get("nee_sign_convention") != "positive_to_atmosphere":
        raise ValueError("target-oriented NEE sign convention drift")

    objective = payload.get("training_objective", {})
    if tuple(objective.get("components", ())) != LOSS_COMPONENTS:
        raise ValueError("target-oriented loss component order drift")
    coefficients = objective.get("coefficients", {})
    if tuple(coefficients) != LOSS_COMPONENTS or {str(name): float(value) for name, value in coefficients.items()} != {
        "one_day_state_anchor": 1.0,
        "seven_day_science": 1.0,
        "thirty_day_science": 1.0,
        "necessary_process_supervision": 0.1,
    }:
        raise ValueError("target-oriented loss coefficients drift")
    if tuple(objective.get("one_day_state_anchor", {}).get("groups", ())) != tuple(StateFeatureGroups._fields):
        raise ValueError("target-oriented one-day state groups drift")
    process_weights = objective.get("necessary_process_supervision", {})
    if (
        process_weights.get("primary_science_direct_label_weight"),
        process_weights.get("state_budget_essential_label_weight"),
        process_weights.get("auxiliary_supervision_label_weight"),
        process_weights.get("exact_or_diagnostic_weight"),
    ) != (0.0, 1.0, 0.25, 0.0):
        raise ValueError("target-oriented process-supervision weights drift")

    acceptance = payload.get("acceptance", {})
    if acceptance.get("absolute_not_control_relative") is not True:
        raise ValueError("target-oriented acceptance must be absolute")
    short = acceptance.get("short_horizon", {})
    expected_short = {
        "7": (0.35, 0.10),
        "30": (0.25, 0.07),
    }
    for horizon, (rmse, bias) in expected_short.items():
        gate = short.get(horizon, {})
        if (
            gate.get("normalized_rmse_maximum"),
            gate.get("absolute_normalized_signed_bias_maximum"),
        ) != (rmse, bias):
            raise ValueError(f"target-oriented {horizon}-day gate drift")
    annual = acceptance.get("annual_365", {})
    if (
        annual.get("annual_integrated_flux_median_relative_bias_maximum"),
        annual.get("annual_integrated_flux_p95_relative_bias_maximum"),
        annual.get("year_end_stock_median_relative_error_maximum"),
        annual.get("year_end_stock_p95_relative_error_maximum"),
    ) != (0.02, 0.05, 0.02, 0.05):
        raise ValueError("target-oriented annual gross-flux/stock gate drift")
    hard = acceptance.get("hard_gates", {})
    if (
        hard.get("budget_residual_absolute_tolerance"),
        hard.get("stock_negative_absolute_tolerance"),
    ) != (2.0e-12, 2.0e-12):
        raise ValueError("target-oriented hard physical gate drift")


def load_target_oriented_contract(
    path: str | Path = DEFAULT_CONTRACT_PATH,
) -> TargetOrientedContract:
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    _validate_contract_payload(payload)
    observed = contract_sha256(payload)
    if payload.get("contract_sha256") != observed:
        raise ValueError("target-oriented contract SHA256 drift")
    if observed != FROZEN_CONTRACT_SHA256:
        raise ValueError("target-oriented contract is not the frozen v1")
    return TargetOrientedContract(payload=payload, source_path=source, sha256=observed)


def resolve_target_output_hierarchy(
    learned_schema: NeuralOutputSchema,
    complete_schema: NeuralOutputSchema,
    contract: TargetOrientedContract,
) -> OutputHierarchy:
    hierarchy = contract.payload["output_hierarchy"]
    if len(complete_schema.fields) != int(hierarchy["complete_field_count"]):
        raise ValueError("target-oriented complete schema count drift")
    if len(learned_schema.fields) != int(hierarchy["learned_field_count"]):
        raise ValueError("target-oriented learned schema count drift")
    scalar_count = sum(int(np.prod(field.shape, dtype=np.int64)) for field in learned_schema.fields)
    if scalar_count != int(hierarchy["learned_scalar_count"]):
        raise ValueError("target-oriented learned scalar count drift")
    if tuple(learned_schema.labels) != tuple(complete_schema.labels):
        raise ValueError("target-oriented learned/complete label registry drift")

    complete_names = {field.name for field in complete_schema.fields}
    learned_names = {field.name for field in learned_schema.fields}
    exact_names = set(contract.exact_fields)
    if learned_names & exact_names or learned_names | exact_names != complete_names:
        raise ValueError("target-oriented learned/exact field partition drift")

    labels = set(complete_schema.labels)
    declared_labels = set().union(*(set(values) for values in contract.labels_by_category.values()))
    if labels != declared_labels:
        missing = sorted(labels - declared_labels)
        extra = sorted(declared_labels - labels)
        raise ValueError(f"target-oriented label registry drift; missing={missing}, extra={extra}")
    label_category = {
        label: category
        for category, category_labels in contract.labels_by_category.items()
        for label in category_labels
    }
    fields_by_category: dict[str, list[str]] = {category: [] for category in LEARNED_CATEGORIES}
    for field in learned_schema.fields:
        label = learned_schema.labels[field.label_index]
        fields_by_category[label_category[label]].append(field.name)
    if sum(len(values) for values in fields_by_category.values()) != len(learned_schema.fields):
        raise ValueError("target-oriented learned field classification is incomplete")
    complete_by_name = {field.name: field for field in complete_schema.fields}
    expected_counts = hierarchy["expected_category_counts"]
    for category, names in fields_by_category.items():
        scalar_count_by_category = sum(int(np.prod(complete_by_name[name].shape, dtype=np.int64)) for name in names)
        expected = expected_counts[category]
        if (len(names), scalar_count_by_category) != (
            int(expected["field_count"]),
            int(expected["scalar_count"]),
        ):
            raise ValueError(f"target-oriented {category} count drift")
    exact_scalar_count = sum(int(np.prod(complete_by_name[name].shape, dtype=np.int64)) for name in exact_names)
    expected_exact = expected_counts[EXACT_OR_DIAGNOSTIC]
    if (len(exact_names), exact_scalar_count) != (
        int(expected_exact["field_count"]),
        int(expected_exact["scalar_count"]),
    ):
        raise ValueError("target-oriented exact/diagnostic count drift")
    return OutputHierarchy(
        learned_fields_by_category={category: tuple(values) for category, values in fields_by_category.items()},
        exact_or_diagnostic_fields=contract.exact_fields,
        learned_field_count=len(learned_schema.fields),
        learned_scalar_count=scalar_count,
    )


def derive_carbon_flux_identities(
    fluxes: Mapping[str, Any],
) -> CarbonFluxIdentityResult:
    required = set(BASE_FLUX_FIELDS) | {"npp"}
    if set(fluxes) != required:
        missing = sorted(required - set(fluxes))
        extra = sorted(set(fluxes) - required)
        raise ValueError(f"carbon flux identity registry drift; missing={missing}, extra={extra}")
    values = {name: jnp.asarray(fluxes[name], dtype=jnp.float64) for name in required}
    shape = values[BASE_FLUX_FIELDS[0]].shape
    if any(value.shape != shape for value in values.values()):
        raise ValueError("carbon flux identity shapes differ")
    autotrophic = values["maintenance_respiration"] + values["growth_respiration"]
    npp_from_components = values["gpp"] - autotrophic
    nep = npp_from_components - values["heterotrophic_respiration"]
    return CarbonFluxIdentityResult(
        autotrophic_respiration=autotrophic,
        npp_from_components=npp_from_components,
        nep=nep,
        nee=-nep,
        npp_identity_residual=values["npp"] - npp_from_components,
    )


def derive_carbon_stock_identities(
    stocks: Mapping[str, Any],
) -> CarbonStockIdentityResult:
    required = set(BASE_STOCK_FIELDS)
    if set(stocks) != required:
        missing = sorted(required - set(stocks))
        extra = sorted(set(stocks) - required)
        raise ValueError(f"carbon stock identity registry drift; missing={missing}, extra={extra}")
    values = {name: jnp.asarray(stocks[name], dtype=jnp.float64) for name in required}
    shape = values[BASE_STOCK_FIELDS[0]].shape
    if any(value.shape != shape for value in values.values()):
        raise ValueError("carbon stock identity shapes differ")
    total = values["aboveground_biomass_carbon"] + values["belowground_biomass_carbon"]
    return CarbonStockIdentityResult(
        total_live_biomass_carbon_from_parts=total,
        nonliving_ecosystem_carbon=values["ecosystem_carbon"] - total,
        total_live_identity_residual=values["total_live_biomass_carbon"] - total,
    )


def _normalized_error(
    predicted: Any,
    expected: Any,
    defined: Any,
    scale: Any,
) -> NormalizedError:
    predicted = jnp.asarray(predicted, dtype=jnp.float64)
    expected = jnp.asarray(expected, dtype=jnp.float64)
    defined = jnp.asarray(defined, dtype=jnp.bool_)
    scale = jnp.broadcast_to(jnp.asarray(scale, dtype=jnp.float64), predicted.shape)
    if predicted.shape != expected.shape or defined.shape != predicted.shape:
        raise ValueError("normalized target-oriented arrays differ")
    valid = defined & jnp.isfinite(predicted) & jnp.isfinite(expected) & jnp.isfinite(scale) & (scale > 0.0)
    safe_predicted = jnp.where(valid, predicted, 0.0)
    safe_expected = jnp.where(valid, expected, 0.0)
    safe_scale = jnp.where(valid, scale, 1.0)
    residual = (safe_predicted - safe_expected) / safe_scale
    count = jnp.sum(valid, dtype=jnp.float64)
    mean = jnp.where(count > 0.0, jnp.sum(residual) / count, 0.0)
    mse = jnp.where(
        count > 0.0,
        jnp.sum(residual * residual, dtype=jnp.float64) / count,
        0.0,
    )
    return NormalizedError(mse=mse, mean_bias=mean, bias_squared=mean * mean, count=count)


def one_day_state_anchor(
    predicted: StateFeatureGroups,
    expected: StateFeatureGroups,
    predicted_defined: StateDefinedMasks,
    expected_defined: StateDefinedMasks,
    scales: StateFeatureGroups,
    *,
    eligible: StateDefinedMasks | None = None,
) -> OneDayStateAnchorResult:
    if eligible is None:
        eligible = StateDefinedMasks(
            **{
                name: jnp.ones_like(
                    jnp.asarray(getattr(expected_defined, name), dtype=jnp.bool_),
                    dtype=jnp.bool_,
                )
                for name in StateFeatureGroups._fields
            }
        )
    group_mse: dict[str, Any] = {}
    group_bias: dict[str, Any] = {}
    group_count: dict[str, Any] = {}
    group_loss: dict[str, Any] = {}
    for name in StateFeatureGroups._fields:
        error = _normalized_error(
            getattr(predicted, name),
            getattr(expected, name),
            jnp.asarray(getattr(predicted_defined, name), dtype=jnp.bool_)
            & jnp.asarray(getattr(expected_defined, name), dtype=jnp.bool_)
            & jnp.asarray(getattr(eligible, name), dtype=jnp.bool_),
            getattr(scales, name),
        )
        group_mse[name] = error.mse
        group_bias[name] = error.bias_squared
        group_count[name] = error.count
        group_loss[name] = 0.5 * (error.mse + error.bias_squared)
    active = jnp.stack(tuple(group_count[name] > 0.0 for name in StateFeatureGroups._fields))
    values = jnp.stack(tuple(group_loss[name] for name in StateFeatureGroups._fields))
    count = jnp.sum(active, dtype=jnp.float64)
    total = jnp.where(
        count > 0.0,
        jnp.sum(jnp.where(active, values, 0.0)) / count,
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


def _masked_difference(left: MaskedTrajectory, right: MaskedTrajectory) -> MaskedTrajectory:
    return MaskedTrajectory(
        value=jnp.asarray(left.value) - jnp.asarray(right.value),
        defined=jnp.asarray(left.defined, dtype=jnp.bool_) & jnp.asarray(right.defined, dtype=jnp.bool_),
    )


def _independent_science_trajectories(
    fluxes: Mapping[str, MaskedTrajectory],
    stocks: Mapping[str, MaskedTrajectory],
    *,
    flux_basis: tuple[str, ...] = INDEPENDENT_FLUX_BASIS,
) -> tuple[Mapping[str, MaskedTrajectory], Mapping[str, MaskedTrajectory]]:
    if not set(BASE_FLUX_FIELDS) <= set(fluxes):
        raise ValueError("target-oriented science flux registry is incomplete")
    if not set(BASE_STOCK_FIELDS) <= set(stocks):
        raise ValueError("target-oriented science stock registry is incomplete")
    basis = tuple(flux_basis)
    if basis not in SUPPORTED_FLUX_BASES:
        raise ValueError("target-oriented science flux basis is unsupported")
    available_fluxes = {
        "gpp": fluxes["gpp"],
        "maintenance_respiration": fluxes["maintenance_respiration"],
        "growth_respiration": fluxes["growth_respiration"],
        "autotrophic_respiration": _masked_sum(
            fluxes["maintenance_respiration"],
            fluxes["growth_respiration"],
        ),
        "heterotrophic_respiration": fluxes["heterotrophic_respiration"],
    }
    independent_fluxes = {name: available_fluxes[name] for name in basis}
    independent_stocks = {
        "aboveground_biomass_carbon": stocks["aboveground_biomass_carbon"],
        "belowground_biomass_carbon": stocks["belowground_biomass_carbon"],
        "nonliving_ecosystem_carbon": _masked_difference(
            stocks["ecosystem_carbon"],
            stocks["total_live_biomass_carbon"],
        ),
    }
    return independent_fluxes, independent_stocks


def _macro_error(
    errors: Mapping[str, NormalizedError],
    attribute: str,
) -> Any:
    active = jnp.stack(tuple(error.count > 0.0 for error in errors.values()))
    values = jnp.stack(tuple(getattr(error, attribute) for error in errors.values()))
    count = jnp.sum(active, dtype=jnp.float64)
    return jnp.where(
        count > 0.0,
        jnp.sum(jnp.where(active, values, 0.0)) / count,
        0.0,
    )


def science_window_objective(
    predicted_fluxes: Mapping[str, MaskedTrajectory],
    expected_fluxes: Mapping[str, MaskedTrajectory],
    predicted_stocks: Mapping[str, MaskedTrajectory],
    expected_stocks: Mapping[str, MaskedTrajectory],
    *,
    flux_scales: Mapping[str, Any],
    stock_scales: Mapping[str, Any],
    horizon: int,
    flux_basis: tuple[str, ...] = INDEPENDENT_FLUX_BASIS,
) -> ScienceWindowResult:
    if int(horizon) not in (7, 30):
        raise ValueError("target-oriented training horizon must be 7 or 30 days")
    basis = tuple(flux_basis)
    predicted_flux, predicted_stock = _independent_science_trajectories(
        predicted_fluxes,
        predicted_stocks,
        flux_basis=basis,
    )
    expected_flux, expected_stock = _independent_science_trajectories(
        expected_fluxes,
        expected_stocks,
        flux_basis=basis,
    )
    if set(flux_scales) != set(basis):
        raise ValueError("target-oriented independent flux scale registry drift")
    if set(stock_scales) != set(INDEPENDENT_STOCK_BASIS):
        raise ValueError("target-oriented independent stock scale registry drift")

    flux_errors: dict[str, NormalizedError] = {}
    for name in basis:
        predicted = predicted_flux[name]
        expected = expected_flux[name]
        if jnp.asarray(predicted.value).shape[0] < horizon:
            raise ValueError("target-oriented flux trajectory is shorter than horizon")
        defined = jnp.asarray(predicted.defined[:horizon], dtype=jnp.bool_) & jnp.asarray(
            expected.defined[:horizon], dtype=jnp.bool_
        )
        complete = jnp.all(defined, axis=0)
        flux_errors[name] = _normalized_error(
            jnp.sum(jnp.where(defined, predicted.value[:horizon], 0.0), axis=0),
            jnp.sum(jnp.where(defined, expected.value[:horizon], 0.0), axis=0),
            complete,
            horizon * jnp.asarray(flux_scales[name], dtype=jnp.float64),
        )

    stock_errors: dict[str, NormalizedError] = {}
    for name in INDEPENDENT_STOCK_BASIS:
        predicted = predicted_stock[name]
        expected = expected_stock[name]
        if jnp.asarray(predicted.value).shape[0] < horizon:
            raise ValueError("target-oriented stock trajectory is shorter than horizon")
        defined = jnp.asarray(predicted.defined[horizon - 1], dtype=jnp.bool_) & jnp.asarray(
            expected.defined[horizon - 1], dtype=jnp.bool_
        )
        stock_errors[name] = _normalized_error(
            predicted.value[horizon - 1],
            expected.value[horizon - 1],
            defined,
            stock_scales[name],
        )

    flux_mse = _macro_error(flux_errors, "mse")
    flux_bias = _macro_error(flux_errors, "bias_squared")
    stock_mse = _macro_error(stock_errors, "mse")
    stock_bias = _macro_error(stock_errors, "bias_squared")
    total = 0.25 * (flux_mse + flux_bias + stock_mse + stock_bias)
    return ScienceWindowResult(
        total=total,
        flux_mse=flux_mse,
        flux_bias_squared=flux_bias,
        stock_mse=stock_mse,
        stock_bias_squared=stock_bias,
        flux_errors=flux_errors,
        stock_errors=stock_errors,
    )


def necessary_process_supervision_loss(
    process_loss: ProcessLossResult,
    contract: TargetOrientedContract,
) -> NecessaryProcessSupervisionResult:
    labels_by_category = contract.labels_by_category
    expected = set().union(*(set(values) for values in labels_by_category.values()))
    if set(process_loss.label_mse) != expected or set(process_loss.label_count) != expected:
        raise ValueError("target-oriented process loss label registry drift")
    raw_weights = contract.payload["training_objective"]["necessary_process_supervision"]
    weights = {
        PRIMARY_SCIENCE_DIRECT: float(raw_weights["primary_science_direct_label_weight"]),
        STATE_BUDGET_ESSENTIAL: float(raw_weights["state_budget_essential_label_weight"]),
        AUXILIARY_SUPERVISION: float(raw_weights["auxiliary_supervision_label_weight"]),
    }
    category_loss: dict[str, Any] = {}
    category_count: dict[str, Any] = {}
    numerator = jnp.asarray(0.0, dtype=jnp.float64)
    denominator = jnp.asarray(0.0, dtype=jnp.float64)
    for category in LEARNED_CATEGORIES:
        labels = labels_by_category[category]
        active = jnp.stack(tuple(jnp.asarray(process_loss.label_count[label]) > 0.0 for label in labels))
        values = jnp.stack(tuple(process_loss.label_mse[label] for label in labels))
        count = jnp.sum(active, dtype=jnp.float64)
        value = jnp.where(
            count > 0.0,
            jnp.sum(jnp.where(active, values, 0.0)) / count,
            0.0,
        )
        category_loss[category] = value
        category_count[category] = count
        weight = jnp.asarray(weights[category], dtype=jnp.float64)
        numerator = numerator + weight * jnp.sum(jnp.where(active, values, 0.0))
        denominator = denominator + weight * count
    total = jnp.where(denominator > 0.0, numerator / denominator, 0.0)
    return NecessaryProcessSupervisionResult(
        total=total,
        category_loss=category_loss,
        category_active_label_count=category_count,
    )


def compose_target_oriented_loss(
    *,
    one_day_state_anchor_value: Any,
    seven_day_science: Any,
    thirty_day_science: Any,
    necessary_process_supervision: Any,
    contract: TargetOrientedContract,
) -> TargetOrientedLossResult:
    components = {
        "one_day_state_anchor": jnp.asarray(one_day_state_anchor_value, dtype=jnp.float64),
        "seven_day_science": jnp.asarray(seven_day_science, dtype=jnp.float64),
        "thirty_day_science": jnp.asarray(thirty_day_science, dtype=jnp.float64),
        "necessary_process_supervision": jnp.asarray(necessary_process_supervision, dtype=jnp.float64),
    }
    weighted = {name: contract.loss_coefficients[name] * components[name] for name in LOSS_COMPONENTS}
    total = sum(weighted.values(), jnp.asarray(0.0, dtype=jnp.float64))
    return TargetOrientedLossResult(
        total=total,
        components=components,
        weighted_components=weighted,
    )


__all__ = [
    "AUXILIARY_SUPERVISION",
    "CarbonFluxIdentityResult",
    "CarbonStockIdentityResult",
    "COMPONENT_RESOLVED_FLUX_BASIS",
    "DEFAULT_CONTRACT_PATH",
    "EXACT_OR_DIAGNOSTIC",
    "FROZEN_CONTRACT_SHA256",
    "INDEPENDENT_FLUX_BASIS",
    "INDEPENDENT_STOCK_BASIS",
    "NecessaryProcessSupervisionResult",
    "OneDayStateAnchorResult",
    "OutputHierarchy",
    "PRIMARY_SCIENCE_DIRECT",
    "STATE_BUDGET_ESSENTIAL",
    "ScienceWindowResult",
    "SUPPORTED_FLUX_BASES",
    "TargetOrientedContract",
    "TargetOrientedLossResult",
    "compose_target_oriented_loss",
    "contract_sha256",
    "derive_carbon_flux_identities",
    "derive_carbon_stock_identities",
    "load_target_oriented_contract",
    "necessary_process_supervision_loss",
    "one_day_state_anchor",
    "resolve_target_output_hierarchy",
    "science_window_objective",
]
