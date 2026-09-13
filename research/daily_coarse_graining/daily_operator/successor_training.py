"""Daily rollout and bounded loss calibration for the successor pair."""

from __future__ import annotations

from typing import Any, Callable, Mapping, NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from jax_orchidee.stomate.carbon_kernels import ICARBON
from research.daily_coarse_graining.daily_markov_contract import (
    DailyMarkovContract,
    reconstruct_state_fields_compiled,
)

from .long_horizon_objective import (
    ABOVEGROUND_BIOMASS_PARTS,
    BELOWGROUND_BIOMASS_PARTS,
    DEFAULT_HORIZONS,
    ECOSYSTEM_CARBON_COMPONENTS,
    LongHorizonObjectiveResult,
    MaskedTrajectory,
    derive_area_weighted_biomass_carbon,
    derive_area_weighted_pft_carbon_pool,
    derive_ecosystem_carbon,
    long_horizon_biomass_productivity_objective,
)
from .named_biomass_tendency import (
    NamedBiomassTendencyResult,
    named_biomass_tendency_objective,
)
from .types import DailyOperatorInput, DailyOperatorResult, PFTAxisInput

CUMULATIVE_FLUX_FIELDS = (
    "gpp",
    "npp",
    "maintenance_respiration",
    "growth_respiration",
    "heterotrophic_respiration",
)
TERMINAL_STOCK_FIELDS = (
    "aboveground_biomass_carbon",
    "belowground_biomass_carbon",
    "total_live_biomass_carbon",
    "litter_carbon",
    "soil_particulate_organic_carbon",
    "dissolved_organic_carbon",
    "ecosystem_carbon",
)
LONG_HORIZON_COMPONENTS = (
    "cumulative_flux_mse",
    "cumulative_flux_bias_squared",
    "terminal_stock_mse",
    "terminal_stock_bias_squared",
)
CANDIDATE_OPTIMIZED_FLUX_FIELDS = (
    "gpp",
    "npp",
    "growth_respiration",
    "heterotrophic_respiration",
)
CANDIDATE_OBJECTIVE_COMPONENTS = (
    "cumulative_flux_mse",
    "cumulative_flux_bias_squared",
    "tendency_mse",
    "tendency_bias_squared",
)


class DailyOperatorRolloutTrace(NamedTuple):
    result: DailyOperatorResult
    pft_fraction: Any
    active_pft_mask: Any


class BiomassProductivityTrajectories(NamedTuple):
    fluxes: Mapping[str, MaskedTrajectory]
    stocks: Mapping[str, MaskedTrajectory]


class SuccessorTrainingObjective(NamedTuple):
    total: Any
    one_day_anchor: Any
    long_horizon: LongHorizonObjectiveResult


class CandidateSuccessorTrainingObjective(NamedTuple):
    total: Any
    one_day_anchor: Any
    cumulative_flux_mse: Any
    cumulative_flux_bias_squared: Any
    tendency_mse: Any
    tendency_bias_squared: Any
    long_horizon_diagnostics: LongHorizonObjectiveResult
    named_biomass_tendency: NamedBiomassTendencyResult


class NamedBiomassGradientCalibration(NamedTuple):
    coefficient: float
    median_gradient_norms: Mapping[str, float]
    achieved_stock_to_npp_ratios: Mapping[str, float]
    coefficient_within_bounds: bool
    ratios_within_interval: bool
    passed: bool


class GradientRatioPreflight(NamedTuple):
    split_label: str
    anchor_loss: Any
    component_losses: Mapping[str, Any]
    anchor_gradient_norm: Any
    component_gradient_norms: Mapping[str, Any]
    target_gradient_ratios: Mapping[str, float]
    resolved_coefficients: Mapping[str, float]
    achieved_gradient_ratios: Mapping[str, float]


class GradientRatioPreflightError(ValueError):
    """A bounded coefficient preflight failed with host-readable evidence."""

    def __init__(self, message: str, diagnostic: Mapping[str, Any]):
        super().__init__(message)
        self.diagnostic = dict(diagnostic)


class LongHorizonScales(NamedTuple):
    fluxes: Mapping[str, float]
    stocks: Mapping[str, float]


def _compact_pft_views(
    continuous_state: Any,
    discrete_state: Mapping[str, Any],
    template: PFTAxisInput,
    contract: DailyMarkovContract,
) -> PFTAxisInput:
    fields = reconstruct_state_fields_compiled(
        continuous_state,
        discrete_state,
        contract,
    )
    slow = fields["slowproc_stomate_previous_step_state"]
    active_indices = jnp.asarray(contract.active_pft_indices, dtype=jnp.int32)
    if template.pft_state.shape[0] != active_indices.size:
        raise ValueError("rollout template and Markov active PFT widths differ")
    state_parts = []
    for name in ("veget_max", "lai", "biomass"):
        selected = jnp.take(slow[name], active_indices, axis=1)
        state_parts.append(
            jnp.moveaxis(selected, 1, 0).reshape((active_indices.size, -1))
        )
    active = jnp.asarray(template.active_pft_mask, dtype=jnp.bool_)
    pft_state = jnp.concatenate(tuple(state_parts), axis=1)
    pft_state = jnp.where(active[:, None], pft_state, 0.0)
    fraction = jnp.take(slow["veget_max"], active_indices, axis=1)[0]
    fraction = jnp.where(active, fraction, 0.0)
    return template._replace(pft_state=pft_state, pft_fraction=fraction)


def rollout_input_from_template(
    template: DailyOperatorInput,
    continuous_state: Any,
    discrete_state: Mapping[str, Any],
    contract: DailyMarkovContract,
) -> DailyOperatorInput:
    """Replace every state-derived cached PFT view before one rollout day."""

    continuous_state = jnp.asarray(continuous_state, dtype=jnp.float64)
    if continuous_state.shape != (contract.continuous_state_width,):
        raise ValueError("rollout continuous-state width drift")
    pft = _compact_pft_views(
        continuous_state,
        discrete_state,
        template.pft,
        contract,
    )
    return template._replace(
        continuous_state=continuous_state,
        discrete_state=discrete_state,
        pft=pft,
    )


def rollout_daily_operator(
    parameters: Any,
    templates: DailyOperatorInput,
    *,
    transition: Callable[[Any, DailyOperatorInput], DailyOperatorResult],
    contract: DailyMarkovContract,
    rematerialize: bool = False,
) -> DailyOperatorRolloutTrace:
    """Run one free daily scan without consulting future teacher states."""

    continuous = jnp.asarray(templates.continuous_state, dtype=jnp.float64)
    if continuous.ndim != 2 or continuous.shape[1] != contract.continuous_state_width:
        raise ValueError("rollout templates require [day, continuous_state]")
    if continuous.shape[0] < 1:
        raise ValueError("rollout window cannot be empty")
    initial_discrete = jax.tree_util.tree_map(
        lambda value: jnp.asarray(value)[0],
        templates.discrete_state,
    )

    def step(carry, template):
        current = rollout_input_from_template(template, carry[0], carry[1], contract)
        result = transition(parameters, current)
        trace = DailyOperatorRolloutTrace(
            result=result,
            pft_fraction=current.pft.pft_fraction,
            active_pft_mask=current.pft.active_pft_mask,
        )
        return (result.next_state.other, result.next_discrete_state), trace

    scan_step = jax.checkpoint(step) if rematerialize else step
    _, trace = jax.lax.scan(
        scan_step,
        (continuous[0], initial_discrete),
        templates,
    )
    return trace


def _state_leaf(contract: DailyMarkovContract, field: str) -> Any:
    matches = tuple(
        leaf
        for leaf in contract.state_leaves
        if leaf.component == "slowproc_stomate_previous_step_state"
        and leaf.path == (field,)
        and leaf.start is not None
    )
    if len(matches) != 1:
        raise ValueError(f"long-horizon state field owner drift for {field}")
    return matches[0]


def _positive_l2_scale(value: Any, *, name: str) -> float:
    array = np.asarray(value, dtype=np.float64)
    if array.size < 1 or not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"invalid frozen train/train scale for {name}")
    result = float(np.sqrt(np.sum(array * array, dtype=np.float64)))
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"invalid derived long-horizon scale for {name}")
    return result


def derive_long_horizon_scales(
    next_state_scale: Any,
    fast_gpp_scale: Any,
    contract: DailyMarkovContract,
) -> LongHorizonScales:
    """Propagate accepted train/train column scales to scalar science fields.

    The base statistics do not retain cross-column covariance. Root-sum-square
    propagation therefore supplies one deterministic, positive scale per
    reduced field without rescanning v6 or tuning against rollout results.
    """

    state_scale = np.asarray(next_state_scale, dtype=np.float64)
    if state_scale.shape != (contract.continuous_state_width,):
        raise ValueError("next-state scale width does not match the Markov contract")
    def state_field(field: str) -> np.ndarray:
        leaf = _state_leaf(contract, field)
        value = state_scale[leaf.start : leaf.stop].reshape(leaf.shape)
        if "nvm" not in leaf.axis_names:
            raise ValueError(f"long-horizon state field {field} has no PFT axis")
        pft_axis = leaf.axis_names.index("nvm")
        if value.shape[pft_axis] != len(contract.active_pft_indices):
            raise ValueError(
                f"long-horizon state field {field} is not on the compact PFT axis"
            )
        return value

    fluxes = {
        "gpp": _positive_l2_scale(fast_gpp_scale, name="gpp"),
        "npp": _positive_l2_scale(state_field("npp_daily"), name="npp"),
        "maintenance_respiration": _positive_l2_scale(
            state_field("resp_maint"),
            name="maintenance_respiration",
        ),
        "growth_respiration": _positive_l2_scale(
            state_field("resp_growth"),
            name="growth_respiration",
        ),
        "heterotrophic_respiration": _positive_l2_scale(
            state_field("resp_hetero"),
            name="heterotrophic_respiration",
        ),
    }

    biomass = np.take(state_field("biomass"), ICARBON, axis=-1)
    aboveground = _positive_l2_scale(
        np.take(biomass, ABOVEGROUND_BIOMASS_PARTS, axis=-1),
        name="aboveground_biomass_carbon",
    )
    belowground = _positive_l2_scale(
        np.take(biomass, BELOWGROUND_BIOMASS_PARTS, axis=-1),
        name="belowground_biomass_carbon",
    )
    total_live = _positive_l2_scale(
        biomass,
        name="total_live_biomass_carbon",
    )
    litter = _positive_l2_scale(
        np.concatenate(
            (
                np.take(state_field("litter_above"), ICARBON, axis=-1).reshape(-1),
                np.take(state_field("litter_below"), ICARBON, axis=-1).reshape(-1),
            )
        ),
        name="litter_carbon",
    )
    particulate = _positive_l2_scale(
        state_field("carbon_32l"),
        name="soil_particulate_organic_carbon",
    )
    dissolved = _positive_l2_scale(
        np.take(state_field("DOC"), ICARBON, axis=-1),
        name="dissolved_organic_carbon",
    )
    interception = _positive_l2_scale(
        np.take(state_field("interception_storage"), ICARBON, axis=-1),
        name="canopy_interception_organic_carbon",
    )
    ecosystem = float(
        np.sqrt(
            total_live**2
            + litter**2
            + particulate**2
            + dissolved**2
            + interception**2
        )
    )
    stocks = {
        "aboveground_biomass_carbon": aboveground,
        "belowground_biomass_carbon": belowground,
        "total_live_biomass_carbon": total_live,
        "litter_carbon": litter,
        "soil_particulate_organic_carbon": particulate,
        "dissolved_organic_carbon": dissolved,
        "ecosystem_carbon": ecosystem,
    }
    if tuple(fluxes) != CUMULATIVE_FLUX_FIELDS or tuple(stocks) != TERMINAL_STOCK_FIELDS:
        raise ValueError("derived long-horizon scale registry drift")
    return LongHorizonScales(fluxes=fluxes, stocks=stocks)


def _trajectory_field(
    continuous: Any,
    defined: Any,
    contract: DailyMarkovContract,
    field: str,
) -> tuple[Any, Any]:
    leaf = _state_leaf(contract, field)
    days = continuous.shape[0]
    shape = (days,) + tuple(int(value) for value in leaf.shape)
    return (
        continuous[:, leaf.start : leaf.stop].reshape(shape),
        defined[:, leaf.start : leaf.stop].reshape(shape),
    )


def _normalized_pft_context(
    pft_fraction: Any,
    active_pft_mask: Any,
    *,
    days: int,
    pft_width: int,
) -> tuple[Any, Any]:
    fraction = jnp.asarray(pft_fraction, dtype=jnp.float64)
    active = jnp.asarray(active_pft_mask, dtype=jnp.bool_)
    if fraction.ndim == 2:
        fraction = fraction[:, None, :]
    if active.ndim == 2:
        active = active[:, None, :]
    expected = (days, 1, pft_width)
    if fraction.shape != expected or active.shape != expected:
        raise ValueError(f"trajectory PFT context must have shape {expected}")
    return fraction, active


def _weighted_pft_flux(
    value: Any,
    defined: Any,
    fraction: Any,
    active: Any,
) -> MaskedTrajectory:
    value = jnp.asarray(value, dtype=jnp.float64)
    defined = jnp.asarray(defined, dtype=jnp.bool_)
    if value.shape != fraction.shape or defined.shape != value.shape:
        raise ValueError("PFT flux and area context shapes differ")
    admitted = active & defined
    weighted = jnp.sum(jnp.where(admitted, value * fraction, 0.0), axis=-1)
    output_defined = jnp.all((~active) | defined, axis=-1)
    return MaskedTrajectory(weighted, output_defined)


def _reduced_stock_defined(mask: Any, active: Any, *, pft_axis: int) -> Any:
    mask = jnp.asarray(mask, dtype=jnp.bool_)
    normalized_axis = pft_axis % mask.ndim
    active_shape = [1] * mask.ndim
    active_shape[0] = active.shape[0]
    active_shape[1] = active.shape[1]
    active_shape[normalized_axis] = active.shape[-1]
    relevant = jnp.broadcast_to(jnp.reshape(active, tuple(active_shape)), mask.shape)
    return jnp.all((~relevant) | mask, axis=tuple(range(2, mask.ndim)))


def extract_biomass_productivity_trajectories(
    continuous_state: Any,
    continuous_defined: Any,
    *,
    daily_fast_gpp: Any,
    daily_fast_gpp_defined: Any,
    pft_fraction: Any,
    active_pft_mask: Any,
    contract: DailyMarkovContract,
) -> BiomassProductivityTrajectories:
    """Extract the frozen five fluxes and seven stocks from daily states."""

    continuous = jnp.asarray(continuous_state, dtype=jnp.float64)
    defined = jnp.asarray(continuous_defined, dtype=jnp.bool_)
    if (
        continuous.ndim != 2
        or continuous.shape[1] != contract.continuous_state_width
        or defined.shape != continuous.shape
    ):
        raise ValueError("trajectory state arrays require [day, continuous_state]")
    days = continuous.shape[0]
    pft_width = len(contract.active_pft_indices)
    fraction, active = _normalized_pft_context(
        pft_fraction,
        active_pft_mask,
        days=days,
        pft_width=pft_width,
    )
    gpp = jnp.asarray(daily_fast_gpp, dtype=jnp.float64)
    gpp_defined = jnp.asarray(daily_fast_gpp_defined, dtype=jnp.bool_)
    if gpp.ndim == 2:
        gpp = gpp[:, None, :]
    if gpp_defined.ndim == 2:
        gpp_defined = gpp_defined[:, None, :]

    fluxes: dict[str, MaskedTrajectory] = {
        "gpp": _weighted_pft_flux(gpp, gpp_defined, fraction, active)
    }
    for output_name, state_name in (
        ("npp", "npp_daily"),
        ("maintenance_respiration", "resp_maint"),
        ("growth_respiration", "resp_growth"),
        ("heterotrophic_respiration", "resp_hetero"),
    ):
        value, mask = _trajectory_field(continuous, defined, contract, state_name)
        fluxes[output_name] = _weighted_pft_flux(value, mask, fraction, active)

    biomass, biomass_defined = _trajectory_field(
        continuous, defined, contract, "biomass"
    )
    biomass_stocks = derive_area_weighted_biomass_carbon(
        biomass,
        fraction,
        active,
    )
    biomass_mask = _reduced_stock_defined(biomass_defined, active, pft_axis=2)

    litter_above, litter_above_defined = _trajectory_field(
        continuous, defined, contract, "litter_above"
    )
    litter_below, litter_below_defined = _trajectory_field(
        continuous, defined, contract, "litter_below"
    )
    litter = derive_area_weighted_pft_carbon_pool(
        litter_above,
        fraction,
        active,
        pft_axis=3,
        element_axis=-1,
    ) + derive_area_weighted_pft_carbon_pool(
        litter_below,
        fraction,
        active,
        pft_axis=3,
        element_axis=-1,
    )
    litter_defined = _reduced_stock_defined(
        litter_above_defined, active, pft_axis=3
    ) & _reduced_stock_defined(litter_below_defined, active, pft_axis=3)

    particulate, particulate_defined = _trajectory_field(
        continuous, defined, contract, "carbon_32l"
    )
    particulate_stock = derive_area_weighted_pft_carbon_pool(
        particulate,
        fraction,
        active,
        pft_axis=3,
    )
    particulate_mask = _reduced_stock_defined(
        particulate_defined, active, pft_axis=3
    )

    doc, doc_defined = _trajectory_field(continuous, defined, contract, "DOC")
    doc_stock = derive_area_weighted_pft_carbon_pool(
        doc,
        fraction,
        active,
        pft_axis=2,
        element_axis=-1,
    )
    doc_mask = _reduced_stock_defined(doc_defined, active, pft_axis=2)

    interception, interception_defined = _trajectory_field(
        continuous, defined, contract, "interception_storage"
    )
    interception_stock = derive_area_weighted_pft_carbon_pool(
        interception,
        # This reservoir is already per land area, unlike litter/SOC/DOC.
        # Fortran stomate_soilcarbon.f90::soilcarbon_leak:1225 includes
        # veget_max in deposition;1245-1251 and2326-2332 add storage without it.
        jnp.ones_like(fraction),
        active,
        pft_axis=2,
        element_axis=-1,
    )
    interception_mask = _reduced_stock_defined(
        interception_defined, active, pft_axis=2
    )

    components = {
        "live_biomass": biomass_stocks.total,
        "litter": litter,
        "soil_particulate": particulate_stock,
        "dissolved_organic": doc_stock,
        "canopy_interception_organic": interception_stock,
    }
    if tuple(components) != ECOSYSTEM_CARBON_COMPONENTS:
        raise ValueError("ecosystem-carbon component order drift")
    ecosystem = derive_ecosystem_carbon(components)
    ecosystem_defined = (
        biomass_mask
        & litter_defined
        & particulate_mask
        & doc_mask
        & interception_mask
    )
    stocks = {
        "aboveground_biomass_carbon": MaskedTrajectory(
            biomass_stocks.aboveground, biomass_mask
        ),
        "belowground_biomass_carbon": MaskedTrajectory(
            biomass_stocks.belowground, biomass_mask
        ),
        "total_live_biomass_carbon": MaskedTrajectory(
            biomass_stocks.total, biomass_mask
        ),
        "litter_carbon": MaskedTrajectory(litter, litter_defined),
        "soil_particulate_organic_carbon": MaskedTrajectory(
            particulate_stock, particulate_mask
        ),
        "dissolved_organic_carbon": MaskedTrajectory(doc_stock, doc_mask),
        "ecosystem_carbon": MaskedTrajectory(ecosystem, ecosystem_defined),
    }
    if tuple(fluxes) != CUMULATIVE_FLUX_FIELDS or tuple(stocks) != TERMINAL_STOCK_FIELDS:
        raise ValueError("biomass/productivity trajectory registry drift")
    return BiomassProductivityTrajectories(fluxes=fluxes, stocks=stocks)


def trajectories_from_rollout(
    trace: DailyOperatorRolloutTrace,
    contract: DailyMarkovContract,
) -> BiomassProductivityTrajectories:
    gpp = trace.result.prediction.named_outputs["daily.fast_gpp"]
    gpp_defined = jnp.broadcast_to(
        jnp.asarray(trace.active_pft_mask, dtype=jnp.bool_)[:, None, :],
        jnp.asarray(gpp).shape,
    )
    return extract_biomass_productivity_trajectories(
        trace.result.next_state.other,
        trace.result.next_state_defined.other,
        daily_fast_gpp=gpp,
        daily_fast_gpp_defined=gpp_defined,
        pft_fraction=trace.pft_fraction,
        active_pft_mask=trace.active_pft_mask,
        contract=contract,
    )


def successor_training_objective(
    one_day_anchor: Any,
    predicted: BiomassProductivityTrajectories,
    expected: BiomassProductivityTrajectories,
    *,
    flux_scales: Mapping[str, Any],
    stock_scales: Mapping[str, Any],
    coefficients: Mapping[str, Any],
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> SuccessorTrainingObjective:
    """Join the unit-weight one-day anchor to the calibrated rollout terms."""

    anchor = jnp.asarray(one_day_anchor, dtype=jnp.float64)
    long_horizon = long_horizon_biomass_productivity_objective(
        predicted.fluxes,
        expected.fluxes,
        predicted.stocks,
        expected.stocks,
        flux_scales=flux_scales,
        stock_scales=stock_scales,
        coefficients=coefficients,
        horizons=horizons,
    )
    return SuccessorTrainingObjective(
        total=anchor + long_horizon.total,
        one_day_anchor=anchor,
        long_horizon=long_horizon,
    )


def _selected_flux_macro(
    long_horizon: LongHorizonObjectiveResult,
    attribute: str,
) -> Any:
    horizon_values = []
    for value in long_horizon.by_horizon.values():
        errors = tuple(
            value.cumulative_flux[name]
            for name in CANDIDATE_OPTIMIZED_FLUX_FIELDS
        )
        active = jnp.stack(tuple(error.count > 0.0 for error in errors))
        components = jnp.stack(tuple(getattr(error, attribute) for error in errors))
        count = jnp.sum(active, dtype=jnp.float64)
        horizon_values.append(
            jnp.where(
                count > 0.0,
                jnp.sum(jnp.where(active, components, 0.0), dtype=jnp.float64)
                / jnp.maximum(count, 1.0),
                0.0,
            )
        )
    return jnp.mean(jnp.stack(tuple(horizon_values)))


def candidate_successor_training_objective(
    one_day_anchor: Any,
    predicted: BiomassProductivityTrajectories,
    expected: BiomassProductivityTrajectories,
    *,
    flux_scales: Mapping[str, Any],
    stock_scales: Mapping[str, Any],
    tendency_scales: Mapping[int, Mapping[str, Any]],
    coefficients: Mapping[str, Any],
    horizons: Sequence[int] = DEFAULT_HORIZONS,
) -> CandidateSuccessorTrainingObjective:
    """Join the anchor to selected fluxes and named AGB/BGB tendencies."""

    if tuple(coefficients) != CANDIDATE_OBJECTIVE_COMPONENTS:
        raise ValueError("candidate objective coefficient registry drift")
    anchor = jnp.asarray(one_day_anchor, dtype=jnp.float64)
    diagnostic_coefficients = {name: 0.0 for name in LONG_HORIZON_COMPONENTS}
    long_horizon = long_horizon_biomass_productivity_objective(
        predicted.fluxes,
        expected.fluxes,
        predicted.stocks,
        expected.stocks,
        flux_scales=flux_scales,
        stock_scales=stock_scales,
        coefficients=diagnostic_coefficients,
        horizons=horizons,
    )
    cumulative_flux_mse = _selected_flux_macro(long_horizon, "mse")
    cumulative_flux_bias_squared = _selected_flux_macro(
        long_horizon,
        "bias_squared",
    )
    tendency = named_biomass_tendency_objective(
        predicted.stocks,
        expected.stocks,
        scales=tendency_scales,
        coefficients={"tendency_mse": 1.0, "tendency_bias_squared": 1.0},
        horizons=horizons,
    )
    components = {
        "cumulative_flux_mse": cumulative_flux_mse,
        "cumulative_flux_bias_squared": cumulative_flux_bias_squared,
        "tendency_mse": tendency.tendency_mse,
        "tendency_bias_squared": tendency.tendency_bias_squared,
    }
    total = anchor + sum(
        jnp.asarray(coefficients[name], dtype=jnp.float64) * components[name]
        for name in CANDIDATE_OBJECTIVE_COMPONENTS
    )
    return CandidateSuccessorTrainingObjective(
        total=total,
        one_day_anchor=anchor,
        cumulative_flux_mse=cumulative_flux_mse,
        cumulative_flux_bias_squared=cumulative_flux_bias_squared,
        tendency_mse=tendency.tendency_mse,
        tendency_bias_squared=tendency.tendency_bias_squared,
        long_horizon_diagnostics=long_horizon,
        named_biomass_tendency=tendency,
    )


def calibrate_named_biomass_gradient_coefficient(
    npp_gradient_norms: Sequence[float],
    agb_gradient_norms: Sequence[float],
    bgb_gradient_norms: Sequence[float],
    *,
    target_ratio_interval: tuple[float, float],
    coefficient_interval: tuple[float, float],
) -> NamedBiomassGradientCalibration:
    """Resolve one equal-stock coefficient from train-only median gradients."""

    values = {
        "npp": np.asarray(npp_gradient_norms, dtype=np.float64),
        "aboveground_biomass_carbon": np.asarray(
            agb_gradient_norms,
            dtype=np.float64,
        ),
        "belowground_biomass_carbon": np.asarray(
            bgb_gradient_norms,
            dtype=np.float64,
        ),
    }
    sizes = {array.size for array in values.values()}
    if sizes == {0} or len(sizes) != 1:
        raise ValueError("named biomass gradient samples must be nonempty and aligned")
    if any(np.any(~np.isfinite(array)) or np.any(array <= 0.0) for array in values.values()):
        raise ValueError("named biomass gradient norms must be finite and positive")
    ratio_minimum, ratio_maximum = map(float, target_ratio_interval)
    coefficient_minimum, coefficient_maximum = map(float, coefficient_interval)
    if not 0.0 < ratio_minimum <= ratio_maximum:
        raise ValueError("named biomass gradient ratio interval is invalid")
    if not 0.0 < coefficient_minimum <= coefficient_maximum:
        raise ValueError("named biomass coefficient interval is invalid")

    medians = {name: float(np.median(array)) for name, array in values.items()}
    stock_geometric_mean = float(
        np.sqrt(
            medians["aboveground_biomass_carbon"]
            * medians["belowground_biomass_carbon"]
        )
    )
    coefficient = medians["npp"] / stock_geometric_mean
    achieved = {
        name: coefficient * medians[name] / medians["npp"]
        for name in (
            "aboveground_biomass_carbon",
            "belowground_biomass_carbon",
        )
    }
    coefficient_within_bounds = (
        coefficient_minimum <= coefficient <= coefficient_maximum
    )
    ratios_within_interval = all(
        ratio_minimum <= value <= ratio_maximum for value in achieved.values()
    )
    return NamedBiomassGradientCalibration(
        coefficient=coefficient,
        median_gradient_norms=medians,
        achieved_stock_to_npp_ratios=achieved,
        coefficient_within_bounds=coefficient_within_bounds,
        ratios_within_interval=ratios_within_interval,
        passed=coefficient_within_bounds and ratios_within_interval,
    )


def _tree_l2_norm(tree: Any) -> Any:
    leaves = jax.tree_util.tree_leaves(tree)
    if not leaves:
        return jnp.asarray(0.0, dtype=jnp.float64)
    squared = sum(
        (
            jnp.sum(jnp.asarray(leaf, dtype=jnp.float64) ** 2)
            for leaf in leaves
        ),
        jnp.asarray(0.0, dtype=jnp.float64),
    )
    return jnp.sqrt(squared)


def calibrate_gradient_ratio_coefficients(
    parameters: Any,
    anchor_loss: Callable[[Any], Any],
    component_losses: Mapping[str, Callable[[Any], Any]],
    *,
    target_gradient_ratios: Mapping[str, float],
    coefficient_minimum: float,
    coefficient_maximum: float,
    split_label: str,
    compile_gradients: bool = False,
    combined_component_loss: Callable[[Any], Any] | None = None,
) -> GradientRatioPreflight:
    """Resolve coefficients once from one train/train gradient measurement."""

    if split_label != "train/train":
        raise ValueError("successor coefficient preflight is restricted to train/train")
    if tuple(component_losses) != LONG_HORIZON_COMPONENTS or set(
        target_gradient_ratios
    ) != set(LONG_HORIZON_COMPONENTS):
        raise ValueError("successor gradient-ratio component registry drift")
    if not 0.0 < coefficient_minimum <= coefficient_maximum:
        raise ValueError("successor coefficient interval is invalid")
    anchor_measurement = jax.value_and_grad(anchor_loss)
    if compile_gradients:
        anchor_measurement = jax.jit(anchor_measurement)
    anchor_value, anchor_gradient = anchor_measurement(parameters)
    anchor_norm = _tree_l2_norm(anchor_gradient)
    anchor_norm_host = float(anchor_norm)
    if not jnp.isfinite(anchor_value) or not jnp.isfinite(anchor_norm):
        raise ValueError("one-day anchor loss/gradient is nonfinite")
    if anchor_norm_host <= 0.0:
        raise ValueError("one-day anchor gradient norm must be positive")

    component_names = tuple(LONG_HORIZON_COMPONENTS)

    def stacked_component_losses(current_parameters):
        values = (
            jnp.stack(
                tuple(
                    component_losses[name](current_parameters)
                    for name in component_names
                )
            )
            if combined_component_loss is None
            else jnp.asarray(
                combined_component_loss(current_parameters),
                dtype=jnp.float64,
            )
        )
        if values.shape != (len(component_names),):
            raise ValueError("combined rollout component loss shape drift")
        return values

    def component_measurement(current_parameters):
        values, pullback = jax.vjp(stacked_component_losses, current_parameters)
        basis = jnp.eye(len(component_names), dtype=jnp.float64)
        jacobian = jax.lax.map(lambda seed: pullback(seed)[0], basis)
        return values, jacobian

    if compile_gradients:
        component_measurement = jax.jit(component_measurement)
    measured_values, measured_gradients = component_measurement(parameters)

    component_values = {}
    component_norms = {}
    resolved = {}
    achieved = {}
    for index, name in enumerate(component_names):
        target = float(target_gradient_ratios[name])
        if not 0.0 < target < 1.0:
            raise ValueError(f"invalid target gradient ratio for {name}")
        value = measured_values[index]
        gradient = jax.tree_util.tree_map(
            lambda leaf: leaf[index],
            measured_gradients,
        )
        norm = _tree_l2_norm(gradient)
        norm_host = float(norm)
        if not jnp.isfinite(value) or not jnp.isfinite(norm) or norm_host <= 0.0:
            bad_leaves = []
            element_count = 0
            nonfinite_count = 0
            for path, leaf in jax.tree_util.tree_flatten_with_path(gradient)[0]:
                array = np.asarray(leaf)
                bad_count = int(np.count_nonzero(~np.isfinite(array)))
                element_count += int(array.size)
                nonfinite_count += bad_count
                if bad_count:
                    bad_leaves.append(
                        {
                            "path": jax.tree_util.keystr(path),
                            "shape": list(array.shape),
                            "nonfinite_count": bad_count,
                            "element_count": int(array.size),
                        }
                    )
            value_host = float(value)
            diagnostic = {
                "component": name,
                "loss": value_host if np.isfinite(value_host) else str(value_host),
                "gradient_norm": norm_host if np.isfinite(norm_host) else str(norm_host),
                "gradient_element_count": element_count,
                "gradient_nonfinite_count": nonfinite_count,
                "gradient_nonfinite_leaves": bad_leaves,
            }
            raise GradientRatioPreflightError(
                f"nonfinite or zero rollout gradient for {name}",
                diagnostic,
            )
        coefficient = target * anchor_norm_host / norm_host
        if not coefficient_minimum <= coefficient <= coefficient_maximum:
            raise GradientRatioPreflightError(
                f"resolved coefficient for {name} lies outside the frozen interval",
                {
                    "component": name,
                    "loss": float(value),
                    "anchor_gradient_norm": anchor_norm_host,
                    "component_gradient_norm": norm_host,
                    "target_gradient_ratio": target,
                    "resolved_coefficient": coefficient,
                    "coefficient_minimum": coefficient_minimum,
                    "coefficient_maximum": coefficient_maximum,
                },
            )
        component_values[name] = value
        component_norms[name] = norm
        resolved[name] = coefficient
        achieved[name] = coefficient * norm_host / anchor_norm_host

    return GradientRatioPreflight(
        split_label=split_label,
        anchor_loss=anchor_value,
        component_losses=component_values,
        anchor_gradient_norm=anchor_norm,
        component_gradient_norms=component_norms,
        target_gradient_ratios={
            name: float(target_gradient_ratios[name])
            for name in LONG_HORIZON_COMPONENTS
        },
        resolved_coefficients=resolved,
        achieved_gradient_ratios=achieved,
    )
