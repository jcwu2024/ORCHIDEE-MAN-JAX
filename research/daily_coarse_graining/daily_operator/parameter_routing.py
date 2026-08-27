"""Ownership-aware routing for physical PFT parameters.

The canonical operator input carries one complete, named parameter matrix so
that a single day can be replayed or differentiated. That matrix is not a
license to expose every column to a neural encoder. This module derives the
two disjoint ownership views frozen by the Gate-B2 contract and routes their
members only to named scientific consumers:

* explicit fast-process factors used around learned environmental responses;
* parameters passed directly to source-backed retained daily processes.

Ownership classes remain disjoint, but process consumption is not necessarily
single-route. In particular, ``vcmax25`` is both an explicit fast GPP factor
and an input to the retained daily ``vmax_step`` state update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, NamedTuple

import jax.numpy as jnp

from jax_orchidee.stomate.carbon_kernels import (
    IAGRHRTPN,
    IAGRHRTST,
    IAGRSAPPN,
    IAGRSAPST,
    ICARBRES,
    IFRUIT,
    IHEARTABOVE,
    IHEARTBELOW,
    ILEAF,
    IROOT,
    ISAPABOVE,
    ISAPBELOW,
    root_temperature,
)
from research.daily_coarse_graining.daily_pft_interface import (
    DailyPFTInterfaceContract,
    load_daily_pft_interface_contract,
)

from .process_derivations import (
    NativeForcingFieldLayout,
    derive_native_daily_deterministic_fields,
)

FAST_PROCESS_FACTOR = "explicit_parameterized_fast_process_factor"
RETAINED_EXACT = "retained_exact"
PFT_TRAIT_BARE_SOIL = 0
PFT_TRAIT_IS_TREE = 2
VCMAX25_FACTOR_SCALE = 100.0


@dataclass(frozen=True)
class PFTParameterRouting:
    """Static column ownership derived from the named PFT contract."""

    channel_names: tuple[str, ...]
    fast_process_factor_indices: tuple[int, ...]
    retained_exact_indices: tuple[int, ...]

    def index(self, name: str) -> int:
        try:
            return self.channel_names.index(name)
        except ValueError as exc:
            raise ValueError(f"unknown PFT parameter channel {name!r}") from exc


class FastProcessParameterFactors(NamedTuple):
    vcmax25: Any
    maintenance_temperature_slope: Any


class RetainedExactPFTParameters(NamedTuple):
    alloc_min: Any
    residence_time: Any


def build_pft_parameter_routing(
    contract: DailyPFTInterfaceContract,
) -> PFTParameterRouting:
    """Build disjoint column sets and fail closed on unowned channels."""

    supported = {FAST_PROCESS_FACTOR, RETAINED_EXACT}
    unsupported = tuple(
        channel.name for channel in contract.parameter_channels if channel.ownership_class not in supported
    )
    if unsupported:
        raise ValueError(
            f"the current daily operator has no admitted learned-conditional parameter route: {unsupported}"
        )
    fast = tuple(
        channel.index for channel in contract.parameter_channels if channel.ownership_class == FAST_PROCESS_FACTOR
    )
    retained = tuple(
        channel.index for channel in contract.parameter_channels if channel.ownership_class == RETAINED_EXACT
    )
    if set(fast) & set(retained) or sorted(fast + retained) != list(range(len(contract.parameter_channels))):
        raise ValueError("PFT parameter ownership must be complete and disjoint")
    return PFTParameterRouting(
        channel_names=contract.parameter_names,
        fast_process_factor_indices=fast,
        retained_exact_indices=retained,
    )


DEFAULT_PFT_PARAMETER_ROUTING = build_pft_parameter_routing(load_daily_pft_interface_contract())


def _validated_parameter_matrix(value: Any, routing: PFTParameterRouting) -> Any:
    array = jnp.asarray(value, dtype=jnp.float64)
    if array.ndim != 2 or array.shape[1] != len(routing.channel_names):
        raise ValueError(f"pft_parameters must have shape (n_pft, {len(routing.channel_names)})")
    return array


def select_fast_process_factor_channels(
    pft_parameters: Any,
    *,
    routing: PFTParameterRouting = DEFAULT_PFT_PARAMETER_ROUTING,
) -> Any:
    """Return only parameters owned by replaced fast processes."""

    values = _validated_parameter_matrix(pft_parameters, routing)
    return jnp.take(
        values,
        jnp.asarray(routing.fast_process_factor_indices, dtype=jnp.int32),
        axis=1,
    )


def select_retained_exact_channels(
    pft_parameters: Any,
    *,
    routing: PFTParameterRouting = DEFAULT_PFT_PARAMETER_ROUTING,
) -> Any:
    """Return only parameters consumed by retained source-backed formulas."""

    values = _validated_parameter_matrix(pft_parameters, routing)
    return jnp.take(
        values,
        jnp.asarray(routing.retained_exact_indices, dtype=jnp.int32),
        axis=1,
    )


def fast_process_parameter_factors(
    pft_parameters: Any,
    temperature_celsius: Any,
    *,
    routing: PFTParameterRouting = DEFAULT_PFT_PARAMETER_ROUTING,
) -> FastProcessParameterFactors:
    """Evaluate the named VCMAX25 and maintenance-temperature factors.

    Fortran provenance for the maintenance polynomial is
    ``src_parameters/pft_parameters.f90::config_stomate_pft_parameters``
    lines 4220-4242. The learned architecture still owns the environmental
    modifiers; this function owns only the explicit physical dependence.
    """

    values = _validated_parameter_matrix(pft_parameters, routing)
    temperature = jnp.asarray(temperature_celsius, dtype=jnp.float64)
    if temperature.shape != values.shape[:1]:
        raise ValueError("temperature_celsius must contain one value per PFT")
    vcmax25 = values[:, routing.index("vcmax25")]
    slope_a = values[:, routing.index("maint_resp_slope_a")]
    slope_b = values[:, routing.index("maint_resp_slope_b")]
    slope_c = values[:, routing.index("maint_resp_slope_c")]
    slope = slope_c + slope_b * temperature + slope_a * temperature**2
    return FastProcessParameterFactors(
        vcmax25=vcmax25,
        maintenance_temperature_slope=slope,
    )


def apply_fast_parameter_routes(
    outputs: Mapping[str, Any],
    inputs: Any,
    layout: Any,
    forcing_layout: NativeForcingFieldLayout,
    *,
    maintenance_already_routed: bool = False,
) -> dict[str, Any]:
    """Apply the shared explicit VCMAX25 and maintenance parameter factors."""

    result = dict(outputs)
    pft = inputs.pft
    active = jnp.asarray(pft.active_pft_mask, dtype=jnp.bool_)
    traits = jnp.asarray(pft.pft_traits, dtype=jnp.float64)
    fraction = jnp.asarray(pft.pft_fraction, dtype=jnp.float64)
    bare = active & (fraction > 0.0) & (traits[:, PFT_TRAIT_BARE_SOIL] > 0.5)
    vegetation = active & (fraction > 0.0) & ~bare

    continuous = jnp.asarray(inputs.canonical_input.continuous_state, dtype=jnp.float64)
    tlong_k = continuous[layout.t2m_longterm_start : layout.t2m_longterm_stop][0]
    factors = fast_process_parameter_factors(
        pft.pft_parameters,
        jnp.broadcast_to(tlong_k - 273.15, active.shape),
    )
    base_gpp = jnp.asarray(result["daily.fast_gpp"], dtype=jnp.float64)[0]
    gpp = base_gpp * jnp.maximum(factors.vcmax25, 0.0) / VCMAX25_FACTOR_SCALE
    result["daily.fast_gpp"] = jnp.where(vegetation, gpp, 0.0)[None, :]

    if maintenance_already_routed:
        return result

    deterministic = derive_native_daily_deterministic_fields(
        inputs.forcing,
        forcing_layout,
        model_interval_seconds=layout.model_interval_seconds,
    )
    air_k = jnp.asarray(deterministic.t2m_daily, dtype=jnp.float64).reshape(-1)[0]
    soil_k = jnp.asarray(result["daily.temperature_context.tsoil"], dtype=jnp.float64)
    root_k = root_temperature(
        soil_k[:, None, :],
        jnp.asarray(inputs.process_static.z_soil, dtype=jnp.float64),
        jnp.asarray(inputs.process_static.rprof, dtype=jnp.float64),
    )[0]
    part_temperature = jnp.broadcast_to(air_k, (active.shape[0], 12))
    below = jnp.asarray([ISAPBELOW, IHEARTBELOW, IROOT], dtype=jnp.int32)
    part_temperature = part_temperature.at[:, below].set(root_k[:, None])
    is_tree = traits[:, PFT_TRAIT_IS_TREE] > 0.5
    reserve_temperature = jnp.where(is_tree, air_k, root_k)
    part_temperature = part_temperature.at[:, ICARBRES].set(reserve_temperature)
    above = jnp.asarray(
        [
            ILEAF,
            ISAPABOVE,
            IHEARTABOVE,
            IFRUIT,
            IAGRSAPST,
            IAGRSAPPN,
            IAGRHRTST,
            IAGRHRTPN,
        ],
        dtype=jnp.int32,
    )
    part_temperature = part_temperature.at[:, above].set(air_k)
    multiplier = jnp.maximum(
        1.0 + factors.maintenance_temperature_slope[:, None] * (part_temperature - 273.15),
        0.0,
    )
    base_maintenance = jnp.asarray(
        result["daily.fast_maintenance_respiration"],
        dtype=jnp.float64,
    )[0]
    maintenance = base_maintenance * multiplier
    result["daily.fast_maintenance_respiration"] = jnp.where(
        vegetation[:, None],
        maintenance,
        0.0,
    )[None, ...]
    return result


def retained_exact_pft_parameters(
    pft_parameters: Any,
    *,
    routing: PFTParameterRouting = DEFAULT_PFT_PARAMETER_ROUTING,
) -> RetainedExactPFTParameters:
    """Unpack the exact-process columns without exposing fast factors."""

    values = _validated_parameter_matrix(pft_parameters, routing)
    return RetainedExactPFTParameters(
        alloc_min=values[:, routing.index("alloc_min")],
        residence_time=values[:, routing.index("residence_time")],
    )


def route_retained_stomate_parameters(
    stomate_parameters: Any,
    pft_parameters: Any,
    active_pft_mask: Any,
    active_pft_indices: tuple[int, ...],
    *,
    routing: PFTParameterRouting = DEFAULT_PFT_PARAMETER_ROUTING,
) -> Any:
    """Install runtime-owned parameters into the full STOMATE vectors.

    Inactive compact slots preserve their construction-time values. This
    keeps inactive-slot perturbations isolated while allowing active runtime
    values and gradients to reach allocation and gap mortality.
    """

    values = _validated_parameter_matrix(pft_parameters, routing)
    routed = retained_exact_pft_parameters(values, routing=routing)
    vcmax25 = values[:, routing.index("vcmax25")]
    active = jnp.asarray(active_pft_mask, dtype=jnp.bool_)
    full_indices = jnp.asarray(active_pft_indices, dtype=jnp.int32)
    if active.shape != routed.alloc_min.shape:
        raise ValueError("active_pft_mask must contain one value per compact PFT")
    if full_indices.shape != active.shape:
        raise ValueError("active_pft_indices must contain one full index per compact PFT")

    def restore(base: Any, compact: Any) -> Any:
        result = jnp.asarray(base, dtype=jnp.float64)
        current = jnp.take(result, full_indices, axis=0)
        selected = jnp.where(active, jnp.asarray(compact, dtype=jnp.float64), current)
        return result.at[full_indices].set(selected)

    return stomate_parameters._replace(
        alloc_min=restore(stomate_parameters.alloc_min, routed.alloc_min),
        residence_time=restore(
            stomate_parameters.residence_time,
            routed.residence_time,
        ),
        vcmax25=restore(stomate_parameters.vcmax25, vcmax25),
    )
