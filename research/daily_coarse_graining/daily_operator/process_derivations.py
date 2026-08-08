"""Source-backed deterministic formulas inside the Gate-E1 candidate graph."""

from __future__ import annotations

from typing import Any, Mapping, NamedTuple

import jax.numpy as jnp

from jax_orchidee.driver.dim2 import (
    CP_AIR,
    GRAVITY,
    MOLAR_GAS_CONSTANT_AIR,
)
from jax_orchidee.stomate.carbon_kernels import (
    IAGRHRTPN,
    IAGRHRTST,
    IAGRSAPPN,
    IAGRSAPST,
    ICARBON,
    ICARBRES,
    IFRUIT,
    IHEARTABOVE,
    IHEARTBELOW,
    ILEAF,
    IROOT,
    ISAPABOVE,
    ISAPBELOW,
    NPARTS,
    root_profile_layer_weights,
)
from jax_orchidee.stomate.soilcarbon_kernels import (
    IACT,
    IACTIVE,
    IMETABO,
    IMETBEL,
    IPAS,
    IPASSIVE,
    ISLO,
    ISLOW,
    ISTRABO,
    ISTRBEL,
    NCARB,
    NPOOL,
)

from .types import (
    AssembledDailyOperatorInput,
    GrossDecompositionPrediction,
    LitterInputPrediction,
    NativeForcingInput,
)


class NativeForcingFieldLayout(NamedTuple):
    """Static named slices in ``DailyMarkovContract.native_forcing`` order."""

    rain_start: int
    rain_stop: int
    snow_start: int
    snow_stop: int
    temperature_start: int = -1
    temperature_stop: int = -1


class NativeDailyDeterministicFields(NamedTuple):
    """Exact daily forcing-owned fields without half-hour reconstruction."""

    t2m_daily: Any
    precip_daily: Any
    snowfall_daily: Any
    t2m_min_daily: Any
    t2m_max_daily: Any
    t2mdiag: Any


class EndOfDayDriverFields(NamedTuple):
    """DIM2 restart fields owned by the final supplied forcing record."""

    old_zlev: Any
    old_qair: Any
    old_eair: Any
    rau_old: Any


class DerivedCarbonDecomposition(NamedTuple):
    poc_respiration: Any
    poc_to_doc: Any
    doc_to_poc: Any
    doc_respiration: Any


def integrate_native_precipitation(
    forcing: NativeForcingInput,
    layout: NativeForcingFieldLayout,
) -> tuple[Any, Any]:
    """Integrate current-day Rainf/Snowf rates over native intervals.

    Source ownership: ``stomate_daily.f90::stomate_accu`` and
    ``jax_orchidee.stomate.daily`` lines 468-498. The predecessor record is
    boundary context only and is excluded from the current-day amount.
    """

    values = jnp.asarray(forcing.values, dtype=jnp.float64)
    duration = jnp.asarray(forcing.duration_seconds, dtype=jnp.float64)
    active = jnp.asarray(forcing.record_mask, dtype=jnp.bool_) & ~jnp.asarray(
        forcing.predecessor_mask,
        dtype=jnp.bool_,
    )
    rain_rate = values[:, int(layout.rain_start) : int(layout.rain_stop)]
    snow_rate = values[:, int(layout.snow_start) : int(layout.snow_stop)]
    if rain_rate.shape[1] < 1 or rain_rate.shape != snow_rate.shape:
        raise ValueError("Rainf and Snowf slices must be nonempty and shape matched")
    seconds = jnp.where(active, duration, 0.0)[:, None]
    return jnp.sum(rain_rate * seconds, axis=0), jnp.sum(snow_rate * seconds, axis=0)


def derive_native_daily_deterministic_fields(
    forcing: NativeForcingInput,
    layout: NativeForcingFieldLayout,
    *,
    model_interval_seconds: float,
) -> NativeDailyDeterministicFields:
    """Close the retained forcing fold directly over native records.

    Source ownership: the linear forcing interpolation in
    ``daily_markov_contract.reconstruct_compiled_forcing_day`` followed by
    ``stomate_daily.f90::stomate_accu``. For an interval with ``n`` model
    steps, the discrete linear sum is evaluated analytically as
    ``n*x0 + (x1-x0)*(n+1)/2``; no length-48 array is materialized.
    """

    if layout.temperature_start < 0 or layout.temperature_stop <= layout.temperature_start:
        raise ValueError("native daily forcing requires a nonempty Tair slice")
    model_dt = jnp.asarray(model_interval_seconds, dtype=jnp.float64)
    if float(model_interval_seconds) <= 0.0:
        raise ValueError("model forcing interval must be positive")
    values = jnp.asarray(forcing.values, dtype=jnp.float64)
    mask = jnp.asarray(forcing.record_mask, dtype=jnp.bool_)
    predecessor = jnp.asarray(forcing.predecessor_mask, dtype=jnp.bool_)
    duration = jnp.asarray(forcing.duration_seconds, dtype=jnp.float64)
    temperature = values[:, int(layout.temperature_start) : int(layout.temperature_stop)]
    width = int(temperature.shape[1])
    temperature_sum = jnp.zeros((width,), dtype=jnp.float64)
    minimum = jnp.full((width,), jnp.inf, dtype=jnp.float64)
    maximum = jnp.full((width,), -jnp.inf, dtype=jnp.float64)
    diagnostic = jnp.zeros((width,), dtype=jnp.float64)
    day_seconds = jnp.asarray(0.0, dtype=jnp.float64)
    for index in range(1, int(values.shape[0])):
        valid = mask[index] & ~predecessor[index] & mask[index - 1]
        interval_seconds = jnp.where(valid, duration[index], 0.0)
        steps = interval_seconds / model_dt
        previous = jnp.where(valid, temperature[index - 1], 0.0)
        current = jnp.where(valid, temperature[index], 0.0)
        difference = current - previous
        temperature_sum += model_dt * (
            steps * previous + difference * (steps + 1.0) / 2.0
        )
        first_step = previous + difference / jnp.maximum(steps, 1.0)
        minimum = jnp.minimum(
            minimum,
            jnp.where(valid, jnp.minimum(first_step, current), jnp.inf),
        )
        maximum = jnp.maximum(
            maximum,
            jnp.where(valid, jnp.maximum(first_step, current), -jnp.inf),
        )
        diagnostic = jnp.where(valid, current, diagnostic)
        day_seconds += interval_seconds
    safe_day = jnp.maximum(day_seconds, model_dt)
    rain, snow = integrate_native_precipitation(forcing, layout)
    any_current = jnp.any(mask & ~predecessor)
    return NativeDailyDeterministicFields(
        t2m_daily=temperature_sum / safe_day,
        precip_daily=(rain + snow) * (86400.0 / safe_day),
        snowfall_daily=snow * (model_dt / safe_day),
        t2m_min_daily=jnp.where(any_current, minimum, 0.0),
        t2m_max_daily=jnp.where(any_current, maximum, 0.0),
        t2mdiag=diagnostic,
    )


def derive_end_of_day_driver_fields(
    forcing: NativeForcingInput,
    forcing_indices: Mapping[str, int],
    *,
    scalar_measurement_height: Any,
) -> EndOfDayDriverFields:
    """Derive the four forcing-owned DIM2 restart fields without a 48-step scan.

    Fortran provenance: ``src_driver/dim2_driver.f90:1395-1397`` assigns the
    final interpolated forcing values, while lines 742-747 define
    ``old_eair = cp_air*Tair + cte_grav*zlev``. The accepted JAX source uses
    the same formulas in ``driver/dim2.py::_boundary`` lines 432-433. At the
    daily boundary the final active native record is the interpolation
    endpoint, so no half-hour forcing reconstruction is needed.
    """

    required = ("Tair", "PSurf", "Qair")
    if any(name not in forcing_indices for name in required):
        raise ValueError("end-of-day driver derivation is missing a forcing field")
    values = jnp.asarray(forcing.values, dtype=jnp.float64)
    active = jnp.asarray(forcing.record_mask, dtype=jnp.bool_) & ~jnp.asarray(
        forcing.predecessor_mask,
        dtype=jnp.bool_,
    )
    positions = jnp.arange(values.shape[0], dtype=jnp.int32)
    final_index = jnp.maximum(
        jnp.max(jnp.where(active, positions, -1)),
        jnp.asarray(0, dtype=jnp.int32),
    )

    def final(name: str) -> Any:
        return values[final_index, int(forcing_indices[name])].reshape(1)

    temperature = final("Tair")
    pressure = final("PSurf")
    humidity = final("Qair")
    zlev = jnp.asarray(scalar_measurement_height, dtype=jnp.float64).reshape(1)
    return EndOfDayDriverFields(
        old_zlev=zlev,
        old_qair=humidity,
        old_eair=CP_AIR * temperature + GRAVITY * zlev,
        rau_old=pressure / (MOLAR_GAS_CONSTANT_AIR * temperature),
    )


def derive_litter_input(inputs: AssembledDailyOperatorInput) -> LitterInputPrediction:
    """Apply the exact once-daily litter destination partition.

    Fortran provenance: ``src_stomate/stomate_litter.f90``,
    ``littercalc_leak`` lines 2138-2243. This is the source formula used by
    ``carbon_kernels.littercalc_litter_increments``, with the vegetation mask
    taken from the explicit Gate-B2 bare-soil trait instead of a slot number.
    """

    state = inputs.day_start_process_state
    static = inputs.process_static
    bm_to_litter = jnp.asarray(state.bm_to_litter, dtype=jnp.float64)
    turnover = jnp.asarray(state.turnover_daily, dtype=jnp.float64)
    litterfrac = jnp.asarray(static.litterfrac, dtype=jnp.float64)
    rprof = jnp.asarray(static.rprof, dtype=jnp.float64)
    z_soil = jnp.asarray(static.z_soil, dtype=jnp.float64)
    if bm_to_litter.shape != turnover.shape or bm_to_litter.ndim != 4:
        raise ValueError("litter inputs require [npts,n_pft,nparts,nelements] day-start arrays")
    npts, n_pft, nparts, nelements = bm_to_litter.shape
    if nparts != NPARTS or litterfrac.shape[0] != NPARTS:
        raise ValueError("litter part axes do not match the source partition")
    if rprof.shape != (npts, n_pft):
        raise ValueError("rprof must match the day-start landpoint/PFT axes")
    nslm = int(z_soil.shape[0] - 1)
    weights = root_profile_layer_weights(z_soil, rprof, nslm=nslm)
    flux = bm_to_litter + turnover
    above_parts = jnp.asarray(
        [
            ILEAF,
            ISAPABOVE,
            IHEARTABOVE,
            IFRUIT,
            ICARBRES,
            IAGRSAPST,
            IAGRSAPPN,
            IAGRHRTST,
            IAGRHRTPN,
        ],
        dtype=jnp.int32,
    )
    below_parts = jnp.asarray([ISAPBELOW, IHEARTBELOW, IROOT], dtype=jnp.int32)
    flux_carbon = flux[:, :, :, ICARBON : ICARBON + 1]
    above = jnp.einsum(
        "pk,njpe->njke",
        litterfrac[above_parts, :],
        flux_carbon[:, :, above_parts, :],
    )
    below_parts_flux = jnp.einsum(
        "pk,njpe->njke",
        litterfrac[below_parts, :],
        flux_carbon[:, :, below_parts, :],
    )
    below = below_parts_flux[:, :, :, None, :] * weights[:, :, None, :, None]
    active = jnp.asarray(inputs.pft.active_pft_mask, dtype=jnp.bool_)
    bare = jnp.asarray(inputs.pft.pft_traits, dtype=jnp.float64)[:, 0] > 0.5
    vegetation = active & ~bare
    above = jnp.where(vegetation[None, :, None, None], above, 0.0)
    below = jnp.where(vegetation[None, :, None, None, None], below, 0.0)
    return LitterInputPrediction(
        above=jnp.transpose(above, (0, 2, 1, 3)),
        below=jnp.transpose(below, (0, 2, 1, 3, 4)),
    )


def derive_carbon_decomposition(
    poc: GrossDecompositionPrediction,
    doc: GrossDecompositionPrediction,
    inputs: AssembledDailyOperatorInput,
) -> DerivedCarbonDecomposition:
    """Derive the four dependent carbon labels from predicted gross fluxes.

    Fortran provenance: ``src_stomate/stomate_soilcarbon.f90``,
    ``soilcarbon_leak`` lines 1605-1771. POC and DOC respiration retain the
    exact ``1-CUE`` split; POC-to-DOC and DOC-to-POC retain the source layer,
    flood, pool-fraction, and lignin destination rules.
    """

    poc_regular = jnp.asarray(poc.ordinary, dtype=jnp.float64)
    poc_flood = jnp.asarray(poc.flooded, dtype=jnp.float64)
    doc_regular = jnp.asarray(doc.ordinary, dtype=jnp.float64)
    doc_flood = jnp.asarray(doc.flooded, dtype=jnp.float64)
    if poc_regular.shape != poc_flood.shape or poc_regular.ndim != 5:
        raise ValueError("POC gross fluxes require [npts,ncarb,nelements,n_pft,ndeep]")
    if doc_regular.shape != doc_flood.shape or doc_regular.ndim != 4:
        raise ValueError("DOC gross fluxes require [npts,n_pft,ndeep,npool]")
    npts, ncarb, nelements, n_pft, ndeep = poc_regular.shape
    if ncarb != NCARB or doc_regular.shape != (npts, n_pft, ndeep, NPOOL):
        raise ValueError("gross carbon flux axes do not match the source pools")
    cue = jnp.asarray(inputs.process_static.cue, dtype=jnp.float64)
    if cue.ndim == 0:
        cue = jnp.full((npts,), cue, dtype=jnp.float64)
    if cue.shape != (npts,):
        raise ValueError("CUE must be scalar or landpoint-shaped")
    cue_poc = cue[:, None, None, None, None]
    cue_doc = cue[:, None, None, None]
    poc_total = poc_regular + poc_flood
    doc_total = doc_regular + doc_flood
    poc_respiration = jnp.sum((1.0 - cue_poc) * poc_total, axis=(1, 2, 4))
    doc_respiration = jnp.sum((1.0 - cue_doc) * doc_total, axis=(2, 3))

    sro_bottom = jnp.asarray(inputs.process_static.sro_bottom, dtype=jnp.int32)
    include_flood = jnp.arange(ndeep) >= sro_bottom
    poc_to_doc = jnp.zeros((npts, n_pft, ndeep, NPOOL, nelements), dtype=jnp.float64)
    for source, destination in ((IACTIVE, IACT), (ISLOW, ISLO), (IPASSIVE, IPAS)):
        source_flux = poc_regular[:, source, :, :, :] + jnp.where(
            include_flood[None, None, None, :],
            poc_flood[:, source, :, :, :],
            0.0,
        )
        source_flux = jnp.transpose(source_flux, (0, 2, 3, 1))
        poc_to_doc = poc_to_doc.at[:, :, :, destination, :].set(
            cue[:, None, None, None] * source_flux
        )

    frac_carb = jnp.asarray(inputs.process_static.frac_carb, dtype=jnp.float64)
    lignin_above = jnp.asarray(inputs.day_start_process_state.lignin_struc_above, dtype=jnp.float64)
    lignin_below = jnp.asarray(inputs.day_start_process_state.lignin_struc_below, dtype=jnp.float64)
    if frac_carb.shape != (npts, NCARB, NCARB):
        raise ValueError("frac_carb must have shape [npts,ncarb,ncarb]")
    if lignin_above.shape != (npts, n_pft) or lignin_below.shape != (npts, n_pft, ndeep):
        raise ValueError("lignin state does not match DOC gross-flux axes")

    def frac(source: int, target: int) -> Any:
        return frac_carb[:, source, target][:, None, None]

    active_gain = (
        frac(IPASSIVE, IACTIVE) * doc_total[:, :, :, IPAS]
        + frac(ISLOW, IACTIVE) * doc_total[:, :, :, ISLO]
        + doc_total[:, :, :, IMETBEL]
        + doc_total[:, :, :, ISTRBEL] * (1.0 - lignin_below)
        + doc_total[:, :, :, IMETABO]
        + doc_total[:, :, :, ISTRABO] * (1.0 - lignin_above[:, :, None])
    )
    slow_gain = (
        frac(IPASSIVE, ISLOW) * doc_total[:, :, :, IPAS]
        + frac(IACTIVE, ISLOW) * doc_total[:, :, :, IACT]
        + doc_total[:, :, :, ISTRBEL] * lignin_below
        + doc_total[:, :, :, ISTRABO] * lignin_above[:, :, None]
    )
    passive_gain = (
        frac(IACTIVE, IPASSIVE) * doc_total[:, :, :, IACT]
        + frac(ISLOW, IPASSIVE) * doc_total[:, :, :, ISLO]
    )
    doc_to_poc = cue[:, None, None, None] * jnp.stack(
        (active_gain, slow_gain, passive_gain),
        axis=1,
    )
    return DerivedCarbonDecomposition(
        poc_respiration=poc_respiration,
        poc_to_doc=poc_to_doc,
        doc_to_poc=doc_to_poc,
        doc_respiration=doc_respiration,
    )
