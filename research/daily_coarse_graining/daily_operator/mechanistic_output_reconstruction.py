"""Pure-JAX daily-boundary reconstruction for mechanistic neural outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import jax
import jax.numpy as jnp

from jax_orchidee.sechiba.condveg import condveg_frac_snow, condveg_main_minimal
from jax_orchidee.sechiba.hydrol import (
    USDA_MCR,
    HydrolThermosoilMoistureInputs,
    hydrol_evap_bare_limit_grid_beta,
    hydrol_layer_thickness_from_zz_mm,
    hydrol_litter_grid_diagnostics,
    hydrol_litter_top_diagnostics,
    hydrol_shumdiag,
    hydrol_soil_layer_diagnostics,
    hydrol_soilmoist_aggregates,
    hydrol_stress_aggregates,
    hydrol_water_stress_diagnostics,
)
from jax_orchidee.sechiba.slowproc import slowproc_veget_explicit
from jax_orchidee.sechiba.thermosoil import (
    thermosoil_coef_explicit_snow,
    thermosoil_diaglev_from_ptn,
    thermosoil_final_state,
    thermosoil_getdiff_explicit,
    thermosoil_humlev,
    thermosoil_soilc_tempdiff_fraction,
)
from jax_orchidee.stomate.soilcarbon_kernels import soilcarbon_leak_doc_diffusion
from research.daily_coarse_graining.constrained_daily_replay import (
    CARBON_INVENTORY_FIELDS,
    WATER_INVENTORY_FIELDS,
)
from research.daily_coarse_graining.daily_markov_contract import (
    reconstruct_state_fields_compiled,
)

from .conservative_update import split_signed_inventory
from .fast_owner_reconstruction import ROLE_NEURAL, SuccessorFastOwnerMaterializer
from .input_assembly import PFT_AXES, CanonicalDailyOperatorInputAssembler
from .mechanistic_output_contract import MechanisticOutputContract
from .types import (
    AssembledDailyOperatorInput,
    ConservativeUpdateResult,
    ProcessPrediction,
)

DAILY_DOC_DIFFUSIVITY = 1.0e-5
DOC_BASELINE_EPSILON = 1.0e-6
STEMPDIAG_TENDENCY_BOUND = 100.0
CANOPY_ENDPOINT_AUX = "canopy.endpoint"
CANOPY_ENDPOINT_SCALE_AUX = "canopy.endpoint_scale"
CANOPY_STATUS_LOGITS_AUX = "canopy.status_logits"


def support_aware_nroot_simplex(raw: Any, day_start: Any, active_pft: Any) -> Any:
    """Map residual scores to the source-admitted day-start root support."""

    raw = jnp.asarray(raw, dtype=jnp.float64)
    day_start = jnp.asarray(day_start, dtype=jnp.float64)
    active_pft = jnp.asarray(active_pft, dtype=jnp.bool_)
    if raw.shape != day_start.shape or raw.ndim != 3:
        raise ValueError("nroot coordinate requires matching (npts,n_pft,nslm) arrays")
    if active_pft.shape != (raw.shape[1],):
        raise ValueError("nroot active-PFT mask shape drift")
    finite = jnp.isfinite(day_start) & (jnp.abs(day_start) < 0.5e20)
    allowed = finite & (day_start > 0.0) & active_pft[None, :, None]
    start = jnp.where(allowed, day_start, 0.0)
    start_sum = jnp.sum(start, axis=-1, keepdims=True)
    start = jnp.where(start_sum > 0.0, start / jnp.where(start_sum > 0.0, start_sum, 1.0), 0.0)
    scores = jnp.where(allowed, start + raw, -1.0e30)
    ordered = jnp.sort(scores, axis=-1)[..., ::-1]
    cumulative = jnp.cumsum(ordered, axis=-1)
    rank = jnp.arange(1, raw.shape[-1] + 1, dtype=jnp.float64)
    support = 1.0 + rank * ordered > cumulative
    support_count = jnp.sum(support, axis=-1, keepdims=True)
    safe_count = jnp.maximum(support_count, 1)
    tau_sum = jnp.take_along_axis(cumulative, safe_count - 1, axis=-1)
    tau = (tau_sum - 1.0) / safe_count
    sparse = jnp.maximum(jnp.where(allowed, scores - tau, 0.0), 0.0)
    sparse_sum = jnp.sum(sparse, axis=-1, keepdims=True)
    return jnp.where(sparse_sum > 0.0, sparse / sparse_sum, start)


def signed_endpoint_to_transfers(start: Any, end: Any) -> dict[str, Any]:
    """Convert one signed canopy endpoint into unique storage/debt transfers."""

    start_storage, start_debt = split_signed_inventory(start)
    end_storage, end_debt = split_signed_inventory(end)

    def component(source: Any, target: Any) -> tuple[Any, Any]:
        removed = jnp.maximum(source - target, 0.0)
        outgoing = jnp.where(source > 0.0, removed / source, 0.0)
        incoming = jnp.maximum(target - source, 0.0)
        return jnp.clip(outgoing, 0.0, 1.0), incoming

    storage_out, storage_in = component(start_storage, end_storage)
    debt_out, debt_in = component(start_debt, end_debt)
    return {
        "storage.outgoing_fraction": storage_out,
        "storage.incoming_amount": storage_in,
        "debt.outgoing_fraction": debt_out,
        "debt.incoming_amount": debt_in,
    }


def _hydrol_boundary_vegetation_mapping(
    veget_max: Any,
    soiltile: Any,
    vegtot: Any,
    pref_soil_veg: Any,
) -> tuple[Any, Any]:
    """JIT-compatible subset of the audited post-``hydrol_vegupd`` mapping."""

    veget_max = jnp.asarray(veget_max, dtype=jnp.float64)
    soiltile = jnp.asarray(soiltile, dtype=jnp.float64)
    vegtot = jnp.asarray(vegtot, dtype=jnp.float64)
    pref_zero = jnp.asarray(pref_soil_veg, dtype=jnp.int32) - 1
    routing = (
        jnp.arange(soiltile.shape[1], dtype=jnp.int32)[None, None, :]
        == pref_zero[None, :, None]
    )
    tile_active = soiltile > 1.0e-8
    safe_soiltile = jnp.where(tile_active, soiltile, 1.0)
    vegetmax_soil = jnp.where(
        routing & tile_active[:, None, :] & (vegtot[:, None, None] > 1.0e-8),
        veget_max[:, :, None] / safe_soiltile[:, None, :],
        0.0,
    )
    return tile_active, vegetmax_soil


def reconstruct_hydrology_boundary_diagnostics(
    *,
    mc: Any,
    mcl: Any,
    nroot: Any,
    veget: Any,
    veget_max: Any,
    soiltile: Any,
    pref_soil_veg: Any,
    dz_mm: Any,
    njsc: Any,
    evap_bare_lim_ns: Any,
) -> dict[str, Any]:
    """Evaluate the accepted no-scan HYDROL endpoint diagnostics once."""

    mc = jnp.asarray(mc, dtype=jnp.float64)
    mcl = jnp.asarray(mcl, dtype=jnp.float64)
    nroot = jnp.asarray(nroot, dtype=jnp.float64)
    veget = jnp.asarray(veget, dtype=jnp.float64)
    veget_max = jnp.asarray(veget_max, dtype=jnp.float64)
    soiltile = jnp.asarray(soiltile, dtype=jnp.float64)
    njsc = jnp.asarray(njsc, dtype=jnp.int32)
    if mc.shape != mcl.shape or mc.ndim != 3:
        raise ValueError("HYDROL endpoint moisture arrays must share (npts,nslm,nstm)")
    npts, _nslm, nstm = mc.shape
    if nroot.shape[:2] != veget.shape or veget.shape != veget_max.shape:
        raise ValueError("HYDROL endpoint PFT arrays drift")
    vegtot = jnp.sum(veget_max, axis=1)
    tile_active, vegetmax_soil = _hydrol_boundary_vegetation_mapping(
        veget_max=veget_max,
        soiltile=soiltile,
        vegtot=vegtot,
        pref_soil_veg=jnp.asarray(pref_soil_veg, dtype=jnp.int32),
    )
    mcr = jnp.asarray(USDA_MCR, dtype=jnp.float64)[njsc - 1]
    inferred_under = tile_active & (mc[:, 0, :] < mcr[:, None])
    diagnostics = []
    for tile in range(nstm):
        layer = hydrol_soil_layer_diagnostics(
            mc=mc[:, :, tile],
            mcl=mcl[:, :, tile],
            dz_mm=dz_mm,
            njsc=njsc,
            peat_hydro=False,
            soil_tile_index=tile + 1,
        )
        diagnostics.append(
            hydrol_water_stress_diagnostics(
                layer=layer,
                nroot=nroot,
                vegetmax_soil_tile=vegetmax_soil[:, :, tile],
                is_under_mcr=inferred_under[:, tile],
                njsc=njsc,
                peat_hydro=False,
                soil_tile_index=tile + 1,
                dyn_nroot_larix=False,
                new_watstress=False,
            )
        )
    us = jnp.stack(tuple(item.us for item in diagnostics), axis=2)
    humrelv = jnp.stack(tuple(item.humrelv for item in diagnostics), axis=2)
    vegstressv = jnp.stack(tuple(item.vegstressv for item in diagnostics), axis=2)
    aggregate = hydrol_stress_aggregates(
        humrelv=humrelv,
        vegstressv=vegstressv,
        veget_max=veget_max,
    )
    evap_bare_lim = hydrol_evap_bare_limit_grid_beta(
        evap_bare_lim_ns=evap_bare_lim_ns,
        vegtot=vegtot,
        soiltile=soiltile,
    )
    if us.shape[0] != npts:
        raise ValueError("HYDROL endpoint reconstruction point axis drift")
    return {
        "us": us,
        "humrelv": humrelv,
        "humrel": aggregate.humrel,
        "evap_bare_lim": evap_bare_lim,
    }


def _copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {name: _copy_mapping(child) if isinstance(child, Mapping) else child for name, child in value.items()}


def _restore_pft(base: Any, compact: Any, axis: int, active: tuple[int, ...]) -> Any:
    result = jnp.asarray(base)
    source = jnp.asarray(compact)
    for compact_index, full_index in enumerate(active):
        target_slice = [slice(None)] * result.ndim
        source_slice = [slice(None)] * source.ndim
        target_slice[axis] = int(full_index)
        source_slice[axis] = int(compact_index)
        result = result.at[tuple(target_slice)].set(source[tuple(source_slice)])
    return result


def _unpack_water(
    vector: Any,
    base_fields: Mapping[str, Any],
    active: tuple[int, ...],
) -> dict[str, Any]:
    result = dict(base_fields)
    values = jnp.asarray(vector, dtype=jnp.float64)
    cursor = 0
    for field in WATER_INVENTORY_FIELDS:
        base = jnp.asarray(base_fields[field])
        axis = PFT_AXES.get(field)
        compact_shape = list(base.shape)
        if axis is not None:
            compact_shape[axis] = len(active)
        size = 1
        for extent in compact_shape:
            size *= int(extent)
        if field == "qsintveg":
            storage = values[cursor : cursor + size].reshape(compact_shape)
            cursor += size
            debt = values[cursor : cursor + size].reshape(compact_shape)
            cursor += size
            compact = storage - debt
        else:
            compact = values[cursor : cursor + size].reshape(compact_shape)
            cursor += size
        result[field] = compact if axis is None else _restore_pft(base, compact, axis, active)
    if cursor != values.size:
        raise ValueError("mechanistic water vector does not match the canonical layout")
    return result


def _unpack_carbon(
    vector: Any,
    base_fields: Mapping[str, Any],
    active: tuple[int, ...],
) -> dict[str, Any]:
    result = dict(base_fields)
    values = jnp.asarray(vector, dtype=jnp.float64)
    cursor = 0
    for field in CARBON_INVENTORY_FIELDS:
        base = jnp.asarray(base_fields[field])
        axis = PFT_AXES[field]
        compact_shape = list(base.shape)
        compact_shape[axis] = len(active)
        size = 1
        for extent in compact_shape:
            size *= int(extent)
        compact = values[cursor : cursor + size].reshape(compact_shape)
        cursor += size
        result[field] = _restore_pft(base, compact, axis, active)
    if cursor != values.size:
        raise ValueError("mechanistic carbon vector does not match the canonical layout")
    return result


def _unpack_thermal(
    vector: Any,
    base_fields: Mapping[str, Mapping[str, Any]],
    assembler: CanonicalDailyOperatorInputAssembler,
) -> dict[str, dict[str, Any]]:
    result = _copy_mapping(base_fields)
    values = jnp.asarray(vector, dtype=jnp.float64)
    cursor = 0
    for key in assembler.thermal_keys:
        component, field = key.split(".", 1)
        shape = tuple(jnp.asarray(base_fields[component][field]).shape)
        size = 1
        for extent in shape:
            size *= int(extent)
        result[component][field] = values[cursor : cursor + size].reshape(shape)
        cursor += size
    if cursor != values.size:
        raise ValueError("mechanistic thermal vector does not match the canonical layout")
    return result


def updated_fast_state_fields(
    prediction: ProcessPrediction,
    fast_update: ConservativeUpdateResult,
    assembler: CanonicalDailyOperatorInputAssembler,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Overlay conservative inventory/thermal endpoints on the predicted boundary."""

    active = tuple(int(index) for index in assembler.active_pft_indices)
    fields = _unpack_thermal(
        fast_update.state.thermal,
        prediction.retained_boundary.fast_state_fields,
        assembler,
    )
    hydrol = fields["hydrol_previous_step_state"]
    fields["hydrol_previous_step_state"] = _unpack_water(
        fast_update.state.water,
        hydrol,
        active,
    )
    ok_leak = _unpack_carbon(
        fast_update.state.carbon,
        prediction.retained_boundary.ok_leak_result,
        active,
    )
    return fields, ok_leak


def _paper_vertical_grid() -> Any:
    from jax_orchidee.sechiba.diffuco import cwrr_vertical_soil_grid_from_params

    return cwrr_vertical_soil_grid_from_params(
        depth_max_h=2.0,
        depth_max_t=38.0,
        depth_topthickness=9.77517107e-4,
        depth_cstthickness=2.0,
        depth_geom=2.0,
        ratio_geom_below=1.05,
    )


def _paper_zf_soil_b() -> Any:
    grid = _paper_vertical_grid()
    return jnp.concatenate((jnp.zeros((1,), dtype=jnp.float64), jnp.asarray(grid.zlt, dtype=jnp.float64)))


def _last_forcing_record(inputs: AssembledDailyOperatorInput) -> Any:
    mask = jnp.asarray(inputs.forcing.record_mask, dtype=jnp.bool_)
    indices = jnp.arange(mask.shape[0], dtype=jnp.int32)
    index = jnp.max(jnp.where(mask, indices, -1))
    return jnp.asarray(inputs.forcing.values, dtype=jnp.float64)[index]


def _select_pft(value: Any, active: tuple[int, ...], axis: int) -> Any:
    return jnp.take(
        jnp.asarray(value),
        jnp.asarray(active, dtype=jnp.int32),
        axis=axis,
    )


def reconstruct_thermal_coefficients(
    *,
    ptn: Any,
    shum_ngrnd_permalong: Any,
    ref_soc: Any,
    mc: Any,
    mcl: Any,
    veget_max: Any,
    frac_nobio: Any,
    lai: Any,
    temp_sol: Any,
    temp_sol_pft: Any,
    snow: Any,
    snow_nobio: Any,
    snowrho: Any,
    snowtemp: Any,
    snowdz: Any,
    pressure_hpa: Any,
    njsc: Any,
    hydrol_zz_mm: Any,
    hydrol_dz_mm: Any,
    pref_soil_veg: Any,
    ext_coeff_vegetfrac: Any,
    model_interval_seconds: float,
) -> dict[str, Any]:
    """Evaluate the accepted no-scan source sequence at one daily boundary."""

    ptn = jnp.asarray(ptn, dtype=jnp.float64)
    mc = jnp.asarray(mc, dtype=jnp.float64)
    mcl = jnp.asarray(mcl, dtype=jnp.float64)
    veget_max = jnp.asarray(veget_max, dtype=jnp.float64)
    frac_nobio = jnp.asarray(frac_nobio, dtype=jnp.float64)
    lai = jnp.asarray(lai, dtype=jnp.float64)
    hydrol_zz_mm = jnp.asarray(hydrol_zz_mm, dtype=jnp.float64)
    hydrol_dz_mm = jnp.asarray(hydrol_dz_mm, dtype=jnp.float64)
    pref_soil_veg = jnp.asarray(pref_soil_veg, dtype=jnp.int32)
    ext_coeff_vegetfrac = jnp.asarray(ext_coeff_vegetfrac, dtype=jnp.float64)
    njsc = jnp.asarray(njsc, dtype=jnp.int32)
    if ptn.ndim != 3 or mc.ndim != 3:
        raise ValueError("ptn and mc must have batched THERMOSOIL/HYDROL shapes")
    batch, _ngrnd, nvm = ptn.shape
    if mcl.shape != mc.shape or veget_max.shape != (batch, nvm):
        raise ValueError("thermal boundary PFT or HYDROL shapes drift")

    grid = _paper_vertical_grid()
    vegetation = slowproc_veget_explicit(
        lai=lai,
        frac_nobio=frac_nobio,
        veget_max=veget_max,
        pref_soil_veg=pref_soil_veg,
        ext_coeff_vegetfrac=ext_coeff_vegetfrac,
        nstm=mc.shape[2],
    )
    vegtot = 1.0 - vegetation.totfrac_nobio
    moisture = hydrol_soilmoist_aggregates(
        mc=mc,
        mcl=mcl,
        soiltile=vegetation.soiltile,
        vegtot=vegtot,
        dz_mm=hydrol_dz_mm,
    )
    shumdiag = hydrol_shumdiag(
        mc=mc,
        soil_wet_ns=jnp.zeros_like(mc),
        soilmoist=moisture.soilmoist,
        soiltile=vegetation.soiltile,
        dh_mm=hydrol_layer_thickness_from_zz_mm(hydrol_zz_mm),
        dz_mm=hydrol_dz_mm,
        njsc=njsc,
        peat_hydro=False,
    )
    pref_zero = pref_soil_veg - 1
    thermal_moisture = HydrolThermosoilMoistureInputs(
        shumdiag_perma=shumdiag.shumdiag_perma,
        mc_layh=moisture.mc_layh,
        mcl_layh=moisture.mcl_layh,
        tmc_layh=moisture.soilmoist,
        mc_layh_pft=jnp.take(moisture.mc_layh_s, pref_zero, axis=2),
        mcl_layh_pft=jnp.take(moisture.mcl_layh_s, pref_zero, axis=2),
        tmc_layh_pft=jnp.broadcast_to(
            moisture.soilmoist[:, :, None],
            (batch, mc.shape[1], nvm),
        ),
    )
    humlev = thermosoil_humlev(
        shumdiag_perma=thermal_moisture.shumdiag_perma,
        mc_layh=thermal_moisture.mc_layh,
        mcl_layh=thermal_moisture.mcl_layh,
        tmc_layh=thermal_moisture.tmc_layh,
        mc_layh_pft=thermal_moisture.mc_layh_pft,
        mcl_layh_pft=thermal_moisture.mcl_layh_pft,
        tmc_layh_pft=thermal_moisture.tmc_layh_pft,
        znt=grid.znt,
        zlt=grid.zlt,
        dz5=grid.dz5,
    )
    zx1, _zx2 = thermosoil_soilc_tempdiff_fraction(refsoc=ref_soc, nvm=nvm)
    getdiff = thermosoil_getdiff_explicit(
        ptn=ptn,
        njsc=njsc,
        veget_max=vegetation.veget_max,
        shum_ngrnd_permalong=shum_ngrnd_permalong,
        mc_layt=humlev.mc_layt,
        tmc_layt_pft=humlev.tmc_layt_pft,
        snowrho=snowrho,
        snowtemp=snowtemp,
        pb=pressure_hpa,
        dlt=grid.dlt,
        zx1=zx1,
        ok_laidev=jnp.zeros((nvm,), dtype=jnp.bool_),
        nslm=mc.shape[1],
    )
    final = thermosoil_final_state(
        ptn=ptn,
        pkappa=getdiff.pkappa,
        veget_max=vegetation.veget_max,
        shum_ngrnd_permalong=shum_ngrnd_permalong,
    )
    snow_fraction = condveg_frac_snow(
        snow=snow,
        snow_nobio=snow_nobio,
        snowrho=snowrho,
        snowdz=snowdz,
        ok_explicitsnow=True,
    )
    coefficient = thermosoil_coef_explicit_snow(
        ptn=ptn,
        ptn_pftmean=final.ptn_pftmean,
        pcapa=getdiff.pcapa,
        pkappa=getdiff.pkappa,
        pcapa_snow=getdiff.pcapa_snow,
        pkappa_snow=getdiff.pkappa_snow,
        snowdz=snowdz,
        snowtemp=snowtemp,
        temp_sol_new=temp_sol,
        temp_sol_new_pft=temp_sol_pft,
        veget_max=vegetation.veget_max,
        ok_laidev=jnp.zeros((nvm,), dtype=jnp.bool_),
        dlt=grid.dlt,
        dz1=grid.dz1,
        zlt=grid.zlt,
        dt_sechiba=model_interval_seconds,
        lambda_thermal=grid.lambda_thermal,
        frac_snow_veg=snow_fraction.frac_snow_veg,
        frac_snow_nobio=snow_fraction.frac_snow_nobio,
        totfrac_nobio=vegetation.totfrac_nobio,
    )
    return {
        "cgrnd": coefficient.soil.cgrnd,
        "dgrnd": coefficient.soil.dgrnd,
        "cgrnd_snow": coefficient.cgrnd_snow,
        "dgrnd_snow": coefficient.dgrnd_snow,
        "lambda_snow": coefficient.lambda_snow,
        "pcapa_en": getdiff.pcapa_en,
        "soilcap": coefficient.soilcap,
        "soilcap_pft": coefficient.soil.soilcap_pft,
        "soilflx": coefficient.soilflx,
        "soilflx_pft": coefficient.soil.soilflx_pft,
    }


def _softplus_inverse(value: Any) -> Any:
    value = jnp.asarray(value, dtype=jnp.float64)
    return jnp.where(value > 20.0, value, jnp.log(jnp.expm1(value)))


def compose_doc_baseline_residual(baseline: Any, residual: Any, scale: Any) -> Any:
    """Apply one baseline-centered coordinate with an exact zero-residual identity."""

    baseline = jnp.maximum(jnp.asarray(baseline, dtype=jnp.float64), 0.0)
    residual = jnp.asarray(residual, dtype=jnp.float64)
    scale = jnp.asarray(scale, dtype=jnp.float64)
    scale = jnp.where(jnp.isfinite(scale) & (scale > 0.0), scale, 1.0)
    epsilon = jnp.asarray(DOC_BASELINE_EPSILON, dtype=jnp.float64)
    center = baseline / scale + epsilon
    value = (jax.nn.softplus(_softplus_inverse(center) + residual) - epsilon) * scale
    return jnp.maximum(value, 0.0)


def invert_doc_baseline_residual(target: Any, baseline: Any, scale: Any) -> Any:
    """Recover the finite residual coordinate for a nonnegative DOC target."""

    target = jnp.maximum(jnp.asarray(target, dtype=jnp.float64), 0.0)
    baseline = jnp.maximum(jnp.asarray(baseline, dtype=jnp.float64), 0.0)
    scale = jnp.asarray(scale, dtype=jnp.float64)
    scale = jnp.where(jnp.isfinite(scale) & (scale > 0.0), scale, 1.0)
    epsilon = jnp.asarray(DOC_BASELINE_EPSILON, dtype=jnp.float64)
    return _softplus_inverse(target / scale + epsilon) - _softplus_inverse(
        baseline / scale + epsilon
    )


@dataclass(frozen=True, eq=False)
class DailyBoundaryReconstructor:
    assembler: CanonicalDailyOperatorInputAssembler
    contract: MechanisticOutputContract
    forcing_indices: Mapping[str, int]
    output_scale: Mapping[str, Any]
    fast_owner_materializer: SuccessorFastOwnerMaterializer

    def __post_init__(self) -> None:
        required = {"Tair", "PSurf", "Wind_N", "Wind_E"}
        if not required <= set(self.forcing_indices):
            raise ValueError("mechanistic reconstruction forcing registry is incomplete")
        if self.fast_owner_materializer.markov_contract.sha256 != self.assembler.contract.sha256:
            raise ValueError("mechanistic reconstruction and fast-owner contracts drift")

    @property
    def active_pft_indices(self) -> tuple[int, ...]:
        return tuple(int(index) for index in self.assembler.active_pft_indices)

    def static(self, inputs: AssembledDailyOperatorInput, name: str) -> Any:
        matches = tuple(
            leaf for leaf in self.assembler.contract.static_conditions.landpoint_static_leaves if leaf.name == name
        )
        if len(matches) != 1:
            raise ValueError(f"mechanistic reconstruction requires one {name!r} static leaf")
        leaf = matches[0]
        return jnp.asarray(inputs.static_conditions, dtype=jnp.float64)[leaf.start : leaf.stop].reshape(leaf.shape)

    def doc_active_mask(self, inputs: AssembledDailyOperatorInput, rank: int) -> Any:
        compact = jnp.asarray(inputs.pft.active_pft_mask, dtype=jnp.bool_)
        full_pft_count = int(jnp.asarray(self.output_scale[self.contract.doc_field]).shape[0])
        full = jnp.zeros((full_pft_count,), dtype=jnp.bool_)
        full = full.at[jnp.asarray(self.active_pft_indices, dtype=jnp.int32)].set(compact)
        return full.reshape((full_pft_count, *(1 for _ in range(rank - 1))))

    def canopy_capacity(
        self,
        fields: Mapping[str, Mapping[str, Any]],
        inputs: AssembledDailyOperatorInput,
    ) -> Any:
        slow = fields["slowproc_stomate_previous_step_state"]
        capacity = (
            self.static(inputs, "sechiba_qsint")
            * jnp.asarray(slow["veget"], dtype=jnp.float64)
            * jnp.asarray(slow["lai"], dtype=jnp.float64)
        )
        capacity = _select_pft(capacity, self.active_pft_indices, axis=1)
        bare = jnp.asarray(inputs.pft.pft_traits, dtype=jnp.float64)[:, 0] > 0.5
        return jnp.where(bare[None, :], 0.0, jnp.maximum(capacity, 0.0))

    def doc_start_baseline(self, inputs: AssembledDailyOperatorInput) -> Any:
        fields = reconstruct_state_fields_compiled(
            inputs.canonical_input.continuous_state,
            inputs.canonical_input.discrete_state,
            self.assembler.contract,
        )
        doc = jnp.asarray(
            fields["slowproc_stomate_previous_step_state"]["DOC"],
            dtype=jnp.float64,
        )
        ptn = jnp.asarray(
            fields["thermosoil_previous_step_state"]["ptn"],
            dtype=jnp.float64,
        )
        return soilcarbon_leak_doc_diffusion(
            doc,
            ptn,
            _paper_zf_soil_b(),
            jnp.full((doc.shape[0],), DAILY_DOC_DIFFUSIVITY, dtype=jnp.float64),
        ).doc_flux_diff[0]

    def updated_fast_owner_fields(
        self,
        prediction: ProcessPrediction,
        fast_update: ConservativeUpdateResult,
        inputs: AssembledDailyOperatorInput,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        """Apply learned day-end owners before evaluating exact diagnostics."""

        fields, ok_leak = updated_fast_state_fields(
            prediction,
            fast_update,
            self.assembler,
        )
        exact_outputs = frozenset(
            leaf.output_key
            for leaf in self.fast_owner_materializer.owner_contract.leaves
            if leaf.role == ROLE_NEURAL and leaf.output_key in set(self.contract.exact_fields)
        )
        materialized = self.fast_owner_materializer.materialize(
            fields_by_component=fields,
            daily_fields=prediction.retained_boundary.daily_fields,
            ok_leak_updates=ok_leak,
            final_diagnostics=prediction.retained_boundary.final_entry_diagnostics,
            named_outputs=prediction.named_outputs,
            forcing=inputs.forcing,
            measurement_height=self.static(inputs, "measurement_heights")[0],
            permitted_missing_output_keys=exact_outputs,
        )
        return materialized.fields_by_component, materialized.ok_leak_updates

    def thermal_coefficient_inputs(
        self,
        fields: Mapping[str, Mapping[str, Any]],
        inputs: AssembledDailyOperatorInput,
    ) -> dict[str, Any]:
        """Expose the exact source values used by the thermal coefficient kernel."""

        hydrol = fields["hydrol_previous_step_state"]
        diffuco = fields["diffuco_previous_step_state"]
        thermal = fields["thermosoil_previous_step_state"]
        enerbil = fields["enerbil_previous_step_state"]
        active = self.active_pft_indices
        forcing = _last_forcing_record(inputs)

        def compact(value: Any, axis: int) -> Any:
            return _select_pft(value, active, axis)

        mc = jnp.asarray(hydrol["mc"], dtype=jnp.float64)
        return {
            "ptn": compact(thermal["ptn"], 2),
            "shum_ngrnd_permalong": compact(thermal["shum_ngrnd_permalong"], 2),
            "ref_soc": thermal["refSOC"],
            "mc": mc,
            "mcl": hydrol["mcl"],
            "veget_max": compact(diffuco["veget_max"], 1),
            "frac_nobio": diffuco["frac_nobio"],
            "lai": compact(diffuco["lai"], 1),
            "temp_sol": enerbil["temp_sol"],
            "temp_sol_pft": compact(diffuco["temp_sol_pft"], 1),
            "snow": hydrol["snow"],
            "snow_nobio": hydrol["snow_nobio"],
            "snowrho": hydrol["snowrho"],
            "snowtemp": hydrol["snowtemp"],
            "snowdz": hydrol["snowdz"],
            "pressure_hpa": forcing[self.forcing_indices["PSurf"]][None] / 100.0,
            "njsc": jnp.full(
                (mc.shape[0],),
                self.static(inputs, "njsc").reshape(-1)[0],
                dtype=jnp.int32,
            ),
            "hydrol_zz_mm": self.static(inputs, "hydrol_zz_mm"),
            "hydrol_dz_mm": self.static(inputs, "hydrol_dz_mm"),
            "pref_soil_veg": compact(self.static(inputs, "pref_soil_veg"), 0),
            "ext_coeff_vegetfrac": compact(
                self.static(inputs, "ext_coeff_vegetfrac"),
                0,
            ),
            "model_interval_seconds": float(
                self.assembler.contract.native_forcing.model_interval_seconds
            ),
        }

    def doc_trapezoid_baseline(
        self,
        prediction: ProcessPrediction,
        fast_update: ConservativeUpdateResult,
        inputs: AssembledDailyOperatorInput,
    ) -> Any:
        fields, ok_leak = self.updated_fast_owner_fields(
            prediction,
            fast_update,
            inputs,
        )
        start_fields = reconstruct_state_fields_compiled(
            inputs.canonical_input.continuous_state,
            inputs.canonical_input.discrete_state,
            self.assembler.contract,
        )
        start_doc = jnp.asarray(
            start_fields["slowproc_stomate_previous_step_state"]["DOC"],
            dtype=jnp.float64,
        )
        start_ptn = jnp.asarray(
            start_fields["thermosoil_previous_step_state"]["ptn"],
            dtype=jnp.float64,
        )
        end_doc = jnp.asarray(ok_leak["DOC"], dtype=jnp.float64)
        end_ptn = jnp.asarray(
            fields["thermosoil_previous_step_state"]["ptn"],
            dtype=jnp.float64,
        )
        diffusivity = jnp.full(
            (start_doc.shape[0],),
            DAILY_DOC_DIFFUSIVITY,
            dtype=jnp.float64,
        )
        start_flux = soilcarbon_leak_doc_diffusion(
            start_doc,
            start_ptn,
            _paper_zf_soil_b(),
            diffusivity,
        ).doc_flux_diff
        end_flux = soilcarbon_leak_doc_diffusion(
            end_doc,
            end_ptn,
            _paper_zf_soil_b(),
            diffusivity,
        ).doc_flux_diff
        return 0.5 * (start_flux + end_flux)[0]

    def pre_update_outputs(
        self,
        learned_outputs: Mapping[str, Any],
        inputs: AssembledDailyOperatorInput,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        result = dict(learned_outputs)
        fields = reconstruct_state_fields_compiled(
            inputs.canonical_input.continuous_state,
            inputs.canonical_input.discrete_state,
            self.assembler.contract,
        )
        hydrol = fields["hydrol_previous_step_state"]
        active = self.active_pft_indices
        nroot_key = self.contract.nroot_coordinate_field
        start_nroot = _select_pft(hydrol["nroot"], active, axis=1)
        result[nroot_key] = support_aware_nroot_simplex(
            result[nroot_key],
            start_nroot,
            inputs.pft.active_pft_mask,
        )

        start_qsintveg = _select_pft(hydrol["qsintveg"], active, axis=1)
        capacity = self.canopy_capacity(fields, inputs)
        endpoint_scale = jnp.asarray(
            self.output_scale[self.contract.canopy_debt_scale_field],
            dtype=jnp.float64,
        )
        endpoint_scale = jnp.maximum(
            capacity,
            jnp.where(
                jnp.isfinite(endpoint_scale) & (endpoint_scale > 0.0),
                endpoint_scale,
                1.0e-6,
            ),
        )
        endpoint_raw = jnp.asarray(
            result[self.contract.canopy_endpoint_field],
            dtype=jnp.float64,
        )
        endpoint = jnp.minimum(start_qsintveg + endpoint_scale * endpoint_raw, capacity)
        vegetation = (
            jnp.asarray(inputs.pft.active_pft_mask, dtype=jnp.bool_)
            & (jnp.asarray(inputs.pft.pft_traits, dtype=jnp.float64)[:, 0] < 0.5)
        )
        endpoint = jnp.where(vegetation[None, :], endpoint, 0.0)
        status_logits = jnp.stack(
            tuple(
                jnp.asarray(result[self.contract.canopy_status_fields[status]], dtype=jnp.float64)
                for status in ("storage", "zero", "debt")
            ),
            axis=-1,
        )
        transfers = signed_endpoint_to_transfers(start_qsintveg, endpoint)
        for component, value in transfers.items():
            result[f"inventory.water.qsintveg.{component}"] = value
        auxiliary = {
            CANOPY_ENDPOINT_AUX: endpoint,
            CANOPY_ENDPOINT_SCALE_AUX: endpoint_scale,
            CANOPY_STATUS_LOGITS_AUX: status_logits,
        }

        thermal = fields["thermosoil_previous_step_state"]
        ptn_key = "thermal.thermosoil_previous_step_state.ptn.normalized_tendency"
        stemp_key = "thermal.thermosoil_previous_step_state.stempdiag.normalized_tendency"
        ptn_end = jnp.asarray(thermal["ptn"], dtype=jnp.float64) + STEMPDIAG_TENDENCY_BOUND * jnp.asarray(
            result[ptn_key], dtype=jnp.float64
        )
        stemp_end = thermosoil_diaglev_from_ptn(
            ptn=ptn_end,
            veget_max=fields["diffuco_previous_step_state"]["veget_max"],
            nslm=self.static(inputs, "hydrol_zz_mm").size,
        )
        result[stemp_key] = (
            stemp_end - jnp.asarray(thermal["stempdiag"], dtype=jnp.float64)
        ) / STEMPDIAG_TENDENCY_BOUND

        doc_key = self.contract.doc_field
        residual = jnp.asarray(result[doc_key], dtype=jnp.float64)
        baseline = self.doc_start_baseline(inputs)
        result[doc_key] = jnp.where(
            self.doc_active_mask(inputs, residual.ndim),
            compose_doc_baseline_residual(
                baseline,
                residual,
                self.output_scale[doc_key],
            ),
            0.0,
        )
        result["__mechanistic.doc_residual_coordinate"] = residual
        return result, auxiliary

    def post_update_outputs(
        self,
        prediction: ProcessPrediction,
        fast_update: ConservativeUpdateResult,
        inputs: AssembledDailyOperatorInput,
    ) -> dict[str, Any]:
        outputs = dict(prediction.named_outputs)
        residual = outputs.pop("__mechanistic.doc_residual_coordinate")
        fields, _ok_leak = self.updated_fast_owner_fields(
            prediction,
            fast_update,
            inputs,
        )
        hydrol = fields["hydrol_previous_step_state"]
        diffuco = fields["diffuco_previous_step_state"]
        thermal = fields["thermosoil_previous_step_state"]
        active = self.active_pft_indices
        forcing = _last_forcing_record(inputs)
        frac_nobio = jnp.asarray(diffuco["frac_nobio"], dtype=jnp.float64)
        roughness = condveg_main_minimal(
            snow=hydrol["snow"],
            snow_nobio=hydrol["snow_nobio"],
            snowrho=hydrol["snowrho"],
            snowdz=hydrol["snowdz"],
            veget=diffuco["veget"],
            veget_max=diffuco["veget_max"],
            frac_nobio=frac_nobio,
            totfrac_nobio=jnp.sum(frac_nobio, axis=1),
            zlev=jnp.full(
                (frac_nobio.shape[0],),
                self.static(inputs, "measurement_heights")[0],
                dtype=jnp.float64,
            ),
            height=diffuco["height"],
            temp_air=forcing[self.forcing_indices["Tair"]][None],
            pb=forcing[self.forcing_indices["PSurf"]][None] / 100.0,
            u=forcing[self.forcing_indices["Wind_N"]][None],
            v=forcing[self.forcing_indices["Wind_E"]][None],
            lai=diffuco["lai"],
            emis_scal=1.0,
            impaze=False,
            rough_dyn=True,
            ok_explicitsnow=True,
        )
        outputs["endpoint.fast_state.diffuco_previous_step_state.z0m"] = roughness.z0m
        outputs["endpoint.fast_state.diffuco_previous_step_state.roughheight"] = roughness.roughheight
        outputs["endpoint.fast_state.diffuco_previous_step_state.roughheight_pft"] = _select_pft(
            roughness.roughheight_pft,
            active,
            axis=1,
        )

        mc = jnp.asarray(hydrol["mc"], dtype=jnp.float64)
        njsc = jnp.full(
            (mc.shape[0],),
            self.static(inputs, "njsc").reshape(-1)[0],
            dtype=jnp.int32,
        )
        litter = hydrol_litter_top_diagnostics(
            mc=mc,
            dz_mm=self.static(inputs, "hydrol_dz_mm"),
            dh_mm=hydrol_layer_thickness_from_zz_mm(self.static(inputs, "hydrol_zz_mm")),
            njsc=njsc,
        )
        litter_grid = hydrol_litter_grid_diagnostics(
            litter=litter,
            soiltile=self.static(inputs, "soiltile"),
            vegtot=1.0 - jnp.sum(frac_nobio, axis=1),
            njsc=njsc,
            k_lin_layer1=jnp.ones((50, mc.shape[0]), dtype=jnp.float64),
            ks=self.static(inputs, "hydrol_cwrr_ks"),
        )
        outputs["endpoint.fast_state.diffuco_previous_step_state.drysoil_frac"] = litter_grid.drysoil_frac

        final = thermosoil_final_state(
            ptn=thermal["ptn"],
            pkappa=jnp.zeros_like(thermal["ptn"]),
            veget_max=diffuco["veget_max"],
            shum_ngrnd_permalong=thermal["shum_ngrnd_permalong"],
        )
        outputs["endpoint.fast_state.thermosoil_previous_step_state.gtemp"] = final.gtemp

        def compact(value: Any, axis: int) -> Any:
            return _select_pft(value, active, axis)

        hydrology = reconstruct_hydrology_boundary_diagnostics(
            mc=hydrol["mc"],
            mcl=hydrol["mcl"],
            nroot=compact(hydrol["nroot"], 1),
            veget=compact(diffuco["veget"], 1),
            veget_max=compact(diffuco["veget_max"], 1),
            soiltile=self.static(inputs, "soiltile"),
            pref_soil_veg=compact(self.static(inputs, "pref_soil_veg"), 0),
            dz_mm=self.static(inputs, "hydrol_dz_mm"),
            njsc=njsc,
            evap_bare_lim_ns=hydrol["evap_bare_lim_ns"],
        )
        for name, value in hydrology.items():
            outputs[f"endpoint.fast_state.hydrol_previous_step_state.{name}"] = value

        coefficients = reconstruct_thermal_coefficients(
            **self.thermal_coefficient_inputs(fields, inputs)
        )
        for name, value in coefficients.items():
            component = (
                "enerbil_previous_step_state"
                if name in {"soilcap_pft", "soilflx_pft"}
                else ("thermosoil_previous_step_state")
            )
            outputs[f"endpoint.fast_state.{component}.{name}"] = value

        doc_key = self.contract.doc_field
        outputs[doc_key] = jnp.where(
            self.doc_active_mask(inputs, residual.ndim),
            compose_doc_baseline_residual(
                self.doc_trapezoid_baseline(prediction, fast_update, inputs),
                residual,
                self.output_scale[doc_key],
            ),
            0.0,
        )
        missing = sorted(set(self.contract.exact_fields) - set(outputs))
        if missing:
            raise ValueError(f"mechanistic exact materialization is incomplete: {missing}")
        return outputs


__all__ = [
    "DailyBoundaryReconstructor",
    "compose_doc_baseline_residual",
    "invert_doc_baseline_residual",
    "reconstruct_thermal_coefficients",
    "updated_fast_state_fields",
]
