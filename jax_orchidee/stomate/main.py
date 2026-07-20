"""PFT14 paper-case orchestration for ``stomate_main``.

This module owns process order only.  Scientific algebra remains in the
audited daily, season, carbon, and soil-carbon owners.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, NamedTuple

import jax.numpy as jnp

from jax_orchidee.stomate.carbon_kernels import stomate_littercalc_entry_prep
from jax_orchidee.stomate.daily import reset_daily_on_slow, stomate_daily_process_fold_from_entries
from jax_orchidee.stomate.integration import (
    stomate_daily_prescribe_constraints_alloc_kill_gap_turnover_explicit,
    stomate_ok_leak_explicit,
)
from jax_orchidee.stomate.permafrost import (
    StomatePermafrostControls,
    stomate_permafrost_decomposition_controls,
)
from jax_orchidee.stomate.season import (
    season_annual_step,
    season_biometeorology_step,
    season_memory_step,
)
from jax_orchidee.stomate.soilcarbon_kernels import (
    ICO2AQ,
    IDOCL,
    IDOCR,
    IDRAINAGE,
    IFLOODED,
    IRUNOFF,
)


STOMATE_MAIN_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 2896-3032",
    "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3187-3583",
    "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3589-4364",
    "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3035-3124",
    "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 4861-4937",
    "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 4943-5000",
    "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 5033-5049",
)


@dataclass(frozen=True)
class StomateMainPFT14Switches:
    """Structural switches fixed by the 250919 paper configuration."""

    nvm: int = 14
    ok_stomate: bool = True
    ok_leak: bool = True
    ok_pc: bool = False
    do_daily_permafrost: bool = False
    ok_peat: bool = False
    peat_occur: bool = False
    perma_peat: bool = True
    agri_peat: bool = False
    dyn_peat: bool = False
    ok_dgvm: bool = False
    land_cover_change: bool = False
    disable_fire: bool = True
    agriculture: bool = True
    harvest_agri: bool = True
    lpj_gap_const_mort: bool = True
    tf_doc: bool = True
    cryoturbate: bool = False
    ch4_calcul: bool = False
    spinup_analytic: bool = False
    ok_pheno: bool = True
    ok_rotate: bool = False
    nitrogen_use: bool = False
    ok_laidev: tuple[bool, ...] = (False,) * 14
    stomate_forcing_name: str = "NONE"
    stomate_cforcing_name: str = "NONE"
    cforcing_permafrost_name: str = "NONE"


class StomateOutputBoundary(NamedTuple):
    """Pure-data replacement for history/XIOS side effects."""

    history: dict[str, object]
    xios: dict[str, object]
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3463-3473",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 4861-4936",
        "fortran_source/ORCHIDEE/src_stomate/stomate_lpj.f90::StomateLpj lines 1677-1699 and 2200-2302",
    )


class StomateActivityControlResult(NamedTuple):
    """Deep-moisture preparation and activity controls written by main."""

    mc_peat: jnp.ndarray
    mc_man: jnp.ndarray
    hsdeep_new: jnp.ndarray
    controls: StomatePermafrostControls
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3035-3124",
    )


class StomateMonthlyCarbonState(NamedTuple):
    """Persistent monthly accumulators allocated by ``stomate_init``."""

    co2_flux_monthly: jnp.ndarray
    harvest_above_monthly: jnp.ndarray
    cflux_prod_monthly: jnp.ndarray


class StomateMonthlyCarbonResult(NamedTuple):
    """Daily NEP, monthly diagnostics, and exact next-call accumulator state."""

    co2_flux_daily: jnp.ndarray
    nep: jnp.ndarray
    co2_flux_monthly_history: jnp.ndarray | None
    nonbiofrac_history: jnp.ndarray | None
    net_cflux_prod_monthly_tot: jnp.ndarray | None
    net_harvest_above_monthly_tot: jnp.ndarray | None
    net_co2_flux_monthly_sum: jnp.ndarray | None
    net_biosp_prod_monthly_tot: jnp.ndarray | None
    state: StomateMonthlyCarbonState
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 4851-4937",
    )


class StomateOutputFluxResult(NamedTuple):
    """SECHIBA-facing carbon fluxes from ``stomate_main`` lines 5033-5049."""

    resp_maint: jnp.ndarray
    resp_growth: jnp.ndarray
    resp_hetero: jnp.ndarray
    co2_flux: jnp.ndarray
    temp_growth: jnp.ndarray
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 5033-5049",
    )


class StomateFixedPathControlResult(NamedTuple):
    """Runtime state owned by fixed structural arms in ``stomate_main``."""

    tday_counter: jnp.ndarray
    vday_counter: jnp.ndarray
    phenology_enabled: bool
    provenance: tuple[str, ...] = (
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 2896-2896",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3514-3766",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 3940-4584",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 4732-4839",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 4938-4938",
        "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_main lines 5108-5226",
    )


class StomateMainPFT14Result(NamedTuple):
    """Owner results and exact next-call daily state from one main step."""

    entry: object
    daily: object
    litter_prep: object
    ok_leak: object
    season_memory: object | None
    season_biometeorology: object | None
    season_annual: object | None
    lpj: object | None
    activity_control: StomateActivityControlResult
    monthly_carbon: StomateMonthlyCarbonResult | None
    output_fluxes: StomateOutputFluxResult | None
    fixed_path_control: StomateFixedPathControlResult
    daily_state: dict[str, object]
    carbon_state: dict[str, object]
    output: StomateOutputBoundary
    provenance: tuple[str, ...] = STOMATE_MAIN_PROVENANCE


def validate_stomate_main_pft14_switches(switches: StomateMainPFT14Switches) -> None:
    """Reject structural branches outside the audited paper-case graph."""

    expected = {
        "nvm": 14,
        "ok_stomate": True,
        "ok_leak": True,
        "ok_pc": False,
        "do_daily_permafrost": False,
        "ok_peat": False,
        "peat_occur": False,
        "perma_peat": True,
        "agri_peat": False,
        "dyn_peat": False,
        "ok_dgvm": False,
        "land_cover_change": False,
        "disable_fire": True,
        "agriculture": True,
        "harvest_agri": True,
        "lpj_gap_const_mort": True,
        "tf_doc": True,
        "cryoturbate": False,
        "ch4_calcul": False,
        "spinup_analytic": False,
        "ok_pheno": True,
        "ok_rotate": False,
        "nitrogen_use": False,
    }
    mismatches = tuple(
        f"{name}={getattr(switches, name)!r} (required {value!r})"
        for name, value in expected.items()
        if getattr(switches, name) != value
    )
    if len(switches.ok_laidev) != switches.nvm or any(switches.ok_laidev):
        mismatches += ("ok_laidev must contain 14 fixed False entries",)
    for name in ("stomate_forcing_name", "stomate_cforcing_name", "cforcing_permafrost_name"):
        if getattr(switches, name).strip().upper() != "NONE":
            mismatches += (f"{name} must be 'NONE'",)
    if mismatches:
        raise ValueError("unsupported stomate_main structural switch: " + "; ".join(mismatches))


def stomate_fixed_path_control(
    *,
    initialized: bool,
    do_slow: bool,
    end_of_year: bool,
    dt_stomate: float,
    one_day: float,
    tday_counter,
    vday_counter,
    switches: StomateMainPFT14Switches,
) -> StomateFixedPathControlResult:
    """Own fixed-switch and runtime calendar arms of ``stomate_main``.

    Fortran provenance: ``stomate.f90::stomate_main`` lines 2896,
    3514-3766, 3940-4584, 4732-4839, 4938, and 5108-5226. Child arms below
    disabled parent switches perform no substitute calculation.
    """

    if not initialized:
        raise RuntimeError("stomate_main: Initialization not yet done")
    validate_stomate_main_pft14_switches(switches)
    dt_days = jnp.asarray(dt_stomate) / jnp.asarray(one_day)
    tday = jnp.asarray(tday_counter)
    vday = jnp.asarray(vday_counter)
    if do_slow:
        tday = tday + dt_days
        vday = vday + dt_days
        if end_of_year:
            tday = jnp.zeros_like(tday)
    return StomateFixedPathControlResult(tday, vday, switches.ok_pheno)


def _replace_matching(target: dict[str, object], source: Mapping[str, object]) -> None:
    for name, value in source.items():
        if name in target:
            target[name] = value


def _set_owned(target: dict[str, object], owned: Mapping[str, object], *, boundary: str) -> None:
    conflicts = tuple(
        f"{name}={target[name]!r} (required {value!r})"
        for name, value in owned.items()
        if name in target and target[name] != value
    )
    if conflicts:
        raise ValueError(f"{boundary} overrides fixed structural fields: " + "; ".join(conflicts))
    target.update(owned)


def _state_values(*results: object | None) -> dict[str, object]:
    values: dict[str, object] = {}
    for result in results:
        if result is None:
            continue
        state = getattr(result, "state", None)
        if state is not None and hasattr(state, "_asdict"):
            values.update(state._asdict())
        if hasattr(result, "herbivores"):
            values["herbivores"] = result.herbivores
    return values


def _extend_hydrol_moisture(shumdiag, *, ndeep: int, name: str) -> jnp.ndarray:
    moisture = jnp.asarray(shumdiag)
    if moisture.ndim != 2:
        raise ValueError(f"{name} must have shape (nland, nslm)")
    nland, nslm = moisture.shape
    if nland < 1 or nslm < 1 or nslm > ndeep:
        raise ValueError(f"{name} requires 1 <= nslm <= ndeep")
    if nslm == ndeep:
        return moisture
    return jnp.concatenate(
        (moisture, jnp.broadcast_to(moisture[:, -1:], (nland, ndeep - nslm))),
        axis=1,
    )


def stomate_activity_control_precomputation(
    *,
    tdeep,
    hsdeep,
    shumdiag_peat,
    shumdiag_man,
    zz_deep,
    poor_soils,
    frozen_respiration_func: int,
    is_peat,
    tau_peat=3.1536e8,
    z_tau=1.0e6,
    flux_tot_coeff=(1.2, 1.4, 0.75),
    one_day=86400.0,
) -> StomateActivityControlResult:
    """Implement the fixed-paper activity-control precomputation.

    Fortran provenance: ``src_stomate/stomate.f90``, ``stomate_main``, lines
    3035-3124. The paper graph fixes ``perma_peat=.TRUE.``,
    ``agri_peat=.FALSE.``, and ``nvm=14``. Hydrology layer moisture is extended
    with its last layer exactly as lines 3047-3051 and 3061-3065 specify.
    """

    temp = jnp.asarray(tdeep)
    humidity = jnp.asarray(hsdeep)
    if temp.ndim != 3 or temp.shape != humidity.shape:
        raise ValueError("tdeep and hsdeep must share shape (nland, ndeep, 14)")
    nland, ndeep, nvm = temp.shape
    if nland < 1 or nvm != 14:
        raise ValueError("activity controls require at least one land point and nvm=14")
    mc_peat = _extend_hydrol_moisture(shumdiag_peat, ndeep=ndeep, name="shumdiag_peat")
    mc_man = _extend_hydrol_moisture(shumdiag_man, ndeep=ndeep, name="shumdiag_man")
    if mc_peat.shape[0] != nland or mc_man.shape[0] != nland:
        raise ValueError("hydrology moisture land dimension must match tdeep")
    peat_mask = jnp.asarray(is_peat, dtype=bool)
    if peat_mask.shape != (nvm,):
        raise ValueError("is_peat must have shape (14,)")
    hsdeep_new = jnp.where(peat_mask[None, None, :], mc_peat[:, :, None], humidity)
    poor = jnp.asarray(poor_soils)
    if poor.shape != (nland,):
        raise ValueError("poor_soils must have shape (nland,)")

    controls = stomate_permafrost_decomposition_controls(
        temp - 273.15,
        humidity,
        zz_deep,
        mc_peat,
        poor,
        frozen_respiration_func=frozen_respiration_func,
        perma_peat=True,
        agri_peat=False,
        is_peat=peat_mask,
        tau_peat=tau_peat,
        z_tau=z_tau,
        flux_tot_coeff=flux_tot_coeff,
        one_day=one_day,
    )
    return StomateActivityControlResult(mc_peat, mc_man, hsdeep_new, controls)


def stomate_nep_monthly_step(
    state: StomateMonthlyCarbonState,
    *,
    resp_maint_d,
    resp_growth_d,
    resp_hetero_d,
    co2_fire,
    co2_to_bm_dgvm,
    gpp_daily,
    veget_cov_max,
    contfrac,
    harvest_above,
    convflux,
    cflux_prod10,
    cflux_prod100,
    resolution,
    totfrac_nobio,
    end_of_month: bool,
    one_day=86400.0,
) -> StomateMonthlyCarbonResult:
    """Advance lines 4851-4937 without history, MPI, or mutable globals.

    History payloads are returned as data. The global totals equal the Fortran
    reduced values when the supplied landpoint axis is the complete domain.
    """

    gpp = jnp.asarray(gpp_daily)
    pft_shape = gpp.shape
    if gpp.ndim != 2:
        raise ValueError("daily carbon PFT fields must have shape (nland, nvm)")
    pft_fields = {
        "resp_maint_d": resp_maint_d,
        "resp_growth_d": resp_growth_d,
        "resp_hetero_d": resp_hetero_d,
        "co2_fire": co2_fire,
        "co2_to_bm_dgvm": co2_to_bm_dgvm,
        "veget_cov_max": veget_cov_max,
    }
    arrays = {name: jnp.asarray(value) for name, value in pft_fields.items()}
    bad = tuple(name for name, value in arrays.items() if value.shape != pft_shape)
    if bad:
        raise ValueError("monthly PFT fields must match gpp_daily: " + ", ".join(bad))
    nland, nvm = pft_shape
    co2_month = jnp.asarray(state.co2_flux_monthly)
    harvest_month = jnp.asarray(state.harvest_above_monthly)
    product_month = jnp.asarray(state.cflux_prod_monthly)
    harvest = jnp.asarray(harvest_above)
    products = tuple(jnp.asarray(value) for value in (convflux, cflux_prod10, cflux_prod100))
    resolution_arr = jnp.asarray(resolution)
    contfrac_arr = jnp.asarray(contfrac)
    nonbio = jnp.asarray(totfrac_nobio)
    if co2_month.shape != pft_shape:
        raise ValueError("state.co2_flux_monthly must match gpp_daily")
    if harvest_month.shape != (nland,) or harvest.shape != (nland,):
        raise ValueError("harvest monthly/current fields must have shape (nland,)")
    if any(value.shape != product_month.shape for value in products) or product_month.shape[0] != nland:
        raise ValueError("product monthly/current fields must share shape (nland, nwp)")
    if resolution_arr.shape != (nland, 2) or contfrac_arr.shape != (nland,) or nonbio.shape != (nland,):
        raise ValueError("resolution, contfrac, and totfrac_nobio have invalid land shapes")

    co2_daily = arrays["resp_maint_d"] + arrays["resp_growth_d"] + arrays["resp_hetero_d"]
    co2_daily = co2_daily + arrays["co2_fire"] - arrays["co2_to_bm_dgvm"] - gpp
    nep = jnp.sum(co2_daily * arrays["veget_cov_max"], axis=1) / 1.0e3 / jnp.asarray(one_day) * contfrac_arr
    co2_accum = co2_month + co2_daily
    harvest_accum = harvest_month + harvest
    product_accum = product_month + products[0] + products[1] + products[2]

    if not bool(end_of_month):
        next_state = StomateMonthlyCarbonState(co2_accum, harvest_accum, product_accum)
        return StomateMonthlyCarbonResult(
            co2_daily, nep, None, None, None, None, None, None, next_state
        )

    co2_history = co2_accum
    area_contfrac = resolution_arr[:, 0] * resolution_arr[:, 1] * contfrac_arr
    scaled_co2 = co2_accum.at[:, 1:].multiply(area_contfrac[:, None])
    net_co2 = jnp.sum(scaled_co2[:, 1:] * arrays["veget_cov_max"][:, 1:]) * 1.0e-15
    net_products = jnp.sum(jnp.sum(product_accum, axis=1) * area_contfrac) * 1.0e-15
    net_harvest = jnp.sum(harvest_accum * area_contfrac) * 1.0e-15
    net_biosphere = net_co2 + net_products + net_harvest
    next_state = StomateMonthlyCarbonState(
        jnp.zeros_like(co2_accum),
        jnp.zeros_like(harvest_accum),
        jnp.zeros_like(product_accum),
    )
    return StomateMonthlyCarbonResult(
        co2_daily,
        nep,
        co2_history,
        nonbio,
        net_products,
        net_harvest,
        net_co2,
        net_biosphere,
        next_state,
    )


def stomate_output_fluxes(
    *,
    resp_maint_radia,
    resp_growth_d,
    resp_hetero_radia,
    co2_fire,
    co2_to_bm_dgvm,
    veget_cov_max,
    gpp,
    t2m_month,
    dt_sechiba,
    one_day=86400.0,
) -> StomateOutputFluxResult:
    """Compute the exact STOMATE-to-SECHIBA flux packet, lines 5033-5049."""

    fields = {
        name: jnp.asarray(value)
        for name, value in {
            "resp_maint_radia": resp_maint_radia,
            "resp_growth_d": resp_growth_d,
            "resp_hetero_radia": resp_hetero_radia,
            "co2_fire": co2_fire,
            "co2_to_bm_dgvm": co2_to_bm_dgvm,
            "veget_cov_max": veget_cov_max,
            "gpp": gpp,
        }.items()
    }
    shape = fields["gpp"].shape
    if len(shape) != 2 or any(value.shape != shape for value in fields.values()):
        raise ValueError("STOMATE output flux fields must share shape (nland,nvm)")
    t2m_month = jnp.asarray(t2m_month)
    if t2m_month.shape != (shape[0],):
        raise ValueError("t2m_month must have shape (nland,)")

    cover = fields["veget_cov_max"]
    resp_maint = fields["resp_maint_radia"] * cover
    resp_maint = resp_maint.at[:, 0].set(0.0)
    resp_growth = fields["resp_growth_d"] * cover * jnp.asarray(dt_sechiba) / jnp.asarray(one_day)
    resp_hetero = fields["resp_hetero_radia"] * cover
    co2_flux = (
        resp_hetero
        + resp_maint
        + resp_growth
        + (fields["co2_fire"] - fields["co2_to_bm_dgvm"]) * cover / jnp.asarray(one_day)
        - fields["gpp"]
    )
    return StomateOutputFluxResult(resp_maint, resp_growth, resp_hetero, co2_flux, t2m_month - 273.15)


def _output_boundary(
    ok_leak: object,
    lpj: object | None,
    monthly_carbon: StomateMonthlyCarbonResult | None,
) -> StomateOutputBoundary:
    agg = ok_leak.doc_export.doc_exp_agg
    xios = {
        "zz_EXP_DIC_RUNOFF": agg[:, IRUNOFF, ICO2AQ],
        "zz_EXP_DIC_DRAIN": agg[:, IDRAINAGE, ICO2AQ],
        "zz_EXP_DIC_FLOOD": agg[:, IFLOODED, ICO2AQ],
        "zz_EXP_DOCr_RUNOFF": agg[:, IRUNOFF, IDOCR],
        "zz_EXP_DOCr_DRAIN": agg[:, IDRAINAGE, IDOCR],
        "zz_EXP_DOCr_FLOOD": agg[:, IFLOODED, IDOCR],
        "zz_EXP_DOCl_RUNOFF": agg[:, IRUNOFF, IDOCL],
        "zz_EXP_DOCl_DRAIN": agg[:, IDRAINAGE, IDOCL],
        "zz_EXP_DOCl_FLOOD": agg[:, IFLOODED, IDOCL],
    }
    history: dict[str, object] = {}
    if lpj is not None:
        post_npp = lpj.post_npp
        if post_npp.modelout_fields is not None:
            history.update(post_npp.modelout_fields)
    if monthly_carbon is not None:
        xios["nep"] = monthly_carbon.nep
        history["nep"] = monthly_carbon.nep
        if monthly_carbon.co2_flux_monthly_history is not None:
            history["CO2FLUX"] = monthly_carbon.co2_flux_monthly_history
            history["NONBIOFRAC"] = monthly_carbon.nonbiofrac_history
    return StomateOutputBoundary(history=history, xios=xios)


def _daily_state_after_step(fields: Mapping[str, object], do_slow: bool) -> dict[str, object]:
    state = {name: jnp.asarray(value) for name, value in fields.items()}
    if not do_slow:
        return state
    extrema = {"t2m_min_daily", "t2m_max_daily"}
    state = reset_daily_on_slow(state, True, tuple(name for name in state if name not in extrema))
    if "t2m_min_daily" in state:
        state["t2m_min_daily"] = jnp.full_like(state["t2m_min_daily"], 1.0e33)
    if "t2m_max_daily" in state:
        state["t2m_max_daily"] = jnp.full_like(state["t2m_max_daily"], -1.0e33)
    return state


def stomate_main_pft14_step(
    *,
    entry_payload: Mapping[str, object],
    accumulator_state: object,
    maintenance_inputs: Mapping[str, object],
    turnover_daily: object,
    bm_to_litter: object,
    ok_leak_inputs: Mapping[str, object],
    activity_control_inputs: Mapping[str, object],
    dt_sechiba: float,
    dt_stomate: float,
    do_slow: bool,
    switches: StomateMainPFT14Switches | None = None,
    season_memory_state: object | None = None,
    season_memory_inputs: Mapping[str, object] | None = None,
    season_biometeorology_state: object | None = None,
    season_biometeorology_inputs: Mapping[str, object] | None = None,
    season_annual_state: object | None = None,
    season_annual_inputs: Mapping[str, object] | None = None,
    lpj_inputs: Mapping[str, Mapping[str, object] | None] | None = None,
    monthly_carbon_state: StomateMonthlyCarbonState | None = None,
    monthly_carbon_inputs: Mapping[str, object] | None = None,
    output_flux_inputs: Mapping[str, object] | None = None,
    initialized: bool = True,
    end_of_year: bool = False,
    tday_counter: object = 0.0,
    vday_counter: object = 0.0,
    one_day: float = 86400.0,
) -> StomateMainPFT14Result:
    """Execute the source-backed fixed-switch PFT14 ``stomate_main`` chain.

    ``do_slow`` is a runtime time-boundary branch.  Multi-landpoint arrays and
    continuous PFT14 parameters pass unchanged to their scientific owners.
    """

    fixed = switches or StomateMainPFT14Switches()
    validate_stomate_main_pft14_switches(fixed)
    if float(one_day) != 86400.0:
        raise ValueError("one_day is the fixed Fortran source constant 86400 seconds")
    gpp = jnp.asarray(entry_payload["gpp"])
    if gpp.ndim != 2 or gpp.shape[1] != fixed.nvm:
        raise ValueError("entry_payload['gpp'] must have shape (nland, 14)")
    if gpp.shape[0] < 1:
        raise ValueError("stomate_main requires at least one land point")
    fixed_path_control = stomate_fixed_path_control(
        initialized=initialized,
        do_slow=bool(do_slow),
        end_of_year=bool(end_of_year),
        dt_stomate=dt_stomate,
        one_day=one_day,
        tday_counter=tday_counter,
        vday_counter=vday_counter,
        switches=fixed,
    )

    activity_args = dict(activity_control_inputs)
    fixed_activity = ("perma_peat", "agri_peat", "nvm", "one_day")
    overlap = tuple(name for name in fixed_activity if name in activity_args)
    if overlap:
        raise ValueError("activity_control_inputs must not override fixed fields: " + ", ".join(overlap))
    activity_control = stomate_activity_control_precomputation(**activity_args, one_day=one_day)
    if activity_control.controls.prmfrst_soilc_tempctrl.shape != gpp.shape[:1] + (activity_control.mc_peat.shape[1], fixed.nvm):
        raise ValueError("activity-control land/PFT dimensions must match entry_payload['gpp']")

    maint = dict(maintenance_inputs)
    try:
        resp_maint_part_current = maint.pop("resp_maint_part_current")
    except KeyError as exc:
        raise ValueError("maintenance_inputs requires resp_maint_part_current") from exc
    daily = stomate_daily_process_fold_from_entries(
        entry_payloads=(dict(entry_payload),),
        accumulator_state=accumulator_state,
        resp_maint_part_current=resp_maint_part_current,
        do_slow_flags=(bool(do_slow),),
        dt_sechiba=dt_sechiba,
        dt_stomate=dt_stomate,
        peat_occur=False,
        dates=(entry_payload.get("date"),),
        retain_step_results=True,
        **maint,
    )
    if not daily.ok:
        raise ValueError("incomplete STOMATE daily owner boundary: " + ", ".join(daily.missing_inputs))

    leak_args = dict(ok_leak_inputs)
    owned_leak = {
        "turnover",
        "bm_to_litter",
        "soil_mc_32l",
        "resp_maint_part_radia",
        "flood_root_radia",
        "dt_days",
        "nslm",
        "ndeep",
        "perma_peat",
        "ok_cryoturb",
        "fbact_litter",
        "fbact_soilcarbon",
        "fbact_doc",
        "tprof",
    }
    overlap = tuple(sorted(owned_leak.intersection(leak_args)))
    if overlap:
        raise ValueError("ok_leak_inputs must not override orchestration-owned fields: " + ", ".join(overlap))
    soil_mc = jnp.asarray(leak_args["soil_mc"])
    carbon_32l = jnp.asarray(leak_args["carbon_32l"])
    nslm = soil_mc.shape[1]
    ndeep = carbon_32l.shape[3]
    litter_prep = stomate_littercalc_entry_prep(
        soil_mc,
        turnover_daily,
        bm_to_litter,
        dt_sechiba=dt_sechiba,
        nslm=nslm,
        ndeep=ndeep,
    )
    maintenance_step = daily.maintenance.step_results[-1]
    leak_args.update(
        turnover=litter_prep.turnover_littercalc,
        bm_to_litter=litter_prep.bm_to_littercalc,
        soil_mc_32l=litter_prep.soil_mc_32l,
        resp_maint_part_radia=maintenance_step.resp_maint_part,
        flood_root_radia=daily.maintenance.flood_root_radia,
        dt_days=jnp.asarray(dt_sechiba) / jnp.asarray(one_day),
        nslm=nslm,
        ndeep=ndeep,
        perma_peat=True,
        ok_cryoturb=False,
        fbact_litter=activity_control.controls.prmfrst_soilc_tempctrl,
        fbact_soilcarbon=activity_control.controls.prmfrst_soilc_tempctrl,
        fbact_doc=activity_control.controls.prmfrst_soilc_tempctrl_doc,
        tprof=jnp.asarray(activity_args["tdeep"]),
    )
    ok_leak = stomate_ok_leak_explicit(**leak_args)

    memory = biometeorology = annual = lpj = monthly_carbon = None
    supplied_daily_process = (
        season_memory_state,
        season_memory_inputs,
        season_biometeorology_state,
        season_biometeorology_inputs,
        season_annual_state,
        season_annual_inputs,
        lpj_inputs,
        monthly_carbon_state,
        monthly_carbon_inputs,
    )
    if not do_slow and any(value is not None for value in supplied_daily_process):
        raise ValueError("season/lpj inputs are only reachable when do_slow=True")
    if do_slow:
        required = {
            "season_memory_state": season_memory_state,
            "season_memory_inputs": season_memory_inputs,
            "season_biometeorology_state": season_biometeorology_state,
            "season_biometeorology_inputs": season_biometeorology_inputs,
            "season_annual_state": season_annual_state,
            "season_annual_inputs": season_annual_inputs,
            "lpj_inputs": lpj_inputs,
            "monthly_carbon_state": monthly_carbon_state,
            "monthly_carbon_inputs": monthly_carbon_inputs,
        }
        missing = tuple(name for name, value in required.items() if value is None)
        if missing:
            raise ValueError("do_slow STOMATE step requires " + ", ".join(missing))
        memory = season_memory_step(season_memory_state, **dict(season_memory_inputs))
        season_values = _state_values(memory)
        bio_args = dict(season_biometeorology_inputs)
        _replace_matching(bio_args, season_values)
        bio_args.setdefault(
            "firstcall", bool(dict(season_memory_inputs).get("firstcall", False))
        )
        biometeorology = season_biometeorology_step(season_biometeorology_state, **bio_args)
        annual_args = dict(season_annual_inputs)
        _replace_matching(annual_args, _state_values(memory, biometeorology))
        annual_args.setdefault(
            "firstcall", bool(dict(season_memory_inputs).get("firstcall", False))
        )
        annual = season_annual_step(season_annual_state, **annual_args)

        process_values = _state_values(memory, biometeorology, annual)
        groups = {name: (None if value is None else dict(value)) for name, value in dict(lpj_inputs).items()}
        for required_group in (
            "prescribe_inputs",
            "constraints_inputs",
            "phenology_inputs",
            "alloc_inputs",
            "post_npp_inputs",
        ):
            if groups.get(required_group) is None:
                raise ValueError(f"lpj_inputs requires {required_group}")
        for group in groups.values():
            if group is not None:
                _replace_matching(group, process_values)
        post = groups["post_npp_inputs"]
        for name in ("gpp_daily", "resp_maint_part"):
            if name in post:
                raise ValueError(f"post_npp_inputs must not override orchestration-owned field: {name}")
        post["gpp_daily"] = daily.daily_fields["gpp_daily"]
        post["resp_maint_part"] = daily.daily_fields["resp_maint_part"]
        _set_owned(
            groups["prescribe_inputs"],
            {"ok_dgvm": False, "lpj_gap_const_mort": True},
            boundary="prescribe_inputs",
        )
        _set_owned(
            post,
            {
                "ok_dgvm": False,
                "lpj_gap_const_mort": True,
                "wire_cover": True,
                "update_peatfrac": False,
                "ok_pc_cover": False,
                "wire_harvest_agri": True,
                "wire_vmax": True,
                "ok_nlim_vmax": False,
                "wire_output_diagnostics": True,
                "wire_modelout": True,
            },
            boundary="post_npp_inputs",
        )
        lpj = stomate_daily_prescribe_constraints_alloc_kill_gap_turnover_explicit(**groups)

        monthly_args = dict(monthly_carbon_inputs)
        monthly_owned = (
            "gpp_daily",
            "resp_maint_d",
            "resp_growth_d",
            "co2_fire",
            "co2_to_bm_dgvm",
            "harvest_above",
            "one_day",
        )
        overlap = tuple(name for name in monthly_owned if name in monthly_args)
        if overlap:
            raise ValueError("monthly_carbon_inputs must not override orchestration-owned fields: " + ", ".join(overlap))
        post_npp = lpj.post_npp
        npp_update = post_npp.daily_carbon.npp_update
        if post_npp.harvest_agri is None:
            raise ValueError("fixed HARVEST_AGRI paper path did not return harvest state")
        monthly_carbon = stomate_nep_monthly_step(
            monthly_carbon_state,
            gpp_daily=daily.daily_fields["gpp_daily"],
            resp_maint_d=npp_update.resp_maint,
            resp_growth_d=npp_update.resp_growth,
            co2_fire=jnp.zeros_like(daily.daily_fields["gpp_daily"]),
            co2_to_bm_dgvm=jnp.zeros_like(daily.daily_fields["gpp_daily"]),
            harvest_above=post_npp.harvest_agri.harvest_above,
            one_day=one_day,
            **monthly_args,
        )

    carbon_state = {
        **ok_leak.littercalc._asdict(),
        **ok_leak.soilcarbon._asdict(),
        "interception_storage": ok_leak.interception_storage,
        "prmfrst_soilc_tempctrl": activity_control.controls.prmfrst_soilc_tempctrl,
        "prmfrst_soilc_tempctrl_doc": activity_control.controls.prmfrst_soilc_tempctrl_doc,
    }
    output_fluxes = None
    if output_flux_inputs is not None:
        flux_args = dict(output_flux_inputs)
        for name in ("gpp", "dt_sechiba", "one_day"):
            if name in flux_args:
                raise ValueError(f"output_flux_inputs must not override orchestration-owned field: {name}")
        output_fluxes = stomate_output_fluxes(
            gpp=gpp,
            dt_sechiba=dt_sechiba,
            one_day=one_day,
            **flux_args,
        )
        carbon_state.update(output_fluxes._asdict())
        carbon_state.pop("provenance", None)
    if monthly_carbon is not None:
        carbon_state.update(monthly_carbon.state._asdict())
        carbon_state["co2_flux_daily"] = monthly_carbon.co2_flux_daily
    output = _output_boundary(ok_leak, lpj, monthly_carbon)
    return StomateMainPFT14Result(
        entry=daily.accumulator.step_results[-1].local_prep,
        daily=daily,
        litter_prep=litter_prep,
        ok_leak=ok_leak,
        season_memory=memory,
        season_biometeorology=biometeorology,
        season_annual=annual,
        lpj=lpj,
        activity_control=activity_control,
        monthly_carbon=monthly_carbon,
        output_fluxes=output_fluxes,
        fixed_path_control=fixed_path_control,
        daily_state=_daily_state_after_step(daily.daily_fields, bool(do_slow)),
        carbon_state=carbon_state,
        output=output,
    )
