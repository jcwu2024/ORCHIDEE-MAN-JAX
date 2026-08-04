"""Deterministic source-backed process cases for Gate-D1 gradients."""

from __future__ import annotations

import jax.numpy as jnp

from jax_orchidee.stomate.carbon_kernels import (
    ICARBON,
    ILEAF,
    ISAPABOVE,
    NLEAFAGES,
    NPARTS,
    allocation_step,
    gap_mortality_step,
    maintenance_respiration,
    vmax_step,
)
from research.daily_coarse_graining.physical_parameter_gradients import (
    PhysicalGradientComparison,
    compare_physical_parameter_gradient,
)

PFT14_INDEX = 13


def _maintenance_objective(column: int):
    npts, nvm, nslm = 1, 14, 2
    biomass = jnp.zeros((npts, nvm, NPARTS, 1), dtype=jnp.float64)
    biomass = biomass.at[0, PFT14_INDEX, ILEAF, ICARBON].set(100.0)
    biomass = biomass.at[0, PFT14_INDEX, ISAPABOVE, ICARBON].set(50.0)
    coeff_maint_zero = jnp.zeros((nvm, NPARTS), dtype=jnp.float64).at[PFT14_INDEX, :].set(0.01)
    baseline_slopes = jnp.zeros((nvm, 3), dtype=jnp.float64).at[PFT14_INDEX, 0].set(0.0876862)

    def objective(value):
        slopes = baseline_slopes.at[PFT14_INDEX, column].set(value)
        result = maintenance_respiration(
            biomass=biomass,
            t2m=jnp.asarray([293.15], dtype=jnp.float64),
            t2m_longterm=jnp.asarray([293.15], dtype=jnp.float64),
            stempdiag=jnp.full((npts, nslm), 283.15, dtype=jnp.float64),
            z_soil=jnp.asarray([0.0, 1.0, 2.0], dtype=jnp.float64),
            rprof=jnp.ones((npts, nvm), dtype=jnp.float64),
            sla_calc=jnp.full((npts, nvm), 0.02, dtype=jnp.float64),
            coeff_maint_zero=coeff_maint_zero,
            maint_resp_slope=slopes,
            ext_coeff=jnp.full(nvm, 0.5, dtype=jnp.float64),
            is_tree=jnp.ones(nvm, dtype=bool),
        )
        return jnp.sum(result.resp_maint_part[0, PFT14_INDEX, :])

    return objective


def _vmax_objective():
    npts, nvm = 1, 14
    leaf_age = jnp.full((npts, nvm, NLEAFAGES), 10.0, dtype=jnp.float64)
    leaf_frac = jnp.zeros_like(leaf_age).at[:, :, 0].set(1.0)

    def objective(value):
        vcmax25 = jnp.full(nvm, 50.0, dtype=jnp.float64).at[PFT14_INDEX].set(value)
        result = vmax_step(
            leaf_age=leaf_age,
            leaf_frac=leaf_frac,
            vcmax25=vcmax25,
            n_limfert=jnp.ones((npts, nvm), dtype=jnp.float64),
            leaf_timecst=jnp.full(nvm, 100.0, dtype=jnp.float64),
            leafagecrit=jnp.full(nvm, 100.0, dtype=jnp.float64),
            pheno_type=jnp.zeros(nvm, dtype=jnp.int32),
            leaf_tab=jnp.zeros(nvm, dtype=jnp.int32),
            ok_laidev=jnp.zeros(nvm, dtype=bool),
            dt_days=0.0,
        )
        return result.vcmax[0, PFT14_INDEX]

    return objective


def _allocation_objective():
    npts, nvm, nslm = 1, 14, 2
    biomass = jnp.zeros((npts, nvm, NPARTS, 1), dtype=jnp.float64)
    biomass = biomass.at[0, PFT14_INDEX, ILEAF, ICARBON].set(10.0)
    leaf_age = jnp.zeros((npts, nvm, NLEAFAGES), dtype=jnp.float64)
    leaf_frac = jnp.zeros_like(leaf_age).at[0, PFT14_INDEX, 0].set(1.0)
    pft = jnp.zeros((npts, nvm), dtype=jnp.float64)
    active = pft.at[0, PFT14_INDEX].set(1.0)
    is_tree = jnp.zeros(nvm, dtype=bool).at[PFT14_INDEX].set(True)

    def objective(value):
        alloc_min = jnp.full(nvm, 0.2, dtype=jnp.float64).at[PFT14_INDEX].set(value)
        result = allocation_step(
            lai=active,
            veget_max=active,
            senescence=jnp.zeros((npts, nvm), dtype=bool),
            when_growthinit=jnp.ones((npts, nvm), dtype=jnp.float64),
            moiavail_week=jnp.full((npts, nvm), 0.7, dtype=jnp.float64),
            tsoil_month=jnp.full((npts, nslm), 283.15, dtype=jnp.float64),
            soilhum_month=jnp.full((npts, nslm), 0.7, dtype=jnp.float64),
            biomass=biomass,
            age=jnp.zeros((npts, nvm), dtype=jnp.float64),
            leaf_age=leaf_age,
            leaf_frac=leaf_frac,
            z_soil=jnp.asarray([0.0, 1.0, 2.0], dtype=jnp.float64),
            sla_calc=jnp.full((npts, nvm), 0.02, dtype=jnp.float64),
            natural=jnp.ones(nvm, dtype=bool),
            pasture=jnp.zeros(nvm, dtype=bool),
            is_tree=is_tree,
            ok_LAIdev=jnp.zeros(nvm, dtype=bool),
            r0=jnp.full(nvm, 0.35, dtype=jnp.float64),
            s0=jnp.full(nvm, 0.35, dtype=jnp.float64),
            ext_coeff=jnp.full(nvm, 0.5, dtype=jnp.float64),
            lai_max=jnp.full(nvm, 12.0, dtype=jnp.float64),
            lai_max_to_happy=jnp.full(nvm, 0.5, dtype=jnp.float64),
            tau_leafinit=jnp.full(nvm, 10.0, dtype=jnp.float64),
            alloc_min=alloc_min,
            alloc_max=jnp.full(nvm, 0.8, dtype=jnp.float64),
            demi_alloc=jnp.full(nvm, 100.0, dtype=jnp.float64),
            alloc_agr_st=jnp.zeros(nvm, dtype=jnp.float64),
            alloc_agr_pn=jnp.zeros(nvm, dtype=jnp.float64),
        )
        return result.f_alloc[0, PFT14_INDEX, ISAPABOVE]

    return objective


def _gap_inputs(*, active: bool):
    npts, nvm = 1, 14
    biomass = jnp.zeros((npts, nvm, NPARTS, 1), dtype=jnp.float64)
    biomass = biomass.at[0, PFT14_INDEX, ILEAF, ICARBON].set(365.0)
    pft_present = jnp.zeros((npts, nvm), dtype=bool)
    if active:
        pft_present = pft_present.at[0, PFT14_INDEX].set(True)
    is_tree = jnp.zeros(nvm, dtype=bool).at[PFT14_INDEX].set(True)
    return {
        "npp_longterm": jnp.zeros((npts, nvm), dtype=jnp.float64),
        "turnover_longterm": jnp.zeros_like(biomass),
        "lm_lastyearmax": jnp.ones((npts, nvm), dtype=jnp.float64),
        "pft_present": pft_present,
        "biomass": biomass,
        "ind": jnp.ones((npts, nvm), dtype=jnp.float64),
        "bm_to_litter": jnp.zeros_like(biomass),
        "t2m_min_daily": jnp.asarray([280.0], dtype=jnp.float64),
        "tmin_spring_time": jnp.zeros((npts, nvm), dtype=jnp.float64),
        "sla_calc": jnp.full((npts, nvm), 0.02, dtype=jnp.float64),
        "natural": jnp.ones(nvm, dtype=bool),
        "is_tree": is_tree,
        "pasture": jnp.zeros(nvm, dtype=bool),
        "tmin_crit": jnp.full(nvm, jnp.inf, dtype=jnp.float64),
        "leaf_tab": jnp.zeros(nvm, dtype=jnp.int32),
        "pheno_type": jnp.zeros(nvm, dtype=jnp.int32),
    }


def _residence_time_objective(*, active: bool):
    inputs = _gap_inputs(active=active)

    def objective(value):
        residence_time = jnp.full(14, 30.0, dtype=jnp.float64).at[PFT14_INDEX].set(value)
        result = gap_mortality_step(
            **inputs,
            availability_fact=jnp.full(14, 0.14, dtype=jnp.float64),
            residence_time=residence_time,
            dt_days=1.0,
            lpj_gap_const_mort=True,
        )
        return result.biomass[0, PFT14_INDEX, ILEAF, ICARBON]

    return objective


def _availability_threshold_objective():
    inputs = _gap_inputs(active=True)

    def objective(value):
        availability = jnp.full(14, 0.14, dtype=jnp.float64).at[PFT14_INDEX].set(value)
        result = gap_mortality_step(
            **inputs,
            availability_fact=availability,
            residence_time=jnp.full(14, 30.0, dtype=jnp.float64),
            dt_days=1.0,
            lpj_gap_const_mort=False,
            min_avail=0.01,
        )
        return result.mortality_fraction[0, PFT14_INDEX]

    return objective


def run_process_gradient_cases() -> tuple[PhysicalGradientComparison, ...]:
    """Run deterministic process-level coverage for all Gate-D1 pair classes."""

    specs = (
        ("process.vmax__vcmax25", "smooth_active", _vmax_objective(), 63.2061836, 6.32061836e-4),
        ("process.maintenance__maint_resp_slope_c", "smooth_active", _maintenance_objective(0), 0.0876862, 1.0e-6),
        ("process.maintenance__maint_resp_slope_b", "smooth_active", _maintenance_objective(1), 0.001, 1.0e-7),
        ("process.allocation__alloc_min", "smooth_active", _allocation_objective(), 0.2019211, 2.0e-6),
        ("process.gap__residence_time", "smooth_active", _residence_time_objective(active=True), 50.6583452, 5.06583452e-4),
        ("process.gap_inactive__residence_time", "inactive", _residence_time_objective(active=False), 50.6583452, 5.06583452e-4),
        ("process.gap_threshold__availability_fact", "threshold_adjacent", _availability_threshold_objective(), 0.01, 1.0e-6),
    )
    return tuple(
        compare_physical_parameter_gradient(
            objective,
            pair_id=pair_id,
            pair_class=pair_class,
            parameter_value=value,
            finite_difference_step=step,
        )
        for pair_id, pair_class, objective, value, step in specs
    )
