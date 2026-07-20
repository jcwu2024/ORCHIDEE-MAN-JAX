"""Source-faithful helpers owned by STOMATE initialization procedures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from jax_orchidee.stomate.carbon_kernels import (
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
)


DATA_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_stomate/stomate_data.f90::data lines 261-348",
    "fortran_source/ORCHIDEE/src_stomate/stomate_data.f90::data lines 367-400",
    "fortran_source/ORCHIDEE/src_stomate/stomate_data.f90::data lines 408-426",
    "fortran_source/ORCHIDEE/src_stomate/stomate_data.f90::data lines 453-457 and 504-538",
    "fortran_source/ORCHIDEE/src_stomate/stomate_data.f90::data lines 574-588",
)
SORT_ASCENDING_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_stomate/stomate.f90::sort_ascending lines 12278-12311",
)
STOMATE_INIT_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_init lines 6296-6330",
    "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_init lines 7178-7379",
    "fortran_source/ORCHIDEE/src_stomate/stomate.f90::stomate_init lines 8250-8275",
)


@dataclass(frozen=True)
class StomateDataConstants:
    alpha_tree: float
    alpha_grass: float
    bm_sapl_leaf: tuple[float, float, float, float]
    pipe_tune1: float
    mass_ratio_heart_sap: float
    pipe_k1: float
    bm_sapl_carbres: float
    bm_sapl_sapabove: float
    pipe_density: float
    pipe_tune2: float
    pipe_tune3: float
    dia_coeff: tuple[float, float]
    bm_sapl_heartabove: float
    bm_sapl_heartbelow: float
    init_sapl_mass_leaf_nat: float
    init_sapl_mass_leaf_agri: float
    init_sapl_mass_carbres: float
    init_sapl_mass_root: float
    init_sapl_mass_fruit: float
    migrate_tree: float
    migrate_grass: float
    pipe_tune4: float
    maxdia_coeff: tuple[float, float]
    cn_sapl_init: float
    lai_initmin_tree: float
    lai_initmin_grass: float
    undef: float
    min_stomate: float
    zero_celsius: float = 273.15


class StomateDataResult(NamedTuple):
    bm_sapl: jnp.ndarray
    migrate: jnp.ndarray
    maxdia: jnp.ndarray
    cn_sapl: jnp.ndarray
    tmin_crit: jnp.ndarray
    tcm_crit: jnp.ndarray
    leaf_timecst: jnp.ndarray
    lai_initmin: jnp.ndarray
    provenance: tuple[str, ...] = DATA_PROVENANCE


def _vector(name: str, value, nvm: int, dtype=None) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.shape != (nvm,):
        raise ValueError(f"{name} must have shape ({nvm},), got {array.shape}")
    return array.copy()


def stomate_data_owner(
    *,
    bm_sapl,
    migrate,
    maxdia,
    cn_sapl,
    lai_initmin,
    sla,
    is_tree,
    pheno_type,
    ok_laidev,
    natural,
    is_grassland_manag,
    sp_densitesem,
    sp_pgrainmaxi,
    tmin_crit,
    tcm_crit,
    pheno_model,
    pheno_gdd_crit,
    senescence_type,
    senescence_temp,
    senescence_hum,
    nosenescence_hum,
    leafagecrit,
    nleafages: int,
    ok_dgvm: bool,
    use_age_class: bool,
    constants: StomateDataConstants,
) -> StomateDataResult:
    """Materialize PFT parameters in ``stomate_data::data`` source order.

    Fortran initializes the arrays before this routine and loops from PFT 2;
    therefore all destination arrays are explicit inputs and PFT 1 is retained.
    """

    bm = np.asarray(bm_sapl).copy()
    if bm.ndim != 3 or bm.shape[1] != NPARTS or bm.shape[2] <= ICARBON:
        raise ValueError("bm_sapl must have shape (nvm,12,nelements>=1)")
    nvm = bm.shape[0]
    scalar_vectors = {
        "migrate": migrate,
        "maxdia": maxdia,
        "cn_sapl": cn_sapl,
        "lai_initmin": lai_initmin,
        "sla": sla,
        "is_tree": is_tree,
        "pheno_type": pheno_type,
        "ok_laidev": ok_laidev,
        "natural": natural,
        "is_grassland_manag": is_grassland_manag,
        "sp_densitesem": sp_densitesem,
        "sp_pgrainmaxi": sp_pgrainmaxi,
        "tmin_crit": tmin_crit,
        "tcm_crit": tcm_crit,
        "pheno_model": pheno_model,
        "senescence_type": senescence_type,
        "senescence_hum": senescence_hum,
        "nosenescence_hum": nosenescence_hum,
        "leafagecrit": leafagecrit,
    }
    values = {name: _vector(name, value, nvm) for name, value in scalar_vectors.items()}
    gdd = np.asarray(pheno_gdd_crit)
    sen_temp = np.asarray(senescence_temp)
    if gdd.shape != (nvm, 3) or sen_temp.shape != (nvm, 3):
        raise ValueError("pheno_gdd_crit and senescence_temp must have shape (nvm,3)")
    if nleafages < 1:
        raise ValueError("nleafages must be positive")

    c = constants
    migrate_out = values["migrate"].astype(float)
    maxdia_out = values["maxdia"].astype(float)
    cn_out = values["cn_sapl"].astype(float)
    lai_out = values["lai_initmin"].astype(float)
    tmin_out = values["tmin_crit"].astype(float)
    tcm_out = values["tcm_crit"].astype(float)
    sla_values = values["sla"].astype(float)

    for j in range(1, nvm):
        if bool(values["is_tree"][j]):
            alpha = c.alpha_tree
            b = c.bm_sapl_leaf
            leaf = ((b[0] * c.pipe_tune1 * (c.mass_ratio_heart_sap * b[1] * sla_values[j] / (np.pi * c.pipe_k1)) ** b[2]) / sla_values[j]) ** b[3]
            bm[j, ILEAF, ICARBON] = leaf
            bm[j, ICARBRES, ICARBON] = c.bm_sapl_carbres * leaf if values["pheno_type"][j] != 1 else 0.0
            csa_sap = leaf / (c.pipe_k1 / sla_values[j])
            dia = (c.mass_ratio_heart_sap * csa_sap * c.dia_coeff[0] / np.pi) ** c.dia_coeff[1]
            sap = c.bm_sapl_sapabove * c.pipe_density * csa_sap * c.pipe_tune2 * dia**c.pipe_tune3
            bm[j, ISAPABOVE, ICARBON] = sap
            bm[j, ISAPBELOW, ICARBON] = sap
            bm[j, IHEARTABOVE, ICARBON] = c.bm_sapl_heartabove * sap
            bm[j, IHEARTBELOW, ICARBON] = c.bm_sapl_heartbelow * sap
        else:
            alpha = c.alpha_grass
            if bool(values["ok_laidev"][j]):
                if bool(values["natural"][j]):
                    raise ValueError("data lines 303-312: ok_LAIdev and natural cannot both be true")
                bm[j, ILEAF, ICARBON] = 0.0
                bm[j, ICARBRES, ICARBON] = values["sp_densitesem"][j] * values["sp_pgrainmaxi"][j]
            else:
                natural_or_managed = bool(values["natural"][j]) or bool(values["is_grassland_manag"][j])
                initial_leaf = c.init_sapl_mass_leaf_nat if natural_or_managed else c.init_sapl_mass_leaf_agri
                bm[j, ILEAF, ICARBON] = initial_leaf / sla_values[j]
                bm[j, ICARBRES, ICARBON] = c.init_sapl_mass_carbres * bm[j, ILEAF, ICARBON]
            bm[j, ISAPABOVE:IHEARTBELOW + 1, ICARBON] = 0.0

        bm[j, IROOT, ICARBON] = c.init_sapl_mass_root / alpha * bm[j, ILEAF, ICARBON]
        bm[j, IFRUIT, ICARBON] = c.init_sapl_mass_fruit * bm[j, ILEAF, ICARBON]
        bm[j, 8:12, ICARBON] = 0.0
        if (not ok_dgvm) and use_age_class:
            bm[j, :, ICARBON] *= 0.05

        migrate_out[j] = c.migrate_tree if bool(values["is_tree"][j]) else c.migrate_grass
        if bool(values["is_tree"][j]):
            maxdia_out[j] = (c.pipe_tune4 / ((c.pipe_tune2 * c.pipe_tune3) / c.maxdia_coeff[0] ** c.pipe_tune3)) ** (1.0 / (c.pipe_tune3 - 1.0)) * c.maxdia_coeff[1]
            cn_out[j] = c.cn_sapl_init
        else:
            maxdia_out[j] = c.undef
            cn_out[j] = 1.0

        tmin_out[j] = tmin_out[j] + c.zero_celsius if abs(tmin_out[j] - c.undef) > c.min_stomate else c.undef
        tcm_out[j] = tcm_out[j] + c.zero_celsius if abs(tcm_out[j] - c.undef) > c.min_stomate else c.undef
        model = str(values["pheno_model"][j]).strip()
        if model in {"moigdd", "humgdd"} and np.any(gdd[j] == c.undef):
            raise ValueError("data lines 453-457: phenology GDD parameters contain undef")
        sen_type = str(values["senescence_type"][j]).strip().lower()
        if sen_type in {"cold", "mixed"} and np.any(sen_temp[j] == c.undef):
            raise ValueError("data lines 504-508: senescence temperature contains undef")
        if sen_type in {"dry", "mixed"} and values["senescence_hum"][j] == c.undef:
            raise ValueError("data lines 519-523: senescence_hum is undef")
        if sen_type in {"dry", "mixed"} and values["nosenescence_hum"][j] == c.undef:
            raise ValueError("data lines 535-539: nosenescence_hum is undef")
        lai_out[j] = c.lai_initmin_tree if bool(values["is_tree"][j]) else c.lai_initmin_grass

    leaf_timecst = values["leafagecrit"].astype(float) / float(nleafages)
    return StomateDataResult(
        jnp.asarray(bm), jnp.asarray(migrate_out), jnp.asarray(maxdia_out), jnp.asarray(cn_out),
        jnp.asarray(tmin_out), jnp.asarray(tcm_out), jnp.asarray(leaf_timecst), jnp.asarray(lai_out)
    )


def sort_ascending_owner(array) -> jnp.ndarray:
    """Fortran selection sort, preserving strict-``<`` and NaN behavior."""

    result = np.asarray(array).copy()
    if result.ndim != 2:
        raise ValueError("array must have shape (kjpindex,months_num)")
    for i in range(result.shape[0]):
        for j in range(result.shape[1] - 1):
            minimum = result[i, j]
            location = j
            for k in range(j + 1, result.shape[1]):
                if result[i, k] < minimum:
                    minimum = result[i, k]
                    location = k
            result[i, j], result[i, location] = result[i, location], result[i, j]
    return jnp.asarray(result)


class StomateInitLifecycleResult(NamedTuple):
    diagnostic_index_fortran: int
    leak_daily_state: dict[str, jnp.ndarray]
    allocation_only_lines: tuple[str, ...]
    provenance: tuple[str, ...] = STOMATE_INIT_PROVENANCE


def stomate_init_lifecycle_owner(
    *,
    npts: int,
    nvm: int,
    ndeep: int,
    npool: int,
    nelements: int,
    nlitt: int,
    nstm: int,
    nslm: int,
    nflow: int,
    ncarb: int,
    diagnostic_index_fortran: int = 1,
    ok_stomate: bool = True,
    ok_dgvm: bool = False,
    ok_co2: bool = True,
    ok_leak: bool = True,
    dtype=jnp.float64,
) -> StomateInitLifecycleResult:
    """Own the five reachable lifecycle arms in ``stomate_init``.

    Lines 7178-7379 are allocation only. Their scientific initialization is
    performed at lines 8250-8275 and represented below by exact zero arrays.
    """

    dimensions = (npts, nvm, ndeep, npool, nelements, nlitt, nstm, nslm, nflow, ncarb)
    if any(value < 1 for value in dimensions):
        raise ValueError("all STOMATE dimensions must be positive")
    if (not ok_stomate) and ok_dgvm:
        raise ValueError("stomate_init lines 6318-6323: DGVM requires STOMATE")
    if (not ok_co2) and ok_stomate:
        raise ValueError("stomate_init lines 6325-6330: STOMATE requires CO2/GPP")
    if not ok_leak:
        raise NotImplementedError("paper PFT14 requires ok_leak=True at stomate_init lines 7178 and 8250")

    p, v, d, q, e = npts, nvm, ndeep, npool, nelements
    shapes = {
        "soilcarbon_input_DOC_daily": (p, v, d, q, e),
        "floodcarbon_input_daily": (p, v, q, e),
        "litter_above_Cforcing_daily": (p, nlitt, v, e),
        "litter_below_Cforcing_daily": (p, nlitt, v, d, e),
        "lignin_struc_above_Cforcing_daily": (p, v),
        "lignin_struc_below_Cforcing_daily": (p, v, d),
        "runoff_per_soil_Cforcing_daily": (p, nstm),
        "runoff2peat_Cforcing_daily": (p, nstm),
        "drainage_per_soil_Cforcing_daily": (p, nstm),
        "wat_flux_Cforcing_daily": (p, nslm, nstm),
        "soil_mc_32l_Cforcing_daily": (p, d, nstm),
        "soil_mc_Cforcing_daily": (p, nslm, nstm),
        "DOC_to_topsoil_Cforcing_daily": (p, nflow),
        "precip2ground_Cforcing_daily": (p, v),
        "interception_storage_Cforcing_daily": (p, v, e),
        "biomass_Cforcing_daily": (p, v, NPARTS, e),
        "fastr_Cforcing_daily": (p,),
        "precip2canopy_Cforcing_daily": (p, v),
        "canopy2ground_Cforcing_daily": (p, v),
        "DOC_to_subsoil_Cforcing_daily": (p, nflow),
        "erodepth_Cforcing_daily": (p, v),
        "seddep_Cforcing_daily": (p,),
        "pocdep_Cforcing_daily": (p, ncarb),
        "flood_frac_Cforcing_daily": (p,),
    }
    state = {name: jnp.zeros(shape, dtype=dtype) for name, shape in shapes.items()}
    diagnostic = min(max(int(diagnostic_index_fortran), 1), npts)
    return StomateInitLifecycleResult(
        diagnostic,
        state,
        ("lines 7178-7379: allocation contract only; values assigned at lines 8250-8275",),
    )
