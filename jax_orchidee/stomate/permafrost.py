"""Permafrost temperature/moisture controls for STOMATE carbon processes.

This module implements source-closed local algebra from ``microactem`` and
the immediate ``stomate_main`` conversion of returned time constants to
decomposition rates. It does not run deep carbon, litter, or soil-carbon pool
updates.
"""

from __future__ import annotations

from typing import NamedTuple

from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np


ZERO_CELSIUS = 273.15
STOMATE_TAU_SECONDS = 4.699e6
ONE_DAY_SECONDS = 86400.0
DEFAULT_FLUX_TOT_COEFF = (1.2, 1.4, 0.75)
FORTRAN_EPSILON_R4 = jnp.finfo(jnp.float32).eps

MICROACTEM_PROVENANCE = (
    "Fortran source truth: src_stomate/stomate_permafrost_soilcarbon.f90, "
    "microactem lines 2468-2703.",
    "Fortran source truth: src_stomate/stomate.f90, stomate_main lines "
    "3034-3124 calls microactem and converts returned time constants to "
    "day^-1 decomposition controls.",
    "Fortran source truth: src_parameters/constantes_soil_var.f90 lines "
    "174-175 defines tau_peat and z_tau defaults; paper run overrides are in "
    "configs/orchidee_man_250919.yaml structural_overrides.",
)


class MicroactemControlResult(NamedTuple):
    """Intermediate controls and final ``microactem`` time constants."""

    temp_control: jnp.ndarray
    moisture_control: jnp.ndarray
    peat_moisture_control: jnp.ndarray | None
    fbact_seconds: jnp.ndarray


class StomatePermafrostControls(NamedTuple):
    """Day^-1 controls passed onward from ``stomate_main``."""

    prmfrst_soilc_tempctrl: jnp.ndarray
    prmfrst_soilc_tempctrl_doc: jnp.ndarray
    fbact_seconds: jnp.ndarray
    fbact_doc_seconds: jnp.ndarray


def microactem_temperature_control(temp_celsius, frozen_respiration_func: int, *, epsilon=FORTRAN_EPSILON_R4):
    """Compute ``microactem`` temperature control.

    Fortran provenance: ``src_stomate/stomate_permafrost_soilcarbon.f90``,
    function ``microactem``, lines 2512-2563. ``temp`` is Celsius; Fortran
    immediately converts it to Kelvin before applying one of five frozen
    respiration branches.
    """

    temp_kelvin = jnp.asarray(temp_celsius) + ZERO_CELSIUS
    q10 = 2.0
    normal = jnp.exp(jnp.log(q10) * (temp_kelvin - (ZERO_CELSIUS + 30.0)) / 10.0)
    if int(frozen_respiration_func) == 0:
        result = jnp.minimum(1.0, normal)
    elif int(frozen_respiration_func) == 1:
        below = jnp.asarray(epsilon, dtype=temp_kelvin.dtype)
        linear = (temp_kelvin - (ZERO_CELSIUS - 1.0)) * jnp.exp(
            jnp.log(q10) * (ZERO_CELSIUS - (ZERO_CELSIUS + 30.0)) / 10.0
        )
        result = jnp.where(temp_kelvin > ZERO_CELSIUS, normal, jnp.where(temp_kelvin > ZERO_CELSIUS - 1.0, linear, below))
    elif int(frozen_respiration_func) == 2:
        below = jnp.asarray(epsilon, dtype=temp_kelvin.dtype)
        linear = ((temp_kelvin - (ZERO_CELSIUS - 3.0)) / 3.0) * jnp.exp(
            jnp.log(q10) * (ZERO_CELSIUS - (ZERO_CELSIUS + 30.0)) / 10.0
        )
        result = jnp.where(temp_kelvin > ZERO_CELSIUS, normal, jnp.where(temp_kelvin > ZERO_CELSIUS - 3.0, linear, below))
    elif int(frozen_respiration_func) == 3:
        frozen = jnp.exp(jnp.log(100.0) * (temp_kelvin - ZERO_CELSIUS) / 10.0) * jnp.exp(
            jnp.log(q10) * (-30.0) / 10.0
        )
        result = jnp.where(temp_kelvin > ZERO_CELSIUS, normal, frozen)
    elif int(frozen_respiration_func) == 4:
        frozen = jnp.exp(jnp.log(1000.0) * (temp_kelvin - ZERO_CELSIUS) / 10.0) * jnp.exp(
            jnp.log(q10) * (-30.0) / 10.0
        )
        result = jnp.where(temp_kelvin > ZERO_CELSIUS, normal, frozen)
    else:
        raise ValueError(f"frozen_respiration_func not in Fortran microactem list: {frozen_respiration_func}")
    return jnp.maximum(jnp.minimum(1.0, result), jnp.asarray(epsilon, dtype=temp_kelvin.dtype))


def microactem_moisture_control(moist_in, limit_decomp_moisture: int):
    """Compute non-peat ``microactem`` moisture control.

    Fortran provenance: ``src_stomate/stomate_permafrost_soilcarbon.f90``,
    function ``microactem``, lines 2571-2582. Branch 0 applies the standard
    ORCHIDEE quadratic clipped to [0.25, 1]; branch 1 sets moisture control to
    one for the DOC call.
    """

    moist = jnp.asarray(moist_in)
    if int(limit_decomp_moisture) == 0:
        result = -1.1 * moist * moist + 2.4 * moist - 0.29
        return jnp.maximum(0.25, jnp.minimum(1.0, result))
    if int(limit_decomp_moisture) == 1:
        return jnp.ones_like(moist)
    raise ValueError(f"limit_decomp_moisture not in Fortran microactem list: {limit_decomp_moisture}")


def _moyano_peat_moisture_table() -> np.ndarray:
    """Materialize the source-defined constant table outside JAX tracing."""

    mc = 0.01 + 0.02 * jnp.arange(45, dtype=jnp.float64)
    pcsr = (
        0.97509
        - 0.48212 * mc
        + 1.83997 * (mc**2)
        - 1.56379 * (mc**3)
        + 0.09867 * 1.2
        + 1.39944 * 0.05
        + 0.17938 * 0.3
        - 0.30307 * mc * 1.2
        - 0.30885 * mc * 0.3
    )
    sr = jnp.cumprod(pcsr)
    sr = sr / jnp.max(sr)
    ind = int(np.asarray(jnp.argmax(sr)))
    prefix = sr[: ind + 1]
    scaled_prefix = (prefix - jnp.min(prefix)) / jnp.max(prefix - jnp.min(prefix))
    return np.asarray(sr.at[: ind + 1].set(scaled_prefix), dtype=np.float64)


_MOYANO_PEAT_MOISTURE_TABLE = _moyano_peat_moisture_table()


def moyano_peat_moisture_lookup(mc_peat, *, epsilon=FORTRAN_EPSILON_R4):
    """Evaluate the peat moisture lookup table used by ``microactem``.

    Fortran provenance: ``src_stomate/stomate_permafrost_soilcarbon.f90``,
    function ``microactem``, lines 2605-2653. The same table appears as
    ``control_moist_func_peat`` in ``stomate_litter.f90`` lines 1516-1569.
    """

    corgmat = jnp.asarray(_MOYANO_PEAT_MOISTURE_TABLE, dtype=jnp.asarray(mc_peat).dtype)
    mc_ind = jnp.minimum(44, jnp.maximum(0, jnp.floor(jnp.asarray(mc_peat) / 0.02).astype(jnp.int32)))
    return jnp.minimum(1.0, jnp.maximum(jnp.asarray(epsilon, dtype=corgmat.dtype), corgmat[mc_ind]))


def microactem(
    temp_celsius,
    frozen_respiration_func: int,
    limit_decomp_moisture: int,
    moist_in,
    zi_soil,
    mc_peat,
    *,
    perma_peat: bool = False,
    agri_peat: bool = False,
    is_peat=None,
    tau_peat=3.1536e8,
    z_tau=1.0e6,
    flux_tot_coeff=DEFAULT_FLUX_TOT_COEFF,
    stomate_tau=STOMATE_TAU_SECONDS,
    epsilon=FORTRAN_EPSILON_R4,
) -> MicroactemControlResult:
    """Compute ``microactem`` time constants in seconds.

    Fortran provenance: ``src_stomate/stomate_permafrost_soilcarbon.f90``,
    function ``microactem``, lines 2468-2703. The function returns residence
    time controls in seconds; ``stomate_main`` converts them to rates later.
    ``is_peat`` is a PFT-length boolean array and must be supplied explicitly
    when ``perma_peat`` is active.
    """

    temp = jnp.asarray(temp_celsius)
    moist = jnp.asarray(moist_in)
    zi = jnp.asarray(zi_soil)
    mc_peat_arr = jnp.asarray(mc_peat)
    if temp.shape != moist.shape:
        raise ValueError("temp_celsius and moist_in must have the same shape (npts, ndeep, nvm)")
    if temp.ndim != 3:
        raise ValueError("temp_celsius must have shape (npts, ndeep, nvm)")
    npts, ndeep, nvm = temp.shape
    if zi.shape != (ndeep,):
        raise ValueError("zi_soil must have shape (ndeep,)")
    if mc_peat_arr.shape != (npts, ndeep):
        raise ValueError("mc_peat must have shape (npts, ndeep)")

    temp_control = microactem_temperature_control(temp, frozen_respiration_func, epsilon=epsilon)
    moisture_control = microactem_moisture_control(moist, limit_decomp_moisture)

    peat_moisture = None
    if perma_peat:
        if is_peat is None:
            raise ValueError("is_peat must be supplied explicitly when perma_peat is true")
        peat_mask = jnp.asarray(is_peat, dtype=bool)
        if peat_mask.shape != (nvm,):
            raise ValueError("is_peat must have shape (nvm,)")
        peat_moisture_2d = moyano_peat_moisture_lookup(mc_peat_arr, epsilon=epsilon)
        peat_moisture = jnp.where(peat_mask[None, None, :], peat_moisture_2d[:, :, None], 1.0)
        if agri_peat:
            agri_mask = (jnp.arange(nvm) == 14) | (jnp.arange(nvm) == 15)
            agri_moisture = moyano_peat_moisture_lookup(moist, epsilon=epsilon)
            moisture_control = jnp.where(agri_mask[None, None, :], agri_moisture, moisture_control)
        tau_depth = jnp.where(
            jnp.arange(ndeep) < 12,
            jnp.asarray(tau_peat) * jnp.exp(zi / jnp.asarray(z_tau)),
            jnp.asarray(tau_peat) * jnp.exp(zi[11] / jnp.asarray(z_tau)),
        )
        peat_tau = tau_depth[:, None]
        fbact_seconds = jnp.where(
            peat_mask[None, None, :],
            peat_tau[None, :, :] / (peat_moisture * temp_control),
            jnp.asarray(stomate_tau) / (moisture_control * temp_control),
        )
    else:
        fbact_seconds = jnp.asarray(stomate_tau) / (moisture_control * temp_control)

    coeff0 = jnp.asarray(flux_tot_coeff)[0]
    agri_divide_mask = (jnp.arange(nvm) == 14) | (jnp.arange(nvm) == 15)
    fbact_seconds = jnp.where(agri_divide_mask[None, None, :], fbact_seconds / coeff0, fbact_seconds)
    return MicroactemControlResult(
        temp_control=temp_control,
        moisture_control=moisture_control,
        peat_moisture_control=peat_moisture,
        fbact_seconds=fbact_seconds,
    )


def stomate_permafrost_decomposition_controls(
    temp_celsius,
    moist_in,
    zi_soil,
    mc_peat,
    poor_soils,
    *,
    frozen_respiration_func: int,
    perma_peat: bool,
    agri_peat: bool,
    is_peat=None,
    tau_peat=3.1536e8,
    z_tau=1.0e6,
    flux_tot_coeff=DEFAULT_FLUX_TOT_COEFF,
    one_day=ONE_DAY_SECONDS,
) -> StomatePermafrostControls:
    """Build ``stomate_main`` permafrost decomposition controls.

    Fortran provenance: ``src_stomate/stomate.f90``, subroutine
    ``stomate_main``, lines 3093-3124. It calls ``microactem`` twice, with
    ``limit_decomp_moisture`` equal to 1 for ``fbact`` and 0 for
    ``fbact_doc`` in this source, converts seconds to day^-1 rates, and
    multiplies each grid-cell slab by ``(2 - poor_soils) * 0.5``.
    """

    fbact = microactem(
        temp_celsius,
        frozen_respiration_func,
        1,
        moist_in,
        zi_soil,
        mc_peat,
        perma_peat=perma_peat,
        agri_peat=agri_peat,
        is_peat=is_peat,
        tau_peat=tau_peat,
        z_tau=z_tau,
        flux_tot_coeff=flux_tot_coeff,
    ).fbact_seconds
    fbact_doc = microactem(
        temp_celsius,
        frozen_respiration_func,
        0,
        moist_in,
        zi_soil,
        mc_peat,
        perma_peat=perma_peat,
        agri_peat=agri_peat,
        is_peat=is_peat,
        tau_peat=tau_peat,
        z_tau=z_tau,
        flux_tot_coeff=flux_tot_coeff,
    ).fbact_seconds
    poor = jnp.asarray(poor_soils)
    factor = (2.0 - poor) * 0.5
    rate = (1.0 / fbact) * jnp.asarray(one_day) * factor[:, None, None]
    rate_doc = (1.0 / fbact_doc) * jnp.asarray(one_day) * factor[:, None, None]
    return StomatePermafrostControls(
        prmfrst_soilc_tempctrl=rate,
        prmfrst_soilc_tempctrl_doc=rate_doc,
        fbact_seconds=fbact,
        fbact_doc_seconds=fbact_doc,
    )
