"""Source-routed completion owners for the remaining SECHIBA science procedures.

The functions in this module compose process kernels owned by ``diffuco``,
``enerbil``, ``condveg``, and ``slowproc``.  Restart functions own scientific
field selection; NetCDF/IOIPSL transport remains outside this module.
"""

from __future__ import annotations

from typing import Mapping, NamedTuple

import jax.numpy as jnp
import numpy as np

from jax_orchidee.driver.interpolation_core4cont import interpweight_2dcont_routed
from jax_orchidee.driver.interpolation_core12 import (
    Aggregate,
    InterpolationSource,
    InterpolationTarget,
)


DIFFUCO_COMPLETION_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/diffuco.f90::diffuco_main lines 629-736",
    "fortran_source/ORCHIDEE/src_sechiba/diffuco.f90::diffuco_finalize lines 773-800",
    "fortran_source/ORCHIDEE/src_sechiba/diffuco.f90::diffuco_trans lines 1820-1910",
)
ENERBIL_COMPLETION_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/enerbil.f90::enerbil_flux lines 1435-1648",
)
CONDVEG_COMPLETION_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/condveg.f90::condveg_finalize lines 500-526",
    "fortran_source/ORCHIDEE/src_sechiba/condveg.f90::condveg_background_soilalb lines 1210-1284",
)
SLOWPROC_COMPLETION_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_initialize lines 252-304",
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_main lines 1076-1136",
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_finalize lines 1215-1284",
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::get_soilcorr_usda lines 5563-5621",
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_checkveget lines 5824-5947",
    "fortran_source/ORCHIDEE/src_sechiba/slowproc.f90::slowproc_change_frac lines 5967-6026",
)


class DiffucoTransResult(NamedTuple):
    vbeta3: jnp.ndarray
    vbeta3pot: jnp.ndarray
    rveget: jnp.ndarray
    cimean: jnp.ndarray
    vbetaco2: jnp.ndarray


class DiffucoMainRoutingResult(NamedTuple):
    drag: object
    qsatt: jnp.ndarray
    transpiration: object
    xios: tuple[tuple[str, jnp.ndarray], ...]
    history_primary: tuple[tuple[str, jnp.ndarray], ...]
    history_secondary: tuple[tuple[str, jnp.ndarray], ...]


class EnerbilFluxRoutingResult(NamedTuple):
    flux: object
    snow: object
    correction: object
    tair: jnp.ndarray


class CondvegBackgroundSoilAlbedoResult(NamedTuple):
    soilalb_bg: jnp.ndarray
    aalb_bg: jnp.ndarray


class SlowprocVegetationCheck(NamedTuple):
    epsilon: float
    warnings: tuple[str, ...]


class SlowprocHistoryRoutingResult(NamedTuple):
    npp: jnp.ndarray
    primary: tuple[tuple[str, jnp.ndarray], ...]
    secondary: tuple[tuple[str, jnp.ndarray], ...]


def _float64(value):
    return jnp.asarray(value, dtype=jnp.float64)


def diffuco_trans_source_routed(
    *,
    swnet,
    temp_air,
    pb,
    qair,
    rau,
    u,
    v,
    q_cdrag,
    humrel,
    veget,
    veget_max,
    lai,
    qsintveg,
    qsintmax,
    rstruct,
    vbeta23,
    kzero,
    rveg_pft,
    min_wind: float = 0.1,
    min_sechiba: float = 1.0e-8,
    rayt_cste: float = 100.0,
    defc_plus: float = 23.0,
    defc_mult: float = 0.5,
    undef_sechiba: float = 1.0e20,
) -> DiffucoTransResult:
    """Execute legacy ``diffuco_trans`` without CO2 photosynthesis.

    Provenance: ``diffuco.f90::diffuco_trans`` lines 1820-1910.  PFT1 remains
    at source initialization values; PFTs 2..nvm use the exact canopy-water,
    activity, stomatal-resistance, beta, and potential-beta formulas.
    """

    from .enerbil import qsat_moisture_qsatcalc

    swnet, temp_air, pb, qair, rau, u, v, q_cdrag = map(
        _float64, (swnet, temp_air, pb, qair, rau, u, v, q_cdrag)
    )
    humrel, veget, veget_max, lai, qsintveg, qsintmax, rstruct, vbeta23 = map(
        _float64, (humrel, veget, veget_max, lai, qsintveg, qsintmax, rstruct, vbeta23)
    )
    if humrel.ndim != 2:
        raise ValueError("humrel must have shape (npts,nvm)")
    shape = humrel.shape
    for name, value in (
        ("veget", veget),
        ("veget_max", veget_max),
        ("lai", lai),
        ("qsintveg", qsintveg),
        ("qsintmax", qsintmax),
        ("rstruct", rstruct),
        ("vbeta23", vbeta23),
    ):
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}")
    for name, value in (
        ("swnet", swnet),
        ("temp_air", temp_air),
        ("pb", pb),
        ("qair", qair),
        ("rau", rau),
        ("u", u),
        ("v", v),
        ("q_cdrag", q_cdrag),
    ):
        if value.shape != (shape[0],):
            raise ValueError(f"{name} must have shape (npts,)")
    kzero, rveg_pft = _float64(kzero), _float64(rveg_pft)
    if kzero.shape != (shape[1],) or rveg_pft.shape != (shape[1],):
        raise ValueError("kzero and rveg_pft must have shape (nvm,)")

    qsatt = qsat_moisture_qsatcalc(temp_air, pb)
    zdefconc = rau * jnp.maximum(qsatt - qair, 0.0)
    speed = jnp.maximum(min_wind, jnp.sqrt(u * u + v * v))[:, None]
    storage = qsintmax > min_sechiba
    zqsvegrap = jnp.where(
        storage, jnp.maximum(0.0, qsintveg / jnp.where(storage, qsintmax, 1.0)), 0.0
    )
    pft_not_bare = jnp.arange(shape[1])[None, :] > 0
    active = (
        pft_not_bare
        & (veget * lai > min_sechiba)
        & (kzero[None, :] > min_sechiba)
        & (swnet[:, None] > min_sechiba)
    )
    safe_swnet = jnp.where(active, swnet[:, None], 1.0)
    safe_lai = jnp.where(active, lai, 1.0)
    safe_kzero = jnp.where(active, kzero[None, :], 1.0)
    rveget_calc = (
        ((safe_swnet + rayt_cste) / safe_swnet)
        * ((defc_plus + defc_mult * zdefconc[:, None]) / safe_kzero)
        / safe_lai
    )
    rveget_min = (defc_plus / safe_kzero) / safe_lai
    denominator = 1.0 + speed * q_cdrag[:, None] * rveg_pft[None, :] * (
        rveget_calc + rstruct
    )
    dry_beta = veget * (1.0 - zqsvegrap) * humrel / denominator
    wet_beta = jnp.minimum(vbeta23, veget * zqsvegrap * humrel / denominator)
    potential_denominator = 1.0 + speed * q_cdrag[:, None] * rveg_pft[None, :] * (
        rveget_min + rstruct
    )
    vbeta3pot_calc = jnp.maximum(0.0, veget_max / potential_denominator)
    rveget = jnp.where(active, rveget_calc, undef_sechiba)
    vbeta3 = jnp.where(active, dry_beta + wet_beta, 0.0)
    vbeta3pot = jnp.where(active, vbeta3pot_calc, 0.0)
    zeros = jnp.zeros(shape, dtype=jnp.float64)
    return DiffucoTransResult(vbeta3, vbeta3pot, rveget, zeros, zeros)


def diffuco_main_source_routed(
    *,
    ldq_cdrag_from_gcm: bool,
    drag_inputs: Mapping[str, object],
    pb,
    temp_sol,
    ok_co2: bool,
    trans_inputs: Mapping[str, object],
    almaoutput: bool = False,
    hist2_id: int = -1,
    ok_bvoc: bool = False,
) -> DiffucoMainRoutingResult:
    """Route the six scientific/output arms in ``diffuco_main`` lines 629-736."""

    from .diffuco import (
        diffuco_drag_boundary_explicit,
        diffuco_trans_co2_c3_pft_explicit,
    )
    from .enerbil import qsat_moisture_qsatcalc

    if ok_bvoc:
        raise NotImplementedError(
            "diffuco_main chemistry_bvoc lines 683-688 requires the chemistry owner"
        )
    drag = diffuco_drag_boundary_explicit(
        ldq_cdrag_from_gcm=ldq_cdrag_from_gcm, **dict(drag_inputs)
    )
    qsatt = qsat_moisture_qsatcalc(_float64(temp_sol), _float64(pb))
    routed = dict(trans_inputs)
    routed["q_cdrag"] = drag.q_cdrag
    if ok_co2:
        if "pft_index" not in routed:
            raise ValueError("pft_index is required for the PFT-local CO2 owner")
        pft_index = int(routed.pop("pft_index"))
        if pft_index < 0 or pft_index >= drag.q_cdrag_pft.shape[1]:
            raise ValueError("pft_index must select a q_cdrag_pft column")
        routed["q_cdrag_pft"] = drag.q_cdrag_pft[:, pft_index]
        routed["wind"] = jnp.sqrt(
            _float64(drag_inputs["u"]) ** 2 + _float64(drag_inputs["v"]) ** 2
        )
        routed["qsatt"] = qsat_moisture_qsatcalc(
            _float64(routed["t2m"]), _float64(routed["pb"])
        )
        transpiration = diffuco_trans_co2_c3_pft_explicit(**routed)
    else:
        transpiration = diffuco_trans_source_routed(**routed)
    wind = jnp.sqrt(_float64(drag_inputs["u"]) ** 2 + _float64(drag_inputs["v"]) ** 2)
    xios = (
        ("q_cdrag", drag.q_cdrag),
        ("cdrag_pft", drag.q_cdrag_pft),
        ("raero", drag.raero),
        ("wind", wind),
        ("qsatt", qsatt),
    )
    primary = (
        ()
        if almaoutput
        else (
            ("raero", drag.raero),
            ("cdrag", drag.q_cdrag),
            ("cdrag_pft", drag.q_cdrag_pft),
            ("Wind", wind),
            ("qsatt", qsatt),
        )
    )
    secondary = (
        tuple(item for item in primary if item[0] != "cdrag_pft")
        if (not almaoutput and hist2_id > 0)
        else ()
    )
    return DiffucoMainRoutingResult(
        drag, qsatt, transpiration, xios, primary, secondary
    )


def diffuco_finalize_restart_packet(
    *, rstruct, q_cdrag_pft, ok_co2: bool, leaf_ci=None
):
    """Select exact restart fields from ``diffuco_finalize`` lines 788-798."""

    packet = {"rstruct": _float64(rstruct), "cdrag_pft": _float64(q_cdrag_pft)}
    if (
        packet["rstruct"].shape != packet["cdrag_pft"].shape
        or packet["rstruct"].ndim != 2
    ):
        raise ValueError("rstruct and q_cdrag_pft must share shape (npts,nvm)")
    if ok_co2:
        if leaf_ci is None:
            raise ValueError("leaf_ci is required when ok_co2 is true")
        leaf_ci = _float64(leaf_ci)
        if leaf_ci.ndim != 3 or leaf_ci.shape[:2] != packet["rstruct"].shape:
            raise ValueError("leaf_ci must have shape (npts,nvm,nlai)")
        packet["leaf_ci"] = leaf_ci
    return packet


def enerbil_flux_source_routed(
    *, local_inputs, snow_inputs, correction_inputs
) -> EnerbilFluxRoutingResult:
    """Compose all ``enerbil_flux`` calculations in source order."""

    from .enerbil import (
        enerbil_flux_evapot_corr,
        enerbil_flux_explicit_snow_diagnostics,
        enerbil_flux_local_diagnostics,
    )

    flux = enerbil_flux_local_diagnostics(**dict(local_inputs))
    snow_kwargs = dict(snow_inputs)
    snow_kwargs["flux"] = flux
    snow = enerbil_flux_explicit_snow_diagnostics(**snow_kwargs)
    correction_kwargs = dict(correction_inputs)
    correction_kwargs["flux"] = flux
    correction = enerbil_flux_evapot_corr(**correction_kwargs)
    tair = _float64(local_inputs["epot_air"]) / float(
        correction_inputs.get("cp_air", 1004.675)
    )
    return EnerbilFluxRoutingResult(flux, snow, correction, tair)


def enerbil_finalize_restart_packet(
    *,
    evapot,
    evapot_corr,
    temp_sol,
    temp_sol_pft,
    tsol_rad,
    qsurf,
    fluxsens,
    fluxlat,
    vevapp,
    temp_sol_pot,
    q_sol_pot,
):
    """Select every ``enerbil_finalize`` restart value.

    Fortran provenance: ``enerbil.f90::enerbil_finalize`` lines 616-652.
    """

    grid = {
        "evapot": evapot,
        "evapot_corr": evapot_corr,
        "temp_sol": temp_sol,
        "tsolrad": tsol_rad,
        "qsurf": qsurf,
        "fluxsens": fluxsens,
        "fluxlat": fluxlat,
        "evapora": vevapp,
        "tempsolpot": temp_sol_pot,
        "qsolpot": q_sol_pot,
    }
    packet = {name: _float64(value) for name, value in grid.items()}
    npts = packet["temp_sol"].shape[0]
    if any(value.shape != (npts,) for value in packet.values()):
        raise ValueError("ENERBIL finalize grid fields must share shape (npts,)")
    temp_sol_pft = _float64(temp_sol_pft)
    if temp_sol_pft.ndim != 2 or temp_sol_pft.shape[0] != npts:
        raise ValueError("temp_sol_pft must have shape (npts,nvm)")
    packet["temp_sol_pft"] = temp_sol_pft
    return packet


def condveg_background_soilalb_source_routed(
    *,
    bg_alb_vis: InterpolationSource,
    bg_alb_nir: InterpolationSource,
    target: InterpolationTarget,
    aggregate: Aggregate,
    undef_sechiba: float = 1.0e20,
) -> CondvegBackgroundSoilAlbedoResult:
    """Route both background bands through ``interpweight_2Dcont``.

    Provenance: ``condveg_background_soilalb`` lines 1210-1284 and
    ``interpweight.f90::interpweight_2Dcont`` lines 1824-2233. Both source
    reads retain the source bug that passes ``albbg_default(inir)=0.247``.
    ``aalb_bg`` is the availability from the second (NIR) call, matching the
    Fortran output variable overwrite at lines 1270-1273.
    """

    if not isinstance(bg_alb_vis, InterpolationSource) or not isinstance(
        bg_alb_nir, InterpolationSource
    ):
        raise TypeError("bg_alb_vis and bg_alb_nir must be InterpolationSource objects")
    if not isinstance(target, InterpolationTarget):
        raise TypeError("target must be an InterpolationTarget")
    if bg_alb_vis.variable_name != "bg_alb_vis":
        raise ValueError("bg_alb_vis source variable_name must be 'bg_alb_vis'")
    if bg_alb_nir.variable_name != "bg_alb_nir":
        raise ValueError("bg_alb_nir source variable_name must be 'bg_alb_nir'")

    common = dict(
        target=target,
        aggregate=aggregate,
        noneg=False,
        masktype="var",
        maskvalues=(undef_sechiba, undef_sechiba, undef_sechiba),
        typefrac="default",
        defaultvalue=0.247,
        default_no_value=undef_sechiba,
    )
    visible = interpweight_2dcont_routed(bg_alb_vis, **common)
    near_infrared = interpweight_2dcont_routed(bg_alb_nir, **common)
    soilalb_bg = jnp.asarray(
        np.stack((visible.output, near_infrared.output), axis=1), dtype=jnp.float64
    )
    return CondvegBackgroundSoilAlbedoResult(
        soilalb_bg=soilalb_bg,
        aalb_bg=jnp.asarray(near_infrared.availability, dtype=jnp.float64),
    )


def condveg_finalize_restart_packet(
    *,
    z0m,
    z0h,
    roughheight,
    roughheight_pft,
    alb_bg_modis,
    soilalb_bg=None,
    soilalb_dry=None,
    soilalb_wet=None,
    soilalb_moy=None,
):
    """Select exact restart fields from ``condveg_finalize`` lines 514-525."""

    packet = {
        "z0m": _float64(z0m),
        "z0h": _float64(z0h),
        "roughheight": _float64(roughheight),
        "roughheight_pft": _float64(roughheight_pft),
    }
    if alb_bg_modis:
        if soilalb_bg is None:
            raise ValueError("soilalb_bg is required when alb_bg_modis is true")
        packet["soilalbedo_bg"] = _float64(soilalb_bg)
    else:
        for output, value in (
            ("soilalbedo_dry", soilalb_dry),
            ("soilalbedo_wet", soilalb_wet),
            ("soilalbedo_moy", soilalb_moy),
        ):
            if value is None:
                raise ValueError(f"{output} is required when alb_bg_modis is false")
            packet[output] = _float64(value)
    return packet


def slowproc_initialize_source_routed(
    *,
    init_kwargs,
    dt_stomate,
    dt_sechiba,
    ok_stomate,
    qsintcst,
    vcmax_fix=None,
    height_presc=None,
):
    """Compose ``slowproc_initialize`` initialization and derived-state arms."""

    from .slowproc import slowproc_derivvar_explicit, slowproc_init_pft14_explicit

    if float(dt_stomate) < float(dt_sechiba):
        raise ValueError("slowproc_initialize requires dt_stomate >= dt_sechiba")
    initialized = slowproc_init_pft14_explicit(**dict(init_kwargs))
    if ok_stomate:
        qsintmax = _float64(qsintcst) * initialized.veget * initialized.lai
        qsintmax = qsintmax.at[:, 0].set(0.0)
        derived = None
    else:
        if vcmax_fix is None or height_presc is None:
            raise ValueError(
                "vcmax_fix and height_presc are required when ok_stomate is false"
            )
        derived = slowproc_derivvar_explicit(
            veget=initialized.veget,
            lai=initialized.lai,
            vcmax_fix=vcmax_fix,
            height_presc=height_presc,
            qsintcst=qsintcst,
        )
        qsintmax = derived.qsintmax
    return initialized, qsintmax, derived


def get_soilcorr_usda_source_routed(nusda: int = 12):
    """Return the exact silt/sand/clay table from lines 5588-5619."""

    if int(nusda) != 12:
        raise ValueError("get_soilcorr_usda requires exactly 12 USDA classes")
    sand_clay = np.asarray(
        (
            (0.93, 0.03),
            (0.81, 0.06),
            (0.63, 0.11),
            (0.17, 0.19),
            (0.06, 0.10),
            (0.40, 0.20),
            (0.54, 0.27),
            (0.08, 0.33),
            (0.30, 0.33),
            (0.48, 0.41),
            (0.06, 0.46),
            (0.15, 0.55),
        ),
        dtype=np.float64,
    )
    return jnp.asarray(np.column_stack((1.0 - sand_clay.sum(axis=1), sand_clay)))


def slowproc_checkveget_source_routed(
    *,
    frac_nobio,
    veget_max,
    veget,
    tot_bare_soil,
    soiltile,
    ok_dgvm=False,
    min_vegfrac=1.0e-6,
) -> SlowprocVegetationCheck:
    """Apply every fraction invariant in ``slowproc_checkveget``."""

    frac_nobio, veget_max, veget, tot_bare_soil, soiltile = map(
        _float64, (frac_nobio, veget_max, veget, tot_bare_soil, soiltile)
    )
    if (
        veget_max.ndim != 2
        or veget.shape != veget_max.shape
        or frac_nobio.ndim != 2
        or frac_nobio.shape[0] != veget_max.shape[0]
    ):
        raise ValueError("fraction arrays have incompatible shapes")
    npts = veget_max.shape[0]
    if (
        tot_bare_soil.shape != (npts,)
        or soiltile.ndim != 2
        or soiltile.shape[0] != npts
    ):
        raise ValueError("tot_bare_soil and soiltile have incompatible shapes")
    epsilon = float(np.finfo(np.float64).eps * 1000.0)
    failures = []
    if bool(jnp.any((frac_nobio > epsilon) & (frac_nobio < min_vegfrac))):
        failures.append("frac_nobio below min_vegfrac")
    if not ok_dgvm and bool(jnp.any((veget_max > epsilon) & (veget_max < min_vegfrac))):
        failures.append("veget_max below min_vegfrac")
    if bool(
        jnp.any(
            jnp.abs(jnp.sum(frac_nobio, axis=1) + jnp.sum(veget_max, axis=1) - 1.0)
            > epsilon
        )
    ):
        failures.append("veget_max + frac_nobio does not equal 1")
    if bool(jnp.any(jnp.abs(veget[:, 0] - veget_max[:, 0]) > epsilon)):
        failures.append("bare-soil veget differs from veget_max")
    if bool(jnp.any(veget[:, 1:] > veget_max[:, 1:])):
        failures.append("veget exceeds veget_max")
    expected_bare = veget[:, 0] + jnp.sum(veget_max - veget, axis=1)
    if bool(jnp.any(jnp.abs(expected_bare - tot_bare_soil) > epsilon)):
        failures.append("tot_bare_soil is inconsistent")
    warnings = (
        ("soiltile does not sum to one",)
        if bool(jnp.any(jnp.abs(jnp.sum(soiltile, axis=1) - 1.0) > epsilon))
        else ()
    )
    if failures:
        raise ValueError("slowproc_checkveget: " + "; ".join(failures))
    return SlowprocVegetationCheck(epsilon, warnings)


def slowproc_change_frac_source_routed(
    *,
    agri_peat,
    veget_max_new,
    frac_nobio_new,
    lai,
    pref_soil_veg,
    ext_coeff_vegetfrac,
    nstm,
    veget_max_adjusted=None,
    ok_dgvm=False,
):
    """Execute selection, vegetation recomputation, and checks at lines 6003-6024."""

    from .slowproc import slowproc_surface_update_explicit

    if agri_peat:
        if veget_max_adjusted is None:
            raise ValueError("veget_max_adjusted is required when agri_peat is true")
        selected = veget_max_adjusted
    else:
        selected = veget_max_new
    result = slowproc_surface_update_explicit(
        lai=lai,
        frac_nobio=frac_nobio_new,
        veget_max=selected,
        pref_soil_veg=pref_soil_veg,
        ext_coeff_vegetfrac=ext_coeff_vegetfrac,
        nstm=nstm,
        ok_dgvm=ok_dgvm,
    )
    slowproc_checkveget_source_routed(
        frac_nobio=result.vegetation.frac_nobio,
        veget_max=result.vegetation.veget_max,
        veget=result.vegetation.veget,
        tot_bare_soil=result.tot_bare_soil,
        soiltile=result.vegetation.soiltile,
        ok_dgvm=ok_dgvm,
    )
    return result


def slowproc_main_source_routed(**kwargs):
    """Run the production PFT14 main owner and enforce line 1127 checks."""

    from .slowproc import slowproc_main_pft14_step

    result = slowproc_main_pft14_step(**kwargs)
    if result.calendar.do_slow:
        vegetation = result.vegetation_state
        slowproc_checkveget_source_routed(
            frac_nobio=vegetation["frac_nobio"],
            veget_max=vegetation["veget_max"],
            veget=vegetation["veget"],
            tot_bare_soil=vegetation["tot_bare_soil"],
            soiltile=vegetation["soiltile"],
        )
    return result


def slowproc_main_history_routing(
    *, gpp, resp_maint, resp_growth, resp_hetero, hist2_id: int
) -> SlowprocHistoryRoutingResult:
    """Compute NPP and route history fields from lines 1047-1068.

    Fortran provenance: ``slowproc.f90::slowproc_main`` lines 1047-1068.
    PFT 1 is explicitly zero; PFTs 2:nvm use
    ``gpp-resp_growth-resp_maint``. The secondary stream receives the same
    four arrays only when ``hist2_id > 0``.
    """

    arrays = tuple(
        _float64(value) for value in (gpp, resp_maint, resp_growth, resp_hetero)
    )
    shape = arrays[0].shape
    if len(shape) != 2 or any(value.shape != shape for value in arrays[1:]):
        raise ValueError("GPP and respiration fields must share shape (npts,nvm)")
    if shape[1] < 2:
        raise ValueError("slowproc_main requires bare soil and at least one PFT")
    npp = jnp.zeros_like(arrays[0])
    npp = npp.at[:, 1:].set(
        arrays[0][:, 1:] - arrays[2][:, 1:] - arrays[1][:, 1:]
    )
    fields = (
        ("maint_resp", arrays[1]),
        ("hetero_resp", arrays[3]),
        ("growth_resp", arrays[2]),
        ("npp", npp),
    )
    return SlowprocHistoryRoutingResult(
        npp, fields, fields if int(hist2_id) > 0 else ()
    )


def slowproc_finalize_restart_packet(
    *,
    state: Mapping[str, object],
    hydrol_cwrr: bool,
    read_lai: bool,
    map_pft_format: bool,
    ok_stomate: bool,
):
    """Select SLOWPROC restart fields and expose the STOMATE finalize gate."""

    names = [
        "veget",
        "veget_max",
        "lai",
        "frac_nobio",
        "frac_age",
        "njsc",
        "clayfraction",
        "sandfraction",
        "height",
        "peatPET_lastyear",
        "growth_day",
        "GSL",
        "peatPET_thisyear",
        "precipitation_lastsummer",
        "precipitation_thissummer",
        "summerpet_long",
        "summerp_long",
        "peatC",
        "peatC_ok",
        "soil_ph",
        "poor_soils",
        "bulk_density",
    ]
    if hydrol_cwrr:
        names.append("reinf_slope")
    if read_lai:
        names.append("laimap")
    if map_pft_format:
        names.append("veget_year")
    missing = tuple(name for name in names if name not in state)
    if missing:
        raise ValueError(f"slowproc finalize state is missing {missing}")
    packet = {name: jnp.asarray(state[name]) for name in names}
    return packet, bool(ok_stomate)
