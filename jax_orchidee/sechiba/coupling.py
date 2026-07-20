"""Source-backed SECHIBA process coupling helpers.

These helpers assemble already-computed process boundary fields. They do not
compute missing SECHIBA state and do not provide fallback values.
"""

from __future__ import annotations

from typing import Mapping, NamedTuple

from jax_orchidee.sechiba.enerbil_bridge import (
    enerbil_active_precall_trace_fields,
    validate_enerbil_to_hydrol_boundary_payload,
    validate_enerbil_active_module_closure,
)
from jax_orchidee.sechiba.bridge import validate_bridge_payload
from jax_orchidee.sechiba.hydrol import HydrolThermosoilMoistureInputs
from jax_orchidee.sechiba.thermosoil import ThermosoilExplicitStepResult, ThermosoilNoExplicitSnowStepResult
from jax_orchidee.sechiba.enerbil import EnerbilExplicitStepResult


SECHIBA_THERMOSOIL_COUPLING_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 1049-1118",
    "fortran_source/ORCHIDEE/src_sechiba/thermosoil.f90::thermosoil_main lines 893-1032",
)
SECHIBA_ENERBIL_COUPLING_PROVENANCE = (
    "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90::sechiba_main lines 997-1019",
    "fortran_source/ORCHIDEE/src_sechiba/diffuco.f90::diffuco_main lines 389-411",
    "fortran_source/ORCHIDEE/src_sechiba/enerbil.f90::enerbil_main lines 390-598",
)


class SechibaThermosoilBoundary(NamedTuple):
    """Explicit THERMOSOIL call kwargs and checked bridge payload."""

    kwargs: dict[str, object]
    advertised_payload: dict[str, object]
    provenance: tuple[str, ...] = SECHIBA_THERMOSOIL_COUPLING_PROVENANCE


class SechibaEnerbilPrecallBoundary(NamedTuple):
    """Checked pre-call field payload for ``enerbil_main``."""

    payload: dict[str, object]
    provenance: tuple[str, ...] = SECHIBA_ENERBIL_COUPLING_PROVENANCE


def assemble_enerbil_precall_payload(*payloads: Mapping[str, object]) -> SechibaEnerbilPrecallBoundary:
    """Merge and validate the active ``sechiba_main`` to ``enerbil_main`` inputs.

    Fortran provenance: ``sechiba_main`` calls ``diffuco_main`` at lines
    997-1005, then passes DIFFUCO outputs, driver forcing, and previous
    THERMOSOIL/CONDVEG surface state into ``enerbil_main`` at lines 1013-1019.
    This helper only validates field availability.
    """

    merged: dict[str, object] = {}
    for payload in payloads:
        overlap = set(merged).intersection(payload)
        for key in overlap:
            if merged[key] is not payload[key]:
                raise ValueError(f"duplicate ENERBIL pre-call field with different object: {key}")
        merged.update(payload)

    validation = validate_enerbil_active_module_closure(before_payload=merged, after_payload={})
    if validation.missing_pre_fields:
        raise ValueError(f"missing ENERBIL pre-call fields: {validation.missing_pre_fields}")

    return SechibaEnerbilPrecallBoundary(payload=merged)


def enerbil_precall_field_names() -> tuple[str, ...]:
    """Return the active ENERBIL pre-call fields required by the coupling helper."""

    return enerbil_active_precall_trace_fields()


def enerbil_downstream_payload(result: EnerbilExplicitStepResult) -> Mapping[str, object]:
    """Expose ENERBIL outputs consumed by HYDROL, THERMOSOIL, and slowproc.

    Fortran provenance: ``sechiba_main`` calls ``enerbil_main`` at lines
    1013-1019 and passes its evaporation, transpiration, surface-temperature,
    and explicit-snow outputs into ``hydrol_main`` at lines 1049-1072 and
    ``thermosoil_main`` at lines 1109-1118.
    """

    payload = {
        "transpir": result.evapveg_pft.transpir,
        "transpot": result.evapveg_pft.transpot,
        "vevapwet": result.evapveg_pft.vevapwet,
        "vevapnu": result.evapveg_grid.vevapnu,
        "vevapnu_pft": result.evapveg_pft.vevapnu_pft,
        "vevapsno": result.evapveg_grid.vevapsno,
        "vevapflo": result.evapveg_grid.vevapflo,
        "evapot": result.flux.evapot,
        "evapot_corr": result.evapot_corr.evapot_corr,
        "temp_sol_new": result.surftemp.temp_sol_new,
        "temp_sol_new_pft": result.surftemp.temp_sol_new_pft,
        "qsurf": result.flux.qsurf,
    }
    if result.t2mdiag is not None:
        payload["t2mdiag"] = result.t2mdiag
    if result.explicit_snow is not None:
        payload["pgflux"] = result.explicit_snow.pgflux
        payload["temp_sol_add"] = result.explicit_snow.temp_sol_add

    validation = validate_enerbil_to_hydrol_boundary_payload(payload)
    if not validation.ok:
        raise ValueError(f"missing ENERBIL downstream fields: {validation.missing_fields}")
    return payload


def assemble_thermosoil_explicit_kwargs(
    *,
    moisture: HydrolThermosoilMoistureInputs,
    temp_sol_new,
    temp_sol_new_pft,
    snowrho,
    snowtemp,
    snowdz,
    ptn,
    cgrnd,
    dgrnd,
    cgrnd_snow,
    dgrnd_snow,
    lambda_snow,
    njsc,
    veget_max,
    pb,
    dlt,
    dz1,
    zlt,
    znt,
    dz5,
    dt_sechiba,
    lambda_thermal,
    frac_snow_veg,
    frac_snow_nobio,
    totfrac_nobio,
    temp_sol_beg,
    soilcap_initial,
    pcapa_en_previous=None,
    soilc_total=None,
    refsoc=None,
    zx1=None,
    ok_laidev=None,
    veget_mask_2d=None,
    satsoil=False,
    shum_ngrnd_permalong_previous=None,
    ok_shum_ngrnd_permalong=True,
) -> SechibaThermosoilBoundary:
    """Assemble the active ``sechiba_main`` to ``thermosoil_main`` boundary.

    Fortran provenance: ``sechiba_main`` maps HYDROL PFT moisture arrays at
    lines 1093-1102 and calls ``thermosoil_main`` at lines 1109-1118. All
    recurrence, forcing, snow, and organic-carbon inputs are explicit.
    """

    if sum(value is not None for value in (soilc_total, refsoc, zx1)) != 1:
        raise ValueError("provide exactly one of soilc_total, refsoc, or zx1")

    kwargs = {
        "ptn": ptn,
        "cgrnd": cgrnd,
        "dgrnd": dgrnd,
        "cgrnd_snow": cgrnd_snow,
        "dgrnd_snow": dgrnd_snow,
        "temp_sol_new": temp_sol_new,
        "temp_sol_new_pft": temp_sol_new_pft,
        "snowrho": snowrho,
        "snowtemp": snowtemp,
        "snowdz": snowdz,
        "shumdiag_perma": moisture.shumdiag_perma,
        "mc_layh": moisture.mc_layh,
        "mcl_layh": moisture.mcl_layh,
        "tmc_layh": moisture.tmc_layh,
        "mc_layh_pft": moisture.mc_layh_pft,
        "mcl_layh_pft": moisture.mcl_layh_pft,
        "tmc_layh_pft": moisture.tmc_layh_pft,
        "njsc": njsc,
        "veget_max": veget_max,
        "pb": pb,
        "dlt": dlt,
        "dz1": dz1,
        "zlt": zlt,
        "znt": znt,
        "dz5": dz5,
        "dt_sechiba": dt_sechiba,
        "lambda_thermal": lambda_thermal,
        "frac_snow_veg": frac_snow_veg,
        "frac_snow_nobio": frac_snow_nobio,
        "totfrac_nobio": totfrac_nobio,
        "temp_sol_beg": temp_sol_beg,
        "soilcap_initial": soilcap_initial,
        "ok_laidev": ok_laidev,
        "veget_mask_2d": veget_mask_2d,
        "satsoil": satsoil,
        "ok_shum_ngrnd_permalong": ok_shum_ngrnd_permalong,
    }
    if shum_ngrnd_permalong_previous is not None:
        kwargs["shum_ngrnd_permalong_previous"] = shum_ngrnd_permalong_previous
    if pcapa_en_previous is not None:
        kwargs["pcapa_en_previous"] = pcapa_en_previous
    if soilc_total is not None:
        kwargs["soilc_total"] = soilc_total
        organic_key = "soilc_total"
    elif refsoc is not None:
        kwargs["refsoc"] = refsoc
        organic_key = "refSOC"
    else:
        kwargs["zx1"] = zx1
        organic_key = "zx1"

    advertised_payload = {
        **kwargs,
        "soilmoist": moisture.tmc_layh,
        "soilmoist_pft": moisture.tmc_layh_pft,
        "snowdz": snowdz,
        "snowrho": snowrho,
        "snowtemp": snowtemp,
        "frac_snow_veg": frac_snow_veg,
        "frac_snow_nobio": frac_snow_nobio,
        "totfrac_nobio": totfrac_nobio,
        "lambda_snow": lambda_snow,
        organic_key: kwargs[organic_key if organic_key != "refSOC" else "refsoc"],
    }
    validation = validate_bridge_payload(
        advertised_payload,
        groups=("hydrol_to_thermosoil", "enerbil_condveg_to_thermosoil"),
    )
    if not validation.ok:
        missing = {
            group: fields
            for group, fields in validation.missing_by_group.items()
            if fields
        }
        raise ValueError(f"missing THERMOSOIL boundary fields: {missing}")

    return SechibaThermosoilBoundary(kwargs=kwargs, advertised_payload=advertised_payload)


def assemble_thermosoil_no_explicit_snow_kwargs(
    *,
    moisture: HydrolThermosoilMoistureInputs,
    temp_sol_new,
    temp_sol_new_pft,
    snow,
    ptn,
    cgrnd,
    dgrnd,
    cgrnd_snow,
    dgrnd_snow,
    lambda_snow,
    njsc,
    veget_max,
    dlt,
    dz1,
    zlt,
    znt,
    dz5,
    dt_sechiba,
    lambda_thermal,
    temp_sol_beg,
    soilcap_initial,
    pcapa_en_previous=None,
    ok_laidev=None,
    veget_mask_2d=None,
    satsoil=False,
    shum_ngrnd_permalong_previous=None,
    ok_shum_ngrnd_permalong=True,
) -> SechibaThermosoilBoundary:
    """Assemble ``thermosoil_main`` inputs for ``OK_EXPLICITSNOW=n``.

    Fortran provenance: ``sechiba_main`` calls ``thermosoil_main`` at lines
    1109-1118; ``thermosoil_coef`` dispatches to
    ``thermosoil_getdiff_old_thermix_with_snow`` when ``ok_explicitsnow`` is
    false at lines 1499-1507.
    """

    kwargs = {
        "ptn": ptn,
        "cgrnd": cgrnd,
        "dgrnd": dgrnd,
        "cgrnd_snow": cgrnd_snow,
        "dgrnd_snow": dgrnd_snow,
        "temp_sol_new": temp_sol_new,
        "temp_sol_new_pft": temp_sol_new_pft,
        "snow": snow,
        "shumdiag_perma": moisture.shumdiag_perma,
        "mc_layh": moisture.mc_layh,
        "mcl_layh": moisture.mcl_layh,
        "tmc_layh": moisture.tmc_layh,
        "mc_layh_pft": moisture.mc_layh_pft,
        "mcl_layh_pft": moisture.mcl_layh_pft,
        "tmc_layh_pft": moisture.tmc_layh_pft,
        "njsc": njsc,
        "veget_max": veget_max,
        "dlt": dlt,
        "dz1": dz1,
        "zlt": zlt,
        "znt": znt,
        "dz5": dz5,
        "dt_sechiba": dt_sechiba,
        "lambda_thermal": lambda_thermal,
        "temp_sol_beg": temp_sol_beg,
        "soilcap_initial": soilcap_initial,
        "ok_laidev": ok_laidev,
        "veget_mask_2d": veget_mask_2d,
        "satsoil": satsoil,
        "ok_shum_ngrnd_permalong": ok_shum_ngrnd_permalong,
    }
    if shum_ngrnd_permalong_previous is not None:
        kwargs["shum_ngrnd_permalong_previous"] = shum_ngrnd_permalong_previous
    if pcapa_en_previous is not None:
        kwargs["pcapa_en_previous"] = pcapa_en_previous
    advertised_payload = {
        **kwargs,
        "soilmoist": moisture.tmc_layh,
        "soilmoist_pft": moisture.tmc_layh_pft,
        "lambda_snow": lambda_snow,
    }
    validation = validate_bridge_payload(advertised_payload, groups=("hydrol_to_thermosoil",))
    if not validation.ok:
        raise ValueError(
            "missing THERMOSOIL no-explicit hydrol fields: "
            f"{validation.missing_by_group['hydrol_to_thermosoil']}"
        )
    missing = [name for name in ("temp_sol_new", "temp_sol_new_pft", "snow") if advertised_payload.get(name) is None]
    if missing:
        raise ValueError(f"missing THERMOSOIL no-explicit boundary fields: {tuple(missing)}")
    return SechibaThermosoilBoundary(kwargs=kwargs, advertised_payload=advertised_payload)


def thermosoil_downstream_payload(result: ThermosoilExplicitStepResult | ThermosoilNoExplicitSnowStepResult) -> Mapping[str, object]:
    """Expose THERMOSOIL outputs and recurrence for the next production step.

    Fortran provenance: ``thermosoil_main`` computes diagnostic/final fields at
    lines 906-1032 and ``thermosoil_coef`` computes next-step coefficients at
    lines 1518-1725. ``ptn``, ``pcapa_en``, and the begin-state diagnostics
    are included because the next profile/energy transition consumes them
    before recomputing coefficients.
    """

    payload = {
        "stempdiag": result.profile.stempdiag,
        "soilcap": result.coef.soilcap,
        "soilcap_pft": result.coef.soil.soilcap_pft,
        "soilflx": result.coef.soilflx,
        "soilflx_pft": result.coef.soil.soilflx_pft,
        "gtemp": result.final.gtemp,
        "ptnlev1": result.final.ptnlev1,
        "ptn_pftmean": result.final.ptn_pftmean,
        "pkappa_pftmean": result.final.pkappa_pftmean,
        "deephum_prof": result.final.deephum_prof,
        "shum_ngrnd_permalong": result.final.deephum_prof,
        "deeptemp_prof": result.final.deeptemp_prof,
        "cgrnd": result.coef.soil.cgrnd,
        "dgrnd": result.coef.soil.dgrnd,
        "cgrnd_snow": result.coef.cgrnd_snow,
        "dgrnd_snow": result.coef.dgrnd_snow,
        "lambda_snow": result.coef.lambda_snow,
        "ptn": result.profile.ptn,
        "pcapa_en": result.getdiff.pcapa_en,
        "ptn_beg": result.energy.ptn_beg,
        "temp_sol_beg": result.energy.temp_sol_beg,
    }
    validation = validate_bridge_payload(payload, groups=("thermosoil_to_slowproc_enerbil",))
    if not validation.ok:
        raise ValueError(
            "missing THERMOSOIL downstream fields: "
            f"{validation.missing_by_group['thermosoil_to_slowproc_enerbil']}"
        )
    return payload
