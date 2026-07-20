"""Source-ordered production export from SECHIBA final state to restart state."""

from __future__ import annotations

from typing import Mapping

from jax_orchidee.sechiba.condveg import condveg_finalize_restart_packet
from jax_orchidee.sechiba.diffuco import diffuco_finalize_restart_packet
from jax_orchidee.sechiba.enerbil import enerbil_finalize_restart_packet
from jax_orchidee.sechiba.hydrol_thermosoil_completion import (
    explicitsnow_finalize_restart_packet,
    hydrol_finalize,
)
from jax_orchidee.sechiba.restart_io import (
    SechibaRestartState,
    sechiba_restart_state_from_finalize_packets,
)
from jax_orchidee.sechiba.slowproc import slowproc_finalize_restart_packet
from jax_orchidee.sechiba.thermosoil import thermosoil_finalize_restart_packet


def paper_sechiba_restart_state_from_finalize_state(
    state: Mapping[str, object],
    *,
    kjit: int,
) -> SechibaRestartState:
    """Run all paper-path component finalize owners in Fortran order.

    Missing scientific state is an error. The fixed switches correspond to
    the paper run: CWRR hydrology, explicit snow, CO2, MODIS background
    albedo, MAP_PFT_FORMAT, OK_LEAK-driven wet diagnostics, and freezing energy
    correction are active; prescribed LAI is inactive.

    Fortran provenance: ``sechiba.f90::sechiba_finalize`` lines 1800-1923.
    """

    values = dict(state)
    diffuco = diffuco_finalize_restart_packet(
        rstruct=values["rstruct"],
        q_cdrag_pft=values["q_cdrag_pft"],
        ok_co2=True,
        leaf_ci=values["leaf_ci"],
    )
    enerbil = enerbil_finalize_restart_packet(
        **{
            name: values[name]
            for name in (
                "evapot", "evapot_corr", "temp_sol", "temp_sol_pft",
                "tsol_rad", "qsurf", "fluxsens", "fluxlat", "vevapp",
                "temp_sol_pot", "q_sol_pot",
            )
        }
    )
    hydrol_result = hydrol_finalize(
        kjit=int(kjit),
        rest_id=0,
        state=values,
        inputs={
            name: values[name]
            for name in ("snowrho", "snowtemp", "snowdz", "snowheat", "snowgrain")
        },
        restart_writer=lambda request: request,
        use_refsoc_hydrol=False,
        check_waterbal=False,
        ok_explicitsnow=True,
        explicitsnow_finalizer=lambda payload: explicitsnow_finalize_restart_packet(
            **{name: payload[name] for name in (
                "snowrho", "snowtemp", "snowdz", "snowheat", "snowgrain"
            )}
        ),
    )
    hydrol = {request.name: request.value for request in hydrol_result.requests}
    condveg = condveg_finalize_restart_packet(
        z0m=values["z0m"],
        z0h=values["z0h"],
        roughheight=values["roughheight"],
        roughheight_pft=values["roughheight_pft"],
        alb_bg_modis=True,
        soilalb_bg=values["soilalb_bg"],
    )
    thermosoil = thermosoil_finalize_restart_packet(
        ptn=values["ptn"],
        refsoc=values["refsoc"],
        shum_ngrnd_perma=values["shum_ngrnd_perma"],
        cgrnd=values["cgrnd"],
        dgrnd=values["dgrnd"],
        gtemp=values["gtemp"],
        soilcap=values["soilcap"],
        soilcap_pft=values["soilcap_pft"],
        soilflx=values["soilflx"],
        soilflx_pft=values["soilflx_pft"],
        cgrnd_snow=values["cgrnd_snow"],
        dgrnd_snow=values["dgrnd_snow"],
        lambda_snow=values["lambda_snow"],
        ok_shum_ngrnd_permalong=True,
        shum_ngrnd_permalong=values["shum_ngrnd_permalong"],
        ok_Ecorr=True,
        e_soil_lat=values["e_soil_lat"],
    )
    slowproc, stomate_finalize_active = slowproc_finalize_restart_packet(
        state=values,
        hydrol_cwrr=True,
        read_lai=False,
        map_pft_format=True,
        ok_stomate=True,
    )
    if not stomate_finalize_active:
        raise ValueError("paper SECHIBA restart requires active STOMATE finalize")
    if hydrol_result.explicit_snow_result is None:
        raise ValueError("paper SECHIBA restart requires explicit-snow finalize")
    return sechiba_restart_state_from_finalize_packets(
        diffuco=diffuco,
        enerbil=enerbil,
        hydrol=hydrol,
        explicit_snow=hydrol_result.explicit_snow_result,
        condveg=condveg,
        thermosoil=thermosoil,
        slowproc=slowproc,
    )
