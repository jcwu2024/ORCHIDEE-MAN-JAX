from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.enerbil import enerbil_explicit_local_step  # noqa: E402
from jax_orchidee.sechiba.enerbil_bridge import enerbil_active_precall_trace_fields  # noqa: E402
from jax_orchidee.sechiba.diffuco import (  # noqa: E402
    diffuco_pft14_local_enerbil_precall_explicit,
)
from jax_orchidee.sechiba.hydrol import build_mineral_cwrr_tables  # noqa: E402
from jax_orchidee.sechiba.sechiba_step import (  # noqa: E402
    sechiba_explicit_coupled_step,
    sechiba_explicit_coupled_step_from_local_diffuco,
)
from jax_orchidee.stomate.entry import (  # noqa: E402
    assemble_stomate_main_payload,
    stomate_driver_entry_source,
    stomate_enerbil_entry_source,
    stomate_hydrol_entry_source,
    stomate_no_routing_entry_source,
)


class _MiniDriverPayload:
    pass


def _enerbil_inputs():
    return {
        "temp_sol": np.asarray([280.0], dtype=np.float64),
        "temp_sol_pft": np.asarray([[279.0, 281.0, 282.0]], dtype=np.float64),
        "lwdown": np.asarray([338.0], dtype=np.float64),
        "swnet": np.asarray([120.0], dtype=np.float64),
        "pb": np.asarray([1010.0], dtype=np.float64),
        "emis": np.asarray([0.96], dtype=np.float64),
        "ok_laidev": np.asarray([False, True, False]),
        "epot_air": np.asarray([282000.0], dtype=np.float64),
        "petAcoef": np.asarray([0.01], dtype=np.float64),
        "petBcoef": np.asarray([281500.0], dtype=np.float64),
        "qair": np.asarray([0.006], dtype=np.float64),
        "peqAcoef": np.asarray([0.1], dtype=np.float64),
        "peqBcoef": np.asarray([0.012], dtype=np.float64),
        "soilflx": np.asarray([8.0], dtype=np.float64),
        "soilflx_pft": np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64),
        "rau": np.asarray([1.2], dtype=np.float64),
        "u": np.asarray([1.0], dtype=np.float64),
        "v": np.asarray([0.5], dtype=np.float64),
        "q_cdrag": np.asarray([0.01], dtype=np.float64),
        "q_cdrag_pft": np.asarray([[0.011, 0.012, 0.013]], dtype=np.float64),
        "vbeta": np.asarray([0.4], dtype=np.float64),
        "vbeta_pft": np.asarray([[0.1, 0.2, 0.3]], dtype=np.float64),
        "valpha": np.asarray([1.0], dtype=np.float64),
        "vbeta1": np.asarray([0.2], dtype=np.float64),
        "vbeta2": np.asarray([[0.10, 0.05, 0.02]], dtype=np.float64),
        "vbeta3": np.asarray([[0.30, 0.10, 0.03]], dtype=np.float64),
        "vbeta3pot": np.asarray([[0.40, 0.20, 0.04]], dtype=np.float64),
        "vbeta4": np.asarray([0.3], dtype=np.float64),
        "vbeta4_pft": np.asarray([[0.30, 0.20, 0.10]], dtype=np.float64),
        "vbeta5": np.asarray([0.05], dtype=np.float64),
        "soilcap": np.asarray([5.0e5], dtype=np.float64),
        "soilcap_pft": np.asarray([[4.0e5, 4.5e5, 5.0e5]], dtype=np.float64),
        "veget_max": np.asarray([[1.0, 0.8, 1.0]], dtype=np.float64),
        "dt_sechiba": 1800.0,
        "precip_rain": np.asarray([4.0], dtype=np.float64),
        "snowdz": np.asarray([[0.01, 0.02, 0.04]], dtype=np.float64),
        "temp_air": np.asarray([285.0], dtype=np.float64),
        "pgflux": np.asarray([0.0], dtype=np.float64),
    }


def _local_diffuco_payload_for_enerbil(enerbil_inputs):
    bundle = _local_diffuco_boundary_bundle(enerbil_inputs)
    return diffuco_pft14_local_enerbil_precall_explicit(
        **bundle["inputs"],
        pft_output_backgrounds=bundle["backgrounds"],
        passthrough=bundle["passthrough"],
    ).payload


def _local_diffuco_boundary_bundle(enerbil_inputs):
    pft_index = 2
    nvm = 3
    humrel = np.asarray([[0.0, 0.6, 0.7]], dtype=np.float64)
    veget = np.asarray([[0.0, 0.2, 0.35]], dtype=np.float64)
    veget_max = enerbil_inputs["veget_max"]
    lai = np.asarray([[0.0, 0.8, 0.7]], dtype=np.float64)
    qsintveg = np.asarray([[0.0, 0.01, 0.02]], dtype=np.float64)
    qsintmax = np.asarray([[0.0, 0.20, 0.25]], dtype=np.float64)
    rstruct = np.asarray([[20.0, 22.0, 25.0]], dtype=np.float64)
    ok_laidev = np.asarray([False, False, False])
    trans_inputs = {
        "swdown": np.asarray([350.0], dtype=np.float64),
        "qsurf": np.asarray([0.008], dtype=np.float64),
        "t2m": enerbil_inputs["temp_air"],
        "pb": enerbil_inputs["pb"],
        "wind": np.sqrt(enerbil_inputs["u"] * enerbil_inputs["u"] + enerbil_inputs["v"] * enerbil_inputs["v"]),
        "temp_growth": np.asarray([25.0], dtype=np.float64),
        "ca": np.asarray([410.0], dtype=np.float64),
        "vcmax": np.asarray([42.0], dtype=np.float64),
        "tphoto_min": 273.15,
        "tphoto_max": 330.0,
        "ext_coeff": 0.5,
        "a1": 0.85,
        "b1": 0.02,
        "stress_gs": 0.1,
        "e_kmc": 79430.0,
        "e_kmo": 36380.0,
        "e_sco": -24460.0,
        "e_gamma_star": 37830.0,
        "e_rd": 46390.0,
        "e_jmax": 43540.0,
        "d_jmax": 200000.0,
        "asj": 659.70,
        "bsj": -0.75,
        "e_vcmax": 58520.0,
        "d_vcmax": 200000.0,
        "asv": 668.39,
        "bsv": -1.07,
        "e_gm": 49600.0,
        "d_gm": 437400.0,
        "s_gm": 1400.0,
        "arjv": 2.59,
        "brjv": -0.035,
        "gm25": 0.12,
        "stress_gm": 0.2,
        "g0": 0.01,
        "kmc25": 405.0,
        "kmo25": 278400.0,
        "sco25": 2590.0,
        "gamma_star25": 42.75,
        "alpha_ll": 0.3,
        "theta": 0.7,
        "stress_vcmax": 0.2,
        "rveg_pft": 1.0,
        "rstruct_const": 25.0,
        "control_salinity": np.asarray([1.0], dtype=np.float64),
        "control_inudate": np.asarray([1.0], dtype=np.float64),
        "nlai": 3,
        "laimax": 1.5,
        "lai_level_depth": 0.2,
    }
    backgrounds = {
        name: np.zeros((1, nvm), dtype=np.float64)
        for name in ("gpp", "gsmean", "rveget", "rstruct", "cimean")
    }
    return {
        "inputs": {
            "pft_index": pft_index,
            "trans_co2_inputs": trans_inputs,
            "ldq_cdrag_from_gcm": False,
            "u": enerbil_inputs["u"],
            "v": enerbil_inputs["v"],
            "zlev": np.asarray([30.0], dtype=np.float64),
            "z0h": np.asarray([0.06], dtype=np.float64),
            "z0m": np.asarray([0.15], dtype=np.float64),
            "roughheight": np.asarray([3.0], dtype=np.float64),
            "roughheight_pft": np.asarray([[0.0, 3.5, 5.0]], dtype=np.float64),
            "temp_sol": enerbil_inputs["temp_sol"],
            "temp_sol_pft": enerbil_inputs["temp_sol_pft"],
            "temp_air": enerbil_inputs["temp_air"],
            "qsurf": trans_inputs["qsurf"],
            "qair": enerbil_inputs["qair"],
            "pb": enerbil_inputs["pb"],
            "rau": enerbil_inputs["rau"],
            "humrel": humrel,
            "veget": veget,
            "veget_max": veget_max,
            "lai": lai,
            "qsintveg": qsintveg,
            "qsintmax": qsintmax,
            "rstruct": rstruct,
            "ok_laidev": ok_laidev,
            "snow": np.asarray([0.05], dtype=np.float64),
            "frac_nobio": np.asarray([[0.0]], dtype=np.float64),
            "totfrac_nobio": np.asarray([0.0], dtype=np.float64),
            "snow_nobio": np.asarray([[0.0]], dtype=np.float64),
            "frac_snow_veg": np.asarray([0.1], dtype=np.float64),
            "frac_snow_nobio": np.asarray([[0.0]], dtype=np.float64),
            "evapot": np.asarray([0.2], dtype=np.float64),
            "evapot_corr": np.asarray([0.2], dtype=np.float64),
            "flood_frac": np.asarray([0.0], dtype=np.float64),
            "flood_res": np.asarray([0.0], dtype=np.float64),
            "tot_bare_soil": np.asarray([0.20], dtype=np.float64),
            "evap_bare_lim": np.asarray([0.25], dtype=np.float64),
            "vbeta3_background": np.asarray([[0.0, 0.05, 0.0]], dtype=np.float64),
            "dt_sechiba": enerbil_inputs["dt_sechiba"],
        },
        "backgrounds": backgrounds,
        "passthrough": {
            "lai": lai,
            "veget_max": veget_max,
        },
    }


def _hydrol_inputs(enerbil_result):
    npts = 1
    nvm = 3
    nstm = 6
    nslm = 7
    veget_max = np.asarray([[0.2, 0.4, 0.4]], dtype=np.float64)
    veget = np.asarray([[0.1, 0.3, 0.3]], dtype=np.float64)
    soiltile = np.asarray([[0.20, 0.20, 0.15, 0.20, 0.10, 0.15]], dtype=np.float64)
    vegtot = np.sum(veget_max, axis=1)
    pref_soil_veg = np.asarray([1, 2, 4], dtype=np.int32)
    dz_mm = np.asarray([0.0, 3.0, 6.0, 8.0, 10.0, 12.0, 14.0], dtype=np.float64)
    zz_mm = np.asarray([1.0, 4.0, 10.0, 18.0, 28.0, 40.0, 54.0], dtype=np.float64)
    mc = np.zeros((npts, nslm, nstm), dtype=np.float64)
    for jst in range(nstm):
        mc[:, :, jst] = np.linspace(0.30, 0.42, nslm, dtype=np.float64)[None, :] + 0.01 * jst
    humrelv = np.zeros((npts, nvm, nstm), dtype=np.float64)
    us = np.zeros((npts, nvm, nstm, nslm), dtype=np.float64)
    for jv, jst_fortran in enumerate(pref_soil_veg):
        jst = int(jst_fortran) - 1
        humrelv[:, jv, jst] = 0.5
        us[:, jv, jst, :] = np.asarray([0.0, 0.10, 0.12, 0.10, 0.08, 0.06, 0.04], dtype=np.float64)
    tables = build_mineral_cwrr_tables(njsc=2, mcs=np.asarray([0.50]), z_m=zz_mm / 1000.0)
    return {
        "inputs": {
            "precip_rain": np.asarray([4.0], dtype=np.float64),
            "vevapwet": enerbil_result.evapveg_pft.vevapwet,
            "veget": veget,
            "veget_max": veget_max,
            "qsintmax": np.asarray([[0.2, 0.5, 0.6]], dtype=np.float64),
            "qsintveg": np.asarray([[0.0, 0.1, 0.2]], dtype=np.float64),
            "tot_melt": np.asarray([0.0], dtype=np.float64),
            "vegtot": vegtot,
            "throughfall_by_pft": np.asarray([0.15, 0.25, 0.35], dtype=np.float64),
            "pref_soil_veg": pref_soil_veg,
            "soiltile": soiltile,
            "vevapflo": enerbil_result.evapveg_grid.vevapflo,
            "flood_frac": np.asarray([0.0], dtype=np.float64),
            "flood_res": np.asarray([1.0], dtype=np.float64),
            "subsinksoil": np.asarray([0.0], dtype=np.float64),
            "vevapnu": enerbil_result.evapveg_grid.vevapnu,
            "vevapnu_pft": enerbil_result.evapveg_pft.vevapnu_pft,
            "transpir": enerbil_result.evapveg_pft.transpir,
            "humrel": np.asarray([[0.0, 0.5, 0.5]], dtype=np.float64),
            "humrelv": humrelv,
            "us": us,
            "ae_ns": np.zeros((npts, nstm), dtype=np.float64),
            "evap_bare_lim": np.asarray([0.2], dtype=np.float64),
            "evap_bare_lim_ns": np.full((npts, nstm), 0.02, dtype=np.float64),
            "water2infilt": np.full((npts, nstm), 0.02, dtype=np.float64),
            "reinfiltration_soil": np.zeros((npts, nstm), dtype=np.float64),
            "mc": mc,
            "mcl": mc.copy(),
            "profil_froz": np.zeros_like(mc),
            "mcr": np.asarray([0.057], dtype=np.float64),
            "mcs": np.asarray([0.50], dtype=np.float64),
            "dz_mm": dz_mm,
            "dt_days": 1.0 / 48.0,
            "free_drain_coef": np.ones((npts, nstm), dtype=np.float64),
            "resolv": np.ones((npts, nstm), dtype=bool),
            "kfact_root": np.ones((npts, nslm, nstm), dtype=np.float64),
            "ks": np.asarray([10.0], dtype=np.float64),
            "kfact": np.ones(nslm, dtype=np.float64),
            "tmc": np.zeros((npts, nstm), dtype=np.float64),
            "njsc": np.asarray([2], dtype=np.int32),
            "reinf_slope": np.asarray([0.25], dtype=np.float64),
            "zz_mm": zz_mm,
            "zmaxh_m": 2.0,
            "zwt_force": np.full((npts, nstm), 1.0e20, dtype=np.float64),
            "min_sechiba": 1.0e-8,
        },
        "diagnostics": {
            "nroot": np.tile(np.asarray([0.0, 0.10, 0.15, 0.20, 0.20, 0.20, 0.15], dtype=np.float64), (npts, nvm, 1)),
            "dz_mm": dz_mm,
            "dh_mm": np.asarray([1.0, 3.0, 6.0, 8.0, 10.0, 12.0, 14.0], dtype=np.float64),
            "njsc": np.asarray([2], dtype=np.int32),
            "soiltile": soiltile,
            "vegtot": vegtot,
            "mcs": np.asarray([0.50], dtype=np.float64),
            "mineral_tables": tables,
            "vevapnu": enerbil_result.evapveg_grid.vevapnu,
            "tot_melt": np.asarray([0.0], dtype=np.float64),
        },
        "tables": tables,
    }


def _downstream_sechiba_inputs(all_enerbil_inputs, diffuco_payload):
    diffuco_fields = set(diffuco_payload.payload)
    enerbil_inputs = {name: value for name, value in all_enerbil_inputs.items() if name not in diffuco_fields}
    for name in ("zlev", "swdown", "snowdz_1", "qsurf", "evapot", "evapot_corr"):
        enerbil_inputs[name] = object()
    expected_enerbil_source = {**diffuco_payload.payload, **enerbil_inputs}
    expected_enerbil = enerbil_explicit_local_step(
        **{field: expected_enerbil_source[field] for field in (
            "temp_sol",
            "temp_sol_pft",
            "lwdown",
            "swnet",
            "pb",
            "emis",
            "ok_laidev",
            "epot_air",
            "petAcoef",
            "petBcoef",
            "qair",
            "peqAcoef",
            "peqBcoef",
            "soilflx",
            "soilflx_pft",
            "rau",
            "u",
            "v",
            "q_cdrag",
            "q_cdrag_pft",
            "vbeta",
            "vbeta_pft",
            "valpha",
            "vbeta1",
            "vbeta2",
            "vbeta3",
            "vbeta3pot",
            "vbeta4",
            "vbeta4_pft",
            "vbeta5",
            "soilcap",
            "soilcap_pft",
            "veget_max",
            "dt_sechiba",
            "precip_rain",
            "snowdz",
            "temp_air",
            "pgflux",
        )}
    )
    hydrol_bundle = _hydrol_inputs(expected_enerbil)
    hydrol_inputs = hydrol_bundle["inputs"]
    hydrol_inputs["mineral_tables"] = hydrol_bundle["tables"]
    condveg_inputs = {
        "snow": np.asarray([0.1], dtype=np.float64),
        "snow_age": np.asarray([2.0], dtype=np.float64),
        "snow_nobio": np.asarray([[0.0]], dtype=np.float64),
        "snow_nobio_age": np.asarray([0.0], dtype=np.float64),
        "snowrho": np.asarray([[100.0, 200.0, 300.0]], dtype=np.float64),
        "snowdz": enerbil_inputs["snowdz"],
        "veget": hydrol_inputs["veget"],
        "veget_max": hydrol_inputs["veget_max"],
        "frac_nobio": np.zeros((1, 1), dtype=np.float64),
        "totfrac_nobio": np.zeros(1, dtype=np.float64),
        "zlev": np.asarray([20.0], dtype=np.float64),
        "height": np.asarray([[0.1, 5.0, 7.0]], dtype=np.float64),
        "temp_air": all_enerbil_inputs["temp_air"],
        "pb": all_enerbil_inputs["pb"],
        "u": all_enerbil_inputs["u"],
        "v": all_enerbil_inputs["v"],
        "lai": np.asarray([[0.0, 2.0, 2.5]], dtype=np.float64),
        "emis_scal": 0.96,
        "drysoil_frac": np.asarray([0.4], dtype=np.float64),
        "tot_bare_soil": np.asarray([0.20], dtype=np.float64),
        "alb_bg_modis": True,
        "soilalb_bg": np.asarray([[0.12, 0.25]], dtype=np.float64),
        "alb_leaf_vis": np.asarray([0.00, 0.04, 0.06], dtype=np.float64),
        "alb_leaf_nir": np.asarray([0.00, 0.20, 0.24], dtype=np.float64),
        "snowa_aged_vis": np.asarray([0.35, 0.14, 0.18], dtype=np.float64),
        "snowa_aged_nir": np.asarray([0.35, 0.14, 0.18], dtype=np.float64),
        "snowa_dec_vis": np.asarray([0.45, 0.10, 0.60], dtype=np.float64),
        "snowa_dec_nir": np.asarray([0.45, 0.06, 0.52], dtype=np.float64),
    }
    thermosoil_inputs = {
        "ptn": np.asarray([[[280.0 + jg + jv for jv in range(3)] for jg in range(7)]], dtype=np.float64),
        "cgrnd": np.asarray([[[275.0 + jg + jv for jv in range(3)] for jg in range(6)]], dtype=np.float64),
        "dgrnd": np.asarray([[[0.2 + 0.01 * jg for _ in range(3)] for jg in range(6)]], dtype=np.float64),
        "cgrnd_snow": np.asarray([[260.0, 261.0, 262.0]], dtype=np.float64),
        "dgrnd_snow": np.asarray([[0.1, 0.2, 0.3]], dtype=np.float64),
        "lambda_snow": np.asarray([0.5], dtype=np.float64),
        "snowtemp": np.asarray([[270.0, 268.0, 266.0]], dtype=np.float64),
        "njsc": np.asarray([1], dtype=np.int32),
        "veget_max": hydrol_inputs["veget_max"],
        "pb": all_enerbil_inputs["pb"],
        "dlt": np.asarray([0.1, 0.2, 0.4, 0.8, 1.2, 1.8, 2.4], dtype=np.float64),
        "dz1": np.asarray([10.0, 3.0, 2.0, 1.5, 1.2, 1.0], dtype=np.float64),
        "zlt": np.asarray([0.1, 0.3, 0.7, 1.2, 1.8, 2.5, 3.3], dtype=np.float64),
        "znt": np.asarray([0.05, 0.2, 0.5, 0.95, 1.5, 2.15, 2.9], dtype=np.float64),
        "dz5": np.asarray([0.2, 0.4, 0.45, 0.5, 0.55, 0.6], dtype=np.float64),
        "dt_sechiba": 1800.0,
        "lambda_thermal": 0.5,
        "totfrac_nobio": np.asarray([0.0], dtype=np.float64),
        "temp_sol_beg": np.asarray([278.0], dtype=np.float64),
        "soilcap_initial": np.asarray([1000.0], dtype=np.float64),
        "refsoc": np.asarray([[0.0, 10.0, 20.0, 25.0, 30.0, 35.0, 40.0]], dtype=np.float64),
        "ok_laidev": np.asarray([False, True, False]),
    }
    slowproc_inputs = {
        "lai": condveg_inputs["lai"],
        "frac_nobio": condveg_inputs["frac_nobio"],
        "veget_max": hydrol_inputs["veget_max"],
        "pref_soil_veg": hydrol_inputs["pref_soil_veg"],
        "ext_coeff_vegetfrac": np.asarray([0.0, 0.5, 0.3], dtype=np.float64),
        "nstm": 6,
    }
    return {
        "enerbil_inputs": enerbil_inputs,
        "hydrol_inputs": hydrol_inputs,
        "hydrol_diagnostic_inputs": hydrol_bundle["diagnostics"],
        "condveg_inputs": condveg_inputs,
        "thermosoil_inputs": thermosoil_inputs,
        "slowproc_inputs": slowproc_inputs,
        "expected_enerbil": expected_enerbil,
    }


def _stomate_driver_source(all_enerbil_inputs, downstream, result):
    driver = _MiniDriverPayload()
    driver.kjpindex = 1
    driver.nbindex = 1
    driver.kindex = np.asarray([1], dtype=np.int32)
    driver.lalo = np.asarray([[21.0, 109.0]], dtype=np.float64)
    driver.contfrac = np.asarray([1.0], dtype=np.float64)
    driver.resolution = np.asarray([[111106.0, 111111.0]], dtype=np.float64)
    driver.neighbours = np.asarray([[-1, -1, -1, -1, -1, -1, -1, -1]], dtype=np.int32)
    driver.u = all_enerbil_inputs["u"]
    driver.v = all_enerbil_inputs["v"]
    driver.precip_rain = all_enerbil_inputs["precip_rain"]
    driver.precip_snow = np.asarray([0.0], dtype=np.float64)
    driver.swdown = np.asarray([350.0], dtype=np.float64)
    driver.pb = all_enerbil_inputs["pb"]
    driver.clay_frac = np.asarray([0.1], dtype=np.float64)
    driver.bulk_dens = np.asarray([1.3], dtype=np.float64)
    driver.soil_ph = np.asarray([6.0], dtype=np.float64)
    driver.poor_soils = np.asarray([0.0], dtype=np.float64)
    return stomate_driver_entry_source(
        driver,
        kjit=1,
    )


def test_sechiba_explicit_coupled_step_runs_current_local_chain_in_source_order():
    all_enerbil_inputs = _enerbil_inputs()
    diffuco_payload = _local_diffuco_payload_for_enerbil(all_enerbil_inputs)
    downstream = _downstream_sechiba_inputs(all_enerbil_inputs, diffuco_payload)

    result = sechiba_explicit_coupled_step(
        diffuco_payload=diffuco_payload,
        enerbil_inputs=downstream["enerbil_inputs"],
        hydrol_inputs=downstream["hydrol_inputs"],
        hydrol_diagnostic_inputs=downstream["hydrol_diagnostic_inputs"],
        condveg_inputs=downstream["condveg_inputs"],
        thermosoil_inputs=downstream["thermosoil_inputs"],
        slowproc_inputs=downstream["slowproc_inputs"],
    )

    assert result.provenance[0].endswith("sechiba.f90::sechiba_main lines 997-1216")
    assert "vbeta5" in result.diffuco_local_fields
    assert "q_cdrag" in result.diffuco_passthrough_fields
    np.testing.assert_allclose(
        np.asarray(result.enerbil_payload["temp_sol_new"]),
        np.asarray(downstream["expected_enerbil"].surftemp.temp_sol_new),
    )
    np.testing.assert_allclose(
        np.asarray(result.hydrol_thermosoil_moisture.mc_layh),
        np.asarray(result.hydrol_diagnostics.mc_layh),
    )
    assert result.condveg.albedo is not None
    assert result.condveg.albedo_snow is not None
    assert result.condveg.alb_bare is not None
    assert result.condveg.alb_veget is not None
    assert result.condveg.albedo_snow_mean is not None
    np.testing.assert_allclose(np.asarray(result.condveg.emis), np.asarray([0.96]))
    np.testing.assert_allclose(
        np.asarray(result.thermosoil.profile.stempdiag),
        np.asarray(result.thermosoil_payload["stempdiag"]),
    )
    assert result.slowproc is not None
    np.testing.assert_allclose(
        np.asarray(result.slowproc.tot_bare_soil),
        np.asarray(
            result.slowproc.vegetation.veget_max[:, 0]
            + np.sum(
                np.asarray(result.slowproc.vegetation.veget_max)[:, 1:]
                - np.asarray(result.slowproc.vegetation.veget)[:, 1:],
                axis=1,
            )
        ),
    )

    stomate_payload = assemble_stomate_main_payload(
        _stomate_driver_source(all_enerbil_inputs, downstream, result),
        stomate_enerbil_entry_source(
            t2mdiag=result.enerbil_payload["t2mdiag"],
            evapot_corr=result.enerbil_payload["evapot_corr"],
            temp_sol=all_enerbil_inputs["temp_sol"],
        ),
        stomate_hydrol_entry_source(
            diagnostics=result.hydrol_diagnostics,
            outputs=result.hydrol_outputs,
        ),
        stomate_no_routing_entry_source(
            kjpindex=1,
            nflow=2,
            river_routing=False,
            nbp_glo=1,
        ),
        {
            "gpp": result.diffuco_payload["gpp"],
            "humrel": downstream["hydrol_inputs"]["humrel"],
            "veget": result.slowproc.vegetation.veget,
            "veget_max": result.slowproc.vegetation.veget_max,
            "totfrac_nobio": result.slowproc.vegetation.totfrac_nobio,
            "stempdiag": result.thermosoil_payload["stempdiag"],
            "snow": downstream["condveg_inputs"]["snow"],
            "snowdz": downstream["condveg_inputs"]["snowdz"],
            "snowrho": downstream["condveg_inputs"]["snowrho"],
        }
    )
    assert not stomate_payload.ready
    np.testing.assert_allclose(
        np.asarray(stomate_payload.payload["temp_sol"]),
        np.asarray(all_enerbil_inputs["temp_sol"]),
    )
    assert not np.array_equal(
        np.asarray(stomate_payload.payload["temp_sol"]),
        np.asarray(result.enerbil_payload["temp_sol_new"]),
    )
    for name in (
        "gpp",
        "humrel",
        "veget",
        "veget_max",
        "totfrac_nobio",
        "temp_sol",
        "t2mdiag",
        "evapot_corr",
        "stempdiag",
        "snow",
        "snowdz",
        "snowrho",
        "shumdiag",
        "litterhumdiag",
        "tmc_topgrass",
        "shumdiag_peat",
        "mc_peat_above",
        "shumdiag_croppeat",
        "mc_croppeat_above",
        "shumdiag_man",
        "mc_man_above",
        "wtp",
        "liqwt_ratio",
        "soil_mc",
        "wat_flux",
        "drainage_per_soil",
        "runoff_per_soil",
        "runoff2peat",
        "precip2canopy",
        "precip2ground",
        "canopy2ground",
        "DOC_to_topsoil",
        "DOC_to_subsoil",
        "flood_frac",
        "fastr",
    ):
        assert name in stomate_payload.covered_inputs
    for name in (
        "kjit",
        "kjpindex",
        "index",
        "lalo",
        "resolution",
        "neighbours",
        "contfrac",
        "clay",
        "t2m",
        "t2m_min",
        "t2m_max",
        "precip_rain",
        "precip_snow",
        "swdown",
        "pb",
        "wspeed",
        "bulk_dens",
        "soil_ph",
        "poor_soils",
    ):
        assert name in stomate_payload.covered_inputs
    assert "fwet_new" in stomate_payload.missing_by_source["hydrol"]
    assert "biomass" in stomate_payload.missing_by_source["stomate_state"]


def test_sechiba_explicit_coupled_step_from_local_diffuco_matches_payload_entrypoint():
    all_enerbil_inputs = _enerbil_inputs()
    diffuco_bundle = _local_diffuco_boundary_bundle(all_enerbil_inputs)
    manual_diffuco = diffuco_pft14_local_enerbil_precall_explicit(
        **diffuco_bundle["inputs"],
        pft_output_backgrounds=diffuco_bundle["backgrounds"],
        passthrough=diffuco_bundle["passthrough"],
    )
    downstream = _downstream_sechiba_inputs(all_enerbil_inputs, manual_diffuco.payload)

    manual_step = sechiba_explicit_coupled_step(
        diffuco_payload=manual_diffuco.payload,
        enerbil_inputs=downstream["enerbil_inputs"],
        hydrol_inputs=downstream["hydrol_inputs"],
        hydrol_diagnostic_inputs=downstream["hydrol_diagnostic_inputs"],
        condveg_inputs=downstream["condveg_inputs"],
        thermosoil_inputs=downstream["thermosoil_inputs"],
        slowproc_inputs=downstream["slowproc_inputs"],
    )
    local_step = sechiba_explicit_coupled_step_from_local_diffuco(
        diffuco_inputs=diffuco_bundle["inputs"],
        pft_output_backgrounds=diffuco_bundle["backgrounds"],
        diffuco_passthrough=diffuco_bundle["passthrough"],
        enerbil_inputs=downstream["enerbil_inputs"],
        hydrol_inputs=downstream["hydrol_inputs"],
        hydrol_diagnostic_inputs=downstream["hydrol_diagnostic_inputs"],
        condveg_inputs=downstream["condveg_inputs"],
        thermosoil_inputs=downstream["thermosoil_inputs"],
        slowproc_inputs=downstream["slowproc_inputs"],
    )

    assert local_step.provenance[1].endswith("diffuco.f90::diffuco_main lines 629-710")
    np.testing.assert_allclose(
        np.asarray(local_step.diffuco.payload.payload["q_cdrag"]),
        np.asarray(manual_diffuco.payload.payload["q_cdrag"]),
    )
    np.testing.assert_allclose(
        np.asarray(local_step.diffuco.boundary.process_chain.closure.comb.vbeta),
        np.asarray(manual_diffuco.boundary.process_chain.closure.comb.vbeta),
    )
    np.testing.assert_allclose(
        np.asarray(local_step.step.enerbil_payload["temp_sol_new"]),
        np.asarray(manual_step.enerbil_payload["temp_sol_new"]),
    )
    np.testing.assert_allclose(
        np.asarray(local_step.step.hydrol_thermosoil_moisture.mc_layh),
        np.asarray(manual_step.hydrol_thermosoil_moisture.mc_layh),
    )
    np.testing.assert_allclose(
        np.asarray(local_step.step.condveg.albedo),
        np.asarray(manual_step.condveg.albedo),
    )
    np.testing.assert_allclose(
        np.asarray(local_step.step.condveg.albedo_snow),
        np.asarray(manual_step.condveg.albedo_snow),
    )
    np.testing.assert_allclose(
        np.asarray(local_step.step.condveg.albedo_snow_mean),
        np.asarray(manual_step.condveg.albedo_snow_mean),
    )
    np.testing.assert_allclose(
        np.asarray(local_step.step.thermosoil_payload["stempdiag"]),
        np.asarray(manual_step.thermosoil_payload["stempdiag"]),
    )
    assert local_step.step.slowproc is not None
    assert manual_step.slowproc is not None
    np.testing.assert_allclose(
        np.asarray(local_step.step.slowproc.tot_bare_soil),
        np.asarray(manual_step.slowproc.tot_bare_soil),
    )
