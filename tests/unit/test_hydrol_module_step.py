from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.hydrol import (  # noqa: E402
    PEAT_MCR,
    PEAT_MCS,
    POROS_ORG,
    SOILC_MAX,
    USDA_AVAN,
    USDA_MCF,
    USDA_MCR,
    USDA_MCS,
    USDA_MCW,
    USDA_NVAN,
    USDA_VG_M,
    USDA_VG_PSI_FC,
    USDA_VG_PSI_WP,
    HYDROL_FIRST_STEP_PRECALL_REQUIRED_FIELDS,
    assemble_hydrol_first_step_precall_payload,
    build_mineral_cwrr_tables,
    build_peat_cwrr_tables,
    hydrol_after_under_mcr_runoff_peat_tide_routing,
    hydrol_canop_interception,
    hydrol_flood_reservoir,
    hydrol_first_step_profil_froz_from_restart_thermosoil,
    hydrol_static_precall_template_from_slowproc,
    hydrol_initial_tmc_from_restart_mc,
    hydrol_layer_moisture_content,
    hydrol_module_diagnostics,
    hydrol_module_explicit_step,
    hydrol_module_outputs,
    hydrol_nroot_from_humcste,
    hydrol_refsoc_1d_thresholds,
    hydrol_runoff_peat_post_loop_reinjection,
    run_hydrol_first_step_module_from_precall,
    hydrol_soil_froz_profile,
    hydrol_soil_all_tiles_explicit_step,
    hydrol_split_soil_fluxes,
    hydrol_to_thermosoil_moisture_inputs,
    hydrol_vegupd_static_state,
)
from jax_orchidee.driver.init import parse_run_def, parse_run_def_bool, parse_run_def_float, parse_run_def_indexed_vector  # noqa: E402
from jax_orchidee.driver.orchestration import (  # noqa: E402
    _add_hydrol_to_previous_fields,
    _empty_previous_step_state_fields,
    paper_1961_driver_timestep_scaffold,
)
from jax_orchidee.driver.stomate_boundary import paper_case_cwrr_vertical_soil_grid_from_used_run_def  # noqa: E402
from jax_orchidee.driver.trace import read_static_fields_from_fixed_format_trace_dir  # noqa: E402


CONFIG = ROOT / "configs" / "orchidee_man_250919.yaml"
USED_RUN_DEF = ROOT / "outputs" / "server_1961_trace_full_20260623" / "run" / "used_run.def"
FULL_TRACE_DIR = ROOT / "outputs" / "server_1961_trace_full_20260623" / "traces"


def _module_inputs():
    npts = 1
    nvm = 4
    nstm = 6
    nslm = 7
    veget_max = np.asarray([[0.1, 0.3, 0.2, 0.4]], dtype=np.float64)
    veget = np.asarray([[0.1, 0.24, 0.12, 0.30]], dtype=np.float64)
    soiltile = np.asarray([[0.20, 0.20, 0.10, 0.25, 0.05, 0.20]], dtype=np.float64)
    vegtot = np.sum(veget_max, axis=1)
    pref_soil_veg = np.asarray([1, 1, 2, 4], dtype=np.int32)
    dz_mm = np.asarray([0.0, 3.0, 6.0, 8.0, 10.0, 12.0, 14.0], dtype=np.float64)
    zz_mm = np.asarray([1.0, 4.0, 10.0, 18.0, 28.0, 40.0, 54.0], dtype=np.float64)
    mc = np.zeros((npts, nslm, nstm), dtype=np.float64)
    for jst in range(nstm):
        mc[:, :, jst] = np.linspace(0.30, 0.42, nslm, dtype=np.float64)[None, :] + 0.01 * jst
    humrelv = np.zeros((npts, nvm, nstm), dtype=np.float64)
    us = np.zeros((npts, nvm, nstm, nslm), dtype=np.float64)
    for jv, jst_fortran in enumerate(pref_soil_veg):
        jst = jst_fortran - 1
        humrelv[:, jv, jst] = 0.5
        us[:, jv, jst, :] = np.asarray([0.0, 0.10, 0.12, 0.10, 0.08, 0.06, 0.04])
    return {
        "precip_rain": np.asarray([4.0], dtype=np.float64),
        "vevapwet": np.asarray([[0.0, 0.02, 0.01, 0.03]], dtype=np.float64),
        "veget": veget,
        "veget_max": veget_max,
        "qsintmax": np.asarray([[0.0, 0.5, 0.4, 0.6]], dtype=np.float64),
        "qsintveg": np.asarray([[0.0, 0.1, 0.1, 0.2]], dtype=np.float64),
        "tot_melt": np.asarray([0.0], dtype=np.float64),
        "vegtot": vegtot,
        "throughfall_by_pft": np.asarray([0.0, 0.25, 0.35, 0.45], dtype=np.float64),
        "pref_soil_veg": pref_soil_veg,
        "soiltile": soiltile,
        "vevapflo": np.asarray([0.1], dtype=np.float64),
        "flood_frac": np.asarray([0.0], dtype=np.float64),
        "flood_res": np.asarray([1.0], dtype=np.float64),
        "subsinksoil": np.asarray([0.0], dtype=np.float64),
        "vevapnu": np.asarray([0.05], dtype=np.float64),
        "vevapnu_pft": np.asarray([[0.0, 0.02, 0.01, 0.02]], dtype=np.float64),
        "transpir": np.asarray([[0.0, 0.03, 0.02, 0.04]], dtype=np.float64),
        "humrel": np.asarray([[0.0, 0.5, 0.5, 0.5]], dtype=np.float64),
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
    }


def _runoff_tide_inputs():
    return {
        "soil_tile_index": 4,
        "ru_ns": np.asarray([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]], dtype=np.float64),
        "runoff2peat": np.full((1, 6), 99.0, dtype=np.float64),
        "soiltile": np.asarray([[0.20, 0.30, 0.10, 0.25, 0.05, 0.10]], dtype=np.float64),
        "water2infilt": np.zeros((1, 6), dtype=np.float64),
        "run2peat": np.asarray([9.0], dtype=np.float64),
        "run2man": np.asarray([8.0], dtype=np.float64),
        "wt_ab": np.asarray([0.0], dtype=np.float64),
        "wt_ab_tide": np.asarray([0.7], dtype=np.float64),
        "peat_hydro": True,
        "ok_ru2peat": True,
        "ok_wt_ab": False,
        "tides": False,
        "wtp_tide": 0.0,
        "dwtp_tide": 0.0,
    }


def test_hydrol_refsoc_1d_thresholds_follow_hydrol_var_init_formula():
    njsc = np.asarray([2, 5], dtype=np.int32)
    refsoc = np.asarray([0.0, SOILC_MAX * 2.0], dtype=np.float64)

    result = hydrol_refsoc_1d_thresholds(njsc=njsc, refSOC_1d=refsoc)

    tex = njsc - 1
    zx1 = np.minimum(refsoc / SOILC_MAX, 1.0)
    expected_mcs = zx1 * POROS_ORG + (1.0 - zx1) * np.asarray(USDA_MCS)[tex]
    mcr = np.asarray(USDA_MCR)[tex]
    alpha = np.asarray(USDA_AVAN)[tex]
    vg_n = np.asarray(USDA_NVAN)[tex]
    vg_m = np.asarray(USDA_VG_M)[tex]
    expected_mcw = mcr + (expected_mcs - mcr) / (1.0 + (alpha * np.asarray(USDA_VG_PSI_WP)[tex]) ** vg_n) ** vg_m
    expected_mcf = mcr + (expected_mcs - mcr) / (1.0 + (alpha * np.asarray(USDA_VG_PSI_FC)[tex]) ** vg_n) ** vg_m

    np.testing.assert_allclose(np.asarray(result.zx1), zx1)
    np.testing.assert_allclose(np.asarray(result.mcs), expected_mcs)
    np.testing.assert_allclose(np.asarray(result.mcw), expected_mcw)
    np.testing.assert_allclose(np.asarray(result.mcf), expected_mcf)
    assert np.asarray(result.mcs)[0] == USDA_MCS[1]


def test_hydrol_refsoc_1d_thresholds_disabled_returns_mineral_values():
    njsc = np.asarray([2, 5], dtype=np.int32)

    result = hydrol_refsoc_1d_thresholds(njsc=njsc, refSOC_1d=None, use_refSOC_hydrol=False)

    tex = njsc - 1
    np.testing.assert_allclose(np.asarray(result.mcs), np.asarray(USDA_MCS)[tex])
    np.testing.assert_allclose(np.asarray(result.mcw), np.asarray(USDA_MCW)[tex])
    np.testing.assert_allclose(np.asarray(result.mcf), np.asarray(USDA_MCF)[tex])


def test_hydrol_static_template_carries_refsoc_hydrol_thresholds():
    slowproc_restart = SimpleNamespace(
        lai=np.zeros((1, 2), dtype=np.float64),
        frac_nobio=np.zeros((1, 1), dtype=np.float64),
        veget=np.asarray([[0.0, 1.0]], dtype=np.float64),
        veget_max=np.asarray([[0.0, 1.0]], dtype=np.float64),
        soiltile=np.asarray([[0.0, 1.0]], dtype=np.float64),
        totfrac_nobio=np.asarray([0.0], dtype=np.float64),
    )

    template = hydrol_static_precall_template_from_slowproc(
        slowproc_restart=slowproc_restart,
        pref_soil_veg=np.asarray([1, 2], dtype=np.int32),
        nstm=2,
        ext_coeff_vegetfrac=np.ones(2, dtype=np.float64),
        hydrol_njsc=np.asarray([5], dtype=np.int32),
        refSOC_1d=np.asarray([SOILC_MAX], dtype=np.float64),
        use_refSOC_hydrol=True,
    )

    assert "refSOC_1d" in template
    np.testing.assert_allclose(np.asarray(template["refSOC_1d"]), [SOILC_MAX])
    np.testing.assert_allclose(np.asarray(template["mcs"]), [POROS_ORG])
    assert np.asarray(template["mcw"]).shape == (1,)
    assert np.asarray(template["mcf"]).shape == (1,)
    assert "refSOC_1d" in template["_source_kernel_inputs"]


def _module_diagnostic_inputs(result, inputs):
    npts, nslm, _ = result.soil.mc.shape
    nvm = inputs["veget_max"].shape[1]
    nroot = np.zeros((npts, nvm, nslm), dtype=np.float64)
    nroot[:, :, 1:] = 1.0 / (nslm - 1)
    return {
        "nroot": nroot,
        "dz_mm": inputs["dz_mm"],
        "dh_mm": np.asarray([1.0, 3.0, 6.0, 8.0, 10.0, 12.0, 14.0], dtype=np.float64),
        "njsc": inputs["njsc"],
        "soiltile": inputs["soiltile"],
        "vegtot": inputs["vegtot"],
        "mcs": inputs["mcs"],
        "vevapnu": inputs["vevapnu"],
        "tot_melt": inputs["tot_melt"],
    }


def _explicit_snow_module_precall(*, omit_restart_field: str | None = None):
    inputs = _module_inputs()
    npts = inputs["veget_max"].shape[0]
    nvm = inputs["veget_max"].shape[1]
    nstm = inputs["soiltile"].shape[1]
    nslm = inputs["mc"].shape[1]
    nsnow = 3
    nnobio = 1
    snowdz = np.zeros((npts, nsnow), dtype=np.float64)
    snowrho = np.full((npts, nsnow), 50.0, dtype=np.float64)
    snowtemp = np.full((npts, nsnow), 273.15, dtype=np.float64)
    snowheat = np.zeros((npts, nsnow), dtype=np.float64)
    snowgrain = np.zeros((npts, nsnow), dtype=np.float64)
    snowliq = np.zeros((npts, nsnow), dtype=np.float64)

    enerbil_payload = {
        "temp_sol_new": np.asarray([268.0], dtype=np.float64),
        "transpir": inputs["transpir"],
        "transpot": np.asarray([[0.0, 0.04, 0.03, 0.05]], dtype=np.float64),
        "vevapwet": inputs["vevapwet"],
        "vevapnu": inputs["vevapnu"],
        "vevapnu_pft": inputs["vevapnu_pft"],
        "vevapsno": np.zeros(npts, dtype=np.float64),
        "vevapflo": inputs["vevapflo"],
        "evapot": np.asarray([1.0], dtype=np.float64),
        "evapot_corr": np.asarray([1.0], dtype=np.float64),
        "pgflux": np.asarray([-5.0], dtype=np.float64),
        "temp_sol_add": np.zeros(npts, dtype=np.float64),
        "soilcap": np.full(npts, 2.0e6, dtype=np.float64),
    }
    diffuco_payload = {
        "qsintmax": inputs["qsintmax"],
        "qsintveg": inputs["qsintveg"],
        "evap_bare_lim": inputs["evap_bare_lim"],
        "flood_frac": inputs["flood_frac"],
        "flood_res": inputs["flood_res"],
        "snow": np.zeros(npts, dtype=np.float64),
        "snowdz": snowdz,
        "snowrho": snowrho,
        "snowtemp": snowtemp,
        "snow_age": np.zeros(npts, dtype=np.float64),
        "snow_nobio": np.zeros((npts, nnobio), dtype=np.float64),
        "snow_nobio_age": np.zeros((npts, nnobio), dtype=np.float64),
        "snowheat": snowheat,
        "snowgrain": snowgrain,
        "snowliq": snowliq,
    }
    hydrol_restart_values = {
        "moistc": np.transpose(inputs["mc"], (0, 2, 1)),
        "moistcl": np.transpose(inputs["mcl"], (0, 2, 1)),
        "ae_ns": inputs["ae_ns"],
        "water2infilt": inputs["water2infilt"],
        "free_drain_coef": inputs["free_drain_coef"],
        "zwt_force": inputs["zwt_force"],
        "evap_bare_lim_ns": inputs["evap_bare_lim_ns"],
        "us": inputs["us"],
        "humrel": inputs["humrel"],
        "njsc": inputs["njsc"],
        "profil_froz_hydro_ns": inputs["profil_froz"],
        "temp_hydro": np.full((npts, nslm), 268.0, dtype=np.float64),
        "stempdiag": np.full((npts, nslm), 268.0, dtype=np.float64),
        "wt_ab": np.zeros(npts, dtype=np.float64),
        "wt_ab_tide": np.zeros(npts, dtype=np.float64),
        "snowheat": snowheat,
        "snowgrain": snowgrain,
        "snowliq": snowliq,
        "subsnownobio": np.zeros((npts, nnobio), dtype=np.float64),
        "grndflux": np.zeros(npts, dtype=np.float64),
        "snowmelt": np.zeros(npts, dtype=np.float64),
        "soilflxresid": np.zeros(npts, dtype=np.float64),
    }
    if omit_restart_field is not None:
        hydrol_restart_values.pop(omit_restart_field, None)
    hydrol_restart = SimpleNamespace(**hydrol_restart_values)
    slowproc_restart = SimpleNamespace(
        lai=np.zeros((npts, nvm), dtype=np.float64),
        frac_nobio=np.zeros((npts, nnobio), dtype=np.float64),
        veget=inputs["veget"],
        veget_max=inputs["veget_max"],
        soiltile=inputs["soiltile"],
        totfrac_nobio=np.zeros(npts, dtype=np.float64),
        tot_bare_soil=inputs["veget_max"][:, 0] + np.sum(inputs["veget_max"][:, 1:] - inputs["veget"][:, 1:], axis=1),
    )
    thermosoil_restart = SimpleNamespace(
        ptn=np.full((npts, nslm, nvm), 268.0, dtype=np.float64),
        gtemp=np.asarray([268.0], dtype=np.float64),
        lambda_snow=np.asarray([0.5], dtype=np.float64),
        cgrnd_snow=np.asarray([[260.0, 258.0, 0.0]], dtype=np.float64),
        dgrnd_snow=np.asarray([[0.10, 0.20, 0.0]], dtype=np.float64),
    )
    precall = assemble_hydrol_first_step_precall_payload(
        enerbil_payload=enerbil_payload,
        hydrol_restart=hydrol_restart,
        slowproc_restart=slowproc_restart,
        pref_soil_veg=inputs["pref_soil_veg"],
        nstm=nstm,
        ext_coeff_vegetfrac=np.ones(nvm, dtype=np.float64),
        precip_rain=np.zeros(npts, dtype=np.float64),
        precip_snow=np.asarray([6.0], dtype=np.float64),
        diffuco_payload=diffuco_payload,
        thermosoil_restart=thermosoil_restart,
        throughfall_by_pft=np.full(nvm, 30.0, dtype=np.float64),
        humcste=np.ones(nvm, dtype=np.float64),
        zz_mm=inputs["zz_mm"],
        diaglev_m=inputs["zz_mm"] / 1000.0,
        peat_hydro=False,
        ok_dgvm=False,
        ok_freeze_cwrr=True,
        dt_days=inputs["dt_days"],
    )
    return precall, inputs


def test_hydrol_runoff_to_peat_without_tides_matches_fortran_tile4_branch():
    inputs = _runoff_tide_inputs()

    result = hydrol_after_under_mcr_runoff_peat_tide_routing(**inputs)

    mineral_sum = 1.0 * 0.20 + 2.0 * 0.30 + 3.0 * 0.10
    np.testing.assert_allclose(np.asarray(result.run2peat), [mineral_sum])
    np.testing.assert_allclose(np.asarray(result.run2man), [8.0])
    np.testing.assert_allclose(np.asarray(result.ru_ns), [[0.0, 0.0, 0.0, 4.0, 5.0, 6.0]])
    np.testing.assert_allclose(np.asarray(result.runoff2peat), np.zeros((1, 6)))
    np.testing.assert_allclose(np.asarray(result.water2infilt), np.zeros((1, 6)))


def test_hydrol_tide_routing_splits_mineral_runoff_between_peat_and_mangrove_tiles():
    inputs = _runoff_tide_inputs()
    inputs["tides"] = True

    result = hydrol_after_under_mcr_runoff_peat_tide_routing(**inputs)

    mineral_sum = 1.0 * 0.20 + 2.0 * 0.30 + 3.0 * 0.10
    split_den = 0.25 + 0.10
    np.testing.assert_allclose(np.asarray(result.run2peat), [0.25 / split_den * mineral_sum])
    np.testing.assert_allclose(np.asarray(result.run2man), [0.10 / split_den * mineral_sum])
    np.testing.assert_allclose(np.asarray(result.runoff2peat)[0, :3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(np.asarray(result.ru_ns)[0, :3], [0.0, 0.0, 0.0])


def test_hydrol_tide_routing_uses_tile6_area_when_peat_tile4_is_absent():
    inputs = _runoff_tide_inputs()
    inputs["tides"] = True
    inputs["soiltile"] = np.asarray([[0.20, 0.30, 0.10, 0.0, 0.05, 0.10]], dtype=np.float64)

    result = hydrol_after_under_mcr_runoff_peat_tide_routing(**inputs)

    mineral_sum = 1.0 * 0.20 + 2.0 * 0.30 + 3.0 * 0.10
    np.testing.assert_allclose(np.asarray(result.run2peat), [0.0])
    np.testing.assert_allclose(np.asarray(result.run2man), [0.10 * mineral_sum])
    np.testing.assert_allclose(np.asarray(result.ru_ns)[0, :3], [0.0, 0.0, 0.0])


def test_hydrol_tide_reservoir_sign_cases_match_fortran_writeback():
    cases = [
        (0.5, 0.2, 0.0, 0.5 + 4.0, 0.5 + 4.0),
        (0.0, 0.0, 4.0, 0.0, 0.0),
        # hydrol.f90 lines 6280-6285: the <= 0 / <= 0 branch is tested before
        # the <= 0 / < 0 branch, so this source-order case leaves runoff intact.
        (0.0, -0.2, 4.0, 0.0, 0.0),
        (0.5, -0.2, 4.2, 0.5, 0.5),
    ]
    for wtp_tide, dwtp_tide, expected_ru4, expected_wt_ab_tide, expected_infilt4 in cases:
        inputs = _runoff_tide_inputs()
        inputs.update(
            tides=True,
            peat_hydro=False,
            ok_ru2peat=False,
            wtp_tide=wtp_tide,
            dwtp_tide=dwtp_tide,
        )

        result = hydrol_after_under_mcr_runoff_peat_tide_routing(**inputs)

        np.testing.assert_allclose(np.asarray(result.ru_ns)[0, 3], expected_ru4)
        np.testing.assert_allclose(np.asarray(result.wt_ab_tide), [expected_wt_ab_tide])
        np.testing.assert_allclose(np.asarray(result.water2infilt)[0, 3], expected_infilt4)


def test_hydrol_tide_reservoir_sign_cases_are_jittable():
    inputs = _runoff_tide_inputs()
    inputs.update(tides=True, peat_hydro=False, ok_ru2peat=False)
    inputs.pop("wtp_tide")
    inputs.pop("dwtp_tide")

    transition = jax.jit(
        lambda wtp_tide, dwtp_tide: hydrol_after_under_mcr_runoff_peat_tide_routing(
            **inputs,
            wtp_tide=wtp_tide,
            dwtp_tide=dwtp_tide,
        )
    )
    result = transition(jnp.asarray(0.5), jnp.asarray(-0.2))

    np.testing.assert_allclose(np.asarray(result.ru_ns)[0, 3], 4.2)
    np.testing.assert_allclose(np.asarray(result.wt_ab_tide), [0.5])
    np.testing.assert_allclose(np.asarray(result.water2infilt)[0, 3], 0.5)


def test_hydrol_runoff_peat_post_loop_reinjects_peat_mangrove_and_above_surface_reservoirs():
    ru_ns = np.asarray([[0.0, 0.0, 0.0, 4.0, 5.0, 6.0]], dtype=np.float64)
    soiltile = np.asarray([[0.20, 0.30, 0.10, 0.25, 0.05, 0.10]], dtype=np.float64)
    water2infilt = np.zeros((1, 6), dtype=np.float64)
    tmc = np.ones((1, 6), dtype=np.float64)

    result = hydrol_runoff_peat_post_loop_reinjection(
        ru_ns=ru_ns,
        water2infilt=water2infilt,
        tmc=tmc,
        soiltile=soiltile,
        run2peat=np.asarray([0.35], dtype=np.float64),
        run2man=np.asarray([0.14], dtype=np.float64),
        wt_ab=np.asarray([0.2], dtype=np.float64),
        peat_hydro=True,
        agri_peat=True,
        ok_ru2peat=True,
        tides=True,
    )

    expected_run2peat = 0.35 + 5.0 * 0.05
    np.testing.assert_allclose(np.asarray(result.ru_ns)[0, 4], 0.0)
    np.testing.assert_allclose(np.asarray(result.run2peat), [expected_run2peat])
    np.testing.assert_allclose(np.asarray(result.water2infilt)[0, 3], expected_run2peat / 0.25 + 0.2)
    np.testing.assert_allclose(np.asarray(result.water2infilt)[0, 5], 0.14 / 0.10 + 0.2)
    np.testing.assert_allclose(np.asarray(result.tmc), tmc + np.asarray(result.water2infilt))
    np.testing.assert_allclose(np.asarray(result.tmc_soil), tmc)


def test_hydrol_soil_froz_profile_matches_source_linear_and_correction_formula():
    temp = np.array([[272.15, 273.15, 274.15]], dtype=np.float64)
    mc = np.full((1, 3, 1), 0.30, dtype=np.float64)
    dh = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    result = hydrol_soil_froz_profile(
        temp_hydro=temp,
        mc=mc,
        njsc=np.array([2], dtype=np.int32),
        dh_mm=dh,
        ok_thermodynamical_freezing=False,
        fr_dt=2.0,
        froz_frac_corr=1.0,
        smtot_corr=2.0,
        max_froz_hydro=1.0,
    )

    base = np.array([[1.0, 0.5, 0.0]], dtype=np.float64)
    froz_frac_moy = np.sum(dh * base[0]) / np.sum(dh)
    smtot_moy = np.sum(dh[:-1] * (mc[0, :-1, 0] / 0.41)) / np.sum(dh[:-1])
    expected = base * froz_frac_moy * smtot_moy**2
    np.testing.assert_allclose(np.asarray(result)[0, :, 0], expected[0])


def test_hydrol_first_step_precall_assembly_closes_enerbil_boundary_without_fabricating_hydrol_internals():
    values = parse_run_def(USED_RUN_DEF)
    static_truth = read_static_fields_from_fixed_format_trace_dir(FULL_TRACE_DIR)
    scaffold = paper_1961_driver_timestep_scaffold(
        CONFIG,
        year=1961,
        tstep=0,
        static_trace_fields=static_truth,
        used_run_def_path=USED_RUN_DEF,
        root=ROOT,
    )

    assert scaffold.enerbil_first_step_local is not None
    assert scaffold.diffuco_first_step_precall is not None
    assert scaffold.first_step_restart_state is not None
    nvm = scaffold.first_step_restart_state.slowproc.lai.shape[1]
    assembly = assemble_hydrol_first_step_precall_payload(
        enerbil_payload=scaffold.enerbil_first_step_local.payload,
        hydrol_restart=scaffold.first_step_restart_state.hydrol,
        slowproc_restart=scaffold.first_step_restart_state.slowproc,
        pref_soil_veg=scaffold.step.run_scalars.pref_soil_veg,
        nstm=scaffold.step.run_scalars.nstm,
        ext_coeff_vegetfrac=parse_run_def_indexed_vector(values, "EXT_COEFF_VEGETFRAC", nvm),
        precip_rain=scaffold.payload.precip_rain,
        precip_snow=scaffold.payload.precip_snow,
        diffuco_payload=scaffold.diffuco_first_step_precall.payload,
        thermosoil_restart=scaffold.first_step_restart_state.thermosoil,
        throughfall_by_pft=parse_run_def_indexed_vector(values, "PERCENT_THROUGHFALL_PFT", nvm),
        humcste=parse_run_def_indexed_vector(values, "HYDROL_HUMCSTE", nvm),
        zz_mm=scaffold.stomate_pre_step.pre_step.boundary_inputs.ok_leak_inputs["z_soil"][1:] * 1000.0,
        diaglev_m=scaffold.stomate_pre_step.pre_step.boundary_inputs.ok_leak_inputs["z_soil"][1:],
        peat_hydro=parse_run_def_bool(values["PEAT_HYDRO"]),
        ok_dgvm=parse_run_def_bool(values["STOMATE_OK_DGVM"]),
        ok_explicitsnow=parse_run_def_bool(values["OK_EXPLICITSNOW"]),
        ok_freeze_cwrr=parse_run_def_bool(values["OK_FREEZE_CWRR"]),
        ok_thermodynamical_freezing=parse_run_def_bool(values["OK_THERMODYNAMICAL_FREEZING"]),
        fr_dt=parse_run_def_float(values, "FR_DT"),
        froz_frac_corr=parse_run_def_float(values, "FROZ_FRAC_CORR"),
        smtot_corr=parse_run_def_float(values, "SMTOT_CORR"),
        max_froz_hydro=parse_run_def_float(values, "MAX_FROZ_HYDRO"),
        dt_days=parse_run_def_float(values, "DT_SECHIBA") / 86400.0,
    )

    assert assembly.ok
    assert assembly.enerbil_boundary.ok
    assert set(assembly.enerbil_inputs) >= {
        "temp_sol_new",
        "transpir",
        "transpot",
        "vevapwet",
        "vevapnu",
        "vevapnu_pft",
        "vevapsno",
        "vevapflo",
        "evapot",
        "evapot_corr",
        "pgflux",
        "temp_sol_add",
    }
    assert "mc" in assembly.restart_inputs
    assert "mcl" in assembly.restart_inputs
    assert np.asarray(assembly.payload["mc"]).shape == (1, 11, 6)
    assert np.asarray(assembly.payload["mcl"]).shape == (1, 11, 6)
    np.testing.assert_allclose(
        np.asarray(assembly.payload["mc"])[:, :, 3],
        scaffold.first_step_restart_state.hydrol.moistc[:, 3, :],
    )
    np.testing.assert_allclose(np.asarray(assembly.payload["soiltile"]), [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    np.testing.assert_allclose(np.asarray(assembly.payload["throughfall_by_pft"]), np.full((14,), 0.30))
    assert "humrelv" not in assembly.missing_inputs
    assert "us" not in assembly.missing_inputs
    assert "evap_bare_lim_ns" not in assembly.missing_inputs
    assert "profil_froz" not in assembly.missing_inputs
    assert np.asarray(assembly.payload["us"]).shape == (1, 14, 6, 11)
    assert np.asarray(assembly.payload["humrelv"]).shape == (1, 14, 6)
    np.testing.assert_allclose(
        np.asarray(assembly.payload["humrelv"]),
        np.sum(np.asarray(assembly.payload["us"]), axis=3),
    )
    assert "kfact_root" not in assembly.missing_inputs
    assert np.asarray(assembly.payload["kfact_root"]).shape == (1, 11, 6)
    assert np.any(np.asarray(assembly.payload["kfact_root"])[0, :, 3] > 1.0)
    assert np.asarray(assembly.payload["profil_froz"]).shape == (1, 11, 6)
    assert np.asarray(assembly.payload["temp_hydro"]).shape == (1, 11)
    assert np.asarray(assembly.payload["stempdiag"]).shape == (1, 11)
    expected_profil, expected_stempdiag, expected_temp_hydro = hydrol_first_step_profil_froz_from_restart_thermosoil(
        thermosoil_restart=scaffold.first_step_restart_state.thermosoil,
        veget_max=assembly.payload["veget_max"],
        mc=assembly.payload["mc"],
        njsc=scaffold.first_step_restart_state.hydrol.njsc,
        zz_mm=scaffold.stomate_pre_step.pre_step.boundary_inputs.ok_leak_inputs["z_soil"][1:] * 1000.0,
        diaglev_m=scaffold.stomate_pre_step.pre_step.boundary_inputs.ok_leak_inputs["z_soil"][1:],
        ok_explicitsnow=parse_run_def_bool(values["OK_EXPLICITSNOW"]),
        peat_hydro=parse_run_def_bool(values["PEAT_HYDRO"]),
        ok_thermodynamical_freezing=parse_run_def_bool(values["OK_THERMODYNAMICAL_FREEZING"]),
        fr_dt=parse_run_def_float(values, "FR_DT"),
        froz_frac_corr=parse_run_def_float(values, "FROZ_FRAC_CORR"),
        smtot_corr=parse_run_def_float(values, "SMTOT_CORR"),
        max_froz_hydro=parse_run_def_float(values, "MAX_FROZ_HYDRO"),
    )
    np.testing.assert_allclose(np.asarray(assembly.payload["profil_froz"]), np.asarray(expected_profil))
    np.testing.assert_allclose(np.asarray(assembly.payload["stempdiag"]), np.asarray(expected_stempdiag))
    np.testing.assert_allclose(np.asarray(assembly.payload["temp_hydro"]), np.asarray(expected_temp_hydro))
    assert "same_step_enerbil_evaporation_inputs" not in assembly.missing_inputs
    assert set(assembly.payload).issubset(
        set(HYDROL_FIRST_STEP_PRECALL_REQUIRED_FIELDS)
        | {
            "snowdz",
            "wt_ab",
            "wt_ab_tide",
            "stempdiag",
            "temp_hydro",
            "dz_mm",
            "zz_mm",
            "dt_days",
            "njsc",
                "mcr",
                "mcs",
                "mcw",
                "mcf",
                "soilcap",
                "gtemp",
            "lambda_snow",
            "cgrnd_snow",
            "dgrnd_snow",
        }
    )


def test_hydrol_first_step_precall_accepts_cold_start_freeze_profile_without_thermosoil_restart():
    npts = 1
    nvm = 3
    nstm = 2
    nslm = 4
    enerbil_payload = {
        "temp_sol_new": np.asarray([280.0]),
        "pgflux": np.asarray([0.0]),
        "temp_sol_add": np.asarray([0.0]),
        "vevapsno": np.asarray([0.0]),
        "vevapwet": np.zeros((npts, nvm)),
        "vevapflo": np.asarray([0.0]),
        "transpir": np.zeros((npts, nvm)),
        "transpot": np.zeros((npts, nvm)),
        "vevapnu": np.asarray([0.0]),
        "vevapnu_pft": np.zeros((npts, nvm)),
        "evapot": np.asarray([0.0]),
        "evapot_corr": np.asarray([0.0]),
    }
    diffuco_payload = {
        "qsintmax": np.zeros((npts, nvm)),
        "qsintveg": np.zeros((npts, nvm)),
        "evap_bare_lim": np.asarray([0.0]),
        "flood_frac": np.asarray([0.0]),
        "flood_res": np.asarray([0.0]),
        "snow": np.asarray([0.0]),
        "snowdz": np.zeros((npts, 3)),
        "snowrho": np.full((npts, 3), 50.0),
        "snowtemp": np.full((npts, 3), 273.15),
        "snow_age": np.asarray([0.0]),
        "snow_nobio": np.zeros((npts, 1)),
        "snow_nobio_age": np.zeros((npts, 1)),
    }
    profil = np.zeros((npts, nslm, nstm), dtype=np.float64)
    temp_hydro = np.full((npts, nslm), 280.0, dtype=np.float64)
    hydrol_restart = SimpleNamespace(
        moistc=np.full((npts, nstm, nslm), 0.3, dtype=np.float64),
        moistcl=np.full((npts, nstm, nslm), 0.3, dtype=np.float64),
        ae_ns=np.zeros((npts, nstm)),
        water2infilt=np.zeros((npts, nstm)),
        free_drain_coef=np.ones((npts, nstm)),
        zwt_force=np.full((npts, nstm), 1.0e20),
        evap_bare_lim_ns=np.zeros((npts, nstm)),
        us=np.zeros((npts, nvm, nstm, nslm)),
        humrel=np.zeros((npts, nvm)),
        njsc=np.asarray([2], dtype=np.int32),
        profil_froz_hydro_ns=profil,
        temp_hydro=temp_hydro,
    )
    slowproc_restart = SimpleNamespace(
        lai=np.zeros((npts, nvm)),
        frac_nobio=np.zeros((npts, 1)),
        veget_max=np.asarray([[0.2, 0.3, 0.5]], dtype=np.float64),
    )

    assembly = assemble_hydrol_first_step_precall_payload(
        enerbil_payload=enerbil_payload,
        hydrol_restart=hydrol_restart,
        slowproc_restart=slowproc_restart,
        pref_soil_veg=np.asarray([1, 1, 2], dtype=np.int32),
        nstm=nstm,
        ext_coeff_vegetfrac=np.ones((nvm,), dtype=np.float64),
        precip_rain=np.asarray([0.0]),
        precip_snow=np.asarray([0.0]),
        diffuco_payload=diffuco_payload,
        thermosoil_restart=None,
        throughfall_by_pft=np.zeros((nvm,), dtype=np.float64),
        humcste=np.ones((nvm,), dtype=np.float64),
        zz_mm=np.asarray([1.0, 4.0, 10.0, 18.0], dtype=np.float64),
        ok_freeze_cwrr=True,
        dt_days=0.5,
    )

    assert assembly.ok
    assert "profil_froz" in assembly.restart_inputs
    assert "temp_hydro" in assembly.restart_inputs
    np.testing.assert_allclose(np.asarray(assembly.payload["profil_froz"]), profil)
    np.testing.assert_allclose(np.asarray(assembly.payload["temp_hydro"]), temp_hydro)


def test_hydrol_precall_uses_explicit_slowproc_vegetation_when_lai_was_reset_to_zero():
    npts = 1
    nvm = 3
    nstm = 2
    slowproc_restart = SimpleNamespace(
        lai=np.zeros((npts, nvm), dtype=np.float64),
        frac_nobio=np.zeros((npts, 1), dtype=np.float64),
        veget=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
        veget_max=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
        soiltile=np.asarray([[0.0, 1.0]], dtype=np.float64),
        tot_bare_soil=np.asarray([0.0], dtype=np.float64),
    )

    static = hydrol_static_precall_template_from_slowproc(
        slowproc_restart=slowproc_restart,
        pref_soil_veg=np.asarray([1, 1, 2], dtype=np.int32),
        nstm=nstm,
        ext_coeff_vegetfrac=np.ones((nvm,), dtype=np.float64),
        hydrol_njsc=np.asarray([2], dtype=np.int32),
    )
    vegupd = hydrol_vegupd_static_state(
        veget=static["veget"],
        veget_max=static["veget_max"],
        soiltile=static["soiltile"],
        vegtot=static["vegtot"],
        pref_soil_veg=np.asarray([1, 1, 2], dtype=np.int32),
    )

    np.testing.assert_allclose(np.asarray(static["veget"]), slowproc_restart.veget)
    np.testing.assert_allclose(np.asarray(static["soiltile"]), slowproc_restart.soiltile)
    np.testing.assert_allclose(np.asarray(vegupd.frac_bare), np.zeros((npts, nvm)))
    np.testing.assert_allclose(np.asarray(vegupd.frac_bare_ns), np.zeros((npts, nstm)))


def test_hydrol_module_explicit_step_matches_manual_source_order_assembly():
    inputs = _module_inputs()
    tables = build_mineral_cwrr_tables(njsc=2, mcs=inputs["mcs"], z_m=inputs["zz_mm"] / 1000.0)

    module = hydrol_module_explicit_step(**inputs, mineral_tables=tables)

    vegupd = hydrol_vegupd_static_state(
        veget=inputs["veget"],
        veget_max=inputs["veget_max"],
        soiltile=inputs["soiltile"],
        vegtot=inputs["vegtot"],
        pref_soil_veg=inputs["pref_soil_veg"],
    )
    canop = hydrol_canop_interception(
        precip_rain=inputs["precip_rain"],
        vevapwet=inputs["vevapwet"],
        veget_max=inputs["veget_max"],
        veget=inputs["veget"],
        qsintmax=inputs["qsintmax"],
        qsintveg=inputs["qsintveg"],
        tot_melt=inputs["tot_melt"],
        vegtot=inputs["vegtot"],
        throughfall_by_pft=inputs["throughfall_by_pft"],
    )
    flood = hydrol_flood_reservoir(
        vevapflo=inputs["vevapflo"],
        flood_frac=inputs["flood_frac"],
        flood_res=inputs["flood_res"],
        precisol=canop.precisol,
        subsinksoil=inputs["subsinksoil"],
    )
    split = hydrol_split_soil_fluxes(
        veget_max=inputs["veget_max"],
        soiltile=inputs["soiltile"],
        precisol=flood.precisol,
        vevapnu=inputs["vevapnu"],
        vevapnu_pft=inputs["vevapnu_pft"],
        transpir=inputs["transpir"],
        humrel=inputs["humrel"],
        humrelv=inputs["humrelv"],
        us=inputs["us"],
        ae_ns=inputs["ae_ns"],
        evap_bare_lim=inputs["evap_bare_lim"],
        evap_bare_lim_ns=inputs["evap_bare_lim_ns"],
        frac_bare_ns=vegupd.frac_bare_ns,
        tot_bare_soil=np.sum(np.asarray(vegupd.frac_bare) * inputs["veget_max"], axis=1),
        vegetmax_soil=vegupd.vegetmax_soil,
        vegtot=inputs["vegtot"],
        pref_soil_veg=inputs["pref_soil_veg"],
    )
    soil = hydrol_soil_all_tiles_explicit_step(
        water2infilt=inputs["water2infilt"],
        ae_ns=split.ae_ns,
        subsinksoil=flood.subsinksoil,
        precisol_ns=split.precisol_ns,
        reinfiltration_soil=inputs["reinfiltration_soil"],
        is_crop_soil=False,
        mc=inputs["mc"],
        mcl=inputs["mcl"],
        profil_froz=inputs["profil_froz"],
        mcr=inputs["mcr"],
        mcs=inputs["mcs"],
        dz_mm=inputs["dz_mm"],
        dt_days=inputs["dt_days"],
        free_drain_coef=inputs["free_drain_coef"],
        rootsink=split.rootsink,
        resolv=inputs["resolv"],
        mask_soiltile=vegupd.mask_soiltile,
        kfact_root=inputs["kfact_root"],
        ks=inputs["ks"],
        kfact=inputs["kfact"],
        soiltile=inputs["soiltile"],
        tmc=inputs["tmc"],
        njsc=inputs["njsc"],
        mineral_tables=tables,
        reinf_slope=inputs["reinf_slope"],
        zz_mm=inputs["zz_mm"],
        zmaxh_m=inputs["zmaxh_m"],
        zwt_force=inputs["zwt_force"],
    )

    np.testing.assert_allclose(np.asarray(module.canop.precisol), np.asarray(canop.precisol))
    np.testing.assert_allclose(np.asarray(module.flood.precisol), np.asarray(flood.precisol))
    np.testing.assert_allclose(np.asarray(module.split.precisol_ns), np.asarray(split.precisol_ns))
    np.testing.assert_allclose(np.asarray(module.soil.mc), np.asarray(soil.mc))
    np.testing.assert_allclose(np.asarray(module.soil.tmc), np.asarray(soil.tmc))


def test_hydrol_module_explicit_step_runs_pft14_peat_tide_path():
    inputs = _module_inputs()
    z_m = inputs["zz_mm"] / 1000.0
    mineral_tables = build_mineral_cwrr_tables(njsc=2, mcs=np.asarray([0.50]), z_m=z_m)
    peat_tables = build_peat_cwrr_tables(z_m)
    inputs["mcr"] = np.asarray([PEAT_MCR], dtype=np.float64)
    inputs["mcs"] = np.asarray([PEAT_MCS], dtype=np.float64)
    inputs["mc"][:, :, 3] = np.linspace(0.30, 0.45, inputs["mc"].shape[1], dtype=np.float64)[None, :]
    inputs["mcl"] = inputs["mc"].copy()

    result = hydrol_module_explicit_step(
        **inputs,
        mineral_tables=mineral_tables,
        peat_tables=peat_tables,
        peat_hydro=True,
        tides=True,
        ok_ru2peat=True,
        ok_wt_ab=True,
        wtp_tide=0.5,
        dwtp_tide=0.2,
    )

    assert len(result.soil.tile_results) == 6
    assert result.soil.tile_results[3].coef_before_infilt is not None
    assert np.asarray(result.soil.wt_ab_tide)[0] >= 0.0
    assert np.asarray(result.soil.water2infilt[0, 3]) >= 0.0


def test_hydrol_module_explicit_step_applies_finite_water_table_forcing_to_prognostic_state():
    """hydrol.f90:6116-6180 executes inside the production module wrapper."""

    inputs = _module_inputs()
    inputs["zwt_force"] = np.full_like(inputs["zwt_force"], 1.0e20)
    inputs["zwt_force"][:, 0] = 0.010
    tables = build_mineral_cwrr_tables(njsc=2, mcs=inputs["mcs"], z_m=inputs["zz_mm"] / 1000.0)

    forced = hydrol_module_explicit_step(**inputs, mineral_tables=tables)
    baseline_inputs = dict(inputs)
    baseline_inputs["zwt_force"] = np.full_like(inputs["zwt_force"], 1.0e20)
    baseline = hydrol_module_explicit_step(**baseline_inputs, mineral_tables=tables)

    assert not np.allclose(np.asarray(forced.soil.mc[:, :, 0]), np.asarray(baseline.soil.mc[:, :, 0]))
    assert not np.allclose(np.asarray(forced.soil.dr_ns[:, 0]), np.asarray(baseline.soil.dr_ns[:, 0]))


def test_hydrol_module_outputs_expose_downstream_soil_flux_fields():
    inputs = _module_inputs()
    tables = build_mineral_cwrr_tables(njsc=2, mcs=inputs["mcs"], z_m=inputs["zz_mm"] / 1000.0)
    result = hydrol_module_explicit_step(**inputs, mineral_tables=tables)

    outputs = hydrol_module_outputs(result, soiltile=inputs["soiltile"])

    np.testing.assert_allclose(np.asarray(outputs.runoff_per_soil), np.asarray(result.soil.ru_ns))
    np.testing.assert_allclose(np.asarray(outputs.drainage_per_soil), np.asarray(result.soil.dr_ns))
    np.testing.assert_allclose(np.asarray(outputs.runoff2peat), np.asarray(result.soil.runoff2peat))
    np.testing.assert_allclose(np.asarray(outputs.tmc), np.asarray(result.soil.tmc))
    np.testing.assert_allclose(np.asarray(outputs.tmc_soil), np.asarray(result.soil.tmc_soil))
    np.testing.assert_allclose(np.asarray(outputs.wat_flux), np.asarray(result.soil.qflux))
    np.testing.assert_allclose(
        np.asarray(outputs.wtd),
        np.sum(inputs["soiltile"] * np.asarray(result.soil.wtd_ns), axis=1),
    )
    np.testing.assert_allclose(np.asarray(outputs.precip2ground), np.asarray(result.canop.precip2ground))
    np.testing.assert_allclose(np.asarray(outputs.floodout), np.asarray(result.flood.floodout))


def test_hydrol_first_step_module_closure_runs_outputs_diagnostics_and_thermosoil_boundary():
    values = parse_run_def(USED_RUN_DEF)
    static_truth = read_static_fields_from_fixed_format_trace_dir(FULL_TRACE_DIR)
    scaffold = paper_1961_driver_timestep_scaffold(
        CONFIG,
        year=1961,
        tstep=0,
        static_trace_fields=static_truth,
        used_run_def_path=USED_RUN_DEF,
        root=ROOT,
    )
    assert scaffold.hydrol_first_step_precall is not None
    assert scaffold.first_step_restart_state is not None
    grid = paper_case_cwrr_vertical_soil_grid_from_used_run_def(USED_RUN_DEF)
    nvm = scaffold.first_step_restart_state.slowproc.lai.shape[1]

    closure = run_hydrol_first_step_module_from_precall(
        scaffold.hydrol_first_step_precall,
        humcste=parse_run_def_indexed_vector(values, "HYDROL_HUMCSTE", nvm),
        dz_mm=grid.dnh * 1000.0,
        dh_mm=grid.dlh * 1000.0,
        zz_mm=grid.znh * 1000.0,
        altmax=scaffold.first_step_restart_state.stomate.altmax,
        ks=parse_run_def_indexed_vector(values, "CWRR_KS", 12),
        reinf_slope=[parse_run_def_float(values, "SLOPE")],
        zmaxh_m=parse_run_def_float(values, "DEPTH_MAX_H"),
        doponds=parse_run_def_bool(values["DO_PONDS"]),
        peat_hydro=parse_run_def_bool(values["PEAT_HYDRO"]),
        ok_freeze_cwrr=parse_run_def_bool(values["OK_FREEZE_CWRR"]),
        ok_pc=parse_run_def_bool(values["OK_PC"]),
        ok_leak=parse_run_def_bool(values["OK_LEAK"]),
        ok_ru2peat=parse_run_def_bool(values["OK_RU2PEAT"]),
        ok_wt_ab=parse_run_def_bool(values["OK_WT_AB"]),
        tides=parse_run_def_bool(values["TIDES"]),
        agri_peat=parse_run_def_bool(values["AGRI_PEAT"]),
        dyn_nroot_larix=parse_run_def_bool(values["dyn_nroot_larix"]),
        is_tree=scaffold.step.run_scalars.is_tree,
        znt=grid.diaglev,
        run2peat=scaffold.first_step_restart_state.hydrol.run2peat,
        run2man=scaffold.first_step_restart_state.hydrol.run2man,
    )

    assert closure.ok
    assert closure.missing_inputs == ()
    assert closure.routing_zero is not None
    assert set(closure.routing_zero) == {
        "riverflow",
        "coastalflow",
        "returnflow",
        "reinfiltration",
        "irrigation",
        "sed_depositiontot",
        "poc_depositiontot",
        "stream_inflow",
        "stream_outflow",
        "stream_frac",
        "flood_res",
        "fastr",
    }
    for value in closure.routing_zero.values():
        np.testing.assert_array_equal(np.asarray(value), np.zeros_like(value))
    np.testing.assert_array_equal(
        np.asarray(closure.module_inputs["reinfiltration_soil"]),
        np.zeros_like(np.asarray(closure.module_inputs["soiltile"])),
    )
    np.testing.assert_array_equal(
        np.asarray(closure.module_inputs["irrigation_soil"]),
        np.zeros_like(np.asarray(closure.module_inputs["soiltile"])),
    )
    assert closure.snow_state is not None
    np.testing.assert_allclose(np.asarray(closure.snow_state.snow), [0.0])
    np.testing.assert_allclose(np.asarray(closure.snow_state.snowdz), [[0.0, 0.0, 0.0]])
    np.testing.assert_allclose(np.asarray(closure.snow_state.snowrho), [[50.0, 50.0, 50.0]])
    np.testing.assert_allclose(np.asarray(closure.snow_state.snowtemp), [[273.15, 273.15, 273.15]])
    assert np.asarray(closure.outputs.wat_flux).shape == (1, 11, 6)
    assert np.asarray(closure.diagnostics.humrel).shape == (1, 14)
    assert np.asarray(closure.thermosoil_moisture.mc_layh_pft).shape == (1, 11, 14)
    np.testing.assert_allclose(
        np.asarray(closure.module_inputs["tmc"]),
        np.asarray(
            hydrol_initial_tmc_from_restart_mc(
                mc=scaffold.hydrol_first_step_precall.payload["mc"],
                water2infilt=scaffold.hydrol_first_step_precall.payload["water2infilt"],
                dz_mm=grid.dnh * 1000.0,
            )
        ),
    )
    expected_nroot = hydrol_nroot_from_humcste(
        humcste=parse_run_def_indexed_vector(values, "HYDROL_HUMCSTE", nvm),
        dz_mm=grid.dnh * 1000.0,
        zz_mm=grid.znh * 1000.0,
        altmax=scaffold.first_step_restart_state.stomate.altmax,
        ok_pc=parse_run_def_bool(values["OK_PC"]),
        ok_leak=parse_run_def_bool(values["OK_LEAK"]),
    )
    np.testing.assert_allclose(np.asarray(closure.diagnostic_inputs["nroot"]), np.asarray(expected_nroot))


def test_hydrol_first_step_module_closure_runs_complete_explicit_snow_branch():
    precall, inputs = _explicit_snow_module_precall()

    closure = run_hydrol_first_step_module_from_precall(
        precall,
        temp_air=np.asarray([268.0], dtype=np.float64),
        pb=np.asarray([1013.0], dtype=np.float64),
        u=np.zeros(1, dtype=np.float64),
        v=np.zeros(1, dtype=np.float64),
        humcste=np.ones(inputs["veget_max"].shape[1], dtype=np.float64),
        dz_mm=inputs["dz_mm"],
        dh_mm=np.asarray([1.0, 3.0, 6.0, 8.0, 10.0, 12.0, 14.0], dtype=np.float64),
        zz_mm=inputs["zz_mm"],
        ks=inputs["ks"],
        reinf_slope=inputs["reinf_slope"],
        zmaxh_m=inputs["zmaxh_m"],
        doponds=False,
        peat_hydro=False,
        ok_freeze_cwrr=True,
        is_tree=np.asarray([False, True, False, True]),
        znt=inputs["zz_mm"] / 1000.0,
        dt_sechiba=1800.0,
    )

    assert precall.ok
    assert closure.ok
    assert closure.missing_inputs == ()
    assert closure.snow_step is not None
    assert closure.snow_state is closure.snow_step.snow_state
    np.testing.assert_allclose(
        np.asarray(closure.module_inputs["tot_melt"]),
        np.asarray(closure.snow_state.tot_melt),
    )
    np.testing.assert_allclose(
        np.asarray(closure.module_inputs["subsinksoil"]),
        np.asarray(closure.snow_step.subsinksoil),
    )
    np.testing.assert_allclose(
        np.asarray(closure.snow_state.snow),
        np.asarray(closure.snow_state.snowrho * closure.snow_state.snowdz).sum(axis=1),
    )
    assert np.asarray(closure.snow_state.snowdz).shape == (1, 3)
    assert np.all(np.asarray(closure.snow_state.snowdz) >= 0.0)
    assert np.all(np.asarray(closure.snow_step.snowliq) >= 0.0)
    assert any("hydrol.f90::hydrol_main lines 1177-1197" in item for item in closure.snow_step.provenance)

    fields = _empty_previous_step_state_fields()
    _add_hydrol_to_previous_fields(fields, closure)
    hydrol_state = fields["hydrol_previous_step_state"]
    for name in (
        "snow",
        "snowdz",
        "snowrho",
        "snowtemp",
        "snowheat",
        "snowgrain",
        "snow_age",
        "snow_nobio",
        "snow_nobio_age",
        "snowliq",
        "soilflxresid",
    ):
        assert name in hydrol_state
    np.testing.assert_allclose(np.asarray(hydrol_state["snowliq"]), np.asarray(closure.snow_step.snowliq))


def test_hydrol_explicit_snow_melt_feeds_same_step_canopy_and_soil():
    precall, inputs = _explicit_snow_module_precall()
    payload = dict(precall.payload)
    payload.update(
        precip_rain=np.zeros(1, dtype=np.float64),
        precip_snow=np.zeros(1, dtype=np.float64),
        snow=np.asarray([30.0], dtype=np.float64),
        snowdz=np.full((1, 3), 0.05, dtype=np.float64),
        snowrho=np.full((1, 3), 200.0, dtype=np.float64),
        snowtemp=np.full((1, 3), 275.0, dtype=np.float64),
        snowheat=np.zeros((1, 3), dtype=np.float64),
        temp_sol_new=np.asarray([280.0], dtype=np.float64),
        gtemp=np.asarray([280.0], dtype=np.float64),
    )
    precall = precall._replace(payload=payload)

    closure = run_hydrol_first_step_module_from_precall(
        precall,
        temp_air=np.asarray([280.0], dtype=np.float64),
        pb=np.asarray([1013.0], dtype=np.float64),
        u=np.zeros(1, dtype=np.float64),
        v=np.zeros(1, dtype=np.float64),
        humcste=np.ones(inputs["veget_max"].shape[1], dtype=np.float64),
        dz_mm=inputs["dz_mm"],
        dh_mm=np.asarray([1.0, 3.0, 6.0, 8.0, 10.0, 12.0, 14.0], dtype=np.float64),
        zz_mm=inputs["zz_mm"],
        ks=inputs["ks"],
        reinf_slope=inputs["reinf_slope"],
        zmaxh_m=inputs["zmaxh_m"],
        doponds=False,
        peat_hydro=False,
        ok_freeze_cwrr=True,
        is_tree=np.asarray([False, True, False, True]),
        znt=inputs["zz_mm"] / 1000.0,
        dt_sechiba=1800.0,
    )

    melt = np.asarray(closure.snow_state.tot_melt)
    assert closure.ok
    assert np.all(melt > 0.0)
    np.testing.assert_allclose(np.asarray(closure.module_inputs["tot_melt"]), melt)
    np.testing.assert_allclose(np.asarray(closure.module.canop.precisol).sum(axis=1), melt)


def test_hydrol_first_step_module_closure_runs_bucket_snow_branch_when_explicit_snow_disabled():
    precall, inputs = _explicit_snow_module_precall()

    closure = run_hydrol_first_step_module_from_precall(
        precall,
        temp_air=np.asarray([268.0], dtype=np.float64),
        pb=np.asarray([1013.0], dtype=np.float64),
        u=np.zeros(1, dtype=np.float64),
        v=np.zeros(1, dtype=np.float64),
        humcste=np.ones(inputs["veget_max"].shape[1], dtype=np.float64),
        dz_mm=inputs["dz_mm"],
        dh_mm=np.asarray([1.0, 3.0, 6.0, 8.0, 10.0, 12.0, 14.0], dtype=np.float64),
        zz_mm=inputs["zz_mm"],
        ks=inputs["ks"],
        reinf_slope=inputs["reinf_slope"],
        zmaxh_m=inputs["zmaxh_m"],
        doponds=False,
        peat_hydro=False,
        ok_freeze_cwrr=True,
        ok_explicitsnow=False,
        is_tree=np.asarray([False, True, False, True]),
        znt=inputs["zz_mm"] / 1000.0,
        dt_sechiba=1800.0,
    )

    assert precall.ok
    assert closure.ok
    assert closure.missing_inputs == ()
    assert closure.snow_step is None
    assert closure.snow_state is None
    assert closure.bucket_snow_step is not None
    assert np.asarray(closure.bucket_snow_step.snow).shape == (1,)
    assert np.asarray(closure.bucket_snow_step.snow_nobio).shape == (1, 1)
    assert any("hydrol_snow lines 4693-4966" in item for item in closure.bucket_snow_step.provenance)


def test_hydrol_first_step_module_closure_reports_missing_explicit_snow_inputs_without_defaults():
    precall, inputs = _explicit_snow_module_precall(omit_restart_field="soilflxresid")

    closure = run_hydrol_first_step_module_from_precall(
        precall,
        temp_air=np.asarray([268.0], dtype=np.float64),
        pb=np.asarray([1013.0], dtype=np.float64),
        u=np.zeros(1, dtype=np.float64),
        v=np.zeros(1, dtype=np.float64),
        humcste=np.ones(inputs["veget_max"].shape[1], dtype=np.float64),
        dz_mm=inputs["dz_mm"],
        dh_mm=np.asarray([1.0, 3.0, 6.0, 8.0, 10.0, 12.0, 14.0], dtype=np.float64),
        zz_mm=inputs["zz_mm"],
        ks=inputs["ks"],
        reinf_slope=inputs["reinf_slope"],
        zmaxh_m=inputs["zmaxh_m"],
        doponds=False,
        peat_hydro=False,
        ok_freeze_cwrr=True,
        is_tree=np.asarray([False, True, False, True]),
        znt=inputs["zz_mm"] / 1000.0,
        dt_sechiba=1800.0,
    )

    assert precall.ok
    assert not closure.ok
    assert "explicit_snow:soilflxresid" in closure.missing_inputs
    assert closure.module is None
    assert closure.snow_step is None
    assert closure.snow_state is None


def test_hydrol_module_diagnostics_expose_downstream_water_state_boundary():
    inputs = _module_inputs()
    tables = build_mineral_cwrr_tables(njsc=2, mcs=inputs["mcs"], z_m=inputs["zz_mm"] / 1000.0)
    result = hydrol_module_explicit_step(**inputs, mineral_tables=tables)
    npts, nslm, nstm = result.soil.mc.shape
    nvm = inputs["veget_max"].shape[1]
    nroot = np.zeros((npts, nvm, nslm), dtype=np.float64)
    nroot_profile = np.asarray([0.0, 0.10, 0.15, 0.20, 0.20, 0.20, 0.15], dtype=np.float64)
    nroot[:, :, 1:] = nroot_profile[1:]
    nroot[:, :, 1:] = nroot[:, :, 1:] / nroot[:, :, 1:].sum(axis=2, keepdims=True)
    dh_mm = np.asarray([1.0, 3.0, 6.0, 8.0, 10.0, 12.0, 14.0], dtype=np.float64)

    diagnostics = hydrol_module_diagnostics(
        result,
        nroot=nroot,
        dz_mm=inputs["dz_mm"],
        dh_mm=dh_mm,
        njsc=inputs["njsc"],
        soiltile=inputs["soiltile"],
        vegtot=inputs["vegtot"],
        mcs=inputs["mcs"],
        mineral_tables=tables,
        vevapnu=inputs["vevapnu"],
        tot_melt=inputs["tot_melt"],
        irrigation=np.zeros_like(inputs["vegtot"]),
        returnflow=np.zeros_like(inputs["vegtot"]),
        reinfiltration=np.zeros_like(inputs["vegtot"]),
        peat_hydro=False,
    )

    assert len(diagnostics.layer_by_tile) == nstm
    assert diagnostics.us.shape == (npts, nvm, nstm, nslm)
    assert diagnostics.humrel.shape == inputs["veget_max"].shape
    assert diagnostics.shumdiag.shape == (npts, nslm)
    assert diagnostics.mc_layh_s.shape == result.soil.mc.shape
    assert diagnostics.tmc_litter.shape == inputs["soiltile"].shape
    assert diagnostics.k_litt.shape == inputs["vegtot"].shape
    assert np.all(np.asarray(diagnostics.humrel[:, 0]) == 0.0)
    assert np.all(np.asarray(diagnostics.soil_wet_ns) >= 0.0)
    assert np.all(np.asarray(diagnostics.soil_wet_ns) <= 1.0)
    assert np.all(np.asarray(diagnostics.soil_wet_litter) >= 0.0)
    assert np.all(np.asarray(diagnostics.soil_wet_litter) <= 1.0)

    expected_soilmoist = np.zeros((npts, nslm), dtype=np.float64)
    for jst in range(nstm):
        expected_soilmoist += inputs["soiltile"][:, jst, None] * np.asarray(
            hydrol_layer_moisture_content(np.asarray(result.soil.mc)[:, :, jst], inputs["dz_mm"])
        )
    expected_soilmoist *= inputs["vegtot"][:, None]
    np.testing.assert_allclose(np.asarray(diagnostics.soilmoist), expected_soilmoist)
    np.testing.assert_allclose(np.asarray(diagnostics.mcl_layh_s), np.asarray(result.soil.mc))
    np.testing.assert_allclose(
        np.asarray(diagnostics.humtot),
        np.sum(inputs["vegtot"][:, None] * inputs["soiltile"] * np.asarray(result.soil.tmc), axis=1),
    )
    np.testing.assert_allclose(np.asarray(diagnostics.vevapnu), inputs["vevapnu"] + np.asarray(result.flood.subsinksoil) * inputs["vegtot"])


def test_hydrol_module_masks_final_drainage_for_absent_soil_tile():
    inputs = _module_inputs()
    absent_tile = 3
    inputs["soiltile"] = inputs["soiltile"].copy()
    inputs["soiltile"][0, 0] += inputs["soiltile"][0, absent_tile]
    inputs["soiltile"][0, absent_tile] = 0.0
    for field in ("mc", "mcl", "profil_froz", "kfact_root"):
        inputs[field] = inputs[field].copy()
        inputs[field][:, :, absent_tile] = 0.0
    for field in (
        "ae_ns",
        "evap_bare_lim_ns",
        "water2infilt",
        "reinfiltration_soil",
    ):
        inputs[field] = inputs[field].copy()
        inputs[field][:, absent_tile] = 0.0

    tables = build_mineral_cwrr_tables(
        njsc=2,
        mcs=inputs["mcs"],
        z_m=inputs["zz_mm"] / 1000.0,
    )
    result = hydrol_module_explicit_step(**inputs, mineral_tables=tables)

    assert np.asarray(result.soil.dr_ns)[0, absent_tile] == 0.0


def test_hydrol_diag_soil_aggregates_frozen_profile_with_tile_mask():
    """Fortran hydrol.f90::hydrol_diag_soil lines 9045-9080."""

    inputs = _module_inputs()
    inputs["ok_freeze_cwrr"] = True
    inputs["temp_hydro"] = np.full(inputs["mc"].shape[:2], 280.0, dtype=np.float64)
    tables = build_mineral_cwrr_tables(njsc=2, mcs=inputs["mcs"], z_m=inputs["zz_mm"] / 1000.0)
    result = hydrol_module_explicit_step(**inputs, mineral_tables=tables)
    diagnostic_inputs = _module_diagnostic_inputs(result, inputs)
    diagnostics = hydrol_module_diagnostics(
        result,
        **diagnostic_inputs,
        mineral_tables=tables,
        ok_freeze_cwrr=True,
    )
    expected = np.sum(
        inputs["vegtot"][:, None, None]
        * inputs["soiltile"][:, None, :]
        * (inputs["vegtot"] > 1.0e-8)[:, None, None]
        * inputs["profil_froz"],
        axis=2,
    )
    np.testing.assert_allclose(np.asarray(diagnostics.profil_froz_hydro), expected)


def test_hydrol_module_diagnostics_use_same_step_alt_residual_evap_bare_limit():
    inputs = _module_inputs()
    tables = build_mineral_cwrr_tables(njsc=2, mcs=inputs["mcs"], z_m=inputs["zz_mm"] / 1000.0)
    baseline = hydrol_module_explicit_step(**inputs, mineral_tables=tables)
    result = hydrol_module_explicit_step(
        **inputs,
        mineral_tables=tables,
        evapot=np.asarray([2.0], dtype=np.float64),
        evapot_corr=np.asarray([20.0], dtype=np.float64),
    )

    diagnostics = hydrol_module_diagnostics(
        result,
        **_module_diagnostic_inputs(result, inputs),
        mineral_tables=tables,
        evap_bare_limit_alt=result.soil.evap_bare_limit_alt,
        tmcint=result.soil.tmcint_for_evap_bare_limit,
        evapot=np.asarray([2.0], dtype=np.float64),
    )

    assert result.soil.evap_bare_limit_alt is not None
    assert result.soil.tmcint_for_evap_bare_limit.shape == inputs["soiltile"].shape
    np.testing.assert_allclose(np.asarray(result.soil.mc), np.asarray(baseline.soil.mc))
    np.testing.assert_allclose(np.asarray(result.soil.tmc), np.asarray(baseline.soil.tmc))
    assert np.max(np.abs(np.asarray(result.soil.evap_bare_limit_alt.mc_after) - np.asarray(result.soil.mc))) > 0.0
    np.testing.assert_allclose(
        np.asarray(diagnostics.evap_bare_lim_ns),
        np.asarray([[1.0, 0.20532379091953755, 0.0, 0.0, 0.0, 0.0]], dtype=np.float64),
    )
    expected_grid_beta = np.sum(
        np.asarray(diagnostics.evap_bare_lim_ns)
        * np.asarray(inputs["vegtot"])[:, None]
        * np.asarray(inputs["soiltile"]),
        axis=1,
    )
    np.testing.assert_allclose(
        np.asarray(diagnostics.evap_bare_lim),
        expected_grid_beta,
    )


def test_hydrol_module_diagnostics_refuses_alt_residual_without_same_step_state():
    inputs = _module_inputs()
    tables = build_mineral_cwrr_tables(njsc=2, mcs=inputs["mcs"], z_m=inputs["zz_mm"] / 1000.0)
    result = hydrol_module_explicit_step(
        **inputs,
        mineral_tables=tables,
        evapot=np.asarray([2.0], dtype=np.float64),
        evapot_corr=np.asarray([20.0], dtype=np.float64),
    )

    with pytest.raises(ValueError, match="tmcint, evapot"):
        hydrol_module_diagnostics(
            result,
            **_module_diagnostic_inputs(result, inputs),
            mineral_tables=tables,
            evap_bare_limit_alt=result.soil.evap_bare_limit_alt,
        )


def test_hydrol_to_thermosoil_moisture_inputs_follow_sechiba_pft_tile_mapping():
    inputs = _module_inputs()
    tables = build_mineral_cwrr_tables(njsc=2, mcs=inputs["mcs"], z_m=inputs["zz_mm"] / 1000.0)
    result = hydrol_module_explicit_step(**inputs, mineral_tables=tables)
    npts, nslm, nstm = result.soil.mc.shape
    nvm = inputs["veget_max"].shape[1]
    nroot = np.zeros((npts, nvm, nslm), dtype=np.float64)
    nroot[:, :, 1:] = 1.0 / (nslm - 1)
    dh_mm = np.asarray([1.0, 3.0, 6.0, 8.0, 10.0, 12.0, 14.0], dtype=np.float64)
    diagnostics = hydrol_module_diagnostics(
        result,
        nroot=nroot,
        dz_mm=inputs["dz_mm"],
        dh_mm=dh_mm,
        njsc=inputs["njsc"],
        soiltile=inputs["soiltile"],
        vegtot=inputs["vegtot"],
        mcs=inputs["mcs"],
        mineral_tables=tables,
        vevapnu=inputs["vevapnu"],
        tot_melt=inputs["tot_melt"],
    )

    payload = hydrol_to_thermosoil_moisture_inputs(
        diagnostics,
        pref_soil_veg=inputs["pref_soil_veg"],
    )

    assert payload.provenance[0].endswith("sechiba.f90::sechiba_main lines 1093-1118")
    np.testing.assert_allclose(np.asarray(payload.shumdiag_perma), np.asarray(diagnostics.shumdiag_perma))
    np.testing.assert_allclose(np.asarray(payload.mc_layh), np.asarray(diagnostics.mc_layh))
    np.testing.assert_allclose(np.asarray(payload.mcl_layh), np.asarray(diagnostics.mcl_layh))
    np.testing.assert_allclose(np.asarray(payload.tmc_layh), np.asarray(diagnostics.soilmoist))

    for jv, jst_fortran in enumerate(inputs["pref_soil_veg"]):
        jst = int(jst_fortran) - 1
        np.testing.assert_allclose(
            np.asarray(payload.mc_layh_pft)[:, :, jv],
            np.asarray(diagnostics.mc_layh_s)[:, :, jst],
        )
        np.testing.assert_allclose(
            np.asarray(payload.mcl_layh_pft)[:, :, jv],
            np.asarray(diagnostics.mcl_layh_s)[:, :, jst],
        )
        np.testing.assert_allclose(
            np.asarray(payload.tmc_layh_pft)[:, :, jv],
            np.asarray(diagnostics.soilmoist),
        )

    assert np.asarray(payload.mc_layh_pft).shape == (npts, nslm, nvm)
