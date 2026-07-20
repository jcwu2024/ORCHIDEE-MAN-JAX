from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.sechiba.diffuco import (
    ControlInundateInputs,
    MIN_SECHIBA,
    aboveground_root_state,
    assemble_diffuco_first_step_precall_payload,
    assemble_pft14_local_enerbil_precall_kwargs_from_first_step_precall,
    assemble_pft14_control_inundation_first_step,
    assemble_pft14_control_salinity_first_step,
    alpha_ll_for_pft_from_run_def,
    cwrr_diaglev_from_vertical_soil_params,
    diffuco_aero_explicit,
    diffuco_bare_cwrr_beta,
    diffuco_comb_explicit,
    diffuco_comb_final_beta_bundle,
    diffuco_comb_final_beta_bundle_no_dew,
    diffuco_coupled_drag_boundary_explicit,
    diffuco_coupled_q_cdrag_pft,
    diffuco_dew_vegetation_coefficient,
    diffuco_drag_boundary_explicit,
    diffuco_flood_beta_explicit,
    diffuco_inter_explicit,
    diffuco_pft14_c3_beta_process_chain_explicit,
    diffuco_pft14_c3_chain_with_inter_explicit,
    diffuco_qsatt_explicit,
    diffuco_raerod_explicit,
    diffuco_snow_beta_explicit,
    diffuco_arrhenius,
    diffuco_arrhenius_modified,
    diffuco_apply_pft14_assimtot_controls,
    diffuco_c3_assimilation_yin_layer,
    diffuco_c3_canopy_layer_integrals,
    diffuco_pft14_c3_beta_closure_explicit,
    diffuco_pft14_after_main_payload,
    diffuco_pft14_local_enerbil_precall_explicit,
    diffuco_pft14_local_process_boundary_explicit,
    diffuco_pft14_enerbil_precall_payload,
    diffuco_lai_light_table,
    diffuco_photo_temperature_response,
    diffuco_trans_co2_c3_pft_explicit,
    diffuco_trans_co2_activity,
    diffuco_trans_co2_outputs_from_fvcb,
    diffuco_vpd_boundary_conductance,
    diffuco_control_inundation_input_coverage,
    humcste_from_pft_to_mtc,
    humcste_use_from_humcste,
    inundated_root_fraction,
    mangrove_control_salinity,
    mangrove_control_inundation,
    pft14_gpp_from_assimilation,
    pft14_trans_co2_parameter_inputs_from_run_def,
    rprof_from_humcste_use,
    read_diffuco_first_step_restart_state,
    run_pft14_local_enerbil_precall_from_first_step_precall,
    root_fraction_by_soil_layer,
    root_ventilation_fraction,
    soil_inundated_layer_count,
    z_soil_from_diaglev,
)
from jax_orchidee.driver.trace import read_static_fields_from_fixed_format_trace_dir
from jax_orchidee.sechiba.diffuco_bridge import (
    first_active_pft14_diffuco_after_main_payload,
    first_active_pft14_diffuco_trans_co2_payload,
    first_pft14_diffuco_after_main_payload,
    first_pft14_diffuco_trans_co2_payload,
    read_diffuco_trans_co2_records,
    validate_diffuco_after_main_trace_payload,
)
from jax_orchidee.driver.init import parse_run_def
from jax_orchidee.sechiba.enerbil import qsat_moisture_qsatcalc
from jax_orchidee.stomate.carbon_kernels import (
    IAGRSAPPN,
    IAGRSAPST,
    IAGRHRTPN,
    IAGRHRTST,
    ICARBON,
    NPARTS,
)


def _config_scalar(name: str) -> float:
    for line in (ROOT / "configs" / "orchidee_man_250919.yaml").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{name}:"):
            return float(stripped.split(":", maxsplit=1)[1].strip())
    raise KeyError(name)


CONTROL_SALINITY_MIN = _config_scalar("CONTROL_SALINITY_MIN")
CONTROL_INUDATE_MIN = _config_scalar("CONTROL_INUDATE_MIN")
AGB_AGR_VEN_ALL_ST = _config_scalar("AGB_AGR_VEN_ALL_ST")
AGB_AGR_VEN_ALL_PN = _config_scalar("AGB_AGR_VEN_ALL_PN")
H_AGR_MAX_ST = _config_scalar("H_AGR_MAX_ST")
H_AGR_MAX_PN = _config_scalar("H_AGR_MAX_PN")
BRIDGE_TRACE_DIR = ROOT / "outputs" / "server_1961_bridge_trace_20260624" / "traces"
USED_RUN_DEF = ROOT / "outputs" / "server_1961_trace_full_20260623" / "run" / "used_run.def"
DIFFUCO_TRACE_ROOT = ROOT / "outputs" / "server_1961_diffuco_active_after_main_trace_20260624_2244" / "traces"
DIFFUCO_USED_RUN_DEF = ROOT / "outputs" / "server_1961_diffuco_active_after_main_trace_20260624_2244" / "run" / "used_run.def"
REFERENCE_SECHIBA_START = (
    ROOT
    / "reference/case_001_071/OUT/orc_calibrate_250919_sen/"
    / "arg2_1.0/001.0-071.0/I10/S2_63.206_0.0876_0.2019_50.658/sechiba_start.nc"
)


def _used_run_def_scalar(name: str) -> float:
    for line in USED_RUN_DEF.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{name} "):
            return float(stripped.split("=", maxsplit=1)[1].strip())
    raise KeyError(name)


def _used_run_def_indexed(name: str, index: int) -> float:
    return _used_run_def_scalar(f"{name}__{index:05d}")


def _run_def_scalar(path: Path, name: str) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{name} "):
            return stripped.split("=", maxsplit=1)[1].strip()
    raise KeyError(name)


def _run_def_float(path: Path, name: str) -> float:
    return float(_run_def_scalar(path, name))


def _run_def_indexed_float(path: Path, name: str, index: int = 14) -> float:
    return _run_def_float(path, f"{name}__{index:05d}")


def _run_def_indexed_bool(path: Path, name: str, index: int = 14) -> bool:
    return _run_def_scalar(path, f"{name}__{index:05d}").upper().startswith("T")


def test_mangrove_control_salinity_uses_fortran_quadratic_and_config_floor():
    salinity = np.asarray([0.0, 30.0, 35.0], dtype=np.float64)

    control = mangrove_control_salinity(salinity, control_salinity_min=CONTROL_SALINITY_MIN)

    raw = -0.0228 * salinity**2.0 + 1.37 * salinity - 19.6
    assert np.allclose(np.asarray(control), np.maximum(raw, CONTROL_SALINITY_MIN))
    assert np.asarray(control)[0] == pytest.approx(CONTROL_SALINITY_MIN)
    assert np.asarray(control)[1] > CONTROL_SALINITY_MIN
    assert np.asarray(control)[2] == pytest.approx(CONTROL_SALINITY_MIN)


def test_first_step_control_salinity_assembly_requires_exact_salinity():
    missing = assemble_pft14_control_salinity_first_step(
        control_salinity_min=CONTROL_SALINITY_MIN,
    )

    assert not missing.ok
    assert missing.missing_inputs == ("salinity",)
    assert "control_salinity" not in missing.payload

    static_truth = read_static_fields_from_fixed_format_trace_dir(BRIDGE_TRACE_DIR)
    assembly = assemble_pft14_control_salinity_first_step(
        salinity=static_truth.salinity,
        control_salinity_min=CONTROL_SALINITY_MIN,
    )

    assert assembly.ok
    assert assembly.missing_inputs == ()
    assert set(("salinity", "control_salinity")) <= set(assembly.source_inputs)
    expected = mangrove_control_salinity(static_truth.salinity, control_salinity_min=CONTROL_SALINITY_MIN)
    np.testing.assert_allclose(np.asarray(assembly.payload["control_salinity"]), np.asarray(expected))


def test_diffuco_first_step_restart_reader_normalizes_precall_state_shapes():
    restart = read_diffuco_first_step_restart_state(REFERENCE_SECHIBA_START)

    assert restart.path.name == "sechiba_start.nc"
    assert restart.temp_sol.shape == (1,)
    assert restart.temp_sol_pft.shape == (1, 14)
    assert restart.qsurf.shape == (1,)
    assert restart.snowdz.shape == (1, 3)
    assert restart.snowrho.shape == (1, 3)
    assert restart.frac_nobio.shape == (1, 1)
    assert restart.veget.shape == (1, 14)
    assert restart.roughheight_pft.shape == (1, 14)
    assert restart.humrel.shape == (1, 14)
    assert np.allclose(restart.temp_sol, [304.63800152])
    assert np.allclose(restart.qsurf, [0.01873809])
    assert np.allclose(restart.veget[0, 13], 0.973181842386765)
    assert np.allclose(restart.veget_max[0, 13], 1.0)
    assert np.allclose(restart.z0m, [2.83563931])


def test_diffuco_first_step_precall_assembly_materializes_exact_restart_and_source_fields():
    restart = read_diffuco_first_step_restart_state(REFERENCE_SECHIBA_START)
    albedo = {"albedo": np.asarray([[0.04526854, 0.22753724]], dtype=np.float64)}
    driver = {
        "u": np.asarray([1.0]),
        "v": np.asarray([2.0]),
        "zlev": np.asarray([30.0]),
        "temp_air": np.asarray([300.0]),
        "qair": np.asarray([0.01]),
        "pb": np.asarray([1000.0]),
        "swdown": np.asarray([400.0]),
        "ccanopy": np.asarray([400.0]),
        "precip_rain": np.asarray([0.0]),
        "lalo": np.asarray([[0.0, 0.0]]),
        "neighbours": np.zeros((1, 8), dtype=np.int32),
        "resolution": np.asarray([[1.0, 1.0]]),
    }
    derivvar = {
        "qsintmax": np.zeros((1, 14), dtype=np.float64),
        "assim_param": np.ones((1, 14, 1), dtype=np.float64),
        "height": restart.height,
        "temp_growth": np.asarray([25.0]),
    }

    assembly = assemble_diffuco_first_step_precall_payload(
        driver_or_intersurf_payload=driver,
        restart_payload=restart.as_payload(),
        driver_albedo_payload=albedo,
        slowproc_derivvar_payload=derivvar,
        ok_explicitsnow=True,
        river_routing=True,
        nbp_glo=1,
    )

    assert not assembly.ok
    assert "temp_sol" in assembly.restart_inputs
    assert "roughheight_pft" in assembly.restart_inputs
    assert "u" in assembly.driver_inputs
    assert "rau" in assembly.source_kernel_inputs
    assert "swnet" in assembly.source_kernel_inputs
    assert "totfrac_nobio" in assembly.source_kernel_inputs
    assert "tot_bare_soil" in assembly.source_kernel_inputs
    assert "frac_snow_veg" in assembly.source_kernel_inputs
    assert "frac_snow_nobio" in assembly.source_kernel_inputs
    assert "flood_frac" in assembly.source_kernel_inputs
    assert "flood_res" in assembly.source_kernel_inputs
    assert "qsintmax" in assembly.source_kernel_inputs
    assert "temp_sol" not in assembly.missing_inputs
    assert "control_salinity" in assembly.missing_local_process_inputs
    assert "frac_snow_veg" not in assembly.missing_local_process_inputs
    assert "frac_snow_nobio" not in assembly.missing_local_process_inputs
    assert "flood_frac" not in assembly.missing_local_process_inputs
    assert "flood_res" not in assembly.missing_local_process_inputs
    np.testing.assert_allclose(np.asarray(assembly.payload["frac_snow_veg"]), [0.0])
    np.testing.assert_allclose(np.asarray(assembly.payload["frac_snow_nobio"]), [[0.0]])
    np.testing.assert_allclose(np.asarray(assembly.payload["flood_frac"]), [0.0])
    np.testing.assert_allclose(np.asarray(assembly.payload["flood_res"]), [0.0])
    np.testing.assert_allclose(np.asarray(assembly.payload["totfrac_nobio"]), [0.0])
    np.testing.assert_allclose(
        np.asarray(assembly.payload["tot_bare_soil"]),
        restart.veget_max[:, 0] + np.sum(restart.veget_max[:, 1:] - restart.veget[:, 1:], axis=1),
    )


def test_pft14_trans_co2_parameter_inputs_derive_alpha_ll_through_mtc_mapping():
    values = parse_run_def(USED_RUN_DEF)

    assert alpha_ll_for_pft_from_run_def(values, pft_fortran_index=14) == pytest.approx(0.3)
    params = pft14_trans_co2_parameter_inputs_from_run_def(values)

    assert params["alpha_ll"] == pytest.approx(0.3)
    assert params["rveg_pft"] == pytest.approx(_run_def_indexed_float(USED_RUN_DEF, "RVEG_PFT"))
    assert params["rstruct_const"] == pytest.approx(_run_def_indexed_float(USED_RUN_DEF, "RSTRUCT_CONST"))
    assert params["ok_laidev"] == _run_def_indexed_bool(USED_RUN_DEF, "OK_LAIDEV")


def _full_trace_first_step_diffuco_precall():
    from jax_orchidee.driver.orchestration import paper_1961_driver_timestep_scaffold

    static_truth = read_static_fields_from_fixed_format_trace_dir(BRIDGE_TRACE_DIR)
    scaffold = paper_1961_driver_timestep_scaffold(
        ROOT / "configs" / "orchidee_man_250919.yaml",
        year=1961,
        tstep=0,
        static_trace_fields=static_truth,
        used_run_def_path=USED_RUN_DEF,
        root=ROOT,
    )
    assert scaffold.diffuco_first_step_precall is not None
    assert scaffold.diffuco_first_step_precall.ok
    return scaffold.diffuco_first_step_precall


def test_pft14_local_enerbil_precall_kwargs_from_first_step_precall_are_source_backed():
    precall = _full_trace_first_step_diffuco_precall()
    values = parse_run_def(USED_RUN_DEF)

    kwargs = assemble_pft14_local_enerbil_precall_kwargs_from_first_step_precall(
        precall,
        run_def_values=values,
    )

    assert kwargs["pft_index"] == 13
    assert kwargs["ldq_cdrag_from_gcm"] is False
    assert kwargs["dt_sechiba"] == pytest.approx(_run_def_float(USED_RUN_DEF, "DT_SECHIBA"))
    assert kwargs["min_wind"] == pytest.approx(_run_def_float(USED_RUN_DEF, "MIN_WIND"))
    assert kwargs["rough_dyn"] is True
    assert kwargs["ok_snowfact"] is True
    assert kwargs["trans_co2_inputs"]["alpha_ll"] == pytest.approx(0.3)
    np.testing.assert_allclose(
        kwargs["trans_co2_inputs"]["vcmax"],
        np.asarray(precall.payload["assim_param"])[:, 13, 0],
    )
    assert kwargs["pft_output_backgrounds"]["gpp"].shape == (1, 14)
    assert kwargs["pft_output_backgrounds"]["rveget"][0, 0] == pytest.approx(1.0e20)
    assert "q_cdrag" not in kwargs["passthrough"]


def test_pft14_local_enerbil_precall_runs_from_first_step_precall_without_after_main_trace():
    precall = _full_trace_first_step_diffuco_precall()
    values = parse_run_def(USED_RUN_DEF)

    assembly = run_pft14_local_enerbil_precall_from_first_step_precall(
        precall,
        run_def_values=values,
    )

    payload = assembly.result.payload.payload
    for name in (
        "q_cdrag",
        "q_cdrag_pft",
        "raero",
        "qsatt",
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
        "gpp",
        "gsmean",
        "rveget",
        "rstruct",
        "cimean",
    ):
        assert name in payload
    assert set(("q_cdrag", "q_cdrag_pft", "raero", "qsatt")) <= set(assembly.result.payload.passthrough_fields)
    np.testing.assert_allclose(np.asarray(payload["q_cdrag"]), np.asarray(assembly.result.boundary.drag.q_cdrag))
    np.testing.assert_allclose(np.asarray(payload["vbeta"]), np.asarray(assembly.result.boundary.process_chain.closure.comb.vbeta))
    assert np.asarray(payload["vbeta"]).shape == (1,)
    assert np.asarray(payload["vbeta_pft"]).shape == (1, 14)
    assert np.asarray(payload["gpp"]).shape == (1, 14)


def test_root_fraction_by_soil_layer_matches_fortran_exponential_profile():
    z_soil = np.asarray([0.0, 0.5, 1.5, 3.0], dtype=np.float64)
    rprof = np.asarray([1.0, 2.0], dtype=np.float64)

    frac = np.asarray(root_fraction_by_soil_layer(z_soil, rprof))

    rpc = 1.0 / (1.0 - np.exp(-z_soil[-1] / rprof))
    expected = rpc[:, None] * (
        np.exp(-z_soil[:-1] / rprof[:, None]) - np.exp(-z_soil[1:] / rprof[:, None])
    )
    assert np.allclose(frac, expected)
    assert np.allclose(frac.sum(axis=-1), np.ones(2))


def test_z_soil_from_diaglev_is_only_direct_fortran_boundary_construction():
    diaglev = np.asarray([0.25, 1.0, 2.0], dtype=np.float64)

    z_soil = np.asarray(z_soil_from_diaglev(diaglev))

    assert np.allclose(z_soil, [0.0, 0.25, 1.0, 2.0])


def test_cwrr_diaglev_from_vertical_soil_params_closes_paper_case_z_soil():
    diaglev = np.asarray(
        cwrr_diaglev_from_vertical_soil_params(
            depth_max_h=_used_run_def_scalar("DEPTH_MAX_H"),
            depth_max_t=_used_run_def_scalar("DEPTH_MAX_T"),
            depth_topthickness=_used_run_def_scalar("DEPTH_TOPTHICK"),
            depth_cstthickness=_used_run_def_scalar("DEPTH_CSTTHICK"),
            depth_geom=_used_run_def_scalar("DEPTH_GEOM"),
            ratio_geom_below=_used_run_def_scalar("RATIO_GEOM_BELOW"),
        )
    )
    z_soil = np.asarray(z_soil_from_diaglev(diaglev))

    expected_diaglev = np.asarray(
        [
            0.0004887585535,
            0.001955034214,
            0.005865102642,
            0.013685239498,
            0.02932551321,
            0.060606060634,
            0.123167155484,
            0.24828934518,
            0.498533724572,
            0.999022482356,
            1.749755620589,
        ],
        dtype=np.float64,
    )

    assert diaglev.shape == (11,)
    assert np.allclose(diaglev, expected_diaglev)
    assert np.allclose(z_soil[0], 0.0)
    assert np.allclose(z_soil[1:], expected_diaglev)


def test_rprof_from_humcste_use_requires_explicit_hydrol_state():
    humcste_use = np.asarray([[2.0, 4.0]], dtype=np.float64)

    rprof = np.asarray(rprof_from_humcste_use(humcste_use))

    assert np.allclose(rprof, [[0.5, 0.25]])


def test_paper_case_humcste_use_and_rprof_are_constructible_from_local_used_run_def():
    pft_to_mtc = np.arange(1, 15, dtype=np.int32)
    pft_to_mtc[13] = int(_used_run_def_indexed("PFT_TO_MTC", 14))
    humcste_from_table = np.asarray(
        humcste_from_pft_to_mtc(
            pft_to_mtc,
            zmaxh=_used_run_def_scalar("DEPTH_MAX_H"),
        )
    )
    humcste_from_used_run = np.asarray([_used_run_def_indexed("HYDROL_HUMCSTE", idx) for idx in range(1, 15)])
    humcste_use = np.asarray(humcste_use_from_humcste(humcste_from_table, npts=1))
    rprof = np.asarray(rprof_from_humcste_use(humcste_use))

    assert np.allclose(humcste_from_table, humcste_from_used_run)
    assert humcste_use.shape == (1, 14)
    assert humcste_use[0, 13] == pytest.approx(0.8)
    assert rprof[0, 13] == pytest.approx(1.25)


def test_soil_inundated_layer_count_matches_fortran_count_and_clip():
    z_soil = np.asarray([0.0, 0.5, 1.5, 3.0], dtype=np.float64)
    tide_height = np.asarray([[-0.1, -0.6, -2.0, -4.0, 0.2]], dtype=np.float64)

    n_soil_inudate = np.asarray(soil_inundated_layer_count(tide_height, z_soil, nslm=3))

    assert n_soil_inudate.tolist() == [[1, 2, 2, 2, 0]]


def test_inundated_root_fraction_uses_fortran_one_based_tail_slice_semantics():
    frac_root_soil = np.asarray([[0.2, 0.3, 0.5]], dtype=np.float64)
    n_soil_inudate = np.asarray([[0, 1, 2]], dtype=np.int64)

    frac_root_inudate = np.asarray(inundated_root_fraction(frac_root_soil, n_soil_inudate))

    assert np.allclose(frac_root_inudate, [[1.0, 0.8, 0.5]])


def test_aboveground_root_state_and_ventilation_follow_fortran_formulas():
    biomass = np.zeros((1, 14, NPARTS, 1), dtype=np.float64)
    biomass[0, 13, IAGRSAPST, ICARBON] = 60.0
    biomass[0, 13, IAGRHRTST, ICARBON] = 40.0
    biomass[0, 13, IAGRSAPPN, ICARBON] = 10.0
    biomass[0, 13, IAGRHRTPN, ICARBON] = 10.0
    tide_height = np.asarray([[-1.0, 0.05, 0.5]], dtype=np.float64)

    agr = aboveground_root_state(
        biomass,
        h_agr_max_st=H_AGR_MAX_ST,
        h_agr_max_pn=H_AGR_MAX_PN,
    )
    ventilation = root_ventilation_fraction(
        tide_height,
        agr,
        agb_agr_ven_all_st=AGB_AGR_VEN_ALL_ST,
        agb_agr_ven_all_pn=AGB_AGR_VEN_ALL_PN,
    )

    agb_st = 100.0
    agb_pn = 20.0
    h_st = min((10.0 ** ((np.log10(agb_st * 0.02 * 1000.0 / 1000.0) + 2.945) / 2.546) / 3.1415 / 100.0) * 7.56 + 0.5, H_AGR_MAX_ST)
    h_pn = min(agb_pn * 0.02 * 0.085, H_AGR_MAX_PN)
    tide_pos = np.maximum(tide_height, 0.0)
    ven_st = min(agb_st / AGB_AGR_VEN_ALL_ST, 1.0) * np.maximum(h_st - tide_pos, 0.0) / h_st
    ven_pn = min(agb_pn / AGB_AGR_VEN_ALL_PN, 1.0) * np.maximum(h_pn - tide_pos, 0.0) / h_pn

    assert np.asarray(agr.agb_agr_st)[0] == pytest.approx(agb_st)
    assert np.asarray(agr.agb_agr_pn)[0] == pytest.approx(agb_pn)
    assert np.asarray(agr.h_agr_st)[0] == pytest.approx(h_st)
    assert np.asarray(agr.h_agr_pn)[0] == pytest.approx(h_pn)
    assert np.allclose(np.asarray(ventilation.frac_root_ven_st), ven_st)
    assert np.allclose(np.asarray(ventilation.frac_root_ven_pn), ven_pn)
    assert np.allclose(np.asarray(ventilation.frac_root_ventilate), np.minimum(0.5 * ven_st + 0.5 * ven_pn, 1.0))


def test_mangrove_control_inundation_combines_root_anoxia_time_mean_and_floor_factor():
    z_soil = np.asarray([0.0, 0.5, 1.5, 3.0], dtype=np.float64)
    rprof = np.ones((1, 14), dtype=np.float64)
    tide_height = np.asarray([[-0.1, -0.6, 0.2]], dtype=np.float64)
    biomass = np.zeros((1, 14, NPARTS, 1), dtype=np.float64)
    biomass[0, 13, IAGRSAPST, ICARBON] = 60.0
    biomass[0, 13, IAGRHRTST, ICARBON] = 40.0
    biomass[0, 13, IAGRSAPPN, ICARBON] = 10.0
    biomass[0, 13, IAGRHRTPN, ICARBON] = 10.0

    result = mangrove_control_inundation(
        ControlInundateInputs(
            z_soil=z_soil,
            rprof=rprof,
            tide_height=tide_height,
            biomass=biomass,
        ),
        control_inudate_min=CONTROL_INUDATE_MIN,
        agb_agr_ven_all_st=AGB_AGR_VEN_ALL_ST,
        agb_agr_ven_all_pn=AGB_AGR_VEN_ALL_PN,
        h_agr_max_st=H_AGR_MAX_ST,
        h_agr_max_pn=H_AGR_MAX_PN,
    )

    frac_root_soil = np.asarray(root_fraction_by_soil_layer(z_soil, np.asarray([1.0])))
    n_soil_inudate = np.asarray([[1, 2, 0]])
    frac_root_inudate = np.asarray(inundated_root_fraction(frac_root_soil, n_soil_inudate))
    agr = aboveground_root_state(
        biomass,
        h_agr_max_st=H_AGR_MAX_ST,
        h_agr_max_pn=H_AGR_MAX_PN,
    )
    ventilation = root_ventilation_fraction(
        tide_height,
        agr,
        agb_agr_ven_all_st=AGB_AGR_VEN_ALL_ST,
        agb_agr_ven_all_pn=AGB_AGR_VEN_ALL_PN,
    )
    frac_root_anoxia = np.maximum(frac_root_inudate - np.asarray(ventilation.frac_root_ventilate), 0.0)
    expected_control = 1.0 - (1.0 - CONTROL_INUDATE_MIN) * frac_root_anoxia.mean(axis=-1)

    assert np.allclose(np.asarray(result.n_soil_inudate), n_soil_inudate)
    assert np.allclose(np.asarray(result.frac_root_inudate), frac_root_inudate)
    assert np.allclose(np.asarray(result.frac_root_anoxia), frac_root_anoxia)
    assert np.allclose(np.asarray(result.control_inudate), expected_control)
    assert CONTROL_INUDATE_MIN <= np.asarray(result.control_inudate)[0] <= 1.0


def test_control_inundation_input_coverage_reports_trace_covered_and_missing_inputs():
    static_truth = read_static_fields_from_fixed_format_trace_dir(BRIDGE_TRACE_DIR)
    assert static_truth.tide_height.shape == (1, 584)

    run_def_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            ROOT / "fortran_run_scripts" / "paper_250919" / "run.def.vn",
            ROOT
            / "fortran_run_scripts"
            / "paper_250919"
            / "sen_reference_arg2_1.0_001.0-071.0"
            / "I10"
            / "S2_63.206_0.0876_0.2019_50.658"
            / "run.def_63.206_0.0876_0.2019_50.658",
        )
    ).upper()
    assert "HYDROL_HUMCSTE" not in run_def_text

    coverage = diffuco_control_inundation_input_coverage(
        full_tide_height_available=static_truth.tide_height is not None,
        after_diffuco_tide_height_1_available=True,
        restart_biomass_available=True,
    )

    assert not coverage.complete
    assert coverage.covered_inputs == ("tide_height",)
    assert coverage.missing_inputs == ("z_soil", "rprof", "biomass")
    assert coverage.tide_height.status == "covered"
    assert "Full tide_height" in coverage.tide_height.detail
    assert coverage.z_soil.status == "missing"
    assert "diaglev" in coverage.z_soil.detail
    assert coverage.rprof.status == "missing"
    assert "humcste/PFT_TO_MTC" in coverage.rprof.detail
    assert coverage.biomass.status == "missing"
    assert "does not identify" in coverage.biomass.detail


def test_control_inundation_input_coverage_reports_local_source_closed_z_soil_and_rprof():
    static_truth = read_static_fields_from_fixed_format_trace_dir(BRIDGE_TRACE_DIR)

    coverage = diffuco_control_inundation_input_coverage(
        diaglev_source_available=True,
        humcste_source_available=True,
        full_tide_height_available=static_truth.tide_height is not None,
        restart_biomass_available=True,
    )

    assert not coverage.complete
    assert coverage.covered_inputs == ("z_soil", "rprof", "tide_height")
    assert coverage.missing_inputs == ("biomass",)
    assert coverage.z_soil.status == "constructible"
    assert "znt(1:nslm)" in coverage.z_soil.detail
    assert coverage.rprof.status == "constructible"
    assert "PFT_TO_MTC/HYDROL_HUMCSTE" in coverage.rprof.detail
    assert coverage.biomass.status == "missing"


def test_control_inundation_input_coverage_distinguishes_restart_output_from_start_boundary():
    coverage = diffuco_control_inundation_input_coverage(
        diaglev_source_available=True,
        humcste_source_available=True,
        full_tide_height_available=True,
        stomate_restart_output_biomass_available=True,
    )

    assert not coverage.complete
    assert coverage.covered_inputs == ("z_soil", "rprof", "tide_height")
    assert coverage.missing_inputs == ("biomass",)
    assert coverage.biomass.status == "missing"
    assert "end-of-run saved state" in coverage.biomass.detail
    assert "STOMATE_RESTART_FILEIN" in coverage.biomass.detail


def test_control_inundation_input_coverage_covers_first_step_cold_start_zero_biomass():
    coverage = diffuco_control_inundation_input_coverage(
        diaglev_source_available=True,
        humcste_source_available=True,
        full_tide_height_available=True,
        first_step_cold_start_biomass_zeroed=True,
    )

    assert coverage.complete
    assert coverage.covered_inputs == ("z_soil", "rprof", "tide_height", "biomass")
    assert coverage.missing_inputs == ()
    assert coverage.biomass.status == "covered"
    assert "cold start" in coverage.biomass.detail
    assert "zero" in coverage.biomass.detail


def test_control_inundation_input_coverage_covers_first_step_stomate_start_file_only():
    coverage = diffuco_control_inundation_input_coverage(
        diaglev_source_available=True,
        humcste_source_available=True,
        full_tide_height_available=True,
        first_step_stomate_start_biomass_available=True,
        restart_biomass_available=True,
    )

    assert coverage.complete
    assert coverage.covered_inputs == ("z_soil", "rprof", "tide_height", "biomass")
    assert coverage.missing_inputs == ()
    assert coverage.biomass.status == "covered"
    assert "STOMATE_RESTART_FILEIN" in coverage.biomass.detail
    assert "before sechiba_main calls diffuco_main" in coverage.biomass.detail


def test_first_step_control_inundation_assembly_materializes_z_soil_rprof_and_requires_full_tide():
    pft_to_mtc = np.arange(1, 15, dtype=np.int32)
    pft_to_mtc[13] = int(_used_run_def_indexed("PFT_TO_MTC", 14))
    hydrol_humcste = np.asarray([_used_run_def_indexed("HYDROL_HUMCSTE", idx) for idx in range(1, 15)])
    biomass = np.zeros((1, 14, NPARTS, 1), dtype=np.float64)

    assembly = assemble_pft14_control_inundation_first_step(
        depth_max_h=_used_run_def_scalar("DEPTH_MAX_H"),
        depth_max_t=_used_run_def_scalar("DEPTH_MAX_T"),
        depth_topthickness=_used_run_def_scalar("DEPTH_TOPTHICK"),
        depth_cstthickness=_used_run_def_scalar("DEPTH_CSTTHICK"),
        depth_geom=_used_run_def_scalar("DEPTH_GEOM"),
        ratio_geom_below=_used_run_def_scalar("RATIO_GEOM_BELOW"),
        pft_to_mtc=pft_to_mtc,
        hydrol_humcste=hydrol_humcste,
        biomass=biomass,
        control_inudate_min=CONTROL_INUDATE_MIN,
        agb_agr_ven_all_st=AGB_AGR_VEN_ALL_ST,
        agb_agr_ven_all_pn=AGB_AGR_VEN_ALL_PN,
        h_agr_max_st=H_AGR_MAX_ST,
        h_agr_max_pn=H_AGR_MAX_PN,
    )

    assert not assembly.ok
    assert assembly.inputs is None
    assert assembly.result is None
    assert assembly.missing_inputs == ("tide_height",)
    assert set(("z_soil", "humcste", "humcste_use", "rprof", "biomass")) <= set(assembly.source_inputs)
    assert np.asarray(assembly.payload["z_soil"]).shape == (12,)
    assert np.asarray(assembly.payload["rprof"]).shape == (1, 14)
    assert np.asarray(assembly.payload["rprof"])[0, 13] == pytest.approx(1.25)


def test_first_step_control_inundation_assembly_runs_when_full_tide_and_biomass_are_exact():
    static_truth = read_static_fields_from_fixed_format_trace_dir(BRIDGE_TRACE_DIR)
    pft_to_mtc = np.arange(1, 15, dtype=np.int32)
    pft_to_mtc[13] = int(_used_run_def_indexed("PFT_TO_MTC", 14))
    hydrol_humcste = np.asarray([_used_run_def_indexed("HYDROL_HUMCSTE", idx) for idx in range(1, 15)])
    biomass = np.zeros((1, 14, NPARTS, 1), dtype=np.float64)
    biomass[0, 13, IAGRSAPST, ICARBON] = 60.0
    biomass[0, 13, IAGRHRTST, ICARBON] = 40.0
    biomass[0, 13, IAGRSAPPN, ICARBON] = 10.0
    biomass[0, 13, IAGRHRTPN, ICARBON] = 10.0

    assembly = assemble_pft14_control_inundation_first_step(
        depth_max_h=_used_run_def_scalar("DEPTH_MAX_H"),
        depth_max_t=_used_run_def_scalar("DEPTH_MAX_T"),
        depth_topthickness=_used_run_def_scalar("DEPTH_TOPTHICK"),
        depth_cstthickness=_used_run_def_scalar("DEPTH_CSTTHICK"),
        depth_geom=_used_run_def_scalar("DEPTH_GEOM"),
        ratio_geom_below=_used_run_def_scalar("RATIO_GEOM_BELOW"),
        pft_to_mtc=pft_to_mtc,
        hydrol_humcste=hydrol_humcste,
        tide_height=static_truth.tide_height,
        biomass=biomass,
        control_inudate_min=CONTROL_INUDATE_MIN,
        agb_agr_ven_all_st=AGB_AGR_VEN_ALL_ST,
        agb_agr_ven_all_pn=AGB_AGR_VEN_ALL_PN,
        h_agr_max_st=H_AGR_MAX_ST,
        h_agr_max_pn=H_AGR_MAX_PN,
    )

    assert assembly.ok
    assert assembly.inputs is not None
    assert assembly.result is not None
    assert assembly.missing_inputs == ()
    assert "control_inudate" in assembly.payload

    direct = mangrove_control_inundation(
        assembly.inputs,
        control_inudate_min=CONTROL_INUDATE_MIN,
        agb_agr_ven_all_st=AGB_AGR_VEN_ALL_ST,
        agb_agr_ven_all_pn=AGB_AGR_VEN_ALL_PN,
        h_agr_max_st=H_AGR_MAX_ST,
        h_agr_max_pn=H_AGR_MAX_PN,
    )
    np.testing.assert_allclose(
        np.asarray(assembly.result.control_inudate),
        np.asarray(direct.control_inudate),
    )
    assert np.asarray(assembly.result.control_inudate).shape == (1,)


def test_first_step_control_inundation_assembly_infers_multiland_npts_from_inputs():
    pft_to_mtc = np.arange(1, 15, dtype=np.int32)
    pft_to_mtc[13] = int(_used_run_def_indexed("PFT_TO_MTC", 14))
    hydrol_humcste = np.asarray([_used_run_def_indexed("HYDROL_HUMCSTE", idx) for idx in range(1, 15)])
    tide_height = np.asarray(
        [
            [-0.2, 0.0, 0.3],
            [0.1, 0.4, 0.8],
        ],
        dtype=np.float64,
    )
    biomass = np.zeros((2, 14, NPARTS, 1), dtype=np.float64)
    biomass[:, 13, IAGRSAPST, ICARBON] = [60.0, 30.0]
    biomass[:, 13, IAGRHRTST, ICARBON] = [40.0, 20.0]
    biomass[:, 13, IAGRSAPPN, ICARBON] = [10.0, 5.0]
    biomass[:, 13, IAGRHRTPN, ICARBON] = [10.0, 5.0]

    assembly = assemble_pft14_control_inundation_first_step(
        depth_max_h=_used_run_def_scalar("DEPTH_MAX_H"),
        depth_max_t=_used_run_def_scalar("DEPTH_MAX_T"),
        depth_topthickness=_used_run_def_scalar("DEPTH_TOPTHICK"),
        depth_cstthickness=_used_run_def_scalar("DEPTH_CSTTHICK"),
        depth_geom=_used_run_def_scalar("DEPTH_GEOM"),
        ratio_geom_below=_used_run_def_scalar("RATIO_GEOM_BELOW"),
        pft_to_mtc=pft_to_mtc,
        hydrol_humcste=hydrol_humcste,
        tide_height=tide_height,
        biomass=biomass,
        control_inudate_min=CONTROL_INUDATE_MIN,
        agb_agr_ven_all_st=AGB_AGR_VEN_ALL_ST,
        agb_agr_ven_all_pn=AGB_AGR_VEN_ALL_PN,
        h_agr_max_st=H_AGR_MAX_ST,
        h_agr_max_pn=H_AGR_MAX_PN,
    )

    assert assembly.ok
    np.testing.assert_allclose(np.asarray(assembly.payload["rprof"][:, 13]), np.asarray([1.25, 1.25]))
    assert np.asarray(assembly.result.control_inudate).shape == (2,)
    assert np.all(np.asarray(assembly.result.control_inudate) >= CONTROL_INUDATE_MIN)


def test_first_step_control_inundation_assembly_rejects_mismatched_land_dimensions():
    pft_to_mtc = np.arange(1, 15, dtype=np.int32)
    hydrol_humcste = np.asarray([_used_run_def_indexed("HYDROL_HUMCSTE", idx) for idx in range(1, 15)])
    tide_height = np.zeros((2, 3), dtype=np.float64)
    biomass = np.zeros((1, 14, NPARTS, 1), dtype=np.float64)

    with pytest.raises(ValueError, match="land-point dimension"):
        assemble_pft14_control_inundation_first_step(
            depth_max_h=_used_run_def_scalar("DEPTH_MAX_H"),
            depth_max_t=_used_run_def_scalar("DEPTH_MAX_T"),
            depth_topthickness=_used_run_def_scalar("DEPTH_TOPTHICK"),
            depth_cstthickness=_used_run_def_scalar("DEPTH_CSTTHICK"),
            depth_geom=_used_run_def_scalar("DEPTH_GEOM"),
            ratio_geom_below=_used_run_def_scalar("RATIO_GEOM_BELOW"),
            pft_to_mtc=pft_to_mtc,
            hydrol_humcste=hydrol_humcste,
            tide_height=tide_height,
            biomass=biomass,
            control_inudate_min=CONTROL_INUDATE_MIN,
            agb_agr_ven_all_st=AGB_AGR_VEN_ALL_ST,
            agb_agr_ven_all_pn=AGB_AGR_VEN_ALL_PN,
            h_agr_max_st=H_AGR_MAX_ST,
            h_agr_max_pn=H_AGR_MAX_PN,
        )


def test_control_inundation_input_coverage_marks_constructible_only_with_explicit_upstream_state():
    coverage = diffuco_control_inundation_input_coverage(
        diaglev_available=True,
        humcste_use_available=True,
        full_tide_height_available=True,
        diffuco_call_biomass_available=True,
    )

    assert coverage.complete
    assert coverage.covered_inputs == ("z_soil", "rprof", "tide_height", "biomass")
    assert coverage.z_soil.status == "constructible"
    assert coverage.rprof.status == "constructible"


def test_pft14_gpp_applies_salinity_and_inundation_controls_before_conversion():
    assimtot = np.asarray([2.0, 4.0], dtype=np.float64)
    veget_max = np.asarray([1.0, 0.25], dtype=np.float64)
    dt_sechiba = 1800.0
    control_salinity = np.asarray([0.5, 0.75], dtype=np.float64)
    control_inudate = np.asarray([CONTROL_INUDATE_MIN, 1.0], dtype=np.float64)

    gpp = pft14_gpp_from_assimilation(
        assimtot,
        veget_max=veget_max,
        dt_sechiba=dt_sechiba,
        control_salinity=control_salinity,
        control_inudate=control_inudate,
    )

    expected = assimtot * control_salinity * control_inudate * 12.0e-6 * veget_max * dt_sechiba
    assert np.allclose(np.asarray(gpp), expected)


def test_pft14_assimtot_controls_feed_cimean_not_only_gpp():
    assimtot = np.asarray([[10.0]], dtype=np.float64)
    controlled = diffuco_apply_pft14_assimtot_controls(
        assimtot,
        control_salinity=np.asarray([[0.5]], dtype=np.float64),
        control_inudate=np.asarray([[0.8]], dtype=np.float64),
        jv=14,
    )

    result = diffuco_trans_co2_outputs_from_fvcb(
        assimilate=np.asarray([[True]], dtype=bool),
        assimtot=controlled,
        rdtot=np.asarray([[1.0]], dtype=np.float64),
        gstot=np.asarray([[2.0]], dtype=np.float64),
        leaf_gs_top=np.asarray([[0.4]], dtype=np.float64),
        gamma_star=np.asarray([[40.0]], dtype=np.float64),
        fvpd=np.asarray([[0.6]], dtype=np.float64),
        g0var=np.asarray([[0.01]], dtype=np.float64),
        laisum=np.asarray([[1.5]], dtype=np.float64),
        ilai=np.asarray([[1]], dtype=np.int32),
        laitab=np.asarray([0.0, 0.5], dtype=np.float64),
        veget_max=np.asarray([[0.7]], dtype=np.float64),
        humrel=np.asarray([[0.8]], dtype=np.float64),
        zqsvegrap=np.asarray([[0.2]], dtype=np.float64),
        vbeta23=np.asarray([[0.05]], dtype=np.float64),
        t2m=np.asarray([300.0], dtype=np.float64),
        pb=np.asarray([1000.0], dtype=np.float64),
        wind=np.asarray([2.0], dtype=np.float64),
        q_cdrag=np.asarray([0.01], dtype=np.float64),
        q_cdrag_pft=np.asarray([[0.01]], dtype=np.float64),
        rveg_pft=np.asarray([1.0], dtype=np.float64),
        ca=np.asarray([410.0], dtype=np.float64),
        rstruct_const=np.asarray([25.0], dtype=np.float64),
        ok_laidev=np.asarray([False]),
        dt_sechiba=1800.0,
    )

    expected_assimtot = assimtot * 0.5 * 0.8
    expected_cimean = 0.6 * (expected_assimtot + 1.0) / (2.0 - 0.01 * 1.5) + 40.0
    expected_gpp = expected_assimtot * 12.0e-6 * 0.7 * 1800.0
    np.testing.assert_allclose(np.asarray(controlled), expected_assimtot)
    np.testing.assert_allclose(np.asarray(result.cimean), expected_cimean)
    np.testing.assert_allclose(np.asarray(result.gpp), expected_gpp)


def test_pft14_gpp_helper_leaves_non_pft14_assimilation_uncontrolled():
    gpp = pft14_gpp_from_assimilation(
        np.asarray([2.0], dtype=np.float64),
        veget_max=np.asarray([1.0], dtype=np.float64),
        dt_sechiba=1800.0,
        control_salinity=np.asarray([0.5], dtype=np.float64),
        control_inudate=np.asarray([0.5], dtype=np.float64),
        jv=13,
    )

    assert np.asarray(gpp)[0] == pytest.approx(2.0 * 12.0e-6 * 1800.0)


def test_diffuco_trans_co2_activity_matches_source_masks_and_water_inputs():
    result = diffuco_trans_co2_activity(
        swdown=np.asarray([100.0, 0.0], dtype=np.float64),
        humrel=np.asarray([[0.5, 0.6, 0.7], [0.5, 0.0, 0.7]], dtype=np.float64),
        veget=np.asarray([[0.2, 0.3, 0.4], [0.2, 0.3, 0.4]], dtype=np.float64),
        veget_max=np.asarray([[0.2, 0.3, 0.4], [0.2, 0.3, 0.0]], dtype=np.float64),
        lai=np.asarray([[0.005, 0.005, 0.02], [0.02, 0.02, 0.02]], dtype=np.float64),
        qsintveg=np.asarray([[0.0, 0.2, -0.1], [0.1, 0.2, 0.3]], dtype=np.float64),
        qsintmax=np.asarray([[0.0, 0.4, 0.5], [0.2, 0.0, 0.6]], dtype=np.float64),
        temp_growth=np.asarray([25.0, 25.0], dtype=np.float64),
        tphoto_min=np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        tphoto_max=np.asarray([40.0, 40.0, 40.0], dtype=np.float64),
        ok_laidev=np.asarray([False, True, False]),
    )

    expected_lai_active = np.asarray(
        [
            [False, True, True],
            [True, True, False],
        ]
    )
    expected_assimilate = np.asarray(
        [
            [False, True, True],
            [False, False, False],
        ]
    )
    expected_zqsvegrap = np.asarray(
        [
            [0.0, 0.5, 0.0],
            [0.5, 0.0, 0.5],
        ],
        dtype=np.float64,
    )

    np.testing.assert_array_equal(np.asarray(result.lai_active), expected_lai_active)
    np.testing.assert_array_equal(np.asarray(result.assimilate), expected_assimilate)
    np.testing.assert_allclose(np.asarray(result.zqsvegrap), expected_zqsvegrap)
    np.testing.assert_allclose(
        np.asarray(result.water_lim),
        np.asarray([[0.5, 0.6, 0.7], [0.5, 0.0, 0.7]], dtype=np.float64),
    )


def test_diffuco_trans_co2_activity_requires_explicit_pft_parameter_vectors():
    with pytest.raises(ValueError, match="ok_laidev"):
        diffuco_trans_co2_activity(
            swdown=np.asarray([100.0], dtype=np.float64),
            humrel=np.ones((1, 2), dtype=np.float64),
            veget=np.ones((1, 2), dtype=np.float64),
            veget_max=np.ones((1, 2), dtype=np.float64),
            lai=np.ones((1, 2), dtype=np.float64),
            qsintveg=np.zeros((1, 2), dtype=np.float64),
            qsintmax=np.ones((1, 2), dtype=np.float64),
            temp_growth=np.asarray([25.0], dtype=np.float64),
            tphoto_min=np.zeros(2, dtype=np.float64),
            tphoto_max=np.ones(2, dtype=np.float64) * 40.0,
            ok_laidev=np.asarray([False]),
        )


def test_diffuco_lai_light_table_matches_fortran_discretization():
    ext_coeff = np.asarray([0.5, 0.7], dtype=np.float64)
    result = diffuco_lai_light_table(ext_coeff)

    nlai = 20
    laimax = 12.0
    depth = 0.15
    jl = np.arange(nlai + 1, dtype=np.float64)
    expected_laitab = laimax * (np.exp(depth * jl) - 1.0) / (np.exp(depth * nlai) - 1.0)
    expected_light = np.exp(-ext_coeff[:, None] * expected_laitab[:-1][None, :])

    assert np.asarray(result.laitab).shape == (21,)
    assert np.asarray(result.light).shape == (2, 20)
    np.testing.assert_allclose(np.asarray(result.laitab), expected_laitab)
    np.testing.assert_allclose(np.asarray(result.light), expected_light)
    assert np.asarray(result.laitab)[0] == pytest.approx(0.0)
    assert np.asarray(result.laitab)[-1] == pytest.approx(12.0)


def test_diffuco_lai_light_table_allows_explicit_runtime_constants():
    result = diffuco_lai_light_table(
        np.asarray([0.2], dtype=np.float64),
        nlai=3,
        laimax=6.0,
        lai_level_depth=0.3,
    )
    jl = np.arange(4, dtype=np.float64)
    expected_laitab = 6.0 * (np.exp(0.3 * jl) - 1.0) / (np.exp(0.3 * 3.0) - 1.0)

    np.testing.assert_allclose(np.asarray(result.laitab), expected_laitab)
    np.testing.assert_allclose(np.asarray(result.light), np.exp(-0.2 * expected_laitab[:-1])[None, :])


def test_diffuco_vpd_boundary_conductance_matches_source_formulas():
    qsurf = np.asarray([0.010, 0.008], dtype=np.float64)
    qsatt = np.asarray([0.014, 0.012], dtype=np.float64)
    t2m = np.asarray([300.0, 285.0], dtype=np.float64)
    pb = np.asarray([1000.0, 990.0], dtype=np.float64)
    water_lim = np.asarray([[0.5, 0.8], [0.2, 0.9]], dtype=np.float64)
    a1 = np.asarray([0.9, 0.8], dtype=np.float64)
    b1 = np.asarray([0.05, 0.10], dtype=np.float64)
    stress_gs = np.asarray([0.4, 0.2], dtype=np.float64)

    result = diffuco_vpd_boundary_conductance(
        qsurf=qsurf,
        qsatt=qsatt,
        t2m=t2m,
        pb=pb,
        water_lim=water_lim,
        a1=a1,
        b1=b1,
        stress_gs=stress_gs,
    )

    vapor_air = qsurf * pb / (0.622 + qsurf * 0.378)
    vapor_sat = qsatt * pb / (0.622 + qsatt * 0.378)
    air_relhum = vapor_air / vapor_sat
    vpd = (vapor_sat - vapor_air) / 10.0
    bounded = np.minimum(1.0 - MIN_SECHIBA, np.maximum(MIN_SECHIBA, a1[None, :] - b1[None, :] * vpd[:, None]))
    fvpd = (1.0 / (1.0 / bounded - 1.0)) * np.maximum(1.0 - stress_gs[None, :], water_lim)
    gb_h2o = 0.04 * 44.6 * (273.15 / t2m) * (pb / 1013.0)
    gb_co2 = gb_h2o / 1.6

    np.testing.assert_allclose(np.asarray(result.air_relhum), air_relhum)
    np.testing.assert_allclose(np.asarray(result.vpd), vpd)
    np.testing.assert_allclose(np.asarray(result.fvpd), fvpd)
    np.testing.assert_allclose(np.asarray(result.gb_h2o), gb_h2o)
    np.testing.assert_allclose(np.asarray(result.gb_co2), gb_co2)


def test_diffuco_arrhenius_helpers_match_fortran_formulas():
    temp = np.asarray([273.15, 298.0, 310.0], dtype=np.float64)
    energy_act = 65000.0
    energy_deact = 200000.0
    entropy = np.asarray([650.0, 660.0, 670.0], dtype=np.float64)

    arr = diffuco_arrhenius(temp, 298.0, energy_act)
    expected_arr = np.exp(((temp - 298.0) * energy_act) / (298.0 * 8.314 * temp))
    mod = diffuco_arrhenius_modified(temp, 298.0, energy_act, energy_deact, entropy)
    expected_mod = expected_arr * (
        (1.0 + np.exp((298.0 * entropy - energy_deact) / (298.0 * 8.314)))
        / (1.0 + np.exp((temp * entropy - energy_deact) / (8.314 * temp)))
    )

    np.testing.assert_allclose(np.asarray(arr), expected_arr)
    np.testing.assert_allclose(np.asarray(mod), expected_mod)


def test_diffuco_photo_temperature_response_matches_source_temperature_bundle():
    t2m = np.asarray([273.15, 298.0, 310.0], dtype=np.float64)
    temp_growth = np.asarray([10.0, 25.0, 36.0], dtype=np.float64)
    vcmax = np.asarray([40.0, 50.0, 60.0], dtype=np.float64)
    water_lim = np.asarray([0.2, 0.7, 0.9], dtype=np.float64)
    params = {
        "e_kmc": 79430.0,
        "e_kmo": 36380.0,
        "e_sco": -24460.0,
        "e_gamma_star": 37830.0,
        "e_rd": 46390.0,
        "e_jmax": 43540.0,
        "d_jmax": 200000.0,
        "asj": 650.0,
        "bsj": -0.50,
        "e_vcmax": 58520.0,
        "d_vcmax": 200000.0,
        "asv": 668.0,
        "bsv": -1.07,
        "e_gm": 49600.0,
        "d_gm": 437400.0,
        "s_gm": 1400.0,
        "arjv": 2.59,
        "brjv": -0.035,
        "gm25": 0.10,
        "stress_gm": 0.35,
        "stress_gs": 0.25,
        "g0": 0.01,
        "kmc25": 404.9,
        "kmo25": 278400.0,
        "sco25": 2590.0,
        "gamma_star25": 42.75,
    }

    result = diffuco_photo_temperature_response(
        t2m=t2m,
        temp_growth=temp_growth,
        vcmax=vcmax,
        water_lim=water_lim,
        **params,
    )

    growth = np.maximum(11.0, np.minimum(temp_growth, 35.0))
    t_kmc = np.asarray(diffuco_arrhenius(t2m, 298.0, params["e_kmc"]))
    t_kmo = np.asarray(diffuco_arrhenius(t2m, 298.0, params["e_kmo"]))
    t_sco = np.asarray(diffuco_arrhenius(t2m, 298.0, params["e_sco"]))
    t_gamma = np.asarray(diffuco_arrhenius(t2m, 298.0, params["e_gamma_star"]))
    t_rd = np.asarray(diffuco_arrhenius(t2m, 298.0, params["e_rd"]))
    s_jmax = params["asj"] + params["bsj"] * growth
    t_jmax = np.asarray(diffuco_arrhenius_modified(t2m, 298.0, params["e_jmax"], params["d_jmax"], s_jmax))
    s_vcmax = params["asv"] + params["bsv"] * growth
    t_vcmax = np.asarray(diffuco_arrhenius_modified(t2m, 298.0, params["e_vcmax"], params["d_vcmax"], s_vcmax))
    t_gm = np.asarray(diffuco_arrhenius_modified(t2m, 298.0, params["e_gm"], params["d_gm"], params["s_gm"]))

    np.testing.assert_allclose(np.asarray(result.s_jmax_acclim_temp), s_jmax)
    np.testing.assert_allclose(np.asarray(result.s_vcmax_acclim_temp), s_vcmax)
    np.testing.assert_allclose(np.asarray(result.t_kmc), t_kmc)
    np.testing.assert_allclose(np.asarray(result.t_kmo), t_kmo)
    np.testing.assert_allclose(np.asarray(result.t_sco), t_sco)
    np.testing.assert_allclose(np.asarray(result.t_gamma_star), t_gamma)
    np.testing.assert_allclose(np.asarray(result.t_rd), t_rd)
    np.testing.assert_allclose(np.asarray(result.t_jmax), t_jmax)
    np.testing.assert_allclose(np.asarray(result.t_vcmax), t_vcmax)
    np.testing.assert_allclose(np.asarray(result.t_gm), t_gm)
    np.testing.assert_allclose(np.asarray(result.vc), vcmax * t_vcmax)
    np.testing.assert_allclose(np.asarray(result.vj), (params["arjv"] + params["brjv"] * growth) * vcmax * t_jmax)
    np.testing.assert_allclose(np.asarray(result.gm), params["gm25"] * t_gm * np.maximum(1.0 - params["stress_gm"], water_lim))
    np.testing.assert_allclose(np.asarray(result.g0var), params["g0"] * np.maximum(1.0 - params["stress_gs"], water_lim))
    np.testing.assert_allclose(np.asarray(result.kmc), params["kmc25"] * t_kmc)
    np.testing.assert_allclose(np.asarray(result.kmo), params["kmo25"] * t_kmo)
    np.testing.assert_allclose(np.asarray(result.sco), params["sco25"] * t_sco)
    np.testing.assert_allclose(np.asarray(result.gamma_star), params["gamma_star25"] * t_gamma)
    np.testing.assert_allclose(np.asarray(result.low_gamma_star), 0.5 / (params["sco25"] * t_sco))
    assert np.asarray(result.s_jmax_acclim_temp)[0] == pytest.approx(params["asj"] + params["bsj"] * 11.0)
    assert np.asarray(result.s_jmax_acclim_temp)[2] == pytest.approx(params["asj"] + params["bsj"] * 35.0)


def _manual_c3_yin_layer(vc2, jj, rd, ca, gamma_star, kmc, kmo, sco, gm, gb_co2, g0var, fvpd):
    def root_for(x1, x2):
        a = g0var * (x2 + gamma_star) + (g0var / gm + fvpd) * (x1 - rd)
        b = ca * (x1 - rd) - gamma_star * x1 - rd * x2
        c = ca + x2 + (1.0 / gm + 1.0 / gb_co2) * (x1 - rd)
        d = x2 + gamma_star + (x1 - rd) / gm
        m = 1.0 / gm + (g0var / gm + fvpd) * (1.0 / gm + 1.0 / gb_co2)
        p = -(d + (x1 - rd) / gm + a * (1.0 / gm + 1.0 / gb_co2) + (g0var / gm + fvpd) * c) / m
        q = (d * (x1 - rd) + a * c + (g0var / gm + fvpd) * b) / m
        r = -a * b / m
        qq = (p**2 - 3.0 * q) / 9.0
        uu = (2.0 * p**3 - 9.0 * p * q + 27.0 * r) / 54.0
        valid = qq >= 0.0 and abs(uu / (qq**1.5)) <= 1.0
        root = np.nan
        if valid:
            psi = np.arccos(uu / (qq**1.5))
            root = -2.0 * np.sqrt(qq) * np.cos(psi / 3.0) - p / 3.0
        return root, valid, p, q, r, qq, uu

    x1_vc = vc2
    x2_vc = kmc * (1.0 + 2.0 * gamma_star * sco / kmo)
    root_vc, valid_vc, p_vc, q_vc, r_vc, qq_vc, uu_vc = root_for(x1_vc, x2_vc)
    x1_j = jj / 4.0
    x2_j = 2.0 * gamma_star
    root_j, valid_j, p_j, q_j, r_j, qq_j, uu_j = root_for(x1_j, x2_j)
    a1 = root_vc if valid_vc and root_vc < 9999.0 else 9999.0
    info = 2.0 if valid_vc and root_vc < 9999.0 else 0.0
    x1, x2 = x1_j, x2_j
    p, q, r, qq, uu = p_j, q_j, r_j, qq_j, uu_j
    if valid_j and root_j < a1:
        a1 = root_j
        info = 2.0
    elif valid_j:
        x1, x2 = x1_vc, x2_vc
        info = 1.0
    if (not (valid_vc or valid_j)) or a1 < -rd:
        a1 = -rd
    cc = (gamma_star * x1 + (a1 + rd) * x2) / max(MIN_SECHIBA, x1 - (a1 + rd))
    leaf_ci = cc + a1 / gm
    ci_star = gamma_star - rd / gm
    if abs(a1 + rd) < MIN_SECHIBA:
        gs = g0var
    else:
        gs = g0var + (a1 + rd) / (leaf_ci - ci_star) * fvpd
    return {
        "assimi": a1,
        "cc": cc,
        "leaf_ci": leaf_ci,
        "ci_star": ci_star,
        "gs": gs,
        "info_limitphoto": info,
        "x1": x1,
        "x2": x2,
        "p": p,
        "q": q,
        "r": r,
        "qq": qq,
        "uu": uu,
        "root_vc": root_vc,
        "root_j": root_j,
        "valid_vc": valid_vc,
        "valid_j": valid_j,
    }


def test_diffuco_c3_assimilation_yin_layer_matches_fortran_cubic_state_machine():
    inputs = {
        "vc2": np.asarray([42.0, 20.0], dtype=np.float64),
        "jj": np.asarray([120.0, 40.0], dtype=np.float64),
        "rd": np.asarray([0.8, 0.5], dtype=np.float64),
        "ca": np.asarray([410.0, 390.0], dtype=np.float64),
        "gamma_star": np.asarray([42.0, 40.0], dtype=np.float64),
        "kmc": np.asarray([405.0, 390.0], dtype=np.float64),
        "kmo": np.asarray([278400.0, 270000.0], dtype=np.float64),
        "sco": np.asarray([2590.0, 2500.0], dtype=np.float64),
        "gm": np.asarray([0.10, 0.08], dtype=np.float64),
        "gb_co2": np.asarray([1.20, 1.10], dtype=np.float64),
        "g0var": np.asarray([0.01, 0.02], dtype=np.float64),
        "fvpd": np.asarray([0.6, 0.4], dtype=np.float64),
    }

    result = diffuco_c3_assimilation_yin_layer(**inputs)
    expected = [
        _manual_c3_yin_layer(*(inputs[name][idx] for name in inputs))
        for idx in range(2)
    ]

    for field in (
        "assimi",
        "cc",
        "leaf_ci",
        "ci_star",
        "gs",
        "info_limitphoto",
        "x1",
        "x2",
        "p",
        "q",
        "r",
        "qq",
        "uu",
        "root_vc",
        "root_j",
        "valid_vc",
        "valid_j",
    ):
        np.testing.assert_allclose(
            np.asarray(getattr(result, field)),
            np.asarray([item[field] for item in expected], dtype=np.float64),
            rtol=1e-12,
            atol=1e-12,
        )
    assert set(np.asarray(result.info_limitphoto).tolist()) <= {1.0, 2.0}


def test_diffuco_c3_assimilation_yin_layer_fallback_sets_assimilation_to_minus_rd_and_g0():
    result = diffuco_c3_assimilation_yin_layer(
        vc2=np.asarray([-20.0], dtype=np.float64),
        jj=np.asarray([-80.0], dtype=np.float64),
        rd=np.asarray([1.0], dtype=np.float64),
        ca=np.asarray([400.0], dtype=np.float64),
        gamma_star=np.asarray([42.0], dtype=np.float64),
        kmc=np.asarray([405.0], dtype=np.float64),
        kmo=np.asarray([278400.0], dtype=np.float64),
        sco=np.asarray([2590.0], dtype=np.float64),
        gm=np.asarray([0.1], dtype=np.float64),
        gb_co2=np.asarray([1.2], dtype=np.float64),
        g0var=np.asarray([0.01], dtype=np.float64),
        fvpd=np.asarray([0.6], dtype=np.float64),
    )

    np.testing.assert_allclose(np.asarray(result.assimi), np.asarray([-1.0]))
    np.testing.assert_allclose(np.asarray(result.gs), np.asarray([0.01]))
    assert bool(np.asarray(result.fallback)[0])


def test_diffuco_c3_canopy_layer_integrals_match_fortran_lai_loop():
    assimilate = np.asarray([True, True, False], dtype=bool)
    lai = np.asarray([0.7, 1.4, 0.9], dtype=np.float64)
    laitab = np.asarray([0.0, 0.4, 0.9, 1.5], dtype=np.float64)
    light = np.asarray([0.9, 0.6, 0.3], dtype=np.float64)
    inputs = {
        "assimilate": assimilate,
        "lai": lai,
        "laitab": laitab,
        "light": light,
        "vc": np.asarray([45.0, 38.0, 50.0], dtype=np.float64),
        "vj": np.asarray([90.0, 80.0, 100.0], dtype=np.float64),
        "vcmax": np.asarray([42.0, 36.0, 48.0], dtype=np.float64),
        "t_rd": np.asarray([1.1, 0.9, 1.0], dtype=np.float64),
        "water_lim": np.asarray([0.7, 0.5, 0.8], dtype=np.float64),
        "swdown": np.asarray([500.0, 350.0, 450.0], dtype=np.float64),
        "ca": np.asarray([410.0, 395.0, 400.0], dtype=np.float64),
        "gamma_star": np.asarray([42.0, 40.0, 41.0], dtype=np.float64),
        "kmc": np.asarray([405.0, 390.0, 400.0], dtype=np.float64),
        "kmo": np.asarray([278400.0, 270000.0, 275000.0], dtype=np.float64),
        "sco": np.asarray([2590.0, 2500.0, 2550.0], dtype=np.float64),
        "gm": np.asarray([0.10, 0.08, 0.09], dtype=np.float64),
        "gb_co2": np.asarray([1.20, 1.10, 1.15], dtype=np.float64),
        "g0var": np.asarray([0.01, 0.02, 0.015], dtype=np.float64),
        "fvpd": np.asarray([0.6, 0.4, 0.5], dtype=np.float64),
        "ext_coeff": 0.5,
        "alpha_ll": 0.372,
        "theta": 0.7,
        "stress_vcmax": 0.2,
    }

    result = diffuco_c3_canopy_layer_integrals(**inputs)

    n_vcmax = 1.0 - 0.7 * (1.0 - light)
    stress = np.maximum(1.0 - inputs["stress_vcmax"], inputs["water_lim"])
    vc2 = inputs["vc"][:, None] * n_vcmax[None, :] * stress[:, None]
    vj2 = inputs["vj"][:, None] * n_vcmax[None, :] * stress[:, None]
    rd = inputs["vcmax"][:, None] * n_vcmax[None, :] * 0.01 * inputs["t_rd"][:, None] * stress[:, None]
    iabs = inputs["swdown"][:, None] * 4.6 * 0.5 * inputs["ext_coeff"] * light[None, :]
    jj = (
        inputs["alpha_ll"] * iabs
        + vj2
        - np.sqrt((inputs["alpha_ll"] * iabs + vj2) ** 2 - 4.0 * inputs["theta"] * vj2 * inputs["alpha_ll"] * iabs)
    ) / (2.0 * inputs["theta"])
    layer_width = laitab[1:] - laitab[:-1]
    calculate = assimilate[:, None] & (laitab[:-1][None, :] <= lai[:, None])
    layer = diffuco_c3_assimilation_yin_layer(
        vc2=vc2,
        jj=jj,
        rd=rd,
        ca=inputs["ca"][:, None],
        gamma_star=inputs["gamma_star"][:, None],
        kmc=inputs["kmc"][:, None],
        kmo=inputs["kmo"][:, None],
        sco=inputs["sco"][:, None],
        gm=inputs["gm"][:, None],
        gb_co2=inputs["gb_co2"][:, None],
        g0var=inputs["g0var"][:, None],
        fvpd=inputs["fvpd"][:, None],
    )
    assimi = np.where(calculate, np.asarray(layer.assimi), 0.0)
    gs = np.where(calculate, np.asarray(layer.gs), 0.0)
    leaf_ci = np.where(calculate, np.asarray(layer.leaf_ci), inputs["ca"][:, None])
    laisum = np.sum(np.where(laitab[:-1][None, :] <= lai[:, None], layer_width[None, :], 0.0), axis=1)

    np.testing.assert_allclose(np.asarray(result.vc2), vc2)
    np.testing.assert_allclose(np.asarray(result.vj2), vj2)
    np.testing.assert_allclose(np.asarray(result.rd), rd)
    np.testing.assert_allclose(np.asarray(result.iabs), iabs)
    np.testing.assert_allclose(np.asarray(result.jj), jj)
    np.testing.assert_allclose(np.asarray(result.assimtot), np.sum(assimi * layer_width[None, :], axis=1))
    np.testing.assert_allclose(np.asarray(result.rdtot), np.sum(np.where(calculate, rd, 0.0) * layer_width[None, :], axis=1))
    np.testing.assert_allclose(np.asarray(result.gstot), np.sum(gs * layer_width[None, :], axis=1))
    np.testing.assert_allclose(np.asarray(result.leaf_gs_top), np.where(calculate[:, 0], gs[:, 0], 0.0))
    np.testing.assert_array_equal(np.asarray(result.ilai), np.maximum(1, np.sum(calculate, axis=1)))
    np.testing.assert_allclose(np.asarray(result.laisum), laisum)
    np.testing.assert_allclose(
        np.asarray(result.cim),
        np.where(
            laisum > 0.0,
            np.sum(np.where(laitab[:-1][None, :] <= lai[:, None], leaf_ci * layer_width[None, :], 0.0), axis=1) / laisum,
            0.0,
        ),
    )


def test_diffuco_trans_co2_outputs_from_fvcb_match_fortran_output_layer():
    assimilate = np.asarray([[True, True], [False, True]])
    assimtot = np.asarray([[5.0, 3.0], [4.0, 2.0]], dtype=np.float64)
    rdtot = np.asarray([[0.5, 0.2], [0.3, 0.1]], dtype=np.float64)
    gstot = np.asarray([[0.40, 0.30], [0.20, 0.25]], dtype=np.float64)
    leaf_gs_top = np.asarray([[0.10, 0.05], [0.08, 0.04]], dtype=np.float64)
    gamma_star = np.asarray([[40.0, 42.0], [41.0, 43.0]], dtype=np.float64)
    fvpd = np.asarray([[0.8, 0.7], [0.9, 0.6]], dtype=np.float64)
    g0var = np.asarray([[0.01, 0.02], [0.01, 0.02]], dtype=np.float64)
    laisum = np.asarray([[1.0, 1.5], [1.0, 2.0]], dtype=np.float64)
    ilai = np.asarray([[1, 2], [1, 3]], dtype=np.int32)
    laitab = np.asarray([0.0, 0.2, 0.5, 1.0, 1.5], dtype=np.float64)
    veget_max = np.asarray([[0.5, 0.4], [0.5, 0.3]], dtype=np.float64)
    humrel = np.asarray([[0.6, 0.7], [0.5, 0.0]], dtype=np.float64)
    zqsvegrap = np.asarray([[0.25, 0.5], [0.1, 0.2]], dtype=np.float64)
    vbeta23 = np.asarray([[0.05, 0.02], [0.03, 0.04]], dtype=np.float64)
    t2m = np.asarray([300.0, 290.0], dtype=np.float64)
    pb = np.asarray([1000.0, 990.0], dtype=np.float64)
    wind = np.asarray([2.0, 0.05], dtype=np.float64)
    q_cdrag = np.asarray([0.01, 0.02], dtype=np.float64)
    q_cdrag_pft = np.asarray([[0.011, 0.012], [0.021, 0.022]], dtype=np.float64)
    rveg_pft = np.asarray([1.0, 1.5], dtype=np.float64)
    ca = np.asarray([410.0, 390.0], dtype=np.float64)
    rstruct_const = np.asarray([25.0, 2.0], dtype=np.float64)
    ok_laidev = np.asarray([False, True])
    dt_sechiba = 1800.0

    result = diffuco_trans_co2_outputs_from_fvcb(
        assimilate=assimilate,
        assimtot=assimtot,
        rdtot=rdtot,
        gstot=gstot,
        leaf_gs_top=leaf_gs_top,
        gamma_star=gamma_star,
        fvpd=fvpd,
        g0var=g0var,
        laisum=laisum,
        ilai=ilai,
        laitab=laitab,
        veget_max=veget_max,
        humrel=humrel,
        zqsvegrap=zqsvegrap,
        vbeta23=vbeta23,
        t2m=t2m,
        pb=pb,
        wind=wind,
        q_cdrag=q_cdrag,
        q_cdrag_pft=q_cdrag_pft,
        rveg_pft=rveg_pft,
        ca=ca,
        rstruct_const=rstruct_const,
        ok_laidev=ok_laidev,
        dt_sechiba=dt_sechiba,
    )

    gsmean = np.where(assimilate, gstot, 0.0)
    denom = gsmean - g0var * laisum
    cimean = np.where(
        assimilate,
        np.where(np.abs(denom) > MIN_SECHIBA, fvpd * (assimtot + rdtot) / denom + gamma_star, gamma_star),
        ca[:, None],
    )
    gpp = np.where(assimilate, assimtot * 12.0e-6 * veget_max * dt_sechiba, 0.0)
    scale = 0.0244 * (t2m[:, None] / 273.15) * (1013.0 / pb[:, None]) * 1.6
    gstot_m = scale * gstot
    gstop_m = scale * leaf_gs_top * laitab[ilai]
    rveget = np.where(assimilate, 1.0 / gstop_m, 1.0e20)
    rstruct = np.where(assimilate, np.maximum(1.0 / gstot_m - np.where(assimilate, 1.0 / gstop_m, 0.0), MIN_SECHIBA), rstruct_const[None, :])
    drag = np.where(ok_laidev[None, :], q_cdrag_pft, q_cdrag[:, None])
    speed = np.maximum(0.1, wind)
    cresist = 1.0 / (1.0 + speed[:, None] * drag * (rveg_pft[None, :] * (rveget + rstruct)))
    vbeta3 = np.where(
        assimilate,
        np.where(
            humrel >= MIN_SECHIBA,
            veget_max * (1.0 - zqsvegrap) * cresist + np.minimum(vbeta23, veget_max * zqsvegrap * cresist),
            0.0,
        ),
        0.0,
    )
    vbeta3pot = np.where(assimilate, np.maximum(0.0, veget_max * cresist), 0.0)

    np.testing.assert_allclose(np.asarray(result.gsmean), gsmean)
    np.testing.assert_allclose(np.asarray(result.cimean), cimean)
    np.testing.assert_allclose(np.asarray(result.gpp), gpp)
    np.testing.assert_allclose(np.asarray(result.rveget), rveget)
    np.testing.assert_allclose(np.asarray(result.rstruct), rstruct)
    np.testing.assert_allclose(np.asarray(result.vbeta3), vbeta3)
    np.testing.assert_allclose(np.asarray(result.vbeta3pot), vbeta3pot)
    np.testing.assert_allclose(np.asarray(result.cresist), np.where(assimilate, cresist, 0.0))
    assert np.asarray(result.vbeta3)[1, 1] == pytest.approx(0.0)
    assert np.asarray(result.vbeta3pot)[1, 1] > 0.0


def test_diffuco_trans_co2_outputs_from_fvcb_validates_lai_index_bounds():
    with pytest.raises(ValueError, match="ilai"):
        diffuco_trans_co2_outputs_from_fvcb(
            assimilate=np.ones((1, 1), dtype=bool),
            assimtot=np.ones((1, 1), dtype=np.float64),
            rdtot=np.zeros((1, 1), dtype=np.float64),
            gstot=np.ones((1, 1), dtype=np.float64),
            leaf_gs_top=np.ones((1, 1), dtype=np.float64),
            gamma_star=np.ones((1, 1), dtype=np.float64),
            fvpd=np.ones((1, 1), dtype=np.float64),
            g0var=np.zeros((1, 1), dtype=np.float64),
            laisum=np.ones((1, 1), dtype=np.float64),
            ilai=np.asarray([[3]], dtype=np.int32),
            laitab=np.asarray([0.0, 1.0, 2.0], dtype=np.float64),
            veget_max=np.ones((1, 1), dtype=np.float64),
            humrel=np.ones((1, 1), dtype=np.float64),
            zqsvegrap=np.zeros((1, 1), dtype=np.float64),
            vbeta23=np.zeros((1, 1), dtype=np.float64),
            t2m=np.asarray([300.0], dtype=np.float64),
            pb=np.asarray([1000.0], dtype=np.float64),
            wind=np.asarray([1.0], dtype=np.float64),
            q_cdrag=np.asarray([0.01], dtype=np.float64),
            q_cdrag_pft=np.asarray([[0.01]], dtype=np.float64),
            rveg_pft=np.asarray([1.0], dtype=np.float64),
            ca=np.asarray([410.0], dtype=np.float64),
            rstruct_const=np.asarray([25.0], dtype=np.float64),
            ok_laidev=np.asarray([False]),
            dt_sechiba=1800.0,
        )


def _pft14_c3_trans_inputs():
    return {
        "swdown": np.asarray([500.0, 350.0], dtype=np.float64),
        "qsurf": np.asarray([0.010, 0.009], dtype=np.float64),
        "qsatt": np.asarray([0.014, 0.013], dtype=np.float64),
        "t2m": np.asarray([300.0, 296.0], dtype=np.float64),
        "pb": np.asarray([1000.0, 990.0], dtype=np.float64),
        "wind": np.asarray([2.0, 1.5], dtype=np.float64),
        "temp_growth": np.asarray([25.0, 28.0], dtype=np.float64),
        "ca": np.asarray([410.0, 395.0], dtype=np.float64),
        "vcmax": np.asarray([42.0, 36.0], dtype=np.float64),
        "humrel": np.asarray([0.7, 0.5], dtype=np.float64),
        "veget": np.asarray([0.35, 0.25], dtype=np.float64),
        "veget_max": np.asarray([0.45, 0.35], dtype=np.float64),
        "lai": np.asarray([0.7, 1.4], dtype=np.float64),
        "qsintveg": np.asarray([0.02, 0.04], dtype=np.float64),
        "qsintmax": np.asarray([0.20, 0.25], dtype=np.float64),
        "vbeta23": np.asarray([0.04, 0.03], dtype=np.float64),
        "q_cdrag": np.asarray([0.010, 0.012], dtype=np.float64),
        "q_cdrag_pft": np.asarray([0.011, 0.013], dtype=np.float64),
        "tphoto_min": 273.15,
        "tphoto_max": 330.0,
        "ok_laidev": False,
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
        "alpha_ll": 0.372,
        "theta": 0.7,
        "stress_vcmax": 0.2,
        "rveg_pft": 1.0,
        "rstruct_const": 25.0,
        "dt_sechiba": 1800.0,
        "control_salinity": np.asarray([0.8, 0.9], dtype=np.float64),
        "control_inudate": np.asarray([0.7, 1.0], dtype=np.float64),
        "nlai": 3,
        "laimax": 1.5,
        "lai_level_depth": 0.2,
    }


def test_diffuco_trans_co2_c3_pft_explicit_matches_source_order_helpers():
    inputs = _pft14_c3_trans_inputs()

    result = diffuco_trans_co2_c3_pft_explicit(**inputs)

    activity = diffuco_trans_co2_activity(
        swdown=inputs["swdown"],
        humrel=inputs["humrel"][:, None],
        veget=inputs["veget"][:, None],
        veget_max=inputs["veget_max"][:, None],
        lai=inputs["lai"][:, None],
        qsintveg=inputs["qsintveg"][:, None],
        qsintmax=inputs["qsintmax"][:, None],
        temp_growth=inputs["temp_growth"],
        tphoto_min=np.asarray([inputs["tphoto_min"]], dtype=np.float64),
        tphoto_max=np.asarray([inputs["tphoto_max"]], dtype=np.float64),
        ok_laidev=np.asarray([inputs["ok_laidev"]]),
    )
    lai_light = diffuco_lai_light_table(
        np.asarray([inputs["ext_coeff"]], dtype=np.float64),
        nlai=inputs["nlai"],
        laimax=inputs["laimax"],
        lai_level_depth=inputs["lai_level_depth"],
    )
    vpd = diffuco_vpd_boundary_conductance(
        qsurf=inputs["qsurf"],
        qsatt=inputs["qsatt"],
        t2m=inputs["t2m"],
        pb=inputs["pb"],
        water_lim=activity.water_lim,
        a1=np.asarray([inputs["a1"]]),
        b1=np.asarray([inputs["b1"]]),
        stress_gs=np.asarray([inputs["stress_gs"]]),
    )
    photo = diffuco_photo_temperature_response(
        t2m=inputs["t2m"],
        temp_growth=inputs["temp_growth"],
        vcmax=inputs["vcmax"],
        water_lim=activity.water_lim[:, 0],
        e_kmc=inputs["e_kmc"],
        e_kmo=inputs["e_kmo"],
        e_sco=inputs["e_sco"],
        e_gamma_star=inputs["e_gamma_star"],
        e_rd=inputs["e_rd"],
        e_jmax=inputs["e_jmax"],
        d_jmax=inputs["d_jmax"],
        asj=inputs["asj"],
        bsj=inputs["bsj"],
        e_vcmax=inputs["e_vcmax"],
        d_vcmax=inputs["d_vcmax"],
        asv=inputs["asv"],
        bsv=inputs["bsv"],
        e_gm=inputs["e_gm"],
        d_gm=inputs["d_gm"],
        s_gm=inputs["s_gm"],
        arjv=inputs["arjv"],
        brjv=inputs["brjv"],
        gm25=inputs["gm25"],
        stress_gm=inputs["stress_gm"],
        stress_gs=inputs["stress_gs"],
        g0=inputs["g0"],
        kmc25=inputs["kmc25"],
        kmo25=inputs["kmo25"],
        sco25=inputs["sco25"],
        gamma_star25=inputs["gamma_star25"],
    )
    canopy = diffuco_c3_canopy_layer_integrals(
        assimilate=activity.assimilate[:, 0],
        lai=inputs["lai"],
        laitab=lai_light.laitab,
        light=lai_light.light[0],
        vc=photo.vc,
        vj=photo.vj,
        vcmax=inputs["vcmax"],
        t_rd=photo.t_rd,
        water_lim=activity.water_lim[:, 0],
        swdown=inputs["swdown"],
        ca=inputs["ca"],
        gamma_star=photo.gamma_star,
        kmc=photo.kmc,
        kmo=photo.kmo,
        sco=photo.sco,
        gm=photo.gm,
        gb_co2=vpd.gb_co2,
        g0var=photo.g0var,
        fvpd=vpd.fvpd[:, 0],
        ext_coeff=inputs["ext_coeff"],
        alpha_ll=inputs["alpha_ll"],
        theta=inputs["theta"],
        stress_vcmax=inputs["stress_vcmax"],
    )
    controlled = diffuco_apply_pft14_assimtot_controls(
        canopy.assimtot,
        control_salinity=inputs["control_salinity"],
        control_inudate=inputs["control_inudate"],
        jv=14,
    )
    expected = diffuco_trans_co2_outputs_from_fvcb(
        assimilate=activity.assimilate,
        assimtot=controlled[:, None],
        rdtot=canopy.rdtot[:, None],
        gstot=canopy.gstot[:, None],
        leaf_gs_top=canopy.leaf_gs_top[:, None],
        gamma_star=photo.gamma_star[:, None],
        fvpd=vpd.fvpd,
        g0var=photo.g0var[:, None],
        laisum=canopy.laisum[:, None],
        ilai=canopy.ilai[:, None],
        laitab=lai_light.laitab,
        veget_max=inputs["veget_max"][:, None],
        humrel=inputs["humrel"][:, None],
        zqsvegrap=activity.zqsvegrap,
        vbeta23=inputs["vbeta23"][:, None],
        t2m=inputs["t2m"],
        pb=inputs["pb"],
        wind=inputs["wind"],
        q_cdrag=inputs["q_cdrag"],
        q_cdrag_pft=inputs["q_cdrag_pft"][:, None],
        rveg_pft=np.asarray([inputs["rveg_pft"]]),
        ca=inputs["ca"],
        rstruct_const=np.asarray([inputs["rstruct_const"]]),
        ok_laidev=np.asarray([inputs["ok_laidev"]]),
        dt_sechiba=inputs["dt_sechiba"],
    )

    np.testing.assert_allclose(np.asarray(result.controlled_assimtot), np.asarray(controlled))
    np.testing.assert_allclose(np.asarray(result.canopy.assimtot), np.asarray(canopy.assimtot))
    np.testing.assert_allclose(np.asarray(result.canopy.cim), np.asarray(canopy.cim))
    np.testing.assert_allclose(np.asarray(result.output.gpp), np.asarray(expected.gpp))
    np.testing.assert_allclose(np.asarray(result.output.cimean), np.asarray(expected.cimean))
    np.testing.assert_allclose(np.asarray(result.output.vbeta3), np.asarray(expected.vbeta3))
    np.testing.assert_allclose(np.asarray(result.output.vbeta3pot), np.asarray(expected.vbeta3pot))


def _server_pft14_trans_inputs(row: dict[str, object]) -> dict[str, object]:
    used = DIFFUCO_USED_RUN_DEF
    return {
        "swdown": np.asarray([row["swdown"]], dtype=np.float64),
        "qsurf": np.asarray([row["qsurf"]], dtype=np.float64),
        "qsatt": np.asarray([row["qsatt"]], dtype=np.float64),
        "t2m": np.asarray([row["t2m"]], dtype=np.float64),
        "pb": np.asarray([row["pb"]], dtype=np.float64),
        "wind": np.asarray([row["wind"]], dtype=np.float64),
        "temp_growth": np.asarray([row["temp_growth"]], dtype=np.float64),
        "ca": np.asarray([row["ca"]], dtype=np.float64),
        "vcmax": np.asarray([row["vcmax"]], dtype=np.float64),
        "humrel": np.asarray([row["humrel"]], dtype=np.float64),
        "veget": np.asarray([row["veget"]], dtype=np.float64),
        "veget_max": np.asarray([row["veget_max"]], dtype=np.float64),
        "lai": np.asarray([row["lai"]], dtype=np.float64),
        "qsintveg": np.asarray([row["qsintveg"]], dtype=np.float64),
        "qsintmax": np.asarray([row["qsintmax"]], dtype=np.float64),
        "vbeta23": np.asarray([row["vbeta23"]], dtype=np.float64),
        "q_cdrag": np.asarray([row["q_cdrag"]], dtype=np.float64),
        "q_cdrag_pft": np.asarray([row["q_cdrag_pft"]], dtype=np.float64),
        "tphoto_min": _run_def_indexed_float(used, "TPHOTO_MIN"),
        "tphoto_max": _run_def_indexed_float(used, "TPHOTO_MAX"),
        "ok_laidev": _run_def_indexed_bool(used, "OK_LAIDEV"),
        "ext_coeff": _run_def_indexed_float(used, "EXT_COEFF"),
        "a1": _run_def_indexed_float(used, "A1"),
        "b1": _run_def_indexed_float(used, "B1"),
        "stress_gs": _run_def_indexed_float(used, "STRESS_GS"),
        "e_kmc": _run_def_indexed_float(used, "E_KMC"),
        "e_kmo": _run_def_indexed_float(used, "E_KMO"),
        "e_sco": _run_def_indexed_float(used, "E_SCO"),
        "e_gamma_star": _run_def_indexed_float(used, "E_GAMMA_STAR"),
        "e_rd": _run_def_indexed_float(used, "E_RD"),
        "e_jmax": _run_def_indexed_float(used, "E_JMAX"),
        "d_jmax": _run_def_indexed_float(used, "D_JMAX"),
        "asj": _run_def_indexed_float(used, "ASJ"),
        "bsj": _run_def_indexed_float(used, "BSJ"),
        "e_vcmax": _run_def_indexed_float(used, "E_VCMAX"),
        "d_vcmax": _run_def_indexed_float(used, "D_VCMAX"),
        "asv": _run_def_indexed_float(used, "ASV"),
        "bsv": _run_def_indexed_float(used, "BSV"),
        "e_gm": _run_def_indexed_float(used, "E_GM"),
        "d_gm": _run_def_indexed_float(used, "D_GM"),
        "s_gm": _run_def_indexed_float(used, "S_GM"),
        "arjv": _run_def_indexed_float(used, "ARJV"),
        "brjv": _run_def_indexed_float(used, "BRJV"),
        "gm25": _run_def_indexed_float(used, "GM25"),
        "stress_gm": _run_def_indexed_float(used, "STRESS_GM"),
        "g0": _run_def_indexed_float(used, "G0"),
        "kmc25": _run_def_indexed_float(used, "KMC25"),
        "kmo25": _run_def_indexed_float(used, "KMO25"),
        "sco25": _run_def_indexed_float(used, "SCO25"),
        "gamma_star25": _run_def_indexed_float(used, "gamma_star25"),
        # Fortran source truth: src_parameters/constantes_mtc.f90 lines 440-442
        # gives alpha_LL_mtc(2)=0.3, and used_run.def maps PFT14 to MTC2.
        "alpha_ll": 0.3,
        "theta": _run_def_indexed_float(used, "THETA"),
        "stress_vcmax": _run_def_indexed_float(used, "STRESS_VCMAX"),
        "rveg_pft": _run_def_indexed_float(used, "RVEG_PFT"),
        "rstruct_const": _run_def_indexed_float(used, "RSTRUCT_CONST"),
        "dt_sechiba": _run_def_float(used, "DT_SECHIBA"),
        "control_salinity": np.asarray([row["control_salinity"]], dtype=np.float64),
        "control_inudate": np.asarray([row["control_inudate"]], dtype=np.float64),
    }


def test_server_1961_diffuco_trans_co2_pft14_inactive_records_match_fortran_trace():
    records = read_diffuco_trans_co2_records(root=DIFFUCO_TRACE_ROOT, limit=4)

    assert len(records) == 4
    assert all(record["lai"] == pytest.approx(0.0) for record in records)
    for row in records:
        result = diffuco_trans_co2_c3_pft_explicit(**_server_pft14_trans_inputs(row))

        assert not bool(np.asarray(result.activity.assimilate)[0, 0])
        assert not bool(np.asarray(result.activity.lai_active)[0, 0])
        np.testing.assert_allclose(np.asarray(result.output.gpp)[0, 0], row["gpp"], atol=0.0, rtol=0.0)
        np.testing.assert_allclose(np.asarray(result.output.gsmean)[0, 0], row["gsmean"], atol=0.0, rtol=0.0)
        np.testing.assert_allclose(np.asarray(result.output.rveget)[0, 0], row["rveget"], rtol=1e-15)
        np.testing.assert_allclose(np.asarray(result.output.rstruct)[0, 0], row["rstruct"], rtol=1e-15)
        np.testing.assert_allclose(np.asarray(result.output.cimean)[0, 0], row["cimean"], rtol=1e-15)
        np.testing.assert_allclose(np.asarray(result.output.vbeta3)[0, 0], row["vbeta3"], atol=0.0, rtol=0.0)
        np.testing.assert_allclose(np.asarray(result.output.vbeta3pot)[0, 0], row["vbeta3pot"], atol=0.0, rtol=0.0)
        np.testing.assert_allclose(np.asarray(result.canopy.laisum)[0], row["laisum"], rtol=1e-14)
        np.testing.assert_allclose(np.asarray(result.canopy.cim)[0], row["cim"], rtol=1e-15)
        assert int(np.asarray(result.canopy.ilai)[0]) == row["ilai"]
        np.testing.assert_allclose(np.asarray(result.photo_temperature.gamma_star)[0], row["gamma_star"], rtol=1e-13)
        np.testing.assert_allclose(np.asarray(result.vpd_boundary.fvpd)[0, 0], row["fvpd"], rtol=1e-15)
        np.testing.assert_allclose(np.asarray(result.photo_temperature.g0var)[0], row["g0var"], rtol=1e-15)


def test_server_1961_diffuco_trans_co2_pft14_first_active_record_matches_fortran_trace():
    records = read_diffuco_trans_co2_records(root=DIFFUCO_TRACE_ROOT, limit=None)
    row = next(record for record in records if record["lai"] > 0.01 or record["gpp"] > 0.0)

    assert row["kjit"] == 49
    assert row["lai"] == pytest.approx(0.779116943429268)
    result = diffuco_trans_co2_c3_pft_explicit(**_server_pft14_trans_inputs(row))

    assert bool(np.asarray(result.activity.assimilate)[0, 0])
    assert bool(np.asarray(result.activity.lai_active)[0, 0])
    np.testing.assert_allclose(np.asarray(result.output.gpp)[0, 0], row["gpp"], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(result.output.gsmean)[0, 0], row["gsmean"], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(result.output.rveget)[0, 0], row["rveget"], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(result.output.rstruct)[0, 0], row["rstruct"], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(result.output.cimean)[0, 0], row["cimean"], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(result.output.vbeta3)[0, 0], row["vbeta3"], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(result.output.vbeta3pot)[0, 0], row["vbeta3pot"], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(result.controlled_assimtot)[0], row["assimtot"], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(result.canopy.rdtot)[0], row["rdtot"], rtol=1e-12)
    # The trace writes Fortran's local gstot after lines 2915-2916 mutate it
    # from mol m-2 s-1 into m s-1; the pre-conversion canopy integral is
    # already checked through gsmean above.
    np.testing.assert_allclose(np.asarray(result.output.gstot_m_per_s)[0, 0], row["gstot"], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(result.canopy.leaf_gs_top)[0], row["leaf_gs_top"], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(result.canopy.laisum)[0], row["laisum"], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(result.canopy.cim)[0], row["cim"], rtol=1e-12)
    assert int(np.asarray(result.canopy.ilai)[0]) == row["ilai"]
    np.testing.assert_allclose(np.asarray(result.photo_temperature.gamma_star)[0], row["gamma_star"], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(result.vpd_boundary.fvpd)[0, 0], row["fvpd"], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(result.photo_temperature.g0var)[0], row["g0var"], rtol=1e-12)


def test_server_1961_diffuco_pft14_active_main_beta_closure_matches_after_main_trace():
    trans_row = first_active_pft14_diffuco_trans_co2_payload()
    after_main = first_active_pft14_diffuco_after_main_payload()
    assert after_main is not None
    assert after_main["kjit"] == trans_row["kjit"] == 49

    trans = diffuco_trans_co2_c3_pft_explicit(**_server_pft14_trans_inputs(trans_row))
    nvm = 14
    pft_index = 13

    def scalar(value):
        return np.asarray([value], dtype=np.float64)

    def pft14_column(value):
        array = np.zeros((1, nvm), dtype=np.float64)
        array[0, pft_index] = value
        return array

    closure = diffuco_pft14_c3_beta_closure_explicit(
        trans_co2=trans,
        pft_index=pft_index,
        humrel=pft14_column(after_main["humrel"]),
        qair=scalar(after_main["qair"]),
        temp_air=scalar(after_main["temp_air"]),
        qsatt=scalar(trans_row["qsatt"]),
        veget=pft14_column(after_main["veget"]),
        veget_max=pft14_column(after_main["veget_max"]),
        lai=pft14_column(after_main["lai"]),
        tot_bare_soil=scalar(after_main["tot_bare_soil"]),
        evap_bare_lim=scalar(after_main["evap_bare_lim"]),
        vbeta1=scalar(after_main["vbeta1"]),
        vbeta2=pft14_column(after_main["vbeta2"]),
        vbeta3_background=np.zeros((1, nvm), dtype=np.float64),
        qsintmax=pft14_column(after_main["qsintmax"]),
    )

    np.testing.assert_allclose(np.asarray(closure.vbeta3)[0, pft_index], after_main["vbeta3"], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(closure.vbeta3pot)[0, pft_index], after_main["vbeta3pot"], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(closure.bare.vbeta4)[0], after_main["vbeta4"], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(np.asarray(closure.bare.vbeta4_pft)[0, pft_index], after_main["vbeta4_pft"], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(np.asarray(closure.comb.valpha)[0], after_main["valpha"], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(np.asarray(closure.comb.vbeta)[0], after_main["vbeta"], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(closure.comb.vbeta_pft)[0, pft_index], after_main["vbeta_pft"], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(np.asarray(closure.comb.vbeta1)[0], after_main["vbeta1"], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(np.asarray(closure.comb.vbeta2)[0, pft_index], after_main["vbeta2"], atol=0.0, rtol=0.0)
    assert not bool(np.asarray(closure.comb.toveg)[0])
    assert not bool(np.asarray(closure.comb.tosnow)[0])


def test_diffuco_pft14_after_main_payload_exports_only_source_backed_and_explicit_fields():
    trans_row = first_active_pft14_diffuco_trans_co2_payload()
    after_main = first_active_pft14_diffuco_after_main_payload()
    assert after_main is not None
    trans = diffuco_trans_co2_c3_pft_explicit(**_server_pft14_trans_inputs(trans_row))
    nvm = 14
    pft_index = 13

    def scalar(value):
        return np.asarray([value], dtype=np.float64)

    def pft14_column(value):
        array = np.zeros((1, nvm), dtype=np.float64)
        array[0, pft_index] = value
        return array

    closure = diffuco_pft14_c3_beta_closure_explicit(
        trans_co2=trans,
        pft_index=pft_index,
        humrel=pft14_column(after_main["humrel"]),
        qair=scalar(after_main["qair"]),
        temp_air=scalar(after_main["temp_air"]),
        qsatt=scalar(trans_row["qsatt"]),
        veget=pft14_column(after_main["veget"]),
        veget_max=pft14_column(after_main["veget_max"]),
        lai=pft14_column(after_main["lai"]),
        tot_bare_soil=scalar(after_main["tot_bare_soil"]),
        evap_bare_lim=scalar(after_main["evap_bare_lim"]),
        vbeta1=scalar(after_main["vbeta1"]),
        vbeta2=pft14_column(after_main["vbeta2"]),
        vbeta3_background=np.zeros((1, nvm), dtype=np.float64),
        qsintmax=pft14_column(after_main["qsintmax"]),
    )
    passthrough_names = ("q_cdrag", "q_cdrag_pft", "vbeta5", "lai", "veget", "veget_max", "qair", "temp_air")
    payload = diffuco_pft14_after_main_payload(
        closure,
        pft_index=pft_index,
        passthrough={name: scalar(after_main[name]) for name in passthrough_names},
    )

    for name in (
        "gpp",
        "gsmean",
        "rveget",
        "rstruct",
        "cimean",
        "vbeta3",
        "vbeta3pot",
        "valpha",
        "vbeta",
        "vbeta_pft",
        "vbeta1",
        "vbeta2",
        "vbeta4",
        "vbeta4_pft",
        "humrel",
    ):
        assert name in payload.local_fields
        np.testing.assert_allclose(np.asarray(payload.payload[name])[0], after_main[name], rtol=1e-12)

    assert "vbeta5" in payload.passthrough_fields
    assert "vbeta5" not in payload.local_fields
    np.testing.assert_allclose(np.asarray(payload.payload["vbeta5"])[0], after_main["vbeta5"], rtol=0.0, atol=0.0)
    assert payload.provenance[1].endswith("diffuco.f90::diffuco_main lines 668-710")


def test_diffuco_pft14_after_main_payload_rejects_passthrough_over_local_outputs():
    trans_inputs = _pft14_c3_trans_inputs()
    trans = diffuco_trans_co2_c3_pft_explicit(**trans_inputs)
    pft_index = 2
    npts, nvm = 2, 4
    closure = diffuco_pft14_c3_beta_closure_explicit(
        trans_co2=trans,
        pft_index=pft_index,
        humrel=np.ones((npts, nvm), dtype=np.float64),
        qair=np.asarray([0.010, 0.009], dtype=np.float64),
        temp_air=np.asarray([300.0, 296.0], dtype=np.float64),
        qsatt=np.asarray([0.014, 0.013], dtype=np.float64),
        veget=np.ones((npts, nvm), dtype=np.float64) * 0.2,
        veget_max=np.ones((npts, nvm), dtype=np.float64) * 0.3,
        lai=np.ones((npts, nvm), dtype=np.float64),
        tot_bare_soil=np.asarray([0.20, 0.25], dtype=np.float64),
        evap_bare_lim=np.asarray([0.25, 0.30], dtype=np.float64),
        vbeta1=np.asarray([0.1, 0.2], dtype=np.float64),
        vbeta2=np.zeros((npts, nvm), dtype=np.float64),
        vbeta3_background=np.zeros((npts, nvm), dtype=np.float64),
        qsintmax=np.ones((npts, nvm), dtype=np.float64) * 0.2,
    )

    with pytest.raises(ValueError, match="overlap local DIFFUCO outputs"):
        diffuco_pft14_after_main_payload(closure, pft_index=pft_index, passthrough={"gpp": object()})


def test_diffuco_pft14_payload_exports_comb_vbeta4_after_warm_dew_rewrite():
    """Fortran diffuco_comb mutates vbeta4 in place before ENERBIL consumes it."""

    trans_inputs = _pft14_c3_trans_inputs()
    trans = diffuco_trans_co2_c3_pft_explicit(**trans_inputs)
    trans = trans._replace(
        output=trans.output._replace(
            vbeta3=np.full((2, 1), 0.75, dtype=np.float64),
            vbeta3pot=np.full((2, 1), 0.8, dtype=np.float64),
        )
    )
    pft_index = 2
    npts, nvm = 2, 4
    humrel = np.ones((npts, nvm), dtype=np.float64) * 0.5
    veget = np.ones((npts, nvm), dtype=np.float64) * 0.1
    veget_max = np.ones((npts, nvm), dtype=np.float64) * 0.2
    lai = np.ones((npts, nvm), dtype=np.float64)
    qsintmax = np.ones((npts, nvm), dtype=np.float64) * 0.2
    tot_bare_soil = np.asarray([0.456282860493586, 0.25], dtype=np.float64)
    closure = diffuco_pft14_c3_beta_closure_explicit(
        trans_co2=trans,
        pft_index=pft_index,
        humrel=humrel,
        qair=np.asarray([0.008904067178567251, 0.009], dtype=np.float64),
        temp_air=np.asarray([295.0, 296.0], dtype=np.float64),
        qsatt=np.asarray([0.00887168167779843, 0.007], dtype=np.float64),
        veget=veget,
        veget_max=veget_max,
        lai=lai,
        tot_bare_soil=tot_bare_soil,
        evap_bare_lim=np.zeros(npts, dtype=np.float64),
        vbeta1=np.asarray([0.2, 0.3], dtype=np.float64),
        vbeta2=np.zeros((npts, nvm), dtype=np.float64),
        vbeta3_background=np.zeros((npts, nvm), dtype=np.float64),
        qsintmax=qsintmax,
    )
    backgrounds = {
        name: np.zeros((npts, nvm), dtype=np.float64)
        for name in ("gpp", "gsmean", "rveget", "rstruct", "cimean")
    }

    assert np.asarray(closure.bare.vbeta4)[0] == pytest.approx(0.0)
    np.testing.assert_allclose(np.asarray(closure.comb.vbeta4), tot_bare_soil)
    assert np.asarray(closure.vbeta3)[0, pft_index] > 0.0
    np.testing.assert_allclose(np.asarray(closure.comb.vbeta3), np.zeros((npts, nvm), dtype=np.float64))

    after_main = diffuco_pft14_after_main_payload(closure, pft_index=pft_index)
    enerbil_precall = diffuco_pft14_enerbil_precall_payload(
        closure,
        pft_index=pft_index,
        pft_output_backgrounds=backgrounds,
    )

    np.testing.assert_allclose(np.asarray(after_main.payload["vbeta4"]), tot_bare_soil)
    np.testing.assert_allclose(np.asarray(enerbil_precall.payload["vbeta4"]), tot_bare_soil)
    np.testing.assert_allclose(np.asarray(after_main.payload["vbeta3"]), np.zeros(npts, dtype=np.float64))
    np.testing.assert_allclose(np.asarray(enerbil_precall.payload["vbeta3"]), np.zeros((npts, nvm), dtype=np.float64))


def test_diffuco_pft14_enerbil_precall_payload_keeps_non_pft14_background_explicit():
    trans_inputs = _pft14_c3_trans_inputs()
    trans = diffuco_trans_co2_c3_pft_explicit(**trans_inputs)
    pft_index = 2
    npts, nvm = 2, 4
    humrel = np.asarray(
        [[0.0, 0.6, trans_inputs["humrel"][0], 0.4], [0.0, 0.5, trans_inputs["humrel"][1], 0.3]],
        dtype=np.float64,
    )
    vbeta2 = np.asarray([[0.0, 0.03, 0.04, 0.02], [0.0, 0.02, 0.03, 0.01]], dtype=np.float64)
    closure = diffuco_pft14_c3_beta_closure_explicit(
        trans_co2=trans,
        pft_index=pft_index,
        humrel=humrel,
        qair=np.asarray([0.010, 0.009], dtype=np.float64),
        temp_air=np.asarray([300.0, 296.0], dtype=np.float64),
        qsatt=np.asarray([0.014, 0.013], dtype=np.float64),
        veget=np.asarray(
            [[0.0, 0.2, trans_inputs["veget"][0], 0.1], [0.0, 0.1, trans_inputs["veget"][1], 0.2]],
            dtype=np.float64,
        ),
        veget_max=np.asarray(
            [[0.0, 0.25, trans_inputs["veget_max"][0], 0.15], [0.0, 0.15, trans_inputs["veget_max"][1], 0.25]],
            dtype=np.float64,
        ),
        lai=np.asarray(
            [[0.0, 0.8, trans_inputs["lai"][0], 0.5], [0.0, 0.7, trans_inputs["lai"][1], 0.6]],
            dtype=np.float64,
        ),
        tot_bare_soil=np.asarray([0.20, 0.25], dtype=np.float64),
        evap_bare_lim=np.asarray([0.25, 0.30], dtype=np.float64),
        vbeta1=np.asarray([0.1, 0.2], dtype=np.float64),
        vbeta2=vbeta2,
        vbeta3_background=np.asarray([[0.0, 0.05, 0.0, 0.04], [0.0, 0.04, 0.0, 0.03]], dtype=np.float64),
        qsintmax=np.asarray([[0.0, 0.2, 0.3, 0.1], [0.0, 0.2, 0.3, 0.1]], dtype=np.float64),
    )
    backgrounds = {name: np.full((npts, nvm), fill, dtype=np.float64) for fill, name in enumerate(
        ("gpp", "gsmean", "rveget", "rstruct", "cimean"),
        start=1,
    )}

    payload = diffuco_pft14_enerbil_precall_payload(
        closure,
        pft_index=pft_index,
        pft_output_backgrounds=backgrounds,
        passthrough={"q_cdrag": trans_inputs["q_cdrag"]},
    )

    for name in ("gpp", "gsmean", "rveget", "rstruct", "cimean"):
        expected = backgrounds[name].copy()
        expected[:, pft_index] = np.asarray(getattr(trans.output, name))[:, 0]
        np.testing.assert_allclose(np.asarray(payload.payload[name]), expected)

    np.testing.assert_allclose(np.asarray(payload.payload["vbeta3"]), np.asarray(closure.vbeta3))
    np.testing.assert_allclose(np.asarray(payload.payload["vbeta_pft"]), np.asarray(closure.comb.vbeta_pft))
    assert "q_cdrag" in payload.passthrough_fields
    assert payload.provenance[-1].endswith("sechiba.f90::sechiba_main lines 1013-1019")


def test_diffuco_pft14_enerbil_precall_payload_requires_pft_backgrounds():
    trans_inputs = _pft14_c3_trans_inputs()
    trans = diffuco_trans_co2_c3_pft_explicit(**trans_inputs)
    npts, nvm = 2, 4
    closure = diffuco_pft14_c3_beta_closure_explicit(
        trans_co2=trans,
        pft_index=2,
        humrel=np.ones((npts, nvm), dtype=np.float64),
        qair=np.asarray([0.010, 0.009], dtype=np.float64),
        temp_air=np.asarray([300.0, 296.0], dtype=np.float64),
        qsatt=np.asarray([0.014, 0.013], dtype=np.float64),
        veget=np.ones((npts, nvm), dtype=np.float64) * 0.2,
        veget_max=np.ones((npts, nvm), dtype=np.float64) * 0.3,
        lai=np.ones((npts, nvm), dtype=np.float64),
        tot_bare_soil=np.asarray([0.20, 0.25], dtype=np.float64),
        evap_bare_lim=np.asarray([0.25, 0.30], dtype=np.float64),
        vbeta1=np.asarray([0.1, 0.2], dtype=np.float64),
        vbeta2=np.zeros((npts, nvm), dtype=np.float64),
        vbeta3_background=np.zeros((npts, nvm), dtype=np.float64),
        qsintmax=np.ones((npts, nvm), dtype=np.float64) * 0.2,
    )

    with pytest.raises(ValueError, match="missing PFT output backgrounds"):
        diffuco_pft14_enerbil_precall_payload(closure, pft_index=2, pft_output_backgrounds={"gpp": np.zeros((npts, nvm))})


def test_diffuco_pft14_c3_beta_closure_chains_trans_bare_and_comb_no_dew():
    trans_inputs = _pft14_c3_trans_inputs()
    trans = diffuco_trans_co2_c3_pft_explicit(**trans_inputs)
    pft_index = 2
    npts, nvm = 2, 4
    humrel = np.asarray(
        [
            [0.0, 0.6, trans_inputs["humrel"][0], 0.4],
            [0.0, 0.5, trans_inputs["humrel"][1], 0.3],
        ],
        dtype=np.float64,
    )
    veget = np.asarray(
        [
            [0.0, 0.2, trans_inputs["veget"][0], 0.1],
            [0.0, 0.1, trans_inputs["veget"][1], 0.2],
        ],
        dtype=np.float64,
    )
    veget_max = np.asarray(
        [
            [0.0, 0.25, trans_inputs["veget_max"][0], 0.15],
            [0.0, 0.15, trans_inputs["veget_max"][1], 0.25],
        ],
        dtype=np.float64,
    )
    lai = np.asarray(
        [
            [0.0, 0.8, trans_inputs["lai"][0], 0.5],
            [0.0, 0.7, trans_inputs["lai"][1], 0.6],
        ],
        dtype=np.float64,
    )
    vbeta2 = np.asarray([[0.0, 0.03, 0.04, 0.02], [0.0, 0.02, 0.03, 0.01]], dtype=np.float64)
    vbeta3_background = np.asarray([[0.0, 0.05, 0.0, 0.04], [0.0, 0.04, 0.0, 0.03]], dtype=np.float64)
    evap_bare_lim = np.asarray([0.25, 0.30], dtype=np.float64)

    result = diffuco_pft14_c3_beta_closure_explicit(
        trans_co2=trans,
        pft_index=pft_index,
        humrel=humrel,
        qair=np.asarray([0.010, 0.009], dtype=np.float64),
        temp_air=np.asarray([300.0, 296.0], dtype=np.float64),
        qsatt=np.asarray([0.014, 0.013], dtype=np.float64),
        veget=veget,
        veget_max=veget_max,
        lai=lai,
        tot_bare_soil=np.asarray([0.20, 0.25], dtype=np.float64),
        evap_bare_lim=evap_bare_lim,
        vbeta1=np.asarray([0.1, 0.2], dtype=np.float64),
        vbeta2=vbeta2,
        vbeta3_background=vbeta3_background,
        qsintmax=np.asarray([[0.0, 0.2, 0.3, 0.1], [0.0, 0.2, 0.3, 0.1]], dtype=np.float64),
    )

    expected_vbeta3 = vbeta3_background.copy()
    expected_vbeta3[:, pft_index] = np.asarray(trans.output.vbeta3)[:, 0]
    expected_vbeta3pot = np.zeros((npts, nvm), dtype=np.float64)
    expected_vbeta3pot[:, pft_index] = np.asarray(trans.output.vbeta3pot)[:, 0]
    expected_bare = diffuco_bare_cwrr_beta(
        evap_bare_lim=evap_bare_lim,
        vbeta2=vbeta2,
        vbeta3=expected_vbeta3,
        veget_max=veget_max,
    )
    expected_comb = diffuco_comb_explicit(
        humrel=humrel,
        qair=np.asarray([0.010, 0.009], dtype=np.float64),
        temp_air=np.asarray([300.0, 296.0], dtype=np.float64),
        qsatt=np.asarray([0.014, 0.013], dtype=np.float64),
        veget=veget,
        veget_max=veget_max,
        lai=lai,
        tot_bare_soil=np.asarray([0.20, 0.25], dtype=np.float64),
        vbeta1=np.asarray([0.1, 0.2], dtype=np.float64),
        vbeta2=vbeta2,
        vbeta3=expected_vbeta3,
        vbeta4=expected_bare.vbeta4,
        vbeta4_pft=expected_bare.vbeta4_pft,
        qsintmax=np.asarray([[0.0, 0.2, 0.3, 0.1], [0.0, 0.2, 0.3, 0.1]], dtype=np.float64),
    )

    np.testing.assert_allclose(np.asarray(result.vbeta3), expected_vbeta3)
    np.testing.assert_allclose(np.asarray(result.vbeta3pot), expected_vbeta3pot)
    np.testing.assert_allclose(np.asarray(result.bare.vbeta4), np.asarray(expected_bare.vbeta4))
    np.testing.assert_allclose(np.asarray(result.bare.vbeta4_pft), np.asarray(expected_bare.vbeta4_pft))
    np.testing.assert_allclose(np.asarray(result.comb.vbeta), np.asarray(expected_comb.vbeta))
    np.testing.assert_allclose(np.asarray(result.comb.vbeta_pft), np.asarray(expected_comb.vbeta_pft))


def test_diffuco_inter_explicit_matches_fortran_interception_branches():
    qair = np.asarray([0.010, 0.012], dtype=np.float64)
    qsatt = np.asarray([0.020, 0.016], dtype=np.float64)
    rau = np.asarray([1.2, 1.1], dtype=np.float64)
    u = np.asarray([3.0, 0.05], dtype=np.float64)
    v = np.asarray([4.0, 0.0], dtype=np.float64)
    q_cdrag = np.asarray([0.010, 0.020], dtype=np.float64)
    q_cdrag_pft = np.asarray([[0.015, 0.020, 0.025], [0.011, 0.012, 0.013]], dtype=np.float64)
    humrel = np.asarray([[0.0, 0.6, 0.8], [0.0, 0.0, 0.7]], dtype=np.float64)
    veget = np.asarray([[0.7, 0.5, 0.4], [0.8, 0.3, 0.2]], dtype=np.float64)
    qsintveg = np.asarray([[0.5, 0.02, 0.04], [0.3, 0.02, -0.01]], dtype=np.float64)
    qsintmax = np.asarray([[1.0, 0.10, 0.20], [1.0, 0.10, 0.20]], dtype=np.float64)
    rstruct = np.asarray([[30.0, 20.0, 10.0], [25.0, 15.0, 8.0]], dtype=np.float64)
    ok_laidev = np.asarray([False, False, True])
    dt_sechiba = 1800.0

    result = diffuco_inter_explicit(
        qair=qair,
        qsatt=qsatt,
        rau=rau,
        u=u,
        v=v,
        q_cdrag=q_cdrag,
        q_cdrag_pft=q_cdrag_pft,
        humrel=humrel,
        veget=veget,
        qsintveg=qsintveg,
        qsintmax=qsintmax,
        rstruct=rstruct,
        ok_laidev=ok_laidev,
        dt_sechiba=dt_sechiba,
    )

    speed = np.maximum(0.1, np.sqrt(u * u + v * v))
    zqsvegrap = np.maximum(0.0, qsintveg / qsintmax)
    expected_vbeta2 = np.zeros_like(humrel)
    expected_vbeta23 = np.zeros_like(humrel)
    expected_ziltest = np.zeros_like(humrel)
    expected_zrapp = np.zeros_like(humrel)
    for ji in range(2):
        for jv in range(1, 3):
            if veget[ji, jv] > MIN_SECHIBA and qsintveg[ji, jv] > 0.0:
                drag = q_cdrag_pft[ji, jv] if ok_laidev[jv] else q_cdrag[ji]
                beta = veget[ji, jv] * zqsvegrap[ji, jv] / (1.0 + speed[ji] * drag * rstruct[ji, jv])
                ziltest = dt_sechiba * beta * speed[ji] * q_cdrag[ji] * rau[ji] * (qsatt[ji] - qair[ji])
                expected_ziltest[ji, jv] = ziltest
                if ziltest > 0.0:
                    zrapp = qsintveg[ji, jv] / ziltest
                    expected_zrapp[ji, jv] = zrapp
                    if zrapp < 1.0:
                        if humrel[ji, jv] >= MIN_SECHIBA:
                            expected_vbeta23[ji, jv] = max(beta - beta * zrapp, 0.0)
                        beta = beta * zrapp
                expected_vbeta2[ji, jv] = beta

    np.testing.assert_allclose(np.asarray(result.zqsvegrap), zqsvegrap)
    np.testing.assert_allclose(np.asarray(result.ziltest), expected_ziltest)
    np.testing.assert_allclose(np.asarray(result.zrapp), expected_zrapp)
    np.testing.assert_allclose(np.asarray(result.vbeta2), expected_vbeta2)
    np.testing.assert_allclose(np.asarray(result.vbeta23), expected_vbeta23)
    assert not bool(np.asarray(result.active)[0, 0])
    assert bool(np.asarray(result.limited_by_storage)[0, 1])
    assert bool(np.asarray(result.limited_by_storage)[1, 1]) == bool(expected_zrapp[1, 1] < 1.0)
    assert not bool(np.asarray(result.limited_by_storage)[1, 2])


def test_diffuco_inter_explicit_validates_fortran_boundary_shapes():
    with pytest.raises(ValueError, match="humrel must have shape"):
        diffuco_inter_explicit(
            qair=np.asarray([0.01]),
            qsatt=np.asarray([0.02]),
            rau=np.asarray([1.2]),
            u=np.asarray([1.0]),
            v=np.asarray([0.0]),
            q_cdrag=np.asarray([0.01]),
            q_cdrag_pft=np.asarray([[0.01]]),
            humrel=np.asarray([0.5]),
            veget=np.asarray([[0.5]]),
            qsintveg=np.asarray([[0.01]]),
            qsintmax=np.asarray([[0.1]]),
            rstruct=np.asarray([[20.0]]),
            ok_laidev=np.asarray([False]),
            dt_sechiba=1800.0,
        )


def test_diffuco_pft14_c3_chain_with_inter_matches_manual_source_order_chain():
    trans_inputs = _pft14_c3_trans_inputs()
    pft_index = 2
    npts, nvm = 2, 4
    humrel = np.asarray(
        [[0.0, 0.6, trans_inputs["humrel"][0], 0.4], [0.0, 0.5, trans_inputs["humrel"][1], 0.3]],
        dtype=np.float64,
    )
    veget = np.asarray(
        [[0.0, 0.2, trans_inputs["veget"][0], 0.1], [0.0, 0.1, trans_inputs["veget"][1], 0.2]],
        dtype=np.float64,
    )
    veget_max = np.asarray(
        [[0.0, 0.25, trans_inputs["veget_max"][0], 0.15], [0.0, 0.15, trans_inputs["veget_max"][1], 0.25]],
        dtype=np.float64,
    )
    lai = np.asarray(
        [[0.0, 0.8, trans_inputs["lai"][0], 0.5], [0.0, 0.7, trans_inputs["lai"][1], 0.6]],
        dtype=np.float64,
    )
    qsintveg = np.asarray(
        [[0.0, 0.01, trans_inputs["qsintveg"][0], 0.02], [0.0, 0.02, trans_inputs["qsintveg"][1], 0.01]],
        dtype=np.float64,
    )
    qsintmax = np.asarray(
        [[0.0, 0.2, trans_inputs["qsintmax"][0], 0.1], [0.0, 0.2, trans_inputs["qsintmax"][1], 0.1]],
        dtype=np.float64,
    )
    q_cdrag_pft = np.asarray(
        [[0.010, 0.012, trans_inputs["q_cdrag_pft"][0], 0.014], [0.012, 0.014, trans_inputs["q_cdrag_pft"][1], 0.016]],
        dtype=np.float64,
    )
    ok_laidev = np.asarray([False, False, trans_inputs["ok_laidev"], False])
    rstruct = np.asarray([[20.0, 22.0, trans_inputs["rstruct_const"], 18.0], [21.0, 23.0, trans_inputs["rstruct_const"], 19.0]])
    vbeta3_background = np.asarray([[0.0, 0.05, 0.0, 0.04], [0.0, 0.04, 0.0, 0.03]], dtype=np.float64)
    common_trans_inputs = {
        key: value
        for key, value in trans_inputs.items()
        if key
        not in {
            "qsatt",
            "humrel",
            "veget",
            "veget_max",
            "lai",
            "qsintveg",
            "qsintmax",
            "vbeta23",
            "q_cdrag",
            "q_cdrag_pft",
            "ok_laidev",
            "dt_sechiba",
        }
    }

    result = diffuco_pft14_c3_chain_with_inter_explicit(
        pft_index=pft_index,
        trans_co2_inputs=common_trans_inputs,
        qair=np.asarray([0.010, 0.009], dtype=np.float64),
        qsatt=np.asarray([0.014, 0.013], dtype=np.float64),
        temp_air=np.asarray([300.0, 296.0], dtype=np.float64),
        rau=np.asarray([1.2, 1.1], dtype=np.float64),
        u=np.asarray([1.8, 1.2], dtype=np.float64),
        v=np.asarray([0.5, 0.4], dtype=np.float64),
        q_cdrag=trans_inputs["q_cdrag"],
        q_cdrag_pft=q_cdrag_pft,
        humrel=humrel,
        veget=veget,
        veget_max=veget_max,
        lai=lai,
        qsintveg=qsintveg,
        qsintmax=qsintmax,
        rstruct=rstruct,
        ok_laidev=ok_laidev,
        tot_bare_soil=np.asarray([0.20, 0.25], dtype=np.float64),
        evap_bare_lim=np.asarray([0.25, 0.30], dtype=np.float64),
        vbeta1=np.asarray([0.1, 0.2], dtype=np.float64),
        vbeta3_background=vbeta3_background,
        dt_sechiba=trans_inputs["dt_sechiba"],
    )
    manual_inter = diffuco_inter_explicit(
        qair=np.asarray([0.010, 0.009], dtype=np.float64),
        qsatt=np.asarray([0.014, 0.013], dtype=np.float64),
        rau=np.asarray([1.2, 1.1], dtype=np.float64),
        u=np.asarray([1.8, 1.2], dtype=np.float64),
        v=np.asarray([0.5, 0.4], dtype=np.float64),
        q_cdrag=trans_inputs["q_cdrag"],
        q_cdrag_pft=q_cdrag_pft,
        humrel=humrel,
        veget=veget,
        qsintveg=qsintveg,
        qsintmax=qsintmax,
        rstruct=rstruct,
        ok_laidev=ok_laidev,
        dt_sechiba=trans_inputs["dt_sechiba"],
    )
    manual_trans = diffuco_trans_co2_c3_pft_explicit(
        **common_trans_inputs,
        qsatt=np.asarray([0.014, 0.013], dtype=np.float64),
        humrel=humrel[:, pft_index],
        veget=veget[:, pft_index],
        veget_max=veget_max[:, pft_index],
        lai=lai[:, pft_index],
        qsintveg=qsintveg[:, pft_index],
        qsintmax=qsintmax[:, pft_index],
        vbeta23=np.asarray(manual_inter.vbeta23)[:, pft_index],
        q_cdrag=trans_inputs["q_cdrag"],
        q_cdrag_pft=q_cdrag_pft[:, pft_index],
        ok_laidev=bool(ok_laidev[pft_index]),
        dt_sechiba=trans_inputs["dt_sechiba"],
    )
    manual_closure = diffuco_pft14_c3_beta_closure_explicit(
        trans_co2=manual_trans,
        pft_index=pft_index,
        humrel=humrel,
        qair=np.asarray([0.010, 0.009], dtype=np.float64),
        temp_air=np.asarray([300.0, 296.0], dtype=np.float64),
        qsatt=np.asarray([0.014, 0.013], dtype=np.float64),
        veget=veget,
        veget_max=veget_max,
        lai=lai,
        tot_bare_soil=np.asarray([0.20, 0.25], dtype=np.float64),
        evap_bare_lim=np.asarray([0.25, 0.30], dtype=np.float64),
        vbeta1=np.asarray([0.1, 0.2], dtype=np.float64),
        vbeta2=manual_inter.vbeta2,
        vbeta3_background=vbeta3_background,
        qsintmax=qsintmax,
    )

    np.testing.assert_allclose(np.asarray(result.inter.vbeta2), np.asarray(manual_inter.vbeta2))
    np.testing.assert_allclose(np.asarray(result.trans_co2.output.vbeta3), np.asarray(manual_trans.output.vbeta3))
    np.testing.assert_allclose(np.asarray(result.closure.comb.vbeta), np.asarray(manual_closure.comb.vbeta))
    np.testing.assert_allclose(np.asarray(result.closure.comb.vbeta_pft), np.asarray(manual_closure.comb.vbeta_pft))


def test_diffuco_coupled_drag_boundary_matches_fortran_gcm_drag_branch():
    q_cdrag = np.asarray([0.01, 0.02], dtype=np.float64)
    u = np.asarray([3.0, 0.01], dtype=np.float64)
    v = np.asarray([4.0, 0.02], dtype=np.float64)

    q_cdrag_pft = diffuco_coupled_q_cdrag_pft(q_cdrag, nvm=4)
    expected_pft = np.asarray([[0.01, 0.01, 0.01, 0.01], [0.02, 0.02, 0.02, 0.02]], dtype=np.float64)
    np.testing.assert_allclose(np.asarray(q_cdrag_pft), expected_pft)

    raero = diffuco_raerod_explicit(u=u, v=v, q_cdrag=q_cdrag)
    speed = np.maximum(0.1, np.sqrt(u * u + v * v))
    np.testing.assert_allclose(np.asarray(raero), 1.0 / (q_cdrag * speed))

    boundary = diffuco_coupled_drag_boundary_explicit(u=u, v=v, q_cdrag=q_cdrag, nvm=4)
    np.testing.assert_allclose(np.asarray(boundary.q_cdrag), q_cdrag)
    np.testing.assert_allclose(np.asarray(boundary.q_cdrag_pft), expected_pft)
    np.testing.assert_allclose(np.asarray(boundary.raero), np.asarray(raero))


def _manual_diffuco_aero(
    *,
    u,
    v,
    zlev,
    z0h,
    z0m,
    roughheight,
    roughheight_pft,
    temp_sol,
    temp_sol_pft,
    temp_air,
    qsurf,
    qair,
    snow,
    ok_laidev,
    min_wind=0.1,
    snowcri=1.5,
    ok_snowfact=False,
    rough_dyn=False,
):
    cp_air = 1004.675
    cte_grav = 9.80665
    msmlr_air = 28.964e-3
    msmlr_h2o = 18.02e-3
    retv = msmlr_air / msmlr_h2o - 1.0
    rvtmp2 = (4.0 * msmlr_air) / (3.5 * msmlr_h2o) - 1.0
    cepdu2 = 0.1**2
    ct_karman = 0.41
    cb = 5.0
    cc = 5.0
    cd = 5.0

    npts, nvm = roughheight_pft.shape
    q_cdrag = np.zeros(npts, dtype=np.float64)
    q_cdrag_pft = np.zeros((npts, nvm), dtype=np.float64)
    zri = np.zeros(npts, dtype=np.float64)
    zri_pft = np.zeros((npts, nvm), dtype=np.float64)
    cd_neut = np.zeros(npts, dtype=np.float64)
    cd_tmp = np.zeros(npts, dtype=np.float64)
    cd_tmp_pft = np.zeros((npts, nvm), dtype=np.float64)
    snowfact = np.zeros(npts, dtype=np.float64)

    for ji in range(npts):
        speed = np.sqrt(u[ji] * u[ji] + v[ji] * v[ji])
        zg = zlev[ji] * cte_grav
        zdphi = zg / cp_air
        ztvd = (temp_air[ji] + zdphi / (1.0 + rvtmp2 * qair[ji])) * (1.0 + retv * qair[ji])
        ztvs = temp_sol[ji] * (1.0 + retv * qsurf[ji])
        zdu2 = max(cepdu2, speed**2)
        zri[ji] = max(min(zg * (ztvd - ztvs) / (zdu2 * ztvd), 5.0), -5.0)
        snowfact[ji] = 10.0 if snow[ji] > snowcri and ok_snowfact and not rough_dyn else 1.0
        cd_neut[ji] = ct_karman**2 / (
            np.log((zlev[ji] + roughheight[ji]) / z0m[ji])
            * np.log((zlev[ji] + roughheight[ji]) / z0h[ji])
        )
        if zri[ji] >= 0.0:
            zscf = np.sqrt(1.0 + cd * abs(zri[ji]))
            cd_tmp[ji] = cd_neut[ji] / (1.0 + 3.0 * cb * zri[ji] * zscf)
        else:
            zscf = 1.0 / (
                1.0
                + 3.0
                * cb
                * cc
                * cd_neut[ji]
                * np.sqrt(abs(zri[ji]) * ((zlev[ji] + roughheight[ji]) / z0m[ji] / snowfact[ji]))
            )
            cd_tmp[ji] = cd_neut[ji] * (1.0 - 3.0 * cb * zri[ji] * zscf)
        q_cdrag[ji] = max(cd_tmp[ji], 1.0e-4 / max(speed, min_wind))

        for jv in range(nvm):
            if jv >= 1 and ok_laidev[jv]:
                ztvs_jv = temp_sol_pft[ji, jv] * (1.0 + retv * qsurf[ji])
                zri_pft[ji, jv] = max(min(zg * (ztvd - ztvs_jv) / (zdu2 * ztvd), 5.0), -5.0)
                cd_neut_jv = ct_karman**2 / (
                    np.log((zlev[ji] + roughheight_pft[ji, jv]) / z0m[ji])
                    * np.log((zlev[ji] + roughheight_pft[ji, jv]) / z0h[ji])
                )
                if zri_pft[ji, jv] >= 0.0:
                    zscf_jv = np.sqrt(1.0 + cd * abs(zri_pft[ji, jv]))
                    cd_tmp_pft[ji, jv] = cd_neut_jv / (1.0 + 3.0 * cb * zri_pft[ji, jv] * zscf_jv)
                else:
                    zscf_jv = 1.0 / (
                        1.0
                        + 3.0
                        * cb
                        * cc
                        * cd_neut_jv
                        * np.sqrt(
                            abs(zri_pft[ji, jv])
                            * ((zlev[ji] + roughheight_pft[ji, jv]) / z0m[ji] / snowfact[ji])
                        )
                    )
                    cd_tmp_pft[ji, jv] = cd_neut_jv * (1.0 - 3.0 * cb * zri_pft[ji, jv] * zscf_jv)
                q_cdrag_pft[ji, jv] = max(cd_tmp_pft[ji, jv], 1.0e-4 / max(speed, min_wind))
            else:
                zri_pft[ji, jv] = zri[ji]
                cd_tmp_pft[ji, jv] = cd_tmp[ji]
                q_cdrag_pft[ji, jv] = q_cdrag[ji]

    return {
        "q_cdrag": q_cdrag,
        "q_cdrag_pft": q_cdrag_pft,
        "raero": 1.0 / (q_cdrag * np.maximum(min_wind, np.sqrt(u * u + v * v))),
        "zri": zri,
        "zri_pft": zri_pft,
        "cd_tmp": cd_tmp,
        "cd_tmp_pft": cd_tmp_pft,
        "snowfact": snowfact,
    }


def test_diffuco_aero_explicit_matches_fortran_louis_stability_branches():
    inputs = {
        "u": np.asarray([2.0, 1.5], dtype=np.float64),
        "v": np.asarray([0.5, 0.4], dtype=np.float64),
        "zlev": np.asarray([35.0, 30.0], dtype=np.float64),
        "z0h": np.asarray([0.08, 0.06], dtype=np.float64),
        "z0m": np.asarray([0.20, 0.15], dtype=np.float64),
        "roughheight": np.asarray([4.0, 3.0], dtype=np.float64),
        "roughheight_pft": np.asarray([[0.0, 4.5, 7.0], [0.0, 3.5, 5.0]], dtype=np.float64),
        "temp_sol": np.asarray([295.0, 304.0], dtype=np.float64),
        "temp_sol_pft": np.asarray([[295.0, 296.0, 294.5], [304.0, 306.0, 303.0]], dtype=np.float64),
        "temp_air": np.asarray([300.0, 298.0], dtype=np.float64),
        "qsurf": np.asarray([0.010, 0.012], dtype=np.float64),
        "qair": np.asarray([0.008, 0.010], dtype=np.float64),
        "snow": np.asarray([0.5, 2.0], dtype=np.float64),
        "ok_laidev": np.asarray([False, True, False]),
        "ok_snowfact": True,
        "rough_dyn": False,
    }

    result = diffuco_aero_explicit(**inputs)
    expected = _manual_diffuco_aero(**inputs)

    assert expected["zri"][0] > 0.0
    assert expected["zri"][1] < 0.0
    np.testing.assert_allclose(np.asarray(result.q_cdrag), expected["q_cdrag"])
    np.testing.assert_allclose(np.asarray(result.q_cdrag_pft), expected["q_cdrag_pft"])
    np.testing.assert_allclose(np.asarray(result.raero), expected["raero"])
    np.testing.assert_allclose(np.asarray(result.zri), expected["zri"])
    np.testing.assert_allclose(np.asarray(result.zri_pft), expected["zri_pft"])
    np.testing.assert_allclose(np.asarray(result.cd_tmp), expected["cd_tmp"])
    np.testing.assert_allclose(np.asarray(result.cd_tmp_pft), expected["cd_tmp_pft"])
    np.testing.assert_allclose(np.asarray(result.snowfact), expected["snowfact"])
    np.testing.assert_allclose(np.asarray(result.q_cdrag_pft[:, 0]), np.asarray(result.q_cdrag))
    np.testing.assert_allclose(np.asarray(result.q_cdrag_pft[:, 2]), np.asarray(result.q_cdrag))
    assert np.asarray(result.snowfact)[1] == pytest.approx(10.0)


def test_diffuco_aero_explicit_rough_dyn_disables_snowfact_like_paper_config():
    base = {
        "u": np.asarray([1.5], dtype=np.float64),
        "v": np.asarray([0.4], dtype=np.float64),
        "zlev": np.asarray([30.0], dtype=np.float64),
        "z0h": np.asarray([0.06], dtype=np.float64),
        "z0m": np.asarray([0.15], dtype=np.float64),
        "roughheight": np.asarray([3.0], dtype=np.float64),
        "roughheight_pft": np.asarray([[0.0, 3.5]], dtype=np.float64),
        "temp_sol": np.asarray([304.0], dtype=np.float64),
        "temp_sol_pft": np.asarray([[304.0, 306.0]], dtype=np.float64),
        "temp_air": np.asarray([298.0], dtype=np.float64),
        "qsurf": np.asarray([0.012], dtype=np.float64),
        "qair": np.asarray([0.010], dtype=np.float64),
        "snow": np.asarray([2.0], dtype=np.float64),
        "ok_laidev": np.asarray([False, True]),
        "ok_snowfact": True,
    }

    snow_smoothed = diffuco_aero_explicit(**base, rough_dyn=False)
    paper_config = diffuco_aero_explicit(**base, rough_dyn=True)

    assert np.asarray(snow_smoothed.snowfact)[0] == pytest.approx(10.0)
    assert np.asarray(paper_config.snowfact)[0] == pytest.approx(1.0)
    assert np.asarray(snow_smoothed.q_cdrag)[0] != pytest.approx(np.asarray(paper_config.q_cdrag)[0])


def test_diffuco_aero_explicit_validates_shapes():
    base = {
        "u": np.asarray([1.0], dtype=np.float64),
        "v": np.asarray([0.5], dtype=np.float64),
        "zlev": np.asarray([30.0], dtype=np.float64),
        "z0h": np.asarray([0.05], dtype=np.float64),
        "z0m": np.asarray([0.10], dtype=np.float64),
        "roughheight": np.asarray([2.0], dtype=np.float64),
        "roughheight_pft": np.asarray([[0.0, 2.0]], dtype=np.float64),
        "temp_sol": np.asarray([300.0], dtype=np.float64),
        "temp_sol_pft": np.asarray([[300.0, 301.0]], dtype=np.float64),
        "temp_air": np.asarray([299.0], dtype=np.float64),
        "qsurf": np.asarray([0.010], dtype=np.float64),
        "qair": np.asarray([0.009], dtype=np.float64),
        "snow": np.asarray([0.0], dtype=np.float64),
        "ok_laidev": np.asarray([False, True]),
    }
    with pytest.raises(ValueError, match="v must have shape"):
        diffuco_aero_explicit(**{**base, "v": np.asarray([0.5, 0.6], dtype=np.float64)})
    with pytest.raises(ValueError, match="roughheight_pft must have shape"):
        diffuco_aero_explicit(**{**base, "roughheight_pft": np.asarray([0.0, 2.0], dtype=np.float64)})
    with pytest.raises(ValueError, match="ok_laidev must have shape"):
        diffuco_aero_explicit(**{**base, "ok_laidev": np.asarray([True], dtype=bool)})


def test_diffuco_drag_boundary_explicit_selects_fortran_gcm_or_local_aero_branch():
    u = np.asarray([2.0, 1.5], dtype=np.float64)
    v = np.asarray([0.5, 0.4], dtype=np.float64)
    supplied_q_cdrag = np.asarray([0.01, 0.02], dtype=np.float64)
    gcm = diffuco_drag_boundary_explicit(
        ldq_cdrag_from_gcm=True,
        u=u,
        v=v,
        q_cdrag=supplied_q_cdrag,
        nvm=3,
    )
    expected_gcm = diffuco_coupled_drag_boundary_explicit(u=u, v=v, q_cdrag=supplied_q_cdrag, nvm=3)
    np.testing.assert_allclose(np.asarray(gcm.q_cdrag), np.asarray(expected_gcm.q_cdrag))
    np.testing.assert_allclose(np.asarray(gcm.q_cdrag_pft), np.asarray(expected_gcm.q_cdrag_pft))
    np.testing.assert_allclose(np.asarray(gcm.raero), np.asarray(expected_gcm.raero))

    local_inputs = {
        "u": u,
        "v": v,
        "zlev": np.asarray([35.0, 30.0], dtype=np.float64),
        "z0h": np.asarray([0.08, 0.06], dtype=np.float64),
        "z0m": np.asarray([0.20, 0.15], dtype=np.float64),
        "roughheight": np.asarray([4.0, 3.0], dtype=np.float64),
        "roughheight_pft": np.asarray([[0.0, 4.5, 7.0], [0.0, 3.5, 5.0]], dtype=np.float64),
        "temp_sol": np.asarray([295.0, 304.0], dtype=np.float64),
        "temp_sol_pft": np.asarray([[295.0, 296.0, 294.5], [304.0, 306.0, 303.0]], dtype=np.float64),
        "temp_air": np.asarray([300.0, 298.0], dtype=np.float64),
        "qsurf": np.asarray([0.010, 0.012], dtype=np.float64),
        "qair": np.asarray([0.008, 0.010], dtype=np.float64),
        "snow": np.asarray([0.5, 2.0], dtype=np.float64),
        "ok_laidev": np.asarray([False, True, False]),
        "ok_snowfact": True,
    }
    local = diffuco_drag_boundary_explicit(ldq_cdrag_from_gcm=False, **local_inputs)
    expected_local = diffuco_aero_explicit(**local_inputs)
    np.testing.assert_allclose(np.asarray(local.q_cdrag), np.asarray(expected_local.q_cdrag))
    np.testing.assert_allclose(np.asarray(local.q_cdrag_pft), np.asarray(expected_local.q_cdrag_pft))
    np.testing.assert_allclose(np.asarray(local.raero), np.asarray(expected_local.raero))


def test_diffuco_drag_boundary_explicit_requires_branch_inputs():
    with pytest.raises(ValueError, match="q_cdrag and nvm are required"):
        diffuco_drag_boundary_explicit(ldq_cdrag_from_gcm=True, u=np.asarray([1.0]), v=np.asarray([0.0]), q_cdrag=None)
    with pytest.raises(ValueError, match="missing diffuco_aero inputs"):
        diffuco_drag_boundary_explicit(ldq_cdrag_from_gcm=False, u=np.asarray([1.0]), v=np.asarray([0.0]))


def test_diffuco_coupled_drag_boundary_validates_shapes():
    with pytest.raises(ValueError, match="q_cdrag must have shape"):
        diffuco_coupled_q_cdrag_pft(np.asarray([[0.01]]), nvm=2)
    with pytest.raises(ValueError, match="nvm must be positive"):
        diffuco_coupled_q_cdrag_pft(np.asarray([0.01]), nvm=0)
    with pytest.raises(ValueError, match="v must have shape"):
        diffuco_raerod_explicit(u=np.asarray([1.0, 2.0]), v=np.asarray([1.0]), q_cdrag=np.asarray([0.01, 0.02]))


def test_diffuco_qsatt_explicit_reuses_fortran_qsatcalc_kernel():
    temp_sol = np.asarray([280.25, 295.75], dtype=np.float64)
    pb = np.asarray([1000.0, 990.0], dtype=np.float64)

    np.testing.assert_allclose(
        np.asarray(diffuco_qsatt_explicit(temp_sol=temp_sol, pb=pb)),
        np.asarray(qsat_moisture_qsatcalc(temp_sol, pb)),
    )


def test_diffuco_snow_beta_explicit_matches_fortran_snow_and_nobio_limiters():
    qair = np.asarray([0.010, 0.012], dtype=np.float64)
    qsatt = np.asarray([0.020, 0.016], dtype=np.float64)
    rau = np.asarray([1.2, 1.1], dtype=np.float64)
    u = np.asarray([3.0, 0.05], dtype=np.float64)
    v = np.asarray([4.0, 0.0], dtype=np.float64)
    q_cdrag = np.asarray([0.010, 0.020], dtype=np.float64)
    snow = np.asarray([0.05, 0.40], dtype=np.float64)
    frac_nobio = np.asarray([[0.10, 0.20], [0.05, 0.10]], dtype=np.float64)
    totfrac_nobio = np.asarray([0.30, 0.15], dtype=np.float64)
    snow_nobio = np.asarray([[0.01, 0.50], [0.10, 0.02]], dtype=np.float64)
    frac_snow_veg = np.asarray([0.50, 0.25], dtype=np.float64)
    frac_snow_nobio = np.asarray([[0.40, 0.80], [0.50, 0.25]], dtype=np.float64)
    dt_sechiba = 1800.0

    result = diffuco_snow_beta_explicit(
        qair=qair,
        qsatt=qsatt,
        rau=rau,
        u=u,
        v=v,
        q_cdrag=q_cdrag,
        snow=snow,
        frac_nobio=frac_nobio,
        totfrac_nobio=totfrac_nobio,
        snow_nobio=snow_nobio,
        frac_snow_veg=frac_snow_veg,
        frac_snow_nobio=frac_snow_nobio,
        dt_sechiba=dt_sechiba,
    )

    speed = np.maximum(0.1, np.sqrt(u * u + v * v))
    veg_base = (1.0 - totfrac_nobio) * frac_snow_veg
    veg_subtest = dt_sechiba * veg_base * speed * q_cdrag * rau * (qsatt - qair)
    expected_veg = veg_base.copy()
    expected_veg_zrapp = np.zeros_like(veg_base)
    for ji in range(2):
        if veg_subtest[ji] > 0.0:
            zrapp = snow[ji] / veg_subtest[ji]
            expected_veg_zrapp[ji] = zrapp
            if zrapp < 1.0:
                expected_veg[ji] *= zrapp

    nobio_base = frac_nobio * frac_snow_nobio
    expected_nobio = nobio_base.copy()
    expected_nobio_zrapp = np.zeros_like(nobio_base)
    expected_nobio_subtest = np.zeros_like(nobio_base)
    for jv in range(2):
        for ji in range(2):
            subtest = dt_sechiba * nobio_base[ji, jv] * speed[ji] * q_cdrag[ji] * rau[ji] * (qsatt[ji] - qair[ji])
            expected_nobio_subtest[ji, jv] = subtest
            if subtest > 0.0:
                zrapp = snow_nobio[ji, jv] / subtest
                expected_nobio_zrapp[ji, jv] = zrapp
                if zrapp < 1.0:
                    expected_nobio[ji, jv] *= zrapp
    expected_vbeta1 = expected_veg + expected_nobio.sum(axis=1)

    np.testing.assert_allclose(np.asarray(result.vegetation_base), veg_base)
    np.testing.assert_allclose(np.asarray(result.vegetation_subtest), veg_subtest)
    np.testing.assert_allclose(np.asarray(result.vegetation_zrapp), expected_veg_zrapp)
    np.testing.assert_allclose(np.asarray(result.nobio_subtest), expected_nobio_subtest)
    np.testing.assert_allclose(np.asarray(result.nobio_zrapp), expected_nobio_zrapp)
    np.testing.assert_allclose(np.asarray(result.nobio_add), expected_nobio)
    np.testing.assert_allclose(np.asarray(result.vbeta1), expected_vbeta1)


def test_diffuco_flood_beta_explicit_matches_fortran_flood_limiter():
    qair = np.asarray([0.010, 0.012, 0.011], dtype=np.float64)
    qsatt = np.asarray([0.020, 0.016, 0.009], dtype=np.float64)
    rau = np.asarray([1.2, 1.1, 1.0], dtype=np.float64)
    u = np.asarray([3.0, 0.05, 1.0], dtype=np.float64)
    v = np.asarray([4.0, 0.0, 0.0], dtype=np.float64)
    q_cdrag = np.asarray([0.010, 0.020, 0.030], dtype=np.float64)
    evapot = np.asarray([2.0, 0.0, 3.0], dtype=np.float64)
    evapot_corr = np.asarray([1.0, 5.0, 6.0], dtype=np.float64)
    flood_frac = np.asarray([0.40, 0.30, 0.20], dtype=np.float64)
    flood_res = np.asarray([0.01, 1.0, 1.0], dtype=np.float64)
    dt_sechiba = 1800.0

    result = diffuco_flood_beta_explicit(
        qair=qair,
        qsatt=qsatt,
        rau=rau,
        u=u,
        v=v,
        q_cdrag=q_cdrag,
        evapot=evapot,
        evapot_corr=evapot_corr,
        flood_frac=flood_frac,
        flood_res=flood_res,
        dt_sechiba=dt_sechiba,
    )

    base = flood_frac.copy()
    positive_evapot = evapot > MIN_SECHIBA
    base[positive_evapot] = flood_frac[positive_evapot] * evapot_corr[positive_evapot] / evapot[positive_evapot]
    speed = np.maximum(0.1, np.sqrt(u * u + v * v))
    subtest = dt_sechiba * base * speed * q_cdrag * rau * (qsatt - qair)
    zrapp = np.zeros_like(base)
    expected = base.copy()
    for ji in range(3):
        if subtest[ji] > 0.0:
            zrapp[ji] = flood_res[ji] / subtest[ji]
            if zrapp[ji] < 1.0:
                expected[ji] *= zrapp[ji]

    np.testing.assert_allclose(np.asarray(result.base), base)
    np.testing.assert_allclose(np.asarray(result.subtest), subtest)
    np.testing.assert_allclose(np.asarray(result.zrapp), zrapp)
    np.testing.assert_allclose(np.asarray(result.vbeta5), expected)


def test_diffuco_pft14_c3_beta_process_chain_matches_manual_source_order_chain():
    trans_inputs = _pft14_c3_trans_inputs()
    pft_index = 2
    npts, nvm = 2, 4
    qair = np.asarray([0.010, 0.009], dtype=np.float64)
    qsatt = np.asarray([0.014, 0.013], dtype=np.float64)
    temp_air = np.asarray([300.0, 296.0], dtype=np.float64)
    rau = np.asarray([1.2, 1.1], dtype=np.float64)
    u = np.asarray([1.8, 1.2], dtype=np.float64)
    v = np.asarray([0.5, 0.4], dtype=np.float64)
    humrel = np.asarray(
        [[0.0, 0.6, trans_inputs["humrel"][0], 0.4], [0.0, 0.5, trans_inputs["humrel"][1], 0.3]],
        dtype=np.float64,
    )
    veget = np.asarray(
        [[0.0, 0.2, trans_inputs["veget"][0], 0.1], [0.0, 0.1, trans_inputs["veget"][1], 0.2]],
        dtype=np.float64,
    )
    veget_max = np.asarray(
        [[0.0, 0.25, trans_inputs["veget_max"][0], 0.15], [0.0, 0.15, trans_inputs["veget_max"][1], 0.25]],
        dtype=np.float64,
    )
    lai = np.asarray(
        [[0.0, 0.8, trans_inputs["lai"][0], 0.5], [0.0, 0.7, trans_inputs["lai"][1], 0.6]],
        dtype=np.float64,
    )
    qsintveg = np.asarray(
        [[0.0, 0.01, trans_inputs["qsintveg"][0], 0.02], [0.0, 0.02, trans_inputs["qsintveg"][1], 0.01]],
        dtype=np.float64,
    )
    qsintmax = np.asarray(
        [[0.0, 0.2, trans_inputs["qsintmax"][0], 0.1], [0.0, 0.2, trans_inputs["qsintmax"][1], 0.1]],
        dtype=np.float64,
    )
    q_cdrag_pft = np.asarray(
        [[0.010, 0.012, trans_inputs["q_cdrag_pft"][0], 0.014], [0.012, 0.014, trans_inputs["q_cdrag_pft"][1], 0.016]],
        dtype=np.float64,
    )
    ok_laidev = np.asarray([False, False, trans_inputs["ok_laidev"], False])
    rstruct = np.asarray([[20.0, 22.0, trans_inputs["rstruct_const"], 18.0], [21.0, 23.0, trans_inputs["rstruct_const"], 19.0]])
    vbeta3_background = np.asarray([[0.0, 0.05, 0.0, 0.04], [0.0, 0.04, 0.0, 0.03]], dtype=np.float64)
    common_trans_inputs = {
        key: value
        for key, value in trans_inputs.items()
        if key
        not in {
            "qsatt",
            "humrel",
            "veget",
            "veget_max",
            "lai",
            "qsintveg",
            "qsintmax",
            "vbeta23",
            "q_cdrag",
            "q_cdrag_pft",
            "ok_laidev",
            "dt_sechiba",
        }
    }

    result = diffuco_pft14_c3_beta_process_chain_explicit(
        pft_index=pft_index,
        trans_co2_inputs=common_trans_inputs,
        qair=qair,
        qsatt=qsatt,
        temp_air=temp_air,
        rau=rau,
        u=u,
        v=v,
        q_cdrag=trans_inputs["q_cdrag"],
        q_cdrag_pft=q_cdrag_pft,
        humrel=humrel,
        veget=veget,
        veget_max=veget_max,
        lai=lai,
        qsintveg=qsintveg,
        qsintmax=qsintmax,
        rstruct=rstruct,
        ok_laidev=ok_laidev,
        snow=np.asarray([0.05, 0.40], dtype=np.float64),
        frac_nobio=np.asarray([[0.10, 0.20], [0.05, 0.10]], dtype=np.float64),
        totfrac_nobio=np.asarray([0.30, 0.15], dtype=np.float64),
        snow_nobio=np.asarray([[0.01, 0.50], [0.10, 0.02]], dtype=np.float64),
        frac_snow_veg=np.asarray([0.50, 0.25], dtype=np.float64),
        frac_snow_nobio=np.asarray([[0.40, 0.80], [0.50, 0.25]], dtype=np.float64),
        evapot=np.asarray([2.0, 0.0], dtype=np.float64),
        evapot_corr=np.asarray([1.0, 5.0], dtype=np.float64),
        flood_frac=np.asarray([0.40, 0.30], dtype=np.float64),
        flood_res=np.asarray([0.01, 1.0], dtype=np.float64),
        tot_bare_soil=np.asarray([0.20, 0.25], dtype=np.float64),
        evap_bare_lim=np.asarray([0.25, 0.30], dtype=np.float64),
        vbeta3_background=vbeta3_background,
        dt_sechiba=trans_inputs["dt_sechiba"],
    )
    manual_snow = diffuco_snow_beta_explicit(
        qair=qair,
        qsatt=qsatt,
        rau=rau,
        u=u,
        v=v,
        q_cdrag=trans_inputs["q_cdrag"],
        snow=np.asarray([0.05, 0.40], dtype=np.float64),
        frac_nobio=np.asarray([[0.10, 0.20], [0.05, 0.10]], dtype=np.float64),
        totfrac_nobio=np.asarray([0.30, 0.15], dtype=np.float64),
        snow_nobio=np.asarray([[0.01, 0.50], [0.10, 0.02]], dtype=np.float64),
        frac_snow_veg=np.asarray([0.50, 0.25], dtype=np.float64),
        frac_snow_nobio=np.asarray([[0.40, 0.80], [0.50, 0.25]], dtype=np.float64),
        dt_sechiba=trans_inputs["dt_sechiba"],
    )
    manual_flood = diffuco_flood_beta_explicit(
        qair=qair,
        qsatt=qsatt,
        rau=rau,
        u=u,
        v=v,
        q_cdrag=trans_inputs["q_cdrag"],
        evapot=np.asarray([2.0, 0.0], dtype=np.float64),
        evapot_corr=np.asarray([1.0, 5.0], dtype=np.float64),
        flood_frac=np.asarray([0.40, 0.30], dtype=np.float64),
        flood_res=np.asarray([0.01, 1.0], dtype=np.float64),
        dt_sechiba=trans_inputs["dt_sechiba"],
    )
    manual_chain = diffuco_pft14_c3_chain_with_inter_explicit(
        pft_index=pft_index,
        trans_co2_inputs=common_trans_inputs,
        qair=qair,
        qsatt=qsatt,
        temp_air=temp_air,
        rau=rau,
        u=u,
        v=v,
        q_cdrag=trans_inputs["q_cdrag"],
        q_cdrag_pft=q_cdrag_pft,
        humrel=humrel,
        veget=veget,
        veget_max=veget_max,
        lai=lai,
        qsintveg=qsintveg,
        qsintmax=qsintmax,
        rstruct=rstruct,
        ok_laidev=ok_laidev,
        tot_bare_soil=np.asarray([0.20, 0.25], dtype=np.float64),
        evap_bare_lim=np.asarray([0.25, 0.30], dtype=np.float64),
        vbeta1=manual_snow.vbeta1,
        vbeta3_background=vbeta3_background,
        dt_sechiba=trans_inputs["dt_sechiba"],
    )

    np.testing.assert_allclose(np.asarray(result.snow.vbeta1), np.asarray(manual_snow.vbeta1))
    np.testing.assert_allclose(np.asarray(result.flood.vbeta5), np.asarray(manual_flood.vbeta5))
    np.testing.assert_allclose(np.asarray(result.inter.vbeta2), np.asarray(manual_chain.inter.vbeta2))
    np.testing.assert_allclose(np.asarray(result.closure.comb.vbeta1), np.asarray(manual_chain.closure.comb.vbeta1))
    np.testing.assert_allclose(np.asarray(result.closure.comb.vbeta), np.asarray(manual_chain.closure.comb.vbeta))
    after_payload = diffuco_pft14_after_main_payload(result, pft_index=pft_index)
    assert "vbeta5" in after_payload.local_fields
    np.testing.assert_allclose(np.asarray(after_payload.payload["vbeta5"]), np.asarray(result.flood.vbeta5))
    backgrounds = {name: np.zeros((npts, nvm), dtype=np.float64) for name in ("gpp", "gsmean", "rveget", "rstruct", "cimean")}
    precall_payload = diffuco_pft14_enerbil_precall_payload(
        result,
        pft_index=pft_index,
        pft_output_backgrounds=backgrounds,
    )
    assert "vbeta5" in precall_payload.local_fields
    np.testing.assert_allclose(np.asarray(precall_payload.payload["vbeta5"]), np.asarray(result.flood.vbeta5))


def test_diffuco_pft14_beta_process_chain_can_compute_qsatt_from_temp_sol_and_pb():
    trans_inputs = _pft14_c3_trans_inputs()
    pft_index = 2
    npts, nvm = 2, 4
    temp_sol = np.asarray([280.25, 295.75], dtype=np.float64)
    pb = np.asarray([1000.0, 990.0], dtype=np.float64)
    qsatt = np.asarray(diffuco_qsatt_explicit(temp_sol=temp_sol, pb=pb))
    qair = np.asarray([0.010, 0.009], dtype=np.float64)
    temp_air = np.asarray([300.0, 296.0], dtype=np.float64)
    rau = np.asarray([1.2, 1.1], dtype=np.float64)
    u = np.asarray([1.8, 1.2], dtype=np.float64)
    v = np.asarray([0.5, 0.4], dtype=np.float64)
    humrel = np.asarray(
        [[0.0, 0.6, trans_inputs["humrel"][0], 0.4], [0.0, 0.5, trans_inputs["humrel"][1], 0.3]],
        dtype=np.float64,
    )
    veget = np.asarray(
        [[0.0, 0.2, trans_inputs["veget"][0], 0.1], [0.0, 0.1, trans_inputs["veget"][1], 0.2]],
        dtype=np.float64,
    )
    veget_max = np.asarray(
        [[0.0, 0.25, trans_inputs["veget_max"][0], 0.15], [0.0, 0.15, trans_inputs["veget_max"][1], 0.25]],
        dtype=np.float64,
    )
    lai = np.asarray(
        [[0.0, 0.8, trans_inputs["lai"][0], 0.5], [0.0, 0.7, trans_inputs["lai"][1], 0.6]],
        dtype=np.float64,
    )
    qsintveg = np.asarray(
        [[0.0, 0.01, trans_inputs["qsintveg"][0], 0.02], [0.0, 0.02, trans_inputs["qsintveg"][1], 0.01]],
        dtype=np.float64,
    )
    qsintmax = np.asarray(
        [[0.0, 0.2, trans_inputs["qsintmax"][0], 0.1], [0.0, 0.2, trans_inputs["qsintmax"][1], 0.1]],
        dtype=np.float64,
    )
    q_cdrag_pft = np.asarray(
        [[0.010, 0.012, trans_inputs["q_cdrag_pft"][0], 0.014], [0.012, 0.014, trans_inputs["q_cdrag_pft"][1], 0.016]],
        dtype=np.float64,
    )
    ok_laidev = np.asarray([False, False, trans_inputs["ok_laidev"], False])
    rstruct = np.asarray([[20.0, 22.0, trans_inputs["rstruct_const"], 18.0], [21.0, 23.0, trans_inputs["rstruct_const"], 19.0]])
    vbeta3_background = np.asarray([[0.0, 0.05, 0.0, 0.04], [0.0, 0.04, 0.0, 0.03]], dtype=np.float64)
    common_trans_inputs = {
        key: value
        for key, value in trans_inputs.items()
        if key
        not in {
            "qsatt",
            "humrel",
            "veget",
            "veget_max",
            "lai",
            "qsintveg",
            "qsintmax",
            "vbeta23",
            "q_cdrag",
            "q_cdrag_pft",
            "ok_laidev",
            "dt_sechiba",
            "pb",
        }
    }
    base_kwargs = dict(
        pft_index=pft_index,
        trans_co2_inputs={**common_trans_inputs, "pb": pb},
        qair=qair,
        temp_air=temp_air,
        rau=rau,
        u=u,
        v=v,
        q_cdrag=trans_inputs["q_cdrag"],
        q_cdrag_pft=q_cdrag_pft,
        humrel=humrel,
        veget=veget,
        veget_max=veget_max,
        lai=lai,
        qsintveg=qsintveg,
        qsintmax=qsintmax,
        rstruct=rstruct,
        ok_laidev=ok_laidev,
        snow=np.asarray([0.05, 0.40], dtype=np.float64),
        frac_nobio=np.asarray([[0.10, 0.20], [0.05, 0.10]], dtype=np.float64),
        totfrac_nobio=np.asarray([0.30, 0.15], dtype=np.float64),
        snow_nobio=np.asarray([[0.01, 0.50], [0.10, 0.02]], dtype=np.float64),
        frac_snow_veg=np.asarray([0.50, 0.25], dtype=np.float64),
        frac_snow_nobio=np.asarray([[0.40, 0.80], [0.50, 0.25]], dtype=np.float64),
        evapot=np.asarray([2.0, 0.0], dtype=np.float64),
        evapot_corr=np.asarray([1.0, 5.0], dtype=np.float64),
        flood_frac=np.asarray([0.40, 0.30], dtype=np.float64),
        flood_res=np.asarray([0.01, 1.0], dtype=np.float64),
        tot_bare_soil=np.asarray([0.20, 0.25], dtype=np.float64),
        evap_bare_lim=np.asarray([0.25, 0.30], dtype=np.float64),
        vbeta3_background=vbeta3_background,
        dt_sechiba=trans_inputs["dt_sechiba"],
    )

    explicit = diffuco_pft14_c3_beta_process_chain_explicit(**base_kwargs, qsatt=qsatt)
    computed = diffuco_pft14_c3_beta_process_chain_explicit(**base_kwargs, temp_sol=temp_sol, pb=pb)
    np.testing.assert_allclose(np.asarray(computed.snow.vbeta1), np.asarray(explicit.snow.vbeta1))
    np.testing.assert_allclose(np.asarray(computed.inter.vbeta2), np.asarray(explicit.inter.vbeta2))
    np.testing.assert_allclose(np.asarray(computed.closure.comb.vbeta), np.asarray(explicit.closure.comb.vbeta))

    with pytest.raises(ValueError, match="qsatt is required"):
        diffuco_pft14_c3_beta_process_chain_explicit(**base_kwargs)


def test_diffuco_pft14_beta_process_chain_recomputes_trans_co2_qsatt_from_t2m():
    row = first_active_pft14_diffuco_trans_co2_payload()
    trans_inputs = _server_pft14_trans_inputs(row)
    pft_index = 2
    npts, nvm = 1, 4
    humrel = np.zeros((npts, nvm), dtype=np.float64)
    veget = np.zeros((npts, nvm), dtype=np.float64)
    veget_max = np.zeros((npts, nvm), dtype=np.float64)
    lai = np.zeros((npts, nvm), dtype=np.float64)
    qsintveg = np.zeros((npts, nvm), dtype=np.float64)
    qsintmax = np.ones((npts, nvm), dtype=np.float64)
    q_cdrag_pft = np.full((npts, nvm), row["q_cdrag_pft"], dtype=np.float64)
    rstruct = np.full((npts, nvm), trans_inputs["rstruct_const"], dtype=np.float64)
    for target, source in (
        (humrel, "humrel"),
        (veget, "veget"),
        (veget_max, "veget_max"),
        (lai, "lai"),
        (qsintveg, "qsintveg"),
        (qsintmax, "qsintmax"),
    ):
        target[:, pft_index] = trans_inputs[source]
    common_trans_inputs = {
        key: value
        for key, value in trans_inputs.items()
        if key
        not in {
            "qsatt",
            "humrel",
            "veget",
            "veget_max",
            "lai",
            "qsintveg",
            "qsintmax",
            "vbeta23",
            "q_cdrag",
            "q_cdrag_pft",
            "ok_laidev",
            "dt_sechiba",
        }
    }

    result = diffuco_pft14_c3_beta_process_chain_explicit(
        pft_index=pft_index,
        trans_co2_inputs=common_trans_inputs,
        qair=np.asarray([row["qsurf"]], dtype=np.float64),
        qsatt=np.asarray([row["qsatt"] * 2.0], dtype=np.float64),
        temp_air=np.asarray([row["t2m"]], dtype=np.float64),
        rau=np.asarray([1.2], dtype=np.float64),
        u=np.asarray([row["wind"]], dtype=np.float64),
        v=np.asarray([0.0], dtype=np.float64),
        q_cdrag=np.asarray([row["q_cdrag"]], dtype=np.float64),
        q_cdrag_pft=q_cdrag_pft,
        humrel=humrel,
        veget=veget,
        veget_max=veget_max,
        lai=lai,
        qsintveg=qsintveg,
        qsintmax=qsintmax,
        rstruct=rstruct,
        ok_laidev=np.asarray([False, False, trans_inputs["ok_laidev"], False]),
        snow=np.asarray([0.0], dtype=np.float64),
        frac_nobio=np.zeros((npts, 1), dtype=np.float64),
        totfrac_nobio=np.asarray([0.0], dtype=np.float64),
        snow_nobio=np.zeros((npts, 1), dtype=np.float64),
        frac_snow_veg=np.asarray([0.0], dtype=np.float64),
        frac_snow_nobio=np.zeros((npts, 1), dtype=np.float64),
        evapot=np.asarray([1.0], dtype=np.float64),
        evapot_corr=np.asarray([1.0], dtype=np.float64),
        flood_frac=np.asarray([0.0], dtype=np.float64),
        flood_res=np.asarray([0.0], dtype=np.float64),
        tot_bare_soil=np.asarray([0.0], dtype=np.float64),
        evap_bare_lim=np.asarray([0.0], dtype=np.float64),
        vbeta3_background=np.zeros((npts, nvm), dtype=np.float64),
        dt_sechiba=trans_inputs["dt_sechiba"],
    )

    np.testing.assert_allclose(np.asarray(result.trans_co2.vpd_boundary.fvpd)[0, 0], row["fvpd"], rtol=1.0e-12)
    np.testing.assert_allclose(np.asarray(result.trans_co2.controlled_assimtot)[0], row["assimtot"], rtol=1.0e-12)


def _pft14_local_boundary_inputs(*, ldq_cdrag_from_gcm=False):
    trans_inputs = _pft14_c3_trans_inputs()
    pft_index = 2
    npts, nvm = 2, 4
    temp_sol = np.asarray([280.25, 295.75], dtype=np.float64)
    temp_sol_pft = np.asarray(
        [[280.25, 281.0, 282.0, 280.5], [295.75, 296.5, 297.0, 295.0]],
        dtype=np.float64,
    )
    pb = np.asarray([1000.0, 990.0], dtype=np.float64)
    qair = np.asarray([0.010, 0.009], dtype=np.float64)
    temp_air = np.asarray([300.0, 296.0], dtype=np.float64)
    rau = np.asarray([1.2, 1.1], dtype=np.float64)
    u = np.asarray([1.8, 1.2], dtype=np.float64)
    v = np.asarray([0.5, 0.4], dtype=np.float64)
    humrel = np.asarray(
        [[0.0, 0.6, trans_inputs["humrel"][0], 0.4], [0.0, 0.5, trans_inputs["humrel"][1], 0.3]],
        dtype=np.float64,
    )
    veget = np.asarray(
        [[0.0, 0.2, trans_inputs["veget"][0], 0.1], [0.0, 0.1, trans_inputs["veget"][1], 0.2]],
        dtype=np.float64,
    )
    veget_max = np.asarray(
        [[0.0, 0.25, trans_inputs["veget_max"][0], 0.15], [0.0, 0.15, trans_inputs["veget_max"][1], 0.25]],
        dtype=np.float64,
    )
    lai = np.asarray(
        [[0.0, 0.8, trans_inputs["lai"][0], 0.5], [0.0, 0.7, trans_inputs["lai"][1], 0.6]],
        dtype=np.float64,
    )
    qsintveg = np.asarray(
        [[0.0, 0.01, trans_inputs["qsintveg"][0], 0.02], [0.0, 0.02, trans_inputs["qsintveg"][1], 0.01]],
        dtype=np.float64,
    )
    qsintmax = np.asarray(
        [[0.0, 0.2, trans_inputs["qsintmax"][0], 0.1], [0.0, 0.2, trans_inputs["qsintmax"][1], 0.1]],
        dtype=np.float64,
    )
    ok_laidev = np.asarray([False, False, trans_inputs["ok_laidev"], False])
    common_trans_inputs = {
        key: value
        for key, value in trans_inputs.items()
        if key
        not in {
            "qsatt",
            "humrel",
            "veget",
            "veget_max",
            "lai",
            "qsintveg",
            "qsintmax",
            "vbeta23",
            "q_cdrag",
            "q_cdrag_pft",
            "ok_laidev",
            "dt_sechiba",
            "pb",
        }
    }
    base = dict(
        pft_index=pft_index,
        trans_co2_inputs={**common_trans_inputs, "pb": pb},
        ldq_cdrag_from_gcm=ldq_cdrag_from_gcm,
        u=u,
        v=v,
        q_cdrag=np.asarray(trans_inputs["q_cdrag"], dtype=np.float64) if ldq_cdrag_from_gcm else None,
        nvm=nvm if ldq_cdrag_from_gcm else None,
        zlev=np.asarray([35.0, 30.0], dtype=np.float64),
        z0h=np.asarray([0.08, 0.06], dtype=np.float64),
        z0m=np.asarray([0.20, 0.15], dtype=np.float64),
        roughheight=np.asarray([4.0, 3.0], dtype=np.float64),
        roughheight_pft=np.asarray([[0.0, 4.5, 7.0, 2.0], [0.0, 3.5, 5.0, 2.5]], dtype=np.float64),
        temp_sol=temp_sol,
        temp_sol_pft=temp_sol_pft,
        temp_air=temp_air,
        qsurf=np.asarray(trans_inputs["qsurf"], dtype=np.float64),
        qair=qair,
        pb=pb,
        rau=rau,
        humrel=humrel,
        veget=veget,
        veget_max=veget_max,
        lai=lai,
        qsintveg=qsintveg,
        qsintmax=qsintmax,
        rstruct=np.asarray(
            [[20.0, 22.0, trans_inputs["rstruct_const"], 18.0], [21.0, 23.0, trans_inputs["rstruct_const"], 19.0]],
            dtype=np.float64,
        ),
        ok_laidev=ok_laidev,
        snow=np.asarray([0.05, 0.40], dtype=np.float64),
        frac_nobio=np.asarray([[0.10, 0.20], [0.05, 0.10]], dtype=np.float64),
        totfrac_nobio=np.asarray([0.30, 0.15], dtype=np.float64),
        snow_nobio=np.asarray([[0.01, 0.50], [0.10, 0.02]], dtype=np.float64),
        frac_snow_veg=np.asarray([0.50, 0.25], dtype=np.float64),
        frac_snow_nobio=np.asarray([[0.40, 0.80], [0.50, 0.25]], dtype=np.float64),
        evapot=np.asarray([2.0, 0.0], dtype=np.float64),
        evapot_corr=np.asarray([1.0, 5.0], dtype=np.float64),
        flood_frac=np.asarray([0.40, 0.30], dtype=np.float64),
        flood_res=np.asarray([0.01, 1.0], dtype=np.float64),
        tot_bare_soil=np.asarray([0.20, 0.25], dtype=np.float64),
        evap_bare_lim=np.asarray([0.25, 0.30], dtype=np.float64),
        vbeta3_background=np.asarray([[0.0, 0.05, 0.0, 0.04], [0.0, 0.04, 0.0, 0.03]], dtype=np.float64),
        dt_sechiba=trans_inputs["dt_sechiba"],
        ok_snowfact=True,
        rough_dyn=True,
    )
    return base


def test_diffuco_pft14_local_process_boundary_runs_drag_qsatt_and_beta_chain_in_source_order():
    inputs = _pft14_local_boundary_inputs(ldq_cdrag_from_gcm=False)
    result = diffuco_pft14_local_process_boundary_explicit(**inputs)

    expected_drag = diffuco_drag_boundary_explicit(
        ldq_cdrag_from_gcm=False,
        u=inputs["u"],
        v=inputs["v"],
        zlev=inputs["zlev"],
        z0h=inputs["z0h"],
        z0m=inputs["z0m"],
        roughheight=inputs["roughheight"],
        roughheight_pft=inputs["roughheight_pft"],
        temp_sol=inputs["temp_sol"],
        temp_sol_pft=inputs["temp_sol_pft"],
        temp_air=inputs["temp_air"],
        qsurf=inputs["qsurf"],
        qair=inputs["qair"],
        snow=inputs["snow"],
        ok_laidev=inputs["ok_laidev"],
        ok_snowfact=inputs["ok_snowfact"],
        rough_dyn=inputs["rough_dyn"],
    )
    expected_qsatt = diffuco_qsatt_explicit(temp_sol=inputs["temp_sol"], pb=inputs["pb"])
    expected_chain = diffuco_pft14_c3_beta_process_chain_explicit(
        pft_index=inputs["pft_index"],
        trans_co2_inputs=inputs["trans_co2_inputs"],
        qair=inputs["qair"],
        qsatt=expected_qsatt,
        temp_air=inputs["temp_air"],
        rau=inputs["rau"],
        u=inputs["u"],
        v=inputs["v"],
        q_cdrag=expected_drag.q_cdrag,
        q_cdrag_pft=expected_drag.q_cdrag_pft,
        humrel=inputs["humrel"],
        veget=inputs["veget"],
        veget_max=inputs["veget_max"],
        lai=inputs["lai"],
        qsintveg=inputs["qsintveg"],
        qsintmax=inputs["qsintmax"],
        rstruct=inputs["rstruct"],
        ok_laidev=inputs["ok_laidev"],
        snow=inputs["snow"],
        frac_nobio=inputs["frac_nobio"],
        totfrac_nobio=inputs["totfrac_nobio"],
        snow_nobio=inputs["snow_nobio"],
        frac_snow_veg=inputs["frac_snow_veg"],
        frac_snow_nobio=inputs["frac_snow_nobio"],
        evapot=inputs["evapot"],
        evapot_corr=inputs["evapot_corr"],
        flood_frac=inputs["flood_frac"],
        flood_res=inputs["flood_res"],
        tot_bare_soil=inputs["tot_bare_soil"],
        evap_bare_lim=inputs["evap_bare_lim"],
        vbeta3_background=inputs["vbeta3_background"],
        dt_sechiba=inputs["dt_sechiba"],
    )

    np.testing.assert_allclose(np.asarray(result.drag.q_cdrag), np.asarray(expected_drag.q_cdrag))
    np.testing.assert_allclose(np.asarray(result.drag.q_cdrag_pft), np.asarray(expected_drag.q_cdrag_pft))
    np.testing.assert_allclose(np.asarray(result.qsatt), np.asarray(expected_qsatt))
    np.testing.assert_allclose(np.asarray(result.process_chain.snow.vbeta1), np.asarray(expected_chain.snow.vbeta1))
    np.testing.assert_allclose(np.asarray(result.process_chain.inter.vbeta2), np.asarray(expected_chain.inter.vbeta2))
    np.testing.assert_allclose(np.asarray(result.process_chain.closure.comb.vbeta), np.asarray(expected_chain.closure.comb.vbeta))


def test_diffuco_pft14_local_process_boundary_keeps_explicit_gcm_drag_branch():
    inputs = _pft14_local_boundary_inputs(ldq_cdrag_from_gcm=True)
    result = diffuco_pft14_local_process_boundary_explicit(**inputs)

    expected_drag = diffuco_coupled_drag_boundary_explicit(
        u=inputs["u"],
        v=inputs["v"],
        q_cdrag=inputs["q_cdrag"],
        nvm=inputs["nvm"],
    )
    np.testing.assert_allclose(np.asarray(result.drag.q_cdrag), np.asarray(expected_drag.q_cdrag))
    np.testing.assert_allclose(np.asarray(result.drag.q_cdrag_pft), np.asarray(expected_drag.q_cdrag_pft))
    assert np.asarray(result.drag.q_cdrag_pft)[0, 2] == pytest.approx(inputs["q_cdrag"][0])


def test_diffuco_pft14_local_enerbil_precall_payload_exports_local_boundary_fields():
    inputs = _pft14_local_boundary_inputs(ldq_cdrag_from_gcm=False)
    npts, nvm = inputs["veget"].shape
    backgrounds = {name: np.zeros((npts, nvm), dtype=np.float64) for name in ("gpp", "gsmean", "rveget", "rstruct", "cimean")}

    result = diffuco_pft14_local_enerbil_precall_explicit(
        **inputs,
        pft_output_backgrounds=backgrounds,
        passthrough={"lai": inputs["lai"], "veget_max": inputs["veget_max"]},
    )

    payload = result.payload.payload
    assert "q_cdrag" in result.payload.passthrough_fields
    assert "q_cdrag_pft" in result.payload.passthrough_fields
    assert "qsatt" in result.payload.passthrough_fields
    np.testing.assert_allclose(np.asarray(payload["q_cdrag"]), np.asarray(result.boundary.drag.q_cdrag))
    np.testing.assert_allclose(np.asarray(payload["q_cdrag_pft"]), np.asarray(result.boundary.drag.q_cdrag_pft))
    np.testing.assert_allclose(np.asarray(payload["qsatt"]), np.asarray(result.boundary.qsatt))
    np.testing.assert_allclose(np.asarray(payload["vbeta"]), np.asarray(result.boundary.process_chain.closure.comb.vbeta))
    with pytest.raises(ValueError, match="pass-through fields overlap local DIFFUCO boundary fields"):
        diffuco_pft14_local_enerbil_precall_explicit(
            **inputs,
            pft_output_backgrounds=backgrounds,
            passthrough={"q_cdrag": np.asarray([0.0, 0.0], dtype=np.float64)},
        )


def test_diffuco_pft14_local_enerbil_precall_jit_matches_explicit_path():
    inputs = _pft14_local_boundary_inputs(ldq_cdrag_from_gcm=False)
    npts, nvm = inputs["veget"].shape
    backgrounds = {name: np.zeros((npts, nvm), dtype=np.float64) for name in ("gpp", "gsmean", "rveget", "rstruct", "cimean")}
    trans_inputs = dict(inputs["trans_co2_inputs"])
    trans_inputs["lai_light"] = diffuco_lai_light_table(
        np.asarray([trans_inputs["ext_coeff"]], dtype=np.float64),
        nlai=trans_inputs["nlai"],
        laimax=trans_inputs["laimax"],
        lai_level_depth=trans_inputs["lai_level_depth"],
    )
    inputs = {**inputs, "trans_co2_inputs": trans_inputs}

    explicit = diffuco_pft14_local_enerbil_precall_explicit(
        **inputs,
        pft_output_backgrounds=backgrounds,
        passthrough={"lai": inputs["lai"], "veget_max": inputs["veget_max"]},
    )
    compiled = diffuco_pft14_local_enerbil_precall_explicit(
        **inputs,
        pft_output_backgrounds=backgrounds,
        passthrough={"lai": inputs["lai"], "veget_max": inputs["veget_max"]},
        use_jit=True,
    )

    for name in ("gpp", "gsmean", "rveget", "rstruct", "cimean", "vbeta", "vbeta_pft", "vbeta3"):
        np.testing.assert_allclose(
            np.asarray(compiled.payload.payload[name]),
            np.asarray(explicit.payload.payload[name]),
            rtol=1e-12,
            atol=1e-12,
        )


def test_diffuco_bare_cwrr_beta_uses_scalar_hydrol_evap_bare_lim():
    evap_bare_lim = np.asarray([0.30, 0.90], dtype=np.float64)
    vbeta2 = np.asarray([[0.10, 0.05, 0.00], [0.20, 0.10, 0.05]], dtype=np.float64)
    vbeta3 = np.asarray([[0.20, 0.10, 0.05], [0.30, 0.25, 0.10]], dtype=np.float64)
    veget_max = np.asarray([[0.50, 0.20, 0.00], [0.70, 0.30, 0.10]], dtype=np.float64)

    result = diffuco_bare_cwrr_beta(
        evap_bare_lim=evap_bare_lim,
        vbeta2=vbeta2,
        vbeta3=vbeta3,
        veget_max=veget_max,
    )

    expected_vbeta4 = np.minimum(evap_bare_lim, 1.0 - np.sum(vbeta2 + vbeta3, axis=1))
    expected_pft = np.where(
        veget_max > 0.0,
        np.minimum(evap_bare_lim[:, None], veget_max - (vbeta2 + vbeta3)),
        0.0,
    )
    np.testing.assert_allclose(np.asarray(result.vbeta4), expected_vbeta4)
    np.testing.assert_allclose(np.asarray(result.vbeta4_pft), expected_pft)
    assert np.asarray(result.vbeta4_pft)[0, 2] == pytest.approx(0.0)


def test_diffuco_bare_cwrr_beta_does_not_accept_evap_bare_lim_ns_shape():
    with pytest.raises(ValueError, match="evap_bare_lim"):
        diffuco_bare_cwrr_beta(
            evap_bare_lim=np.asarray([[0.2, 0.3]], dtype=np.float64),
            vbeta2=np.zeros((1, 2), dtype=np.float64),
            vbeta3=np.zeros((1, 2), dtype=np.float64),
            veget_max=np.ones((1, 2), dtype=np.float64),
        )


def test_diffuco_comb_final_beta_bundle_sums_and_clears_below_min_sechiba():
    vbeta2 = np.asarray([[2.0e-9, 1.0e-9], [0.2, 0.1]], dtype=np.float64)
    vbeta3 = np.asarray([[3.0e-9, 1.0e-9], [0.05, 0.15]], dtype=np.float64)
    vbeta4 = np.asarray([1.0e-9, 0.25], dtype=np.float64)

    result = diffuco_comb_final_beta_bundle(vbeta2, vbeta3, vbeta4, min_sechiba=MIN_SECHIBA)

    assert np.allclose(np.asarray(result.valpha), np.ones(2))
    assert np.asarray(result.vbeta)[0] == pytest.approx(0.0)
    assert np.asarray(result.vbeta2)[0].tolist() == [0.0, 0.0]
    assert np.asarray(result.vbeta3)[0].tolist() == [0.0, 0.0]
    assert np.asarray(result.vbeta4)[0] == pytest.approx(0.0)
    assert np.asarray(result.vbeta)[1] == pytest.approx(0.75)
    assert np.allclose(np.asarray(result.vbeta2)[1], vbeta2[1])
    assert np.allclose(np.asarray(result.vbeta3)[1], vbeta3[1])
    assert np.asarray(result.vbeta4)[1] == pytest.approx(vbeta4[1])


def test_diffuco_comb_no_dew_guard_requires_explicit_inactive_proof():
    vbeta2 = np.asarray([[0.2, 0.1]], dtype=np.float64)
    vbeta3 = np.asarray([[0.05, 0.15]], dtype=np.float64)
    vbeta4 = np.asarray([0.25], dtype=np.float64)

    with pytest.raises(ValueError, match="qsatt and qair"):
        diffuco_comb_final_beta_bundle_no_dew(vbeta2, vbeta3, vbeta4)
    with pytest.raises(ValueError, match="dew branch is active"):
        diffuco_comb_final_beta_bundle_no_dew(
            vbeta2,
            vbeta3,
            vbeta4,
            qsatt=np.asarray([0.010], dtype=np.float64),
            qair=np.asarray([0.011], dtype=np.float64),
        )

    result = diffuco_comb_final_beta_bundle_no_dew(
        vbeta2,
        vbeta3,
        vbeta4,
        qsatt=np.asarray([0.012], dtype=np.float64),
        qair=np.asarray([0.011], dtype=np.float64),
    )
    assert np.asarray(result.vbeta)[0] == pytest.approx(0.75)


def test_diffuco_comb_explicit_no_dew_matches_final_bundle_shortcut():
    humrel = np.asarray([[0.6, 0.7]], dtype=np.float64)
    vbeta2 = np.asarray([[0.2, 0.1]], dtype=np.float64)
    vbeta3 = np.asarray([[0.05, 0.15]], dtype=np.float64)
    vbeta4 = np.asarray([0.25], dtype=np.float64)
    vbeta4_pft = np.asarray([[0.10, 0.15]], dtype=np.float64)

    result = diffuco_comb_explicit(
        humrel=humrel,
        qair=np.asarray([0.010], dtype=np.float64),
        temp_air=np.asarray([280.0], dtype=np.float64),
        qsatt=np.asarray([0.012], dtype=np.float64),
        veget=np.asarray([[0.0, 0.4]], dtype=np.float64),
        veget_max=np.asarray([[0.0, 0.5]], dtype=np.float64),
        lai=np.asarray([[0.0, 2.0]], dtype=np.float64),
        tot_bare_soil=np.asarray([0.3], dtype=np.float64),
        vbeta1=np.asarray([0.2], dtype=np.float64),
        vbeta2=vbeta2,
        vbeta3=vbeta3,
        vbeta4=vbeta4,
        vbeta4_pft=vbeta4_pft,
        qsintmax=np.asarray([[0.0, 0.2]], dtype=np.float64),
    )
    shortcut = diffuco_comb_final_beta_bundle(vbeta2, vbeta3, vbeta4, min_sechiba=MIN_SECHIBA)

    np.testing.assert_allclose(np.asarray(result.valpha), np.asarray(shortcut.valpha))
    np.testing.assert_allclose(np.asarray(result.vbeta), np.asarray(shortcut.vbeta))
    np.testing.assert_allclose(np.asarray(result.vbeta2), np.asarray(shortcut.vbeta2))
    np.testing.assert_allclose(np.asarray(result.vbeta3), np.asarray(shortcut.vbeta3))
    np.testing.assert_allclose(np.asarray(result.vbeta4), np.asarray(shortcut.vbeta4))
    np.testing.assert_allclose(np.asarray(result.vbeta1), np.asarray([0.2]))
    np.testing.assert_allclose(np.asarray(result.vbeta_pft), np.zeros_like(vbeta2))
    np.testing.assert_allclose(np.asarray(result.humrel), humrel)
    assert not bool(np.asarray(result.toveg)[0])
    assert not bool(np.asarray(result.tosnow)[0])


def test_diffuco_comb_explicit_warm_dew_rewrites_interception_and_humrel():
    humrel = np.asarray([[0.5, 0.7, 0.9]], dtype=np.float64)
    veget = np.asarray([[0.0, 0.4, 0.2]], dtype=np.float64)
    lai = np.asarray([[0.0, 1.0, 2.0]], dtype=np.float64)
    qsintmax = np.asarray([[0.0, 0.3, 0.4]], dtype=np.float64)
    vbeta4_pft = np.asarray([[0.05, 0.10, 0.20]], dtype=np.float64)
    tot_bare_soil = np.asarray([0.25], dtype=np.float64)

    result = diffuco_comb_explicit(
        humrel=humrel,
        qair=np.asarray([0.012], dtype=np.float64),
        temp_air=np.asarray([280.0], dtype=np.float64),
        qsatt=np.asarray([0.010], dtype=np.float64),
        veget=veget,
        veget_max=np.asarray([[0.0, 0.5, 0.3]], dtype=np.float64),
        lai=lai,
        tot_bare_soil=tot_bare_soil,
        vbeta1=np.asarray([0.2], dtype=np.float64),
        vbeta2=np.asarray([[0.01, 0.02, 0.03]], dtype=np.float64),
        vbeta3=np.asarray([[0.10, 0.20, 0.30]], dtype=np.float64),
        vbeta4=np.asarray([0.15], dtype=np.float64),
        vbeta4_pft=vbeta4_pft,
        qsintmax=qsintmax,
    )

    coeff = np.asarray(diffuco_dew_vegetation_coefficient(lai))
    expected_vbeta2 = np.asarray([[0.0, coeff[0, 1] * veget[0, 1], coeff[0, 2] * veget[0, 2]]])
    expected_vbeta3 = np.zeros_like(humrel)
    expected_vbeta_pft = vbeta4_pft + expected_vbeta2
    expected_vbeta = tot_bare_soil + expected_vbeta2.sum(axis=1)

    assert bool(np.asarray(result.toveg)[0])
    assert not bool(np.asarray(result.tosnow)[0])
    np.testing.assert_allclose(np.asarray(result.vbeta1), np.asarray([0.0]))
    np.testing.assert_allclose(np.asarray(result.vbeta4), tot_bare_soil)
    np.testing.assert_allclose(np.asarray(result.vbeta2), expected_vbeta2)
    np.testing.assert_allclose(np.asarray(result.vbeta3), expected_vbeta3)
    np.testing.assert_allclose(np.asarray(result.humrel), np.zeros_like(humrel))
    np.testing.assert_allclose(np.asarray(result.vbeta_pft), expected_vbeta_pft)
    np.testing.assert_allclose(np.asarray(result.vbeta), expected_vbeta)


def test_diffuco_comb_explicit_freezing_dew_routes_to_snow_beta():
    veget_max = np.asarray([[0.0, 0.5, 0.25]], dtype=np.float64)
    result = diffuco_comb_explicit(
        humrel=np.asarray([[0.5, 0.7, 0.9]], dtype=np.float64),
        qair=np.asarray([0.012], dtype=np.float64),
        temp_air=np.asarray([270.0], dtype=np.float64),
        qsatt=np.asarray([0.010], dtype=np.float64),
        veget=np.asarray([[0.0, 0.4, 0.2]], dtype=np.float64),
        veget_max=veget_max,
        lai=np.asarray([[0.0, 1.0, 2.0]], dtype=np.float64),
        tot_bare_soil=np.asarray([0.25], dtype=np.float64),
        vbeta1=np.asarray([0.2], dtype=np.float64),
        vbeta2=np.asarray([[0.01, 0.02, 0.03]], dtype=np.float64),
        vbeta3=np.asarray([[0.10, 0.20, 0.30]], dtype=np.float64),
        vbeta4=np.asarray([0.15], dtype=np.float64),
        vbeta4_pft=np.asarray([[0.05, 0.10, 0.20]], dtype=np.float64),
        qsintmax=np.asarray([[0.0, 0.3, 0.4]], dtype=np.float64),
    )

    assert not bool(np.asarray(result.toveg)[0])
    assert bool(np.asarray(result.tosnow)[0])
    np.testing.assert_allclose(np.asarray(result.vbeta1), np.asarray([1.0]))
    np.testing.assert_allclose(np.asarray(result.vbeta4), np.asarray([0.0]))
    np.testing.assert_allclose(np.asarray(result.vbeta2), np.zeros_like(veget_max))
    np.testing.assert_allclose(np.asarray(result.vbeta3), np.zeros_like(veget_max))
    np.testing.assert_allclose(np.asarray(result.humrel), np.zeros_like(veget_max))
    np.testing.assert_allclose(np.asarray(result.vbeta_pft), veget_max)
    np.testing.assert_allclose(np.asarray(result.vbeta), np.asarray([0.0]))


def test_diffuco_after_main_trace_cannot_validate_untraced_control_internals():
    payload = first_pft14_diffuco_after_main_payload()
    validation = validate_diffuco_after_main_trace_payload(payload)

    assert validation.parse_and_coverage_ok
    assert not validation.complete
    assert validation.uncovered_internal_diagnostics == ("control_salinity/control_inudate",)
    assert "control_salinity" not in payload
    assert "control_inudate" not in payload
    assert "control_salinity/control_inudate" not in validation.covered_contract_fields
    assert payload["jv"] == 14
    assert payload["salinity"] == pytest.approx(33.227593421936)
