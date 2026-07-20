from __future__ import annotations

import numpy as np
import pytest

from jax_orchidee.sechiba.hydrol import (
    PEAT_MCS,
    PEAT_MCR,
    USDA_MCS,
    USDA_MCR,
    hydrol_var_init,
)


def _inputs(*, npts=1, nstm=3):
    nvm = 14
    nslm = 11
    # Explicit source geometry boundary; values are in metres.
    znh = np.geomspace(0.001, 2.0, nslm)
    dnh = np.empty(nslm)
    dnh[0] = znh[0]
    dnh[1:] = np.diff(znh)
    dlh = np.empty(nslm)
    dlh[:-1] = (dnh[:-1] + dnh[1:]) / 2.0
    dlh[-1] = dnh[-1] / 2.0
    veget_max = np.zeros((npts, nvm))
    veget_max[:, 13] = 1.0
    soiltile = np.zeros((npts, nstm))
    soiltile[:, min(2, nstm - 1)] = 1.0
    return {
        "veget": veget_max.copy(),
        "veget_max": veget_max,
        "soiltile": soiltile,
        "njsc": np.full(npts, 6, dtype=np.int32),
        "znh_m": znh,
        "dnh_m": dnh,
        "dlh_m": dlh,
        "humcste": np.linspace(0.2, 4.0, nvm),
        "altmax": np.zeros((npts, nvm)),
    }


def _integrated_layers(mc, dz):
    out = np.zeros_like(mc)
    out[:, 0, :] = dz[1] * (3.0 * mc[:, 0, :] + mc[:, 1, :]) / 8.0
    for jsl in range(1, mc.shape[1] - 1):
        out[:, jsl, :] = (
            dz[jsl] * (3.0 * mc[:, jsl, :] + mc[:, jsl - 1, :]) / 8.0
            + dz[jsl + 1] * (3.0 * mc[:, jsl, :] + mc[:, jsl + 1, :]) / 8.0
        )
    out[:, -1, :] = dz[-1] * (3.0 * mc[:, -1, :] + mc[:, -2, :]) / 8.0
    return out


def test_pft14_default_initialization_follows_source_order():
    result = hydrol_var_init(**_inputs())

    np.testing.assert_allclose(result.zz, _inputs()["znh_m"] * 1000.0)
    np.testing.assert_allclose(result.mc, 0.3)
    np.testing.assert_allclose(result.mcl, result.mc)
    np.testing.assert_allclose(result.qsintveg, 0.0)
    np.testing.assert_allclose(result.mcs, USDA_MCS[5])
    np.testing.assert_allclose(result.mx_eau_var, 2.0 * 1000.0 * USDA_MCS[5])
    # Lines 4141-4143 only warn here; they do not normalize the ordinary branch.
    assert np.all(np.sum(result.nroot, axis=2) < 1.0)
    np.testing.assert_allclose(result.humcste_use[0, 13], _inputs()["humcste"][13])
    np.testing.assert_allclose(result.drysoil_frac, 0.5)
    np.testing.assert_allclose(result.profil_froz_hydro_ns, 0.0)
    assert result.mineral_tables.mc_lin.shape == (51, 1)
    assert result.peat_tables is None
    assert "lines 3946-4624" in result.provenance[0]


def test_restart_state_and_fortran_layer_integrals_are_preserved():
    inputs = _inputs(npts=2)
    nslm = len(inputs["znh_m"])
    mc = np.linspace(0.12, 0.42, 2 * nslm * 3).reshape(2, nslm, 3)
    mcl = mc - 0.01
    water2infilt = np.asarray([[1.0, 2.0, 3.0], [0.25, 0.5, 0.75]])
    resdist = np.asarray([[0.2, 0.3, 0.5], [0.1, 0.2, 0.7]])
    vegtot_old = np.asarray([0.8, 0.6])
    drysoil = np.asarray([999999.0, 0.2])
    qsintveg = np.arange(28, dtype=float).reshape(2, 14) / 100.0
    humrelv = np.full((2, 14, 3), 0.7)

    result = hydrol_var_init(
        **inputs,
        mc=mc,
        mcl=mcl,
        water2infilt=water2infilt,
        resdist=resdist,
        vegtot_old=vegtot_old,
        drysoil_frac=drysoil,
        qsintveg=qsintveg,
        humrelv=humrelv,
    )
    integrated = _integrated_layers(mc, np.asarray(result.dz))
    expected_tmc = np.sum(integrated, axis=1) + water2infilt
    expected_soilmoist = np.sum(integrated * resdist[:, None, :], axis=2) * vegtot_old[:, None]

    np.testing.assert_allclose(result.mcl, mcl)
    np.testing.assert_allclose(result.tmc, expected_tmc)
    np.testing.assert_allclose(result.soilmoist, expected_soilmoist)
    np.testing.assert_allclose(result.tmc_litter, np.sum(integrated[:, :4, :], axis=1))
    np.testing.assert_allclose(result.tmc_trampling, np.sum(integrated[:, :6, :], axis=1))
    np.testing.assert_allclose(result.drysoil_frac, drysoil)
    np.testing.assert_allclose(result.qsintveg, qsintveg)
    np.testing.assert_allclose(result.humrelv[:, 0, :], 0.0)
    np.testing.assert_allclose(result.humrelv[:, 1:, :], 0.7)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("CWRR_NKS_N0", -1.0),
        ("CWRR_NKS_POWER", -1.0),
        ("CWRR_AKS_A0", -1.0),
        ("KFACT_DECAY_RATE", -1.0),
        ("KFACT_STARTING_DEPTH", 0.0),
        ("KFACT_MAX", 9.99),
    ],
)
def test_parameter_checks_match_fortran_conditions(key, value):
    with pytest.raises(ValueError):
        hydrol_var_init(**_inputs(), config_values={key: value})


def test_negative_aks_power_is_accepted_because_fortran_checks_nk_rel():
    result = hydrol_var_init(**_inputs(), config_values={"CWRR_AKS_POWER": -0.25})
    assert result.parameters["CWRR_AKS_POWER"] == -0.25


def test_peat_tile_thresholds_capacity_and_diagnostics_follow_active_branches():
    inputs = _inputs(nstm=6)
    inputs["soiltile"][:] = np.asarray([0.1, 0.1, 0.1, 0.2, 0.25, 0.25])
    inputs["veget"][:] = inputs["veget_max"]
    result = hydrol_var_init(**inputs, peat_hydro=True, tides=True, agri_peat=True)

    np.testing.assert_allclose(result.tmcs[0, :3], 2.0 * 1000.0 * USDA_MCS[5])
    np.testing.assert_allclose(result.tmcs[0, 3:], 2.0 * 1000.0 * PEAT_MCS)
    np.testing.assert_allclose(result.tmcr[0, :3], 2.0 * 1000.0 * USDA_MCR[5])
    np.testing.assert_allclose(result.tmcr[0, 3:], 2.0 * 1000.0 * PEAT_MCR)
    expected_capacity = 2.0 * 1000.0 * (USDA_MCS[5] * 0.3 + PEAT_MCS * 0.7)
    np.testing.assert_allclose(result.mx_eau_var, expected_capacity)
    np.testing.assert_allclose(result.shumdiag_peat, result.shumdiag_croppeat)
    np.testing.assert_allclose(result.shumdiag_peat, result.shumdiag_man)
    assert result.peat_tables is not None


def test_refsoc_thresholds_are_applied_before_mineral_table_build():
    result = hydrol_var_init(
        **_inputs(),
        use_refSOC_hydrol=True,
        refSOC_1d=np.asarray([130000.0]),
    )
    np.testing.assert_allclose(result.mcs, 0.92)
    np.testing.assert_allclose(result.mineral_tables.mc_lin[-1], result.mcs)
    dz = np.asarray(result.dz)
    litter_depth = dz[1] / 2.0 + np.sum((dz[1:4] + dz[2:5]) / 2.0)
    np.testing.assert_allclose(result.tmc_litter_sat, 0.92 * litter_depth)


def test_permafrost_alt_branch_zeros_deep_roots_then_normalizes():
    inputs = _inputs()
    inputs["altmax"][:, 13] = 0.2
    result = hydrol_var_init(**inputs, ok_pc=True)

    active = inputs["znh_m"] >= 0.2
    np.testing.assert_allclose(result.nroot[0, 13, active], 0.0)
    np.testing.assert_allclose(np.sum(result.nroot[0, 13]), 1.0, atol=1.0e-12)
