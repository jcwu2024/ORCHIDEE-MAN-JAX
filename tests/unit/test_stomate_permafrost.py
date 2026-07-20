from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from jax_orchidee.stomate.permafrost import (
    FORTRAN_EPSILON_R4,
    MICROACTEM_PROVENANCE,
    STOMATE_TAU_SECONDS,
    microactem,
    microactem_moisture_control,
    microactem_temperature_control,
    moyano_peat_moisture_lookup,
    stomate_permafrost_decomposition_controls,
)


def _peat_lookup_expected(values):
    mc = 0.01 + 0.02 * np.arange(45, dtype=np.float64)
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
    sr = np.cumprod(pcsr)
    sr = sr / np.max(sr)
    ind = int(np.argmax(sr))
    corgmat = sr.copy()
    corgmat[: ind + 1] = corgmat[: ind + 1] - np.min(corgmat[: ind + 1])
    corgmat[: ind + 1] = corgmat[: ind + 1] / np.max(corgmat[: ind + 1])
    indices = np.minimum(44, np.maximum(0, np.floor(np.asarray(values) / 0.02).astype(np.int32)))
    return np.minimum(1.0, np.maximum(np.finfo(np.float32).eps, corgmat[indices]))


def test_microactem_temperature_control_matches_frozen_respiration_branches():
    temp = np.asarray([-4.0, -0.5, 0.0, 10.0, 30.0, 40.0], dtype=np.float64)

    branch0 = np.asarray(microactem_temperature_control(temp, 0))
    branch1 = np.asarray(microactem_temperature_control(temp, 1))
    branch2 = np.asarray(microactem_temperature_control(temp, 2))
    branch3 = np.asarray(microactem_temperature_control(temp, 3))
    branch4 = np.asarray(microactem_temperature_control(temp, 4))

    normal = np.exp(np.log(2.0) * (temp - 30.0) / 10.0)
    assert np.allclose(branch0, np.maximum(np.minimum(1.0, normal), np.finfo(np.float32).eps))
    assert branch1[0] == pytest.approx(float(FORTRAN_EPSILON_R4))
    assert branch1[1] == pytest.approx(0.5 * np.exp(np.log(2.0) * -30.0 / 10.0))
    assert branch2[0] == pytest.approx(float(FORTRAN_EPSILON_R4))
    assert branch2[1] == pytest.approx((2.5 / 3.0) * np.exp(np.log(2.0) * -30.0 / 10.0))
    assert branch3[0] == pytest.approx(np.exp(np.log(100.0) * -4.0 / 10.0) * np.exp(np.log(2.0) * -30.0 / 10.0))
    assert branch4[0] == pytest.approx(np.exp(np.log(1000.0) * -4.0 / 10.0) * np.exp(np.log(2.0) * -30.0 / 10.0))
    assert branch0[-1] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="frozen_respiration_func"):
        microactem_temperature_control(temp, 99)


def test_microactem_moisture_control_matches_standard_and_doc_branches():
    moist = np.asarray([0.0, 0.5, 1.0, 2.0], dtype=np.float64)

    standard = np.asarray(microactem_moisture_control(moist, 0))
    doc = np.asarray(microactem_moisture_control(moist, 1))

    expected = np.maximum(0.25, np.minimum(1.0, -1.1 * moist * moist + 2.4 * moist - 0.29))
    assert np.allclose(standard, expected)
    assert np.allclose(doc, np.ones_like(moist))
    with pytest.raises(ValueError, match="limit_decomp_moisture"):
        microactem_moisture_control(moist, 7)


def test_moyano_peat_moisture_lookup_matches_fortran_table_formula():
    values = np.asarray([0.0, 0.01, 0.02, 0.54, 0.9, 1.5], dtype=np.float64)

    result = np.asarray(moyano_peat_moisture_lookup(values))

    assert np.allclose(result, _peat_lookup_expected(values))


def test_microactem_nonpeat_time_constant_uses_stomate_tau_temperature_and_moisture():
    temp = np.asarray([[[10.0, 30.0]]], dtype=np.float64)
    moist = np.asarray([[[0.5, 1.0]]], dtype=np.float64)

    result = microactem(
        temp,
        frozen_respiration_func=0,
        limit_decomp_moisture=0,
        moist_in=moist,
        zi_soil=np.asarray([0.1], dtype=np.float64),
        mc_peat=np.asarray([[0.5]], dtype=np.float64),
    )

    temp_control = np.maximum(np.minimum(1.0, np.exp(np.log(2.0) * (temp - 30.0) / 10.0)), np.finfo(np.float32).eps)
    moist_control = np.maximum(0.25, np.minimum(1.0, -1.1 * moist * moist + 2.4 * moist - 0.29))
    expected = STOMATE_TAU_SECONDS / (temp_control * moist_control)
    assert np.allclose(np.asarray(result.fbact_seconds), expected)
    assert result.peat_moisture_control is None


def test_microactem_peat_path_uses_explicit_is_peat_tau_and_depth_cap():
    temp = np.ones((1, 13, 3), dtype=np.float64) * 30.0
    moist = np.ones_like(temp) * 0.5
    zi_soil = np.linspace(0.1, 1.3, 13, dtype=np.float64)
    mc_peat = np.ones((1, 13), dtype=np.float64) * 0.54
    is_peat = np.asarray([False, True, False])

    result = microactem(
        temp,
        frozen_respiration_func=0,
        limit_decomp_moisture=1,
        moist_in=moist,
        zi_soil=zi_soil,
        mc_peat=mc_peat,
        perma_peat=True,
        is_peat=is_peat,
        tau_peat=100.0,
        z_tau=2.0,
    )

    peat_moisture = _peat_lookup_expected(mc_peat)
    expected_tau = 100.0 * np.exp(zi_soil / 2.0)
    expected_tau[12] = 100.0 * np.exp(zi_soil[11] / 2.0)
    fbact = np.asarray(result.fbact_seconds)
    assert np.allclose(fbact[0, :, 1], expected_tau / peat_moisture[0, :])
    assert np.allclose(fbact[0, :, 0], STOMATE_TAU_SECONDS)
    assert np.allclose(fbact[0, :, 2], STOMATE_TAU_SECONDS)
    assert np.asarray(result.peat_moisture_control).shape == (1, 13, 3)
    with pytest.raises(ValueError, match="is_peat"):
        microactem(
            temp,
            frozen_respiration_func=0,
            limit_decomp_moisture=1,
            moist_in=moist,
            zi_soil=zi_soil,
            mc_peat=mc_peat,
            perma_peat=True,
        )


def test_microactem_agri_peat_flux_coeff_applies_to_fortran_pfts_15_and_16_only():
    temp = np.ones((1, 1, 16), dtype=np.float64) * 30.0
    moist = np.ones_like(temp) * 0.54
    zi_soil = np.asarray([0.1], dtype=np.float64)
    mc_peat = np.asarray([[0.54]], dtype=np.float64)

    result = microactem(
        temp,
        frozen_respiration_func=0,
        limit_decomp_moisture=0,
        moist_in=moist,
        zi_soil=zi_soil,
        mc_peat=mc_peat,
        perma_peat=False,
        agri_peat=True,
        flux_tot_coeff=(1.2, 1.4, 0.75),
    )

    fbact = np.asarray(result.fbact_seconds)
    base = STOMATE_TAU_SECONDS / np.asarray(microactem_moisture_control(moist, 0))
    assert fbact[0, 0, 13] == pytest.approx(base[0, 0, 13])
    assert fbact[0, 0, 14] == pytest.approx(base[0, 0, 14] / 1.2)
    assert fbact[0, 0, 15] == pytest.approx(base[0, 0, 15] / 1.2)


def test_stomate_permafrost_decomposition_controls_follow_stomate_main_conversion():
    temp = np.ones((2, 1, 2), dtype=np.float64) * 30.0
    moist = np.ones_like(temp) * 0.5
    zi_soil = np.asarray([0.1], dtype=np.float64)
    mc_peat = np.ones((2, 1), dtype=np.float64) * 0.5
    poor_soils = np.asarray([0.0, 0.5], dtype=np.float64)

    result = stomate_permafrost_decomposition_controls(
        temp,
        moist,
        zi_soil,
        mc_peat,
        poor_soils,
        frozen_respiration_func=0,
        perma_peat=False,
        agri_peat=False,
    )

    fbact = STOMATE_TAU_SECONDS
    moist_control = np.maximum(0.25, np.minimum(1.0, -1.1 * moist * moist + 2.4 * moist - 0.29))
    fbact_doc = STOMATE_TAU_SECONDS / moist_control
    factor = ((2.0 - poor_soils) * 0.5)[:, None, None]
    assert np.allclose(np.asarray(result.prmfrst_soilc_tempctrl), (1.0 / fbact) * 86400.0 * factor)
    assert np.allclose(np.asarray(result.prmfrst_soilc_tempctrl_doc), (1.0 / fbact_doc) * 86400.0 * factor)
    assert any("microactem lines 2468-2703" in item for item in MICROACTEM_PROVENANCE)

