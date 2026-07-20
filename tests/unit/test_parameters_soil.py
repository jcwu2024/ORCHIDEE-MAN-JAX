from __future__ import annotations

import numpy as np
import pytest

from jax_orchidee.parameters.soil import SoilParameterError, SoilParameters, config_soil_parameters


def test_impose_gate_and_read_order_recompute_snow_capacity():
    initial = SoilParameters(sn_dens=300.0, sn_capa=123.0)
    skipped = config_soil_parameters(
        {"SNOW_DENSITY": 400.0, "TAU_PEAT": 7.0}, ok_sechiba=False,
        impose_param=True, hydrol_cwrr=True, initial=initial,
    )
    assert (skipped.sn_dens, skipped.sn_capa, skipped.tau_peat) == (300.0, 123.0, 7.0)
    applied = config_soil_parameters(
        {"SNOW_DENSITY": 400.0}, ok_sechiba=True,
        impose_param=True, hydrol_cwrr=True, initial=initial,
    )
    assert (applied.sn_dens, applied.sn_capa) == (400.0, 840000.0)


@pytest.mark.parametrize("key,value", [
    ("DRY_SOIL_HEAT_CAPACITY", 0.0), ("DRY_SOIL_HEAT_COND", -1.0),
    ("WET_SOIL_HEAT_CAPACITY", 0.0), ("WET_SOIL_HEAT_COND", 0.0),
    ("SNOW_HEAT_COND", 0.0), ("SNOW_DENSITY", 0.0),
    ("NOBIO_WATER_CAPAC_VOLUMETRI", 0.0), ("SECHIBA_QSINT", 0.0),
    ("SOIL_LAYERS_DISCRE_METHOD", 3),
])
def test_imposed_parameter_guards(key, value):
    with pytest.raises(SoilParameterError):
        config_soil_parameters({key: value}, ok_sechiba=True, impose_param=True, hydrol_cwrr=True)


@pytest.mark.parametrize("overrides", [
    {"CHOISNEL_DIFF_MIN": 0.0},
    {"CHOISNEL_DIFF_MIN": 0.2, "CHOISNEL_DIFF_MAX": 0.2},
    {"CHOISNEL_DIFF_EXP": 0.0}, {"CHOISNEL_RSOL_CSTE": 0.0}, {"HCRIT_LITTER": 0.0},
])
def test_choisnel_guards_are_only_active_outside_cwrr(overrides):
    with pytest.raises(SoilParameterError):
        config_soil_parameters(overrides, ok_sechiba=True, impose_param=True, hydrol_cwrr=False)
    config_soil_parameters(overrides, ok_sechiba=True, impose_param=True, hydrol_cwrr=True)


def test_freeze_dependent_defaults_then_specific_overrides():
    state = config_soil_parameters(
        {
            "OK_FREEZE": "y", "READ_REFTEMP": "n", "OK_FREEZE_THERMIX": "y",
            "OK_ECORR": "n", "OK_SNOWFACT": "n", "OK_FREEZE_CWRR": "y",
            "OK_THERMODYNAMICAL_FREEZING": "n", "SMCMAX_FAO__00002": 0.5,
            "CHECK_CWRR": "y",
        },
        ok_sechiba=False, impose_param=False, hydrol_cwrr=True,
    )
    assert not state.read_reftemp and state.ok_freeze_thermix and not state.ok_ecorr
    assert not state.ok_snowfact and state.ok_freeze_cwrr
    assert not state.ok_thermodynamical_freezing
    assert state.smcmax_fao == (0.41, 0.5, 0.41)
    assert state.check_cwrr and not state.check_cwrr2


def test_ecorr_cross_guard_and_thermodynamical_value_preservation():
    with pytest.raises(SoilParameterError, match="OK_ECORR"):
        config_soil_parameters(
            {"OK_ECORR": "y", "OK_FREEZE_THERMIX": "n"},
            ok_sechiba=False, impose_param=False, hydrol_cwrr=True,
        )
    initial = SoilParameters(ok_thermodynamical_freezing=False)
    state = config_soil_parameters(
        {"OK_FREEZE_CWRR": "n", "OK_THERMODYNAMICAL_FREEZING": "y"},
        ok_sechiba=False, impose_param=False, hydrol_cwrr=True, initial=initial,
    )
    assert not state.ok_thermodynamical_freezing


def test_fortran_d_exponents_and_strict_integer_reading():
    state = config_soil_parameters(
        {"TAU_PEAT": "3.1536D8", "SMCMAX_FAO": "4.1d-1, 4.3D-1, 4.1D-1"},
        ok_sechiba=False, impose_param=False, hydrol_cwrr=True,
    )
    assert state.tau_peat == 3.1536e8
    assert state.smcmax_fao == (0.41, 0.43, 0.41)
    with pytest.raises(ValueError, match="must be an integer"):
        config_soil_parameters(
            {"SOIL_LAYERS_DISCRE_METHOD": "1.5"},
            ok_sechiba=True, impose_param=True, hydrol_cwrr=True,
        )
    array_state = config_soil_parameters(
        {"SMCMAX_FAO": np.asarray([0.4, 0.5, 0.6])},
        ok_sechiba=False, impose_param=False, hydrol_cwrr=True,
    )
    assert array_state.smcmax_fao == (0.4, 0.5, 0.6)
