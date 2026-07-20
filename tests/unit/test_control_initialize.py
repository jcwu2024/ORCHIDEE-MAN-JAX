from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jax_orchidee.parameters.control import (
    ControlInitializationError,
    control_initialize,
    paper_control_initialize,
)


ROOT = Path(__file__).resolve().parents[2]
USED_RUN_DEF = ROOT / "outputs/server_1961_trace_full_20260623/run/used_run.def"


def test_paper_control_configuration_and_cwrr_diaglev_are_source_backed():
    state = paper_control_initialize(USED_RUN_DEF)

    assert state.dt_sechiba == 1800.0
    assert (state.soil_classif, state.nscm) == ("usda", 12)
    assert state.hydrol_cwrr and state.river_routing and state.ok_leak
    assert state.ok_stomate and state.ok_co2
    assert not state.ok_pc and not state.ok_dgvm and not state.ok_bvoc
    assert state.nvm == 14 and state.nslm == state.diaglev.size
    assert state.diaglev[-1] < state.zmaxh
    assert state.parameter_initializers == (
        "pft_parameters_main",
        "activate_sub_models",
        "veget_config",
        "config_pft_parameters",
        "config_sechiba_parameters",
        "config_sechiba_pft_parameters",
        "config_soil_parameters",
        "config_co2_parameters",
        "config_stomate_parameters",
        "config_stomate_pft_parameters",
    )


@pytest.mark.parametrize("soil_classif,nscm", [("zobler", 3), ("fao", 3), ("none", 3), ("usda", 12)])
def test_soil_classification_select_case_has_only_source_legal_values(soil_classif, nscm):
    state = control_initialize(
        {"SOILTYPE_CLASSIF": soil_classif},
        dt=1800.0,
        ok_freeze_thermix=False,
    )
    assert state.nscm == nscm


def test_illegal_soil_classification_is_not_approximated():
    with pytest.raises(ControlInitializationError, match="unsupported SOILTYPE_CLASSIF"):
        control_initialize({"SOILTYPE_CLASSIF": "soilgrids"}, dt=1800.0)


def test_routing_gates_conditional_reads_and_rotation_applies_source_forcing():
    state = control_initialize(
        {
            "RIVER_ROUTING": "n",
            "DO_IRRIGATION": "y",
            "DO_FLOODPLAINS": "y",
            "OK_ROTATE": "n",
            "DYN_PLNTDT": "y",
            "CYC_ROT_MAX": "3",
        },
        dt=1800.0,
    )
    assert not state.do_irrigation and not state.do_floodplains
    assert state.dyn_plntdt and state.cyc_rot_max == 1

    rotating = control_initialize(
        {"OK_ROTATE": "y", "DYN_PLNTDT": "y", "CYC_ROT_MAX": "3"},
        dt=1800.0,
    )
    assert not rotating.dyn_plntdt and rotating.cyc_rot_max == 3


def test_consistency_rules_follow_fortran_order():
    state = control_initialize(
        {
            "OK_LEAK": "y",
            "OK_PC": "y",
            "STOMATE_OK_STOMATE": "n",
            "STOMATE_OK_CO2": "n",
            "STOMATE_OK_DGVM": "y",
            "CANOPY_MULTILAYER": "y",
            "CANOPY_EXTINCTION": "n",
        },
        dt=1800.0,
    )
    assert not state.ok_pc
    assert state.ok_stomate and not state.ok_co2
    assert state.ok_radcanopy
    assert "config_dgvm_parameters" in state.parameter_initializers

    with pytest.raises(ControlInitializationError, match="cannot both be true"):
        control_initialize(
            {"STOMATE_OK_DGVM": "y", "OK_ROTATE": "y"},
            dt=1800.0,
        )


def test_freezing_guards_reject_invalid_cwrr_and_choisnel_structures():
    with pytest.raises(ControlInitializationError, match="DEPTH_MAX_T >= 11"):
        control_initialize(
            {"HYDROL_CWRR": "y"},
            dt=1800.0,
            znt=np.array([0.1, 0.5, 1.5]),
            nslm=3,
            zmaxt=10.0,
            ok_freeze_thermix=True,
        )

    with pytest.raises(ControlInitializationError, match="THERMOSOIL_NBLEV >= 11"):
        control_initialize(
            {"HYDROL_CWRR": "n", "THERMOSOIL_NBLEV": "7"},
            dt=1800.0,
            ok_freeze_thermix=True,
        )


def test_choisnel_diaglev_and_parameter_call_guards_match_source():
    state = control_initialize(
        {
            "HYDROL_CWRR": "n",
            "DEPTH_MAX_H": "4",
            "THERMOSOIL_NBLEV": "11",
            "IMPOSE_PARAM": "n",
            "STOMATE_OK_STOMATE": "y",
        },
        dt=1800.0,
        ok_freeze_thermix=True,
    )
    assert state.nslm == 11 and state.ngrnd == 11
    np.testing.assert_allclose(state.diaglev[-1], 4.0)
    np.testing.assert_array_less(state.diaglev[:-1], state.diaglev[1:])
    assert state.parameter_initializers == (
        "pft_parameters_main",
        "activate_sub_models",
        "veget_config",
        "config_soil_parameters",
    )


def test_invalid_literal_values_fail_at_the_control_boundary():
    with pytest.raises(ControlInitializationError, match="invalid OK_LEAK boolean"):
        control_initialize({"OK_LEAK": "sometimes"}, dt=1800.0)
    with pytest.raises(ControlInitializationError, match="invalid NVM integer"):
        control_initialize({"NVM": "14.5"}, dt=1800.0)


def test_fortran_d_exponents_are_accepted_for_float_and_integral_controls():
    state = control_initialize(
        {
            "DEPTH_MAX_H": "4D0",
            "THERMOSOIL_NBLEV": "11d0",
            "NVM": "1.4D1",
        },
        dt=1800.0,
    )

    assert state.zmaxh == 4.0
    assert state.ngrnd == 11
    assert state.nvm == 14


@pytest.mark.parametrize("value", ["1.45D1", "NaN", "Inf", True])
def test_integer_controls_reject_nonintegral_or_nonfinite_fortran_tokens(value):
    with pytest.raises(ControlInitializationError, match="invalid NVM integer"):
        control_initialize({"NVM": value}, dt=1800.0)


@pytest.mark.parametrize("value", ["NaN", "-Inf", False])
def test_float_controls_reject_nonfinite_or_boolean_tokens(value):
    with pytest.raises(ControlInitializationError, match="invalid DEPTH_MAX_H number"):
        control_initialize({"DEPTH_MAX_H": value}, dt=1800.0)
