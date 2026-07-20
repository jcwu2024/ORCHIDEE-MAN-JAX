from __future__ import annotations

import re
from pathlib import Path

import pytest

from jax_orchidee.parameters.constantes import (
    PrintLevelReader,
    _SECHIBA_AFTER_WATSTRESS,
    _SECHIBA_BEFORE_IMPAZE,
    _SECHIBA_IMPAZE,
    _STOMATE_BEFORE_CH4,
    _STOMATE_CH4,
    _VEGET_UNCONDITIONAL_1,
    activate_sub_models,
    config_sechiba_parameters,
    config_stomate_parameters,
    veget_config,
)


def _defaults(keys, value=0.0):
    return {key: value for key in keys}


def _fortran_getin_order(procedure: str) -> list[str]:
    source = (Path(__file__).resolve().parents[2] / "fortran_source/ORCHIDEE/src_parameters/constantes.f90").read_text()
    start = source.index(f"SUBROUTINE {procedure}")
    end = source.index(f"END SUBROUTINE {procedure}", start)
    active_source = "\n".join(
        line for line in source[start:end].splitlines() if not line.lstrip().startswith("!")
    )
    return [key.upper() for key in re.findall(r"getin_p\('([^']+)'", active_source, re.IGNORECASE)]


def test_order_tables_match_every_active_fortran_getin_call():
    expected = {
        "veget_config": list(_VEGET_UNCONDITIONAL_1)
        + ["IMPOSE_SOILT", "LAI_MAP", "MAP_PFT_FORMAT", "VEGET_REINIT", "VEGET_YEAR"],
        "config_sechiba_parameters": list(_SECHIBA_BEFORE_IMPAZE)
        + [key for key, _ in _SECHIBA_IMPAZE]
        + ["NEW_WATSTRESS", "ALPHA_WATSTRESS"]
        + list(_SECHIBA_AFTER_WATSTRESS),
        "config_stomate_parameters": list(_STOMATE_BEFORE_CH4)
        + list(_STOMATE_CH4)
        + ["CODESLA", "PERCENT_RESIDUAL"],
    }
    for procedure, jax_order in expected.items():
        assert _fortran_getin_order(procedure) == jax_order


def test_activate_sub_models_guards_and_nested_override_order():
    defaults = {
        "OK_HERBIVORES": False, "TREAT_EXPANSION": False, "HARVEST_AGRI": True,
        "DISABLE_FIRE": False, "ALLOW_DEFOREST_FIRE": False, "SPINUP_ANALYTIC": False,
        "MOIST_FUNC_MOYANO": False, "DANS_RESTART": False, "PRIMING": True,
        "OK_TF_DOC": True, "OK_ROTATE": False,
    }
    result = activate_sub_models(
        defaults,
        {"LPJ_GAP_CONST_MORT": "y", "GLUC_USE_AGE_CLASS": "y", "GLUC_NAGEC_TREE": "3"},
        ok_stomate=True,
        ok_dgvm=True,
    )
    assert result.values["LPJ_GAP_CONST_MORT"] is True
    assert result.values["NAGEC_TREE"] == 3
    assert result.values["NAGEC_HERB"] == 1
    assert result.warnings[0].severity == 1
    assert result.read_order.index("FIRE_DISABLE") < result.read_order.index("ALLOW_DEFOREST_FIRE")
    assert result.read_order.index("GLUC_USE_AGE_CLASS") < result.read_order.index("GLUC_NAGEC_TREE")


def test_activate_sub_models_false_arms_skip_stomate_fire_and_age_reads():
    assert activate_sub_models({}, {}, ok_stomate=False, ok_dgvm=False).read_order == ()
    defaults = {
        "OK_HERBIVORES": False, "TREAT_EXPANSION": False, "HARVEST_AGRI": True,
        "DISABLE_FIRE": True, "SPINUP_ANALYTIC": False, "MOIST_FUNC_MOYANO": False,
        "DANS_RESTART": False, "PRIMING": True, "OK_TF_DOC": True, "OK_ROTATE": False,
    }
    result = activate_sub_models(defaults, {}, ok_stomate=True, ok_dgvm=False)
    assert result.values["LPJ_GAP_CONST_MORT"] is True
    assert "ALLOW_DEFOREST_FIRE" not in result.read_order
    assert "GLUC_NAGEC_TREE" not in result.read_order


@pytest.mark.parametrize("impose,map_format,expected", [(False, False, 18), (True, False, 19), (False, True, 20)])
def test_veget_config_three_arms(impose, map_format, expected):
    defaults = _defaults(_VEGET_UNCONDITIONAL_1, False)
    defaults.update({"READ_LAI": False, "MAP_PFT_FORMAT": map_format})
    defaults["IMPOSE_VEG"] = impose
    if impose:
        defaults["IMPOSE_SOILT"] = False
    if map_format:
        defaults.update({"VEGET_REINIT": True, "VEGET_YEAR_ORIG": 1})
    result = veget_config(defaults, {})
    assert len(result.read_order) == expected


def test_config_sechiba_parameters_preserves_order_derived_value_and_three_arms():
    keys = _SECHIBA_BEFORE_IMPAZE + ("NEW_WATSTRESS",) + _SECHIBA_AFTER_WATSTRESS
    defaults = _defaults(keys)
    defaults.update({"SNOWCRI": 1.5, "IMPOSE_AZE": False, "NEW_WATSTRESS": True, "ALPHA_WATSTRESS": 2.0})
    result = config_sechiba_parameters(defaults, {"SNOWCRI": "2.5D0"})
    assert result.values["SNEIGE"] == pytest.approx(0.0025)
    assert result.read_order.index("SNOWCRI") < result.read_order.index("IRRIG_DOSMAX")
    assert "CONDVEG_Z0" not in result.read_order
    assert result.read_order.index("NEW_WATSTRESS") < result.read_order.index("ALPHA_WATSTRESS")


def test_config_sechiba_impose_aze_requires_only_active_branch_defaults():
    keys = _SECHIBA_BEFORE_IMPAZE + ("NEW_WATSTRESS",) + _SECHIBA_AFTER_WATSTRESS
    defaults = _defaults(keys)
    defaults.update({"IMPOSE_AZE": True, "NEW_WATSTRESS": False})
    with pytest.raises(KeyError, match="Z0_SCAL"):
        config_sechiba_parameters(defaults, {})


def test_config_stomate_ch4_arm_is_structural_and_ordered():
    defaults = _defaults(_STOMATE_BEFORE_CH4 + ("CODESLA", "PRC_RESIDUAL"))
    without = config_stomate_parameters(defaults, {}, ch4_calcul=False)
    assert "NVERT" not in without.read_order
    with_ch4_defaults = {**defaults, **_defaults(_STOMATE_CH4)}
    with_ch4 = config_stomate_parameters(with_ch4_defaults, {"NVERT": "12"}, ch4_calcul=True)
    assert with_ch4.read_order.index("GREEN_AGE_DEC") < with_ch4.read_order.index("NVERT")
    assert with_ch4.read_order.index("ALPHA_CH4") < with_ch4.read_order.index("CODESLA")


def test_get_printlev_first_and_later_arms_match_fortran_saved_state():
    reader = PrintLevelReader()
    assert reader.get_printlev("sechiba", {"PRINTLEV": "2", "PRINTLEV_sechiba": "4"}) == 4
    assert reader.get_printlev("stomate", {"PRINTLEV": "0"}) == 2
    assert reader.get_printlev("sechiba", {"PRINTLEV_sechiba": "3"}) == 3


def test_malformed_override_is_rejected_instead_of_guessed():
    reader = PrintLevelReader()
    with pytest.raises(ValueError, match="not an integer"):
        reader.get_printlev("x", {"PRINTLEV": "1.5"})


def test_structural_flags_and_uninitialized_values_are_not_guessed():
    with pytest.raises(TypeError, match="structural logical"):
        activate_sub_models({}, {}, ok_stomate="false", ok_dgvm=False)
    with pytest.raises(TypeError, match="structural logical"):
        config_stomate_parameters({}, {}, ch4_calcul="false")
    with pytest.raises(KeyError, match="AGRICULTURE"):
        veget_config({}, {"AGRICULTURE": "y"})
