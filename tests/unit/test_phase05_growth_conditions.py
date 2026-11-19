"""Source-reference checks at the complete, standalone daily-carbon boundary."""
from __future__ import annotations

import importlib.util
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def module():
    spec = importlib.util.spec_from_file_location(
        "phase05_growth_oracle", ROOT / "scripts/dev/oracle_phase05_growth_conditions.py"
    )
    m = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(m)
    return m


@lru_cache(maxsize=1)
def fixture():
    return json.loads((ROOT / "tests/fixtures/phase05_growth_reference.json").read_text())


@lru_cache(maxsize=32)
def execute(case, compiled=False):
    m = module()
    return m.run_current_chain(m.decode(fixture()["cases"][case]["inputs"]), public_wrapper=compiled)


def assert_reference(actual, expected):
    assert set(actual) == set(expected)
    for name, values in actual.items():
        ref = expected[name]
        assert list(values.shape) == ref["shape"], name
        np.testing.assert_allclose(
            values.ravel(order="F"), ref["values"],
            rtol=fixture()["rtol"], atol=fixture()["atol"], err_msg=name,
        )


@pytest.mark.external_data
def test_original_growth_source_identities():
    missing = [record["file"] for record in fixture()["source_identities"].values()
               if not (ROOT / record["file"]).is_file()]
    if missing:
        pytest.skip("Original growth source identity NOT EXECUTED; missing: " + ", ".join(missing))
    assert fixture()["source_identities"] == module().source_identities()


def test_complete_boundary_inputs():
    assert set(fixture()["cases"]) == {"restart", "cold", "sla_bounds", "restart_reentry", "cold_reentry", "sla_bounds_reentry"}
    for case in fixture()["cases"].values():
        inputs = module().decode(case["inputs"])
        assert inputs["prescribe_inputs"]["biomass"].shape == (6, 14, 12, 1)
        assert inputs["alloc_inputs"]["tsoil_month"].shape == (6, 2)
        assert inputs["constraints_inputs"]["is_peat"][13]
        assert inputs["constraints_inputs"]["agriculture"]
        assert inputs["constraints_inputs"]["undef"] == -9999.
        assert inputs["post_npp_inputs"]["min_stomate"] == 1e-8
        assert inputs["prescribe_inputs"]["pheno_is_none"][13]


@pytest.mark.parametrize("case", ["restart", "cold", "sla_bounds", "restart_reentry", "cold_reentry", "sla_bounds_reentry"])
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "public_static_jit"])
def test_full_original_growth_chain_reference(case, compiled):
    assert_reference(module().chain_fields(execute(case, compiled)), fixture()["cases"][case]["expected"])


@pytest.mark.parametrize("case", ["restart", "cold"])
def test_actual_returned_carry_enters_next_real_component(case):
    m = module()
    original_inputs = m.decode(fixture()["cases"][case]["inputs"])
    first = execute(case)
    actual_carry = m.carry_inputs(original_inputs, first)
    expected_carry = m.decode(fixture()["cases"][case+"_reentry"]["inputs"])
    for group in actual_carry:
        for name, value in actual_carry[group].items():
            if isinstance(value, np.ndarray):
                np.testing.assert_allclose(value, expected_carry[group][name], rtol=1e-12, atol=1e-14, err_msg=group+"."+name)
    # Consume the JAX return itself in the maintained public compiled chain.
    # Fortran's reentry reference used its own directly returned state.
    second = m.run_current_chain(actual_carry, public_wrapper=True)
    assert_reference(m.chain_fields(second), fixture()["cases"][case+"_reentry"]["expected"])
    assert not actual_carry["prescribe_inputs"]["firstcall"]
    assert not actual_carry["prescribe_inputs"]["stomate_restart_none"]


def test_source_floor_and_bare_soil_sla_carry_are_both_observable():
    inputs = module().decode(fixture()["cases"]["restart"]["inputs"])
    result = execute("restart")
    # Original npp_calc loops SLA over j=2,nvm; bare-soil SLA must retain carry.
    np.testing.assert_array_equal(result.post_npp.daily_carbon.age_sla.sla_calc[:, 0], inputs["post_npp_inputs"]["sla_calc"][:, 0])
    assert float(result.post_npp.daily_carbon.npp_update.biomass[1, 13, 7, 0]) == 1e-8
    assert float(result.post_npp.daily_carbon.npp_update.resp_maint[1, 13]) < 0.
    # Positive allocation and gap/turnover fluxes are present in the same tuple.
    assert float(result.allocation.f_alloc[0, 13, 0]) > 0.
    assert float(result.post_npp.gap.bm_to_litter[3, 13, 1, 0]) > 0.
    assert float(result.post_npp.turnover.turnover[3, 13, 0, 0]) > 0.
    np.testing.assert_array_equal(result.post_npp.bm_to_litter, result.post_npp.gap.bm_to_litter)


def test_source_ordered_sla_clamps_cover_upper_young_and_lower_arms():
    bounded = execute("sla_bounds").post_npp.daily_carbon.age_sla.sla_calc
    assert float(bounded[0, 13]) == .03
    assert float(bounded[4, 13]) == .01
    # The absent cell has weighted age zero, so the middle ELSEWHERE arm wins.
    assert float(execute("restart").post_npp.daily_carbon.age_sla.sla_calc[5, 13]) == .03


def test_original_caller_keeps_turnover_lai_until_final_setlai():
    # stomate_lpj updates slai after alloc at1113, passes entry lai to turn
    # at1264, and refreshes lai itself only at1553.
    inputs = module().decode(fixture()["cases"]["restart"]["inputs"])
    result = execute("restart")
    np.testing.assert_array_equal(result.post_npp.turnover.lai, inputs["alloc_inputs"]["lai"])
    assert float(result.post_npp.turnover.lai[0, 13]) != float(result.post_npp.lai_after_setlai[0, 13])


@pytest.mark.parametrize("case", [
    "paper_natural_peat_tcm", "paper_natural_peat_undef", "natural_no_agri_tcm", "natural_no_agri_undef",
    "peat_only_tcm", "peat_only_undef", "agriculture_only_tcm", "agriculture_only_undef",
    "excluded_mask_tcm", "excluded_mask_undef", "seasonal_tree_tcm", "seasonal_tree_undef",
])
def test_original_constraints_mask_temperature_matrix(case):
    from jax_orchidee.stomate.carbon_kernels import constraints_step
    row = fixture()["constraints_cases"][case]
    result = constraints_step(**module().decode(row["inputs"]))
    assert_reference({"adapted": np.asarray(result.adapted), "regenerate": np.asarray(result.regenerate)}, row["expected"])
