"""Portable finite reference checks; original-source identity is a separate gate."""

from __future__ import annotations

import importlib.util
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
CASES = ("environment_partition", "age_efficiency", "partition_upper", "partition_lower", "fraction_floor", "water_limited", "source_default_upper")
ALL_CASES = CASES + tuple(case + "_reentry" for case in CASES)


@lru_cache(maxsize=1)
def module():
    spec = importlib.util.spec_from_file_location("phase06_growth_arithmetic", ROOT / "scripts/dev/oracle_phase06_growth_arithmetic.py")
    m = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(m)
    return m


@lru_cache(maxsize=1)
def fixture():
    return json.loads((ROOT / "tests/fixtures/phase06_growth_arithmetic_reference.json").read_text(encoding="ascii"))


@lru_cache(maxsize=32)
def execute(case, compiled):
    m = module()
    return m.current(m.base.decode(fixture()["cases"][case]["inputs"]), compiled)


def assert_reference(actual, expected):
    assert set(actual) == set(expected)
    for field, value in actual.items():
        ref = expected[field]
        assert list(value.shape) == ref["shape"], field
        np.testing.assert_allclose(value.ravel(order="F"), ref["values"], rtol=1e-12, atol=1e-14, err_msg=field)


@pytest.mark.parametrize("case", ALL_CASES)
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "public_static_jit"])
def test_original_complete_arithmetic_reference(case, compiled):
    assert_reference(module().base.chain_fields(execute(case, compiled)), fixture()["cases"][case]["expected"])


@pytest.mark.parametrize("case", CASES)
def test_actual_returned_carry_enters_public_next_component(case):
    m = module()
    x = m.base.decode(fixture()["cases"][case]["inputs"])
    carry = m.base.carry_inputs(x, execute(case, True))
    independent = m.base.decode(fixture()["cases"][case + "_reentry"]["inputs"])
    for group in carry:
        for name, value in carry[group].items():
            if isinstance(value, np.ndarray):
                np.testing.assert_allclose(value, independent[group][name], rtol=1e-12, atol=1e-14, err_msg=group + "." + name)
    actual_next = m.current(carry, True)
    assert_reference(m.base.chain_fields(actual_next), fixture()["cases"][case + "_reentry"]["expected"])


@pytest.mark.parametrize("case", CASES)
def test_actual_turnover_return_enters_real_vmax(case):
    m = module()
    x = m.base.decode(fixture()["cases"][case]["inputs"])
    result = m.actual_vmax_consumer(x, execute(case, True))
    expected = fixture()["cases"][case]["expected"]
    assert_reference({name: np.asarray(getattr(result, name)) for name in ("leaf_age", "leaf_frac", "vcmax")}, {name: expected["vmax." + name] for name in ("leaf_age", "leaf_frac", "vcmax")})


def test_observed_original_arithmetic_operands_activate_required_arms():
    arms = {row["arm"] for row in module().arm_evidence(fixture()["observations"])}
    assert {
        "water.lower", "water.interior", "water.upper_equality",
        "humidity.lower", "humidity.interior", "humidity.upper_equality",
        "temperature.lower", "temperature.interior", "temperature.upper",
        "light.inner_floor", "light.interior", "light.outer_high_lai",
        "nitrogen.lower", "nitrogen.interior", "partition.root_floor", "partition.root_interior",
        "partition.leaf_lower", "partition.leaf_interior", "partition.leaf_upper",
        "turnover.age_cap", "turnover.below_cap", "vmax.fraction_zero", "vmax.fraction_positive", "vmax.fraction_negative",
        "vmax.efficiency_rising", "vmax.efficiency_plateau", "vmax.efficiency_declining", "vmax.efficiency_floor",
    } <= arms


def test_parameter_domains_and_independent_reference_shapes():
    m = module()
    assert set(fixture()["cases"]) == set(ALL_CASES)
    for case in CASES:
        x = m.base.decode(fixture()["cases"][case]["inputs"])
        al, po = x["alloc_inputs"], x["post_npp_inputs"]
        assert np.all(1.0 - al["r0"] - al["s0"] >= 0)
        assert np.all(al["s0"] != 1)
        np.testing.assert_array_equal(po["leaf_timecst"], po["vmax_leafagecrit"] / 4.0)
        assert x["constraints_inputs"]["is_peat"][13]
        assert not x["post_npp_inputs"]["ok_dgvm"]
        for name, row in fixture()["cases"][case]["expected"].items():
            assert row["shape"] == list(m.base.reference_shape(name))


def test_observed_floor_and_cap_results_follow_the_triggering_operands():
    for row in module().arm_evidence(fixture()["observations"]):
        arm, values, cells = row["arm"], row["observed_operands"], row["cells_zero_based"]
        if arm == "turnover.age_cap":
            np.testing.assert_array_equal(np.asarray(values["rate"])[cells], .99)
        elif arm == "vmax.fraction_negative":
            assert np.all(np.asarray(values["before"])[cells] < 0)
            np.testing.assert_array_equal(np.asarray(values["after"])[cells], 0)
        elif arm == "partition.leaf_lower":
            np.testing.assert_array_equal(np.asarray(values["bounded"])[cells], .2)
        elif arm == "partition.leaf_upper":
            np.testing.assert_array_equal(np.asarray(values["bounded"])[cells], np.asarray(values["configured_maximum"])[cells])
        elif arm == "vmax.efficiency_floor":
            np.testing.assert_array_equal(np.asarray(values["efficiency"])[cells], .3)
        elif arm == "vmax.efficiency_plateau":
            np.testing.assert_array_equal(np.asarray(values["efficiency"])[cells], 1)


def test_source_default_and_explicit_upper_configuration_remain_distinct():
    default = np.asarray(execute("source_default_upper", True).allocation.f_alloc)
    configured = np.asarray(execute("partition_upper", True).allocation.f_alloc)
    assert np.max(np.abs(default - configured)) > .05
    source_inputs = module().base.decode(fixture()["cases"]["source_default_upper"]["inputs"])
    configured_inputs = module().base.decode(fixture()["cases"]["partition_upper"]["inputs"])
    assert "max_l_to_lsr" not in source_inputs["allocation_kwargs"]
    assert configured_inputs["allocation_kwargs"]["max_l_to_lsr"] == .6


@pytest.mark.parametrize("case", ALL_CASES)
def test_complete_return_is_finite_and_preserves_pool_and_fraction_bounds(case):
    fields = module().base.chain_fields(execute(case, True))
    assert all(np.all(np.isfinite(value)) for value in fields.values())
    for name, value in fields.items():
        if name.endswith(".biomass") or name.endswith(".leaf_frac"):
            assert np.all(value >= 0), name
    fractions = fields["vmax.leaf_frac"]
    assert np.all(fractions <= 1.0 + 1e-14)
    totals = np.sum(fractions, axis=-1)
    assert np.all((totals == 0) | np.isclose(totals, 1, rtol=1e-12, atol=1e-14))
    np.testing.assert_allclose(np.sum(fields["allocation.f_alloc"][:, 13], axis=-1), 1, rtol=1e-12, atol=1e-14)


@pytest.mark.external_data
def test_original_fortran_source_identity_gate():
    if not (ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate_alloc.f90").exists():
        pytest.skip("Original Fortran package is external to portable numerical tests")
    assert fixture()["source_identities"] == module().identities()
