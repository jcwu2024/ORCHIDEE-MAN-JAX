"""Portable strict gates; outside-paper diagnostic failures stay explicit."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.dev.oracle_phase06_daily_memory import FIXTURE, run, sha

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    return run(tmp_path_factory.mktemp("phase06_daily_memory"))


@pytest.fixture(scope="module")
def reference():
    return json.loads(FIXTURE.read_text())


@pytest.mark.parametrize(
    "case",
    [
        "paper_tau_below",
        "paper_tau_equal",
        "paper_tau_above_year",
        "paper_clamp_equalities",
        "diagnostic_defined_memory",
        "diagnostic_dgvm",
    ],
)
def test_original_full_state_and_real_narrow_jit_strict_gates(evidence, case):
    checks = evidence["cases"][case]["checks"]
    assert len(checks) == 123
    assert all(check["passed"] for check in checks), [check for check in checks if not check["passed"]]
    assert evidence["rtol"] == 1e-12
    assert evidence["atol"] == 1e-14


def test_actual_compiled_memory_reaches_public_carbon_consumer(evidence):
    assert all(check["passed"] for check in evidence["consumer_join"]["checks"])
    assert evidence["fixed_paper_gate_passed"]
    assert evidence["status"] == "complete_with_outside_scope_failures"


@pytest.mark.parametrize(
    "left,right",
    [
        ("gdd_warm", "gdd_defined"),
        ("midsummer_week", "midsummer_long"),
        ("mid_defined", "mid_warm"),
        ("ncd_defined", "ncd_cold"),
    ],
)
def test_original_observer_independently_observed_all_four_conjunct_combinations(reference, left, right):
    obs = reference["original_observations"]["diagnostic_defined_memory"]
    actual = set(zip(obs[left + ".pft14"], obs[right + ".pft14"]))
    assert actual == {(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)}


def test_original_observer_spring_finite_and_humidity_physical_exclusions(reference):
    obs = reference["original_observations"]["diagnostic_defined_memory"]
    for left, right in (("spring_first_left", "spring_first_right"), ("spring_second_left", "spring_second_right")):
        assert set(zip(obs[left + ".pft14"], obs[right + ".pft14"])) == {(0.0, 1.0), (1.0, 0.0), (1.0, 1.0)}
    assert set(zip(obs["hum_defined.pft14"], obs["hum_newmin.pft14"])) == {(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)}
    for missing in range(4):
        flags = obs[f"critical_defined.pft{9 + missing}"]
        assert flags == [float(index != missing) for index in range(4)]
    assert obs["critical_defined.pft14"] == [1.0] * 4


def test_original_implicit_caps_include_equalities_and_both_sides(reference):
    obs = reference["original_observations"]
    assert [
        obs[name]["tau_candidate.pft0"][0] for name in ("paper_tau_below", "paper_tau_equal", "paper_tau_above_year")
    ] == [1094.0, 1095.0, 1096.0]
    equal = obs["paper_clamp_equalities"]
    assert equal["tlong_raw.pft0"][1] == equal["tlong_low.pft0"][0] == 253.1
    assert equal["tlong_raw.pft0"][3] == equal["tlong_high.pft0"][0] == 303.1
    bio = obs["diagnostic_defined_memory"]
    assert bio["ncd_candidate.pft14"][:3] == [1094.0, 1095.0, 1096.0]
    assert bio["ncd_max.pft14"] == [1095.0]
    assert bio["ncd_cold.pft14"][:3] == [1.0] * 3


def test_outside_paper_boundary_failures_remain_strict_failures(evidence, reference):
    # This verifies preservation of failures; it is not a scientific pass for
    # DGVM and does not change original reference numbers or thresholds.
    for case in ("diagnostic_frac_exact_min", "diagnostic_frac_valid_cover_min"):
        failures = evidence["outside_scope_strict_failures"][case]
        assert {item["name"] for item in failures} == {
            prefix + name
            for prefix in ("original.", "original_compiled.")
            for name in ("maxfpc_lastyear", "maxfpc_thisyear")
        }
        assert all(not item["passed"] and item["failed_coordinates"] == 1 for item in failures)
        assert reference["original_observations"][case]["fpc_fraction.pft14"][5] == 1e-8
        assert reference["original_observations"][case]["fpc_frac.pft14"][5] == 0
    valid = reference["complete_inputs"]["diagnostic_frac_valid_cover_min"]
    veget, maximal = np.asarray(valid["v"]), np.asarray(valid["vd"])
    assert np.all((veget >= 0) & (veget <= maximal))
    np.testing.assert_allclose(maximal.sum(axis=1), np.ones(8), rtol=0, atol=1e-15)
    assert valid["domain"].startswith("nonpaper")


@pytest.mark.external_data
def test_original_source_and_template_identity_audit_separate_from_numerical_gates(reference):
    missing = [path for path in reference["immutable_assets"] if not (ROOT / path).is_file()]
    if missing:
        pytest.skip("External source assets unavailable: " + ", ".join(missing))
    assert all(sha(ROOT / path) == expected for path, expected in reference["immutable_assets"].items())
