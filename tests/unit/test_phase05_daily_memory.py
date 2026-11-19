from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.dev.oracle_phase05_daily_memory import run, sha

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def evidence(tmp_path_factory):
    return run(tmp_path_factory.mktemp("phase05_daily_memory"))


def test_historical_source_inputs_and_strict_references_are_unchanged(evidence):
    assert evidence["source_identity_mode"] == "external_not_requested"
    assert {key: len(value) for key, value in evidence["saved_reference_checks"].items()} == {
        "season": 38,
        "daily_maintenance": 5,
        "instant_daily_prep": 14,
    }
    assert evidence["rtol"] == 1e-12
    assert evidence["atol"] == 1e-14
    assert all(item["passed"] for group in evidence["saved_reference_checks"].values() for item in group)


@pytest.mark.parametrize(
    "case", ["paper_reference", "paper_reference_noleap", "paper_reference_cap", "paper_reference_floor"]
)
def test_original_season_reference_and_actual_daily_reset_carry(evidence, case):
    component = evidence["paper_components"][case]
    assert len(component["oracle_checks"]) == 38
    assert all(item["passed"] for item in component["oracle_checks"] + component["checks"])


def test_maintenance_actual_fold_and_narrow_jit_reuse_same_fortran_tuple(evidence):
    assert all(item["passed"] for item in evidence["actual_maintenance_component"]["checks"])


def test_actual_next_consumers_receive_season_returns(evidence):
    component = evidence["actual_consumer_join"]
    assert component["level"] == "actual_consumer_operands_and_returns_only"
    assert all(item["passed"] for item in component["checks"])


@pytest.mark.parametrize("case", ["paper_reference", "paper_reference_noleap", "paper_reference_cap"])
def test_actual_common_predecessor_updates_solar_gdd_before_ngd(evidence, case):
    component = evidence["paper_components"][case]
    assert component["actual_pre_step_pft14_check"]["passed"], component["actual_pre_step_pft14_check"]
    assert component["actual_writeback_gdd_check"]["passed"], component["actual_writeback_gdd_check"]


def test_reference_fixture_declares_separate_microdomains():
    fixture = json.loads((ROOT / "tests/fixtures/phase05_daily_memory_reference.json").read_text())
    assert fixture["paper_reference"]["ngd_minus5"][26:] == [0.98, 0.98]
    assert fixture["paper_reference_noleap"]["gdd_init_date"] != fixture["paper_reference"]["gdd_init_date"]


def test_original_year_tseason_output_observed_without_changing_old_case(evidence):
    assert evidence["year_tseason_extension"]["comparison"]["passed"]


@pytest.mark.external_data
def test_external_original_source_and_historical_asset_identities():
    fixture = json.loads((ROOT / "tests/fixtures/phase05_daily_memory_reference.json").read_text())
    missing = [path for path in fixture["immutable_assets"] if not (ROOT / path).is_file()]
    if missing:
        pytest.skip("External original-source/historical assets unavailable: " + ", ".join(missing))
    assert all(sha(ROOT / path) == expected for path, expected in fixture["immutable_assets"].items())
