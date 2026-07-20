from __future__ import annotations

from pathlib import Path

from jax_orchidee.driver.paper_binding import (
    PAPER_POINT_VARYING_RUN_DEF_KEYS,
    audit_paper_landpoint_binding,
    expected_paper_domain_limits,
    varying_case_run_def_keys,
)
from jax_orchidee.driver.reference_layout import inventory_paper_references, resolve_paper_landpoint_reference


ROOT = Path(__file__).resolve().parents[2]


def test_landpoint_id_decodes_archived_two_degree_forcing_zoom():
    assert expected_paper_domain_limits("001.0-071.0") == {
        "LIMIT_WEST": -180.0,
        "LIMIT_EAST": -178.0,
        "LIMIT_SOUTH": -20.0,
        "LIMIT_NORTH": -18.0,
    }
    assert expected_paper_domain_limits("319.0-057.0") == {
        "LIMIT_WEST": 138.0,
        "LIMIT_EAST": 140.0,
        "LIMIT_SOUTH": -34.0,
        "LIMIT_NORTH": -32.0,
    }


def test_all_669_archived_cases_have_only_the_declared_point_varying_keys():
    references = inventory_paper_references(ROOT)

    assert len(references) == 669
    assert set(varying_case_run_def_keys(references)) == set(PAPER_POINT_VARYING_RUN_DEF_KEYS)


def test_high_risk_point_bindings_reach_their_real_jax_consumers():
    for landpoint_id in ("001.0-071.0", "069.0-119.0", "319.0-057.0"):
        result = audit_paper_landpoint_binding(ROOT, resolve_paper_landpoint_reference(ROOT, landpoint_id))

        assert result.passed, result.errors
        assert all(result.checks.values())
        assert len(result.package_signature_sha256) == 64


def test_case_sensitive_submission_key_does_not_override_fortran_used_truth():
    result = audit_paper_landpoint_binding(ROOT, resolve_paper_landpoint_reference(ROOT, "069.0-119.0"))

    assert result.checks["fortran_used_static_values_preserved"]
    assert not result.errors
