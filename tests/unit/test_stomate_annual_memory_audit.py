from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.dev.audit_stomate_annual_memory_fields import build_annual_memory_audit  # noqa: E402


def test_stomate_annual_memory_audit_has_source_and_jax_state_for_core_fields():
    audit = build_annual_memory_audit()
    summary = audit["summary"]

    assert summary["annual_memory_fields"] == 56
    assert summary["missing_jax_field"] == 0
    assert summary["missing_source"] == 0
    assert summary["groups"]["season_memory"] == 16
    assert summary["groups"]["leak_deep_carbon_doc"] == 11

    by_fortran = {record["fortran_name"]: record for record in audit["records"]}
    assert by_fortran["PFTpresent"]["jax_field"] == "pft_present"
    assert by_fortran["maint_resp"]["jax_field"] == "resp_maint_part"
    assert by_fortran["fixed_cryoturb_depth"]["jax_field"] == "fixed_cryoturbation_depth"
    assert by_fortran["carbon_32l_a"]["jax_field"] == "carbon_32l"
    assert by_fortran["freedoc"]["jax_field"] == "DOC"
