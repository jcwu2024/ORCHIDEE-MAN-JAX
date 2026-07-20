from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/reference_mode/micro_oracles/intersurf_main_batch_c"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/intersurf.f90"
OWNER = "pft14-owner-contract-81281ca5ba43"


def _json(name: str) -> dict:
    return json.loads((OUTPUT / name).read_text(encoding="ascii"))


def test_intersurf_main_owner_has_original_bytes_and_genuine_scientific_call_graph() -> None:
    manifest = _json("source_manifest.json")
    owner = manifest["intersurf_main_2d"]
    extracted = OUTPUT / "extracted_original_bytes/intersurf_main_2d.f90"
    assert owner["start_line"] == 458
    assert owner["end_line"] == 761
    assert owner["byte_exact_in_compile_unit"] is True
    assert hashlib.sha256(extracted.read_bytes()).hexdigest() == owner["sha256"]
    assert extracted.read_bytes() in SOURCE.read_bytes()

    graph = _json("call_graph.json")
    assert graph["scientific_stub_count"] == 0
    assert set(graph["genuine_scientific_procedures"]) >= {
        "sechiba_var_init",
        "enerbil_begin",
        "enerbil_surftemp",
        "enerbil_flux",
        "condveg_z0cdrag",
        "condveg_albedo",
        "qsatcalc",
    }
    assert all(
        record["byte_exact_in_compile_unit"]
        for record in manifest.values()
        if record["role"] != "full_direct_callee_audit_extract"
    )
    assert manifest["sechiba_main"]["role"] == "full_direct_callee_audit_extract"
    assert (OUTPUT / "extracted_original_bytes/sechiba_main.f90").is_file()


def test_intersurf_main_owner_comparison_covers_every_export_and_boundary_at_tight_tolerance() -> None:
    comparison = _json("comparison.json")
    assert comparison["status"] == "passed"
    assert comparison["rtol"] <= 1.0e-12
    assert comparison["all_exported_flux_state_writebacks_compared"] is True
    assert set(comparison["exported_fields"]) == {
        "z0m", "coastalflow", "riverflow", "tsol_rad", "vevapp",
        "temp_sol_new", "qsurf", "albedo", "fluxsens", "fluxlat",
        "emis", "cdrag",
    }
    assert {case["case_id"] for case in comparison["cases"]} == {
        "native_history_unsynced_window",
        "native_history_sync_gate_disabled",
    }
    assert all(item["passed"] for item in comparison["comparisons"])
    assert comparison["formal_audit"]["complete"] is True


def test_intersurf_main_six_canonical_arms_are_mapped_and_formally_accepted() -> None:
    coverage = _json("branch_coverage.json")
    evidence = _json("owner_region_evidence.json")
    audit = _json("formal_audit_report.json")
    record = evidence["records"][0]

    assert coverage["branch_complete"] is True
    assert coverage["required_arm_count"] == 6
    assert coverage["covered_arm_count"] == 6
    assert set(coverage["canonical_arm_ids"]) == set(record["required_arm_ids"])
    assert (OUTPUT / "oracle.f90.gcov").is_file()
    assert coverage["fatal_boundary_arm_ids"] == []
    assert record["owner_region_id"] == OWNER
    assert record["passed"] is True
    assert set(record["gcov_arm_ids"]) | set(record["source_proof_arm_ids"]) == set(record["required_arm_ids"])
    assert record["direct_state_writeback_comparison"]["rtol"] <= 1.0e-12
    assert audit["complete"] is True
    assert audit["counts"]["passed_owner_regions"] == 1
