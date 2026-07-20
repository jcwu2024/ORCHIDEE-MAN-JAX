from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
FRAGMENT = ROOT / "docs/source_audits/oracle_families/hydrol_thermosoil.yaml"
LEDGER = ROOT / "docs/source_audits/pft14_reachable_ledger.yaml"
INVENTORY = ROOT / "outputs/reference_mode/fortran_control_flow_inventory.json"
RESOLUTION = ROOT / "outputs/reference_mode/pft14_control_flow_resolution.json"
DISPOSITIONS = ROOT / "docs/source_audits/pft14_arm_dispositions_sechiba.yaml"


def _document(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _span_hash(path: Path, start: int, end: int) -> str:
    lines = path.read_bytes().splitlines(keepends=True)
    return hashlib.sha256(b"".join(lines[start - 1 : end])).hexdigest()


def test_hydrol_lane_fragment_contract_and_source_hashes():
    document = _document(FRAGMENT)
    assert document["schema_version"] == 2
    assert document["families"]
    for family in document["families"]:
        required = {
            "id",
            "runner",
            "ledger_entries",
            "procedures",
            "input_asset",
            "comparison_asset",
        }
        if "branch_coverage_asset" not in family:
            required |= {"output_contract", "branch_cases"}
        assert required <= family.keys()
        assert ":" in family["runner"]
        for procedure in family["procedures"]:
            if "source_file" not in procedure:
                source = ROOT / "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90"
                assert _span_hash(source, procedure["start_line"], procedure["end_line"]) == procedure["expected_span_sha256"]
                continue
            assert {
                "source_file",
                "procedure",
                "procedure_kind",
                "start_line",
                "end_line",
                "expected_source_sha256",
                "expected_span_sha256",
                "dependencies",
                "compilation",
                "numerical_oracle_status",
            } <= procedure.keys()
            assert "use_modules" in procedure["dependencies"]
            compilation = procedure["compilation"]
            assert compilation["status"] == "passed"
            assert compilation["standalone_compilable"] is False
            assert compilation["compiler_available"] is True
            assert {
                "compiler",
                "compiler_version",
                "harness",
                "result_asset",
            } <= compilation.keys()
            assert procedure["numerical_oracle_status"] == "verified"
            source = ROOT / procedure["source_file"]
            assert (
                hashlib.sha256(source.read_bytes()).hexdigest()
                == procedure["expected_source_sha256"]
            )
            assert (
                _span_hash(source, procedure["start_line"], procedure["end_line"])
                == procedure["expected_span_sha256"]
            )


def test_hydrol_lane_branch_cases_cover_declared_control_flow_arms():
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    inventory_arm_ids = {item["id"] for item in inventory["control_flow_arms"]}
    resolution = json.loads(RESOLUTION.read_text(encoding="utf-8"))
    disposition_rules = {
        item["branch_id"]: item["decisions"]
        for item in _document(DISPOSITIONS)["rules"]
    }
    for family in _document(FRAGMENT)["families"]:
        if "branch_coverage_asset" in family:
            coverage = json.loads((ROOT / family["branch_coverage_asset"]).read_text(encoding="ascii"))
            assert coverage["branch_complete"] is True
            assert coverage["missing_disposition"] == []
            assert coverage["missing_case_assignment"] == []
            covered_entries = (
                set(coverage["ledger_entries"])
                if coverage.get("ledger_entries")
                else {coverage["ledger_entry"]}
            )
            assert covered_entries == set(family["ledger_entries"])
            continue
        cases_by_entry = {entry: [] for entry in family["ledger_entries"]}
        for case in family["branch_cases"]:
            assert {
                "id",
                "ledger_entry",
                "fortran_lines",
                "condition",
                "expected_arm",
                "input_case",
                "control_flow_arm_ids",
            } <= case.keys()
            assert case["ledger_entry"] in cases_by_entry
            declared = set(family["required_control_flow_arm_ids"])
            assert set(case["control_flow_arm_ids"]) <= declared
            assert set(case["control_flow_arm_ids"]) <= inventory_arm_ids
            cases_by_entry[case["ledger_entry"]].append(case)
        for entry, cases in cases_by_entry.items():
            assert cases
            declared = set(family["required_control_flow_arm_ids"])
            covered = {arm for case in cases for arm in case["control_flow_arm_ids"]}
            assert covered == declared
            authoritative = set()
            for row in resolution["rows"]:
                if entry not in row.get("process_ledger_ids", []):
                    continue
                decisions = disposition_rules.get(row["control_flow_id"])
                assert decisions is not None
                for side, decision in decisions.items():
                    if decision["disposition"] in {"pft14_active", "pft14_conditional"}:
                        authoritative.add(f"{row['control_flow_id']}:{str(side).lower()}")
            assert declared == authoritative


def test_hydrol_lane_cached_comparisons_are_verified():
    for family in _document(FRAGMENT)["families"]:
        result_asset = ROOT / (
            family["result_asset"]
            if "result_asset" in family
            else family["procedures"][0]["compilation"]["result_asset"]
        )
        result = json.loads(result_asset.read_text(encoding="ascii"))
        assert result["family"] == family["id"]
        assert result["status"] == "passed"
        assert set(result["verified_ledger_entries"]) == set(family["ledger_entries"])
        assert result["comparisons"] and all(
            item["passed"] for item in result["comparisons"]
        )


def test_canopy_bare_soil_undefined_output_is_not_claimed():
    family = next(
        item
        for item in _document(FRAGMENT)["families"]
        if item["id"] == "hydrol_canopy_infiltration_freeze"
    )
    undefined = family["fortran_undefined_contract"]
    assert len(undefined) == 1
    assert undefined[0]["field"] == "canopy2ground(:,1)"
    assert undefined[0]["fortran_lines"] == "5066-5090"
    assert "canopy2ground" not in family["output_contract"]
    assert "canopy2ground_pft2_to_pft14" in family["output_contract"]


def test_path_evidence_families_do_not_claim_verified_ledger_entries():
    document = _document(FRAGMENT)
    for family in document["path_evidence_families"]:
        result_asset = family.get("result_asset")
        if result_asset is None and family.get("procedures"):
            result_asset = family["procedures"][0].get("compilation", {}).get("result_asset")
        assert result_asset is not None
        result = json.loads((ROOT / result_asset).read_text(encoding="ascii"))
        assert result["status"] == "passed"
        assert result["verified_ledger_entries"] == []
        assert set(result["path_evidence_ledger_entries"]) == set(family["ledger_entries"])


def test_hydrol_soil_owner_uses_real_initializer_and_rebuildable_inputs():
    family = next(
        item
        for item in _document(FRAGMENT)["families"]
        if item["id"] == "hydrol_soil_owner_cwrr_water_root_peat_alt"
    )
    coverage = json.loads((ROOT / family["branch_coverage_asset"]).read_text(encoding="ascii"))
    assert coverage["branch_complete"] is True
    assert coverage["missing_disposition"] == []
    assert coverage["missing_case_assignment"] == []
    assert family["procedures"][0]["procedure"] == "hydrol_var_init"
    inputs = json.loads((ROOT / family["input_asset"]).read_text(encoding="ascii"))
    assert inputs["schema_version"] == 2
    assert inputs["common"]["compiler_nbint"] == 50
    assert [case["name"] for case in inputs["cases"]] == [
        "cwrr_baseline",
        "forced_water_table",
        "dynamic_root",
        "peat_tide",
        "alt_bare_residual",
        "pond_storage",
        "tide_nonpositive",
        "tide_falling",
        "tide_zero_boundary",
        "new_water_stress_root_threshold",
        "freeze_aware_cwrr",
        "soil_resistance_bare_evaporation",
        "peat_nontidal",
        "agricultural_peat",
        "permafrost_peat",
        "forced_peat_water_table",
    ]
    source = ROOT / "fortran_source/ORCHIDEE/src_sechiba/hydrol.f90"
    for procedure in family["procedures"]:
        assert (
            _span_hash(source, procedure["start_line"], procedure["end_line"])
            == procedure["expected_span_sha256"]
        )


def test_thermosoil_recurrence_keeps_undefined_ok_zimov_explicit():
    pending = {
        item["id"]: item for item in _document(FRAGMENT)["pending_families"]
    }
    recurrence = pending["thermosoil_cwrr_recurrence"]
    assert recurrence["ledger_entries"] == ["thermosoil.active.cwrr_recurrence"]
    assert recurrence["blockers"] == [
        {
            "id": "thermosoil_main.local_ok_zimov",
            "fortran_lines": "877,991-993",
            "detail": "A local uninitialized LOGICAL SAVE shadows the initialized module ok_zimov; no paper build flag explicitly initializes local SAVE storage.",
        }
    ]
