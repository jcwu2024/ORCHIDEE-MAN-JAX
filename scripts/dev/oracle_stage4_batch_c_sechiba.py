"""Formal evidence assembly for the closed ``sechiba_main`` fixed-tail owner."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.dev.extract_fortran_micro_oracle import extract_procedure_bytes  # noqa: E402


FAMILY = "sechiba_batch_c"
SOURCE = ROOT / "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90"
OUTPUT = ROOT / "outputs/reference_mode/micro_oracles" / FAMILY
CONTRACTS = ROOT / "outputs/reference_mode/pft14_arm_contract_classes.json"
SOURCE_PROOFS = ROOT / "outputs/reference_mode/pft14_source_proof_evidence.json"

OWNER_IDS = ("pft14-owner-contract-beb73c2fad63",)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build(output: Path = OUTPUT) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    contracts = _load(CONTRACTS)
    proofs = _load(SOURCE_PROOFS)
    proof_ids = {
        str(record["arm_id"])
        for record in proofs.get("records", [])
        if record.get("passed") is True
    }
    gcov_ids: set[str] = set()
    fatal_ids: set[str] = set()
    full_irrigation_witness_path = output / "sechiba_main_full_irrigation_witness.json"
    full_irrigation_witness = _load(full_irrigation_witness_path) if full_irrigation_witness_path.is_file() else {}
    full_irrigation_passed = bool(full_irrigation_witness.get("cases")) and all(
        case.get("passed") is True for case in full_irrigation_witness.get("cases", [])
    ) and full_irrigation_witness.get("passed") is True
    if full_irrigation_passed:
        gcov_ids.update(str(arm) for arm in full_irrigation_witness.get("passed_arm_ids", []))
    fixed_tail_path = output / "sechiba_main_fixed_tail_comparison.json"
    fixed_tail = _load(fixed_tail_path) if fixed_tail_path.is_file() else {}
    fixed_tail_passed = fixed_tail.get("status") == "passed"
    if fixed_tail_passed:
        gcov_ids.update(str(arm) for arm in fixed_tail.get("gcov_arm_ids", []))
        fatal_ids.update(str(arm) for arm in fixed_tail.get("fatal_witness_arm_ids", []))
    execution_ids = gcov_ids | fatal_ids
    owners = {
        str(record["owner_region_id"]): record
        for record in contracts.get("owner_regions", [])
        if record.get("owner_region_id") in OWNER_IDS
    }
    if set(owners) != set(OWNER_IDS):
        raise RuntimeError("Batch C owner contract IDs drifted")

    spans: dict[str, dict[str, object]] = {}
    for procedure in sorted({str(owner["fortran_procedure"]) for owner in owners.values()}):
        span = extract_procedure_bytes(SOURCE, procedure)
        asset = f"owner_{procedure}.f90"
        (output / asset).write_bytes(span.span_bytes)
        spans[procedure] = {
            "asset": asset,
            "start_line": span.start_line,
            "end_line": span.end_line,
            "sha256": span.span_sha256,
        }

    records = []
    arm_records = []
    for owner_id in OWNER_IDS:
        owner = owners[owner_id]
        required = sorted(str(arm) for arm in owner["arm_ids"])
        pinned = sorted(arm for arm in required if arm in proof_ids)
        executed = sorted(arm for arm in required if arm in execution_ids)
        covered = sorted(set(pinned) | set(executed))
        missing = sorted(set(required) - set(covered))
        records.append({
            "owner_region_id": owner_id,
            "base_region_id": owner["base_region_id"],
            "fortran_procedure": owner["fortran_procedure"],
            "jax_owners": owner["jax_owners"],
            "comparison_asset": f"outputs/reference_mode/micro_oracles/{FAMILY}/sechiba_main_fixed_tail_comparison.json",
            "component_comparison_assets": [
                f"outputs/reference_mode/micro_oracles/{FAMILY}/sechiba_main_full_irrigation_witness.json",
                f"outputs/reference_mode/micro_oracles/{FAMILY}/sechiba_main_fixed_tail_comparison.json",
            ],
            "branch_coverage_asset": f"outputs/reference_mode/micro_oracles/{FAMILY}/branch_coverage.json",
            "required_arm_ids": required,
            "gcov_arm_ids": sorted(set(required) & gcov_ids),
            "fatal_witness_arm_ids": sorted(set(required) & fatal_ids),
            "source_proof_arm_ids": pinned,
            "covered_arm_ids": covered,
            "missing_arm_ids": missing,
            "direct_state_writeback_comparison": {
                "status": "passed" if full_irrigation_passed and fixed_tail_passed else "failed",
                "comparison_count": len(fixed_tail.get("comparisons", [])),
                "state_asset": f"outputs/reference_mode/micro_oracles/{FAMILY}/sechiba_main_fixed_tail_comparison.json",
                "full_irrigation_asset": f"outputs/reference_mode/micro_oracles/{FAMILY}/sechiba_main_full_irrigation_witness.json",
                "tolerance_policy": fixed_tail.get("tolerance_policy"),
            },
            "passed": not missing and full_irrigation_passed and fixed_tail_passed,
        })
        arm_records.extend({
            "arm_id": arm,
            "owner_region_id": owner_id,
            "support": (
                "pinned_source_proof" if arm in proof_ids
                else "exact_fatal_witness" if arm in fatal_ids
                else "real_gcov" if arm in gcov_ids
                else "unclosed"
            ),
        } for arm in required)

    covered = [item["arm_id"] for item in arm_records if item["support"] != "unclosed"]
    missing = [item["arm_id"] for item in arm_records if item["support"] == "unclosed"]
    owner_complete = all(record["passed"] for record in records)
    coverage = {
        "schema_version": 1,
        "family": FAMILY,
        "branch_complete": not missing and owner_complete,
        "required_arm_count": len(arm_records),
        "covered_arm_count": len(covered),
        "missing_arm_ids": missing,
        "execution_witness_arm_ids": sorted(execution_ids),
        "fatal_witness_arm_ids": sorted(fatal_ids),
        "source_proof_arm_ids": sorted(arm for arm in covered if arm in proof_ids),
        "case_matrix": fixed_tail.get("case_matrix", []),
        "arms": arm_records,
        "policy": "Only GCOV, exact fatal witnesses, and pre-existing pinned source proofs can support an arm.",
    }
    evidence = {
        "schema_version": 1,
        "family": FAMILY,
        "complete": not missing and owner_complete,
        "records": records,
    }
    inputs = {
        "schema_version": 1,
        "owner_ids": list(OWNER_IDS),
        "source_file": SOURCE.relative_to(ROOT).as_posix(),
        "source_file_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "owner_procedure_spans": spans,
        "minimal_genuine_dependencies": [
            "constantes", "constantes_soil", "pft_parameters", "routing", "hydrol",
            "diffuco_main", "enerbil_main", "hydrol_main", "condveg_main",
            "thermosoil_main", "slowproc_main", "restart/history transport callbacks",
        ],
        "case_matrix": fixed_tail.get("case_matrix", []),
        "original_fragment_harness": f"outputs/reference_mode/micro_oracles/{FAMILY}/sechiba_main_fixed_tail_oracle.f90",
        "gcov_artifact": f"outputs/reference_mode/micro_oracles/{FAMILY}/sechiba_main_fixed_tail_oracle.f90.gcov",
        "fatal_witness_asset": f"outputs/reference_mode/micro_oracles/{FAMILY}/sechiba_main_fixed_tail_fatal_witness.json",
    }
    (output / "inputs.json").write_text(json.dumps(inputs, indent=2) + "\n", encoding="ascii")
    (output / "branch_coverage.json").write_text(json.dumps(coverage, indent=2) + "\n", encoding="ascii")
    (output / "owner_region_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="ascii")
    return coverage


if __name__ == "__main__":
    report = build()
    print(json.dumps({"branch_complete": report["branch_complete"], "missing": len(report["missing_arm_ids"])}, indent=2))
