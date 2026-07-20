"""Rebuild the complete Batch C SECHIBA evidence family from its witnesses.

Each SECHIBA owner is produced by a separately executable oracle.  This
assembler deliberately merges those immutable witness assets so rerunning one
owner cannot erase the other five from ``owner_region_evidence.json``.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FAMILY = "sechiba_batch_c"
OUTPUT = ROOT / "outputs/reference_mode/micro_oracles" / FAMILY
CONTRACTS = ROOT / "outputs/reference_mode/pft14_arm_contract_classes.json"
SOURCE_PROOFS = ROOT / "outputs/reference_mode/pft14_source_proof_evidence.json"

OWNER_IDS = (
    "pft14-owner-contract-d5ed3bbbb641",
    "pft14-owner-contract-bee72b6b6bf0",
    "pft14-owner-contract-8437dcaf9cc2",
    "pft14-owner-contract-79f78d322bec",
    "pft14-owner-contract-198a52d8c6f6",
    "pft14-owner-contract-beb73c2fad63",
)


def _load(name: str) -> dict[str, object]:
    return json.loads((OUTPUT / name).read_text(encoding="utf-8"))


def _passed_witness(name: str, *flags: str) -> bool:
    path = OUTPUT / name
    if not path.is_file():
        return False
    witness = _load(name)
    return all(bool(witness.get(flag)) for flag in flags)


def main() -> None:
    contracts = json.loads(CONTRACTS.read_text(encoding="utf-8"))
    source_proofs = json.loads(SOURCE_PROOFS.read_text(encoding="utf-8"))
    pinned = {
        str(record["arm_id"])
        for record in source_proofs["records"]
        if record.get("passed") is True
    }
    owners = {
        str(owner["owner_region_id"]): owner
        for owner in contracts["owner_regions"]
        if str(owner["owner_region_id"]) in OWNER_IDS
    }
    if set(owners) != set(OWNER_IDS):
        raise RuntimeError("SECHIBA Batch C owner contract IDs drifted")

    first_call = _passed_witness(
        "sechiba_init_first_call_witness.json", "passed"
    )
    restget = _passed_witness(
        "sechiba_initialize_restget_witness.json",
        "passed_fortran_control_flow",
    )
    output_dispatch = _passed_witness(
        "sechiba_main_output_dispatch_witness.json", "passed"
    )
    lai_mean = _passed_witness("sechiba_main_lai_mean_witness.json", "passed")
    tail = _load("sechiba_main_fixed_tail_comparison.json")
    tail_ok = tail.get("status") == "passed"
    tail_gcov = set(tail.get("gcov_arm_ids", [])) if tail_ok else set()
    tail_fatal = set(tail.get("fatal_witness_arm_ids", [])) if tail_ok else set()
    irrigation = _load("sechiba_main_full_irrigation_witness.json")
    irrigation_gcov = (
        set(irrigation.get("passed_arm_ids", []))
        if irrigation.get("passed") is True
        else set()
    )

    explicit: dict[str, set[str]] = {owner_id: set() for owner_id in OWNER_IDS}
    if first_call:
        explicit["pft14-owner-contract-bee72b6b6bf0"].add(
            "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90:1983:if:true"
        )
    if restget:
        explicit["pft14-owner-contract-79f78d322bec"].update(
            {
                "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90:774:if:false",
                "fortran_source/ORCHIDEE/src_sechiba/sechiba.f90:774:if:true",
            }
        )
    if output_dispatch:
        explicit["pft14-owner-contract-198a52d8c6f6"].update(
            _load("sechiba_main_output_dispatch_witness.json")["arms"]
        )
    if lai_mean:
        explicit["pft14-owner-contract-198a52d8c6f6"].update(
            _load("sechiba_main_lai_mean_witness.json")["arms"]
        )
    explicit["pft14-owner-contract-beb73c2fad63"].update(tail_gcov | irrigation_gcov)

    records = []
    arm_rows = []
    for owner_id in OWNER_IDS:
        owner = owners[owner_id]
        required = set(owner["arm_ids"])
        source_backed = required & pinned
        gcov_backed = required & explicit[owner_id]
        fatal_backed = required & tail_fatal if owner_id.endswith("beb73c2fad63") else set()
        covered = source_backed | gcov_backed | fatal_backed
        missing = required - covered
        record = {
            "owner_region_id": owner_id,
            "base_region_id": owner["base_region_id"],
            "fortran_procedure": owner["fortran_procedure"],
            "jax_owners": owner["jax_owners"],
            "comparison_asset": f"outputs/reference_mode/micro_oracles/{FAMILY}/sechiba_main_fixed_tail_comparison.json",
            "branch_coverage_asset": f"outputs/reference_mode/micro_oracles/{FAMILY}/branch_coverage.json",
            "required_arm_ids": sorted(required),
            "gcov_arm_ids": sorted(gcov_backed),
            "fatal_witness_arm_ids": sorted(fatal_backed),
            "source_proof_arm_ids": sorted(source_backed),
            "covered_arm_ids": sorted(covered),
            "missing_arm_ids": sorted(missing),
            "passed": not missing,
        }
        records.append(record)
        for arm in sorted(required):
            arm_rows.append(
                {
                    "arm_id": arm,
                    "owner_region_id": owner_id,
                    "support": (
                        "real_gcov" if arm in gcov_backed else
                        "exact_fatal_witness" if arm in fatal_backed else
                        "pinned_source_proof" if arm in source_backed else "unclosed"
                    ),
                }
            )

    evidence = {
        "schema_version": 1,
        "family": FAMILY,
        "complete": all(record["passed"] for record in records),
        "records": records,
    }
    coverage = {
        "schema_version": 1,
        "family": FAMILY,
        "branch_complete": evidence["complete"],
        "required_arm_count": len(arm_rows),
        "covered_arm_count": sum(row["support"] != "unclosed" for row in arm_rows),
        "missing_arm_ids": [row["arm_id"] for row in arm_rows if row["support"] == "unclosed"],
        "arms": arm_rows,
        "policy": "Only retained GCOV, exact fatal witnesses, and pinned source proofs support an arm.",
    }
    (OUTPUT / "owner_region_evidence.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="ascii"
    )
    (OUTPUT / "branch_coverage.json").write_text(
        json.dumps(coverage, indent=2) + "\n", encoding="ascii"
    )
    print(json.dumps({"complete": evidence["complete"], "records": len(records)}, indent=2))
    if not evidence["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
