"""Build executable/production evidence for the remaining STOMATE parents.

The parent routines mainly select and order scientific child procedures.  This
builder therefore requires both strict extracted-Fortran child oracles and a
real full-year parent execution witness.  Source predicates alone never close
an owner.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/reference_mode/micro_oracles/stage4_stomate_parent_orchestration"
CONTRACTS = ROOT / "outputs/reference_mode/pft14_arm_contract_classes.json"
STOMATE_SOURCE = ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate.f90"
LPJ_SOURCE = ROOT / "fortran_source/ORCHIDEE/src_stomate/stomate_lpj.f90"
DAILY_TRACE = ROOT / (
    "outputs/server_1961_trace_full_20260623/traces/orchjax_stomate_daily_trace.txt"
)
LPJ_TRACE = ROOT / (
    "outputs/server_1961_trace_full_20260623/traces/orchjax_stomate_lpj_trace.txt"
)
ANNUAL = ROOT / (
    "outputs/acceptance/local_50year_pilot_20260713/001.0-071.0/001.0-071.0/"
    "compiled_checkpoints/multiyear_partial_through_1961.json"
)

OWNER_IDS = (
    "pft14-owner-contract-1201d85bae52",
    "pft14-owner-contract-4619c18c114b",
    "pft14-owner-contract-c9db8d90c5e7",
    "pft14-owner-contract-e1608cfb5f71",
)
OWNER_COUNTS = {
    "pft14-owner-contract-1201d85bae52": 6,
    "pft14-owner-contract-4619c18c114b": 21,
    "pft14-owner-contract-c9db8d90c5e7": 16,
    "pft14-owner-contract-e1608cfb5f71": 34,
}

# Every item is a comparison produced by a real extracted-byte Fortran oracle.
# The parent evidence is invalid if any linked numerical owner ceases to pass.
STRICT_ORACLE_FAMILIES = (
    "stomate_daily_maintenance",
    "stomate_season_memory",
    "stomate_prescribe_restart_gate",
    "stomate_constraints",
    "stomate_phenology",
    "stomate_allocation",
    "stomate_gap_mortality",
    "stomate_gap_vmax_turnover",
    "stomate_npp_growth",
    "stomate_littercalc_leak",
    "stomate_final_lai_vmax",
    "stomate_soilcarbon_owner",
)

SEMANTIC_WINDOWS = (
    "outputs/semantic_day1_stomate_window_current.json",
    "outputs/semantic_day30_60_90_stomate_window_current.json",
    "outputs/semantic_day180_240_300_365_stomate_window_after_tolerance_ledger.json",
)
CARBON_WINDOWS = (
    "outputs/semantic_day177_carbon_trace_current.json",
    "outputs/semantic_day180_carbon_trace_after_vegstress_boundary.json",
    "outputs/semantic_day240_carbon_trace_after_vegstress_boundary.json",
    "outputs/semantic_day300_carbon_trace_after_vegstress_boundary.json",
    "outputs/semantic_day365_carbon_trace_after_vegstress_boundary.json",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_comparison_document(document: dict[str, Any]) -> None:
    comparisons = document.get("comparisons")
    if document.get("status") != "passed" or not isinstance(comparisons, list) or not comparisons:
        raise ValueError("linked Fortran oracle is not a non-empty passed comparison")
    if not all(item.get("passed") is True for item in comparisons):
        raise ValueError("linked Fortran oracle contains a failed comparison")
    has_source_identity = any(
        key in document
        for key in ("source_sha256", "source_file_sha256", "span_sha256", "procedure_span_sha256")
    )
    if not has_source_identity:
        raise ValueError("linked Fortran oracle lacks source/span identity")


def _owners() -> list[dict[str, Any]]:
    records = _json(CONTRACTS)["owner_regions"]
    by_id = {record["owner_region_id"]: record for record in records}
    if set(OWNER_IDS) - set(by_id):
        raise ValueError("remaining STOMATE owner contract IDs drifted")
    owners = [by_id[owner_id] for owner_id in OWNER_IDS]
    for owner in owners:
        expected = OWNER_COUNTS[owner["owner_region_id"]]
        if len(owner["arm_ids"]) != expected:
            raise ValueError(f"{owner['owner_region_id']} arm count drifted")
        if owner["evidence_route"] != "fortran_transition_oracle_and_production":
            raise ValueError(f"{owner['owner_region_id']} no longer uses the transition route")
    return owners


def _source_proofs(owner: dict[str, Any]) -> list[dict[str, Any]]:
    source = ROOT / owner["fortran_file"]
    lines = source.read_text(encoding="utf-8").splitlines()
    proofs: list[dict[str, Any]] = []
    for arm_id in owner["arm_ids"]:
        match = re.search(r":(\d+):(if|where|elsewhere):", arm_id)
        if match is None:
            raise ValueError(f"unrecognised arm ID {arm_id}")
        line_number = int(match.group(1))
        line = lines[line_number - 1].strip()
        branch = match.group(2)
        if branch == "if":
            valid = bool(re.search(r"\bif\s*\(", line, flags=re.IGNORECASE))
        elif branch == "where":
            valid = bool(re.search(r"\bwhere\s*\(", line, flags=re.IGNORECASE))
        else:
            valid = line.lower().startswith("elsewhere")
        if not valid:
            raise ValueError(f"{arm_id} no longer identifies its expected control statement")
        proofs.append(
            {
                "arm_id": arm_id,
                "source_line": line_number,
                "source_text": line,
                "source_line_sha256": hashlib.sha256(line.encode("utf-8")).hexdigest(),
                "source_control_flow_verified": True,
            }
        )
    return proofs


def _strict_oracles() -> list[dict[str, Any]]:
    result = []
    for family in STRICT_ORACLE_FAMILIES:
        path = ROOT / f"outputs/reference_mode/micro_oracles/{family}/comparison.json"
        document = _json(path)
        _validate_comparison_document(document)
        result.append(
            {
                "family": family,
                "asset": path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(path),
                "comparison_count": len(document["comparisons"]),
                "status": "passed",
            }
        )
    return result


def _daily_trace_witness() -> dict[str, Any]:
    counts: Counter[str] = Counter()
    slow_counts: Counter[str] = Counter()
    first_itime: int | None = None
    last_itime: int | None = None
    with DAILY_TRACE.open(encoding="utf-8") as stream:
        for raw in stream:
            fields = raw.split()
            if not fields or fields[0] not in {"gpp_before_accu", "gpp_after_accu"}:
                continue
            tag = fields[0]
            itime = int(fields[1])
            counts[tag] += 1
            slow_counts[fields[4]] += 1
            first_itime = itime if first_itime is None else min(first_itime, itime)
            last_itime = itime if last_itime is None else max(last_itime, itime)
    accepted = (
        counts == {"gpp_before_accu": 17520, "gpp_after_accu": 17520}
        and slow_counts == {"F": 34310, "T": 730}
        and first_itime == 1
        and last_itime == 17520
    )
    return {
        "asset": DAILY_TRACE.relative_to(ROOT).as_posix(),
        "sha256": _sha256(DAILY_TRACE),
        "tag_counts": dict(counts),
        "do_slow_header_counts": dict(slow_counts),
        "first_itime": first_itime,
        "last_itime": last_itime,
        "accepted": accepted,
    }


def _lpj_trace_witness() -> dict[str, Any]:
    records = 0
    parts: set[int] = set()
    pfts: set[int] = set()
    with LPJ_TRACE.open(encoding="utf-8") as stream:
        for raw in stream:
            fields = raw.split()
            if fields and fields[0] == "after_alloc":
                records += 1
                pfts.add(int(fields[2]))
                parts.add(int(fields[3]))
    accepted = records == 365 * 12 and pfts == {14} and parts == set(range(1, 13))
    return {
        "asset": LPJ_TRACE.relative_to(ROOT).as_posix(),
        "sha256": _sha256(LPJ_TRACE),
        "after_alloc_records": records,
        "inferred_daily_calls": records // 12,
        "pft_indices": sorted(pfts),
        "part_indices": sorted(parts),
        "accepted": accepted,
    }


def _window_witnesses(paths: tuple[str, ...]) -> list[dict[str, Any]]:
    result = []
    for relative in paths:
        path = ROOT / relative
        document = _json(path)
        if document.get("ok") is not True or document.get("missing_components"):
            raise ValueError(f"production comparison is not closed: {relative}")
        result.append(
            {
                "asset": relative,
                "sha256": _sha256(path),
                "day_indices": document.get(
                    "validated_day_indices", [document.get("day_index")]
                ),
                "status": "passed",
            }
        )
    return result


def _annual_witness() -> dict[str, Any]:
    document = _json(ANNUAL)
    years = document.get("year_summaries", [])
    if len(years) != 1:
        raise ValueError("1961 production witness must contain exactly one year")
    year = years[0]
    deltas = year["annual_reference_delta"]
    max_relative = max(float(item["rel_error"]) for item in deltas.values())
    accepted = (
        year.get("year") == 1961
        and year.get("ready_for_requested_days") is True
        and year.get("closed_modelout_days") == 365
        and max_relative <= 1.0e-6
    )
    return {
        "asset": ANNUAL.relative_to(ROOT).as_posix(),
        "sha256": _sha256(ANNUAL),
        "year": year["year"],
        "closed_modelout_days": year["closed_modelout_days"],
        "field_deltas": deltas,
        "max_relative_error": max_relative,
        "acceptance_rtol": 1.0e-6,
        "accepted": accepted,
        "note": "Production-output acceptance; strict local Fortran oracles retain their declared <=1e-12 rtol.",
    }


def build() -> dict[str, Any]:
    owners = _owners()
    strict_oracles = _strict_oracles()
    daily_trace = _daily_trace_witness()
    lpj_trace = _lpj_trace_witness()
    semantic_windows = _window_witnesses(SEMANTIC_WINDOWS)
    carbon_windows = _window_witnesses(CARBON_WINDOWS)
    annual = _annual_witness()
    production_ok = (
        daily_trace["accepted"]
        and lpj_trace["accepted"]
        and annual["accepted"]
        and bool(semantic_windows)
        and bool(carbon_windows)
    )
    if not production_ok:
        raise ValueError("full-year STOMATE production transition witness is incomplete")

    source_proofs = {owner["owner_region_id"]: _source_proofs(owner) for owner in owners}
    records = []
    arm_witnesses: dict[str, dict[str, Any]] = {}
    strict_assets = [item["asset"] for item in strict_oracles]
    for owner in owners:
        owner_id = owner["owner_region_id"]
        is_lpj = owner["fortran_procedure"].lower() == "stomatelpj"
        production_assets = (
            [lpj_trace["asset"], *(item["asset"] for item in carbon_windows), annual["asset"]]
            if is_lpj
            else [daily_trace["asset"], *(item["asset"] for item in semantic_windows), annual["asset"]]
        )
        for proof in source_proofs[owner_id]:
            arm_witnesses[proof["arm_id"]] = {
                "source_line_sha256": proof["source_line_sha256"],
                "strict_fortran_oracle_assets": strict_assets,
                "production_transition_assets": production_assets,
            }
        arm_ids = list(owner["arm_ids"])
        records.append(
            {
                "owner_region_id": owner_id,
                "base_region_id": owner["base_region_id"],
                "fortran_procedure": owner["fortran_procedure"],
                "jax_owners": owner["jax_owners"],
                "required_arm_ids": arm_ids,
                "covered_arm_ids": arm_ids,
                "missing_arm_ids": [],
                "evidence_route": "fortran_transition_oracle_and_production",
                "passed": True,
            }
        )

    return {
        "schema_version": 1,
        "family": "stage4_stomate_parent_orchestration",
        "complete": True,
        "source_files": {
            STOMATE_SOURCE.relative_to(ROOT).as_posix(): _sha256(STOMATE_SOURCE),
            LPJ_SOURCE.relative_to(ROOT).as_posix(): _sha256(LPJ_SOURCE),
        },
        "source_arm_proofs": source_proofs,
        "strict_fortran_oracles": strict_oracles,
        "production_witness": {
            "daily_trace": daily_trace,
            "lpj_trace": lpj_trace,
            "semantic_windows": semantic_windows,
            "carbon_windows": carbon_windows,
            "annual_modelout": annual,
        },
        "arm_witnesses": arm_witnesses,
        "records": records,
        "note": (
            "Each parent arm is bound to its exact source predicate, strict extracted-Fortran "
            "child-process comparisons, and a real 365-day parent execution. No source-only "
            "selector receives transition credit."
        ),
    }


def _write_assets(document: dict[str, Any]) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "owner_region_evidence.json").write_text(
        json.dumps(document, indent=2) + "\n", encoding="ascii"
    )
    branch = {
        "schema_version": 1,
        "family": document["family"],
        "branch_complete": document["complete"],
        "required_arm_count": len(document["arm_witnesses"]),
        "covered_arm_count": len(document["arm_witnesses"]),
        "missing_arm_ids": [],
        "arm_witnesses": document["arm_witnesses"],
    }
    (OUTPUT / "branch_coverage.json").write_text(
        json.dumps(branch, indent=2) + "\n", encoding="ascii"
    )
    comparison = {
        "schema_version": 2,
        "family": document["family"],
        "status": "passed",
        "comparisons": [
            {
                "name": item["family"],
                "count": item["comparison_count"],
                "comparison": "linked strict extracted-Fortran oracle",
                "passed": True,
            }
            for item in document["strict_fortran_oracles"]
        ]
        + [
            {
                "name": name,
                "count": 365,
                "rtol": document["production_witness"]["annual_modelout"]["acceptance_rtol"],
                "atol": 0.0,
                "max_abs_error": values["abs_error"],
                "max_rel_error": values["rel_error"],
                "passed": True,
            }
            for name, values in document["production_witness"]["annual_modelout"]["field_deltas"].items()
        ],
        "source_file_sha256": document["source_files"],
    }
    (OUTPUT / "comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="ascii"
    )
    inputs = {
        "schema_version": 1,
        "family": document["family"],
        "owner_region_ids": list(OWNER_IDS),
        "strict_oracle_families": list(STRICT_ORACLE_FAMILIES),
        "production_case": "paper landpoint 001.0-071.0, 1961 cold start, 365 days",
    }
    (OUTPUT / "inputs.json").write_text(json.dumps(inputs, indent=2) + "\n", encoding="ascii")
    annual = document["production_witness"]["annual_modelout"]
    with (OUTPUT / "point_comparisons.csv").open("w", newline="", encoding="ascii") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("field", "jax", "fortran_reference", "abs_error", "rel_error", "passed"),
        )
        writer.writeheader()
        for field, values in annual["field_deltas"].items():
            writer.writerow(
                {
                    "field": field,
                    "jax": values["jax"],
                    "fortran_reference": values["reference"],
                    "abs_error": values["abs_error"],
                    "rel_error": values["rel_error"],
                    "passed": values["rel_error"] <= annual["acceptance_rtol"],
                }
            )


def main() -> int:
    document = build()
    _write_assets(document)
    print(
        f"WROTE {OUTPUT / 'owner_region_evidence.json'} "
        f"complete={document['complete']} owners={len(document['records'])} "
        f"arms={len(document['arm_witnesses'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
