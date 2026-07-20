from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/dev/audit_pft14_arm_equivalence_regions.py"
SPEC = importlib.util.spec_from_file_location("audit_pft14_arm_equivalence_regions", SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(AUDIT)


def _arm(line: int, *, procedure: str = "step") -> dict[str, object]:
    branch_id = f"fortran_source/ORCHIDEE/src_sechiba/a.f90:{line}:if"
    return {
        "arm_id": f"{branch_id}:true",
        "branch_id": branch_id,
        "arm": "true",
        "file": "fortran_source/ORCHIDEE/src_sechiba/a.f90",
        "procedure": procedure,
        "line": line,
        "source": "IF (condition) THEN",
        "evidence_status": "source_owner_only",
    }


def _resolution(
    arm: dict[str, object],
    *,
    owner_statuses: list[str] | None = None,
    process_ids: list[str] | None = None,
) -> dict[str, object]:
    statuses = owner_statuses or []
    return {
        "control_flow_id": arm["branch_id"],
        "source_owner_ids": [f"owner.{index}" for index, _ in enumerate(statuses)],
        "source_owner_statuses": statuses,
        "jax_owners": [f"jax_orchidee.module.owner_{index}" for index, _ in enumerate(statuses)],
        "process_ledger_ids": process_ids or [],
    }


def test_every_owner_only_arm_is_uniquely_partitioned() -> None:
    owner_arm = _arm(10)
    inactive_arm = _arm(20, procedure="inactive_step")
    report = AUDIT.build_report(
        {"records": [owner_arm, inactive_arm]},
        {
            "rows": [
                _resolution(owner_arm, owner_statuses=["partial", "implemented"]),
                _resolution(inactive_arm, process_ids=["a.inactive.step"]),
            ]
        },
        {"entries": []},
        {
            "entries": [
                {
                    "id": "a.inactive.step",
                    "reachability": "inactive",
                    "condition": "A fixed PFT14 switch makes this procedure unreachable.",
                }
            ]
        },
    )

    assert report["partition_complete"]
    assert report["counts"]["assigned_arms"] == 2
    assert report["counts"]["equivalence_regions"] == 2
    assert report["counts"]["transition_oracle_regions"] == 1
    assert report["counts"]["source_reclassification_regions"] == 1
    assert report["counts"]["known_implementation_gap_regions"] == 0
    assert len({item["arm_id"] for item in report["arm_assignments"]}) == 2
    assert all(region["closed"] is False for region in report["regions"])
    inactive_region = next(
        region for region in report["regions"] if not region["requires_executable_evidence"]
    )
    assert inactive_region["source_proof_rationale"]


def test_inventory_category_controls_region_without_filename_heuristics() -> None:
    arm = _arm(10, procedure="reader")
    report = AUDIT.build_report(
        {"records": [arm]},
        {"rows": [_resolution(arm, owner_statuses=["implemented"])]},
        {
            "entries": [
                {
                    "id": "state.reader",
                    "category": "state_io",
                    "fortran_file": arm["file"],
                    "fortran_subroutine": "reader",
                    "reason": "Restores a prognostic field.",
                    "gap": "Some optional namespaces remain open.",
                }
            ]
        },
        {"entries": []},
    )

    region = report["regions"][0]
    assert region["region_kind"] == "state_io_transition_required"
    assert region["classification_basis"] == "process_inventory_extension"
    assert region["procedure_level_gap_context"]
    assert region["requires_executable_evidence"]
    assert not region["known_implementation_gap"]


def test_partial_owner_alone_cannot_be_partitioned_as_implemented() -> None:
    arm = _arm(10)
    try:
        AUDIT.build_report(
            {"records": [arm]},
            {"rows": [_resolution(arm, owner_statuses=["partial"])]},
            {"entries": []},
            {"entries": []},
        )
    except ValueError as exc:
        assert "no extension" in str(exc)
    else:
        raise AssertionError("partial owner was incorrectly accepted as implemented evidence")


def test_non_owner_only_records_are_outside_this_partition() -> None:
    arm = _arm(10)
    arm["evidence_status"] = "fortran_oracle_and_production"
    report = AUDIT.build_report(
        {"records": [arm]},
        {"rows": []},
        {"entries": []},
        {"entries": []},
    )

    assert report["partition_complete"]
    assert report["counts"]["owner_only_arms"] == 0
    assert report["regions"] == []


def test_repository_owner_only_backlog_is_complete_and_uniquely_partitioned() -> None:
    arm_evidence = json.loads(
        (ROOT / "outputs/reference_mode/pft14_arm_evidence_gap.json").read_text()
    )
    report = AUDIT.build_report(
        arm_evidence,
        json.loads(
            (ROOT / "outputs/reference_mode/pft14_control_flow_resolution.json").read_text()
        ),
        yaml.safe_load(
            (ROOT / "docs/source_audits/pft14_process_inventory_extensions.yaml").read_text()
        ),
        yaml.safe_load(
            (ROOT / "docs/source_audits/pft14_reachable_ledger.yaml").read_text()
        ),
    )
    expected = sum(
        record.get("evidence_status") == "source_owner_only"
        for record in arm_evidence["records"]
    )

    assert report["partition_complete"]
    assert report["counts"]["owner_only_arms"] == expected
    assert report["counts"]["assigned_arms"] == expected
    assert not any(report["partition_errors"].values())
    assert all(
        region["source_proof_rationale"]
        for region in report["regions"]
        if not region["requires_executable_evidence"]
    )
    assert all(region["closed"] is False for region in report["regions"])
