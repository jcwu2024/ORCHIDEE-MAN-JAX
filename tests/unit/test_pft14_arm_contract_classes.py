from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/dev/audit_pft14_arm_contract_classes.py"
SPEC = importlib.util.spec_from_file_location("audit_pft14_arm_contract_classes", SCRIPT)
AUDIT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(AUDIT)


def _repository_report() -> dict[str, object]:
    return AUDIT.build_report(
        json.loads(
            (ROOT / "outputs/reference_mode/pft14_arm_equivalence_regions.json").read_text()
        ),
        json.loads((ROOT / "outputs/reference_mode/pft14_arm_disposition.json").read_text()),
        yaml.safe_load(
            (ROOT / "docs/source_audits/pft14_arm_contract_policy.yaml").read_text()
        ),
    )


def test_repository_contract_classification_is_complete_and_unique() -> None:
    report = _repository_report()
    assert report["classification_complete"]
    assert report["counts"]["classified_arms"] == report["counts"]["input_arms"]
    assert not any(report["classification_errors"].values())
    assert all(not record["closed"] for record in report["arm_classifications"])
    assert len({record["arm_id"] for record in report["arm_classifications"]}) == report[
        "counts"
    ]["input_arms"]


def test_source_proofs_are_narrow_and_retain_source_rationale() -> None:
    report = _repository_report()
    proofs = [
        record
        for record in report["arm_classifications"]
        if record["evidence_route"] == "source_proof"
    ]
    assert proofs
    assert all(record["rationale"] for record in proofs)
    assert all(
        record["edge_class"]
        in {
            "paper_static_dispatch",
            "inactive_process",
            "valid_input_continuation",
            None,
        }
        for record in proofs
    )
    assert all(
        record["disposition_source"] != "automatic"
        for record in proofs
        if record["edge_class"] == "valid_input_continuation"
    )


def test_scientific_and_driver_transitions_still_require_fortran_oracle() -> None:
    report = _repository_report()
    records = report["arm_classifications"]
    for contract_class in {"scientific_state_transition", "driver_input_transition"}:
        selected = [record for record in records if record["contract_class"] == contract_class]
        assert selected
        numerical = [record for record in selected if record["edge_class"] is None]
        assert numerical
        assert all(
            record["evidence_route"] == "fortran_transition_oracle_and_production"
            for record in numerical
        )
        assert all(
            record["owner_evidence_route"]
            == "fortran_transition_oracle_and_production"
            for record in selected
        )


def test_source_proof_edge_does_not_erase_owner_evidence_requirement() -> None:
    report = _repository_report()
    static_science = [
        record
        for record in report["arm_classifications"]
        if record["edge_class"] == "paper_static_dispatch"
        and record["contract_class"] == "scientific_state_transition"
    ]
    assert static_science
    assert all(record["evidence_route"] == "source_proof" for record in static_science)
    assert all(
        record["owner_evidence_route"] == "fortran_transition_oracle_and_production"
        for record in static_science
    )


def test_state_and_output_contracts_are_not_misreported_as_process_equations() -> None:
    report = _repository_report()
    states = [
        record
        for record in report["arm_classifications"]
        if record["contract_class"] == "state_io_transition"
    ]
    outputs = [
        record
        for record in report["arm_classifications"]
        if record["contract_class"] == "output_lifecycle_transition"
    ]
    assert states and outputs
    assert all(
        record["evidence_route"]
        in {"fortran_state_io_oracle_and_lifecycle", "source_proof"}
        for record in states
    )
    assert all(
        record["evidence_route"]
        in {"output_acceptance_and_source_contract", "source_proof"}
        for record in outputs
    )
