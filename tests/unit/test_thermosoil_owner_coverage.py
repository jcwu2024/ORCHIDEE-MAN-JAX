from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ORACLES = ROOT / "outputs/reference_mode/micro_oracles"


def _evidence(family: str) -> tuple[dict[str, object], dict[str, object]]:
    output = ORACLES / family
    return (
        json.loads((output / "branch_coverage.json").read_text()),
        json.loads((output / "owner_region_evidence.json").read_text()),
    )


def test_thermosoil_conductivity_owner_evidence_is_complete() -> None:
    coverage, owners = _evidence("thermosoil_cond_matrix")
    assert coverage["branch_complete"]
    assert coverage["required_arm_count"] == coverage["covered_arm_count"] == 33
    assert owners["complete"]
    assert len(owners["records"]) == 3
    assert all(record["passed"] for record in owners["records"])


def test_thermosoil_coef_owner_uses_oracle_and_fixed_control_proofs() -> None:
    coverage, owners = _evidence("thermosoil_cwrr_recurrence_coef")
    comparison = json.loads(
        (ORACLES / "thermosoil_cwrr_recurrence_coef/comparison.json").read_text()
    )
    assert comparison["status"] == "passed"
    assert coverage["branch_complete"]
    assert owners["complete"]
    assert len(owners["records"]) == 1
    assert owners["records"][0]["passed"]
    assert owners["records"][0]["source_proof_arm_ids"]


def test_thermosoil_day199_wlupdate_owner_evidence_is_complete() -> None:
    coverage, owners = _evidence("thermosoil_day199_recurrence_actual")
    assert coverage["branch_complete"]
    assert coverage["required_arm_count"] == coverage["covered_arm_count"] == 2
    assert owners["complete"]
    assert len(owners["records"]) == 1
    assert owners["records"][0]["fortran_procedure"] == "thermosoil_wlupdate"


def test_thermosoil_aux_transition_owner_evidence_is_complete() -> None:
    coverage, owners = _evidence("thermosoil_aux_transitions")
    assert coverage["branch_complete"]
    assert coverage["required_arm_count"] == coverage["covered_arm_count"] == 24
    assert owners["complete"]
    assert len(owners["records"]) == 4
    assert {record["fortran_procedure"] for record in owners["records"]} == {
        "add_heat_zimov",
        "thermosoil_energy",
        "thermosoil_getdiff_thinsnow",
        "thermosoil_humlev",
    }


def test_thermosoil_finalize_state_io_owner_evidence_is_complete() -> None:
    output = ORACLES / "thermosoil_finalize_state_io"
    comparison = json.loads((output / "comparison.json").read_text())
    coverage = json.loads((output / "state_io_coverage.json").read_text())
    owners = json.loads((output / "owner_region_evidence.json").read_text())
    assert comparison["status"] == "passed"
    assert coverage["complete"]
    assert owners["complete"]
    assert len(owners["records"]) == 1
    assert owners["records"][0]["fortran_procedure"] == "thermosoil_finalize"


def test_sechiba_typed_setvar_owner_families_are_complete() -> None:
    expected = {"sechiba_setvar_serial": (68, 13), "sechiba_setvar_parallel": (74, 13)}
    for family, (arms, owner_count) in expected.items():
        coverage, owners = _evidence(family)
        assert coverage["branch_complete"]
        assert coverage["required_arm_count"] == coverage["covered_arm_count"] == arms
        assert owners["complete"]
        assert len(owners["records"]) == owner_count
