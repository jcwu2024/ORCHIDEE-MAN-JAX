from pathlib import Path

import pytest

from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_joint_calibration import (
    DIRECT_SOURCE,
    SEVEN_SOURCE,
    THIRTY_SOURCE,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_joint_update import (
    FROZEN_CONTRACT_SHA256,
    load_structured_gpp_joint_update_preflight_contract,
    structured_gpp_joint_objectives,
)


def test_structured_gpp_joint_update_contract_binds_calibrated_policy():
    contract = load_structured_gpp_joint_update_preflight_contract()
    assert contract.sha256 == FROZEN_CONTRACT_SHA256
    assert contract.artifact_update == 512
    assert contract.coefficients == {
        DIRECT_SOURCE: 1.0,
        SEVEN_SOURCE: 5.065197169020285,
        THIRTY_SOURCE: 4.594433600967992,
    }
    assert contract.clip_threshold == 46.802861793665286
    assert (
        contract.payload["identity"]["calibration_report_sha256"]
        == "6ff2d03e0719b22c64012834ea3c5b134a740efa59be01d5715e5cf4c24389c0"
    )


def test_structured_gpp_joint_update_contract_uses_inherited_discarded_adamw():
    payload = load_structured_gpp_joint_update_preflight_contract().payload
    proposal = payload["optimizer_proposal"]
    assert proposal["state_source"] == "warm_start_checkpoint_after_250_direct_gpp_updates"
    assert proposal["initial_optimizer_step"] == 250
    assert proposal["proposed_optimizer_step"] == 251
    assert proposal["updates_executed"] == 1
    assert proposal["updates_persisted"] == 0
    assert proposal["checkpoint_written"] is False
    assert payload["execution"]["paid_training_authorized"] is False
    assert payload["execution"]["joint_training_authorized"] is False


def test_structured_gpp_joint_objectives_use_frozen_three_source_sum():
    contract = load_structured_gpp_joint_update_preflight_contract()
    result = structured_gpp_joint_objectives(
        {DIRECT_SOURCE: 2.0, SEVEN_SOURCE: 0.25, THIRTY_SOURCE: 0.5},
        contract.coefficients,
    )
    expected_science = 5.065197169020285 * 0.25 + 4.594433600967992 * 0.5
    assert result["weighted_science"] == pytest.approx(expected_science)
    assert result["joint"] == pytest.approx(2.0 + expected_science)
    with pytest.raises(ValueError, match="source loss is invalid"):
        structured_gpp_joint_objectives(
            {DIRECT_SOURCE: 2.0, SEVEN_SOURCE: -0.25, THIRTY_SOURCE: 0.5},
            contract.coefficients,
        )


def test_structured_gpp_joint_update_runner_is_discarded_and_free_gpu_only():
    root = Path(__file__).resolve().parents[2]
    runner = (
        root / "scripts" / "hpc" / "run_target_oriented_structured_gpp_joint_update_preflight.py"
    ).read_text(encoding="utf-8")
    launcher = (
        root / "scripts" / "hpc" / "run_target_oriented_structured_gpp_joint_update_preflight.sh"
    ).read_text(encoding="utf-8")
    assert "load_structured_gpp_checkpoint" in runner
    assert 'warm_checkpoint["optimizer"]' in runner
    assert "target_oriented_adamw_update" in runner
    assert "PARENT_PARAMETER_COUNT =" not in runner
    assert "ADAPTER_PARAMETER_COUNT =" not in runner
    assert '"optimizer_updates_persisted": 0' in runner
    assert '"checkpoint_written": False' in runner
    assert "write_structured_gpp_checkpoint" not in runner
    assert "write_structured_gpp_recovery" not in runner
    assert "source_to_negative_parameter_step_cosines" in runner
    assert "sbatch" not in launcher
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-120m}" in launcher
    assert "target-oriented-structured-gpp-joint-calibration" in launcher
