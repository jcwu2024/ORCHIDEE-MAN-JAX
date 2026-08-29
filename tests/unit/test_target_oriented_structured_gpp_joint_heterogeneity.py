from pathlib import Path

import jax
import numpy as np

from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_joint_calibration import (
    DIRECT_SOURCE,
    SEVEN_SOURCE,
    THIRTY_SOURCE,
)
from research.daily_coarse_graining.daily_operator.target_oriented_structured_gpp_joint_heterogeneity import (
    ARTIFACT_UPDATES,
    EFFECTIVE_BATCH_SIZES,
    FROZEN_CONTRACT_SHA256,
    grouped_artifact_updates,
    load_structured_gpp_joint_heterogeneity_contract,
    minimum_consistent_effective_batch_size,
)
from scripts.hpc.audit_target_oriented_structured_gpp_joint_gradient_heterogeneity import (
    _group_report,
    _mean_host_gradients,
)


def _group(*cosines):
    return {
        "source_to_combined_cosines": {
            "direct": cosines[0],
            "seven": cosines[1],
            "thirty": cosines[2],
        }
    }


def test_structured_gpp_heterogeneity_contract_freezes_train_only_grid():
    contract = load_structured_gpp_joint_heterogeneity_contract()
    assert contract.sha256 == FROZEN_CONTRACT_SHA256
    assert contract.artifact_updates == ARTIFACT_UPDATES
    assert contract.effective_batch_sizes == EFFECTIVE_BATCH_SIZES
    assert contract.payload["data"]["calibration_artifacts_reused"] is False
    assert contract.payload["data"]["evaluation_artifacts_reused"] is False
    assert contract.payload["effective_batch_analysis"]["optimizer_created"] is False
    assert contract.payload["execution"]["paid_training_authorized"] is False


def test_structured_gpp_heterogeneity_groups_are_nested_nonoverlapping_partitions():
    assert grouped_artifact_updates(ARTIFACT_UPDATES, 1)[0] == (512,)
    assert grouped_artifact_updates(ARTIFACT_UPDATES, 2) == (
        (512, 548),
        (584, 620),
        (656, 692),
        (728, 763),
    )
    assert grouped_artifact_updates(ARTIFACT_UPDATES, 4) == (
        (512, 548, 584, 620),
        (656, 692, 728, 763),
    )
    assert grouped_artifact_updates(ARTIFACT_UPDATES, 8) == (ARTIFACT_UPDATES,)


def test_structured_gpp_minimum_consistent_effective_batch_requires_every_group():
    reports = {
        1: [_group(1.0, 1.0, -0.1), _group(1.0, 1.0, 1.0)],
        2: [_group(1.0, 1.0, 0.1), _group(1.0, -0.1, 1.0)],
        4: [_group(1.0, 0.1, 0.1), _group(0.1, 0.1, 0.1)],
        8: [_group(1.0, 1.0, 1.0)],
    }
    assert minimum_consistent_effective_batch_size(reports) == 4
    reports[4][1] = _group(1.0, 0.0, 1.0)
    assert minimum_consistent_effective_batch_size(reports) == 8
    reports[8][0] = _group(1.0, -0.1, 1.0)
    assert minimum_consistent_effective_batch_size(reports) is None


def test_structured_gpp_grouped_host_gradients_are_exact_arithmetic_means():
    left = {"a": np.asarray([1.0, 3.0], dtype=np.float32)}
    right = {"a": np.asarray([3.0, 5.0], dtype=np.float32)}
    mean = _mean_host_gradients(jax, (left, right))
    np.testing.assert_array_equal(mean["a"], [2.0, 4.0])

    gradients = {
        512: {DIRECT_SOURCE: left, SEVEN_SOURCE: left, THIRTY_SOURCE: left},
        548: {DIRECT_SOURCE: right, SEVEN_SOURCE: right, THIRTY_SOURCE: right},
    }
    report = _group_report(
        jax,
        (512, 548),
        gradients,
        {DIRECT_SOURCE: 1.0, SEVEN_SOURCE: 2.0, THIRTY_SOURCE: 3.0},
    )
    assert report["common_descent_passed"]
    assert all(value == 1.0 for value in report["source_to_combined_cosines"].values())


def test_structured_gpp_heterogeneity_runner_is_zero_update_and_free_gpu_only():
    root = Path(__file__).resolve().parents[2]
    runner = (
        root
        / "scripts"
        / "hpc"
        / "audit_target_oriented_structured_gpp_joint_gradient_heterogeneity.py"
    ).read_text(encoding="utf-8")
    launcher = (
        root
        / "scripts"
        / "hpc"
        / "run_target_oriented_structured_gpp_joint_gradient_heterogeneity.sh"
    ).read_text(encoding="utf-8")
    assert "grouped_artifact_updates" in runner
    assert "minimum_consistent_effective_batch_size" in runner
    assert '"optimizer_created": False' in runner
    assert '"optimizer_updates_executed": 0' in runner
    assert '"checkpoint_written": False' in runner
    assert "target_oriented_adamw_update" not in runner
    assert "write_structured_gpp_checkpoint" not in runner
    assert "sbatch" not in launcher
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-120m}" in launcher
