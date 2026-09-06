from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_conditioned_attention_persistence import (
    load_checkpoint,
    load_recovery,
    load_start_snapshot,
    parameter_trees_identical,
    write_checkpoint,
    write_recovery,
    write_start_snapshot,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_conditioned_attention_persistent_training import (
    FINAL_LOGICAL_UPDATE,
    FROZEN_CONTRACT_SHA256,
    LOGICAL_UPDATES,
    START_COMPLETED_LOGICAL_UPDATE,
    TRAIN_PANEL_GROUP_UPDATES,
    calibrate_persistent_step_l2,
    classify_persistent_smoke,
    load_conditioned_attention_persistent_training_contract,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_direct_generalization import (
    MONITOR_UPDATES,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_objective import (
    GRADIENT_SOURCES,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_persistent_scale_training import (
    ALL_REPORTED_KEYS,
)
from scripts.hpc import run_target_oriented_coupled_gpp_conditioned_attention_persistent_training as runner


def _panel() -> dict:
    return {
        "artifact_updates": list(MONITOR_UPDATES),
        "records": [{"artifact_update": value} for value in MONITOR_UPDATES],
        "source_mean_losses": {name: 1.0 for name in GRADIENT_SOURCES},
        "absolute_key_scores": {horizon: {name: 1.0 for name in ALL_REPORTED_KEYS} for horizon in ("7", "30")},
        "acceptance": {
            horizon: {
                "keys": {
                    name: {
                        "normalized_rmse": 0.1,
                        "normalized_signed_bias": 0.01,
                    }
                    for name in ALL_REPORTED_KEYS
                }
            }
            for horizon in ("7", "30")
        },
        "hard_gate_passed": True,
    }


def _records(last_update: int) -> list[dict]:
    return [
        {
            "logical_update": update,
            "selection": {"accepted": True, "selected_scale": 1.0},
        }
        for update in range(LOGICAL_UPDATES[0], last_update + 1)
    ]


def _monitors(last_update: int) -> list[dict]:
    gate = {"passed": True, "absolute_science": {"passed": False}}
    return [
        {
            "logical_update": update,
            "panel": _panel(),
            "local_gate": gate,
            "cumulative_gate": gate,
        }
        for update in range(LOGICAL_UPDATES[0], last_update + 1)
    ]


def test_conditioned_persistent_contract_freezes_three_disjoint_updates():
    contract = load_conditioned_attention_persistent_training_contract()
    artifacts = [value for panel in contract.train_panels for group in panel for value in group]
    assert contract.sha256 == FROZEN_CONTRACT_SHA256
    assert contract.logical_updates == LOGICAL_UPDATES == (3, 4, 5)
    assert contract.panel_group_updates == TRAIN_PANEL_GROUP_UPDATES
    assert len(artifacts) == len(set(artifacts)) == 120
    assert set(artifacts).isdisjoint(contract.monitor_updates)
    assert tuple(len(registry) for registry in contract.source_registries) == (43, 43, 43)
    assert contract.payload["starting_state"]["required_completed_logical_update"] == 2
    assert contract.payload["fixed_monitor"]["failed_monitor_candidate_discarded"] is True
    assert contract.payload["execution"]["paid_partition_authorized"] is False


def test_conditioned_persistent_calibration_is_recomputed_from_current_registry():
    contract = load_conditioned_attention_persistent_training_contract()
    labels = contract.source_registries[0]
    calibrated = calibrate_persistent_step_l2(
        contract,
        source_registry=labels,
        losses={name: 1.0 for name in labels},
        gradient_l2_norms={name: 2.0 for name in labels},
        direction_cosines={name: 0.25 for name in labels},
        trainable_l2_norm=2_000.0,
    )
    assert calibrated["usable"] is True
    assert calibrated["caps"]["linearized_response"] == pytest.approx(0.04)
    assert calibrated["caps"]["absolute_l2"] == pytest.approx(0.01)
    assert calibrated["caps"]["relative_l2"] == pytest.approx(0.02)
    assert calibrated["full_parameter_delta_l2"] == pytest.approx(0.01)
    assert calibrated["limiting_cap"] == "absolute_l2"


def test_conditioned_persistent_classification_separates_transfer_and_stops():
    assert (
        classify_persistent_smoke(
            completed_logical_update=5,
            stopped_reason=None,
            final_absolute_science_passed=False,
        )
        == "conditioned_attention_three_update_smoke_transfers_absolute_open"
    )
    assert (
        classify_persistent_smoke(
            completed_logical_update=4,
            stopped_reason="monitor_transfer",
            final_absolute_science_passed=None,
        )
        == "conditioned_attention_persistent_smoke_stopped_monitor_transfer"
    )
    with pytest.raises(ValueError):
        classify_persistent_smoke(
            completed_logical_update=5,
            stopped_reason="train_only",
            final_absolute_science_passed=False,
        )


def test_conditioned_start_recovery_and_checkpoint_roundtrip(tmp_path: Path):
    identity = {"contract": "frozen"}
    trainable = {"attention": np.asarray([1.0, 2.0, 3.0], dtype=np.float32)}
    start = tmp_path / "start.pkl"
    written = write_start_snapshot(
        start,
        identity=identity,
        trainable=trainable,
        parameter_count=3,
    )
    loaded = load_start_snapshot(start, identity=identity, parameter_count=3)
    assert written["completed_logical_update"] == START_COMPLETED_LOGICAL_UPDATE
    assert parameter_trees_identical(trainable, loaded["trainable"])

    persistent_identity = dict(identity, start_snapshot_sha256=sha256_file(start))
    recovery = tmp_path / "recovery.pkl"
    write_recovery(
        recovery,
        identity=persistent_identity,
        trainable=trainable,
        completed_logical_update=START_COMPLETED_LOGICAL_UPDATE,
        parameter_count=3,
        update_records=(),
        initial_monitor=_panel(),
        update_monitors=(),
    )
    initial = load_recovery(recovery, identity=persistent_identity, parameter_count=3)
    assert initial is not None
    assert initial["completed_logical_update"] == 2

    write_recovery(
        recovery,
        identity=persistent_identity,
        trainable=trainable,
        completed_logical_update=4,
        parameter_count=3,
        update_records=_records(4),
        initial_monitor=_panel(),
        update_monitors=_monitors(4),
    )
    resumed = load_recovery(recovery, identity=persistent_identity, parameter_count=3)
    assert resumed is not None
    assert resumed["completed_logical_update"] == 4
    assert "optimizer" not in resumed
    assert "fixed_parameters" not in resumed

    checkpoint = tmp_path / "checkpoint-update-0005.pkl"
    write_checkpoint(
        checkpoint,
        identity=persistent_identity,
        trainable=trainable,
        completed_logical_update=FINAL_LOGICAL_UPDATE,
        parameter_count=3,
        update_records=_records(FINAL_LOGICAL_UPDATE),
        initial_monitor=_panel(),
        update_monitors=_monitors(FINAL_LOGICAL_UPDATE),
    )
    restored = load_checkpoint(
        checkpoint,
        expected_sha256=sha256_file(checkpoint),
        identity=persistent_identity,
        parameter_count=3,
    )
    assert restored["completed_logical_update"] == FINAL_LOGICAL_UPDATE


def test_persistence_rejects_monitor_coverage_gap(tmp_path: Path):
    with pytest.raises(ValueError, match="monitor coverage"):
        write_recovery(
            tmp_path / "recovery.pkl",
            identity={"contract": "frozen"},
            trainable={"attention": np.asarray([1.0], dtype=np.float32)},
            completed_logical_update=3,
            parameter_count=1,
            update_records=_records(3),
            initial_monitor=_panel(),
            update_monitors=(),
        )


def test_runner_checks_monitor_before_retaining_candidate_and_recalibrates_each_update():
    update_source = inspect.getsource(runner._train_only_update)
    run_source = inspect.getsource(runner.run)
    loop_source = run_source[run_source.index("for logical_update") :]
    assert update_source.index("initial_science = _evaluate_detailed_monitor") < update_source.index(
        "scheduled = _measure_source_gradients"
    )
    assert "calibrate_persistent_step_l2" in update_source
    assert "replace_conditioned_coupled_gpp_trainable" in update_source
    assert loop_source.index("candidate_monitor = _monitor") < loop_source.index("if not monitor_accepted")
    assert loop_source.index("if not monitor_accepted") < loop_source.index("trainable = candidate_trainable")
    assert "initialize_target_oriented_adamw" not in update_source


def test_conditioned_persistent_launcher_supplies_its_actual_parser_branch():
    root = Path(__file__).resolve().parents[2]
    wrapper = (
        root / "scripts/hpc/run_target_oriented_coupled_gpp_conditioned_attention_persistent_training.sh"
    ).read_text(encoding="utf-8")
    shared = (root / "scripts/hpc/run_target_oriented_coupled_gpp_multi_group_direction.sh").read_text(encoding="utf-8")
    required = {
        option
        for action in runner._parser()._actions
        if action.required
        for option in action.option_strings
        if option.startswith("--")
    }
    module_branch_start = shared.index(
        'elif [[ "$PYTHON_MODULE" == '
        '"scripts.hpc.run_target_oriented_coupled_gpp_conditioned_attention_persistent_training"'
    )
    module_branch_end = shared.index("elif [[", module_branch_start + 8)
    module_branch = shared[module_branch_start:module_branch_end]
    common_command = shared[shared.index('"$PYTHON" -m "$PYTHON_MODULE"') :]
    missing = sorted(option for option in required if option not in module_branch and option not in common_command)
    assert missing == []
    for variable in (
        "PERSISTENT_TRAINING_REPORT",
        "PERSISTENT_ARTIFACT_AWARE_RECOVERY",
        "CONDITIONING_CONTRACT",
        "CONDITIONED_ADMISSION_RESULT",
        "CONDITIONED_DISCARDED_UPDATE_CONTRACT",
        "CONDITIONED_DISCARDED_UPDATE_RESULT",
        "CONDITIONED_PERSISTENT_CONTRACT",
    ):
        assert f"export {variable}=" in wrapper
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-240m}" in wrapper
    assert "sbatch" not in wrapper
    assert "sbatch" not in shared
