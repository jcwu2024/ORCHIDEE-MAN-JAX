from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from research.daily_coarse_graining.daily_operator.successor_fit_checkpoint import sha256_file
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_artifact_aware_persistence import (
    load_checkpoint,
    load_recovery,
    load_start_snapshot,
    parameter_trees_identical,
    write_checkpoint,
    write_recovery,
    write_start_snapshot,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_artifact_aware_persistent_training import (
    FROZEN_CONTRACT_SHA256,
    LOGICAL_UPDATES,
    PRIMARY_SCIENCE_KEYS,
    TRAIN_PANEL_GROUP_UPDATES,
    artifact_aware_monitor_gate,
    load_coupled_gpp_artifact_aware_persistent_training_contract,
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
from scripts.hpc import run_target_oriented_coupled_gpp_artifact_aware_persistent_training as runner


def _monitor(
    *,
    source_multiplier: float = 1.0,
    primary_multiplier: float = 1.0,
    logical_update: int | None = None,
    passed: bool = True,
) -> dict:
    panel = {
        "artifact_updates": list(MONITOR_UPDATES),
        "records": [{"artifact_update": value} for value in MONITOR_UPDATES],
        "source_mean_losses": {name: source_multiplier for name in GRADIENT_SOURCES},
        "absolute_key_scores": {
            horizon: {name: primary_multiplier if name in PRIMARY_SCIENCE_KEYS else 1.0 for name in ALL_REPORTED_KEYS}
            for horizon in ("7", "30")
        },
        "hard_gate_passed": True,
    }
    if logical_update is not None:
        panel.update(
            logical_update=logical_update,
            gate={
                "logical_update": logical_update,
                "passed": passed,
                "stop_training": not passed,
            },
        )
    return panel


def _records(last_update: int) -> list[dict]:
    return [
        {
            "logical_update": update,
            "selection": {"accepted": True, "selected_scale": 0.5},
        }
        for update in range(2, last_update + 1)
    ]


def test_artifact_aware_persistent_contract_freezes_disjoint_rotating_panels():
    contract = load_coupled_gpp_artifact_aware_persistent_training_contract()
    artifacts = [value for panel in contract.train_panels for group in panel for value in group]
    assert contract.sha256 == FROZEN_CONTRACT_SHA256
    assert contract.logical_updates == LOGICAL_UPDATES
    assert contract.panel_group_updates == TRAIN_PANEL_GROUP_UPDATES
    assert len(artifacts) == len(set(artifacts)) == 240
    assert set(artifacts).isdisjoint(contract.monitor_updates)
    assert tuple(len(registry) for registry in contract.source_registries) == (43,) * 6
    assert contract.payload["execution"]["paid_partition_authorized"] is False


def test_final_monitor_gate_requires_gpp_and_primary_science_improvement():
    contract = load_coupled_gpp_artifact_aware_persistent_training_contract()
    initial = _monitor()
    improved = _monitor(source_multiplier=0.99, primary_multiplier=0.99)
    gate = artifact_aware_monitor_gate(
        contract,
        logical_update=7,
        initial=initial,
        current=improved,
    )
    assert gate["passed"] is True
    assert gate["checks"]["direct_gpp_mean_strictly_improves"] is True
    assert gate["checks"]["primary_science_macro_strictly_improves"] is True

    regressed = _monitor(source_multiplier=1.02, primary_multiplier=0.99)
    failed = artifact_aware_monitor_gate(
        contract,
        logical_update=7,
        initial=initial,
        current=regressed,
    )
    assert failed["passed"] is False
    assert failed["stop_training"] is True
    assert failed["checks"]["non_gpp_source_means_within_final_cap"] is False


def test_exact_start_recovery_and_final_checkpoint_roundtrip(tmp_path: Path):
    base_identity = {"contract": "frozen"}
    trainable = {"attention": np.asarray([1.0, 2.0, 3.0], dtype=np.float32)}
    start = tmp_path / "start.pkl"
    written_start = write_start_snapshot(
        start,
        identity=base_identity,
        trainable=trainable,
        parameter_count=3,
    )
    loaded_start = load_start_snapshot(start, identity=base_identity, parameter_count=3)
    assert written_start["completed_logical_update"] == 1
    assert parameter_trees_identical(trainable, loaded_start["trainable"])
    with pytest.raises(FileExistsError):
        write_start_snapshot(
            start,
            identity=base_identity,
            trainable=trainable,
            parameter_count=3,
        )

    identity = dict(base_identity, start_snapshot_sha256=sha256_file(start))
    recovery = tmp_path / "recovery.pkl"
    write_recovery(
        recovery,
        identity=identity,
        trainable=trainable,
        completed_logical_update=1,
        parameter_count=3,
        update_records=(),
        initial_monitor=_monitor(),
        milestone_monitors=(),
    )
    initial_recovery = load_recovery(recovery, identity=identity, parameter_count=3)
    assert initial_recovery is not None
    assert initial_recovery["completed_logical_update"] == 1

    write_recovery(
        recovery,
        identity=identity,
        trainable=trainable,
        completed_logical_update=4,
        parameter_count=3,
        update_records=_records(4),
        initial_monitor=_monitor(),
        milestone_monitors=(_monitor(logical_update=4),),
    )
    loaded_recovery = load_recovery(recovery, identity=identity, parameter_count=3)
    assert loaded_recovery is not None
    assert loaded_recovery["completed_logical_update"] == 4
    assert "optimizer" not in loaded_recovery
    assert "fixed_parameters" not in loaded_recovery

    checkpoint = tmp_path / "checkpoint-update-0007.pkl"
    write_checkpoint(
        checkpoint,
        identity=identity,
        trainable=trainable,
        completed_logical_update=7,
        parameter_count=3,
        update_records=_records(7),
        initial_monitor=_monitor(),
        milestone_monitors=(
            _monitor(logical_update=4),
            _monitor(logical_update=7),
        ),
    )
    loaded_checkpoint = load_checkpoint(
        checkpoint,
        expected_sha256=sha256_file(checkpoint),
        identity=identity,
        parameter_count=3,
    )
    assert loaded_checkpoint["completed_logical_update"] == 7
    assert parameter_trees_identical(trainable, loaded_checkpoint["trainable"])


def test_persistence_rejects_update_and_milestone_coverage_gaps(tmp_path: Path):
    trainable = {"attention": np.asarray([1.0], dtype=np.float32)}
    common = dict(
        path=tmp_path / "recovery.pkl",
        identity={"contract": "frozen"},
        trainable=trainable,
        completed_logical_update=4,
        parameter_count=1,
        initial_monitor=_monitor(),
    )
    with pytest.raises(ValueError, match="update coverage"):
        write_recovery(
            **common,
            update_records=_records(4)[1:],
            milestone_monitors=(_monitor(logical_update=4),),
        )
    with pytest.raises(ValueError, match="milestone coverage"):
        write_recovery(
            **common,
            update_records=_records(4),
            milestone_monitors=(),
        )


def test_resume_initializes_context_without_reconstructing_history(monkeypatch):
    sentinel = {"context": "initialized"}
    monkeypatch.setattr(runner, "_initialization_only", lambda _args: sentinel)

    def fail_reconstruction(*_args, **_kwargs):
        raise AssertionError("resume attempted historical reconstruction")

    monkeypatch.setattr(runner, "_reconstruct_start", fail_reconstruction)
    bootstrap, start, replay, mode = runner._prepare_bootstrap(
        None,
        start_exists=True,
        parent_contract=None,
        parent_result={},
        schedule=None,
    )
    assert bootstrap is sentinel
    assert start is None
    assert replay is None
    assert mode == "identity_exact_resume"


def test_persistent_launcher_supplies_every_required_cli_argument():
    root = Path(__file__).resolve().parents[2]
    shared = (root / "scripts/hpc/run_target_oriented_coupled_gpp_multi_group_direction.sh").read_text(encoding="utf-8")
    wrapper = (root / "scripts/hpc/run_target_oriented_coupled_gpp_artifact_aware_persistent_training.sh").read_text(
        encoding="utf-8"
    )
    required_options = {
        option
        for action in runner._parser()._actions
        if action.required
        for option in action.option_strings
        if option.startswith("--")
    }
    missing = sorted(option for option in required_options if option not in shared and option not in wrapper)
    assert missing == []
    assert "sbatch" not in wrapper
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-240m}" in wrapper
    assert "ARTIFACT_AWARE_PERSISTENT_CONTRACT" in shared
    assert "CONTINUITY_RESULT" in shared
