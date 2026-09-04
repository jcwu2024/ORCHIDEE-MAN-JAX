from __future__ import annotations

from pathlib import Path

import pytest

from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_artifact_aware_continuity import (
    ARTIFACT_AWARE_RESULT_SHA256,
    FROZEN_CONTRACT_SHA256,
    LOGICAL_UPDATES,
    TRAIN_PANEL_GROUP_UPDATES,
    classify_artifact_aware_continuity,
    load_coupled_gpp_artifact_aware_continuity_contract,
    monitor_trajectory_passed,
)
from scripts.hpc import audit_target_oriented_coupled_gpp_artifact_aware_continuity as runner


def test_continuity_contract_freezes_three_disjoint_artifact_panels():
    contract = load_coupled_gpp_artifact_aware_continuity_contract()
    artifacts = [value for panel in contract.train_panels for group in panel for value in group]
    assert contract.sha256 == FROZEN_CONTRACT_SHA256
    assert contract.logical_updates == LOGICAL_UPDATES
    assert contract.panel_group_updates == TRAIN_PANEL_GROUP_UPDATES
    assert tuple(len(panel) for panel in contract.train_panels) == (5, 5, 5)
    assert tuple(sum(len(group) for group in panel) for panel in contract.train_panels) == (40, 40, 40)
    assert len(artifacts) == len(set(artifacts)) == 120
    assert set(artifacts).isdisjoint(contract.monitor_updates)
    assert tuple(len(registry) for registry in contract.source_registries) == (43, 43, 43)
    assert contract.payload["monitor_evaluation"]["individual_artifact_changes_reported_but_not_gated"] is True
    assert not any(contract.payload["retention"].values())
    assert contract.payload["execution"]["paid_partition_authorized"] is False


def test_continuity_first_panel_repeats_predecessor_then_rotates():
    contract = load_coupled_gpp_artifact_aware_continuity_contract()
    assert contract.panel_group_updates[0] == (2, 11, 12, 13, 14)
    assert contract.panel_group_updates[1] == (3, 15, 16, 17, 18)
    assert contract.panel_group_updates[2] == (4, 19, 20, 21, 22)
    assert contract.train_panels[0][0] == (317, 657, 366, 681, 671, 312, 993, 984)


def test_continuity_classification_requires_all_updates_before_monitor():
    assert (
        classify_artifact_aware_continuity(
            accepted_update_count=2,
            required_update_count=3,
            monitor_relative_changes=None,
        )
        == "artifact_aware_continuity_stops_train_only"
    )
    with pytest.raises(ValueError, match="cannot read the monitor"):
        classify_artifact_aware_continuity(
            accepted_update_count=2,
            required_update_count=3,
            monitor_relative_changes=(-0.1, -0.2),
        )
    assert monitor_trajectory_passed((-0.01, -0.02, -0.03))
    assert not monitor_trajectory_passed((-0.01, 0.001, -0.03))
    assert (
        classify_artifact_aware_continuity(
            accepted_update_count=3,
            required_update_count=3,
            monitor_relative_changes=(-0.01, -0.02, -0.03),
        )
        == "artifact_aware_three_update_continuity_passes_monitor"
    )
    assert (
        classify_artifact_aware_continuity(
            accepted_update_count=3,
            required_update_count=3,
            monitor_relative_changes=(-0.01, 0.001, -0.03),
        )
        == "artifact_aware_three_update_continuity_fails_monitor"
    )


def test_continuity_predecessor_loader_accepts_only_frozen_positive_result(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[2] / "outputs/diagnostics"
    source /= "target-oriented-coupled-gpp-artifact-aware-direction-77d5a45.json"
    if not source.exists():
        pytest.skip("local immutable predecessor report is not installed")
    monkeypatch.setattr(runner, "sha256_file", lambda _path: ARTIFACT_AWARE_RESULT_SHA256)
    contract = load_coupled_gpp_artifact_aware_continuity_contract()
    report = runner._load_artifact_aware_result(
        source,
        contract_sha256=contract.payload["identity"]["artifact_aware_direction_contract_sha256"],
    )
    assert report["selection"]["selected_scale"] == 0.5
    assert report["selection"]["selected_monitor"]["improved_artifact_count"] == 8


def test_monitor_trajectory_builds_three_adjacent_changes_from_four_states(monkeypatch):
    def fake_panel(_args, _context, *, trainable, direct_steps, updates, label):
        del direct_steps, label
        return {
            "mean_loss": float(trainable),
            "artifact_losses": [
                {"artifact_update": int(update), "loss": float(trainable) + index * 0.01}
                for index, update in enumerate(updates)
            ],
        }

    monkeypatch.setattr(runner, "_direct_panel", fake_panel)
    result = runner._monitor_trajectory(
        None,
        {},
        trainables=(4.0, 3.0, 2.0, 1.0),
        direct_steps={},
        monitor_updates=(251, 358),
    )
    assert len(result["state_panels"]) == 4
    assert len(result["relative_to_initial"]) == 3
    assert len(result["consecutive_changes"]) == 3
    assert result["relative_changes"] == pytest.approx((-0.25, -0.5, -0.75))


def test_continuity_runner_defers_monitor_and_never_persists_parameters():
    root = Path(__file__).resolve().parents[2]
    source = (root / "scripts/hpc/audit_target_oriented_coupled_gpp_artifact_aware_continuity.py").read_text(
        encoding="utf-8"
    )
    launcher = (root / "scripts/hpc/run_target_oriented_coupled_gpp_artifact_aware_continuity.sh").read_text(
        encoding="utf-8"
    )
    update_loop = source.index("for logical_update, panel, source_registry in zip(")
    train_only_finished = source.index("accepted_count = len(trainable_trajectory) - 1")
    monitor = source.index("monitor = _monitor_trajectory(")
    assert update_loop < train_only_finished < monitor
    assert "write_coupled_gpp_recovery" not in source
    assert "write_coupled_gpp_checkpoint" not in source
    assert "PYTHON_MODULE=scripts.hpc.audit_target_oriented_coupled_gpp_artifact_aware_continuity" in launcher
    assert "sbatch" not in launcher


def test_continuity_launcher_supplies_every_required_cli_argument():
    root = Path(__file__).resolve().parents[2]
    launcher = (root / "scripts/hpc/run_target_oriented_coupled_gpp_multi_group_direction.sh").read_text(
        encoding="utf-8"
    )
    required_options = {
        option
        for action in runner._parser()._actions
        if action.required
        for option in action.option_strings
        if option.startswith("--")
    }
    missing = sorted(option for option in required_options if option not in launcher)
    assert missing == []
