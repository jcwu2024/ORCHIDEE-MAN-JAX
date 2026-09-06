from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_conditioned_attention_discarded_update import (
    FROZEN_CONTRACT_SHA256,
    TRAINABLE_PARAMETER_COUNT,
    absolute_science_gate,
    calibrate_full_step_l2,
    classify_discarded_update,
    load_conditioned_attention_discarded_update_contract,
    monitor_transfer_gate,
    science_progress_gate,
)
from research.daily_coarse_graining.daily_operator.target_oriented_coupled_gpp_objective import (
    DIRECT_SOURCE,
    GRADIENT_SOURCES,
)
from scripts.hpc import audit_target_oriented_coupled_gpp_conditioned_attention_discarded_update as runner


def _panel(contract, *, multiplier: float = 1.0, direct_artifact_multiplier: float | None = None):
    direct_multiplier = multiplier if direct_artifact_multiplier is None else direct_artifact_multiplier
    acceptance = {}
    for horizon in ("7", "30"):
        acceptance[horizon] = {
            "keys": {
                name: {
                    "normalized_rmse": 0.1 * multiplier,
                    "normalized_signed_bias": 0.02 * multiplier,
                }
                for name in contract.required_science_keys
            }
        }
    return {
        "acceptance": acceptance,
        "source_mean_losses": {name: 1.0 * multiplier for name in GRADIENT_SOURCES},
        "records": [
            {
                "artifact_update": update,
                "source_losses": {name: 1.0 * direct_multiplier for name in GRADIENT_SOURCES},
            }
            for update in contract.monitor_updates
        ],
        "hard_gate_passed": True,
    }


def test_discarded_update_contract_freezes_exact_update2_start_and_update3_panel():
    contract = load_conditioned_attention_discarded_update_contract()
    assert contract.sha256 == FROZEN_CONTRACT_SHA256
    assert len(contract.train_groups) == 5
    assert len(contract.train_artifacts) == len(set(contract.train_artifacts)) == 40
    assert len(contract.source_registry) == 43
    assert contract.train_groups[0] == (655, 783, 959, 302, 550, 705, 261, 580)
    assert set(contract.train_artifacts).isdisjoint(contract.monitor_updates)
    assert contract.payload["starting_state"]["required_completed_logical_update"] == 2
    assert contract.payload["starting_state"]["rejected_update3_applied"] is False
    assert contract.payload["step_calibration"]["old_parent_full_step_l2_inherited"] is False
    assert contract.payload["identity"]["candidate_trainable_parameter_count"] == TRAINABLE_PARAMETER_COUNT
    assert not any(contract.payload["retention"].values())
    assert contract.payload["execution"]["paid_partition_authorized"] is False


def test_architecture_aware_step_calibration_uses_strictest_frozen_cap():
    contract = load_conditioned_attention_discarded_update_contract()
    labels = contract.source_registry
    calibrated = calibrate_full_step_l2(
        contract,
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
    assert calibrated["maximum_predicted_relative_reduction"] == pytest.approx(0.005)


def test_science_and_monitor_gates_separate_progress_from_absolute_acceptance():
    contract = load_conditioned_attention_discarded_update_contract()
    initial = _panel(contract)
    improved = _panel(contract, multiplier=0.99, direct_artifact_multiplier=0.99)
    progress = science_progress_gate(contract, initial, improved, gate_name="train_only_selection")
    absolute = absolute_science_gate(contract, improved)
    monitor = monitor_transfer_gate(contract, initial, improved)
    assert progress["passed"] is True
    assert absolute["passed"] is True
    assert monitor["passed"] is True
    assert monitor["nonregressing_direct_gpp_artifact_count"] == 8

    mixed = _panel(contract, multiplier=0.99, direct_artifact_multiplier=1.01)
    failed = monitor_transfer_gate(contract, initial, mixed)
    assert failed["passed"] is False
    assert failed["checks"]["minimum_nonregressing_direct_gpp_artifacts"] is False


def test_discarded_update_classification_requires_monitor_transfer_before_absolute_claim():
    assert (
        classify_discarded_update(
            common_first_order_direction=True,
            calibration_usable=True,
            selected_scale=0.5,
            monitor_passed=True,
            absolute_science_passed=False,
        )
        == "conditioned_discarded_update_transfers_but_absolute_science_remains_open"
    )
    assert (
        classify_discarded_update(
            common_first_order_direction=False,
            calibration_usable=False,
            selected_scale=None,
            monitor_passed=None,
            absolute_science_passed=None,
        )
        == "no_conditioned_common_direction"
    )
    with pytest.raises(ValueError):
        classify_discarded_update(
            common_first_order_direction=False,
            calibration_usable=True,
            selected_scale=None,
            monitor_passed=None,
            absolute_science_passed=None,
        )


def test_runner_rebuilds_conditioned_steps_and_keeps_monitor_after_selection():
    source = inspect.getsource(runner.run)
    paired_source = inspect.getsource(runner._paired_monitor)
    assert "replace_conditioned_coupled_gpp_trainable" in inspect.getsource(runner._conditioned_bootstrap)
    assert "expected_parameter_count=TRAINABLE_PARAMETER_COUNT" in source
    assert source.index("initial_science = _evaluate_detailed_monitor") < source.index(
        "scheduled = _measure_source_gradients"
    )
    assert source.index("selected_scale = None if selected_record is None") < source.index(
        "paired_monitor = _paired_monitor"
    )
    assert "write_recovery" not in source
    assert "write_checkpoint" not in source
    assert "initialize_target_oriented_adamw" not in source
    assert "loaded = tuple" not in paired_source
    assert paired_source.count("_load_artifact(") == 1
    assert paired_source.index("artifact = _load_artifact") < paired_source.index(
        "before_record, before_statistics"
    )
    assert paired_source.index("before_record, before_statistics") < paired_source.index(
        "after_record, after_statistics"
    )
    assert "del artifact" in paired_source


def test_paired_monitor_evaluates_both_states_before_loading_the_next_artifact(monkeypatch):
    events = []

    def load_artifact(_args, _context, update):
        events.append(("load", update))
        return update

    def evaluate_artifact(
        _context,
        *,
        update,
        artifact,
        parameters,
        trainable,
        bootstrap,
        label,
    ):
        del parameters, trainable, bootstrap
        assert artifact == update
        events.append((label, update))
        return {"artifact_update": update}, {"7": object(), "30": object()}

    def finalize_panel(_context, *, records, acceptance_totals, bootstrap, label):
        del acceptance_totals, bootstrap
        return {"label": label, "records": list(records)}

    monkeypatch.setattr(runner, "_load_artifact", load_artifact)
    monkeypatch.setattr(runner, "_evaluate_loaded_detailed_artifact", evaluate_artifact)
    monkeypatch.setattr(runner, "_add_acceptance_statistics", lambda _totals, _statistics: None)
    monkeypatch.setattr(runner, "_finalize_loaded_detailed_panel", finalize_panel)
    monkeypatch.setattr(
        runner,
        "replace_conditioned_coupled_gpp_trainable",
        lambda _parameters, trainable: trainable,
    )

    result = runner._paired_monitor(
        None,
        {},
        before_trainable="before",
        after_trainable="after",
        bootstrap={"initialized_parameters": object()},
        updates=(251, 358),
    )

    assert events == [
        ("load", 251),
        ("fixed_monitor_initial", 251),
        ("fixed_monitor_candidate", 251),
        ("load", 358),
        ("fixed_monitor_initial", 358),
        ("fixed_monitor_candidate", 358),
    ]
    assert result["artifact_load_count"] == 2
    assert result["each_artifact_loaded_once"] is True


def test_free_launcher_supplies_every_required_argument_without_sbatch():
    root = Path(__file__).resolve().parents[2]
    wrapper = (
        root / "scripts/hpc/run_target_oriented_coupled_gpp_conditioned_attention_discarded_update.sh"
    ).read_text(encoding="utf-8")
    shared = (root / "scripts/hpc/run_target_oriented_coupled_gpp_multi_group_direction.sh").read_text(
        encoding="utf-8"
    )
    required = {
        option
        for action in runner._parser()._actions
        if action.required
        for option in action.option_strings
        if option.startswith("--")
    }
    module_branch_start = shared.index(
        'if [[ "$PYTHON_MODULE" == '
        '"scripts.hpc.audit_target_oriented_coupled_gpp_conditioned_attention_discarded_update"'
    )
    module_branch_end = shared.index("elif [[", module_branch_start)
    module_branch = shared[module_branch_start:module_branch_end]
    common_command = shared[shared.index('"$PYTHON" -m "$PYTHON_MODULE"') :]
    assert sorted(option for option in required if option not in module_branch and option not in common_command) == []
    for variable in (
        "ARTIFACT_AWARE_CONTRACT",
        "MULTI_GROUP_RESULT",
        "CONTINUITY_CONTRACT",
        "ARTIFACT_AWARE_RESULT",
        "ARTIFACT_AWARE_PERSISTENT_CONTRACT",
        "CONTINUITY_RESULT",
        "PERSISTENT_TRAINING_REPORT",
        "PERSISTENT_ARTIFACT_AWARE_RECOVERY",
        "CONDITIONING_CONTRACT",
        "CONDITIONED_ADMISSION_RESULT",
        "CONDITIONED_DISCARDED_UPDATE_CONTRACT",
    ):
        assert f"export {variable}=" in wrapper
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-180m}" in wrapper
    assert "sbatch" not in wrapper
    assert "sbatch" not in shared
    assert "CONDITIONED_ADMISSION_RESULT" in wrapper
    assert "CONDITIONED_DISCARDED_UPDATE_CONTRACT" in wrapper
    assert DIRECT_SOURCE == "direct_gpp_208_teacher_days"
