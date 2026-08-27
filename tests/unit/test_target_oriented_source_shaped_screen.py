from __future__ import annotations

import math
from pathlib import Path

import pytest

from research.daily_coarse_graining.daily_operator.target_oriented_source_shaped_screen import (
    FROZEN_SHA256,
    compare_source_shaped_train_train_groups,
    load_source_shaped_train_train_screen_contract,
)


def _key(score: float, *, passed: bool = True):
    return {
        "normalized_rmse": math.sqrt(2.0 * score),
        "normalized_signed_bias": 0.0,
        "passed": passed,
    }


def _groups(contract, *, maintenance_factor: float, protected_factor: float):
    keys = ("maintenance_respiration", *contract.science_gate["protected_keys"])
    result = []
    for index, update in enumerate(contract.evaluation_updates):
        acceptance = {}
        for horizon in ("7", "30"):
            values = {
                name: _key(
                    1.0
                    * (
                        maintenance_factor
                        if name == "maintenance_respiration"
                        else protected_factor
                    )
                )
                for name in keys
            }
            acceptance[horizon] = {
                "all_key_gates_passed": True,
                "keys": values,
            }
        result.append(
            {
                "update": update,
                "runtime_year": 1961 + index,
                "hard_failure_count": 0,
                "acceptance": acceptance,
            }
        )
    return result


def test_source_shaped_screen_contract_is_frozen_and_disjoint():
    contract = load_source_shaped_train_train_screen_contract()
    assert contract.sha256 == FROZEN_SHA256
    assert contract.evaluation_updates == (251, 358, 465, 572, 679, 786, 893, 1000)

    root = Path(__file__).resolve().parents[2]
    launcher = (root / "scripts/hpc/run_target_oriented_source_shaped_train_train_zero_panel.sh").read_text()
    manifest = "manifests/coarse_graining/target_oriented_flux_basis_amendment_v1.json"
    assert manifest in launcher
    assert (root / manifest).is_file()


def test_source_shaped_one_update_admission_is_ephemeral_and_branch_only():
    root = Path(__file__).resolve().parents[2]
    runner = (
        root
        / "scripts"
        / "hpc"
        / "run_target_oriented_source_shaped_one_update_admission.py"
    ).read_text(encoding="utf-8")
    launcher = (
        root
        / "scripts"
        / "hpc"
        / "run_target_oriented_source_shaped_one_update_admission.sh"
    ).read_text(encoding="utf-8")

    assert "_effective_gradient" in runner
    assert "science_flux_basis=amendment.flux_basis" in runner
    assert "initialize_target_oriented_adamw(branches)" in runner
    assert "target_oriented_adamw_update" in runner
    assert '"training_updates_executed": 1' in runner
    assert '"training_updates_persisted": 0' in runner
    assert '"checkpoint_written": False' in runner
    assert "write_target_oriented_stage_checkpoint" not in runner
    assert "write_target_oriented_recovery" not in runner
    assert "free gln01" in launcher
    assert "sbatch" not in launcher
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-120m}" in launcher

    from scripts.hpc.run_target_oriented_source_shaped_one_update_admission import (
        _replace_source_shaped_branches,
    )

    retained_root = object()
    retained_target = object()
    before = {
        "retained_root": retained_root,
        "target_oriented": {
            "retained_target": retained_target,
            "source_shaped_maintenance_heads": {"old": object()},
        },
    }
    branches = {"aboveground": object(), "belowground": object()}
    after = _replace_source_shaped_branches(before, branches)
    assert after["retained_root"] is retained_root
    assert after["target_oriented"]["retained_target"] is retained_target
    assert after["target_oriented"]["source_shaped_maintenance_heads"] is branches


def test_source_shaped_branch_autodiff_admission_is_free_and_ephemeral():
    root = Path(__file__).resolve().parents[2]
    runner = (
        root
        / "scripts"
        / "hpc"
        / "run_target_oriented_source_shaped_branch_autodiff_admission.py"
    ).read_text(encoding="utf-8")
    launcher = (
        root
        / "scripts"
        / "hpc"
        / "run_target_oriented_source_shaped_branch_autodiff_admission.sh"
    ).read_text(encoding="utf-8")
    preflight = (
        root / "scripts" / "hpc" / "run_target_oriented_full_step_preflight.py"
    ).read_text(encoding="utf-8")

    assert 'steps["make_trainable_gradient_steps"]' in runner
    assert "gradient_parameters=branches" in runner
    assert "gradient_steps=direct_steps" in runner
    assert "def make_trainable_gradient_steps" in preflight
    assert '"training_updates_persisted": 0' in runner
    assert '"checkpoint_written": False' in runner
    assert '"recovery_written": False' in runner
    assert "write_target_oriented_stage_checkpoint" not in runner
    assert "write_target_oriented_recovery" not in runner
    assert "free gln01" in launcher
    assert "sbatch" not in launcher
    assert "RUN_TIMEOUT=${RUN_TIMEOUT:-180m}" in launcher


def test_effective_gradient_requires_paired_trainable_overrides():
    from scripts.hpc.run_target_oriented_full_step_preflight import _effective_gradient

    with pytest.raises(ValueError, match="differentiated parameters and steps"):
        _effective_gradient(
            None,
            None,
            {},
            None,
            {},
            None,
            gradient_parameters={},
        )


def test_source_shaped_relative_gate_requires_large_protected_improvement():
    contract = load_source_shaped_train_train_screen_contract()
    parent = _groups(contract, maintenance_factor=1.0, protected_factor=1.0)
    candidate = _groups(contract, maintenance_factor=0.75, protected_factor=0.99)
    report = compare_source_shaped_train_train_groups(parent, candidate, contract)
    assert report["relative_source_gate_passed"]
    assert report["scientific_promotion_gate_passed"]
    assert report["median_maintenance_score_reduction"] == {
        "7": pytest.approx(0.25),
        "30": pytest.approx(0.25),
    }

    regressed = _groups(contract, maintenance_factor=0.75, protected_factor=1.02)
    failed = compare_source_shaped_train_train_groups(parent, regressed, contract)
    assert not failed["protected_score_gate_passed"]
    assert not failed["relative_source_gate_passed"]
