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
