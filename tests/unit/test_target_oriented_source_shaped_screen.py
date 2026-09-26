from __future__ import annotations

import math
from pathlib import Path

import pytest



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
                name: _key(1.0 * (maintenance_factor if name == "maintenance_respiration" else protected_factor))
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
