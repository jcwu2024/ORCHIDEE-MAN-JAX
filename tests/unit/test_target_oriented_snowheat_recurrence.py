from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining.daily_markov_contract import StateLeafSpec
from research.daily_coarse_graining.daily_operator.target_oriented_snowheat_recurrence import (
    FROZEN_CONTRACT_SHA256,
    SNOWHEAT_STATE_KEY,
    WINDOW_NAMES,
    classify_snowheat_recurrence,
    load_target_oriented_snowheat_recurrence_contract,
    replace_state_leaf_with_teacher,
    resolve_continuous_state_leaf,
)


def _leaf(*, key: str = SNOWHEAT_STATE_KEY, start: int | None = 2, stop: int | None = 5):
    component, field = key.split(".")
    return StateLeafSpec(
        component=component,
        path=(field,),
        shape=(3,),
        full_shape=(1, 3),
        dtype="float64",
        start=start,
        stop=stop,
        classification="continuous",
        source_ref="src_sechiba/explicitsnow.f90 snowheat/snowtemp",
        axis_names=("npts", "nsnow"),
    )


def _records(*, seven_gradient: float, thirty_gradient: float, head_factor: float = 1.0):
    records = {}
    for name in WINDOW_NAMES:
        if name == "seven_day_extreme":
            ratio = seven_gradient
        elif name == "thirty_day_extreme":
            ratio = thirty_gradient
        else:
            ratio = 1.0
        records[name] = {
            "gradient_reduction_ratio": ratio,
            "snowheat_head_reduction_ratio": ratio * head_factor,
        }
    return records


def test_frozen_contract_binds_four_windows_and_diagnostic_boundary() -> None:
    contract = load_target_oriented_snowheat_recurrence_contract()
    assert contract.sha256 == FROZEN_CONTRACT_SHA256
    assert tuple(record["name"] for record in contract.windows) == WINDOW_NAMES
    assert contract.payload["intervention"]["state_key"] == SNOWHEAT_STATE_KEY
    assert contract.payload["intervention"]["allowed_in_final_inference"] is False


def test_state_leaf_is_resolved_by_key_and_only_its_coordinates_are_replaced() -> None:
    leaf = _leaf()
    contract = SimpleNamespace(state_leaves=(leaf,))
    assert resolve_continuous_state_leaf(contract, SNOWHEAT_STATE_KEY) is leaf
    predicted = jnp.arange(8.0, dtype=jnp.float64)
    teacher = jnp.arange(100.0, 108.0, dtype=jnp.float64)
    intervened = np.asarray(replace_state_leaf_with_teacher(predicted, teacher, leaf))
    np.testing.assert_array_equal(intervened[:2], np.asarray(predicted)[:2])
    np.testing.assert_array_equal(intervened[2:5], np.asarray(teacher)[2:5])
    np.testing.assert_array_equal(intervened[5:], np.asarray(predicted)[5:])


def test_state_leaf_resolution_fails_for_missing_or_discrete_owner() -> None:
    with pytest.raises(ValueError, match="owner drift"):
        resolve_continuous_state_leaf(SimpleNamespace(state_leaves=()), SNOWHEAT_STATE_KEY)
    with pytest.raises(ValueError, match="not a continuous slice"):
        resolve_continuous_state_leaf(
            SimpleNamespace(state_leaves=(_leaf(start=None, stop=None),)),
            SNOWHEAT_STATE_KEY,
        )


def test_classifier_requires_both_extreme_windows() -> None:
    contract = load_target_oriented_snowheat_recurrence_contract()
    causal = classify_snowheat_recurrence(
        _records(seven_gradient=20.0, thirty_gradient=12.0),
        contract,
    )
    assert causal["classification"] == "snowheat_recurrence_causal"
    partial = classify_snowheat_recurrence(
        _records(seven_gradient=20.0, thirty_gradient=3.0),
        contract,
    )
    assert partial["classification"] == "snowheat_recurrence_partial"
    absent = classify_snowheat_recurrence(
        _records(seven_gradient=1.5, thirty_gradient=1.0),
        contract,
    )
    assert absent["classification"] == "snowheat_recurrence_not_causal"


def test_launcher_and_step_factory_keep_the_intervention_diagnostic_only() -> None:
    root = Path(__file__).resolve().parents[2]
    launcher = (root / "scripts" / "hpc" / "run_target_oriented_snowheat_recurrence_intervention.sh").read_text(
        encoding="utf-8"
    )
    assert "gln01|ibgln01|ibc13b02n04" in launcher
    assert "sbatch" not in launcher
    assert "audit_target_oriented_snowheat_recurrence" in launcher
    assert "target-oriented-flux-gradient-path-53269af/report.json" in launcher
    source = (root / "scripts" / "hpc" / "run_target_oriented_full_step_preflight.py").read_text(encoding="utf-8")
    assert "diagnostic_teacher_carry_state_key: str | None = None" in source
    assert '"seven_teacher_carry_flux_field_gradient"' in source
    assert '"thirty_teacher_carry_flux_field_gradient"' in source
