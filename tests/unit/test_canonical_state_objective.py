from __future__ import annotations

import numpy as np
import pytest

from research.daily_coarse_graining.canonical_state_objective import (
    STATE_PROCESS_GROUPS,
    stabilized_state_delta_scale,
    state_process_weighting_from_contract,
)
from research.daily_coarse_graining.markov_dataset import (
    FiniteColumnStatistics,
    TrainingStatistics,
)


def _leaf(component, field, start, stop):
    return {
        "component": component,
        "path": [field],
        "start": start,
        "stop": stop,
    }


def _contract():
    components = (
        ("slowproc_stomate_previous_step_state", "npp_daily"),
        ("slowproc_stomate_previous_step_state", "biomass"),
        ("slowproc_stomate_previous_step_state", "litterpart"),
        ("slowproc_stomate_previous_step_state", "DOC"),
        ("slowproc_stomate_previous_step_state", "t2m_week"),
        ("hydrol_previous_step_state", "mc"),
        ("thermosoil_previous_step_state", "temp_sol_beg"),
        ("diffuco_previous_step_state", "rveget"),
    )
    return {
        "continuous_state_width": len(components),
        "state_leaves": [
            _leaf(component, field, index, index + 1)
            for index, (component, field) in enumerate(components)
        ],
    }


def _statistics(state_scale, delta_scale):
    arrays = {}
    for name, scale in (("state", state_scale), ("state_delta", delta_scale)):
        scale = np.asarray(scale, dtype=np.float64)
        arrays[name] = FiniteColumnStatistics(
            count=np.full(scale.shape, 10, dtype=np.uint64),
            mean=np.zeros_like(scale),
            variance=scale**2,
            scale=scale,
        )
    return TrainingStatistics("test", "test", "test", 10, ("test",), {}, arrays)


def test_state_process_weights_cover_contract_and_balance_groups():
    result = state_process_weighting_from_contract(_contract())

    assert tuple(group["id"] for group in result.metadata["groups"]) == (
        STATE_PROCESS_GROUPS
    )
    np.testing.assert_allclose(result.weights, np.full((8,), 1.0 / 8.0))
    assert len(result.sha256) == 64


def test_state_process_weights_reject_unknown_or_incomplete_contract():
    contract = _contract()
    contract["state_leaves"][-1]["component"] = "unknown_state"
    with pytest.raises(ValueError, match="no process owner"):
        state_process_weighting_from_contract(contract)

    contract = _contract()
    contract["continuous_state_width"] += 1
    with pytest.raises(ValueError, match="miss 1 scalar"):
        state_process_weighting_from_contract(contract)


def test_state_delta_scale_has_state_relative_floor_and_audit():
    scale, audit = stabilized_state_delta_scale(
        _statistics([100.0, 2.0, 5.0], [1.0e-8, 0.5, 0.01]),
        floor_ratio=1.0e-3,
    )

    np.testing.assert_allclose(scale, [0.1, 0.5, 0.01])
    assert audit["raised_columns"] == 1
    assert audit["floor_ratio"] == 1.0e-3

    with pytest.raises(ValueError, match="floor ratio"):
        stabilized_state_delta_scale(
            _statistics([1.0], [1.0]), floor_ratio=0.0
        )


@pytest.mark.parametrize(
    ("state_scale", "delta_scale"),
    (([0.0], [1.0]), ([1.0], [0.0]), ([np.nan], [1.0]), ([1.0], [np.inf])),
)
def test_state_delta_scale_rejects_invalid_normalizers(state_scale, delta_scale):
    with pytest.raises(ValueError, match="finite and positive"):
        stabilized_state_delta_scale(
            _statistics(state_scale, delta_scale), floor_ratio=1.0e-3
        )
