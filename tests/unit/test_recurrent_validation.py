from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from research.daily_coarse_graining.daily_operator.recurrent_validation import (
    PHYSICAL_COUNT_NAMES,
    feedback_counts,
    physical_counts,
)
from scripts.hpc.probe_public_rollout import physical_checks
from scripts.hpc.run_trained_forward_screen import feedback_hard_counts
from tests.unit.test_daily_operator_successor_process_head import _markov_contract


def _host_counts(values):
    return {name: int(np.asarray(value)) for name, value in values.items()}


def _invalid_physical_case():
    contract = _markov_contract()
    width = contract.continuous_state_width
    days = 2
    state = np.zeros((days, width), dtype=np.float64)
    observed_defined = np.ones_like(state, dtype=bool)
    target_defined = np.ones_like(state, dtype=bool)

    mismatch_index = 0
    observed_defined[0, mismatch_index] = False
    gsmean = next(
        leaf for leaf in contract.state_leaves
        if leaf.key == "diffuco_previous_step_state.gsmean"
    )
    state[0, gsmean.start] = 0.25
    state[1, gsmean.start + 1] = np.nan
    biomass = next(
        leaf for leaf in contract.state_leaves
        if leaf.key == "slowproc_stomate_previous_step_state.biomass"
    )
    state[0, biomass.start : biomass.stop] = -1.0

    fraction = np.asarray([[0.25, 0.75], [np.nan, 0.75]], dtype=np.float64)
    gpp = np.zeros((days, 1, 2), dtype=np.float64)
    gpp[1, 0, 1] = np.inf
    obs = {
        "state": state,
        "defined": observed_defined,
        "gpp": gpp,
        "gpp_defined": np.ones_like(gpp, dtype=bool),
        "active": np.ones((days, 2), dtype=bool),
        "fraction": fraction,
        "budgets": np.asarray([[0.0, 3e-12], [np.nan, -4e-12]]),
        "discrete": {"flag": np.asarray([0, 1])},
        "learned_output_nonfinite": np.asarray([2, 3], dtype=np.int32),
    }
    targets = {"defined": target_defined}
    expected_discrete = {"flag": np.asarray([1, 1])}
    traits = np.zeros((days, 2, 1), dtype=np.float64)
    traits[:, 0, 0] = 1.0
    templates = SimpleNamespace(pft=SimpleNamespace(pft_traits=traits))
    dynamic = np.ones(width, dtype=bool)
    dynamic[mismatch_index] = False
    scales = SimpleNamespace(
        state_declared_dynamic=SimpleNamespace(other=dynamic)
    )
    return obs, targets, expected_discrete, templates, contract, scales


def test_physical_counts_match_legacy_for_each_nonzero_failure_family():
    args = _invalid_physical_case()
    legacy = physical_checks(*args)
    compact = physical_counts(*args)

    assert PHYSICAL_COUNT_NAMES == tuple(legacy)
    assert tuple(compact) == PHYSICAL_COUNT_NAMES
    assert _host_counts(compact) == legacy
    assert all(value > 0 for value in legacy.values())


def test_physical_counts_are_jittable_scalar_integer_arrays():
    obs, targets, expected_discrete, templates, contract, scales = _invalid_physical_case()
    compiled = jax.jit(
        lambda current_obs, current_targets, current_discrete: physical_counts(
            current_obs,
            current_targets,
            current_discrete,
            templates,
            contract,
            scales,
        )
    )(obs, targets, expected_discrete)

    assert _host_counts(compiled) == physical_checks(
        obs, targets, expected_discrete, templates, contract, scales
    )
    assert all(isinstance(value, jax.Array) for value in compiled.values())
    assert all(jnp.issubdtype(value.dtype, jnp.integer) and value.shape == () for value in compiled.values())


def test_feedback_counts_match_legacy_for_nonzero_optional_failures_and_jit():
    feedback = {
        "enabled": True,
        "canopy_stock_change": {
            "row_valid": jnp.asarray([False, True, False]),
            "scale_valid": jnp.asarray(False),
            "scale_pft_applicable": jnp.asarray([True, False, False]),
            "active_slot_count": jnp.asarray([0, 2, 0]),
        },
        "litterpart_feedback": {
            "invalid_count": jnp.asarray([2, 0, 3]),
            "expected_count": jnp.asarray([0, 4, 0]),
        },
    }
    legacy = feedback_hard_counts(feedback)
    compact = feedback_counts(feedback)
    dynamic_feedback = {
        name: value for name, value in feedback.items() if name != "enabled"
    }
    compiled = jax.jit(
        lambda current: feedback_counts({"enabled": True, **current})
    )(dynamic_feedback)

    assert _host_counts(compact) == legacy == {
        "canopy_change_invalid": 5,
        "canopy_change_empty": 2,
        "litterpart_feedback_invalid": 5,
        "litterpart_feedback_empty": 2,
    }
    assert _host_counts(compiled) == legacy
    assert all(isinstance(value, jax.Array) for value in compiled.values())
    assert all(jnp.issubdtype(value.dtype, jnp.integer) and value.shape == () for value in compiled.values())


def test_feedback_counts_preserve_disabled_and_independent_optional_sections():
    assert feedback_counts({"enabled": False}) == {}
    canopy_only = {
        "enabled": True,
        "canopy_stock_change": {
            "row_valid": jnp.asarray([True]),
            "scale_valid": jnp.asarray(True),
            "scale_pft_applicable": jnp.asarray([True]),
            "active_slot_count": jnp.asarray([1]),
        },
    }
    assert _host_counts(feedback_counts(canopy_only)) == feedback_hard_counts(canopy_only)
