from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from research.daily_coarse_graining.daily_operator.target_oriented_training import (
    one_day_state_anchor_from_sufficient_statistics,
    one_day_state_sufficient_statistics,
)
from research.daily_coarse_graining.daily_operator.types import StateDefinedMasks, StateFeatureGroups
from scripts.hpc.shared_state_anchor_diagnostics import state_coordinate_owners, summarize_state_anchor


def test_attribution_recovers_loss_with_masked_nan_and_cross_field_bias():
    names = StateFeatureGroups._fields
    predicted = StateFeatureGroups(**{name: np.array([3.0, -1.0, np.nan]) for name in names})
    expected = StateFeatureGroups(**{name: np.zeros(3) for name in names})
    defined = StateDefinedMasks(**{name: np.array([True, True, False]) for name in names})
    scales = SimpleNamespace(
        state=StateFeatureGroups(**{name: np.ones(3) for name in names}),
        state_loss_eligible=defined,
    )
    owners = {name: [("a", (0,)), ("b", (0,)), ("c", (0,))] for name in names}
    payload = dict(predicted=predicted, expected=expected,
                   predicted_defined=defined, expected_defined=defined)
    result = summarize_state_anchor(payload, scales, owners)
    actual = one_day_state_anchor_from_sufficient_statistics(one_day_state_sufficient_statistics(
        predicted, expected, defined, defined, scales.state, eligible=defined))
    np.testing.assert_allclose(result["total"], actual.total, rtol=1e-12)
    for group in result["groups"].values():
        assert group["count"] == 2
        assert len(group["largest_coordinates"]) == 2
        np.testing.assert_allclose(sum(field["loss_contribution"] for field in group["fields"]), group["loss"])
        field_b = next(field for field in group["fields"] if field["field"] == "b")
        assert field_b["bias_squared_contribution"] < 0


def test_canonical_owner_map_matches_all_seven_state_group_shapes():
    from tests.unit.test_daily_operator_mechanistic_output import mechanistic_setup

    with (
        patch("tests.unit.test_daily_operator_successor_process_head.initialize_successor_parameters",
              return_value={}),
        patch("tests.unit.test_daily_operator_mechanistic_output.initialize_strong_baseline_parameters",
              return_value={}),
    ):
        setup = mechanistic_setup.__wrapped__()
    owners = state_coordinate_owners(setup["head"].contract)
    for name in StateFeatureGroups._fields:
        assert len(owners[name]) == np.size(getattr(setup["assembled"].state, name)), name
        assert all(owner is not None for owner in owners[name]), name
