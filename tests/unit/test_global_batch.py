from dataclasses import replace

import jax
import numpy as np
import pytest

from jax_orchidee.driver import global_batch as batch
from jax_orchidee.driver import orchestration as driver


def lane(name, value=1.0):
    arrays = ({"x": np.asarray([value], dtype=np.float64), "optional": None},)
    args = batch.BlockArguments(
        arrays, np.ones((7, 48, 1)), np.arange(2, 9, dtype=np.int32), *({"x": np.asarray(value)} for _ in range(6))
    )
    return batch.PreparedLane(name, None, None, None, None, args, (("science", 1),))


def test_stack_all_eight_numeric_groups_and_keep_none():
    lanes = [lane("a", 1), lane("b", 2)]
    args = batch.stack_lanes(lanes)
    assert args.values_by_component[0]["optional"] is None
    np.testing.assert_array_equal(args.values_by_component[0]["x"], [[1], [2]])
    assert args.forcing_days.shape == (2, 7, 48, 1)
    assert args.science_day_numbers.shape == (7,)
    for i in range(3, 9):
        np.testing.assert_array_equal(args[i]["x"], [1, 2])


@pytest.mark.parametrize("change", ["static", "dtype", "none", "days", "shape"])
def test_incompatible_lanes_rejected(change):
    a, b = lane("a"), lane("b")
    args = b.arguments
    if change == "static":
        b = replace(b, static_signature=(("science", 2),))
    elif change == "dtype":
        b = replace(b, arguments=args._replace(forcing_days=args.forcing_days.astype(np.float32)))
    elif change == "none":
        b = replace(b, arguments=args._replace(values_by_component=({"x": np.ones(1), "optional": np.zeros(1)},)))
    elif change == "days":
        b = replace(b, arguments=args._replace(science_day_numbers=args.science_day_numbers + 7))
    elif change == "shape":
        b = replace(b, arguments=args._replace(forcing_days=np.ones((7, 48, 2))))
    with pytest.raises(ValueError):
        batch.stack_lanes([a, b])


def test_vmap_is_applied_to_callable_and_each_lane_is_independent(monkeypatch):
    def scalar(*args):
        return args[0][0]["x"] + args[3]["x"] + args[2][0], args[1].sum()

    monkeypatch.setattr(batch, "make_scalar_transition", lambda lane, **kw: scalar)
    lanes = [lane("a", 1.0), lane("b", 4.0)]
    fn = jax.jit(batch.make_batch_transition(lanes))
    x = fn(*batch.stack_lanes(lanes))
    np.testing.assert_array_equal(x[0], [[4], [10]])
    reversed_x = fn(*batch.stack_lanes(lanes[::-1]))
    np.testing.assert_array_equal(x[0][::-1], reversed_x[0])
    changed_x = fn(*batch.stack_lanes([lanes[0], lane("c", 8.0)]))
    np.testing.assert_array_equal(x[0][0], changed_x[0][0])
    assert not np.array_equal(x[0][1], changed_x[0][1])


def test_factory_retains_capture_validation():
    common = dict(
        context=None, initial_state=None, prebound_hydrol_runtime_static_tables=None, daily_carbon_dispatch={}
    )
    with pytest.raises(ValueError, match="boundary capture"):
        driver._paper_later_day_block_transition_factory("dummy", training_output_projector=lambda x: x, **common)
    with pytest.raises(ValueError, match="stable cache key"):
        driver._paper_later_day_block_transition_factory(
            "dummy", training_output_projector=lambda x: x, capture_pre_daily_training_boundaries=True, **common
        )


@pytest.mark.parametrize("case", ["matched", "unexpected", "one_sided", "inf", "wrong_shape"])
def test_numerical_gate_preserves_only_declared_sentinel(case):
    from scripts.dev.benchmark_global_batch import compare

    expected = {"rough": np.asarray([[np.nan, 2.0]], dtype=np.float64)}
    actual = {"rough": expected["rough"].copy()}
    contract = {"['rough']": {"inner_shape": [1, 2], "bare_soil_index": 0}}
    if case == "unexpected":
        actual["rough"][0, 1] = np.nan
    elif case == "one_sided":
        actual["rough"][0, 0] = 0
    elif case == "inf":
        actual["rough"][0, 0] = np.inf
    elif case == "wrong_shape":
        expected["rough"] = np.asarray([[np.nan], [2.0]])
        actual["rough"] = expected["rough"].copy()
    result = compare(actual, expected, sentinel_paths=contract)
    assert result["passed"] is (case == "matched")
    if case == "matched":
        assert result["allowed_sentinels"][0]["count"] == 1
        assert result["max_bound_ratio"] == 0
