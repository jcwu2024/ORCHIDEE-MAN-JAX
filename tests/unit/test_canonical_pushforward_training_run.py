from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from research.daily_coarse_graining import canonical_pushforward_training_run as run
from research.daily_coarse_graining.canonical_pushforward_training_run import (
    _matched_training_sequence,
    _teacher_targets,
    parse_prefix_curriculum,
)


def test_prefix_curriculum_requires_unique_increasing_supported_days():
    stages = parse_prefix_curriculum("0:8,1:4,3:2,7:1,14:1")

    assert [stage.prefix_days for stage in stages] == [0, 1, 3, 7, 14]
    assert [stage.steps for stage in stages] == [8, 4, 2, 1, 1]
    with pytest.raises(ValueError, match="unique and increasing"):
        parse_prefix_curriculum("3:1,1:1")
    with pytest.raises(ValueError, match="0, 1, 3, 7, or 14"):
        parse_prefix_curriculum("2:1")


def test_matched_sequence_uses_terminal_state_and_same_day_inputs():
    batch = {
        "forcing_native": np.arange(2 * 4 * 5).reshape(2, 4, 5, 1),
        "parameters": np.ones((2, 4, 3)),
        "landpoint_static": np.ones((2, 4, 2)),
        "annual_conditions": np.ones((2, 4, 2)),
        "year": np.full((2, 4), 2004),
        "day_index": np.tile(np.arange(100, 104), (2, 1)),
    }
    compiled = {"forcing": jnp.arange(2 * 4 * 6).reshape(2, 4, 6)}
    terminal = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    teacher_fast = np.asarray([[5.0, 6.0], [7.0, 8.0]])
    teacher_next = terminal + 0.5

    sequence = _matched_training_sequence(
        batch,
        compiled,
        offset=3,
        terminal_states=terminal,
        teacher_fast_targets=teacher_fast,
        teacher_next_states=teacher_next,
    )

    np.testing.assert_array_equal(sequence.teacher_state[:, 0], terminal)
    np.testing.assert_array_equal(sequence.teacher_fast_day_target[:, 0], teacher_fast)
    np.testing.assert_array_equal(sequence.teacher_next_state[:, 0], teacher_next)
    np.testing.assert_array_equal(sequence.day_index[:, 0], np.asarray([103, 103]))
    np.testing.assert_array_equal(
        sequence.retained_tail_inputs["forcing"][:, 0],
        np.asarray(compiled["forcing"][:, 3]),
    )


def test_teacher_targets_query_each_model_visited_state(monkeypatch):
    calls = []

    def fake_query(**kwargs):
        calls.append(kwargs)
        state = np.asarray(kwargs["continuous"])
        discrete = kwargs["discrete"]
        return SimpleNamespace(
            fast_day_target=state + 10.0,
            next_state=state + 1.0,
            next_discrete_state={"flag": np.asarray(discrete["flag"])},
        )

    monkeypatch.setattr(run, "query_teacher_day", fake_query)
    batch = {
        "year": np.asarray([[2004, 2004], [2005, 2005]]),
        "day_index": np.asarray([[100, 101], [200, 201]]),
    }
    states = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    discrete = {"flag": np.asarray([[True], [False]])}

    fast, next_state, next_discrete = _teacher_targets(
        batch=batch,
        offset=1,
        terminal_states=states,
        terminal_discrete_states=discrete,
        config_path=SimpleNamespace(),
        runtime=SimpleNamespace(context=object()),
        templates=object(),
        contract=object(),
    )

    np.testing.assert_array_equal(fast, states + 10.0)
    np.testing.assert_array_equal(next_state, states + 1.0)
    np.testing.assert_array_equal(next_discrete["flag"], discrete["flag"])
    assert [call["day_index"] for call in calls] == [101, 201]
    assert [call["year"] for call in calls] == [2004, 2005]
