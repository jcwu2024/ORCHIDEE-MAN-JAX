from __future__ import annotations

from pathlib import Path

import numpy as np

from research.daily_coarse_graining.markov_dataset import MarkovShardRef
from research.daily_coarse_graining.pushforward_outlier_diagnostic import (
    _leaf_attribution,
    replay_selection,
)


class _Leaf:
    def __init__(self, key, start, stop, family):
        self.key = key
        self.start = start
        self.stop = stop
        self.family = family


def _reference(landpoint, year):
    return MarkovShardRef(
        landpoint_id=landpoint,
        year=year,
        spatial_split="train",
        temporal_split="train",
        path=Path(f"{landpoint}-{year}.npz"),
        sha256="hash",
        contract_sha256="contract",
    )


def test_replay_selection_is_deterministic_and_counts_every_step():
    references = tuple(
        _reference(point, year)
        for point in ("a", "b")
        for year in (1961, 1962, 1963)
    )
    first = replay_selection(
        references,
        curriculum="0:3,1:3,7:3",
        batch_size=2,
        seed=7,
        target_prefix=7,
        target_step=2,
    )
    second = replay_selection(
        references,
        curriculum="0:3,1:3,7:3",
        batch_size=2,
        seed=7,
        target_prefix=7,
        target_step=2,
    )
    assert first[0] == second[0]
    np.testing.assert_array_equal(first[1], second[1])
    assert all(sum(counts.values()) == 3 for counts in first[2].values())


def test_leaf_attribution_identifies_the_dominant_output():
    actual = np.asarray([[0.0, 0.0, 0.0]])
    expected = np.asarray([[20.0, 1.0, 1.0]])
    leaves = (_Leaf("large", 0, 1, "a"), _Leaf("small", 1, 3, "b"))
    report = _leaf_attribution(
        actual,
        expected,
        np.ones(3),
        np.full(3, 1.0 / 3.0),
        leaves,
        owner_name="family",
    )
    assert report["top_leaves"][0]["key"] == "large"
    assert report["top_leaves"][0]["samples"][0]["normalized_rmse"] == 20.0
    assert report["defined_status_mismatches"] == 0
