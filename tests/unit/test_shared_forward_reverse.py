"""Arithmetic contract for the forward-versus-combined timing summary."""
import pytest

from scripts.hpc.probe_shared_forward_reverse import timing_summary


def test_marginal_ad_is_not_labeled_pure_reverse():
    result = timing_summary([2.0, 2.1, 1.9], [5.0, 5.1, 4.9])
    assert result["forward_fraction_of_combined"] == pytest.approx(0.4)
    assert result["marginal_ad_fraction_of_combined"] == pytest.approx(0.6)
    assert "rematerialized" in result["interpretation"]


def test_requires_matched_repeats():
    with pytest.raises(ValueError):
        timing_summary([1.0, 1.1], [2.0, 2.1])
    with pytest.raises(ValueError):
        timing_summary([1.0, 1.1, 1.2], [2.0, 2.1])
