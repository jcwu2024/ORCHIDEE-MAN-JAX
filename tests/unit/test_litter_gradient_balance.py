"""Check vector interactions and the frozen coefficient selection rule."""
import numpy as np
import pytest

from scripts.hpc.litter_gradient_balance import select_litter_weight


def gram(vectors):
    array = np.asarray(vectors, dtype=float)
    return array @ array.T


def test_selection_matches_explicit_gradients_with_recurrent_cancellation():
    a = np.array([[4., 0.], [0., 20000.], [-4., 2.], [0., -2.]])
    b = np.array([[2., 0.], [0., 20000.], [0., 0.], [0., 0.]])
    result = select_litter_weight([gram(a), gram(b)])
    assert result['selected_weight'] == pytest.approx(5e-5)
    for vectors, panel in zip((a, b), result['panels'], strict=True):
        old = vectors[0] + .5*vectors[2] + .5*vectors[3]
        new = old + result['selected_weight'] * vectors[1]
        assert panel['old_mixed_norm'] == pytest.approx(np.linalg.norm(old))
        assert panel['candidate_mixed_norm'] == pytest.approx(np.linalg.norm(new))
        assert panel['weighted_litter_to_old_mixed_ratio'] == pytest.approx(.5)
        assert panel['candidate_old_mixed_cosine'] == pytest.approx(np.dot(new, old)/np.linalg.norm(new)/np.linalg.norm(old))


def test_outlier_cap_and_maximum_weight():
    small = gram([[1.], [100000.], [0.], [0.]])
    large = gram([[100.], [100000.], [0.], [0.]])
    result = select_litter_weight([small, large])
    assert result['selected_weight'] == pytest.approx(.5 / 50500)
    assert np.median([p['weighted_litter_to_old_mixed_ratio'] for p in result['panels']]) == pytest.approx(.5)
    assert max(p['weighted_litter_to_old_mixed_ratio'] for p in result['panels']) <= 1.0000001
    capped = select_litter_weight([small, large], target_median_ratio=.75)
    assert capped['selected_weight'] == pytest.approx(1e-5)
    huge_old = gram([[100.], [1.], [0.], [0.]])
    assert select_litter_weight([huge_old, huge_old])['selected_weight'] == 1e-3


def test_zero_original_gradient_cannot_calibrate_a_relative_weight():
    cancelled = gram([[1.], [1000.], [-2.], [0.]])
    with pytest.raises(ValueError, match='positive norms'):
        select_litter_weight([cancelled, cancelled])
