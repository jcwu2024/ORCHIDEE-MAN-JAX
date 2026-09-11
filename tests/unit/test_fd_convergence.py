from types import SimpleNamespace

import jax
import numpy as np

from research.daily_coarse_graining.daily_operator.fd_convergence import comparison, mask_changes, plateau
from research.daily_coarse_graining.daily_operator.types import StateDefinedMasks


def test_plateau_requires_three_adjacent_small_valid_regimes():
    rows = [{'epsilon': epsilon, 'passed': True, 'mask_stable': True, 'physical_pass': True}
            for epsilon in (1e-3, 3e-4, 1e-4, 3e-5, 1e-5)]
    assert not plateau(rows[:3])['passed']
    assert plateau(rows)['epsilons'] == [3e-4, 1e-4, 3e-5]
    rows[2]['mask_stable'] = False
    assert not plateau(rows)['passed']
    rows[2]['mask_stable'] = True
    rows[3]['physical_pass'] = False
    assert not plateau(rows)['passed']


def test_comparison_preserves_original_tolerance_and_rejects_nan():
    assert comparison(.635347742, .635379711, .635379709)['passed']
    assert not comparison(.466128604, .635381858, .635381852)['passed']
    assert not comparison(float('nan'), .6, .6)['passed']


def test_declared_mask_change_is_reported_even_without_a_hard_failure():
    masks = StateDefinedMasks(**{name: np.ones(2, dtype=bool) for name in StateDefinedMasks._fields})
    baseline = {'predicted_defined': masks, 'next_discrete': {'flag': np.array([True])}}
    current = {'predicted_defined': masks._replace(other=np.array([True, False])),
               'next_discrete': {'flag': np.array([True])}}
    contract = SimpleNamespace(state_leaves=[SimpleNamespace(start=0, stop=2, key='example.rveget')])
    result = mask_changes(jax, baseline, current, contract)
    assert not result['stable'] and result['discrete_changes'] == 0
    assert result['changed_fields'] == [{'field': 'example.rveget', 'global_indices': [1]}]
