from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from research.daily_coarse_graining.daily_operator.neural_architecture import NeuralInputNormalization
from scripts.hpc.audit_conditioned_input_tail import within_regression_cap
from scripts.hpc.audit_conditioned_input_tail_cpu import feature_summary


def test_admission_cannot_hide_a_component_regression_in_total_loss():
    reference = {'total': 1., 'fluxes': {'gpp': {'mse': .5}, 'npp': {'mse': 0.}},
                 'stocks': {'agb': {'mse': .2}}}
    candidate = deepcopy(reference)
    candidate['total'] = .8
    assert within_regression_cap(candidate, reference)
    candidate['stocks']['agb']['mse'] = .22
    assert not within_regression_cap(candidate, reference)
    candidate['stocks']['agb']['mse'] = float('nan')
    assert not within_regression_cap(candidate, reference)


def test_cpu_counts_exclude_undefined_coordinates_and_separate_exposure():
    values = np.array([[2., 1e12], [20., np.nan]])
    valid = np.array([[True, False], [True, False]])
    report = feature_summary(values, valid, np.zeros(2), np.ones(2))
    assert report['defined_count'] == 2
    assert report['distinct_coordinates_above_16'] == 1
    assert report['above_threshold'] == {'16.0': 1, '1000.0': 0, '1000000.0': 0}
    assert report['max_abs_z'] == 20.


def test_doc_scope_resolves_named_contract_field_and_rebuilds_binding(monkeypatch):
    from scripts.hpc import audit_conditioned_input_tail as runner

    @dataclass
    class Head:
        input_normalization: object

    normalization = NeuralInputNormalization(*([np.zeros(8)] * 11))
    leaf = SimpleNamespace(key='slowproc_stomate_previous_step_state.DOC', start=2, stop=5)
    context = {'runtime': {'contract': SimpleNamespace(state_leaves=[leaf])},
               'template': None, 'scales': None, 'target_contract': None}
    bootstrap = {'head': Head(normalization), 'context': context, 'flux_basis': None}
    monkeypatch.setattr(runner, '_make_steps', lambda runtime, head, *a, **kw: {'bound_head': head})
    candidate = runner.with_tail(bootstrap, 16., scope='doc')
    assert candidate['steps']['bound_head'] is candidate['head']
    np.testing.assert_array_equal(candidate['head'].input_normalization.continuous_tail_mask,
                                  [False, False, True, True, True, False, False, False])
    assert bootstrap['head'].input_normalization.continuous_tail_mask is None
