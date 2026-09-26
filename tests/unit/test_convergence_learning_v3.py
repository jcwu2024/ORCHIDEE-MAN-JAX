import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from research.daily_coarse_graining.daily_operator.fd_convergence import admission
from research.daily_coarse_graining.daily_operator.fraction_monitoring import fraction_report




def test_legacy_rejects_coarse_failure_but_v3_requires_a_real_stable_plateau():
    rows = [{'epsilon': e, 'passed': e <= 3e-4, 'mask_stable': True, 'physical_pass': True}
            for e in (1e-3, 3e-4, 1e-4, 3e-5)]
    assert not admission(rows)['passed']
    assert admission(rows, 'mask_stable_convergence_plateau_v1')['passed']
    rows[2]['mask_stable'] = False
    assert not admission(rows, 'mask_stable_convergence_plateau_v1')['passed']
    repeated = [dict(rows[-1]) for _ in range(3)]
    assert not admission(repeated, 'mask_stable_convergence_plateau_v1')['passed']


def test_fraction_monitor_distinguishes_supported_endpoints_and_wrong_saturation():
    leaf = SimpleNamespace(start=0, stop=5, shape=(1,5), key='example.fraction')
    diagnostic = {key: SimpleNamespace(other=np.array(values)) for key, values in {
        'predicted': [0.,1.,1.,0.,np.nan], 'expected': [0.,1.,0.,1.,0.],
        'predicted_defined': [True,True,True,True,False],
        'expected_defined': [True,True,True,True,False],
    }.items()}
    report = fraction_report(diagnostic, leaf)
    assert report['count'] == 4
    assert report['near_zero_count'] == 2 and report['near_one_count'] == 2
    assert report['wrong_upper_saturation_count'] == 1
    assert report['wrong_lower_saturation_count'] == 1
    np.testing.assert_allclose(report['rmse'], np.sqrt(.5))
