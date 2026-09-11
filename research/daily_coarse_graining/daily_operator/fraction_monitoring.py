"""Physical fraction diagnostics for short training, without modifying its loss."""
from __future__ import annotations

import numpy as np


def fraction_report(diagnostic, leaf, tolerance=1e-6):
    selection = slice(leaf.start, leaf.stop)
    predicted = np.asarray(diagnostic['predicted'].other)[selection].reshape(leaf.shape)
    expected = np.asarray(diagnostic['expected'].other)[selection].reshape(leaf.shape)
    valid = (np.asarray(diagnostic['predicted_defined'].other)[selection]
             & np.asarray(diagnostic['expected_defined'].other)[selection]).reshape(leaf.shape)
    valid &= np.isfinite(predicted) & np.isfinite(expected)
    near_upper, near_lower = predicted >= 1-tolerance, predicted <= tolerance
    target_zero = expected <= tolerance
    target_one = expected >= 1-tolerance
    delta = np.subtract(predicted, expected, out=np.zeros_like(predicted, dtype=float), where=valid)
    return {'field': leaf.key, 'predicted': predicted.tolist(), 'teacher': expected.tolist(),
            'defined': valid.tolist(), 'count': int(valid.sum()),
            'near_zero_count': int(np.sum(valid & near_lower)),
            'near_one_count': int(np.sum(valid & near_upper)),
            'teacher_near_zero_count': int(np.sum(valid & target_zero)),
            'teacher_near_one_count': int(np.sum(valid & target_one)),
            'wrong_upper_saturation_count': int(np.sum(valid & target_zero & near_upper)),
            'wrong_lower_saturation_count': int(np.sum(valid & target_one & near_lower)),
            'rmse': float(np.sqrt(np.sum(delta*delta)/max(int(valid.sum()), 1)))}
