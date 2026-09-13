"""Host-only interpretation of saved loss-component gradients; no selection."""
from __future__ import annotations

import numpy as np

COMPONENTS = ('base', 'canopy_stock_change', 'litterpart_feedback')


def joint_gradient_geometry(batch, coefficients):
    """Describe an infinitesimal negative raw-gradient step, not an Adam step.

    The saved three-by-three Gram matrix suffices; no model, GPU or parameter
    checkpoint is needed. Positive directional derivatives mean an increase
    for that component along this particular direction. Finite-step training
    and Adam preconditioning still require actual optimizer experiments.
    """
    weights = np.asarray([coefficients[name] for name in COMPONENTS], dtype=np.float64)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.) or weights[0] != 1.:
        raise ValueError('supply finite nonnegative weights with base fixed to one')
    by_name = {group['name']: group for group in batch['groups']}
    gram = np.diag([by_name[name]['gradient']['l2_norm'] ** 2 for name in COMPONENTS])
    for i, left in enumerate(COMPONENTS):
        for j in range(i + 1, len(COMPONENTS)):
            right = COMPONENTS[j]
            gram[i, j] = gram[j, i] = batch['gradient_cross_geometry'][f'{left}__{right}']['dot']
    if not np.all(np.isfinite(gram)):
        raise ValueError('saved gradient geometry is nonfinite')
    squared_norm = float(weights @ gram @ weights)
    if squared_norm < -1e-10 * max(float(np.max(np.abs(gram))), 1.):
        raise ValueError('saved Gram matrix yields a negative squared norm')
    derivatives = -gram @ weights
    norm = float(np.sqrt(max(squared_norm, 0.)))
    base_norm = float(np.sqrt(gram[0, 0]))
    return {
        'coefficients': dict(zip(COMPONENTS, weights.tolist(), strict=True)),
        'raw_combined_gradient_norm': norm,
        'weighted_component_to_base_norm_ratio': {
            name: float(weights[i] * np.sqrt(gram[i, i]) / base_norm) if base_norm > 0 else None
            for i, name in enumerate(COMPONENTS)
        },
        'component_derivative_along_negative_combined_gradient': dict(
            zip(COMPONENTS, derivatives.tolist(), strict=True)),
        'interpretation': 'infinitesimal raw-gradient direction only; not an Adam update or learning prediction',
    }


def calibration_readout(report, *, coefficients=None):
    """Summarize available completed batches without promoting partial evidence."""
    batches = [batch for batch in report.get('batches', []) if batch.get('completed')]
    result = {
        'status': report['status'], 'stage': report.get('stage'),
        'elapsed_seconds': report.get('elapsed_seconds'),
        'completed_batches': len(batches),
        'ready_for_weight_review': (
            report['status'] == 'completed' and len(batches) == 4
            and report.get('all_reported_losses_finite') is True
            and report.get('all_gradient_values_finite') is True
            and report.get('all_hard_counts_zero') is True),
        'coefficient_selected': None, 'components': {},
    }
    for name in COMPONENTS:
        groups = [next(group for group in batch['groups'] if group['name'] == name) for batch in batches]
        if not groups:
            continue
        result['components'][name] = {
            'batch_losses': [group['loss'] for group in groups],
            'batch_gradient_norms': [group['gradient']['l2_norm'] for group in groups],
            'all_gradients_finite': all(group['gradient']['all_finite'] for group in groups),
            'all_hard_counts_zero': all(group['hard_counts']['all_zero'] for group in groups),
        }
    if coefficients is not None:
        result['joint_gradient_geometry'] = [joint_gradient_geometry(batch, coefficients) for batch in batches]
    result['interpretation'] = 'calibration evidence only; weights and trained science acceptance remain undecided'
    return result
