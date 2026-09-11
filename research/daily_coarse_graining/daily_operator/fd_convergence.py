"""Small host-side rules for finite-difference convergence diagnostics."""
from __future__ import annotations

import numpy as np

EPSILONS = (1e-3, 7e-4, 5e-4, 3e-4, 1e-4, 3e-5, 1e-5)


def comparison(finite_difference, reverse, jvp, *, atol=1e-6, rtol=.01):
    finite = bool(np.all(np.isfinite([finite_difference, reverse, jvp])))
    return {'passed': finite and bool(np.isclose(finite_difference, reverse, atol=atol, rtol=rtol)
                                     and np.isclose(jvp, reverse, atol=atol, rtol=rtol)),
            'relative_fd_error': float(abs(finite_difference-reverse)/max(abs(reverse), 1e-30))}


def plateau(rows):
    """Require three adjacent small, physically valid, mask-stable scales."""
    consecutive = []
    for row in rows:
        if 0 < row['epsilon'] <= 3e-4 and row['passed'] and row['mask_stable'] and row['physical_pass']:
            if consecutive and row['epsilon'] >= consecutive[-1]:
                consecutive = []
            consecutive.append(row['epsilon'])
            if len(consecutive) == 3:
                return {'passed': True, 'epsilons': consecutive}
        else:
            consecutive = []
    return {'passed': False, 'epsilons': []}


def admission(rows, policy='all_declared_epsilons'):
    if policy == 'all_declared_epsilons':
        return {'passed': bool(rows) and all(row['passed'] for row in rows),
                'policy': policy}
    if policy == 'mask_stable_convergence_plateau_v1':
        return {**plateau(rows), 'policy': policy}
    raise ValueError(f'unsupported finite-difference policy: {policy}')


def mask_changes(jax, baseline, current, contract):
    changes = {}
    for group in baseline['predicted_defined']._fields:
        a, b = np.asarray(getattr(baseline['predicted_defined'], group)), np.asarray(getattr(current['predicted_defined'], group))
        changes[group] = int(np.sum(a != b))
    a, b = np.asarray(baseline['predicted_defined'].other), np.asarray(current['predicted_defined'].other)
    fields = []
    for leaf in contract.state_leaves:
        if leaf.start is None:
            continue
        indices = np.flatnonzero(a[leaf.start:leaf.stop] != b[leaf.start:leaf.stop])
        if indices.size:
            fields.append({'field': leaf.key, 'global_indices': (indices+leaf.start).tolist()})
    left, structure = jax.tree_util.tree_flatten(baseline['next_discrete'])
    right, other = jax.tree_util.tree_flatten(current['next_discrete'])
    if structure != other:
        raise ValueError('diagnostic discrete-state trees differ')
    discrete = sum(int(np.sum(np.asarray(x) != np.asarray(y))) for x, y in zip(left, right, strict=True))
    return {'stable': not any(changes.values()) and discrete == 0,
            'state_mask_changes': changes, 'changed_fields': fields, 'discrete_changes': discrete}
