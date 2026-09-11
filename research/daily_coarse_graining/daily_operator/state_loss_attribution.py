"""CPU attribution of the declared grouped state loss, without changing its math."""
from __future__ import annotations

import numpy as np

from research.daily_coarse_graining.constrained_daily_replay import CARBON_INVENTORY_FIELDS, WATER_INVENTORY_FIELDS


def state_field_names(contract, assembler):
    leaves = {leaf.key: leaf for leaf in contract.state_leaves if leaf.start is not None}

    def names(key, *, full=False, suffix=''):
        leaf = leaves[key]
        count = int(np.prod(leaf.full_shape if full else leaf.shape))
        return [key + suffix] * count

    hydrol = 'hydrol_previous_step_state.'
    slow = 'slowproc_stomate_previous_step_state.'
    other = ['<unmapped>'] * contract.continuous_state_width
    for key, leaf in leaves.items():
        other[leaf.start:leaf.stop] = [key] * (leaf.stop - leaf.start)
    water = []
    for field in WATER_INVENTORY_FIELDS:
        if field == 'qsintveg':
            water += names(hydrol + field, suffix=':storage') + names(hydrol + field, suffix=':debt')
        else:
            water += names(hydrol + field)
    return {
        'canopy': names(hydrol + 'qsintveg'), 'soil': names(hydrol + 'mc'),
        'snow': sum((names(hydrol + field) for field in ('snow', 'snow_nobio', 'snowliq')), []),
        'water': water, 'carbon': sum((names(slow + field) for field in CARBON_INVENTORY_FIELDS), []),
        'thermal': sum((names(key, full=True) for key in assembler.thermal_keys), []), 'other': other,
    }


def attribute_state_loss(diagnostic, names):
    rows, groups = [], {}
    active_groups = sum(float(value) > 0 for value in diagnostic['group_count'].values())
    for group, expected_loss in diagnostic['group_loss'].items():
        def flat(key):
            values = diagnostic[key]
            return np.asarray(values[group] if isinstance(values, dict) else getattr(values, group)).reshape(-1)
        a, b, scale = flat('predicted'), flat('expected'), flat('scales')
        valid = flat('predicted_defined') & flat('expected_defined') & flat('eligible')
        valid &= np.isfinite(a) & np.isfinite(b) & np.isfinite(scale) & (scale > 0)
        labels = np.asarray(names[group])
        if labels.shape != a.shape:
            raise ValueError(f'{group} label layout {labels.shape} differs from actual state {a.shape}')
        residual = np.divide(a-b, scale, out=np.zeros_like(a, dtype=float), where=valid)
        count = int(valid.sum())
        denominator = max(count, 1)
        mean = float(residual.sum() / denominator)
        mse = float(np.square(residual).sum() / denominator)
        loss = .5 * (mse + mean*mean)
        if not np.isclose(loss, expected_loss, atol=1e-8, rtol=1e-6):
            raise ValueError(f'{group} exported-array loss {loss} differs from actual {expected_loss}')
        contributions = .5 * (residual**2 + residual*mean) / denominator / max(active_groups, 1)
        sensitivities = np.divide(residual + mean, denominator*max(active_groups, 1)*scale,
                                   out=np.zeros_like(a, dtype=float), where=valid)
        groups[group] = {'loss': loss, 'mse': mse, 'signed_bias': mean, 'count': count,
                         'contribution_to_state_total': loss/max(active_groups, 1)}
        for field in np.unique(labels[valid]):
            selected = valid & (labels == field)
            indices = np.flatnonzero(selected)
            top = indices[np.argsort(abs(contributions[indices]))[-3:][::-1]]
            rows.append({'group': group, 'field': str(field), 'count': len(indices),
                'state_loss_contribution': float(contributions[selected].sum()),
                'squared_error_contribution': float(.5*np.square(residual[selected]).sum()/denominator/max(active_groups, 1)),
                'state_output_gradient_norm': float(np.linalg.norm(sensitivities[selected])),
                'top_coordinates': [{'group_index': int(i), 'predicted': float(a[i]), 'teacher': float(b[i]),
                    'scale': float(scale[i]), 'normalized_error': float(residual[i]),
                    'contribution': float(contributions[i])} for i in top]})
    rows.sort(key=lambda row: abs(row['state_loss_contribution']), reverse=True)
    return {'groups': groups, 'fields': rows, 'reconstructed_state_total': sum(row['state_loss_contribution'] for row in rows)}
