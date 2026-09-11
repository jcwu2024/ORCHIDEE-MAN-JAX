"""Offline loss accounting for exact-zero bare-soil gsmean; no model calls."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from research.daily_coarse_graining.daily_operator.state_loss_attribution import attribute_state_loss


def analyze(states_path, output):
    with states_path.open('rb') as handle:
        saved = pickle.load(handle)
    names = saved['field_names']
    indices = np.flatnonzero(np.asarray(names['other']) == 'diffuco_previous_step_state.gsmean')
    # This export was made under the accepted bare-soil/PFT14 layout. This is
    # an accounting probe for its identified index, not the model's masking rule.
    if indices.tolist() != [10, 11]:
        raise ValueError('this diagnostic requires the recorded two-slot layout')
    rows = []
    for record in saved['states']:
        diagnostic = dict(record['diagnostic'])
        original = attribute_state_loss(diagnostic, names)
        changed = np.asarray(diagnostic['predicted'].other).copy()
        changed[10] = 0.
        diagnostic['predicted'] = diagnostic['predicted']._replace(other=changed)
        losses = dict(diagnostic['group_loss'])
        valid = (np.asarray(diagnostic['predicted_defined'].other)
                 & np.asarray(diagnostic['expected_defined'].other)
                 & np.asarray(diagnostic['eligible'].other))
        scale = np.asarray(diagnostic['scales'].other)
        expected = np.asarray(diagnostic['expected'].other)
        valid &= np.isfinite(changed) & np.isfinite(expected) & np.isfinite(scale) & (scale > 0)
        residual = (changed[valid] - expected[valid]) / scale[valid]
        losses['other'] = float(.5*(np.mean(residual**2) + np.mean(residual)**2))
        diagnostic['group_loss'] = losses
        corrected = attribute_state_loss(diagnostic, names)
        rows.append({'arm': record['arm'], 'sample': record['sample'],
                     'original_state_loss': original['reconstructed_state_total'],
                     'counterfactual_state_loss': corrected['reconstructed_state_total'],
                     'remaining_top_fields': corrected['fields'][:6]})
    result = {'model_calls': 0, 'training_updates': 0,
              'scope': 'Only modify exported other-state index10 for loss accounting; not a physical rollout or forward acceptance',
              'cases': rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    for row in rows:
        print(row['arm'], row['sample'], row['original_state_loss'], '->', row['counterfactual_state_loss'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('states', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    analyze(args.states, args.output)
