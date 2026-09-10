"""Read saved day-nine states and frozen input scales; CPU-only, no model calls."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np


def run(root):
    diagnostic = root / 'runtime/outputs/diagnostics'
    report = json.loads((diagnostic / 'conditioned-score-day10-14991510.json').read_text())
    field = next(r for r in report['daily_state_differences'][8]['fields']
                 if r['field'] == 'slowproc_stomate_previous_step_state.DOC')
    start, stop = field['start'], field['stop']
    with (diagnostic / 'conditioned-score-discrepancy-14991510.trace.pkl').open('rb') as handle:
        compact = pickle.load(handle)['compact']
    with (diagnostic / 'conditioned-score-day10-14991510.trace.pkl').open('rb') as handle:
        full = pickle.load(handle)['full']
    statistics = root / 'runtime/outputs/training/first-neural-process-head-v1-statistics/process_training_statistics.npz'
    with np.load(statistics) as arrays:
        mean = arrays['input__continuous__mean'][start:stop]
        scale = arrays['input__continuous__scale'][start:stop]
    a = np.asarray(compact.result.next_state.other[8, start:stop])
    b = np.asarray(full.result.next_state.other[8, start:stop])
    valid = np.asarray(compact.result.next_state_defined.other[8, start:stop]) & np.asarray(full.result.next_state_defined.other[8, start:stop])
    safe = np.where(np.isfinite(scale.astype('float32')) & (scale.astype('float32') > 0), scale.astype('float32'), 1.)
    za, zb = [(x.astype('float32') - mean.astype('float32')) / safe for x in (a, b)]
    delta = np.where(valid, abs(za - zb), 0)
    selected = np.argsort(delta)[-8:][::-1]
    output = {'status': 'completed', 'model_calls': 0, 'training_updates': 0,
              'field': field['field'], 'day_start_for': 10,
              'defined_coordinates': int(valid.sum()),
              'input_scale_min': float(scale[valid].min()), 'input_scale_max': float(scale[valid].max()),
              'max_raw_difference': float(np.max(abs(a[valid] - b[valid]))),
              'max_float32_normalized_difference': float(delta.max()),
              'max_abs_normalized_compact': float(abs(za[valid]).max()),
              'top_coordinates': [{'global_index': int(start + i), 'local_index': int(i),
                  'compact': float(a[i]), 'full': float(b[i]), 'mean': float(mean[i]), 'scale': float(scale[i]),
                  'normalized_compact': float(za[i]), 'normalized_full': float(zb[i]),
                  'normalized_difference': float(delta[i])} for i in selected]}
    path = diagnostic / 'conditioned-doc-input-scale-14991510.json'
    path.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
    print(json.dumps(output, indent=2, allow_nan=False))


if __name__ == '__main__':
    run(Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX'))
