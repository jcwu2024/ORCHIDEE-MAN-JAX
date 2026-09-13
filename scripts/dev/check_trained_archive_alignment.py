"""Inspect archived versus prepared arrays without loading or executing a model."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.hpc.trained_screen_common import WindowPool


def run(selection, attribution):
    pool = WindowPool(selection)
    report = json.loads(attribution.read_text())
    rows = []
    for record in report['windows']:
        selected = record['selection']
        batch = getattr(pool.load(selected['artifact_id']).batch, selected['component'])
        index, length = selected['window_index'], selected['length']
        path = attribution.parent / Path(record['raw_state_archive']).name
        with np.load(path) as archive:
            arrays = [('start', archive['teacher_start'], batch.inputs.continuous_state[index, :length]),
                      ('endpoint', archive['teacher_state'], batch.expected_next_state.other[index, :length])]
            comparison = {}
            for name, left, right in arrays:
                left, right = np.asarray(left), np.asarray(right)
                finite = np.isfinite(left) & np.isfinite(right)
                comparison[name] = {'equal': bool(np.array_equal(left, right)),
                    'equal_nan': bool(np.array_equal(left, right, equal_nan=True)),
                    'finite_max_difference': float(np.max(np.abs(left[finite]-right[finite]), initial=0.)),
                    'nan_counts': [int(np.isnan(left).sum()), int(np.isnan(right).sum())]}
        rows.append({'artifact': selected['artifact_id'], 'horizon': length, **comparison})
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--attribution', type=Path, required=True)
    args = parser.parse_args()
    run(args.selection, args.attribution)
