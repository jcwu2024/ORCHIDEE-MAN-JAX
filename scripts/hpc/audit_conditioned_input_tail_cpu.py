"""Audit saved Teacher/predicted continuous features without a model execution."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np


def feature_summary(values, valid, mean, scale):
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.)
    z = (np.where(valid, values, mean) - mean) / scale
    magnitude = np.abs(z)
    return {"defined_count": int(valid.sum()), "max_abs_z": float(magnitude[valid].max()),
            "above_threshold": {str(t): int(np.sum(valid & (magnitude > t))) for t in (16., 1e3, 1e6)},
            "distinct_coordinates_above_16": int(np.any(valid & (magnitude > 16), axis=0).sum())}


def run(root):
    diagnostics = root / 'runtime/outputs/diagnostics'
    with (diagnostics / 'conditioned-score-discrepancy-14991510.trace.pkl').open('rb') as handle:
        saved = pickle.load(handle)
    with (diagnostics / 'conditioned-score-day10-14991510.trace.pkl').open('rb') as handle:
        full = pickle.load(handle)['full']
    with np.load(root / 'runtime/outputs/training/first-neural-process-head-v1-statistics/process_training_statistics.npz') as data:
        mean, scale = data['input__continuous__mean'], data['input__continuous__scale']
    compact = saved['compact']
    a, b = np.asarray(compact.result.next_state.other), np.asarray(full.result.next_state.other)
    valid_a, valid_b = [np.asarray(trace.result.next_state_defined.other) for trace in (compact, full)]
    teacher = np.asarray(saved['micro'].inputs.continuous_state)[0]
    # Teacher starts use the input adapter's sentinel convention; no guessed
    # next-day mask is applied to the previous day's input.
    teacher_valid = np.isfinite(teacher) & (np.abs(teacher) < .5e20)
    result = {"status": "completed", "model_calls": 0, "training_updates": 0,
              "scope": "artifact 883/window 3 only; 30 Teacher starts and predicted endpoints",
              "teacher": feature_summary(teacher, teacher_valid, mean, scale),
              "compact": feature_summary(a, valid_a, mean, scale),
              "full": feature_summary(b, valid_b, mean, scale)}
    common = valid_a & valid_b
    safe = np.where(np.isfinite(scale) & (scale > 0), scale, 1.)
    za, zb = [(np.where(common, v, mean) - mean) / safe for v in (a, b)]
    def tail(z):
        return np.where(abs(z) <= 16., z, np.sign(z) * (16. + np.log1p(np.maximum(abs(z) - 16., 0.))))
    result['max_float32_feature_difference'] = {
        'linear': float(np.max(abs(za.astype('float32') - zb.astype('float32')))),
        'tail': float(np.max(abs(tail(za).astype('float32') - tail(zb).astype('float32'))))}
    ranked = np.argsort(np.max(np.where(valid_a, abs(za), 0.), axis=0))[-20:][::-1]
    leaf_rows = json.loads((diagnostics / 'conditioned-score-day10-14991510.json').read_text())['daily_state_differences']
    leaves = {r['field']: r for day in leaf_rows for r in day['fields']}
    result['top_coordinates'] = [{'index': int(i), 'scale': float(scale[i]),
        'max_abs_z': float(np.max(abs(za[:, i]))),
        'leaf': next((name for name, r in leaves.items() if r['start'] <= i < r['stop']), 'not_in_differing_leaf_report')}
        for i in ranked]
    output = diagnostics / 'conditioned-input-tail-cpu-14991510.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    run(Path('/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX'))
