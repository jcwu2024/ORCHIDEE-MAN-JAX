"""Adapt prepared held-out validation windows to the shared physical monitor.

The prepared validation format differs from the TRAIN mixed-batch format.
This adapter only maps existing Teacher arrays; it neither generates targets
nor changes the canonical forward model or any training gradient.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.hpc.successor_rollout_screen_prepared import load_prepared_rollout_artifact

SPLITS = {
    'temporal': ('train', 'validation'),
    'spatial': ('validation', 'train'),
    'joint': ('validation', 'validation'),
}


def artifact_case(jax, artifact, row):
    """Map a single-window validation artifact without changing its day order."""
    horizon = int(row['length'])
    entry = artifact.window_entry
    if (artifact.slice_id != row['slice_id']
            or (entry['landpoint_id'], entry['year'], entry['start_rows'])
            != (row['landpoint_id'], row['year'], [row['start_row']])
            or int(entry['horizon_days']) != horizon):
        raise ValueError('validation artifact and selected window differ')
    batch = artifact.batch
    inputs = batch.rollout_inputs
    if np.shape(inputs.continuous_state)[:2] != (1, horizon):
        raise ValueError('validation artifact must contain one complete selected window')
    templates = jax.tree_util.tree_map(lambda value: value[0], inputs)
    if not np.all(np.diff(np.asarray(templates.day_index).reshape(-1)) == 1):
        raise ValueError('validation days are not consecutive')
    # Validation artifacts store one retained bundle, without a window axis.
    retained = jax.tree_util.tree_map(
        lambda value: np.broadcast_to(np.asarray(value), (horizon,) + np.shape(value)),
        artifact.day_inputs)
    initial = templates.continuous_state[0]
    discrete = jax.tree_util.tree_map(lambda value: value[0], templates.discrete_state)
    targets = {
        'state': batch.expected_next_state[0],
        'defined': batch.expected_next_state_defined[0],
        'gpp': batch.expected_fast_gpp[0],
        'gpp_defined': batch.expected_fast_gpp_defined[0],
        'fraction': templates.pft.pft_fraction,
        'active': templates.pft.active_pft_mask,
    }
    expected_discrete = jax.tree_util.tree_map(
        lambda value: value[0], batch.expected_next_discrete_state)
    return initial, discrete, (templates, retained), targets, expected_discrete


class FreshMethodValidationPool:
    """Read one validation artifact at a time; preserve actual split labels."""

    def __init__(self, manifest_path):
        self.path = Path(manifest_path).resolve()
        manifest = json.loads(self.path.read_text())
        panel = json.loads((self.path.parent / manifest['validation_panel']).read_text())
        if (manifest['status'] != 'passed' or manifest.get('sealed_test_used') is not False
                or manifest.get('teacher_data_regenerated') is not False):
            raise ValueError('validation preparation is not complete or crossed its data boundary')
        if manifest['validation_panel_sha256'] != panel['panel_sha256']:
            raise ValueError('validation panel differs from the prepared manifest')
        for name, splits in SPLITS.items():
            cell = panel['slices'][name]
            if (cell['spatial_split'], cell['temporal_split']) != splits:
                raise ValueError('validation split labels differ from the release policy')
        self.panel = panel
        self.records = {int(record['ordinal']): record for record in manifest['entries']}
        self.rows = []
        for identifier, record in self.records.items():
            name, horizon = record['slice_id'], int(record['horizon_days'])
            entry = record['window_entry']
            if name not in SPLITS or horizon not in (7, 30) or len(entry['start_rows']) != 1:
                raise ValueError('unsupported validation cell or window multiplicity')
            if entry not in panel['slices'][name]['horizons'][str(horizon)]:
                raise ValueError('prepared window is absent from the frozen panel')
            start = int(entry['start_rows'][0])
            self.rows.append(dict(
                artifact_id=identifier, slice_id=name, length=horizon,
                component={7: 'seven_day', 30: 'thirty_day'}[horizon],
                landpoint_id=entry['landpoint_id'], year=int(entry['year']), start_row=start,
                teacher_day_keys=[[entry['landpoint_id'], int(entry['year']), day]
                                  for day in range(start, start + horizon)],
            ))
        per_cell = int(panel['windows_per_slice_per_horizon'])
        counts = {(name, horizon): sum(row['slice_id'] == name and row['length'] == horizon
                                      for row in self.rows)
                  for name in SPLITS for horizon in (7, 30)}
        if len(self.records) != len(manifest['entries']) or any(n != per_cell for n in counts.values()):
            raise ValueError('validation cell population is incomplete')

    def case(self, jax, row):
        record = self.records[row['artifact_id']]
        artifact = load_prepared_rollout_artifact(
            self.path, record, slice_id=record['slice_id'], window_entry=record['window_entry'])
        return artifact_case(jax, artifact, row)


def evaluate_validation_panel(monitor, parameters, pool):
    """Report each held-out split separately, without borrowed science gates."""
    results = {}
    for name, (spatial, temporal) in SPLITS.items():
        result = monitor(
            parameters, pool, [row for row in pool.rows if row['slice_id'] == name],
            panel_name='fresh_method_' + name, development=False, apply_science_gates=False)
        result.update(role='held_out_validation', spatial_split=spatial, temporal_split=temporal)
        results[name] = result
    return dict(
        slices=results, parameter_updates=0, scientific_acceptance=False,
        sealed_test_used=False, panel_sha256=pool.panel['panel_sha256'],
        physical_gate_passed=all(value['physical_gate_passed'] for value in results.values()),
        metric_population_complete=all(value['metric_population_complete'] for value in results.values()),
    )
