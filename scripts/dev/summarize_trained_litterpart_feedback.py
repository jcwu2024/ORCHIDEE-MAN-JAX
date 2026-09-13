"""Read saved forward/intervention evidence; perform no neural or Teacher run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(directory: Path) -> dict:
    attribution = json.loads((directory / 'trained-forward-attribution-v1.json').read_text())
    detail = json.loads((directory / 'trained-gpp-litter-detail-v1.json').read_text())
    group = json.loads((directory / 'trained-gpp-state-interventions-v1.json').read_text())
    leaf = next(row for row in attribution['state_layout'] if row['field'] == 'litterpart')
    selection = slice(leaf['start'], leaf['stop'])
    with np.load(directory / 'v6-process-training-statistics.npz') as statistics:
        mean = statistics['input__continuous__mean'][selection]
        scale = statistics['input__continuous__scale'][selection]
    with np.load(directory / 'v6-training-statistics.npz') as statistics:
        next_scale = statistics['next_state__scale'][selection]
    with np.load(directory / 'successor_fast_owner_statistics.npz') as statistics:
        output_mean = statistics['output__endpoint.ok_leak_aux.litterpart__mean']
        output_scale = statistics['output__endpoint.ok_leak_aux.litterpart__scale']
        loss_scale = statistics['loss_scale__endpoint.ok_leak_aux.litterpart']
    rows = {}
    for horizon in (7, 30):
        values = {name: [] for name in ('teacher_start', 'teacher_state', 'reset_state', 'free_state', 'free_start')}
        for window in attribution['windows']:
            if window['selection']['length'] != horizon:
                continue
            with np.load(directory / Path(window['raw_state_archive']).name) as archive:
                for name in values:
                    if name == 'free_start':
                        array = np.concatenate((archive['teacher_start'][:1], archive['free_state'][:-1]))
                    else:
                        array = archive[name]
                    values[name].append(array[:, selection])
        arrays = {name: np.concatenate(parts) for name, parts in values.items()}
        if not all(np.all(np.isfinite(array)) for array in arrays.values()):
            raise ValueError('archived litterpart contains nonfinite values')
        statistics = {}
        for name, array in arrays.items():
            z = (array - mean) / scale
            target = arrays['teacher_start' if name.endswith('start') else 'teacher_state']
            statistics[name] = {
                'count_per_coordinate': len(array),
                'mean': array.mean(axis=0).tolist(),
                'min': array.min(axis=0).tolist(),
                'max': array.max(axis=0).tolist(),
                'rmse_against_same_phase_teacher': np.sqrt(np.mean((array - target) ** 2, axis=0)).tolist(),
                'absolute_input_z_max': np.max(np.abs(z), axis=0).tolist(),
            }
        contrasts = {}
        for name, metric in detail['summary'][str(horizon)]['gpp_bidirectional_attribution'].items():
            contrasts[name] = dict(metric, **detail['causal_classification']['window_sign_consistency'][str(horizon)][name])
        reproduced = {}
        for name in ('litter_metadata', 'doc_inventory'):
            a = detail['summary'][str(horizon)]['gpp_bidirectional_attribution'][name]
            b = group['summary'][str(horizon)]['gpp_bidirectional_attribution'][name]
            reproduced[name] = {key: float(a[key] - b[key]) for key in ('rescue_fraction', 'damage_fraction')}
        rows[str(horizon)] = {'archive_states': statistics, 'gpp_contrasts': contrasts,
                              'previous_group_contrast_differences': reproduced}
    return {
        'scope': 'saved_diagnostic_panel_only_not_training_or_model_acceptance',
        'leaf': leaf, 'input_mean': mean.tolist(), 'input_scale': scale.tolist(),
        'next_state_scale': next_scale.tolist(), 'output_mean': output_mean.tolist(),
        'output_scale': output_scale.tolist(), 'process_loss_scale': loss_scale.tolist(),
        'normalization_authority': 'process statistics input__continuous; fresh DOC-only tail leaves litterpart unchanged',
        'terminal_vs_fast_limit': 'Archive statistics use terminal states, not a new typed fast-label readback.',
        'run_completed': detail['status'] == 'completed' and len(detail['windows']) == 16,
        'controls_valid': detail['causal_classification']['control_reproduction_valid'],
        'hard_count_sum': sum(sum(v.values()) for v in detail['hard_count_totals'].values()),
        'elapsed_seconds': detail['elapsed_seconds'], 'horizons': rows,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path('outputs'))
    parser.add_argument('--output', type=Path, default=Path('outputs/trained-litterpart-feedback-summary-v1.json'))
    args = parser.parse_args()
    report = summarize(args.directory)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'horizons'}, indent=2))
    print(json.dumps(report['horizons']['30']['archive_states'], indent=2))
