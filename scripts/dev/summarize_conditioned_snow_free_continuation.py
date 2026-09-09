"""Summarize a completed corrected-continuation report without loading model data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median

from research.daily_coarse_graining.daily_operator.corrected_continuation import load_contract
from scripts.hpc.run_conditioned_snow_free_continuation import persistent


def summarize(report):
    contract = load_contract()
    baseline, monitors = report['initial_monitor'], report['monitors']
    rows, previous, artifacts = [], baseline, []
    for record, monitor in zip(report['records'], monitors, strict=True):
        panel = monitor['panel']
        local = persistent.monitor_transfer_gate(contract, previous, panel)
        cumulative = persistent.monitor_transfer_gate(contract, baseline, panel)
        assert local == monitor['local_gate'] and cumulative == monitor['cumulative_gate']
        assert local['passed'] and cumulative['passed']
        evidence = monitor['gradient_evidence']
        assert evidence['calls'] == {'7': 64, '30': 32}
        assert len(evidence['records']) == 96 and all(r['linearized_score_agrees'] for r in evidence['records'])
        artifacts.extend(u for g in record['train_only_panel']['artifact_groups'] for u in g)
        slopes = record['direction']['calibration']['constraint_relative_slopes']
        rows.append({'new_update': record['logical_update'] - 4,
                     'direct_reduction_percent': -100 * cumulative['direct_gpp_mean_relative_change'],
                     'science_reduction_percent': -100 * cumulative['science_progress']['macro_relative_change'],
                     'direct_local_reduction_percent': -100 * local['direct_gpp_mean_relative_change'],
                     'science_local_reduction_percent': -100 * local['science_progress']['macro_relative_change'],
                     'train_seconds': record['timing_seconds']['total'],
                     'gradient_window_seconds': sum(r['seconds'] for r in evidence['records']),
                     'scale': record['selection']['selected_scale'],
                     'parameter_delta_l2': record['selection']['selected_parameter_delta_l2'],
                     'limiting_source': max(slopes, key=slopes.get)})
        previous = panel
    metrics = []
    final = monitors[-1]['panel']
    for horizon in ('7', '30'):
        for key in contract.required_science_keys:
            before, after = (p['acceptance'][horizon]['keys'][key] for p in (baseline, final))
            metrics.append({'horizon': int(horizon), 'key': key,
                'initial_rmse': before['normalized_rmse'], 'final_rmse': after['normalized_rmse'],
                'initial_bias': before['normalized_signed_bias'], 'final_bias': after['normalized_signed_bias'],
                'absolute_pass': report['absolute_science']['checks'][f'{horizon}.{key}']})
    middle = monitors[4]['panel']
    second_half = persistent.monitor_transfer_gate(contract, middle, final)
    return {'status': report['status'], 'completed_updates': len(rows), 'updates': rows, 'metrics': metrics,
            'all_local_cumulative_gates_recomputed': True, 'gradient_score_pairs': 96 * len(rows),
            'train_artifact_exposures': len(artifacts), 'distinct_train_artifacts': len(set(artifacts)),
            'hot_train_median_seconds': median(r['train_seconds'] for r in rows[1:]),
            'last_five_direct_reduction_percent': -100 * second_half['direct_gpp_mean_relative_change'],
            'last_five_science_reduction_percent': -100 * second_half['science_progress']['macro_relative_change'],
            'absolute_science_passed': report['absolute_science']['passed'],
            'reverse_compile_count': report['reverse_compile_count']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    summary = summarize(json.loads(args.report.read_text()))
    destination = args.report.with_suffix('.summary.json')
    destination.write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = summary['updates']
    fig, ax = plt.subplots(figsize=(8, 4.5), layout='constrained')
    for field, label in [('direct_reduction_percent', 'Direct GPP loss'),
                         ('science_reduction_percent', '7/30-day science composite')]:
        ax.plot([0] + [r['new_update'] for r in rows], [0] + [r[field] for r in rows],
                marker='o', linewidth=2, label=label)
    ax.set(xlabel='Additional accepted updates', ylabel='Loss reduction from corrected baseline (%)',
           title='Job 14979295: fixed development monitor', xticks=range(11))
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(args.report.with_suffix('.learning.png'), dpi=170)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
