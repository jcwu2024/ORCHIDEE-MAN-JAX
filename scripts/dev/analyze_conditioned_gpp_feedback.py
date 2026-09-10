"""Compare saved Teacher-state and free-rollout GPP errors without model execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def errors(values):
    value = np.asarray(values, dtype=float)
    if value.size == 0 or not np.isfinite(value).all():
        raise ValueError('analysis requires nonempty finite paired observations')
    return {'count': int(value.size), 'rmse': float(np.sqrt(np.mean(value**2))),
            'bias': float(np.mean(value)), 'mae': float(np.mean(np.abs(value)))}


def trajectory(row, label, key, category='fluxes'):
    value = np.asarray(row[label][category][key], dtype=float)
    if value.shape != (row['horizon'],) or not np.isfinite(value).all():
        raise ValueError('analysis requires one complete trajectory per window')
    return value


def analyze(report, curves):
    if not curves['report_reproduced'] or curves['status'] != 'completed':
        raise ValueError('requires the completed, report-matched curve export')
    rows = curves['windows']
    start = report['plan']['starting_logical_update']
    stop = report['completed_logical_update']
    versions = {f'update{start}': 'independent_before', f'update{stop}': 'independent_after'}
    # The canonical acceptance statistic divides window sums by horizon * scale.
    # Thus physical window-mean RMSE / reported normalized RMSE recovers scale.
    # This is unit conversion of the saved metric, not fitting a new normalizer.
    scales = []
    for label, source in versions.items():
        for split in ('temporal', 'spatial', 'joint'):
            for horizon in (7, 30):
                selected = [r for r in rows if r['slice'] == split and r['horizon'] == horizon]
                mean_errors = [np.mean(trajectory(r, label, 'gpp') - trajectory(r, 'teacher', 'gpp'))
                               for r in selected]
                saved = report[source]['slices'][split][str(horizon)]['keys']['gpp']
                if len(selected) != saved['evaluated_values']:
                    raise ValueError('curve/report window counts differ')
                physical = errors(mean_errors)
                scale = physical['rmse'] / saved['normalized_rmse']
                if not np.isclose(physical['bias'] / scale, saved['normalized_signed_bias'],
                                  rtol=1e-5, atol=1e-7):
                    raise ValueError('saved bias does not share the recovered scale')
                scales.append(scale)
    scale = float(np.mean(scales))
    if not np.allclose(scales, scale, rtol=1e-5, atol=1e-7):
        raise ValueError('GPP acceptance scale is not common across the paired panel')

    result = {'schema_version': 'conditioned_gpp_feedback_analysis_v1', 'model_evaluations': 0,
              'trained_updates': 0, 'gpp_physical_unit': 'gC m-2 land day-1',
              'recovered_frozen_gpp_scale': scale, 'scale_reproduction_cells': len(scales),
              'comparison': {}, 'cumulative_carbon': {}, 'training_evidence': {}}
    for label, source in versions.items():
        cells = []
        for split in ('temporal', 'spatial', 'joint'):
            for horizon in (7, 30):
                selected = [r for r in rows if r['slice'] == split and r['horizon'] == horizon]
                free = errors(np.concatenate([trajectory(r, label, 'gpp') - trajectory(r, 'teacher', 'gpp')
                                              for r in selected]))
                direct = report[source]['direct_teacher_state_gpp']['slices'][split][str(horizon)]
                if free['count'] != direct['evaluated_days']:
                    raise ValueError('direct/free daily counts differ')
                direct_rmse = direct['normalized_rmse'] * scale
                cells.append({'split': split, 'horizon': horizon, 'days': free['count'],
                              'teacher_state_rmse': direct_rmse,
                              'teacher_state_bias': direct['normalized_signed_bias'] * scale,
                              'free_rollout': free,
                              'free_to_teacher_state_rmse_ratio': free['rmse'] / direct_rmse})
        daily = errors(np.concatenate([trajectory(r, label, 'gpp') - trajectory(r, 'teacher', 'gpp')
                                       for r in rows]))
        direct = report[source]['direct_teacher_state_gpp']['aggregate']
        result['comparison'][label] = {'cells': cells, 'aggregate': {
            'days': daily['count'], 'teacher_state_rmse': direct['normalized_rmse'] * scale,
            'teacher_state_bias': direct['normalized_signed_bias'] * scale, 'free_rollout': daily,
            'free_to_teacher_state_rmse_ratio': daily['rmse'] / (direct['normalized_rmse'] * scale)}}
        result['cumulative_carbon'][label] = {}
        for horizon in (7, 30):
            selected = [r for r in rows if r['horizon'] == horizon]
            summary = {}
            for key in ('gpp', 'npp', 'maintenance_respiration', 'growth_respiration', 'autotrophic_respiration'):
                truth = [np.sum(trajectory(r, 'teacher', key)) for r in selected]
                predicted = [np.sum(trajectory(r, label, key)) for r in selected]
                summary[key] = errors(np.asarray(predicted) - truth) | {
                    'teacher_mean': float(np.mean(truth)), 'model_mean': float(np.mean(predicted))}
            for key in ('aboveground_biomass_carbon', 'belowground_biomass_carbon'):
                # With a shared day-zero state, endpoint difference equals error in net change.
                summary[key + '_net_change_error'] = errors([
                    trajectory(r, label, key, 'stocks')[-1] - trajectory(r, 'teacher', key, 'stocks')[-1]
                    for r in selected])
            result['cumulative_carbon'][label][str(horizon)] = summary

    actual = []
    years, artifact_ids = set(), set()
    for record in report['records']:
        selected = next(s for s in record['scale_evaluations']
                        if s['scale'] == record['selection']['selected_scale'])
        actual.append({'logical_update': record['logical_update'],
                       'before_loss': selected['combined_artifact_direct_gpp_before'],
                       'after_loss': selected['combined_artifact_direct_gpp_after']})
        for group in record['baseline']['artifact_groups']:
            for item in group['artifact_losses']:
                years.add(item['runtime_year'])
                artifact_ids.add(item['artifact_update'])
    monitors = [(start, report['initial_monitor'])] + [(m['logical_update'], m['panel']) for m in report['monitors']]
    monitor_rows = []
    for step, panel in monitors:
        monitor_rows.append({'logical_update': step,
                             'direct_loss': panel['source_mean_losses']['direct_gpp_208_teacher_days'],
                             'gpp_30_normalized_rmse': panel['acceptance']['30']['keys']['gpp']['normalized_rmse'],
                             'npp_30_normalized_rmse': panel['acceptance']['30']['keys']['npp']['normalized_rmse']})
    result['training_evidence'] = {
        'actual_train_steps': actual, 'fixed_train_split_monitor': monitor_rows,
        'actual_training_runtime_years': sorted(years), 'distinct_artifact_ids': len(artifact_ids),
        'direct_loss_definition': '0.5 * (normalized active-PFT MSE + squared mean bias)',
        'limitations': ['Changing training batches and checkpoints are not a fixed-set learning curve.',
                       'Fixed monitor uses train/train data, but its eight artifact IDs are excluded from these updates.',
                       'Training active-PFT loss and validation land-area RMSE are not directly interchangeable.',
                       'RMSE difference is not an additive causal decomposition of recurrence error.',
                       'Initial stock values were not exported; shared-initial-state endpoint errors give net-change errors, not net-change levels.']}
    old, new = (result['cumulative_carbon'][label]['30'] for label in versions)
    checks = {key + '_rmse_decreased': new[key]['rmse'] < old[key]['rmse'] for key in
              ('gpp', 'npp', 'aboveground_biomass_carbon_net_change_error', 'belowground_biomass_carbon_net_change_error')}
    checks.update({key + '_rmse_within_2_percent': new[key]['rmse'] <= old[key]['rmse'] * 1.02
                   for key in ('maintenance_respiration', 'autotrophic_respiration')})
    checks['relative_transfer_passed'] = bool(report['independent_transfer_passed'])
    result['physical_progress_review'] = {
        'checks': checks, 'passed': all(checks.values()), 'selects_checkpoint': False,
        'final_model_accepted': False,
        'interpretation': 'Post-stage development review; absolute gates and final reproducible training remain separate.'}
    return result


def plot(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams['font.family'] = 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    current_label = list(result['comparison'])[-1]
    current = result['comparison'][current_label]['cells']
    names = {'temporal': '时间留出', 'spatial': '空间留出', 'joint': '时空留出'}
    x = np.arange(len(current))
    axes[0].bar(x - .18, [c['teacher_state_rmse'] for c in current], .36,
                label='每天给教师状态', color='#527aa3')
    axes[0].bar(x + .18, [c['free_rollout']['rmse'] for c in current], .36,
                label='连续使用模型状态', color='#087f8c')
    axes[0].set_xticks(x, [f"{names[c['split']]}\n{c['horizon']}日" for c in current])
    axes[0].set(ylabel='逐日 GPP RMSE（gC/m²/天）', title='同一批天：状态反馈改变了多少误差？')
    axes[0].legend(fontsize=9)
    monitor = result['training_evidence']['fixed_train_split_monitor']
    train = result['training_evidence']['actual_train_steps']
    axes[1].plot([m['logical_update'] for m in monitor], [m['direct_loss'] for m in monitor],
                 color='#087f8c', label='固定训练划分监测集')
    axes[1].scatter([m['logical_update'] for m in train], [m['after_loss'] for m in train],
                    color='#b68b47', s=25, label='本次更新使用的40包（每次不同）')
    axes[1].set(xlabel='逻辑更新步', ylabel='单日 GPP 训练损失（无量纲）', title='单日拟合仍有残差，固定监测仍在改善')
    axes[1].legend(fontsize=9)
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    figure.suptitle(f'{current_label} 诊断：左侧为留出验证；右侧为训练划分，两个纵轴口径不同', fontsize=12)
    figure.tight_layout()
    figure.savefig(output / 'gpp_feedback.png', dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('curves', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    if str(args.curves) == '-':
        before = report['independent_before']['trajectories']
        after = report['independent_after']['trajectories']
        start, stop = report['plan']['starting_logical_update'], report['completed_logical_update']
        windows = []
        for old, new in zip(before, after, strict=True):
            if any(old[key] != new[key] for key in ('entry', 'slice', 'horizon', 'teacher')):
                raise ValueError('before/after embedded curves do not pair')
            windows.append({k: new[k] for k in ('ordinal', 'slice', 'horizon', 'entry', 'teacher')} |
                           {f'update{start}': old['model'], f'update{stop}': new['model']})
        curves = {'windows': windows, 'status': 'completed', 'report_reproduced': True}
    else:
        curves = json.loads(args.curves.read_text())
    result = analyze(report, curves)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    plot(result, args.output)
    print(json.dumps({k: v['aggregate'] for k, v in result['comparison'].items()}, indent=2))


if __name__ == '__main__':
    main()
