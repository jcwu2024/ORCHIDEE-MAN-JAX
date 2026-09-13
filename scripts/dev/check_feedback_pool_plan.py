"""Read metadata only to verify the declared epoch and rollout exclusions."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from scripts.hpc.probe_one_day_batch import _resolved_reference
from scripts.hpc.run_fresh_start_one_day_pool import pool_plan


def check_plan(catalogue_report_path, configuration_path, selection_path, control_report_path):
    catalogue_report = json.loads(catalogue_report_path.read_text())
    catalogue_path = _resolved_reference(catalogue_report['catalogue'], relative_to=catalogue_report_path.resolve())
    with gzip.open(catalogue_path, 'rt', encoding='utf-8') as handle:
        catalogue = json.load(handle)
    configuration = json.loads(configuration_path.read_text())
    selection = json.loads(selection_path.read_text())
    control = json.loads(control_report_path.read_text())
    sampler, panels = pool_plan(catalogue, configuration)
    used, draws, updates = set(), 0, 0
    for batch in sampler.iter_epoch(0):
        updates += 1
        draws += batch.draw_count
        used.update((row.landpoint_id, row.year, row.start_row) for row in batch.rows)
    monitor_days = {(row['landpoint_id'], row['year'], row['start_row']) for row in panels['excluded_days']}
    rollout_days = {tuple(day) for window in selection['windows'] for day in window['teacher_day_keys']}
    budget = configuration['declared_pool_budget']
    checks = {
        'catalogue_and_selection_accepted': catalogue_report['status'] == selection['status'] == 'passed',
        'declared_epoch_counts': (updates, draws, len(used)) == (budget['updates'], budget['row_draws'], budget['unique_teacher_days']),
        'monitor_excluded': not used.intersection(monitor_days),
        'rollout_excluded_from_train_and_monitor': not rollout_days.intersection(used | monitor_days),
        'same_control_sampling': configuration['sampling'] == control['configuration']['sampling'],
        'same_source_identity': catalogue['source_identity'] == selection['source_identity'] == control['method']['source_identity'],
        'complete_control_epoch': control['status'] == 'completed' and control['completed_updates'] == updates,
    }
    return {'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks,
            'updates': updates, 'draws': draws, 'unique_train_days': len(used),
            'monitor_days': len(monitor_days), 'rollout_days': len(rollout_days),
            'rollout_windows': len(selection['windows']), 'teacher_payload_read': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('catalogue-report', 'pool-protocol', 'selection', 'control-report', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = check_plan(args.catalogue_report, args.pool_protocol, args.selection, args.control_report)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['status'] == 'passed' else 1)
