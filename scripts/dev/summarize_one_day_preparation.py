"""Read saved worker metadata to inventory completed expansion and missing work."""
import argparse
import json
from pathlib import Path


def run(root, plan_path, output):
    plan = json.loads(plan_path.read_text())
    expected = {entry['id'] for entry in plan['entries']}
    records, workers = {}, []
    for path in sorted((root/'chunks').glob('worker-*/prepared_manifest.json')):
        report = json.loads(path.read_text())
        entries = report['entries']
        workers.append({'worker': path.parent.name, 'status': report['status'], 'stage': report['stage'],
                        'completed': len(entries), 'assigned': len(report['entry_ids']),
                        'remaining': len(report['entry_ids'])-len(entries),
                        'elapsed_seconds': report['elapsed_seconds'],
                        'error': report.get('error')})
        for entry in entries:
            identifier = entry['update']
            if identifier in records or identifier not in expected:
                raise ValueError('worker entry identity overlap or unexpected entry')
            file_path = path.parent/entry['path']
            if not file_path.is_file() or file_path.stat().st_size != entry['size_bytes']:
                raise ValueError(f'recorded artifact missing or size differs: {file_path}')
            records[identifier] = entry
    rows = sum(entry['rows'] for entry in records.values())
    result = {'status': 'complete' if set(records) == expected else 'incomplete',
              'root': str(root), 'planned_point_years': len(expected), 'completed_point_years': len(records),
              'remaining_point_years': len(expected-set(records)), 'completed_rows': rows,
              'planned_rows': plan['counts']['combined_unique_train_days'],
              'remaining_rows': plan['counts']['combined_unique_train_days']-rows,
              'completed_compressed_bytes': sum(entry['size_bytes'] for entry in records.values()),
              'completed_uncompressed_bytes': sum(entry['uncompressed_batch_bytes'] for entry in records.values()),
              'summed_preparation_seconds': sum(entry['elapsed_seconds'] for entry in records.values()),
              'all_saved_old_rows_exact': all(entry['old_row_exact'] for entry in records.values()),
              'workers': workers, 'remaining_ids': sorted(expected-set(records)),
              'merge_report_exists': (root/'merge_report.json').exists()}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('workers', 'remaining_ids')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'plan', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.plan, args.output)
