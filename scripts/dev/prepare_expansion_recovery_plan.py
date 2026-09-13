"""Balance remaining committed-prefix work without moving prepared data."""
import argparse
import json
from pathlib import Path


def make_plan(inventory, workers=8):
    if inventory['status'] != 'incomplete' or workers < 1:
        raise ValueError('recovery needs incomplete work and a positive worker count')
    count = len(inventory['workers'])
    total = inventory['planned_point_years']
    queues = [{'rank': rank, 'estimated_seconds': 0., 'assignments': []} for rank in range(workers)]
    pending = []
    for row in inventory['workers']:
        if not row['remaining']:
            continue
        index = int(row['worker'].split('-')[-1])
        pending.append({'old_worker': index, 'first': index*total//count+1,
                        'last': (index+1)*total//count,
                        'remaining': row['remaining'],
                        'estimated_seconds': row['remaining']*row['elapsed_seconds']/max(row['completed'], 1)+45})
    for item in sorted(pending, key=lambda row: (-row['estimated_seconds'], row['old_worker'])):
        target = min(queues, key=lambda row: (row['estimated_seconds'], row['rank']))
        target['assignments'].append(item)
        target['estimated_seconds'] += item['estimated_seconds']
    return {'schema_version': 'one_day_expansion_recovery_v1', 'original_root': inventory['root'],
            'original_job': 15061225, 'remaining_point_years': inventory['remaining_point_years'],
            'remaining_rows': inventory['remaining_rows'], 'workers': queues,
            'estimated_max_worker_seconds': max(row['estimated_seconds'] for row in queues),
            'preserve_committed_artifacts': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = make_plan(json.loads(args.inventory.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'workers'}, indent=2))
