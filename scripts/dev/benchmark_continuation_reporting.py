"""CPU-only replay of old full-report IO versus bounded continuation progress IO."""
from __future__ import annotations

import argparse
import copy
import json
import time

from research.daily_coarse_graining.daily_operator.continuation_reporting import ContinuationReporter
from scripts.hpc.run_successor_process_head_canonical_gate import _atomic_json


def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--events', type=int, default=12)
    args = parser.parse_args()
    original = json.loads(args.report.read_text())
    rows = []
    for count in (1, 9):
        report = copy.deepcopy(original)
        report['records'], report['monitors'] = report['records'][:count], report['monitors'][:count]
        path = args.output_dir / f'old-{count}.json'
        started = time.perf_counter()
        for ordinal in range(args.events):
            report.update(stage='horizon_complete', progress={'ordinal': ordinal, 'horizon': 30})
            _atomic_json(path, report)
        old = (time.perf_counter() - started) / args.events
        # Force every compact write: a conservative comparison without throttling gains.
        new = ContinuationReporter(args.output_dir / f'new-{count}.json', report, write=_atomic_json, interval=0.)
        started = time.perf_counter()
        for ordinal in range(args.events):
            new('horizon_complete', {'progress': {'ordinal': ordinal, 'horizon': 30}})
        bounded = (time.perf_counter() - started) / args.events
        new('completed', {})
        assert json.loads(new.path.read_text())['records'] == report['records']
        rows.append({'retained_updates': count, 'events': args.events, 'old_seconds_per_event': old,
                     'new_seconds_per_event': bounded, 'full_report_bytes': path.stat().st_size,
                     'progress_bytes': new.progress_path.stat().st_size})
    result = {'cpu_only': True, 'rows': rows, 'excludes_gpu_compute': True}
    _atomic_json(args.output_dir / 'benchmark.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
