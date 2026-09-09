"""Bounded progress IO; complete scientific evidence is saved at lifecycle boundaries."""
from __future__ import annotations

import time
from pathlib import Path

PROGRESS_EVENTS = frozenset({'horizon_start', 'horizon_complete', 'reverse_day', 'reverse_compile', 'reverse_memory'})


class ContinuationReporter:
    def __init__(self, path, report, *, write, interval=15., clock=time.perf_counter):
        self.path, self.report, self.write = Path(path), report, write
        self.clock, self.started, self.last_progress = clock, clock(), float('-inf')
        self.interval = interval
        self.progress_path = self.path.with_name(self.path.stem + '.progress.json')
        self.full_writes = self.progress_writes = 0
        self.write_seconds = 0.

    def __call__(self, stage, updates):
        now = self.clock()
        self.report.update(updates)
        self.report.update(stage=stage, elapsed_seconds=now - self.started)
        progress = stage in PROGRESS_EVENTS
        if progress and now - self.last_progress < self.interval:
            return
        began = self.clock()
        if progress:
            payload = {key: self.report[key] for key in
                       ('status', 'stage', 'elapsed_seconds', 'logical_update', 'progress') if key in self.report}
            self.write(self.progress_path, payload)
            self.progress_writes += 1
            self.last_progress = now
        else:
            self.full_writes += 1
            self.report['reporting'] = {'full_writes': self.full_writes, 'progress_writes': self.progress_writes,
                                        'write_seconds_before_current': self.write_seconds}
            self.write(self.path, self.report)
        self.write_seconds += self.clock() - began
        print(f'corrected_continuation stage={stage} elapsed={now - self.started:.1f}s', flush=True)
