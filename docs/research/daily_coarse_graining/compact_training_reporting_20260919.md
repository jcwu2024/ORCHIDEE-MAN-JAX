# Compact progress and periodic full training reports

User authorized implementation after the reporting-overhead explanation.
This changes only report persistence in the next checkout. Active paid
job15279139 stays on immutable `d1183e6`; no GPU test or new paid job is needed
for this reporting-only graph-unchanged revision.

## Behavior

`MixedLearningReporter` writes `report.progress.json` at every event. It contains
current stage/update, latest scalar losses, data counts, memory, latest known
checkpoint and the update/elapsed time of the last complete report. It excludes
full historical updates and monitor trajectories. Stale `next_update` and
`monitor_update` fields are not carried into unrelated stages or completion.

`report.json` retains the complete existing history and scientific fields.
It is flushed every25 optimizer updates and at all non-update stage boundaries,
including monitors, compilation, failures, stop and completion. `--report-interval`
can select another positive interval;1 keeps eager full reports. Diagnostic
profiling remains eager. No history is pruned from memory and no asynchronous
writer, per-step file or new derivative implementation is introduced.

There are two fixed report files per run plus the writer's temporary file,
rather than a file per update. Full report polling can lag normal training by
up to24 updates; use the progress sidecar or Slurm text log for current status.
Science summaries continue to read the complete report and only show recorded
monitor outcomes. Live update progress is not a new accuracy measurement.

## Process interruption and recovery

Before the checkpoint is replaced, `checkpoint_saving` forces all corresponding
update records to the complete report. After replacement, `checkpoint_saved`
flushes the new checkpoint metadata. A report ahead of an old checkpoint is
trimmed through the existing shared recovery path; records are never invented.

If interruption happens after checkpoint rename but before the metadata flush,
resume reconciles that checkpoint's path/update/bytes from the file actually
loaded. A missing save duration remains missing; it is not fabricated as zero.
This also handles a resumed checkpoint already at the final update, where no
further save would repair metadata. Existing matching records are preserved.

Abrupt process death between regular flushes may lose the last up to24 unsaved
telemetry rows. Optimizer recovery still rolls back to the latest checkpoint
(normally every50 updates); all records through that checkpoint are retained.
Gracefully caught failures flush the full report. This is process-interruption
handling, not a power-loss durability claim: the existing report writer uses
atomic replace without fsync. Scientific method, checkpoint schema, parameter
updates, data sampling and loss remain unchanged. Cross-checkout checkpoint
method/provenance checks have not been relaxed.

## Checks and timing evidence

Thirty local reporter/runner/checkpoint tests pass. They cover periodic versus
eager identical complete JSON, preserved history, bounded current progress,
failure/stop/monitor flushes, stale live fields, write failures, and interruptions
before/after actual checkpoint replacement using the shared loader. Independent
Sol/high review identified metadata reconciliation and stale-field issues;
both were fixed and the bounded re-review found no remaining concrete issue.

The CPU cadence probe replays the last25 update events of the saved joint600
report using the actual `json_safe -> _json_value -> json.dumps` chain. It does
not evaluate a model, read Teacher data or write the full reports to disk.
Both policies produce exactly the same final complete JSON. Local results:

| Policy | Full reports | Compact progress | Formatting wall time |
| --- | ---: | ---: | ---: |
| Eager |55|55|44.802s|
| Every25 + boundaries |6|55|5.650s|

This is87.39% less **report-formatting** wall time in that diagnostic, not87%
faster training. Filesystem costs and real numerical compute are excluded.
Both modes include compact sidecars, so the eager probe is slightly more work
than the original single-file writer. The shared-node diagnostic below must
also not be treated as an allocated whole-training benchmark.

Evidence: `outputs/mixed-learning-review/report-cadence-local.json`.

The separate server checkout `runtime/checkouts/mixed-reporting-v1` is at
`16c0bad`. Free test02 replay also PASSED: eager48.260s versus periodic6.093s,
87.37% less formatting time;55 versus6 full writes,55 compact writes each,
exactly identical final complete JSON. Artifact:
`outputs/mixed-learning-review/report-cadence-test02.json`. Actual target-shell
dry-run accepts `--report-interval 25` and propagates a non-default GPU mask3,5.
This dry-run submits and computes nothing. The test01 CPU environment lacks
pytest, so the30-test regression result is local; no package was installed.
The dependency-free server cadence replay ran against the actual deployed
formatter/reporter. Active job15279139 remained running, observed at922updates,
without any checkout or command change.

```powershell
conda run -n ORCJAX python scripts/hpc/probe_mixed_reporting_cadence.py --source-root E:/ORCHIDEE-MAN-JAX --report outputs/staged-pilot-joint-600-report.json --output outputs/mixed-learning-review/report-cadence-local.json --updates 25
```

Use this version for the next separately declared run after ordinary integration
checks; keep running paid code immutable. There is no authorization for another
paid submission or an automatic restart of the existing job in this change.
