# Training wall time and ready science review

This is read-only analysis while approved job15279139 continues unchanged at
immutable `d1183e6`. No extra paid work, Teacher generation, model inference,
loss change or checkpoint selection was performed. Timing inputs are existing
pilot/profile reports and logs, plus a partial snapshot of the active run.

## Findings that affect the next efficiency decision

The dual layout overlaps daily64 and T7+T30. In the admitted fixed-input
profile, median complete update cost was7.222s mixed versus6.536s daily-only.
At the same5415 daily updates, arithmetic lower-frequency estimates are:

| Recurrent frequency | Windows per horizon | Update-only hours | Reduction |
| --- | ---: | ---: | ---: |
| Every update |5415|10.863|0%|
| Every2 updates |2708|10.348|4.75%|
| Every4 updates |1354|10.090|7.12%|

These are NOT learning results. Daily-only profiling omits peer synchronization,
so savings are optimistic for the continuous dual path. Both GPUs remain
allocated; a quiet recurrent GPU does not stop its charge. Removing all recurrent
work would save only9.49% under this fixed-profile comparison. Keeping the
same recurrent-window count instead increases daily updates and runtime.
Lower recurrent frequency may change useful data exposure and learning, but is
not currently justified as a major dual-GPU wall-time optimization.

The earlier changing-data dual integration measured7.599s over two warm
updates, including host reads, validation, merge, optimizer and weight sync.
Daily/recurrent branches averaged7.096/6.633s; merge0.372s and sync0.074s.
Branch timings overlap and must not be added as wall time. Extra devices that
only shorten the already shorter recurrent branch cannot provide proportional
speedup without reassigning work or changing the workload.

An early live snapshot of job15279139 reached113 updates. Updates2-113 have
median7.5567s, mean7.9754s, empirical95th percentile7.6513s and maximum23.4577s;
slow observations are retained. Daily/recurrent branch medians7.0487/6.4046s,
merge0.3616s and peer sync0.0799s support the same daily-critical-path picture.
Ordinary no-save/no-monitor wall cycles have median8.0s and mean8.6775s
(including cold update1); outside-update residual averages0.4574s while only
monitor0 is present. Later reports will be larger. This is early throughput,
not a guarantee for all5415 updates, and no post-training science monitor has
yet been recorded in that snapshot.

## Wall time exceeds the update timer

| Existing run | Python elapsed | Timed updates | Monitors | Daily artifact reads |
| --- | ---: | ---: | ---: | ---: |
| Joint600 |9920.92s|7641.73s|303.34s|84.99s|
| Daily400+mixed400 |10065.59s|7689.41s|303.10s|108.47s|

Daily artifact reading was only0.86%/1.08% of elapsed time in these runs. Its
counter includes gzip/pickle decoding and identity validation, across training
and monitors, and overlaps the other timers. It excludes sample stacking,
host/device transfer and recurrent data reads. Thus it does not establish that
all data preparation is negligible or separately time all storage operations.

Log timestamps show additional costs between ordinary updates, even after
excluding intervals containing monitors or checkpoints:

- Joint:588 adjacent cycles, mean14.635s wall versus12.738s timed work,
  residual1.897s/update. Mean residual grows from0.71s in the first100
  to2.59s in updates501-600.
- Staged:784 cycles, residual1.536s/update on average, rising to2.59s in
  updates701-800. Daily and mixed objectives have different compute costs.

Source attribution: `run_mixed_learning.py::emit` runs before and after every
update, outside `record['seconds']`. It traverses `json_safe(report)` and writes
the entire accumulated report with `_atomic_json`, including all detailed
monitor trajectories, indenting/sorting JSON and calling fsync. Reader-memory
statistics and other Python bookkeeping also lie outside the update timer.
Therefore the measured residual is NOT solely disk IO or solely serialization.
The timestamp is taken before formatting, and log precision is0.1s.

A small free-test02 CPU replay of the actual `json_safe` plus `json.dump`
formatting on the saved14,583,383-byte joint600 report took0.779/0.745s per call;
process CPU times0.770/0.743s. No file writes/fsync, model evaluation or GPU work
were included. Local Windows replay took0.614/0.605s. These are diagnostic
causal evidence for nontrivial formatting work, not an accepted training-speed
benchmark or a promise of the same speedup on the allocated GPU host.

Rolling checkpoint metadata replaces each old duration for the same path, so
the sum of retained checkpoint timings is only a lower bound. Do not assign
the remaining elapsed time entirely to compilation or call it all wasted time.

## Bounded next engineering opportunity

After collecting this run, the most directly supported low-risk efficiency
candidate is separating cheap progress output from full historical report
serialization. A later implementation should retain complete per-update
records in memory, publish compact progress frequently, and flush full reports
at a declared interval and every checkpoint/monitor/completion/failure boundary.
Recovery must pair the optimizer checkpoint with matching report/cursors, and
crash-loss of telemetry must be stated. This changes reporting, not numerical
training, but is not implemented or applied to the active checkout here.

Expected savings must be measured before adoption; no claim that this solves
full-Teacher cost. Microbatch/layout/data-parallel work remains a possible later
route. Current16h maximum charge remains79.36yuan, without automatic extension.
The13-15h forecast needs observed complete wall cycles at later milestones;
7.6s alone is not an end-to-end throughput estimate. Increasing report size,
occasional slow updates and remaining cold paths are runtime uncertainties.

## Reusable analysis entries

`scripts/hpc/summarize_mixed_timing.py` reads saved JSON/logs and writes
`outputs/mixed-learning-review/timing.{json,md}`. It distinguishes process-local
elapsed, summed update time, monitor time, rolling-save lower bounds, overlapping
read counters, warm dual components and no-monitor/no-save wall cycles. Resumed
reports do not subtract cumulative updates from process-local elapsed.

`scripts/hpc/probe_mixed_report_overhead.py` only replays report formatting on
CPU. Evidence: `outputs/mixed-learning-review/report-cpu-overhead-test02.json`
and `report-cpu-overhead-local.json`. No Teacher payload or checkpoint is read.

The companion `scripts/hpc/summarize_mixed_learning.py` prepares all available
milestone curves from report JSON. Interpret the fixed-training and reused
development panels separately; excluded TRAIN daily rows are not independent
held-out sites. Report accumulated GPP/NPP/RA/RH, AGB/BGB change, canopy and
nonliving carbon change in physical units, with per-window predictions/bias
alongside RMSE. Zero-RH Teacher windows cannot certify nonzero RH behavior.
An initial-only active snapshot provides no evidence of new learned accuracy.

Current commands, from the project root:

```powershell
conda run -n ORCJAX python scripts/hpc/summarize_mixed_timing.py --report joint600=outputs/staged-pilot-joint-600-report.json --log joint600=outputs/staged-pilot-joint-600.log --report staged800=outputs/staged-pilot-daily400-mixed400-report.json --log staged800=outputs/staged-pilot-daily400-mixed400.log --report dual3=outputs/mixed-learning-dual-integration-15269596.dual.json --report active15279139=outputs/mixed-learning-dual-pool-15279139-live.json --log active15279139=outputs/mixed-learning-dual-pool-15279139-live.log --profile outputs/mixed-learning-dual-paid-15230086.json --output-dir outputs/mixed-learning-review
conda run -n ORCJAX python scripts/hpc/summarize_mixed_learning.py --report joint=outputs/staged-pilot-joint-600-report.json --report staged=outputs/staged-pilot-daily400-mixed400-report.json --output-dir outputs/mixed-learning-review --title "Matched-compute mixed-learning science review"
conda run -n ORCJAX python scripts/hpc/summarize_mixed_learning.py --report pool_live=outputs/mixed-learning-dual-pool-15279139-live.json --output-dir outputs/mixed-learning-review-active --title "Active pool mixed-learning snapshot"
```

Inputs are snapshots; refresh explicitly before interpreting active job progress.
The active job and its scientific protocol remain authoritative.

The historical and active bundles are separate so random initial predictions
from the active run do not obscure trained pilot curves. The active113 snapshot
has only monitor0 and is explicitly marked as lacking a learned science snapshot.
Chinese guide: `outputs/mixed-learning-review/读图说明.md`. Eight focused tests
pass for timing attribution, partial snapshots, growth versus inventory,
population separation, RH caveats and update-time semantics. Ruff passes;
T30 curves and signed per-window graphics were visually inspected.
