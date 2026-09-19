# Continuous dual training: retained-state evidence and timeout

Job15269596 FAILED124:0 after1:55:03 onibc13b02n02. It started2026-09-18
10:29:25 and ended12:24:28 server time, at the115min inner timeout.
Immutable9cc50dd,2V100/8CPU/64GiB; estimated additive charge9.5108yuan,
below approved9.92yuan. This is not a completed original protocol.

All three real paths performed and saved three updates with changing daily,
T7 andT30 data. Serial and uninterrupted dual finished including persisted
forward replay. Recovered execution loaded update2 in a new process, performed
update3, saved the full checkpoint and completed the final TRAIN/development
monitor. Timeout occurred after that monitor and before final persisted-forward
replay/completion was recorded. There is no reported numerical exception/OOM.
Four process startups, substantial compilation and monitoring dominated time;
the main agent underestimated the overall envelope. Do not repeat all GPU legs
to recover the final comparison. Preserve the original running-status report
and scheduler failure rather than relabeling either completed.

## Read-only evidence recovered on free CPU

`scripts/hpc/audit_saved_dual_integration.py` ran ontest01 with deployedJAX0.4.38,
using the unchanged checkpoint loader, full-tree numerical comparator and
per-update scalar/physical/data comparison. No new training or Teacher reads.
Result `outputs/mixed-learning-dual-integration-15269596.saved-audit.json`:

- Three matched updates each: scalar parity passes, physical counters identical
  and zero, daily/T7/T30 data identical between arms and change across updates.
- Final full parameters and AdamW both pass for dual-vs-serial and
  recovered-vs-dual, with zero coordinates outside existing tolerances.
- Differences are nonzero floating-point differences; do not claim bitwise
  identity. Dual-vs-serial parameter relativeL2 is about8.84e-9; AdamW1.96e-8.
- Shared checkpoint method/data/initialization checks pass. Only the explicit
  serial/dual placement field differs. Optimizer step and both data cursors3.
- Tiny two-GPU deployed continuity fixture also passed in3.5875s.

Claim admitted: bounded continuous changing-data dual updates, synchronization
and checkpoint recovery agree numerically with serial/uninterrupted execution.
Claim not admitted: original all-legs completion, recovered final saved-forward
replay, long-run throughput or scientific model quality. The missing replay is
an operational evidence gap, not an observed mismatch. Keep it explicit; fold
its completion into the next authorized execution rather than repeat the entire
paid test. No additional paid submission has been made.

## Timings and memory

| Path | Update1 seconds | Update2 seconds | Update3 seconds |
| --- | ---: | ---: | ---: |
| Serial |37.485|13.054|12.387|
| Continuous dual |50.750|7.707|7.491|
| Dual2/restart/3 |48.923|7.416|47.686|

For updates2/3, mean12.7201s serial versus7.5992s dual, speedup1.6739x,
40.26% shorter. Only two warm changing-data samples: not a sustained-run promise.
Dual steady branches daily7.207/6.985s, recurrent6.871/6.395s; merge/check/update
0.368/0.376s, peer sync0.073/0.075s. Restart update3 incurs cold helper costs.
Peak live memory dualGPU0~16.14GiB/GPU1~2.66GiB, reserved~26.98GiB each.
Equal-update GPU-hour estimate increases~19.5%; additive CPU+GPU estimate~7.4%
with8CPU in either case. The actual short integration was dominated by startup,
so these warm estimates do not describe its9.51yuan diagnostic bill.

## Next work

Return to a declared larger-coverage direct mixed-learning plan with dual
execution, batch64/micro8 and oneT7+oneT30 window per update as the admitted
starting point. Keep architecture/loss/AD unchanged. Specify initialization,
data coverage, update budget, physical/scientific monitors and checkpoint
selection before launch. Paper method remains fresh/full-network reproducible;
the three-update diagnostic checkpoints are not default learning initialization.
Budget startup separately and avoid four repeated startup/monitor cycles in
one short training job. More windows, micro16 or four-card scaling remain
optional later studies, not new prerequisites. No new paid run approved here.
