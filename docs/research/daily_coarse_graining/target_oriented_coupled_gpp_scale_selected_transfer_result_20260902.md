# Coupled-GPP Scale-Selected 7/30-Day Transfer Result

Date: 2026-09-02

Status: primary transfer gate accepted with a direct-GPP warning

Executable commit: `67a4d40`

Result SHA256: `b77e92fefdce5a427c471118f8b003bf69f01abb23c184ea86ba5eef54e5f625`

## Result

The free `gln01` execution completed in `3,532.066` seconds. It reconstructed
the selected updates 5-7 at scales `1/4`, `1`, and `1/2`, then compared the
state before update 5 with the state after update 7 on the eight held-aside
train/train monitor artifacts.

| Source | Before | After | Change | Artifacts improved |
| --- | ---: | ---: | ---: | ---: |
| direct teacher-state GPP | 0.047871 | 0.066678 | +39.2867% | 1/8 |
| reachable one-day state | 0.029141 | 0.023827 | -18.2362% | report only |
| 7-day component-resolved science | 0.082414 | 0.075567 | -8.3089% | 6/8 |
| 30-day component-resolved science | 0.100720 | 0.087189 | -13.4337% | 8/8 |

Both preregistered horizon means strictly improve and all before/after
physical gates pass, so the primary transfer classification is
`non_test_seven_thirty_day_transfer_accepted`. Fixed parameters and the parent
recovery remain unchanged. No reconstructed parameter tree, optimizer state,
checkpoint, or recovery is retained. Peak JAX bytes in use are
`4,114,680,576`.

## Interpretation

This is the first evidence that several accepted local updates accumulate into
material improvement on independent 7-day and 30-day free rollouts. It is
stronger than the preceding same-batch finite-step results.

It is not sufficient to launch an unguarded 100-update run. The horizon total
macro-averages cumulative GPP, maintenance respiration, growth respiration,
heterotrophic respiration, and terminal AGB/BGB/nonliving carbon errors. The
current report did not retain each key separately, so aggregate improvement
can mask a GPP regression. The independent one-day GPP loss increased by
`39.2867%`, including regressions on seven of eight artifacts. This warning is
scientifically material even though it was report-only in the frozen transfer
gate.

## Next Gate

Freeze and locally qualify a persistent scale-selecting 100-update trial with
the following additions:

- monitor cumulative 7-day and 30-day GPP separately from the aggregate;
- retain the direct teacher-state GPP monitor and require recovery to the
  pre-update-5 baseline before a checkpoint can be selected;
- retain per-key GPP, respiration, NPP, AGB, BGB, and carbon reporting;
- stop or withhold a checkpoint when aggregate improvement is obtained only by
  sacrificing GPP;
- keep the same Attention architecture, conservative updater, retained exact
  tail, train/train data boundary, and per-update common-direction scale
  selection.

No paid run is authorized until this amended monitor and recovery/checkpoint
path pass local and free-GPU admission. Fresh validation, 365-day rollout,
Gate D2, sealed test, and product acceptance remain closed.
