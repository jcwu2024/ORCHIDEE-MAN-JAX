# Shared daily/T7/T30 preparation while daily AD runs

Scope: implement the already agreed shared trajectory objective and prepare
its CPU actual-data interface. This does not alter architecture/output
coordinates or the running immutable158192f daily admission (wrapper100848).
It does not launch GPU work before that daily numerical-AD/update gate passes.

`shared_recurrent_admission_v1.json` fixes the existing daily-v3 architecture,
fresh seed, full-network updates, and native JAX checkpointed daily scan.
Use the first two existing TRAIN calibration artifacts, first T7 and T30
window in each. These four windows are technical admission samples, not an
independent scientific validation set. The bounded512-row TRAIN calibration
already accepted in the v3 forward report is reused; no new Teacher arrays,
dataset expansion or full-statistics pass is required.

The prepared window owns the real day0 continuous/discrete state, consecutive
daily templates, and the per-window retained bundle broadcast along time.
The public canonical step overwrites each later template's state with its own
previous prediction and rebuilds state-dependent PFT views. Teacher states
after day0 are supervision only. Stock-defined masks come from the common
day0 state, not the first expected endpoint, and apply to both trajectories.

Exact joint weights remain those in the shared long-term design:

```
daily contribution = process_day/3 + daily_flux/9 + daily_stock/9
T7 contribution    = T7_flux/9 + T7_stock/9
T30 contribution   = T30_flux/9 + T30_stock/9
```

The three gradients sum before one optimizer update. The existing standalone
daily-v3 objective is unchanged. Do not average three standalone scalar losses
or average gradients a second time. Recurrent terms retain the shared daily
and cumulative-flux penalties, stock changes/trajectories, physical TRAIN
units, living/nonliving balance and required physical/science hard counters.

Local tests cover retained-window indexing, masks, recurrence, exact scalar
and gradient composition, native checkpointed derivatives and hard counters.
`prepare_shared_recurrent_admission.py/.sh` then reads actual prepared data,
checks four forward shapes and first-artifact T7/T30 gradient shapes on CPU.
It never initializes numerical neural weights or executes the full model.
Its report explicitly says no numerical gradient or scientific acceptance.

After daily AD passes and CPU preparation is accepted: complete the bounded
GPU runner, measure T7/T30 forward/physical checks, compiled/live memory,
full-network gradients and directional agreement, then two complete joint
updates. Keep each completed expensive result before a later compilation.
Only after that gate discuss a bounded learning run and independent panels.
No paid submission is part of this preparation.

Local integration:26 tests passed in11.27s across recurrent preparation,
recurrent objective, existing daily objective and shared trajectory loss;
Ruff passed. This includes native checkpoint/plain AD on a small synthetic
recurrence and exact joint value/gradient composition. These are not numerical
full75.58M model gradient results. Seven-stock nonnegativity uses the physical
trajectory masks, separately from the common-initial supervision masks.

Execution snapshot: immutable605c941 deployed to
`runtime/checkouts/shared-recurrent-605c941`; actual target-shell dry-run passed.
Free test02 wrapper45145 runs with20min timeout, report/log prefix
`runtime/outputs/diagnostics/shared-recurrent-605c941-cpu`.
At3m50s elapsed it had loaded the existing TRAIN artifacts and constructed
parameter shapes (157.9s); daily and window graph checks were still pending.
This is a RUNNING snapshot, not a passed result. The GPU admission runner is
not yet complete. Concurrent daily wrapper100848 was still running at43m52s,
in full-gradient compilation, with no numerical AD/update outcome yet.
