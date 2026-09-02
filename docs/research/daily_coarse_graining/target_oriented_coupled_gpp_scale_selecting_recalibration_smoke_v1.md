# Coupled GPP Scale-Selecting Recalibration Smoke V1

## Question

The update-5 audit proves that the recalibrated common direction has a bounded
finite trust region: `1/4` is the largest tested scale that improves direct
GPP, reachable state, 7-day science, and 30-day science together. This smoke
asks whether the same rule continues to work after that selected step, rather
than only at the audited update-5 state.

The frozen machine contract is
`manifests/coarse_graining/target_oriented_coupled_gpp_scale_selecting_recalibration_smoke_v1.json`,
canonical SHA256
`3b7109fcccc5a721c96d1ac49eba2fe92a5a7fa0c8386783d562da33fc1bf8a5`.

## Mechanism

The runner reconstructs the accepted update-5 scale audit in one process and
selects its largest passing scale, `1/4`, as an in-memory state. It then tests
new scheduled updates 6 and 7 on frozen artifact groups. At each update it:

1. recomputes all four source gradients at the current parameters;
2. solves the same minimum-norm common-direction problem;
3. constructs the same dtype-preserving full displacement;
4. evaluates scales `1,1/2,1/4,1/8,1/16,1/32` from largest to smallest;
5. applies only the first scale passing every source and physical gate;
6. stops immediately if no common direction or no tested finite scale passes.

Smaller scales are not evaluated after the first passing scale. This is the
actual proposed step-selection rule and avoids spending free-node time on
scales that cannot be selected.

## Data And Matched Boundary

All artifacts are frozen train/train records from `pft14-daily-teacher-v6`.
Update 5 uses artifacts `965,351,272,363,660,670,548,740`; updates 6 and 7 use
`332,994,933,930,699,916,387,427` and
`675,417,455,803,779,284,385,520`. Each update covers 1,664 direct GPP Teacher
days, 256 reachable-state transitions, 64 seven-day windows, and 32 thirty-day
windows.

There is no architecture control in this smoke. The matched boundary is the
same frozen candidate, data, four objectives, source-gradient solver, full
step norm, scale grid, and physical gate at each consecutive state. The only
new mechanism is selecting the largest passing finite scale instead of always
using the full displacement.

## Acceptance And Retention

The smoke passes only if updates 5, 6, and 7 form three consecutive accepted
scale-selected recalibrations. Every selected step must keep all four source
losses non-worsening at relative tolerance `1e-6`, strictly reduce their
frozen weighted objective, and pass every one-day, 7-day, and 30-day physical
and nonfinite gate.

No proposal, optimizer state, recovery, or checkpoint is written. The final
parameters are discarded after reporting. No fresh validation or sealed-test
value is read; no Teacher data is regenerated; no v5 data, 48-record forcing
reconstruction, or subdaily state scan is used.

## Execution And Next Gate

One free `gln01` GPU execution is allowed with a 120-minute timeout. Paid
training remains closed. If all three updates pass, the next gate is a
separate no-checkpoint non-test 7/30-day transfer audit of the selected final
state. Failure triggers attribution of direction feasibility or finite
curvature without changing the architecture.

Local qualification passes 42 coupled-GPP/reachability tests plus Ruff,
formatting, Python compilation, Bash syntax, and whitespace checks.
