# Continuous-state input tail v1: bounded zero-update admission

Completed: numerically stabilized, rejected as a drop-in update-41 correction.
The original protocol below was frozen before model tests.

Keep update-41 weights, Attention topology, physical states, immutable v6 data
and statistics. Add an optional continuous-state feature transform. For the
standardized feature z=(x-mean)/scale and T=16:

```
f(z) = z                                  for abs(z) <= T
     = sign(z) * (T + log1p(abs(z) - T))    otherwise
```

This is monotone and C1, with derivative 1 inside and
1/(1+abs(z)-T) outside. T=16 is a fixed dimensionless engineering choice,
not fitted to validation or assumed to describe a Gaussian distribution.
The central range remains linear; outlying state information and gradients
remain available. Compute the enabled transform in float64 before returning
float32 neural features. This also avoids overflow of float32 standardized
values before compression. Disabled mode preserves legacy arithmetic.
Nonfinite/undefined inputs are masked before arithmetic. Raw physical state
and retained exact processes are untouched. No changes to output coordinates,
trainable scope, losses or score tolerance.

Apply through every continuous-state neural route: parent field/axis summaries,
PFT state, coordinate-local fields, structured GPP and carbon-feedback tokens.
Historical neural implementations use the same opt-in normalization setting.
Forcing/static/annual normalization is unchanged. The switch defaults to None;
the admission runner alone sets 16. Final training must record this protocol
and reproduce inherited pretraining if it is later adopted.

Matched reference: disabled switch, identical accepted update-41 weights and
identical train-only windows. No parameter updates or checkpoint selection.
Cases fixed before execution: (artifact 883, horizon 30, window 3), plus
ordinary existing train cases (322, 7, 1) and (442, 30, 0), zero-based indices.
The problematic saved day-10 state pair additionally tests the full DOC leaf
and coordinate 1785 in isolation. Do not search validation for a favorable T.

Stages, in order:

1. CPU saved-state audit: all defined continuous coordinates, Teacher day
   starts versus predicted states; count abs(z)>16/1e3/1e6, rank affected
   leaves, quantify paired feature differences. This is not global coverage.
2. Local tests: central identity, boundary derivatives, signed extremes,
   masking, finite JVP/VJP/finite differences and route/default compatibility.
3. Free gln01, one GPU, at most 45 minutes: paired production and compact
   trace on the three cases; saved day-10 DOC leaf/coordinate interventions.
   Require unchanged atol=1e-6, rtol=1e-5 score agreement on each corrected
   case, no defined-mask discrepancy, all physical hard counts zero and
   finite science trajectories. Report GPP/NPP/respiration cumulative error,
   AGB/BGB level and net-change errors in physical units. To advance, each
   case's science total and each named flux/stock MSE may increase by at most
   5% plus 1e-8 relative to the unchanged reference. This is a regression
   screen, not absolute scientific acceptance. Day-10 paired GPP difference
   must be <=1e-5 gC m-2 day-1 for both whole-leaf and coordinate probes.
4. Only if forward admission passes, rebuild derivative helpers with the
   enabled head and run existing one-day VJP/JVP and rounded-FD checks plus
   short recurrence reference and bounded streamed 7/30-day comparisons.
   Use the existing derivative policy; do not compile full 30-day monolithic
   reverse (previous measured temporary requirement 43.12 GiB).

Failure stops promotion, not evidence collection. Successful forward equality
alone does not establish gradients, model accuracy or final acceptance. Paid
training requires a separately concrete approved job. An eventual coverage
comparison must apply the same admitted input protocol to both data pools.

## Implementation and CPU evidence

The default-off setting is `NeuralInputNormalization.continuous_tail_threshold`.
`daily_operator/input_conditioning.py` owns the transform; the admission runner
rebuilds the head and forward bindings with T=16. Old training helper closures
are explicitly not reused. No production training enables it automatically.

Free test02 completed `audit_conditioned_input_tail_cpu.py` using the saved
30-day case. Teacher starts have 115,010 valid coordinate-days, with only four
above 16 (one distinct coordinate); maximum |z| is 21.9563. Compact predictions
have 114,991 defined coordinate-days, 261 above 16 across 24 coordinates. All
30 values above 1e3 and 1e6 belong to DOC coordinate 1785; maximum |z| is
7.42438e9. Other large predicted features include gsmean (282.76), interception
storage (63.14), soil heat flux (~40.14) and temperature (~27.15). This is one
train window, not a global Teacher distribution claim.

Applying the candidate to saved states reduces maximum paired float32 feature
difference over all 30 days from 1024 to 0.20711. That residual includes already
diverged downstream physical states; this offline number is not a new rollout
or evidence of model improvement. Actual rollout admission remains required.
Output: `outputs/conditioned-input-tail-cpu-14991510.json`.

## Local admission and free-GPU execution

Eight new tests pass (5.68s), including true JVP/VJP/rounded finite differences,
C1 boundaries, pre-cast extremes, undefined masks, constructor compatibility,
short recurrence and component regression rules. Thirty existing network and
streamed-adjoint/integration tests pass (521.03s), covering forward/loss,
parameter gradients, bound one-day budgets, inactive PFTs and permutation.
Ruff, Python CLI import, shell syntax and whitespace checks pass. These are
local implementation checks, not the real-v6 derivative admission.

Server checkout fast-forwarded from ce7cd79 to **fe3f6fd** after the preceding
diagnostics and CPU preparation had finished. GPU 1 was idle (3 MiB, 0%);
GPU 0 belongs to another running process and was left untouched. Launched
2026-09-10 at approximately 22:50 China time, free gln01/ibc13b02n04, Python
PID **18115**, one V10032GB, 45-minute timeout, allocator fraction 0.55:

```
GPU_DEVICE=1 EXPECTED_GIT_HEAD=fe3f6fd7977f4b0b8ed1545723c683ee151e4f6b \
  bash scripts/hpc/audit_conditioned_input_tail.sh
```

Working directory: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX`.
Report: `runtime/outputs/diagnostics/conditioned-input-tail-14991510.json`.
Log: `runtime/logs/conditioned-input-tail-14991510.log`.
The free-GPU run completed in 2459.7s, within the 45-minute cap. No parameter
updates or paid jobs. CPU audit's standalone arithmetic clarification was rerun separately
on test02; it uses exact legacy float32 arithmetic for the reference and yields
the same maximum feature gap of 1024.

## Completed result (read 2026-09-11)

All three corrected production/compact score pairs pass the original tolerance;
their absolute differences are 1.17e-7, 1.15e-10 and 1.33e-7. Production trace
statistics and trajectory masks agree, outputs are finite, all hard counts zero.
Coordinate 1785 alone reproduces the complete original 0.2896344 GPP jump;
the corrected coordinate probe gives exactly zero, and the whole-DOC probe
gives 7.96e-7 gC m-2 day-1.

Physical performance is mixed. Each cell shows unchanged -> all-state tail;
cumulative errors and final biomass errors are signed model-minus-Teacher in
gC m-2. Daily RMSE is gC m-2 day-1. These are train-only diagnostic windows.

| Case | Daily GPP RMSE | Cumulative GPP error | Cumulative NPP error | Cumulative maintenance error | Final AGB error | Final BGB error |
| --- | --- | --- | --- | --- | --- | --- |
| 883 / 30 days | 3.007 -> 3.810 | 49.94 -> 87.88 | 41.23 -> 49.76 | -2.06 -> 24.03 | 26.01 -> 29.25 | 14.78 -> 20.05 |
| 322 / 7 days | 3.405 -> 0.176 | 21.59 -> 1.11 | 15.53 -> 0.38 | 0.08 -> 0.61 | 5.89 -> 0.14 | 9.51 -> 0.23 |
| 442 / 30 days | 2.589 -> 2.067 | 74.25 -> 61.42 | 47.66 -> 27.78 | 8.06 -> 22.84 | 27.25 -> 14.48 | 20.12 -> 13.13 |

All three fail at least one declared component regression cap. Do not proceed
to derivative acceptance or continuation under this all-state transform. This
is not evidence that it could not be adapted through training; it does show
that numerical stabilization alone cannot be claimed as forward-preserving.
The next bounded DOC-only scope test is frozen in
[`conditioned_doc_tail_v1_20260911.md`](conditioned_doc_tail_v1_20260911.md).
