# Trained GPP State-Intervention Diagnosis

Date: 2026-09-13

Status: preregistered before numerical execution

## Question and fixed evidence

The fixed final-update-1998 screen has much smaller T30 GPP error under daily
Teacher reset than under free recurrence (RMSE `0.10116` versus `0.87144`
gC/m2/day; aggregate bias `-0.179%` versus `-31.581%`). This diagnostic asks
which day-start recurrent state differences mediate that same-day GPP gap. It
does not change the model, loss, parameters, forcing, retained processes,
selected windows, or training method.

The sixteen accepted screen windows and their 296 days remain fixed. Free
day-start continuous state comes from the saved canonical state archives;
Teacher day-start state and forcing come from the already prepared window.
The earlier screen found zero discrete-state mismatches in all sixteen
windows. The executable must also verify that each prepared previous Teacher
endpoint discrete state equals the following prepared day start before using
that common discrete carry. No Teacher or free trajectory is rerun.

## Interventions

The six direct dynamic inputs named by
`target_oriented_structured_gpp.py:82-88` are tested separately:
`assim_param`, `temp_growth`, `humrel`, `qsintveg`, `control_salinity`, and
`control_inudate`. Disjoint indirect groups cover canopy cover/LAI and leaf
state, canopy organic-carbon `interception_storage` alone, remaining
hydrology, energy plus thermosoil, `DOC` alone, litter inventories, litter
quality/fuel metadata, remaining surface/driver/finalize state, and remaining
slow-process state. These groups were fixed across all windows; no case was
selected from the archived drift magnitudes. Runtime construction must cover
every continuous Markov coordinate exactly once. The output records every
leaf's contract slice and audited source reference.

For every archived day and identical forcing, one compiled one-day operator
evaluates full free state, full Teacher state, Teacher-group into free state
(rescue/sufficiency), and free-group into Teacher state (damage/necessity).
Hybrid outputs never become a later-day carry. Thus the contrast estimates
same-day mediation by the chosen state group; it does not identify a unique
upstream process, a multiday total effect, or behavior under a deployable
intervention policy.

## Measurements and decision rule

GPP is primary; NPP, maintenance respiration, growth respiration, and
heterotrophic respiration are retained as context. Every arm reports daily
curves, defined masks, nonfinite counts, physical hard counts, RMSE, mean
bias, and aggregate bias. Full-free and full-Teacher GPP must reproduce the
saved attribution within its declared replay discrepancy.

Control comparisons preserve all measured results before classification.
They reuse the previous replay allclose tolerances (atol1e-8, rtol1e-6);
the free control also allows its recorded raw-GPP replay discrepancy reduced
by the actual PFT-fraction L1 bound. A mismatch invalidates the confirmed-owner
claim, but does not abort or erase other window observations. Hybrid physical,
mask and discrete counts remain visible alongside continuous effect sizes;
material effects before validity filtering are also reported.

For each group, both directions use the positive free-minus-Teacher-reset GPP
MSE gap as denominator:

```text
rescue fraction = (free MSE - rescue MSE) / gap
damage fraction = (damage MSE - Teacher-reset MSE) / gap
```

A primary T30 owner candidate requires both aggregate fractions to be at
least `10%` and positive rescue and damage in at least six of eight T30
windows. T7 is corroborative and is reported separately because the T7 and
T30 panels contain different selected cases. Failure of every singleton group
does not exonerate recurrence; it motivates interaction tests using these
same archived carries. No threshold is a model-acceptance or architecture
selection gate.

Execution is forward-only on a free `gln01` GPU with no gradient, optimizer,
parameter update, checkpoint write, new Teacher data, or paid Slurm job.
