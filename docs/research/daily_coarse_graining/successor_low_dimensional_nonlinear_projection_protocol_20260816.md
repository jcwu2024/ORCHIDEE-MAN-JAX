# Successor Low-Dimensional Nonlinear Projection Protocol

Date: 2026-08-16
Status: frozen before implementation
Contract: `successor_low_dimensional_nonlinear_projection_v1.json`
Contract SHA256: `db2b60c780e5ec4978e936e88dfe13ed0e00bd23ba997b08fccf7e16ed67d339`

## Question

The single maintenance-gradient projection substantially reduced the failed
proposal's maintenance displacement, but no scalar correction simultaneously
met the maintenance caps, all 40 loss constraints, and the long-horizon effect
gate. This audit asks whether those constraints can be restored by a bounded
nonlinear search in a small, scientifically identified parameter subspace.
It does not change or retrain the Attention network.

## Candidate And Matched Control

The matched control is the exact norm-`0.00025` single-gradient correction
from the completed maintenance projection. It already improves median NPP,
AGB tendency, and BGB tendency, but fails the maintenance, anchor, and some
individual loss constraints. The candidate is the first exact-feasible
iterate produced by the sequential constrained solver below.

The safe microstep remains the source of the maintenance caps. The raw failed
norm-`0.0005` proposal remains a non-promotable reference. All arms reconstruct
the same stopped state and use the same ten train/train groups, 40 point-years,
and 1,200 daily transitions. No validation value is read.

## Frozen Subspace

At the matched-control parameters, construct at most eight ordered directions:

1. the raw common-descent direction;
2. the raw maintenance-RMS correction direction;
3. the relinearized matched-control maintenance-RMS repair gradient;
4. the repair gradient of the matched-control maximum absolute maintenance
   scalar;
5. one severity-weighted aggregate gradient of all positive loss violations;
6. one equal-component NPP/AGB/BGB effect gradient;
7. the two largest still-unrepresented individual loss-violation gradients.

Ordered modified Gram-Schmidt in float64 removes a direction whose residual
norm is below `1e-8` of its original norm. The resulting basis must have four
to eight columns and is then frozen. Every candidate is represented as the
stopped parameters plus coordinates in this basis. No new parameter direction
may be added after the nonlinear iterations begin.

Construction is capped at 103 reverse passes. Maintenance-RMS gradients use
the accepted one-point-year microbatch boundary. This retains the previous
memory fix and never vectorizes four point-years through one reverse pass.

## Sequential Solver

Initialize at the matched-control coordinates. For at most four outer
iterations, recompute exact values and JVP derivatives along every frozen
basis direction for:

- all 40 group-component relative losses;
- the NPP, AGB, and BGB median-effect constraints;
- maintenance RMS;
- all 28,800 signed normalized maintenance-output changes;
- signed progress along the raw direction and total displacement.

A float64 SciPy SLSQP subproblem first performs bounded slack restoration when
the linearization is infeasible, then finds the minimum coordinate correction
when hard linearized feasibility is available. Each coordinate step has
Euclidean radius at most `0.000125`. Only full and half trials are allowed.
There are at most eight exact candidate evaluations, and no trial is retained
unless exact violation merit decreases by the frozen tolerance or all gates
pass. Failure of both trials stops the audit rather than expanding the
subspace or step budget.

This is nonlinear because constraints and their derivatives are recomputed at
every accepted finite iterate. It is not another subdivision of the original
maintenance-gradient direction.

## Acceptance

The first candidate is selected only if exact finite evaluation shows:

- every one of the 40 group-component losses increases by at most `1e-6`;
- median NPP, AGB tendency, and BGB tendency each improve by at least `1e-5`;
- maintenance RMS is at most `0.0005208278724955756` and the maximum absolute
  normalized change is at most `0.005030725887901774`;
- total parameter displacement from the stopped state is between `0.000125`
  and `0.00075`;
- every state, budget, and nonfinite count is zero.

The matched control must reproduce its completed-report values within
`1e-10`. Basis provenance, dropped directions, coordinates, every
linearization, inner-solver status, exact trial, and stop reason are reported.

## Stop And Next Action

No optimizer update, parameter set, recovery state, or checkpoint is written.
No Teacher data are regenerated, no v5 data are used, and no validation or
sealed-test values are evaluated. The only allowed execution is one bounded
free-`gln01` audit; no paid Slurm run is authorized.

If an exact-feasible iterate exists, freeze only its coordinates for the
existing non-test 7/30-day screen. If none exists within four iterations and
eight exact evaluations, close this fixed low-dimensional projection and
return to the training objective and effective data use. Do not enlarge the
network or alter Attention on the basis of this audit alone.
