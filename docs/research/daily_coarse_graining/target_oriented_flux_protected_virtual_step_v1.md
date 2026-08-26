# Target-Oriented Protected Flux Virtual Step v1

Date: 2026-08-26

Status: frozen before zero-update execution

Machine contract:
`manifests/coarse_graining/target_oriented_flux_protected_virtual_step_v1.json`

Contract SHA256:
`0781cd758a3c30c9558801beae451301cb66ad794a5ad8aaf47d8f3473a9053b`

## Question

At the exact update-1000 Attention parameters, does the amended independent
flux objective have one finite parameter direction that improves the
unresolved 7/30-day productivity and respiration targets across eight
materially different train/train groups without sacrificing the already
successful one-day state and stock behavior?

This is a local mechanism test. It is not training, architecture selection,
or promotion.

## Candidate Mechanism

Construct 16 gradients:

```text
8 fixed train/train groups x (7-day component flux, 30-day component flux)
```

Normalize each gradient to unit L2 norm, then solve the minimum-norm simplex
problem over the 16 unit gradients. This prevents update 465's 7-day norm of
`1713.736` from dominating gradients whose median norm is `12.506`.

The direction is admitted only if its realized float32 parameter-tree
implementation has cosine at least `0.001` with every contributing gradient.
The audit verifies the realized direction with a second bounded float64
chunked Gram calculation; it does not rely only on the ideal host solver.

No state or stock gradient participates in the simplex. Those objectives
already pass the formal screen and are protection constraints.

## Matched Reference

The matched reference is the same update-1000 checkpoint with parameter step
norm exactly zero. Every proposed step uses the same model, same 83,755,173
parameters, same prepared train/train groups, same component-resolved basis,
and same forward evaluator.

There is no second neural model and no weak control architecture. This gate
asks whether a direction is useful relative to its own unchanged parent.

## Frozen Step Grid

Apply direct global parameter L2 norms:

```text
0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1
```

The upper bound is close to the last-50-update AdamW parameter-step median
`0.10954`, while the logarithmic lower range tests the local trust region.
Every grid point is evaluated; the grid is not extended after seeing results.

The parameter delta is exactly the negative unit common direction times the
declared norm. AdamW, momentum, adaptive coordinate scaling, weight decay,
and clipping are absent.

## Flux Gates

For both 7 and 30 days, evaluate GPP, maintenance respiration, growth
respiration, autotrophic respiration, NPP, NEP, and NEE in every group.
Heterotrophic respiration remains an exact retained-path guard.

A step must satisfy:

- no independent optimized key in any group increases by more than `1e-6`;
- no derived required key in any group increases by more than `1e-6`;
- every optimized and derived key has median relative decrease at least
  `1e-4` at both horizons;
- every parent-passing absolute acceptance key remains passing.

These rules prevent an improved average GPP/respiration macro from hiding a
regressing NPP or one opposing group.

## Protection Gates

- every one-day state/process group may increase by at most `1e-4` and its
  median may not increase;
- every reported stock key at 7 and 30 days may increase by at most `1%` in
  any group and its median may not increase;
- a parent-failing stock RMSE may increase by at most `1%`;
- all physical hard-failure and nonfinite counts must be zero.

Select the largest grid norm satisfying all gates.

## Data And Execution Boundary

- exact update-1000 Attention checkpoint only;
- v6 train/train prepared updates
  `251, 358, 465, 572, 679, 786, 893, 1000`;
- validation manifest and panel identity may be read for checkpoint identity,
  but validation artifact values are not loaded or evaluated;
- no Teacher regeneration, historical v5, sealed test, optimizer update,
  retained proposal, checkpoint, or recovery;
- free `gln01` only; no paid Slurm task is authorized;
- no 365-day rollout, Gate D2, or product work.

The runner is
`scripts/hpc/audit_target_oriented_flux_protected_virtual_step.py`; the
launcher is
`scripts/hpc/run_target_oriented_flux_protected_virtual_step.sh`.

## Decision Rule

If no common direction exists or no grid point passes, close direct
realization of the amended raw loss and redesign the robust loss or sampling
unit. Do not shrink below the frozen grid, enlarge the batch again, or change
the Attention architecture on this evidence.

If a protected step passes, design one bounded recalibrated flux-update smoke.
The virtual proposal itself is still discarded and does not authorize paid
training.
