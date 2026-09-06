# Coupled GPP Conditioned Attention Persistent Training Local Acceptance

Date: 2026-09-06

Status: accepted local implementation; one free-`gln01` execution is next

Contract:
`manifests/coarse_graining/target_oriented_coupled_gpp_conditioned_attention_persistent_training_v1.json`

Canonical contract SHA256:
`ac65c5be2b4202edad81ad8ad371e47ec8cf39443c4e4eedd2e80ab02e5c9aff`

## Implemented Boundary

The runner starts from the identity-exact accepted update-2 recovery, attaches
the Q/K-conditioned Attention temperatures, writes and reloads an immutable
conditioned start snapshot, and resumes only from an identity-matched atomic
recovery. Fixed parameters and optimizer state are never serialized.

Logical updates 3-5 each recompute 40 individual direct-GPP gradients plus
scheduled state, 7-day, and 30-day gradients. The minimum-norm direction and
the architecture-aware full-step calibration are both recomputed at the
current parameter state. Neither parent step norm, prior conditioned step,
simplex weights, nor an optimizer transform is inherited.

After train-only scale selection, the candidate is evaluated against the
preceding accepted monitor state and the immutable initial monitor state. Both
transfer gates must pass before the candidate can overwrite recovery. The
source order in the runner places the rejection branch before assignment of
candidate parameters to retained state.

## Local Evidence

The complete related suite passed:

```text
57 passed in 374.82s
```

Coverage includes:

- conditioned Attention forward, complete loss, and gradients;
- Q/K scale invariance, bounded logits, masks, PFT permutation, and inactive
  PFT isolation;
- one canonical daily transition and water/carbon budgets;
- the accepted discarded-update contract, gates, paired monitor, and launcher;
- three-update schedule uniqueness and monitor disjointness;
- current-state step recalibration and exclusive terminal classifications;
- immutable start, atomic recovery, update-5 checkpoint, exact round trips,
  and rejection of monitor-coverage gaps;
- source-order proof that monitor evaluation precedes parameter retention;
- the new runner's actual parser branch, including every required inherited
  argument; and
- unchanged artifact-aware persistent, continuity, direction, transfer, and
  geometry helpers.

Ruff lint and format checks, Python compilation, both shell syntax checks,
manifest loading, the actual discarded-result loader against the recovered
real report, CLI `--help`, and `git diff --check` pass.

## Evidence Limit

The orchestration and model boundary are locally accepted, but no persistent
conditioned update has yet run on real Teacher-v6 artifacts. The earlier
discarded step remains the only real-v6 finite-step evidence. This local result
therefore does not claim three-update transfer or scientific accuracy.

No Teacher data were regenerated, no v5 or fresh-validation value was read,
and no optimizer, training recovery, or checkpoint was produced locally.

## Next Gate

Synchronize a clean commit and run the frozen wrapper once on free `gln01`
with a 240-minute timeout. The run may persist only monitor-admitted updates
and may resume from its exact recovery if the process timeout interrupts it.
Paid execution, fresh validation, 365-day rollout, Gate D2, sealed test, and
product acceptance remain closed.
