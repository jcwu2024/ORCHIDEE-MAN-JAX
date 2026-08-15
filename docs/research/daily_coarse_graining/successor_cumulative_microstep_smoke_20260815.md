# Cumulative Microstep Smoke, 2026-08-15

## Motivation

The repeated active-set smoke reaches a state where a newly recalibrated
direction passes all 40 first-order checks. A norm-`0.0005` finite step also
decreases all 40 losses, but its one-step anchor and BGB median effects are
smaller than `1e-4`. Norms at or above `0.001` cross into a strongly nonlinear
regression region. Smaller one-shot steps therefore cannot solve the missing
effect magnitude.

The next question is whether several safe steps, each with a freshly
recalibrated direction, can form one useful cumulative update without crossing
that region.

## Frozen Mechanism

1. Reconstruct the two accepted norm-`0.002` direct updates exactly from the
   update-1000 Attention parent.
2. Start from the repeated smoke's 13-constraint positive support.
3. Recalibrate the active set before every microstep and check all 40 relative
   directional derivatives.
4. Propose exactly one norm-`0.0005` microstep. Apply it only when all 40
   losses are nonincreasing within `1e-6` and hard/nonfinite counts are zero.
5. After every applied microstep, compare all 40 losses with the common
   pre-block baseline. Stop successfully when all remain nonincreasing and
   each of the four component medians has decreased by at least `1e-4`.
6. Stop after at most four microsteps, or before the first unsafe microstep.

The `1e-4` scientific effect gate is unchanged; only its evaluation boundary
moves from each microstep to the cumulative macro-update. Four is the frozen
maximum because linear extrapolation of the observed anchor response reaches
the gate at about four safe microsteps. This is a bounded mechanism test, not
an open-ended step search.

## Endpoint Gate

Run the established six-artifact non-test 7/30-day evaluation only if the
cumulative macro-effect gate passes. Report directional support separately
from the unchanged `0.5%` promotion-scale threshold. A pass authorizes design
of repeated macro-updates; it does not promote a model.

## Boundaries

- architecture remains `source_routed_process_attention_v2`;
- objectives, coefficients, prepared train/train groups, and v6 data remain
  unchanged;
- free `gln01` only, with no paid job;
- no checkpoint, recovery, or final parameter retention;
- no Teacher regeneration, v5 data, sealed test, 365-day rollout, or Gate D2.

The machine contract is
`manifests/coarse_graining/successor_cumulative_microstep_smoke_v1.json`,
SHA256 `be46303e4abe8794ccd700958786cbaeb7c0e7fdd973bc44ad514341c32c545a`.

## Local Admission

The runner reconstructs both accepted direct updates, verifies the first new
microstep against the stopped-state reference, recalibrates before every
later microstep, and skips 7/30-day evaluation unless the cumulative gate
passes. The classification tests cover successful early completion and
bounded failure without validation reads. The complete related unit file
passes `74` tests; Ruff, Python compilation, Bash syntax, canonical contract
hash, and `git diff --check` pass.
