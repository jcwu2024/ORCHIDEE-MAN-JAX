# Coupled GPP Fresh-Moment Admission V1

## Decision

The next bounded experiment keeps the accepted coupled-GPP architecture,
four-source objective, `2e-6` learning rate, clip threshold, prepared
train/train schedule, and finite-step gate unchanged. It changes only AdamW
state ownership: every macro update starts with zero first and second moments
and bias-correction step one, while accepted neural parameters continue to the
next macro update.

This follows the update-2 geometry result. The inherited AdamW displacement
was an ascent direction for direct GPP and failed every tested scale. The
zero-moment displacement was a common descent direction and its full step
improved direct GPP, reachable state, 7-day science, 30-day science, and the
weighted objective. The admission tests whether that mechanism survives one
additional parameter update; it is not a general claim that AdamW moments are
always harmful.

The frozen machine-readable contract is
`manifests/coarse_graining/target_oriented_coupled_gpp_fresh_moment_admission_v1.json`,
canonical SHA256
`a8c362fefb8512cdfe5204dccb2fba4d585f7024366ebe64bf9e5c5a031eda11`.

## Exact Sequence

1. Load the retained update-1 parameters from paid job `14903328` without
   modifying its recovery file.
2. Build the scheduled update-2 macro batch and initialize fresh AdamW
   moments. The optimizer-internal step must be one although the scientific
   update number is two.
3. Require direct GPP, reachable one-day state, 7-day science, and 30-day
   science all to be non-increasing within relative tolerance `1e-6`; require
   the weighted objective to decrease strictly.
4. If update 2 passes, retain its parameters only in memory, build the
   scheduled update-3 macro batch, and initialize fresh AdamW moments again.
5. Apply the same gate to update 3. Stop immediately on a scientific rejection
   and still write the JSON result.
6. Discard every proposed parameter and optimizer state after reporting.

Both proposals must have complete finite gradients, change both trainable
families, preserve fixed parameters and the structured-GPP source checkpoint,
and have zero pre/post physical hard-failure counts. The runner reads no fresh
validation or sealed-test values, regenerates no Teacher data, reconstructs no
48-step forcing, and performs no subdaily candidate state scan.

## Acceptance Boundary

A pass means fresh moments remain viable for two consecutive scheduled macro
updates. It authorizes freezing a separate paid updates-2-through-8 contract;
it does not itself authorize or retain training, evaluate the 96-window
validation panel, or promote a checkpoint.

A rejection means the single successful reset update from the geometry audit
does not transfer to the next scheduled parameter state under this unchanged
objective. Paid continuation then remains closed pending review of the failed
source and finite-step geometry.

## Local Qualification

The contract loader, two-update decision, free-node launcher, reused proposal
path, optimizer-step accounting, and related coupled-GPP boundaries pass 31
unit tests. Ruff, Python compilation, shell syntax, and whitespace checks pass.
The only remaining gate is the no-checkpoint two-update execution on free
`gln01`.
