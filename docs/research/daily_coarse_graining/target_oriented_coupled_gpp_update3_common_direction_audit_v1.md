# Coupled GPP Update-3 Common-Direction Audit V1

## Question

The fresh-moment admission proves that reset AdamW repairs scheduled update 2
but is not a sufficient repeated-training rule. At the in-memory state after
that accepted proposal, fixed-weight update 3 improves the weighted objective,
state, and 30-day loss while regressing direct GPP and 7-day science. Its raw
weighted gradient already has negative cosine with both horizon gradients.

This audit asks whether the same four update-3 source gradients contain a
first-order direction and bounded parameter step that decreases all four
sources. It changes neither the coupled-GPP Attention architecture nor the
scientific labels, data schedule, or physical transition.

The frozen machine contract is
`manifests/coarse_graining/target_oriented_coupled_gpp_update3_common_direction_audit_v1.json`,
canonical SHA256
`956940a1b9cc159fb732980b5bb0f090fc9664adebcf27b22719ec310855cf09`.

## State Reconstruction

The runner loads the unchanged update-1 recovery from paid job `14903328` and
reconstructs fresh-moment update 2 on artifacts
`326,935,542,913,897,836,887,863`. That proposal must remain scientifically
accepted. Its parameters continue only in memory.

At the reconstructed state, the runner recomputes fixed-weight update 3 on
artifacts `968,750,987,728,953,253,646,937`. The predecessor scientific
rejection must remain unchanged. Numeric differences from the predecessor
report are recorded but are not fatal tolerances.

## Common Direction

The four independent gradients are direct GPP, reachable one-day state,
7-day component-resolved science, and 30-day component-resolved science. The
runner measures their `4x4` float64 Gram matrix and uses the repository's
tested minimum-norm simplex solver over unit gradients. This is the standard
MGDA construction specialized to four objectives.

The selected direction must have cosine at least `1e-4` with every source
gradient. Two realizations are evaluated:

- direct: subtract the normalized common gradient with a fixed global
  parameter-step norm;
- fresh AdamW comparator: pass the same common gradient through zero first and
  second moments, bias-correction step one, learning rate `2e-6`, and the
  existing clip threshold.

The reference global step norm is the observed fixed-weight update-3 AdamW
displacement, `0.007758973351050004`. Each realization is evaluated at scales
`1, 1/2, 1/4, 1/8, 1/16`. This compares finite responses at matched local
parameter distances without an open-ended step search.

## Gates And Retention

A scale passes only if all four source losses are non-increasing within
relative tolerance `1e-6`, the existing weighted objective decreases
strictly, and every one-day, 7-day, and 30-day physical hard count is zero.
Direct realization is preferred because it preserves the diagnosed geometric
direction without coordinate-wise optimizer rotation. Fresh AdamW can be
selected only if direct has no admissible scale and the AdamW displacement
itself retains positive cosine with all four source gradients.

Every proposal is discarded. The parent recovery and source checkpoint remain
unchanged, no checkpoint is written, no fresh validation value is read, and
Teacher v6 is not regenerated. The runner is authorized only on free
`gln01`; no paid partition is open.

## Decision Boundary

If a common finite step passes, freeze a separate three-to-five-update free
smoke that recomputes the common direction after every accepted step. Do not
infer repeated-training stability from this single audit.

If no first-order common direction exists, stop optimizer experiments and
redesign the objective hierarchy: long-horizon science becomes primary while
direct GPP and one-day state become bounded protection constraints. If a
common direction exists but no finite scale passes, attribute finite
curvature before changing the architecture or training.

## Local Qualification

The frozen loader, exclusive classification, minimum-norm solver, runner
boundary, launcher, and related coupled-GPP/state paths pass 43 focused and
regression tests. Ruff, Python compilation, Bash syntax, and whitespace checks
pass. The free real-v6 execution remains the active gate.
