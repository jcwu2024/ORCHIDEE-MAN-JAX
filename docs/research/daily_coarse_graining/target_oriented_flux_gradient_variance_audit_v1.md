# Target-Oriented Flux Gradient Variance Audit V1

Status: frozen zero-update diagnostic; local implementation and qualification
accepted; free-GPU execution is the active operation.

## Question

The update-1000 checkpoint learned slow-carbon stocks but did not learn GPP,
NPP, and respiration to the absolute 7/30-day thresholds. Each optimizer
update used only four 30-day windows. This audit asks whether those four
windows produce gradients whose directions vary enough across batches to
block flux learning.

It also tests a separate objective-geometry concern. The frozen long-horizon
training basis contains GPP, total autotrophic respiration, and heterotrophic
respiration. NPP is derived, while maintenance and growth respiration are
optimized only through their sum at 7/30 days. Consequently a stable batch
gradient would not by itself prove that the current loss identifies the
required component fluxes.

## Frozen State And Data

- checkpoint: exact update-1000 stage state, SHA256
  `a566f9d95726984dc03cbf1f1a183dd0d1eae9e3479ce5055c64e571ef6a16c7`;
- prepared data: accepted v6 train/train update-251-to-1000 manifest;
- selected updates: `251, 358, 465, 572, 679, 786, 893, 1000`;
- windows: the four 30-day windows from each selected update, 32 total;
- flux keys: GPP, maintenance respiration, growth respiration, total
  autotrophic respiration, and derived NPP;
- no Teacher regeneration, v5 use, validation tuning, or sealed-test read.

The selected updates are evenly spaced across the completed continuation and
were fixed before reading their gradient results. Each base estimate contains
four distinct point-years. Adjacent base estimates form four disjoint
8-window estimates; adjacent pairs form two disjoint 16-window estimates; all
32 windows form the common reference.

## Computation

For each flux key and each four-window group, compute the exact gradient of
`0.5 * (normalized MSE + normalized signed bias squared)` at the frozen
update-1000 parameters. The existing sufficient-statistic linearization makes
the sum of four one-window VJPs equal to the full four-window gradient.

No AdamW transform is applied. The script forms exact gradient Gram matrices
for a complete, non-overlapping parameter partition:

1. base source encoder;
2. shared science backbone;
3. primary path;
4. state path;
5. auxiliary path.

Their Gram matrices sum to the all-parameter Gram matrix. The report includes
gradient norms, pairwise cosines, cosine to the common 32-window direction,
relative variance, signal-to-noise ratio, and cross-key cosine geometry for
4, 8, 16, and 32 windows. Gram accumulation uses float64 and highest-precision
GPU matrix multiplication; an exactly zero gradient family reports null
cosines rather than inventing a direction.

## Decision Rule

A flux key supports a larger effective batch when:

- its four-window median pairwise cosine is below `0.25`;
- its two disjoint 16-window gradients have cosine at least `0.50`;
- the increase is at least `0.25`.

If at least three of the five keys satisfy all three conditions, the action is
`larger_effective_batch_supported`. If at least four keys already have
four-window median cosine at least `0.50`, the action is
`current_four_window_batch_not_primary_blocker`. Other outcomes are
`mixed_batch_and_objective_geometry`.

Cross-key gradients are reported separately. Maintenance-versus-growth or
NPP-versus-independent-basis cosine below `-0.25` is an objective-geometry
conflict, not evidence that more data alone will solve the problem.

## Boundaries

This audit proposes no optimizer update, writes no checkpoint, and cannot
promote or reject the architecture. Zero physical hard failures, finite Gram
matrices, exact coverage of all `83,755,173` parameters, and a persisted JSON
report are required for a valid diagnosis. The result decides whether to
design a larger-effective-batch continuation or amend the flux objective
before any further paid training.

## Local Qualification

The implementation is
`scripts/hpc/audit_target_oriented_flux_gradient_variance.py`; its free-node
launcher is `scripts/hpc/run_target_oriented_flux_gradient_variance_audit.sh`.
The launcher accepts only `gln01`, binds the accepted checkpoint and prepared
manifest, and contains no scheduler submission or training entry point. The
new focused tests plus the existing prepared-batch regressions pass `14/14`.
Ruff, Python compilation, shell syntax, and `git diff --check` also pass.
