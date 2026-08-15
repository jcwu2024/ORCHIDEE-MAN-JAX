# Decoder-Leaf Attribution, 2026-08-16

## Motivation

The seven-block attribution shows that the output decoder carries `96.1526%`
of the stopped direction's squared norm. Removing the full decoder destroys
common descent, while removing any other block leaves the `28.72%` finite-step
regression unchanged. The remaining bounded question is whether that behavior
can be assigned to one of the decoder's four parameter leaves.

## Frozen Test

Reconstruct the same stopped state and full failed direction. Report geometry
for:

- `decoder_first.kernel`;
- `decoder_first.bias`;
- `decoder_second.kernel`;
- `decoder_second.bias`.

Form four leave-one-leaf-out projections at the same global norm `0.0005`.
For each, evaluate all 40 relative directional derivatives first. Skip the
finite step if any derivative is nonpositive. A leaf is attributed only if its
exclusion preserves all-40 first-order descent and makes the finite step
nonincreasing across all 40 objectives within `1e-6`.

This test does not authorize freezing a decoder leaf. A positive attribution
only opens a bounded decoder preconditioner audit. If every exclusion fails,
raw Euclidean parameter-space direct steps close and the next design must
bound output/function change instead.

## Boundaries

- four projections, one frozen step norm, no step search;
- accepted v6 data and the same ten train/train groups only;
- free `gln01` only;
- no optimizer update, validation read, checkpoint, recovery, retained
  parameters, architecture change, or promotion.

The machine contract is
`manifests/coarse_graining/successor_decoder_leaf_attribution_v1.json`,
SHA256 `08783a377a80b88656a830efbccea6a3b9994f6a3653ea701cea261d463cf721`.

## Local Admission

The geometry registry covers all four decoder leaves and measures their
fractions against the full direction norm. Leaf exclusion preserves the
existing pytree, and the shared attribution runner binds the accepted
seven-block report before entering decoder mode. The complete related unit
file passes `79` tests; Ruff, Python compilation, Bash syntax, canonical
contract hash, and `git diff --check` pass.

## Result

The free-`gln01` audit completed at commit `39cd8ef`. The report is
`runtime/outputs/training/successor-decoder-leaf-39cd8ef/report.json`, SHA256
`513e43373c9e9d5b786a279953899449ef560d25d74d5db1b29671d9519b6c4d`.
The stopped-state reference reproduces below `2.1e-15`; hard and nonfinite
counts are zero.

Decoder direction geometry is:

- `decoder_second.kernel`: `68.4533%` of full squared direction norm,
  step-delta norm `4.13683e-4`, and delta/parameter norm `0.147323%`;
- `decoder_first.kernel`: `23.9256%`, `2.44569e-4`, and `0.00143373%`;
- `decoder_second.bias`: `3.75344%`, `9.68690e-5`, and `0.645335%`;
- `decoder_first.bias`: `0.0201673%`, `7.10058e-6`, and `0.00224562%`.

Removing either second decoder leaf destroys common descent. Excluding the
second kernel gives minimum relative derivative `-0.195122`; excluding its
bias gives `-0.0810206`. Removing either first decoder leaf preserves
first-order descent, but the finite step still regresses by `28.7203%` or
`28.7222%`. No leaf exclusion is safe.

The classification is `parameter_geometry_not_attributed`, with next action
`close_raw_parameter_steps_and_design_function_space_trust_region`. Runtime is
`1937.30` seconds and peak JAX device bytes are `13,827,079,680`. No optimizer
update, validation read, checkpoint, recovery, or candidate parameters are
produced.

## Decision After Execution

Raw Euclidean parameter-space direct steps are closed at this state. The two
second-decoder leaves are jointly essential to first-order descent, while the
first-decoder leaves do not cause the finite-step failure on their own. Do not
continue parameter exclusions, scalar step subdivision, or direct updates.

Before choosing a function-space metric, run one bounded output-locality
design/audit: compare the full direction's first-order JVP with its exact
finite-step change for the named decoded process outputs on the same ten
groups, then rank the fields responsible for the BGB/NPP loss jump. The
resulting normalized output coordinates, not parameter norm, must define any
subsequent trust region. This remains an optimizer/metric change, not a neural
architecture change.
