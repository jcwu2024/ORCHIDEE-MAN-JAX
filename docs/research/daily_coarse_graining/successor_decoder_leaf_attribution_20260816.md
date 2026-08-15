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
