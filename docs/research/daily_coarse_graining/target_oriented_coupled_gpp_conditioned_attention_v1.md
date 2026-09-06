# Coupled GPP Conditioned Attention v1

Date: 2026-09-06

Status: user-approved architecture; frozen for local implementation and one
free zero-update admission, while optimizer updates remain unauthorized

## Evidence And Decision

The accepted causal audit localized the sampled Day-5 GPP jump to
`structured_gpp.potential_blocks[0].attention`, specifically the paired Q/K
slice of head 7 and its selection of the DOC state-summary token. The projected
key bank has RMS about `7.17e7`; changing only `1.58e-6` in the head's Q/K
parameter slices moves its maximum logit from `8.1263` to `7796.5381` and
collapses the softmax weight onto DOC. Float64 retains the jump, so this is a
Q/K conditioning defect rather than ordinary float32 roundoff.

This evidence supports a narrow architecture correction. It does not support
discarding the Transformer, hard-coding head 7 or DOC, increasing parameter
count, or changing the Value and output paths.

## Candidate Mechanism

The candidate is
`target_oriented_coupled_gpp_conditioned_attention_v1`. It changes only the
two attention layers in `structured_gpp.potential_blocks`. After the inherited
linear projections, each query and key vector is normalized independently
inside each 48-coordinate head:

```text
q_hat = q / max(||q||_2, epsilon)
k_hat = k / max(||k||_2, epsilon)
score[b,h,q,k] = tau[b,h] * clip(dot(q_hat,k_hat), -1, 1)
tau[b,h] = 12 * sigmoid(raw_tau[b,h])
```

`epsilon` is `1e-6`. There are two blocks and eight heads, so the candidate
adds only 16 scalar temperature parameters. Every temperature starts at
`sqrt(48) = 6.928203...`, the cosine-space equivalent of ordinary scaled
dot-product attention when projected Q and K coordinates have unit variance.
The upper bound `12` permits stronger-than-standard selection while reducing
the observed unbounded score of `7796.5` to a mathematically bounded score.

The same rule is applied to every potential head. Head 7 and DOC are never
named by production inference. The forcing encoder, limitation and correction
branches, carbon-feedback attention, Q/K/V/output linear projections,
attention masks, residual blocks, FFNs, and positive GPP factorization remain
unchanged.

## Output And Physical Boundary

The candidate still newly controls only:

```text
daily.fast_gpp[compact_pft]
```

It remains attached through the existing process-head object and therefore
enters only through:

```text
bind_daily_operator(
    input_assembler=canonical_assembler,
    process_head=candidate_head,
    retained_tail=canonical_retained_tail_adapter,
)
```

It does not predict next-day carbon state or add neural memory. The
conservative updater and retained exact STOMATE tail continue to carry GPP
into the next day's carbon state. Non-GPP outputs, explicit `vcmax25` routing,
inactive/bare-PFT rules, and forbidden maintenance/allocation/mortality
parameter routes are unchanged. Inference reconstructs no half-hour forcing
and performs no subdaily state scan.

## Matched Mechanism Reference

The matched reference uses the identical parent parameters, Q/K
normalization, Value path, masks, output factorization, data, objective,
optimizer, and seed. Its only difference is that every temperature remains
fixed at `sqrt(48)` instead of being learned. Candidate and reference must
therefore have the same forward result at initialization.

This reference answers a later, narrow question: whether learning different
selection strengths by block and head adds value beyond conditioning Q/K. It
is not a promotion baseline and is not assumed to be a good model. It is not
trained when the candidate fails the absolute 7/30-day science gates, and a
relative win cannot rescue an absolute failure.

## Admission Gates Before Training

Local admission requires exact contract and parameter counts, unchanged
legacy-parent output when conditioning is absent, matched initial
candidate/reference output, finite eager and JIT forward/loss/gradients,
nonzero gradient reaching at least one temperature in each potential block,
finite nonzero inherited Q/K/V/output gradients, absolute conditioned logits
no larger than `12.000001`, bit-identical non-GPP output, inactive-PFT
isolation, PFT permutation equivariance, and one complete daily transition
with water and carbon residuals at most `2e-12`.

The free real-v6 admission additionally reconstructs the already audited
`11/16 -> 12/16` parameter interval in memory. It must reproduce the parent
endpoints, keep every conditioned value finite and nonnegative, make the
learnable candidate and fixed reference endpoints agree at initialization,
and reduce the conditioned endpoint jump to at most `25%` of the original
`2.102579083` jump. This threshold tests the stated mechanism directly; it is
not a claim about scientific accuracy.

Only after both local and free zero-update gates pass may a separate contract
authorize one discarded optimizer proposal. Small bounded training and an
independent 7/30-day screen follow in that order. Promotion still requires the
existing absolute GPP, respiration, NPP, AGB, BGB, total-live-biomass, and
ecosystem-carbon gates. A passing 7/30-day result is required before 365-day
rollout, long-held-out points, Gate D2, sealed test, or product acceptance.

## Execution Boundary

This contract authorizes local implementation/tests and one bounded
zero-update execution on free `gln01`. It authorizes no optimizer update,
checkpoint write, paid execution, Teacher-v6 regeneration, historical-v5
read, fresh-validation read, sealed-test read, 365-day rollout, Gate D2, or
product acceptance.

The machine contract is
`manifests/coarse_graining/target_oriented_coupled_gpp_conditioned_attention_v1.json`.
