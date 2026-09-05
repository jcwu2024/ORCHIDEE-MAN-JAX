# Coupled GPP Attention Head And Token Attribution v1

Date: 2026-09-05

Status: completed; the accepted result is recorded in
`target_oriented_coupled_gpp_attention_head_token_attribution_result_20260905.md`

## Question

Does block-0 attention head 7 causally mediate the sampled
`11/16 -> 12/16` Day-5 GPP jump, and is its saturated selection of the DOC
state-summary token necessary for that jump?

This is a zero-training localization inside the accepted model. It does not
change the architecture, optimizer, parameters, data, or product boundary.

## Frozen Parent And Sample

- accepted parent report:
  `outputs/diagnostics/target-oriented-coupled-gpp-potential-internal-56ec8f2.json`;
- parent SHA256:
  `f781e4a92eeac1dd9bc0f986a55fa83478d18d8c20025eb12a2d74067207e0a3`;
- artifact update `550`, runtime year `1976`, seven-day window index `3`,
  Day-5 index `4`, vegetation PFT index `1`;
- lower and upper potential-block trees fixed at `11/16` and `12/16` of the
  already reconstructed rejected displacement;
- branch input, bank values, non-potential parameters, state, forcing,
  limitation, correction, output normalization, and VCMAX routing fixed at
  the accepted control values.

The executable width is exactly `384`, split into `8` attention heads of
`48` output coordinates each. In a Q or K linear projection, head `h` owns
kernel output columns `[48h:48(h+1)]` and the identical bias slice. Kernel and
bias are always moved together unless a result is explicitly labeled as a
kernel-versus-bias measurement; no such partial intervention is part of this
contract.

## Matched Head Controls

For every block-0 head `h = 0, ..., 7`, evaluate both:

1. sufficiency: insert that head's upper Q and K output slices into the
   otherwise lower potential-block tree;
2. necessity: replace that head's upper Q and K output slices with their
   lower values in the otherwise upper potential-block tree.

The eight non-overlapping slices must exactly cover every output coordinate
of both block-0 Q and K kernels and biases. Report per-head Q and K parameter
delta norms and signed effects relative to the complete lower-to-upper GPP
jump. Absolute sufficiency or necessity capture of `0.10` is descriptive
materiality, not an engineering pass condition.

For head 7, repeat the matched lower-insertion and upper-removal controls for
Q-only, K-only, and paired Q+K slices. This crossed panel distinguishes a
paired head mechanism from the component-level redundancy already seen when
all eight Q or K heads were moved together.

## Logit And Token Controls

At the unmodified lower and upper endpoints, record float32 block-0/head-7
logits, softmax weights, and ranks for:

- bank index `153`, which must resolve from the canonical bank layout to
  `slowproc_stomate_previous_step_state.DOC`;
- bank index `205`, which must resolve to
  `slowproc_stomate_previous_step_state.litterpart`.

Also reproduce head 7's accepted winning-index change `205 -> 153` before
interpreting interventions.

Apply the same six token-removal scenarios to both lower and upper endpoints:

- remove DOC, litterpart, or both from the selected PFT's complete potential
  bank mask, affecting all potential heads and blocks;
- remove DOC, litterpart, or both only from block 0, head 7, selected PFT,
  while every other head and block retains the original mask.

For every scenario report the lower shift, upper removal capture, residual
lower-to-upper effect, and the post-intervention head-7 winner. A token is
descriptively necessary when its upper removal captures at least `0.10` of
the original signed jump. Masking is an attribution intervention, not a
candidate inference rule.

## Engineering Acceptance

The report is accepted only if all of the following hold:

- parent identity, schema, accepted status, and engineering gate reproduce;
- the historical direction, runtime year, control trajectory, target sample,
  lower/upper GPP values, and lower/upper float32 shadow potentials reproduce;
- the non-potential GPP multiplier is equal at the two endpoints within the
  existing factorization tolerance;
- width, head count, head width, projection shapes, and exact non-overlapping
  Q/K head coverage match this contract;
- DOC and litterpart resolve to indices `153` and `205`, are valid at the
  target endpoint, and every intervention removes exactly its declared mask
  entries and no others;
- all baseline, head-control, crossed-control, and token-control values are
  finite, and all reconstructed potentials and GPP values are nonnegative.

Scientific attribution is not an engineering assertion. A passed run may
confirm head-7/DOC mediation, identify another head or distributed mechanism,
or reject token necessity. Any engineering failure is inconclusive and must
be inspected before rerun.

## Allowed Execution And Decision Boundary

Local unit tests and one bounded zero-training execution on free `gln01` are
allowed. The implementation may add diagnostic-only head masks to the traced
shadow branch, but it must leave the production forward unchanged when no
diagnostic mask is supplied.

No architecture edit, optimizer, parameter update, checkpoint, recovery,
Teacher regeneration, historical v5 read, validation read, sealed-test read,
half-hour forcing reconstruction, 48-step scan, paid execution, 365-day
rollout, Gate D2, or product acceptance is allowed. The result must be
discussed with the user before any network architecture is changed.
