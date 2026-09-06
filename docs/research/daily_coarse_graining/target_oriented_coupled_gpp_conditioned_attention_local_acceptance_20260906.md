# Coupled GPP Conditioned Attention Local Acceptance

Date: 2026-09-06

Status: accepted local zero-training implementation; free real-v6 GPU
admission is next

Base commit: `254133823a26ea281ab4673e7a60ab9edf1c054b`

Architecture contract:
`manifests/coarse_graining/target_oriented_coupled_gpp_conditioned_attention_v1.json`

Canonical contract SHA256:
`b9bdd9dde51a8f6f1de2c21cd6678f5a51232011397501459d614b549ca560d4`

## Implemented Mechanism

The candidate preserves the accepted coupled GPP/carbon-feedback Transformer
and changes only the two attention layers in
`structured_gpp.potential_blocks`. Each projected query and key is normalized
independently inside each 48-coordinate head. Attention scores are the clipped
cosine similarity multiplied by one bounded temperature per block and head:

```text
score = tau[block, head] * clip(dot(q_hat, k_hat), -1, 1)
tau = 12 * sigmoid(raw_tau)
```

All 16 temperatures initialize at `sqrt(48)`. The matched mechanism reference
uses the same normalized Q/K path with its temperatures fixed at `sqrt(48)`.
The candidate and reference therefore match at initialization. The reference
is for later mechanism attribution only and is not a promotion baseline.

No production path names head 7 or the DOC token. Value aggregation, output
projection, masks, residuals, FFNs, forcing, limitation, correction, carbon
feedback, GPP factorization, and explicit `vcmax25` routing remain unchanged.

## Boundary And Ownership

The candidate still newly controls only `daily.fast_gpp[compact_pft]` and is
bound through the existing process-head boundary. The conservative updater
and retained exact STOMATE tail are unchanged. It predicts no next-day state,
keeps no neural cross-day hidden state, reconstructs no half-hour forcing, and
performs no subdaily state scan.

The exact parameter counts are:

- coupled GPP parent: `109,464,359`;
- new bounded temperatures: `16`;
- complete candidate: `109,464,375`;
- first-stage trainable structured GPP, carbon feedback, and temperatures:
  `15,960,211`;
- retained parent: `93,504,164`.

## Local Evidence

The combined conditioned-candidate, coupled-parent, and structured-parent
suite passed:

```text
17 passed in 804.14s
```

The result establishes:

- exact contract and parameter ownership;
- finite eager and JIT forward plus complete mechanistic loss;
- finite temperature gradients with nonzero reach into both potential
  blocks;
- finite nonzero inherited potential Q/K/V/output gradients;
- scale-invariant and bounded logits when projected Q/K magnitudes are raised
  by `1e8` to `1e9`;
- initial candidate and fixed-temperature reference agreement;
- bit-identical non-GPP outputs relative to the coupled parent;
- unchanged legacy structured and coupled head behavior;
- inactive-PFT isolation and compact-PFT permutation equivariance;
- a complete canonical `bind_daily_operator(...)` transition with valid
  next-state shape and water/carbon residuals within `2e-12`;
- no half-hour forcing reconstruction, 48-step scan, optimizer, or checkpoint
  path in candidate inference.

Ruff, Python compilation, both launcher shell syntax checks, and
`git diff --check` also pass. The free runner's required CLI options are all
supplied by its launcher and shared execution boundary.

## Evidence Limit

This local result uses bounded synthetic fixtures. It proves implementation,
gradient, interface, and physical-update integrity, but it does not prove that
the known real-v6 `11/16 -> 12/16` GPP jump is removed and it is not a science
or promotion result.

No Teacher-v6 data were regenerated, no historical v5 data were read, no
optimizer update was made, no checkpoint was written, and no fresh
validation, sealed test, 7/30/365-day rollout, Gate D2, or product gate was
opened.

## Next Gate

Run the frozen audit once on free `gln01` against the existing real-v6
train/train artifact and accepted attribution assets. It must:

- reproduce the parent trace and both parent endpoints;
- bound every conditioned logit by `12.000001`;
- reduce the known endpoint jump to at most `25%` of its parent magnitude;
- match the learnable candidate and fixed-temperature reference at
  initialization;
- pass one real-v6 complete forward/loss/gradient transition, state checks,
  and water/carbon budgets;
- perform zero optimizer updates and write no checkpoint.

Only a passing free admission may open a separately frozen discarded
one-update preflight. Paid training remains unauthorized.
