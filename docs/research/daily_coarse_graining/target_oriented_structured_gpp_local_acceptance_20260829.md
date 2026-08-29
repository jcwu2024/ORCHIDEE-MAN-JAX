# Target-Oriented Structured GPP Local Acceptance

Date: 2026-08-29

Status: accepted local implementation; real-v6 zero-update admission pending

Base commit: `a86f256c9f6a7b628c2915510ab0b82f8f73b068`

Architecture contract:
`manifests/coarse_graining/target_oriented_structured_gpp_v1.json`

Canonical contract SHA256:
`ce211fa7c6f5a09bd900e02e31a864b61fb32f6b06ee96d4262402c513d620ea`

## Decision

The user-approved structured GPP specialist is implemented at the stable
daily boundary. It preserves the coordinate-aligned Attention parent and the
accepted distribution-aware maintenance adapter, removes only the old
`daily.fast_gpp` scalar projection, and installs a fresh day-local
Transformer over all 45 native forcing field/time tokens.

The specialist has PFT-shared potential, environmental-limitation, and
bounded-correction branches. The correction factor remains in `[0.25, 4]`.
The final physical GPP route uses the existing explicit `vcmax25 / 100`
factor. It does not multiply `pft_fraction` inside GPP and does not expose
maintenance, allocation, or residence-time parameters to the GPP encoder.

The process head is available through
`build_target_oriented_structured_gpp_head(...)` and remains compatible with
the canonical
`bind_daily_operator(input_assembler=canonical_assembler,
process_head=candidate_head,
retained_tail=canonical_retained_tail_adapter)` boundary.

## Parameter Ownership

The accepted counts are exact:

- distribution-aware parent: `93,504,933`;
- removed old GPP projection: `769`;
- new structured GPP specialist: `12,879,363`;
- combined candidate: `106,383,527`;
- first-stage trainable parameters, when separately authorized: `12,879,363`.

The old GPP projection is an empty tree in the candidate. Parent and
maintenance-adapter weights remain present and are not duplicated.

## Local Evidence

The focused module is
`tests/unit/test_target_oriented_structured_gpp.py`. Its five tests passed in
three bounded local invocations:

1. exact contract and parameter tree: `1 passed in 19.78s`;
2. eager/JIT forward, complete loss, parent isolation, and specialist
   gradients: `1 passed in 214.07s`;
3. parameter routing, PFT behavior, canonical one-day budget, and inference
   graph source audit: `3 passed in 205.71s`.

The evidence establishes:

- the frozen contract hash and all parameter counts are exact;
- all 45 native forcing coordinates are represented as distinct field/time
  tokens before two day-local self-attention blocks;
- eager and JIT outputs and the complete mechanistic loss are finite;
- every specialist-gradient leaf is finite;
- potential, limitation, correction, and all three final projections have
  nonzero gradients;
- GPP has a nonzero gradient to active-vegetation `vcmax25`;
- GPP gradients to all three maintenance slopes, `alloc_min`, and
  `residence_time` are exactly zero;
- every non-GPP learned output is bitwise equal to the accepted
  distribution-aware parent;
- bare soil and inactive/zero-fraction PFT slots produce exact-zero GPP;
- compact-PFT permutation equivariance passes;
- one complete canonical daily transition passes conservative validation and
  water/carbon residuals are within `2e-12`;
- candidate inference contains no half-hour reconstruction and no
  `lax.scan`.

Ruff, Python compilation, package import, and `git diff --check` also pass for
the implementation and focused test files.

## Scope Boundary

This result accepts local implementation only. It contains no optimizer
update, checkpoint, Teacher-data regeneration, historical-v5 join,
validation read, 7/30-day science result, 365-day rollout, Gate D2 evidence,
sealed-test read, or product acceptance.

The next gate is one free-`gln01`, real-v6, zero-update admission. It must
exercise the frozen update-250 distribution-aware parent checkpoint, the
existing v6 train/train boundary, the full specialist gradient, parent
preservation, canonical single-day state/budgets, and the absence of a
subdaily scan. A separate frozen contract is required before any optimizer
update or paid job.
