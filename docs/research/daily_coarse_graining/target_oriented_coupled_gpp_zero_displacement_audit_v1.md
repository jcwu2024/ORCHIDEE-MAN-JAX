# Coupled GPP Exact-Zero Displacement Audit V1

## Question

Artifact-aware persistent training retained logical update 2 and rejected
logical update 3. All 40 direct-GPP artifacts, reachable state, 30-day
science, and every physical gate improved at all six tested scales. The only
rejection was scheduled 7-day science. Its relative regression increased from
`0.0508%` at scale `1/2` to `0.5775%` at scale `1/64` instead of approaching
zero with the parameter displacement.

The existing rejected proposal localizes essentially all of that regression
to artifact `550` (runtime year 1976): its 7-day score rises by `6.1243%` at
scale `1/64`, while the other seven scheduled artifacts improve slightly.
Before changing sampling, loss weights, the finite-step gate, or the Attention
architecture, this audit distinguishes an evaluator-baseline inconsistency
from a genuine nonzero-step rollout discontinuity.

## Bounded Evaluation

The audit loads the exact parameter-only `recovery.pkl` retained after logical
update 2. It evaluates only the eight scheduled logical-update-3 artifacts and
only their 7-day pure-forward objective. It recomputes no gradient, applies no
parameter displacement, performs no training, and writes no parameter state.

For every artifact it compares:

1. the gradient-time baseline stored in `rejected-proposal.json`;
2. a new pure-forward evaluation at the exact recovered parameters;
3. the historical pure-forward value at the smallest nonzero scale `1/64`.

The exact-zero value must reproduce the stored baseline within
`rtol=1e-6, atol=1e-10`. A smallest-step mean regression above `0.1%` is
material. At least `90%` of total positive score displacement on one artifact
classifies the jump as localized.

## Decisions

- `gradient_baseline_not_reproduced_by_exact_zero_forward`: compare the
  gradient wrapper's forward result with the pure-forward path on the
  mismatching artifacts. Do not change training policy.
- `localized_nonzero_step_discontinuity`: trace the first differing rollout
  branch/state on the dominant artifact between exact zero and the smallest
  nonzero step. Do not relax the 7-day gate.
- `distributed_nonzero_step_discontinuity`: trace the affected artifacts
  before changing the sampling rule.
- `no_material_nonzero_step_discontinuity`: audit aggregation and report
  construction.

The audit uses existing v6 artifacts only. It does not regenerate Teacher
data, read v5, inspect fresh validation or sealed test, reconstruct 48 forcing
records, or execute a 48-step candidate scan. Paid execution, further
persistent training, architecture changes, 365-day rollout, Gate D2, and
product acceptance remain closed by this diagnostic.

## Implementation

- classification:
  `research/daily_coarse_graining/daily_operator/target_oriented_coupled_gpp_zero_displacement.py`
- runner:
  `scripts/hpc/audit_target_oriented_coupled_gpp_zero_displacement.py`
- free-node wrapper:
  `scripts/hpc/run_target_oriented_coupled_gpp_zero_displacement.sh`
