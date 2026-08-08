# First Neural Process-Head Real-v6 Smoke Acceptance

Status: **accepted through the pre-training real-v6 smoke gate**.

Date: 2026-08-08.

## Bound Identity

- branch HEAD: `a9672986ea1c4a41160fac3529d3fd716f7f4dd0` plus the uncommitted
  first-candidate implementation worktree;
- candidate: `axial_process_operator_v1`;
- matched control: `factorized_process_control_v1`;
- candidate contract SHA256:
  `59bd5da7ea7eef78e151aef5339167ed640a12c7700d74ef62e744049a8218e6`;
- accepted v6 manifest SHA256:
  `5fc2c0691bfa53252f875f7646f3c1a5bea203c2759a6e947b26bb51dc13aa99`;
- accepted v6 acceptance SHA256:
  `8784ed42dc64ef77c9642f613eb2e5940635edc6e744a1af4ea8634293baa8e3`;
- real-v6 smoke report SHA256:
  `b289a2c2cdd37a020d076a57f9839a6c45ac70b2576e5e45d7103707edec3d15`;
- real-v6 tiny-fit report SHA256:
  `0b9ed1eb2c7d180002f0d4f88b4e2b86c7e2a1b645d0b3e66e3941334d89b912`.

The bounded run used rows 0-15 of train/train landpoint `001.0-071.0`, year
1961, directly on one `gln01` V100. No v6 shard was copied to the workstation,
no Slurm job was submitted, and no v5 data or checkpoint was read.

## Accepted Result

Both arms have exactly `5,464,321` trainable float64 parameters. Candidate
and control loss, forward JVP, and all 194 reverse-gradient leaves were finite
on the real batch. The source graph remains free of reconstructed 48-record
forcing and 48-step state scans inside candidate inference.

The candidate passed the PFT gate with maximum dimensional absolute error
`1.862645149230957e-9` and maximum `atol=1e-10, rtol=1e-12` tolerance ratio
`0.0019773556205130043`. The remaining dimensional error is output-scale ULP
noise. Inactive-slot isolation error was exactly zero.

The one-day conservative construction passed before the canonical retained
tail. Water and carbon residuals were `1.287858708050814e-14` and
`2.492103745588281e-14`; both inventories remained nonnegative. The output
registry contains 101 named fields, 40 supervised labels, and 38,530 elements
per sample.

The accepted bounded-target loss policy gives initial candidate/control macro
losses `1.9784446349791678` and `0.9960949470028091`; the largest label and
field MSE values are O(1). The earlier O(`1e19`) diagnostic was caused by
dividing already normalized unit-interval and signed-unit targets by tiny
empirical standard deviations. Those targets now use their declared unit
range as loss scale. Nonnegative decoders use a softplus inverse-mean baseline
so initialization remains near the frozen train mean without losing strict
positivity.

The predeclared 100-update same-batch tiny-fit passed for both arms without
writing a checkpoint. Candidate loss fell to `0.3483770761454735`, an
`82.39%` reduction; control loss fell to `0.3200109958647023`, a `67.87%`
reduction. This proves bounded graph learnability only and is not model
selection evidence.

## Corrections Made During Admission

The accepted semantic layouts are water `(49, 142)`, carbon `(127, 1222)`,
and thermal `(56, 466)`. The frozen `carbon_32l` PFT axis is 2. Candidate
self-attention now covers each scientific domain as one bounded domain rather
than fixed 64-token chunks; there is still no cross-domain all-token
attention.

All PFT-axis normalization moments are shared across active PFT slots. The
earlier slot-specific smoke moments encoded PFT identity and produced a false
`1.47e7` permutation failure despite shared network weights. This normalization
rule is now part of the pre-training candidate contract.

## Native Retained-tail Follow-up

The construction-time forcing blocker was closed after the real-v6 smoke.
The retained adapter now evaluates the source-equivalent discrete linear
forcing sums directly over native intervals and supplies only the six
forcing-owned daily/final fields plus salinity and tide height. It does not
materialize 48 half-hour records.

The existing cold day 2, ordinary day 3, and 1962 restart day 1 Oracle matrix
passes complete canonical next-state, discrete-state, 26 modelout-field, and
4 modelout-leaf comparisons. Maximum canonical state error is
`3.9335201762469296e-13`. The staged graph contains only length-5 and
length-14 scans. The refreshed comparison SHA256 is
`f953fcecc3819d049f4fcc3c69caee8297f6cf3cb110bbc380b16594f170a3d0`.

## Remaining Pre-training Blockers

This acceptance does not authorize training.

1. The accepted v6 `training_statistics.npz` contains base arrays but no
   train/train moments for the 66 typed process fields. The 16-row moments in
   this smoke are explicitly diagnostic-only. Formal typed moments must be
   fitted over train/train rows, PFT-shared where applicable, frozen, and
   hash-bound before optimization.
The declared candidate/control tiny-fit and cold/ordinary/restart retained-tail
matrix are accepted as described above. Formal optimization and 7/30-day
rollout remain blocked only by the statistics artifact in item 1.

No 365-day, sealed-test, Gate D2, or product-acceptance work is authorized by
this report.
