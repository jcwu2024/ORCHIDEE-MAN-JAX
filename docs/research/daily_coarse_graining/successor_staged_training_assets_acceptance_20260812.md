# Successor Staged-Training Assets Acceptance

Status: **accepted training-input policy; staged runner implementation is next**.

## Decision

The matched successor pair is unchanged:

- `source_routed_process_attention_v2` candidate;
- `source_routed_graph_mlp_control_v2` matched control;
- 8,722,689 parameters per arm;
- the canonical `bind_daily_operator(...)` boundary and retained tail.

The rejected 500-update checkpoints remain rejected and neither arm is
promoted. This acceptance freezes a better-controlled training regime; it is
not evidence that either architecture is successful.

The staged-training contract is
`manifests/coarse_graining/successor_training_staged_v3.json`, canonical
SHA256 `53d21d3a03a1502263b333aaba9719b2d42f096672844497b1a213dd54c23c7b`.
The immutable sufficiency-v2 probe contract and its median report are not
rewritten. V3 instead derives one lower-quantile calibration from the 32 raw
gradient measurements already collected under v2.

## Frozen Coefficients

The one-time shared train/train rule is

```text
candidate coefficient = target ratio * anchor gradient norm
                        / component gradient norm
coefficient = empirical 10th percentile with NumPy method="lower"
```

It yields:

| Component | Coefficient | Maximum measured achieved ratio |
| --- | ---: | ---: |
| cumulative flux MSE | 0.088284572329923 | 0.3517957306415747 |
| cumulative flux bias squared | 0.13244689760336273 | 0.5276658879953050 |
| terminal stock MSE | 0.019877774130136576 | 0.4163165418952551 |
| terminal stock bias squared | 0.029825749684563974 | 0.6244350748668159 |

The source probe is the zero-training d3200df report, file SHA256
`c3398c68487db8a8cb0258830f88c1fed0f1f8b24683ef938cbd46befc92cc7d`.
All 30-day state, defined-mask, discrete-state, nonnegative-stock and declared
budget hard-failure counts in that report are zero.

The accepted coefficient asset is under
`runtime/outputs/training/successor-staged-assets-b800db9/`:

- internal contract SHA256:
  `c2833ee2bc66b31842e135b0b85fe48c6f3ee0b9a7650c77b4ae0889a7916b93`;
- file SHA256:
  `a363608c8ee71b25066c008cac7a59f416f944d2685f08bf1e6cfad8ca729963`.

## Frozen Milestone Panel

The validation panel uses only the non-test slices `train/validation`,
`validation/train`, and `validation/validation`. Each slice contains eight
distinct landpoints and two nonoverlapping 30-day windows per point-year.
The same 48 windows are used for both arms and at updates
`500, 1000, 2000, 5000, 10000`.

Direct comparison with the rejected formal 96-window screen finds zero exact
window overlap. The sealed test remains closed.

The accepted panel is in the same runtime directory:

- internal panel SHA256:
  `1ac7490a2c7067bf3f66faea75836d80a9e6fdc62ebcdc8673caa3f445c70350`;
- file SHA256:
  `da21d84761a8c02662b5eed0c1c86765cdb58cdb7e29a7cbbb9e49576eb91635`.

## Execution Evidence

Commit `b800db9` passed 75 related local tests, including the full successor
process-head file, plus Ruff and `git diff --check`. A free `test01` process
then created both assets in 116.4 seconds. An independent process reloaded
both assets, reproduced all four coefficients, counted 48 windows, verified
eight distinct landpoints per slice and zero within-shard overlap, and
confirmed `sealed_test_used=false`.

No network training, checkpoint write, v5 read, Teacher regeneration, 48-step
candidate scan, or paid resource was used.

## Next Gate

Implement and locally validate:

1. staged preparation for up to 10,000 unique train/train point-years;
2. a matched runner that consumes these frozen coefficients;
3. fixed-panel 7/30-day validation at every declared milestone;
4. recovery/checkpoint selection from validation improvement with zero hard
   failures;
5. free-node smoke and resource measurement before any paid submission.

Paid preparation or training still requires a fresh Explore1000 approval
block. The 365-day rollout, long-held-point validation, Gate D2, sealed test,
and product acceptance remain closed.
