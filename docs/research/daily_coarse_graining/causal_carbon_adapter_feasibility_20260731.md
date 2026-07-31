# Causal Carbon Adapter Real-Shard Feasibility

Date: 2026-07-31

## Decision

Pass the Experiment C train-only feasibility gate. The causal-carbon adapter
is technically ready for formal train-only coefficient calibration. This
result does not show that the candidate improves validation rollout quality
and does not authorize use of the sealed test split.

## Execution

- research commit:
  `79258fd55d300989c86e63c69cafc6128a0927a5`;
- host: free `gln01` GPU test host;
- device: one V100 32 GB, selected as visible GPU 1;
- data: landpoint `001.0-071.0`, year 1961, train/train split;
- updates per arm: 8;
- horizon sequence: `1/3/7/1/3/7/1/3`;
- report SHA256:
  `ca8cacdccb42f20b94b5cd64796118cb170f7fbc7813c036cbc6b1edc8aa7fa8`;
- sealed test used: false.

The immutable server root is:

```text
/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/training/causal-carbon-adapter-feasibility-79258fd
```

A local copy of the report and log is under:

```text
outputs/research/daily_coarse_graining/causal_adapter_feasibility_79258fd
```

## Gate Result

All eight control and candidate updates were applied. Candidate gradient norms
ranged from `0.3823` to `2.8397`; every loss and gradient remained finite.

Exact hard totals:

| Constraint | Count |
|---|---:|
| unexpected defined-status mismatch | 0 |
| declared dynamic-status mismatch | 0 |
| discrete-state mismatch | 0 |
| nonfinite defined value | 0 |
| negative source-constrained carbon stock | 0 |

The 1,954,041-parameter parent remained outside the optimizer. Only 76,877
adapter parameters were optimizer-owned. At initialization and after every
candidate update:

- all 2,714 protected fast-day columns were bit-exact with the parent;
- the dynamic undefined head was bit-exact with the parent;
- maximum protected-column absolute difference was exactly `0.0`.

The final feasibility checkpoints are:

```text
control:
849ed99728d0b31c02d4d00ff796a7e08d2cfcf4fe18a74e15dad951334cb5f7

candidate:
1b12e6c17a28715a206440c5aeb37e8f3c6ea1093d3a45ce6dca536af5d388a9
```

They are plumbing evidence only. Formal training must restart both arms from
the same exact-zero adapter and frozen parent, not from either smoke
checkpoint.

## Interpretation

This closes the implementation risk that a restricted adapter could fail to
differentiate through physical `B_fast` restoration and retained STOMATE.
It also proves that optimization cannot silently modify unrelated fast-day
processes.

The unit coefficients used here were deliberately provisional. The next step
is formal independent-component gradient calibration on train/train data,
followed by a fresh matched A/B from the exact-zero adapter. No temporal,
spatial, joint, or sealed-test sample may influence those coefficients.
