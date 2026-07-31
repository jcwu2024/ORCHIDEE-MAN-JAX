# Experiment C Stock-Adapter Ablation Protocol

Date: 2026-07-31

## Purpose

Experiment C is rejected and remains rejected. This bounded attribution asks
one question before designing a replacement architecture:

> Did the trained flux adapter learn useful corrections that survive when the
> independently learned carbon-stock correction is removed?

It does not retrain or promote the rejected candidate.

## Frozen Intervention

Load the completed Experiment C checkpoints without modification. Apply this
evaluation-only transform to the rollout candidate:

```text
carbon_flux_interface  -> trained candidate parameters
carbon_stock_interface -> exact-zero output weight and bias
```

The frozen parent, protected columns, dynamic undefined head, training
checkpoint, and control arm remain unchanged. The transform and disabled group
ID must be recorded in the screening identity.

## Evidence Population

Use the same deterministic one-shard-per-slice inventory as the accepted
Experiment C screening smoke:

- one temporal model-selection shard;
- one spatial model-selection shard;
- one joint model-selection shard;
- Day 1 teacher-forced and Day 7/30 free rollout;
- no sealed-test reference.

The new control report must reproduce the prior smoke control metrics exactly.
Otherwise the attribution is invalid.

## Decision Rules

The attribution supports a conservation-constrained successor only if all of
the following hold:

1. Every hard count remains zero and both restart-split probes pass.
2. Day-1 GPP and maintenance-respiration interface metrics reproduce the
   unablated candidate exactly; these adapter columns were not changed.
3. For both `carbon_32l` and `deepC_peat`, every slice's Day-1 interface error
   is no worse than 1.05 times the matched control and is at most half the
   unablated candidate error.
4. At least 24 of the 27 predeclared flux-bias gates still pass relative to
   the matched control.
5. Litter and DOC guards do not exceed their existing 1.05 ratio threshold.

Interpretation:

- all rules pass: audit and prototype a flux-form conservative transition;
- stock recovery passes but flux retention fails: the learned corrections
  were compensating for inconsistent stocks, so do not reuse this adapter;
- stock recovery fails: reject the proposed conservative successor before
  implementation and reassess the whole-day replacement boundary.

This smoke is attribution evidence only. It cannot revive Experiment C,
authorize confirmation seeds, inspect sealed test, or justify a user-facing
daily surrogate.
