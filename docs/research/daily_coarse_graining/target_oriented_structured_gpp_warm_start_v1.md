# Structured GPP Warm Start V1

Date: 2026-08-29

Status: frozen after accepted free-GPU preflight; paid execution requires a
separate resource approval

Machine contract:
`manifests/coarse_graining/target_oriented_structured_gpp_warm_start_v1.json`

Contract SHA256:
`1a3b8138b7fcef0cc28f7669d89cdfaae053244b706eebf23b3aa4efcc4818a6`

## Purpose

This stage teaches the new structured GPP specialist the direct daily
photosynthesis mapping before any recursive science gradient is allowed. The
coordinate-aligned Attention parent and distribution-aware maintenance adapter
remain frozen. Only the `12,879,363` parameters under
`target_oriented.structured_gpp` are differentiated and checkpointed.

## Data And Objective

The stage executes 250 updates selected in ascending order from prepared
train/train artifacts `252..511`, excluding the calibration and evaluation
groups. Every update uses all 208 existing Teacher days in its artifact: 32
standalone days, 56 Teacher day starts from 7-day windows, and 120 Teacher day
starts from 30-day windows.

The sole objective is

```text
0.5 * normalized GPP MSE
  + 0.5 * normalized signed GPP bias squared.
```

It uses Teacher-defined active non-bare vegetation and the frozen train/train
GPP scale. It does not recursively roll candidate state and does not include
irreducible frozen-parent state or process losses.

## Optimizer And Persistence

The accepted AdamW settings are beta `0.9/0.95`, epsilon `1e-8`, weight decay
`0.01`, constant learning rate `4e-6`, and global norm clip
`17.256861560753983`. The clip is the accepted p90 of eight train-only
preflight gradients, not a validation-tuned value. Recovery is written every
25 updates and the only retained checkpoint is update 250.

## Endpoint Gate

Direct GPP is evaluated before and after training on eight train/train
artifacts disjoint from the 250 training artifacts. A capability pass requires
at least 10% reduction in the exact aggregate direct-GPP loss, improvement on
at least six of eight artifacts, finite optimizer state, exact preservation of
all parent parameters, and zero final one-day physical hard failures.

The checkpoint is not promotion eligible. A capability pass opens only a
train-only 7/30-day gradient calibration at the warm-start endpoint. Joint
training, validation, 365-day rollout, Gate D2, sealed test, and product
acceptance remain closed.
