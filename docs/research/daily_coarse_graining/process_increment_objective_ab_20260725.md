# Process-Balanced State-Increment Objective A/B

Date: 2026-07-25

## Purpose

Test one bounded objective change before generating more Teacher data. The
experiment asks whether process-balanced next-state supervision plus explicit
daily state-increment supervision reduces recursive carbon-state drift on
already-seen conditions. It does not test spatial-data sufficiency or long-run
readiness.

## Frozen Inputs

- Dataset: the accepted nine-point contract-v5 dataset, with 96,354
  train/train transitions.
- Initialization: the same accepted one-step checkpoint used by the frozen v1
  multistep curriculum.
- Architecture: unchanged 1,963,369-parameter canonical model.
- Curriculum: `1:64,3:64,7:64`, batch size 4, learning rate `1e-4`.
- Seed: `20260724`.
- Splits: train and validation only. Sealed test landpoints and 2008-2010 test
  years remain inaccessible.

The accepted v1 multistep checkpoint is the baseline. The candidate must use
the same initialization and update budget; a larger network, extra Teacher
data, extra updates, or a changed sample seed invalidates the A/B.

## Candidate Objective

`process_increment_v2` keeps the existing fast-day target, undefined-status,
and canonical next-state losses. It changes the next-state scalar weighting
from uniform over 3,854 values to equal total weight over eight source-owned
process groups:

| Process group | Width | Leaves | Weight |
|---|---:|---:|---:|
| STOMATE carbon flux | 114 | 13 | 0.125 |
| STOMATE vegetation structure | 56 | 11 | 0.125 |
| STOMATE litter and turnover | 218 | 8 | 0.125 |
| STOMATE carbon storage | 1,875 | 17 | 0.125 |
| STOMATE environment and phenology memory | 123 | 58 | 0.125 |
| HYDROL | 622 | 47 | 0.125 |
| thermal and energy | 653 | 30 | 0.125 |
| surface exchange and finalize | 193 | 73 | 0.125 |

The assignment covers every continuous state value exactly once. Its contract
weighting SHA256 is
`2ec69cc92a4c81a2694693063596382b8edd5b6f00b4311350f9738001695f85`.

The additional loss compares the predicted daily transition
`S_pred[d+1] - S_pred[d]` with the Teacher transition
`S_teacher[d+1] - S_teacher[d]`. Each value is normalized by train/train-only
`state_delta` statistics. The normalizer is bounded below by
`0.001 * state_scale` to prevent nearly constant state values from dominating.
On the production statistics this floor raises 249 of 3,854 columns; all
resulting scales are finite and positive. Undefined endpoints are excluded by
the existing ORCHIDEE finite/sentinel policy.

Default user behavior remains `canonical_multistep_v1`. The candidate is
enabled explicitly with:

```bash
MULTISTEP_OBJECTIVE=process_increment_v2
STATE_INCREMENT_LOSS_WEIGHT=1.0
STATE_DELTA_FLOOR_RATIO=0.001
```

## Evaluation Matrix

Run seven free-feedback days 100-106 for the same four frozen cases:

| Split | Landpoint | Year |
|---|---|---:|
| train/train | `281.0-095.0` | 2004 |
| train/validation | `281.0-095.0` | 2005 |
| validation/train | `215.0-119.0` | 2004 |
| validation/validation | `215.0-119.0` | 2005 |

Use the existing field-complete drift report. Record daily global normalized
RMSE, process-family RMSE, defined-status mismatches, discrete mismatches, and
at least these named slow states: `biomass`, `litterpart`, `npp_daily`,
`resp_growth`, `resp_maint`, and `resp_hetero`.

## Decision Gate

The candidate advances to a spatial-data pilot only when all conditions hold:

1. zero defined-status and discrete mismatches in all four cases;
2. lower Day-7 global RMSE than v1 on train/train;
3. lower Day-7 normalized RMSE than v1 for both `biomass` and `litterpart` on
   train/train, with no material regression in the other four named carbon
   states;
4. no material regression in train/validation global or named-state metrics;
5. finite losses and gradients throughout the equal-budget curriculum.

Spatial validation is reported but is not expected to close without new
landpoints. If the candidate fails the seen-condition gates, stop this
objective experiment. Do not tune repeatedly, add data, relax masks, inspect
the sealed test split, or claim 30-day/365-day readiness. If it passes, freeze
the objective and select a bounded, source-driven spatial pilot before any
larger Teacher-data production.
