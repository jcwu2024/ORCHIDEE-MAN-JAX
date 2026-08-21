# Target-Oriented Daily Operator Contract v1

Status: **frozen before successor architecture selection**.

Date: 2026-08-21.

## Decision

The daily neural operator is promoted by absolute long-horizon accuracy in
GPP, respiration, NPP/NEP/NEE, live biomass, and ecosystem carbon, not by
forcing every learned process field through an independent 95--99% reduction
gate. The complete operator boundary, conservative water/carbon update, exact
retained tail, and hard physical gates remain unchanged.

The machine contract is
`manifests/coarse_graining/target_oriented_daily_operator_v1.json`, with
SHA256 `7c9857c0ca57ee65cfcf7514a216a217c3b1661f6cc869c6dfb85d3786bfce5c`.
The implementation is
`daily_operator/target_oriented_contract.py`.

This decision does not select a new neural architecture. In particular, it
neither rejects Attention nor authorizes another DOC-specific decoder. The
next architecture must be discussed with the user before implementation.

## Available Science Quantities

Teacher v6 and the existing complete daily operator provide:

| Quantity | Source in the daily operator | Optimization role |
| --- | --- | --- |
| GPP | direct `daily.fast_gpp` process output | independent primary flux |
| Maintenance respiration | direct part-resolved `daily.fast_maintenance_respiration` followed by retained daily aggregation | component and reported gate |
| Growth respiration | retained exact `npp_calc` result | component and reported gate |
| Autotrophic respiration | maintenance plus growth respiration | independent primary flux |
| NPP | retained exact result; `GPP - Ra` identity | required derived report/gate, not a second head |
| Heterotrophic respiration | retained litter/soil result driven by conservative carbon endpoints | independent primary flux |
| NEP | `NPP - heterotrophic respiration` | required derived report/gate |
| NEE | `-NEP`, positive to the atmosphere | required derived report/gate |
| AGB/BGB | source-partitioned 12-part live biomass state | independent primary stocks |
| Total live biomass | `AGB + BGB` | required derived report/gate |
| Nonliving ecosystem carbon | ecosystem carbon minus total live biomass | independent primary stock |
| Ecosystem carbon | live biomass plus declared independent litter/POC/DOC/interception inventories | required derived report/gate |

NPP, total autotrophic respiration, NEP/NEE, total live biomass, and ecosystem
carbon remain scientifically important. They are derived from an independent
basis inside the objective and reported separately so the same carbon identity
is not counted two or three times during optimization.

## Output Hierarchy

The current mechanistic boundary contains 181 complete fields. Twenty are
already reconstructed exactly. All 161 learned fields are classified through
their 48 source-owned labels:

| Class | Fields | Scalar coordinates | Training role |
| --- | ---: | ---: | --- |
| Direct primary science | 2 | 26 | GPP and part-resolved maintenance respiration; supervised through science trajectories |
| State/budget essential | 93 | 3,682 | inventory endpoints, thermal tendencies, daily temperature context, and retained-tail interface state |
| Auxiliary process supervision | 66 | 35,298 | typed water, energy, litter, POC, and DOC process detail; low-weight regularization and diagnosis |
| Exact/diagnostic | 20 | 365 | exact reconstruction checks; zero optimization weight |

The large majority of scalar coordinates are therefore auxiliary, not
promotion-defining primary outputs. This classification does not yet decide
whether the successor should keep all auxiliary learned heads, share them, or
replace some with source-backed reconstruction. That is an architecture and
boundary decision reserved for the next discussion.

## Frozen Training Objective

The architecture-neutral composition is:

```text
L = 1.0 * L_state_1d
  + 1.0 * L_science_7d
  + 1.0 * L_science_30d
  + 0.1 * L_process_necessary
```

- `L_state_1d` is the equal-active-group macro over the seven canonical state
  groups after frozen train/train scale normalization. Each group uses one
  half normalized MSE plus one half squared normalized signed bias.
- `L_science_7d` and `L_science_30d` use the independent flux basis GPP/Ra/Rh
  and independent stock basis AGB/BGB/nonliving ecosystem carbon. Cumulative
  fluxes and terminal stocks each contribute normalized MSE and signed bias.
- `L_process_necessary` excludes the two direct primary labels to avoid
  duplicate science weight. State/budget-essential label MSE has weight `1`;
  auxiliary label MSE has weight `0.25`; the complete term is then multiplied
  by `0.1`.
- Derived NPP, NEP/NEE, total live biomass, and total ecosystem carbon are
  still mandatory report and acceptance keys.

All scales must be fitted from existing Teacher-v6 train/train rows and frozen
before training. This is statistics preparation, not Teacher regeneration.
Historical v5 rows, validation tuning, and sealed-test selection remain
forbidden.

## Absolute Acceptance

Every temporal, spatial, and joint non-test validation slice must pass. There
is no candidate-versus-control promotion criterion.

| Horizon | Per-key normalized RMSE | Per-key absolute normalized signed bias |
| --- | ---: | ---: |
| 7 days | at most `0.35` | at most `0.10` |
| 30 days | at most `0.25` | at most `0.07` |

The keys are GPP, maintenance respiration, growth respiration, total
autotrophic respiration, NPP, heterotrophic respiration, NEP, NEE, AGB, BGB,
total live biomass, nonliving ecosystem carbon, and ecosystem carbon.

After 7/30-day acceptance, the 365-day gate requires:

- annual gross/component flux median and p95 relative bias at most `2%` and
  `5%`;
- NEP/NEE median and p95 absolute normalized bias at most `0.03` and `0.10`,
  because their Teacher denominator can be near zero;
- year-end stock median and p95 relative error at most `2%` and `5%`;
- scale-normalized fallback reporting for all low-denominator cases.

At every horizon, undeclared defined-status mismatches, discrete mismatches,
nonfinite learned/declared-defined values, negative declared stocks, and
water/carbon budget failures have required count zero. Declared dynamic
`rveget` status remains supervised and reported but does not veto an update,
as frozen by the post-preflight state-anchor amendment. Budget and stock
tolerances remain `2e-12`.

## Verification

The focused tests validate the frozen hash, all 48 label categories, the
161/20 learned/exact partition, carbon identities, seven-group one-day state
anchor, independent-basis 7/30-day objective, auxiliary downweighting, and
explicit total-loss coefficients. They do not train a model or access Teacher
data.

## Next Gate

Before any neural implementation, decide with the user:

1. which of the 93 state/budget-essential fields need direct learned heads;
2. whether the 66 auxiliary fields remain shared low-capacity heads, become
   diagnostics reconstructed from primary state transitions, or are omitted
   from optimization while still materialized at the boundary;
3. how the accepted coordinate-aligned Attention evidence should be reused in
   a target-oriented successor without inheriting the rejected v4/v5 physical
   materialization heads.

No GPU training, 365-day rollout, Gate D2, or product acceptance is opened by
this contract alone.
