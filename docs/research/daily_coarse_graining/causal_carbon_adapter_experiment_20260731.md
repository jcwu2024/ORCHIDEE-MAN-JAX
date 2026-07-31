# Causal Carbon Adapter Experiment C

Date: 2026-07-31

## Status

Experiment C is frozen and locally implementation-complete, but it has not
yet passed a real-shard GPU feasibility run. It is a bounded hypothesis, not a
validated neural daily surrogate.

The rejected `mixed_horizon_stability_v1` checkpoint is not reused. Both C
arms start from the completed Experiment B one-step control checkpoint:

```text
8a099c85f8c869398ffe2203b0fa1d558735b0f819ac606d112d452db2fb2ec1
```

The 1,954,041-parameter `axis_process_coupled_v1` parent is frozen. A
zero-initialized 76,877-parameter adapter may alter only 141 causal PFT14
columns. The remaining 2,714 fast-day columns and the dynamic undefined head
remain parent-owned and bit-exact.

## Why This Experiment Exists

The Experiment B screen improved aggregate 30-day state RMSE while worsening
signed carbon tendencies, NPP, growth and maintenance respiration, biomass,
LAI, `carbon_32l`, and `deepC_peat`. Source inspection showed that retained
STOMATE already enforces:

```text
NPP = GPP - maintenance respiration - growth respiration
```

The missing supervision is therefore upstream. Same-day `gpp_daily` and
`resp_maint_part` enter retained STOMATE through `B_fast`; the cross-day
`gpp_daily` state is reset at the daily boundary and cannot supervise the
actual same-day GPP by itself.

## Causal Boundary

The adapter changes four equal-weight PFT14 field groups:

| Field | Adapted columns |
|---|---:|
| `daily_interface.gpp_daily` | 1 |
| `daily_interface.resp_maint_part` | 12 |
| `ok_leak.carbon_32l` | 96 |
| `ok_leak.deepC_peat` | 32 |

The Contract v5 adapter layout SHA256 is:

```text
2ed9111105ddb8bad68f776fd806914e8e9a68f45d531140a7d7dfb6613b0797
```

The causal objective layout SHA256 is:

```text
aae7bcc69a16d672c20e931a1edf98caec4bd472858ee116a47943ec0ccfa1fe
```

Downstream primary supervision covers PFT14 NPP, growth respiration,
maintenance respiration, biomass, LAI, `carbon_32l`, and `deepC_peat`.
Flux-value bias and stock-tendency bias are separate loss components. DOC and
litter are report-only no-regression guards rather than terms that can be
traded against the primary objective.

## Matched Comparison

Both arms receive the same frozen parent, exact-zero adapter, anchor batches,
seed, optimizer, and update count:

- `causal_interface_one_step_control` trains the four field-balanced
  same-day interface groups;
- `causal_interface_rollout_candidate` adds downstream next-state, recursive
  rollout, flux-bias, and stock-tendency-bias losses.

All treatment coefficients must be calibrated on train/train data and frozen
before matched training. Temporal, spatial, and joint validation cannot be
used for coefficient selection. The sealed test split remains untouched.

The frozen protocol is:

```text
manifests/coarse_graining/canonical_669_causal_carbon_adapter_experiment.json
SHA256 35ede512bbc5f55881317cb700d7512d71993391c65f36ed8a5de087c4d4c3e7
```

## Execution Gate

Paid matched training is forbidden until the bounded real-shard feasibility
gate passes. That gate performs eight matched updates, covers 1/3/7-day
horizons, and requires:

- finite loss and adapter gradients;
- every control and candidate update applied;
- frozen parent and protected fast-day columns bit-exact;
- zero unexpected defined-status, discrete-state, nonfinite, and negative
  source-constrained carbon-stock failures;
- train/train data only and no sealed-test access.

Its provisional unit coefficients test plumbing only. They are not carried
into formal training. Formal train-only gradient calibration remains required
after feasibility passes.

Local evidence before the real-shard gate:

- adapter architecture, objective, protocol, and surrounding regression:
  69 tests passed;
- a complete synthetic two-day JIT/reverse-mode path passed through physical
  `B_fast` restoration and a retained transition;
- Ruff, `py_compile`, and diff checks passed.

## Stop Rule

If the feasibility gate fails, stop before paid training and attribute the
implementation failure. If the later full model-selection screen fails,
reject C without validation-driven weight tuning, 365-day debugging, sealed
test inspection, or fallback to the rejected B candidate.
