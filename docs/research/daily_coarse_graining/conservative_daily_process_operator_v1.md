# Conservative Daily Process Operator v1

Status: active architecture decision, not yet implemented or validated.

Date: 2026-08-02.

This decision supersedes both direct next-stock prediction and the proposed
neural 13-driver plus exact 48-step `OK_LEAK` scan as final surrogate
architectures. Those implementations and assets remain historical evidence,
diagnostics, and Teacher Oracles.

## Scientific Boundary

The production surrogate must implement one daily Markov transition:

```text
S[d] + F_native[d] + P[d]
  -> daily fluxes, transfer fractions, and bounded tendencies
  -> one constrained state update
  -> retained source-backed daily processes
  -> S[d+1]
```

`F_native[d]` contains the native forcing records and their times/durations.
The current paper forcing contributes five records to a day window: the
predecessor needed at the boundary and four current six-hour records. The
surrogate may perform deterministic unit conversion, calendar, solar-geometry,
and interval accounting. It must not reconstruct 48 interpolated records or
execute 48 recurrent state updates during inference.

## Model Structure

### Native-forcing encoder

Encode each native record with its timestamp and interval duration, then pool
the short ordered sequence into process-specific daily context. The encoder
must accept a mask so a future forcing product can use a different native
record count without changing the scientific boundary.

### Process-structured state encoder

State is grouped by physical owner rather than flattened as anonymous values:

- canopy and vegetation;
- soil water, groundwater, interception, and snow inventories;
- soil and snow thermal state;
- litter, particulate soil carbon, DOC, and related carbon inventories;
- exact discrete lifecycle state.

Shared encoders operate along PFT, soil-layer, carbon-pool, and biomass-part
axes. Diagnostics and packet mirrors are reconstructed, not independently
predicted.

### Daily process heads

Heads predict quantities that have process meaning:

- water inputs, evapotranspiration, runoff, drainage, and inter-reservoir
  transfers;
- energy-driven bounded temperature tendencies and snow phase transfers;
- canopy exchange, GPP, transpiration, and interception terms;
- litter inputs, respiration/export fluxes, and carbon-pool transfer fractions;
- the compact daily interface consumed by retained exact daily processes.

The network must not independently predict next-day water or carbon stocks.

### Constrained updater

For an inventory `x_i`, an outgoing fraction is parameterized as, for example,

```text
q_i = 1 - exp(-softplus(k_i) * one_day)
```

and destination shares use a normalized nonnegative partition. Therefore total
outflow cannot exceed available stock. Internal transfers cancel in the
combined budget; external input, respiration, and lateral export remain
explicit. Post-hoc clipping is a guard failure, not the conservation method.

Water and carbon use hard inventory identities. Thermal variables use an
energy-backed or explicitly bounded daily tendency contract. Exact discrete
states are updated by retained rules or classifiers with explicit semantics,
not rounded continuous outputs.

## PFT-Extensible Contract

This section is frozen by the machine-readable
`manifests/coarse_graining/daily_neural_pft_interface_v1.json` contract and its
validated adapter in `research/daily_coarse_graining/daily_pft_interface.py`.

PFT14 is the current training and acceptance scope, not an architecture limit.
The model consumes arrays with an explicit PFT axis:

```text
pft_state[n_pft, ...]
pft_parameters[n_pft, ...]
pft_traits[n_pft, ...]
pft_fraction[n_pft]
active_pft_mask[n_pft]
```

Weights are shared along the PFT axis. The model must not use landpoint ID or a
PFT14-specific embedding as a shortcut. Functional traits and structural flags
represent C3/C4, tree/grass/crop, evergreen/deciduous, natural/managed, and
other source-backed branches. A different `n_pft` may require JAX recompilation
but must not require an architecture rewrite.

Required structural tests include:

- zero-fraction inactive PFTs do not change active outputs;
- permuting PFT state, parameters, traits, fractions, and masks produces the
  corresponding output permutation;
- no neural model code hard-codes PFT index 13;
- adding a legal PFT slot does not require changing model code.

Supporting another PFT scientifically still requires its source-backed Teacher
branches, parameter coverage, training data, and held-out validation. PFT14
data alone cannot establish another PFT's behavior.

## Parameter Ownership

Every transition-relevant parameter belongs to one of three classes:

1. retained exact daily process parameter: passed directly to the exact JAX
   process and not relearned;
2. explicit parameterized fast-process factor: its known mathematical role is
   retained and the network predicts only an environmental modifier;
3. learned conditional parameter: passed to the network and varied in
   controlled Teacher data.

The current 84-value condition is four parameter families over 14 PFTs:
`alloc_min`, `residence_time`, `vcmax25`, and three
`maint_resp_slope` coefficients. In the paper data, only four calibrated PFT14
values vary materially by landpoint. A constant input does not teach parameter
sensitivity. Any parameter intended for inversion must either retain its exact
formula or receive controlled perturbation data and pass finite, directional
gradient tests.

Already-daily maintenance, allocation, turnover, phenology, and mortality
should remain exact where practical. In the redesigned boundary this keeps
most paper calibration parameters explicit; `vcmax25` and other parameters
inside replaced subdaily processes require explicit conditioning or retained
parameter factors.

## Data Contract Before Training

The completed Gate C1 inventory is
`manifests/coarse_graining/daily_flux_label_inventory_v1.json`; its readable
decision record is `daily_flux_label_inventory_v1.md`.

Existing shards contain day-start state, native forcing, named conditions,
the historical `B_fast` target, next-day state, and selected diagnostics. They
do not necessarily contain every internal daily transfer needed by the new
updater.

A schema-level inventory must classify every required label as:

- directly present in existing shards;
- exactly derivable from existing fields and a complete source-backed budget;
- absent or non-identifiable and therefore requiring supplemental
  diagnostic-only Teacher capture.

Do not inspect all 669 landpoints manually or regenerate them before this
inventory. Endpoints alone cannot uniquely identify simultaneous inputs,
internal transfers, respiration, and export.

## Non-Neural Admission Gate

Before implementing a neural model, feed true Teacher daily flux labels into
the constrained updater and require:

- accepted next-day canonical state reconstruction at declared field-aware
  tolerances;
- exact discrete and defined-status behavior;
- nonnegative physical inventories without post-hoc clipping;
- complete water and carbon closure and a declared thermal residual;
- cold-start, later-day, restart-year, and active/inactive-mask coverage;
- finite forward- and reverse-mode gradients for every tunable parameter;
- no 48-step forcing reconstruction or state scan in the candidate inference
  graph.

Failure at this gate is a boundary or label problem. It must not trigger a GPU
training search.

## Training and Acceptance Ladder

After non-neural admission:

1. supervised daily flux/rate fit;
2. one-day next-state and budget validation;
3. 7- and 30-day free rollout with truncated backpropagation;
4. 365-day validation without full-year retained graphs;
5. complete-chain validation and one sealed test evaluation;
6. measured CPU/GPU inference cost, including the retained exact daily tail.

Daily flux error, monthly/annual accumulated bias, stock tendency, multiannual
drift, restart equivalence, parameter gradients, and held-out landpoint skill
are separate gates. A single aggregate RMSE cannot promote the model.
