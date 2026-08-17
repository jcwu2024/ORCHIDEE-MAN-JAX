# Mechanistic-Output Strong Baseline Architecture Review v1

Date: 2026-08-17.

Status: user approved and implemented. Bounded real-v6 admission and one-epoch
smoke pass; the 100-epoch one-point result fails four independent expert gates
and is controlled by `mechanistic_output_one_point_qualification_result_20260817.md`.

## Decision Being Reviewed

The 49.75-million-parameter axis-aligned process-expert backbone is retained.
The failed one-point result does not show that Attention or model capacity is
the primary problem. Its remaining loss is concentrated in redundant derived
outputs and physically coupled endpoint families. The proposed successor is
therefore a new output mechanism around the same capable backbone:

`axis_aligned_process_expert_mechanistic_output_v1`.

It remains one daily Markov operator and still binds through:

```python
bind_daily_operator(
    input_assembler=canonical_assembler,
    process_head=candidate_head,
    retained_tail=canonical_retained_tail_adapter,
)
```

No historical checkpoint, v5 data, 48-record forcing reconstruction, 48-step
state scan, or cross-day neural hidden state is admitted.

## Inputs And Backbone

The neural path consumes the day-start Markov state, five supplied native
forcing records, static/trait conditions, calendar context, and the physical
parameters that act inside the replaced fast processes. In particular,
`vcmax25` routes to canopy/GPP and `maint_resp_slope` routes to maintenance
respiration. `alloc_min` and `residence_time` do not enter the neural encoder;
they continue to act in the retained exact daily processes after the neural
fast transition.

The retained backbone has:

- a width-256, two-block, eight-head native-forcing Attention encoder;
- dynamic axis-aligned tokens for PFT, layer, pool, plant-part, snow, soil,
  litter, DOC, and other declared physical coordinates;
- seven independent width-384 process experts with three eight-head
  Attention/feed-forward blocks each;
- bounded directed process messages, initialized near zero;
- shared weights along repeated physical axes and explicit PFT masks.

Attention is used to let a process output select relevant information from
the admitted tokens for the same day. It does not store an unlimited history
and is not being compared with a weak MLP control at this stage. The baseline
must first pass absolute scientific gates on its own.

## Complete Output Boundary

The existing public process boundary contains 181 named fields and 39,371
scalar values. The proposed classification is machine-readable in
`strong_baseline_mechanistic_output_review_v1.json`:

| Class | Fields | Scalars | Handling |
| --- | ---: | ---: | --- |
| Learned primary primitives/endpoints | 149 | 35,895 | Process-owned local heads |
| Exact daily-boundary derived | 16 | 218 | Source-backed kernels after primitives |
| Physical baseline plus learned residual | 1 | 3,136 | DOC diffusion magnitude |
| Auxiliary/status | 15 | 122 | Retained-tail and defined-state interface |
| Total | 181 | 39,371 | Complete boundary unchanged |

The neural heads therefore emit 165 fields and 39,153 scalar values. The
complete process boundary still exposes all 181 fields after exact
reconstruction. Removing only 218 scalar outputs is not intended as parameter
reduction; it removes internally inconsistent supervision that dominated the
failed fit.

### Exact derived fields

The following independent neural heads are removed:

- `z0m`, `roughheight`, `roughheight_pft`, and `drysoil_frac`;
- final `gtemp` and `stempdiag`;
- `cgrnd`, `dgrnd`, `cgrnd_snow`, `dgrnd_snow`, `lambda_snow`, `pcapa_en`,
  `soilcap`, `soilcap_pft`, `soilflx`, and `soilflx_pft`.

They are reconstructed from learned final vegetation, snow, hydrology, energy,
and thermal primitives with existing source-backed JAX kernels. The first six
families pass the 20-point-year boundary panel at maximum error `2.22e-15` or
less. All ten coefficient fields pass all 20 point-years within the frozen
`1e-10 + 1e-12*abs(target)` tolerance. `z0h` is explicitly excluded from this
list because it fails every panel point-year and remains learned.

Exact fields do not receive independent trainable heads or independent loss
terms. Their reconstruction error remains a hard diagnostic gate so a wrong
primitive cannot be hidden by a separate derived-output head.

### Coupled learned energy head

The following quantities remain learned but no longer use unrelated scalar
decoders:

- final surface temperature and PFT surface temperatures;
- `fluxsens`, `vevapp`, `evapot`, `evapot_corr`, and `qsurf`;
- the related daily sensible/latent energy labels.

One surface-energy expert emits shared latent primitives for sensible heat,
water-vapor transfer, surface humidity, and surface temperature. Small
field-specific projections then map those shared primitives to the required
Teacher endpoints and daily integrals. This preserves distinct units and
masks while forcing the endpoints to be conditioned on the same energy state.

### Capacity-aware canopy endpoint

Canopy interception is represented relative to the physical storage capacity
derived from vegetation and LAI. The head predicts normalized occupancy and
bounded input/output transfers. The rare legal negative `qsintveg` state is
represented as separate nonnegative debt with an explicit activity/status
quantity. This keeps the conservative storage-minus-debt identity and avoids
asking one smooth scalar head to fit both ordinary occupancy and a rare debt
regime.

### DOC diffusion baseline and residual

`typed.ok_leak.doc_diffusive_interface` remains a nonnegative magnitude. Its
base value is the mean of source-kernel magnitudes at the day-start and
predicted day-end DOC/temperature boundaries. The litter/POC/DOC expert first
predicts a residual coordinate. After the conservative update has formed the
day-end primitive state, a post-update composer evaluates the two boundary
kernels and applies the residual in the pre-constraint coordinate centered on
that baseline. Zero residual reproduces the physical baseline and the final
value cannot become negative.

An explicit activity quantity handles interfaces whose Teacher magnitude is
exactly zero. This matters because the source temperature cutoff can make a
boundary proxy zero even when the cumulative Teacher label was active during
part of the day.

The label is not treated as a signed transfer. Direction remains owned by the
applicable concentration ordering. The conservative DOC endpoint transition,
not this magnitude alone, owns the next-day DOC stock.

### Auxiliary and status fields

Six daily-summary fields, eight `ok_leak` auxiliary fields, and the `rveget`
defined-state logit remain explicit because the retained daily tail or status
boundary consumes them. They are supervised and reported separately from the
149 primary process fields so their losses cannot dominate or hide primary
scientific competence. This category is not a claim that every auxiliary is
physically negligible.

## Forward Use Of The Outputs

The proposed execution order has an explicit primitive/composition split:

```text
day-start state + native forcing + fast-process parameters
  -> axis-preserving encoders and seven Attention experts
  -> learned primitive transfers, endpoints, auxiliary values, and DOC residual coordinate
  -> one float64 conservative water/carbon/thermal primitive update
  -> exact roughness/hydrology/thermal reconstruction from predicted day end
  -> coupled field materialization and nonnegative DOC-baseline composition
  -> complete 181-field process boundary
  -> canonical retained exact daily tail
  -> next-day state and daily fluxes
```

The outer `bind_daily_operator(...)` API remains unchanged. Internally, the
process prediction must distinguish raw/primitive head outputs from the
post-update complete named-output materialization. This prevents a circular
shortcut in which a day-end-dependent physical baseline is evaluated from the
day-start state.

All neural algebra remains float32. Raw head outputs are cast to float64 before
normalization reversal, physical constraints, exact reconstruction,
conservative updates, retained processes, budgets, and scientific loss.

## Training Objective

One-day pretraining still uses equal expert weight and equal label-family
weight within each expert. The revision changes what is optimized:

1. primary learned primitives/endpoints are supervised in their declared
   physical representations;
2. the DOC head is supervised on the correction around the frozen boundary
   baseline and on zero/nonzero activity;
3. exact derived fields and the final DOC magnitude are evaluated after the
   primitive update, not learned by independent standalone heads;
4. energy consistency is evaluated jointly across temperature, humidity,
   sensible heat, and evaporation outputs;
5. GPP, NPP, maintenance respiration, AGB, BGB, total-live biomass, signed
   tendency bias, states, and budgets remain downstream scientific losses.

Long-horizon training is not used to compensate for a model that cannot learn
one-day physics. The staged order remains local forward/loss/gradient and
one-day state/budget checks, one-point-year memorization, 64-point-year
learnability, complete v6 one-day epochs, and only then 7/30/365-day rollout.

## Qualification And Stop Rule

Before any paid job, the revised local/free gate must prove complete field
coverage, exact reconstruction, DOC baseline composition, PFT isolation,
finite JVP/VJP paths, full forward/loss/gradient, one-day state construction,
and water/carbon budgets. The one-point-year and 64-point-year gates are then
rerun from fresh parameters. Failure still blocks full-data training.

No matched weak control is required for this qualification. The scientific
question is whether the model reaches the absolute one-day and rollout gates,
not whether it is slightly better than the rejected model. Explanatory
ablations become meaningful only after a capable baseline exists.

## Approval Boundary

This document and its machine-readable inventory authorize no network edit,
training-contract change, checkpoint reuse, or GPU job. User approval is
required before implementing the revised process head.
