# Successor Route-Local Decoder Architecture Review v1

Status: **proposal only; user architecture review required before
implementation**.

Date: 2026-08-16

## Decision Question

Should the next formal process-head candidate retain the accepted
source-routed Attention body but replace its universal output decoder with
source-route-local decoders?

No implementation, checkpoint, GPU run, paid job, or architecture promotion
is authorized by this document.

## Evidence Requiring A New Mechanism

The failed mixed-point annealed continuation closes another-training-only
explanation. It used 1,000 unique point-years, a four-point mixed batch, and a
real cosine decay. Training loss decreased and maintenance validation improved
22.3%, but the NPP/AGB/BGB/total-live composite worsened 6.98% and all six
7/30-day cells regressed.

Loss-only repairs are also closed in their tested forms:

- named AGB/BGB tendency training repaired stock-gradient magnitude but made
  all six primary cells worse;
- minimum-norm multi-objective gradients, AdamW handoff variants, direct
  directions, group-aware recalibration, cumulative microsteps, and bounded
  post-training projection did not produce an admissible useful update;
- adding more updates, changing batch diversity, or retuning a static stock
  coefficient would repeat a rejected mechanism.

The architecture attribution is more specific:

- the current two-layer universal output decoder carries 96.1526% of the
  stopped direction's squared norm;
- decoded `daily.fast_maintenance_respiration` accounts for 94.2874% of the
  absolute NPP and 95.1144% of the BGB output-nonlinearity contribution;
- GPP offsets rather than causes that failed finite response;
- the current decoder is shared by 181 named fields and 39,371 scalar output
  elements, even though five source routes already exist in the contract.

This does not prove that shared decoding is the cause. It supplies a bounded,
testable hypothesis that is materially different from wider Attention,
another loss weight, or another optimizer.

## Proposed Candidate

Provisional name: `source_routed_route_local_decoder_attention_v3`.

The candidate retains unchanged:

- five native forcing records and all source-typed state/PFT/static/annual
  tokens;
- two day-local forcing Attention blocks;
- 181 named field queries and three source-masked process-Attention blocks;
- the source adjacency ledger and process graph;
- explicit `vcmax25` and maintenance-parameter factors;
- the conservative updater, canonical retained exact tail, and
  `bind_daily_operator(...)` boundary;
- no cross-day hidden state, 48-record reconstruction, or 48-step state scan.

Only output decoding changes. The existing route key for every field selects
one complete two-layer decoder:

```text
field query + field/PFT/axis embeddings
  -> water decoder                         (width 256)
  -> energy decoder                        (width 256)
  -> carbon-other decoder                  (width 256)
  -> fast-GPP decoder                      (width 256)
  -> fast-maintenance-respiration decoder  (width 256)
  -> existing field constraint and physical parameter route
```

All fields in one route may share that route's decoder. No field may be sent
to more than one route, and all 181 fields must be covered. In particular,
maintenance decoder gradients cannot directly update the GPP, carbon-other,
water, or energy decoder parameters. The Attention body remains shared, so
physically useful cross-process context is not removed.

## Parameter-Matched Control

Provisional name: `source_routed_shared_wide_decoder_attention_control_v3`.

The control uses the identical Attention body, inputs, field queries, source
masks, graph, constraints, data, objective, optimizer, paired initialization
seed, and updates. Every same-shaped non-decoder parameter has exactly the
same initial value in both arms. The control retains one universal decoder but
widens that decoder to 1,280 hidden units.

Exact decoder parameter counts are:

```text
candidate: 5 * ((256 * 256 + 256) + (256 + 1)) = 330,245
control:        (256 * 1280 + 1280) + (1280 + 1) = 330,241
```

Using the current 8,722,689-parameter tree as the shape basis gives estimated
totals of 8,986,885 and 8,986,881, a difference of four parameters. The
control spends all capacity in executable shared decoding; it carries no
unused candidate parameters. This isolates route-local ownership from raw
capacity.

The graph-MLP control is not repeated because this experiment makes no new
claim that Attention is better than fixed aggregation. It asks whether the
already selected Attention body needs output-route isolation.

## Objective And Initialization

Both arms start from new parameters. Neither inherits the v2 checkpoint or
its optimizer state. The existing v6 Teacher data, scales, and prepared daily
artifacts may be reused; no v6 Teacher regeneration or v5 join is allowed.

The proposed shared objective is the previously tested named-biomass
tendency boundary:

- unchanged one-day named-process anchor;
- cumulative GPP, NPP, growth-respiration, and heterotrophic-respiration
  components at 1/7/30 days;
- named AGB and BGB tendencies at 1/7/30 days;
- total live biomass as exact `AGB + BGB` reporting and acceptance;
- long-horizon maintenance as a no-regression guard, while one-day
  maintenance remains in the process anchor.

Reopening this objective is justified only together with route-local decoding.
Its earlier failure occurred under the universal decoder: the tendency arm
clipped on 93/100 updates, improved maintenance 31.05%, and worsened every
primary validation cell. The new hypothesis is that a dedicated maintenance
decoder prevents most of that unintended output coupling. Running the same
objective on both new arms tests that claim directly.

The old update-1000 tendency coefficient is not reused. One train/train-only
preflight must freeze a coefficient shared by both arms, use disjoint groups,
and require finite nonzero NPP/AGB/BGB gradients with each weighted biomass
median within `[0.1, 10]` times the NPP median. Validation values may not set
the coefficient.

## Admission Before Training

Both arms must pass:

- exact five-route and 181-field coverage with no duplicate or orphan field;
- the four-parameter count difference and executable parameter audit;
- eager/JIT forward, loss, JVP, and VJP on local synthetic data;
- real-v6 forward/loss/gradient on one free GPU allocation;
- PFT permutation, inactive-slot, and explicit parameter-route checks;
- one-day state, mask, discrete, stock, water-budget, and carbon-budget gates;
- explicit proof that inference does not reconstruct 48 forcing records or
  execute a 48-step state scan.

Failure stops before training. These checks may not trigger architecture or
capacity changes.

## Bounded Matched Training

If admission passes, the proposed first architecture screen is:

- 500 optimizer updates per arm from scratch;
- the same 2,000 unique prepared train/train point-years for both arms,
  grouped as four point-years per update and used once per arm;
- 32 one-day anchors and four 30-day windows per update;
- 50-update warmup followed by cosine decay from `3e-4` to `3e-5` by update
  500;
- unchanged fixed 48-window non-test temporal/spatial/joint panel, evaluated
  only at the frozen final milestone;
- recoverable state for failure handling, with promotion eligibility decided
  only by the final gate.

This uses existing prepared data and is a bounded architecture A/B, not final
production training. A pass may justify preparing broader v6 train/train
coverage; a fail may not.

## Scientific Gate

Candidate admission requires all of the following:

- zero numerical, state, mask, discrete, stock, and budget hard failures;
- primary NPP/AGB/BGB/total-live composite at least 5% better than the matched
  wide-shared control and at least 1% better than the update-1000 incumbent;
- each of NPP, AGB, BGB, and total-live aggregate RMSE no worse than either
  comparator;
- at least four of six 7/30-day slice-horizon primary cells better than both
  comparators;
- maintenance aggregate regression at most 20% and every maintenance cell at
  most 30% relative to the incumbent;
- one-day process-anchor aggregate no more than 5% worse than the matched
  control.

Gradient clipping, memory, and throughput are reported but are not sole
promotion gates. Training loss alone cannot pass the candidate.

A pass opens broader-data training design, not 365-day validation directly.
If the candidate passes the hard gates and beats the matched control by the
required 5% but does not beat the incumbent, it is not admitted; that result
supports the route-isolation mechanism while showing that this bounded fit is
not yet scientifically adequate. It requires user review and does not
automatically authorize more updates. Route-local decoding is closed only if
the candidate fails its matched-control comparison after both arms complete
the frozen fit without optimization or numerical failure. Neither outcome
authorizes wider/deeper Transformer, RNN/SSM, cross-day memory, or coefficient
search.

## Alternatives Not Recommended First

1. **Wider or deeper Transformer body.** The current parameter count is not
   small, and 96% of the failed direction is in the decoder. More Attention
   capacity does not target the observed failure.
2. **Day-local RNN/SSM forcing encoder.** It remains technically admissible,
   but the forcing sequence has only five records and no evidence locates the
   failure in temporal encoding.
3. **Cross-day learned hidden state.** This changes restart and Markov
   semantics and is excluded.
4. **Fully separate networks for every output field.** It discards useful
   source-shared representation and scales poorly across 181 fields. Five
   source-route decoders are the smallest evidence-backed partition.

## Review Required

Before implementation, the user must review and accept or reject:

1. retaining the current Attention body;
2. the five route-local decoder candidate versus the width-1,280 shared
   decoder control;
3. from-scratch initialization;
4. reopening named-biomass tendency loss only within this new matched
   architecture mechanism;
5. the 500-update, 2,000-point-year bounded screen and its gates.
