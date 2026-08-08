# Successor Process-Head Architecture Options v1

Status: **analysis complete; preferred candidate and matched control proposed;
user architecture decision pending**.

Date: 2026-08-09.

## Decision Boundary

This document compares the admissible successor architecture families and
proposes one bounded experiment. It does not freeze or implement a network.
No checkpoint, GPU run, paid job, 365-day rollout, Gate D2 evaluation, or
product gate is authorized by this analysis.

Every option must enter only through:

```python
bind_daily_operator(
    input_assembler=canonical_assembler,
    process_head=candidate_head,
    retained_tail=canonical_retained_tail_adapter,
)
```

It must use the accepted coherent v6 data only, must not initialize from a
historical neural checkpoint, and must not reconstruct 48 forcing records or
execute a 48-step state scan at inference.

## What The First Pair Established

The first matched pair is useful negative evidence but does not reject
attention. Its candidate/control scores were `0.8219892971` and
`0.8433795294`; the user stopped the comparison for practical equivalence
after one seed. No 7/30-day rollout was run and neither arm was promoted.

The topology, rather than raw model size, contains the clearer bottlenecks:

- the accepted input contains 3,854 continuous state scalars, five native
  forcing records with nine fields per record, two active PFT slots, and
  named static and annual conditions;
- the first architecture made 49 water, 127 carbon, and 56 thermal semantic
  tokens, then collapsed each domain to one mean summary before its process
  queries read it;
- the complete continuous-state, static, and annual vectors were separately
  reduced to four statistics each and combined into one global token;
- 40 label-level queries supplied 101 named output fields containing 38,530
  output scalars;
- the 5,464,321-parameter candidate could optimize stably, so there is no
  evidence that parameter count alone caused the practical tie.

The first candidate did perform attention inside each domain. The important
limitation is that an output query could not directly choose among the
relevant water, carbon, thermal, PFT, forcing-field, or named-condition
tokens. It read four already-pooled domain summaries plus one forcing summary
and one heavily compressed global token. Increasing width while preserving
that information path is not a materially new mechanism.

## Shared Scientific Skeleton

All admissible successors should share the following boundary, regardless of
the neural family selected:

```text
five supplied native forcing records x nine named fields
  + named day-start state/PFT/static/annual tokens
  + exact masks and coordinates
    -> day-local forcing representation
    -> 101 named-field process queries with source-routed inputs
    -> bounded fluxes, transfer fractions, and thermal tendencies
    -> one conservative fast update
    -> canonical retained exact daily tail
    -> next-day state and named productivity/stock diagnostics
```

The 101 field queries may share label-family embeddings and decoders, but one
40-label query may not be the only information path for fields with different
source owners. Scalar output elements still use shared axis-aware decoders;
38,530 independent dense output heads are neither necessary nor admissible.

The successor must not use the full continuous state as a duplicate opaque
`other` vector. It should tokenize named state leaves once, preserve the
already accepted water/carbon/thermal and PFT axes, and add only source-needed
state/static/annual leaves. Landpoint ID, PFT ID, Fortran PFT number, and
teacher targets remain forbidden features.

### Source-Routed Input Families

The implementation must freeze a machine-readable output-to-input adjacency
ledger before initializing weights. Its first-level source families are:

| Output family | Admitted direct context |
| --- | --- |
| GPP/photosynthesis | named radiation/meteorological forcing, PFT vegetation/LAI/biomass/traits, canopy-water and thermal context, explicit `vcmax25` factor |
| Maintenance respiration | PFT biomass/traits, named air/root thermal context, state `t2m_longterm`, explicit maintenance polynomial |
| Water transfers | precipitation/meteorological forcing, day-start canopy/snow/soil-water inventories, PFT surface structure, thermal context |
| Energy and thermal tendencies | radiation/meteorological forcing, thermal/snow/water state, PFT surface structure |
| Litter and soil carbon transfers | day-start litter/POC/DOC inventories, source litter inputs, soil moisture/temperature, named soil/static factors and water-transport context |

These are broad source families, not permission for global all-token
attention. The field-level ledger must cite the existing label inventory,
Fortran owner, or an accepted trace. An edge with no source basis is excluded,
and a source-required input absent from v6 fails closed instead of being
approximated.

### Parameter Routing

The full six-column `pft_parameters` array remains an operator-level carrier,
not an encoder feature matrix:

- `alloc_min` goes only to retained exact allocation;
- `residence_time` goes only to retained exact mortality;
- `vcmax25` goes to the retained `vmax_step` and to the named fast GPP factor;
- `maint_resp_slope_a/b/c` go only to the named fast maintenance factor.

The proposed fast parameterization is explicit:

```text
GPP = nonnegative learned environmental response * nonnegative vcmax25

tl = named state t2m_longterm - 273.15
slope = c + b*tl + a*tl^2
maintenance = nonnegative learned base response
              * source-shaped nonnegative temperature multiplier(slope)
```

The successor may learn environmental and compartment responses, but it may
not infer `vcmax25`, the maintenance coefficients, or `t2m_longterm` through
an opaque embedding. In particular, the first plumbing head's freely learned
temperature proxy is not sufficient for the successor: the polynomial must
receive the named source `t2m_longterm` state. The exact source-shaped
temperature multiplier and its units must be audited before implementation so
that later parameter derivatives have the intended meaning.

## Architecture Families

### A. Broad Process Transformer

Information flow:

```text
forcing-field/time tokens -> small forcing Transformer
named state tokens -> domain encoders
field queries -> broad domain cross-attention -> output decoders
```

This retains attention and removes the four-summary bottleneck. It has no
cross-day hidden state and can represent nonlocal interactions. Its main risk
is that broad domain masks still allow statistically convenient but
source-unjustified dependencies. A matched control would replace
content-dependent attention by a parameter-matched token MLP plus fixed
domain aggregation.

Assessment: admissible, but not preferred. It changes the access path without
using the strongest scientific structure already available.

### B. Day-Local RNN Or SSM With Process Heads

Information flow:

```text
five forcing records -> GRU/LSTM/SSM used only within the day
final/all day-local states + named state tokens -> factorized process heads
```

This is not excluded by the historical RNN/MLP failure. That model compressed
3,854 states to width 128, used a recurrent forcing encoder, predicted a
superseded direct-state boundary, and drifted in rollout. The evidence does
not isolate the short RNN as the failure cause.

At a sequence length of five, however, an SSM's long-sequence advantage is
irrelevant and an RNN's sequential inductive bias is modest. A cross-day
hidden state is not proposed: it would add restart state, change the accepted
Markov meaning, complicate PFT permutation and inactive-slot semantics, and
create an undeclared parameter-gradient path. The full physical day-start
state remains the only cross-day memory.

A matched control would use the same process heads and a parameter-matched
order-aware temporal MLP/DeepSets encoder. Assessment: legal backup for the
forcing encoder, but it does not by itself solve process/state routing.

### C. Source-Masked Physical Graph MLP

Information flow:

```text
named input nodes -> source-typed fixed aggregation
source-adjacent process nodes -> edge-type MLP messages
field nodes -> shared axis-aware output decoders
```

This option directly addresses the pooling problem and makes every process
edge auditable. It is efficient, naturally supports hard masks, and is a good
fit for conservation and retained-tail ownership. Its weakness is that fixed
aggregation may blur which forcing time or soil/pool coordinate matters under
a particular state, even when the correct nodes are available.

A matched control would remove inter-process messages while retaining the
same direct source tokens and parameter budget. Assessment: strongest simple
architecture and the preferred control/fallback.

### D. Source-Routed Process Attention Hybrid

Information flow:

```text
45 forcing field/time tokens -> two day-local attention blocks
named state/PFT/static/annual tokens -------------------------+
                                                              |
101 field queries -- source mask --> direct cross-attention <--+
101 field queries -- source graph --> bounded message passing
                  -> shared constrained decoders
```

Attention is used only along admitted edges. It supplies data-dependent
weighting over the relevant forcing times, soil layers, PFTs, and carbon
compartments; the source graph supplies the allowed context. No domain is
reduced to one summary before the field query reads it. There is no global
all-token self-attention and no cross-day learned hidden state.

This is Transformer-style attention inside a structured process head, not a
generic language-model Transformer. Its relevant attention benefit is
selective access to named scientific context, not preservation of a long text
history. Assessment: preferred successor candidate.

## Proposed Candidate And Matched Control

The preferred candidate is provisionally named
`source_routed_process_attention_v2`. The matched control is provisionally
named `source_routed_graph_mlp_control_v2`.

They share:

- the same 45 forcing field/time tokens and named state token registry;
- the same output-to-input source mask and process graph;
- the same 101 field queries, embeddings, width, depth, constrained decoders,
  parameter routing, optimizer batches, seeds, and loss;
- the same conservative updater and canonical retained tail;
- a trainable parameter-count difference no larger than 2%.

The candidate uses content-dependent multi-head attention along allowed
input and process edges. The control replaces those operations with
token-local MLP transforms, deterministic masked aggregation, and edge-type
MLP messages. Unused attention parameters may not be carried merely to claim
a match; the control must spend its capacity on executable local/edge MLPs.

The proposed capacity envelope is 7-10 million float64 parameters, with a
working width of 256, eight heads for the attention arm, two forcing blocks,
and three source-routed process blocks. This is large enough to remove the
known routing bottlenecks and modestly increase capacity from 5.46 million,
without treating size as the hypothesis. The exact count and measured memory
must be frozen after a shape-only initialization and before any fit.

The primary hypothesis is:

> Given identical named tokens, source edges, capacity, parameter routing,
> conservative update, retained tail, and long-horizon objective,
> content-dependent weighting along source-admitted edges improves 7/30-day
> productivity and carbon-stock accuracy over fixed graph aggregation.

This control tests attention where it is scientifically relevant. It does not
repeat the first pair's comparison between domain attention followed by four
summaries and one pooled-context factorized head.

## Long-Horizon Training Objective

The one-day equal-label process loss remains the anchor with coefficient
`1.0`. A 30-day differentiable daily rollout supplies the already frozen
`H = 1, 7, 30` cumulative GPP/NPP/respiration and terminal
AGB/BGB/litter/POC/DOC/ecosystem-carbon components. A 30-step daily scan is
required for this objective and is distinct from the forbidden 48-step
half-hour state scan.

Final coefficients are not selected in this options document because they
depend on the selected architecture's train/train gradients. After user
selection and local implementation, exactly one fixed train/train microbatch
preflight should set them as follows:

1. Measure the gradient norm of the one-day anchor and each of the four frozen
   long-horizon components at the same initialization.
2. Scale cumulative-flux MSE and terminal-stock MSE toward 10% of the anchor
   gradient norm each; scale their signed-bias components toward 15% each.
3. Clip every coefficient to a predeclared finite interval and fail the
   preflight if a component has a zero/nonfinite gradient or requires a clip
   boundary. The interval must be frozen in the candidate contract.
4. Record the achieved individual and combined gradient ratios. Do not rerun
   the coefficient calculation after seeing validation or rollout metrics.

This gives long-horizon science a nominal auxiliary gradient budget of 50%
of the one-day anchor while deliberately emphasizing systematic bias. It is a
bounded initialization calibration, not validation-driven loss tuning.

## Bounded Admission And Stop Gates

Before any training submission, both arms must pass locally:

- exact parameter count, token/mask/edge registry and executable-graph audit;
- eager/JIT forward, finite one-day and 1/7/30-day losses, JVP and VJP;
- active `vcmax25` and maintenance-factor perturbation/gradient paths;
- zero neural gradient to `alloc_min` and `residence_time`, with their retained
  exact effects still active;
- PFT permutation, inactive-slot isolation, and variable legal `n_pft`;
- cold, ordinary, and restart one-day state/mask/discrete checks;
- water/carbon construction residual at most `2e-12`, no negative declared
  inventories, and no nonfinite defined state.

The first GPU step is a no-checkpoint real-v6 forward/loss/gradient smoke on
`gln01`. Only after that passes may one matched seed receive a bounded small
fit. The small fit must freeze its update count and resource envelope before
submission; it may not expand because either training curve looks promising.

The first 7/30-day screen uses only non-test temporal, spatial, and joint
slices. For either arm to remain scientifically viable it must have zero hard
failures and absolute normalized signed bias no larger than `0.10` for every
frozen key productivity and stock field. The attention hypothesis requires at
least 5% improvement over the graph control in the frozen long-horizon macro
objective, no regression in signed productivity/stock bias, no more than 5%
regression in GPP, NPP, AGB, BGB, or ecosystem carbon, and no more than 10%
regression in any other frozen key field.

An improvement below 5% is treated as practical equivalence for this bounded
mechanism screen, not as a formal statistical equivalence claim. In that case
no extra seed is run to rank a marginal difference. The simpler graph control
may be retained only if it independently passes every absolute and hard gate;
otherwise both arms stop. Training loss alone cannot open 365-day validation,
long-held-point evaluation, Gate D2, or product acceptance.

## Recorded Selection And Implementation Status

The user selected `source_routed_process_attention_v2` against the exactly
matched `source_routed_graph_mlp_control_v2`. The graph arm remains the
scientific control and possible simpler fallback only if it independently
passes every absolute gate; a day-local GRU remains an unopened forcing
ablation. Cross-day neural memory remains excluded.

The machine adjacency, exact configuration, coefficient interval, small-fit
budget, and gates are frozen in
`manifests/coarse_graining/successor_process_head_candidate_v1.json`. The
local implementation and synthetic acceptance are recorded in
`successor_process_head_local_foundation_20260809.md`. Real-v6 canonical-tail
smoke, bounded fit, and rollout comparison remain pending.
