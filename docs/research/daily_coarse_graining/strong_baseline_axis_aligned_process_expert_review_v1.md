# Strong Baseline Axis-Aligned Process-Expert Review v1

Status: **proposal only; user architecture review required before
implementation**.

Date: 2026-08-16

## Decision

The next main task is not another bounded comparison around the rejected v2
checkpoint. It is to design one capable daily-operator baseline and qualify it
against absolute scientific targets.

The route-local-decoder A/B remains an unimplemented diagnostic hypothesis,
but it no longer controls current work. The existing update-1000 checkpoint
is historical diagnostic evidence, not an incumbent that defines success. A
matched control or ablation is deferred until a model independently passes
the strong-baseline gates.

No network code, manifest, checkpoint, GPU run, paid job, or architecture
promotion is authorized by this review.

## Why The Existing Network Is Not A Strong Baseline

### Training coverage

The accepted v6 `train/train` product contains:

- 23,540 point-years;
- 8,591,565 daily transitions.

The first 500-update fit consumed only:

- 500 point-years, or 2.12% of the available point-years;
- 8,000 one-day anchors, or 0.093% of available daily transitions;
- 1,000 thirty-day windows, or 0.349% of nominal day positions.

Later stages increased the number of distinct point-years, but they remained
fixed-update screens rather than complete shuffled epochs over the released
training product. Their results cannot establish what a sufficiently trained
model family can achieve.

### Information bottleneck

The accepted extended schema has 181 fields and 39,371 scalar outputs. The
current route distribution is:

| Source route | Fields | Scalar outputs |
| --- | ---: | ---: |
| water | 86 | 949 |
| energy | 42 | 811 |
| carbon fast GPP | 9 | 55 |
| carbon fast maintenance | 2 | 38 |
| carbon other | 42 | 37,518 |

The current architecture completes all source and process Attention at one
query per field. Only after that aggregation does it expand a field into its
scalar coordinates by adding static axis and position embeddings. Therefore
all 8,512 elements of a large carbon-transfer field, for example, start from
the same already-aggregated field representation. An output coordinate cannot
then query the corresponding input depth, carbon pool, PFT, or DOC pool.

Several process-static arrays are also reduced to four statistics before the
process body. Their layer- or pool-local values are unavailable to a local
output element. A wider universal decoder or a five-way decoder split cannot
restore information that was already aggregated away.

### Optimization and objective evidence

The old runs also combine incomplete data coverage with:

- a historically incorrect rollout bias reduction that was later repaired;
- severe nonnegative-output initialization saturation that was later
  repaired;
- frequent global clipping under a fixed norm of 1.0;
- one-day and 30-day objectives applied before one-day mapping competence was
  established;
- a universal decoder carrying 96.15% of the stopped update direction and a
  maintenance output dominating the failed NPP/BGB nonlinearity.

These facts justify a new baseline foundation. They do not justify ranking
small changes around the old checkpoint.

## Proposed Strong Baseline

Provisional name: `axis_aligned_process_expert_operator_v1`.

The baseline remains a one-day Markov operator:

```text
day-start state + five native forcing records + parameters/static conditions
  -> axis-preserving source encoders
  -> process-owned experts with bounded graph messages
  -> axis-aligned process heads
  -> conservative updater
  -> canonical retained exact daily tail
  -> next-day state
```

It retains the canonical
`bind_daily_operator(input_assembler=canonical_assembler,
process_head=baseline_head,
retained_tail=canonical_retained_tail_adapter)` boundary. It has no landpoint
identity input, cross-day neural hidden state, reconstructed 48-record forcing,
or 48-step state scan.

### 1. Native-forcing encoder

- Encode the five supplied record-by-field observations with their time,
  duration, predecessor flag, field identity, and mask.
- Use two width-256, eight-head day-local Attention blocks.
- Produce both record-local tokens and process-specific forcing summaries.
- Do not expand or interpolate the five records.

Attention is a component of the baseline, not the experiment being tested.

### 2. Axis-aligned source encoders

Preserve dynamic coordinates for every axis that is consumed locally by an
output:

- PFT and plant part;
- hydrology, thermal, deep-soil, and snow depth;
- soil tile and nonbiological tile;
- carbon, litter, DOC, and soil-carbon pool;
- element, litter level, LAI class, and other declared contract axes.

Small arrays may use one token per valid element. Large tensors use shared
axis encoders and factorized axial blocks. Layered static quantities such as
`z_soil`, `rprof`, `litterfrac`, `cue`, `frac_carb`, and `sro_bottom` must
remain aligned with their applicable axes; replacing them with global
mean/standard-deviation/minimum/maximum summaries is forbidden where a local
head consumes the corresponding axis.

Weights remain shared along PFT and repeated physical axes. PFT permutation,
inactive-slot isolation, and variable legal `n_pft` remain structural gates.

### 3. Process experts

Use seven independently parameterized experts:

1. canopy exchange and fast GPP;
2. surface energy;
3. hydrology and snow;
4. soil thermal state;
5. plant maintenance respiration;
6. litter, particulate carbon, and DOC transfers;
7. daily interface and fast-state finalization.

Each expert provisionally uses width 384, eight heads, and three residual
cross-Attention/feed-forward blocks. It receives its source-admitted dynamic
axis tokens and owns its complete representation path; experts do not share a
universal hidden stack or output decoder.

Two directed process-graph message rounds exchange only bounded summary
messages. The graph admits the physical couplings needed by canopy, water,
energy, temperature, maintenance, and soil carbon without giving every output
unrestricted access to every expert parameter. Message gates are residual and
initialized near zero so that each expert begins as a learnable local path.

The exact field-to-expert ledger must cover all 181 fields once and only once
before initialization. The source masks remain a restriction, not merely
documentation.

### 4. Axis-aligned output heads

There is no universal scalar decoder.

For an output element, the owning head combines:

- its expert context;
- the dynamic token for each applicable coordinate;
- the corresponding local day-start inventory or state feature;
- field identity and source-backed static factors;
- gated pairwise axis interactions where the source process requires them.

Thus a carbon transfer at one depth/PFT/pool is conditioned on that local
depth/PFT/pool state rather than only on a global field query. Heads share
weights within a scientific label family, but GPP, maintenance, water, energy,
and soil-carbon families do not share a final MLP.

Existing unit-interval, nonnegative, signed, signed-unit, status-logit,
day-start-delta, mask, and exact parameter-route semantics remain unchanged.
`vcmax25` and maintenance factors retain their explicit fast-process roles;
`alloc_min` and `residence_time` continue to bypass the neural encoder and act
in retained exact processes.

### 5. Capacity envelope

The provisional target is 35-50 million float64 parameters:

- shared forcing and axis encoders: approximately 4-7 million;
- seven process experts and graph messages: approximately 27-38 million;
- axis-aligned family heads: approximately 4-7 million.

The implementation contract must compute the exact executable count and stay
below 60 million parameters. The range is a capacity envelope, not a target
to match another model. It may be adjusted once during local shape design,
before any real training result is observed.

## Strong-Baseline Data Path

No v6 Teacher regeneration and no v5 join are allowed. Training reads the
accepted v6 base and typed shards directly.

The old per-update prepared-artifact path is not suitable for full-data
training: 500 artifacts occupied about 16.7 GB, and repeated preparation can
leave an allocated GPU idle. The strong baseline instead requires:

- a small deterministic epoch/index plan, not duplicated training payloads;
- shard-aware base-plus-typed loading with bounded host caching;
- asynchronous CPU assembly and prefetch ahead of GPU execution;
- one manifest/release admission before the run, not complete per-update
  rehashing on the paid GPU path;
- direct day sampling for one-day pretraining;
- on-the-fly contiguous window slicing for rollout training;
- measured hot-loader throughput before any paid submission.

Every one-day epoch uses all 8,591,565 train/train transitions exactly once in
a new seeded order. A batch mixes landpoints, years, seasons, and process
regimes instead of taking many correlated days from one point-year.

## Training Curriculum

### Phase 0: learnability qualification

Before full training:

1. Memorize all valid days from one train/train point-year. Require at least
   99% reduction in the complete one-day loss and normalized RMSE at most
   0.05 for GPP, maintenance respiration, NPP, AGB tendency, and BGB tendency.
2. Fit a frozen 64-point-year train-only set. Require at least 90% loss
   reduction, finite gradients, and zero hard failures.

These are capacity and optimization tests, not model selection. Failure means
the architecture, normalization, data mapping, or optimizer is not ready; it
blocks full training.

### Phase 1: complete one-day pretraining

- Train for two complete shuffled epochs over all 8,591,565 train/train days.
- Use an effective one-day batch of 256, assembled from multiple point-years.
- Use microbatching and exact gradient accumulation if the device cannot hold
  the effective batch.
- Train the 181 named process fields plus downstream one-day GPP, NPP,
  maintenance, AGB tendency, BGB tendency, and total-live tendency.
- Preserve equal macro weight over scientific label families so the 37,518
  carbon-other scalars cannot dominate by element count.

Only after the one-day absolute gate passes may rollout fine-tuning begin.

### Phase 2: rollout curriculum

1. One seven-day rollout epoch with one rotating window from every train/train
   point-year.
2. Two thirty-day rollout epochs with different deterministic window starts.
3. Each effective update retains 256 diverse one-day anchors and uses an
   effective 16 seven-day or 8 thirty-day windows through gradient
   accumulation.

The loss has four equal scientific groups, all normalized from frozen
train/train statistics:

1. named one-day process fidelity;
2. cumulative GPP/NPP productivity;
3. maintenance, growth, and heterotrophic respiration;
4. AGB, BGB, and total-live biomass tendencies and terminal stocks.

Within each group, normalized squared error and signed-bias squared are
reported separately and equally macro-averaged. There is no validation-driven
coefficient search, minimum-norm gradient balancing, or reuse of the rejected
staged-v3 scalarization.

### Optimizer qualification

- Start from new parameters; no historical checkpoint or optimizer state.
- Use AdamW with a provisional peak learning rate of `1e-4`, 2% warmup, and
  cosine decay to `1e-5` within each curriculum phase.
- Exclude normalization scales and biases from weight decay.
- Freeze the global clip threshold from the 99th percentile of 32 diverse
  train-only preflight gradient norms; do not inherit the arbitrary norm 1.0.
- Report clipping frequency. Persistent clipping after warmup is a failed
  optimization qualification, not evidence of useful training.

## Absolute Scientific Gates

Old-model and control comparisons do not enter these gates.

### One-day gate

On each non-test temporal, spatial, and joint validation slice:

- GPP and maintenance normalized RMSE at most 0.25;
- NPP, AGB tendency, BGB tendency, and total-live tendency normalized RMSE at
  most 0.50;
- absolute normalized signed bias at most 0.05 for every primary quantity;
- at least 90% of all supervised label families have normalized RMSE at most
  0.50, with no family above 1.0;
- zero numerical, mask, discrete, stock, water-budget, and carbon-budget hard
  failures.

### Seven/thirty-day gate

For all six slice-by-horizon cells:

- cumulative GPP and NPP normalized RMSE at most 0.50;
- AGB, BGB, and total-live biomass tendency normalized RMSE at most 0.50;
- absolute normalized signed bias at most 0.05 for each primary quantity;
- terminal AGB, BGB, and total-live relative error at most 5%, with a
  train-scale absolute guard for low-biomass cases;
- zero hard failures and no persistent same-sign day-by-day drift.

Passing this gate opens 365-day validation. It is not yet final success.

### 365-day and held-out gate

Final scientific qualification requires, on all declared non-test slices and
later long-held points:

- absolute relative bias at most 5% for annual cumulative GPP and NPP;
- relative RMSE at most 10% for annual cumulative GPP and NPP;
- terminal AGB, BGB, and total-live relative error at most 5%, with the
  low-biomass absolute guard;
- accepted restart-split equivalence, parameter routes, state validity, and
  water/carbon budgets.

Only after these absolute gates pass should the project freeze the model,
design explanatory controls/ablations, open Gate D2, or inspect the sealed
test split.

## Required Review Before Implementation

The user must review and accept or revise:

1. replacing the route-local A/B with one absolute-capability baseline;
2. the seven process experts and bounded graph-message mechanism;
3. dynamic axis-aligned encoding and output heads;
4. the 35-50 million parameter envelope;
5. full v6 epoch-level one-day pretraining followed by 7/30-day curriculum;
6. the proposed absolute one-day, 7/30-day, and 365-day gates.

After acceptance, the next implementation step is only the field-to-expert
and axis-ownership manifest plus local synthetic shape tests. Paid training
requires a later, separate Explore1000 resource and cost approval.
