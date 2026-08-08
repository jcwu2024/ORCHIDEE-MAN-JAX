# First Neural Process-Head Candidate v1

Status: **first matched seed completed; comparison closed for practical equivalence without promotion**.

Date: 2026-08-07.

## Decision

The first formal Gate-E neural candidate is
`axial_process_operator_v1`. It enters only through:

```python
bind_daily_operator(
    input_assembler=canonical_assembler,
    process_head=candidate_head,
    retained_tail=canonical_retained_tail_adapter,
)
```

The matched control is `factorized_process_control_v1`. Neither arm inherits
a historical neural class or checkpoint. Both consume the accepted coherent
v6 Teacher release and the same named process targets. The sealed test split
is unavailable to model selection.

## Primary Hypothesis

Holding the native-forcing encoder, tokenization, output constraints,
parameter count, optimizer, batches, seeds, and update budget fixed, explicit
axis-wise process interaction will improve held-out process supervision and
7/30-day free rollout relative to independent token heads conditioned on one
pooled context.

The new mechanism is not a repeat of the rejected flat RNN/MLP or structured
FiLM models. It predicts typed process quantities, preserves physical axes,
uses the accepted conservative updater, and retains the exact daily tail. It
does not predict an anonymous next-state vector.

## Candidate

`axial_process_operator_v1` is a hierarchical axial process Transformer:

- `d_model=192`, six attention heads, and `ffn_width=768`;
- two masked Transformer blocks over the supplied native forcing records;
- four shared axial process blocks over PFT, water/soil, carbon/pool/depth,
  and thermal axes;
- bounded process-query cross-attention rather than global all-token
  self-attention;
- domain adapters and shared axis-aware output decoders;
- float64 parameters and computation for the first admission experiment;
- a declared 4-6 million parameter interval.

Only the supplied 3/5/7-record masked sequence is encoded. The graph may not
reconstruct 48 forcing records or execute a 48-step state scan. PFT identity
embeddings and landpoint IDs are forbidden.

## Matched Control

`factorized_process_control_v1` uses the same forcing Transformer, dimensions,
depth, domain adapters, output decoders, and constrained parameterizations.
Its axial and process-query interactions are replaced by independent token
MLPs plus the same fraction/mask-aware pooled context. Its trainable parameter
count must be within 2% of the candidate.

This control isolates the axial interaction mechanism. It is not the E1
linear plumbing fixture and is not a rejected historical model.

## Data And Target Boundary

Training binds the accepted `pft14-daily-teacher-v6` dataset manifest,
acceptance report, and training statistics. Historical v5 rows and
checkpoints are forbidden.

The raw v6 `fast_day_target` is legal only inside a host-side target adapter.
The adapter decodes it by the frozen Markov contract into named inventory,
thermal, daily-interface, and retained-boundary fields. The anonymous vector
is never returned as a model input or loss target. Supplemental fields retain
their declared axes, units, dtype, and masks.

The supervised loss is a mask-aware, standardized mean squared error within
each label followed by an equal-weight macro mean over labels. Large OK_LEAK
tensors therefore cannot dominate small scientific families solely through
element count. Seven labels remain outside neural supervision: native rain and
snow aggregation plus five exact carbon derivations. The three thermal
tendencies are derived exactly from Teacher endpoints for target construction
but remain bounded neural outputs, giving 40 supervised label groups in
total. No rollout loss or open-ended loss-weight search is used in the first
experiment.

Targets already parameterized on a declared physical range are not divided by
an empirical standard deviation a second time. `unit_interval` outgoing
fractions and `signed_unit` normalized thermal tendencies use loss scale 1.
Unbounded signed and nonnegative physical quantities use frozen train/train
scales. This prevents near-constant bounded targets from acquiring unbounded
weight while preserving equal-label macro weighting.

All train/train normalization moments on a PFT axis are pooled across active
PFT slots and then broadcast back to that axis. Per-slot moments are forbidden:
they encode slot identity and break permutation equivariance before the shared
network weights are applied. This rule covers both semantic state scales and
named output mean/scale arrays.

## Fixed Training Budget

- optimizer: AdamW;
- learning rate: `3e-4`, 500-update linear warmup, cosine decay to `3e-5`;
- weight decay: `1e-5`;
- global gradient-norm clipping: `1.0`;
- batch size: 16;
- maximum updates: 10,000 per arm and seed;
- seeds: `20260807`, `20260808`, and `20260809`;
- checkpoint interval: 250 updates;
- checkpoint selection: lowest declared non-test validation macro process
  loss, with all hard physical gates still required.

The budget may be reduced before a paid run if a measured resource limit
requires it, but it may not be enlarged after observing candidate results.
Any resource-driven revision creates a new manifest version and applies
identically to both arms.

The pre-training tiny-fit is a bounded graph-learnability diagnostic, not a
selection result. Each arm starts from the same initialization and performs
exactly 100 AdamW updates on the same 16-row train/train batch with learning
rate `3e-4`, weight decay `1e-5`, and global gradient clipping at `1.0`. It
passes only if all losses remain finite and final loss is at least 20% below
initial loss. No tiny-fit checkpoint is retained.

## Admission Gates

Before paid training, both arms must pass real-v6 eager/JIT forward, finite
loss, forward/reverse gradients, tiny-fit, PFT permutation, inactive-slot
isolation, cold/ordinary/restart one-day checks, and the executable graph
audit. Water and carbon construction residuals must be at most `2e-12`, with
zero illegal inventories and exact masks/discrete state. The dimensional PFT
permutation comparison uses `atol=1e-10, rtol=1e-12`; inactive-slot isolation
keeps an absolute `1e-10` gate. Both maximum absolute error and maximum
tolerance ratio are reported.

On the non-test validation slices, the candidate must:

- improve macro process NRMSE over the matched control by at least 5%;
- avoid more than 5% regression in any named scientific label family;
- avoid regression in signed daily tendency bias;
- improve the declared 7/30-day key-field macro error by at least 5%;
- avoid more than 10% regression in any key rollout field;
- produce zero hard budget, state-validity, mask, discrete, or restart
  failures.

Failure stops the candidate. Training loss alone cannot authorize more
updates, a wider model, 365-day evaluation, Gate D2, or product acceptance.

## Current Work Boundary

The candidate and control are implemented with an exactly shared parameter
tree of 5,464,321 float64 parameters. The bounded `gln01` real-v6
forward/loss/JVP/VJP, permutation, inactive-slot, budget, retained-tail, and
100-update tiny-fit gates passed.

The full train/train process statistics artifact is accepted. Its
implementation streams coherent base and typed v6
shards, computes all 101 named output moments and independent loss scales,
computes semantic water/carbon/thermal input scales, shares every applicable
PFT moment across active slots, supports deterministic partition merging, and
hash-binds the candidate, dataset, contracts, source inventory, and accepted
base statistics. Local vectorized-versus-single-day, accumulator merge, PFT
sharing, and artifact-tamper tests pass.

Approved `cnmix` job `14506391` completed all eight fit steps over 8,591,565
train/train rows and 23,540 source shards. Its batch failed after 31:22 at the
merge call only: the CLI passed an unsupported `base_statistics` argument.
Commit `9e8d4e9` corrected that wiring without changing any statistic or
rescanning data. The eight retained JSON/NPZ partial pairs then merged and a
separate Python process independently loaded the final artifact on `test01`.
Both audits reported 8,591,565 samples and 23,540 shards. The final JSON
SHA256 is
`9bc2cab43b3279e39a0fc2876ca298b2b371d6890986d44ef035859ca925eaab`;
the NPZ SHA256 is
`addf957afbd8338f4aa250155dcd299d568c1070b644e30279f9bf0ea37f9bf8`.

One formal seed pair then completed. The axial candidate's selection score was
`0.8219892971` and the factorized control's was `0.8433795294`; the candidate
was better overall and on temporal/spatial slices but worse on the joint
slice. The user judged the difference practically immaterial and stopped the
comparison after this pair. Remaining seeds and the pair's 7/30-day rollout
were not run, neither arm is promoted, and no formal three-seed equivalence or
full-protocol rejection is claimed. The controlling record is
[`first_neural_process_head_first_pair_decision_20260808.md`](first_neural_process_head_first_pair_decision_20260808.md).

The active work is the source-backed long-horizon biomass/productivity target
and metric foundation in
[`long_horizon_biomass_productivity_objective_v1.md`](long_horizon_biomass_productivity_objective_v1.md),
followed by an explicit successor-architecture discussion before any new
network implementation or GPU submission.
