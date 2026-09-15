# FP64 source inventory for mixed daily/T7/T30 training

## Scope and conclusion

This is a bounded source audit of the active fresh, full-network mixed-learning
path at repository HEAD `fa8a63d96c1ec09d747032aebbc0fd614291f2d9e`.
It covers `daily64/micro8`, native checkpointed T7 and T30 reverse AD, and the
shared one-day operator. It changes no precision policy, model, data, or
training result. The separate `fp64_trace_audit_20260915.md` owns trace
completeness and the primitive measurement protocol.

The most important result is narrower than "the model is FP64":

- all 109,464,375 trainable parameter elements are FP32;
- the dense projections, FFNs, layer norms, masked source summaries, graph
  messages, and the conditioned GPP potential attention are intended FP32;
- the shared ordinary attention implementation historically promotes its FP32
  QK score to FP64 by dividing by the NumPy FP64 result of
  `np.sqrt(float(head_width))`; softmax and the weight/value contraction then
  remain FP64 until the output projection casts back to FP32;
- one active masked neural family pooling operation is explicitly FP64 and
  casts its result straight back to FP32;
- physical output transforms, reconstruction, conservation, retained STOMATE,
  and all active training losses/reductions are intentionally FP64.

Thus FP64 on the neural path is real and avoidable in two well-isolated places.
It is not evidence that all 936 statically counted recurrent FP64 matrix-result
operations are large, nor that they determine wall time. Physical FP64 should
not be globally disabled.

## Active execution path

The runner constructs the route as follows:

1. `scripts/hpc/run_mixed_learning.py:202-218` builds the fresh head, the
   one-day objective, and `jax.value_and_grad(component.checkpointed_loss)`.
2. `scripts/hpc/run_fresh_start_one_day.py:18-75` assembles the mechanistic
   target-oriented head and initializes every parameter from the declared seed.
3. `daily_operator/fresh_start.py:26-65` composes distribution-aware
   maintenance, structured GPP, carbon feedback, and conditioned potential
   attention on the target-oriented Attention backbone.
4. `daily_operator/rollout_loss.py:74-126` executes T7/T30 as `lax.scan`; the
   checkpointed route rematerializes the one-day transition during reverse AD.
5. `daily_operator/model.py:70-109` runs neural prediction, the conservative
   water/carbon/thermal update, and the canonical retained daily tail.

The profile's static lowering counts are:

| Component lowering | FP32 matrix results | FP64 matrix results |
| --- | ---: | ---: |
| daily micro8 gradient | 7,816 | 725 |
| T7 checkpointed gradient | 8,924 | 936 |
| T30 checkpointed gradient | 8,924 | 936 |

These are counts of StableHLO `dot_general`/convolution result types in the
lowered programs. Equal T7/T30 counts are consistent with a shared scan body;
they are not horizon-scaled runtime operation counts. They disclose neither
logical dimensions nor FLOPs nor elapsed-time share. The measured profile is
14.09 seconds/update, with daily/T7/T30 loss-gradient calls taking 72.4% and
the separate recurrent validation/check work another 16.1%; see
`mixed_learning_performance_profile_20260914.md`.

## Exact neural promotion

### Shared attention score path

`daily_operator/strong_baseline_architecture.py:_attention` projects Q, K, and
V through `_linear`, whose input and FP32 parameters force FP32 matrix results
(`:856-857`). Historically, `:887` then used:

```python
logits = einsum(FP32_query, FP32_key) / np.sqrt(float(head_width))
```

With JAX x64 enabled, the NumPy scalar is FP64. A local ORCJAX dtype probe
independently gives:

```text
QK dot: float32
QK / NumPy sqrt scalar: float64
softmax: float64
softmax weights @ FP32 V: float64
```

Casting the scale to the QK result dtype keeps all four values FP32. This is a
dtype promotion, not a deliberate numerical algorithm: the final `_linear`
immediately casts the FP64 attention result back to FP32.

The shared function is used by all of these active paths:

| Caller | Source | Source-derived logical size/depth |
| --- | --- | --- |
| native-forcing self-attention | `strong_baseline_architecture.py:958` | 2 blocks; 45 tokens (5 records x 9 fields), width 256, 8 heads |
| factorized axial self-attention | `strong_baseline_architecture.py:1003-1010` | 1 block; leading groups 7 experts x 18 axes, padded token length 32, width 384, 8 heads |
| seven source-routed experts | `strong_baseline_architecture.py:1306-1315` | 7 x 3 blocks; query length 1, bank length 393 + 140 = 533, width 384 |
| target domain refinement | `target_oriented_attention.py:350-384` | 12 blocks total across 7 domains; query length 1, bank length 393 + 140 + 7 = 540, width 384 |
| science cross/mixer | `target_oriented_attention.py:390-431` | 2 cross blocks (5 by 7) plus 2 mixer blocks (5 by 5), width 384 |
| primary coordinate heads | `target_oriented_attention.py:479-493, 549-565` | 2 fields, 26 coordinates total, 3 blocks per field, width 384 |
| state/budget coordinate heads | `target_oriented_attention.py:479-493, 566-580` | 93 fields, 3,682 coordinates total, 2 blocks per field, width 384; bank length varies with field axes |
| maintenance specialist | `target_oriented_distribution_aware_maintenance.py:218-245` | 2 blocks; 24 queries, 419 bank tokens (393 source + 2 PFT + 24 coordinates), width 768 |
| structured-GPP forcing and branches | `target_oriented_structured_gpp.py:350-370, 526-568` | 2 forcing blocks plus 2 limitation and 1 correction blocks, width 384 |
| carbon-feedback specialist | `target_oriented_coupled_gpp_carbon_feedback.py:344-379` | 2 blocks; 2 PFT queries over 2,386 masked carbon-state coordinates, width 384 |

Expanding the frozen block counts and field inventory gives a source upper
bound of 241 ordinary shared-`_attention` invocations in one model-day
transition: 2 forcing, 1
axial, 21 source-expert, 12 domain-refinement, 4 science-token, 6 primary-field,
186 state-field, 2 maintenance, 2 structured-forcing, 3 unconditioned
structured-GPP branch, and 2 carbon-feedback invocations. QK is computed in
FP32 and only its scaled output is promoted; therefore each surviving primal
call has at most one FP64 dot, the weights/V contraction, before AD introduces
derivative contractions. This source inventory can explain why the static
FP64-dot inventory is neural-heavy, but 241 is not an executed-call claim:
specialist replacement and compiler dead-code elimination may remove parent
work. It also does not assign FLOPs or time because many expressions have query
length one and several are batch-shaped.

The exact base context shape was constructed locally from the source-owned unit
fixture, without training: 20 source families, `source_tokens=(393,384)`, two
compact PFTs, 257 state-field tokens, 8 discrete-field tokens, 30 static-field
tokens, 1 annual token, 49 process-static tokens, 45 forcing tokens, and one
calendar token. The 18 axial extents sum to 140 and have maximum extent 32.

The two conditioned GPP potential blocks are the exception. They use
`bounded_cosine_attention` in `target_oriented_structured_gpp.py:466-523`,
which explicitly casts the temperatures and normalization epsilon to FP32.
Their QK, softmax, and value contraction already remain FP32. The precision
candidate should not alter this path.

The same NumPy-scale pattern exists in older `neural_architecture.py:457` and
`successor_architecture.py:552`. Those implementations are not called by the
fresh head assembled in `fresh_start.py`; changing them would not establish a
speedup for the active mixed route.

### Masked source-family pooling

`target_oriented_attention.py:_source_family_bank` (`:308-326`) is sourced once
for each of seven experts on every model day. It expands a `(20,393)` selection
mask across the 384-wide FP32 source tokens, multiplies and reduces in FP64,
then casts the `(20,384)` result back to FP32. This is about 3.02 million
mask/token elements per expert bank, or 21.13 million across the seven sourced
banks before any compiler elimination. It is a reduction, not necessarily a
BLAS matrix product in the lowering.

This pooling has no physical-unit output and no conservation role. An FP32
implementation is a low-sensitivity candidate, but it must be timed separately:
the materialized broadcast may be limited by fusion or memory traffic, and a
source count does not prove runtime cost. The identical FP64 pattern in
`hierarchical_coordinate_attention.py:210-230` belongs to an older head and is
not active in this fresh route.

### Full-tree reporting norms

`scripts/hpc/run_mixed_learning.py:227-229,549-570` casts every gradient leaf
to FP64, squares it, and reduces it to report the daily, T7, and T30 component
norms. With 109,464,375 parameter/gradient elements, this sources three FP64
conversion/square/reductions over 328,393,125 element visits per update. These
norms do not form the optimizer update. The actual clip and AdamW path computes
its combined norm independently in FP32 in
`target_oriented_training.py:385-421`.

This is a low-science-sensitivity candidate: use FP32 reporting norms, compute
the expensive component norms less frequently, or retain an occasional FP64
audit norm. It is not yet an isolated time share. The profile combines gradient
accumulation/norm/merge, AdamW, and tree-finite checks into 0.542 seconds/update
(3.7%), with AdamW alone about 0.069 seconds. Therefore 3.7% is an optimistic
upper bound for the whole category, not the gain from changing these norms.

## FP64 outside the neural core

### Moderate sensitivity: candidates only after attention

1. **Normalized output to physical units.** `output_transform.py:21-96` casts
   all 39,006 learned scalar coordinates per day, plus means and scales, to
   FP64 before sigmoid/softplus/tanh and delta reconstruction. These nonlinear
   transforms feed recurrent state. FP32 could save elementwise bandwidth, but
   tails, near-boundary unit intervals, large DOC scales, and T30 recurrence
   make this substantially riskier than the attention score change. The primal
   source volume is 2,496,384 learned coordinates for one daily64 update,
   273,042 for one T7 rollout, and 1,170,180 for one T30 rollout, before AD
   rematerialization or compiler fusion; these counts are not time shares.

2. **One-day process and state losses.** `mechanistic_output_loss.py:86-237`
   and `target_oriented_training.py:614-866` form masked residuals, squared
   residual arrays, counts, sums, and bias terms in FP64. A future hybrid could
   evaluate well-scaled residuals in FP32 while retaining an FP64 final
   accumulator, but changing the loss changes the optimizer direction and
   requires a fresh complete-loss gradient/update admission.

3. **Recurrent science losses.** `recurrent_science_loss.py:102-235` converts
   predicted and target trajectories to FP64, then computes daily/cumulative
   flux, endpoint stock, and stock-trajectory losses and counts in FP64.
   The complete continuous carry has 3,854 coordinates, hence at most 26,978
   T7 or 115,620 T30 carried state values before the trajectory extractor
   selects and aggregates its named science fields. Exact loss-array element
   counts then depend on each extracted metric and mask. These reductions are
   scientifically consequential and should not be bundled with an attention
   experiment.

4. **GPP physical factorization.** `target_oriented_structured_gpp.py:730-742`
   casts the bounded neural potential/limitation/correction and physical scale,
   PFT traits, fractions, and vcmax factorization to FP64. This lies directly
   on the current GPP science target and recurrent feedback path; retain it in
   the first optimization.

### High sensitivity: retain FP64

1. **Conservative inventory update.** `conservative_update.py:32-101` uses
   FP64 water/carbon inventory vectors and dense destination-share matrices,
   explicitly tests internal cancellation before large totals cancel, and
   supplies water/carbon budget residuals with a `2e-12` absolute gate. This is
   the wrong first target for FP32.

2. **Mechanistic boundary reconstruction.** `mechanistic_output_reconstruction.py`
   performs support normalization, hydrology stress, thermal diagnostics and
   coefficients, canopy storage/status reconstruction, and DOC baselines in
   FP64 (`:58-176, 357-566, 568-934`). It includes the 3,136-coordinate DOC
   diffusive interface and reconstructs exact/derived daily boundary fields.
   Precision changes here need process-specific parity and recurrence evidence.

3. **Canonical retained tail.** `daily_operator/canonical_retained_tail.py:201-460`
   packs the FP64 fast boundary and runs
   `canonical_retained_tail_dynamic_transition`; the latter invokes the
   source-backed daily STOMATE carbon lifecycle in
   `research/daily_coarse_graining/canonical_retained_tail.py:198-380`.
   Retained state has 3,854 continuous coordinates in the source-owned fixture.
   This path is the Fortran-provenance physical tail, not neural overhead.

4. **Retained physical matrix contractions.** Daily litter/fuel routing uses
   four FP64 `einsum` contractions in
   `jax_orchidee/stomate/carbon_kernels.py:905-908,1073-1082`. They operate on
   physical PFT/plant-part/litter/element axes and are small for the one-point
   PFT14 route compared with the repeated 384/768-wide neural attention. The
   final roughness/albedo reconstruction also contains small FP64 physical
   contractions. Static dot counts must not be used to assign their time.

## Candidate order

| Priority | Change | Sensitivity | Reason |
| --- | --- | --- | --- |
| 1 | typed FP32 scale in active shared `_attention` only | low | removes an accidental promotion; inputs, parameters, and output boundary are already FP32 |
| 2 | reporting-only component norms in FP32 or less often | low | three full-tree FP64 reductions; actual clipping already uses a separate FP32 norm |
| 3 | active `_source_family_bank` reduction in FP32 | low | pure neural pooling followed by an existing FP32 cast; time benefit still unmeasured |
| 4 | normalized loss residual arrays/reductions, one component at a time | moderate | potentially broad arrays, but changes the declared scalar and optimizer direction |
| 5 | normalized-output physical transform | moderate to high | broad elementwise path, but nonlinear boundaries feed T30 recurrence |
| 6 | GPP factorization, conservation, mechanistic reconstruction, retained tail | high | physical/scientific ownership and strict budget/parity gates |

Do not combine precision candidates in the first A/B. Otherwise a result cannot
attribute either numerical change or speedup. Do not use global x64 disablement.

## Numerical and performance gates

For each precision candidate, keep parameters, masks, examples, and compiler
settings fixed between reference and candidate.

1. **Primitive dtype gate.** Lower representative active `_attention` shapes.
   Assert that the candidate QK score, softmax, and weight/value contraction are
   FP32 and that no FP64 conversion remains within that primitive. Preserve
   output shape, exact masked zeros, finite primal, reverse gradient, and JVP.

2. **One-day real-v6 gate.** On the first fixed micro8, compare every learned
   physical output, all next-state coordinates, every named one-day loss term,
   full parameter-gradient norm/direction, and the AdamW proposal. Require
   identical masks/discrete state/hard counts, all finite values, and unchanged
   exact-zero applicability. Keep water/carbon residuals within `2e-12`.

3. **Recurrent gate.** Reuse the preserved first T7 and T30 TRAIN windows and
   native checkpointed AD. Compare every named recurrent term/counter, full
   gradient, clip decision, and one complete mixed optimizer proposal. Require
   all physical hard counts zero. Use normalized per-field state/flux errors,
   not one global absolute tolerance across incompatible physical units.

4. **Tolerance policy.** The existing profile parity used `rtol=1e-5` and
   `atol=1e-6` for scalar/norm replay agreement. A changed-precision gate should
   predeclare comparable scalar tolerances, report maximum normalized
   output/state error, gradient cosine and norm ratio, and reject merely
   loosening tolerances after observing a failure. Very small gradient leaves
   need aggregate direction tests rather than unstable elementwise relative
   error alone.

5. **Timing gate.** Compile outside the measurement, synchronize each sample,
   report median/range and compiled memory, and compare the same primitive and
   then the same full update on V100 and RTX 5090. A representative-shape
   primitive is a mechanism test, not proof of actual training-distribution or
   whole-update speed.

## Expected speedup bounds

There is no defensible end-to-end speedup number from the present evidence.
The affected attention kernels are repeated and sometimes wide, but many have
query length one or small token banks; launch overhead, fusion, rematerialized
AD, and memory traffic can dominate. The partial CUPTI trace cannot repair this
missing attribution.

Hardware peak ratios only set arithmetic ceilings. V100 FP64 is approximately
one half of its FP32 peak, while a consumer RTX 5090 exposes FP64 at roughly
one sixty-fourth of FP32. Therefore leaving accidental DGEMM in place can make
an RTX 5090 a poor replacement despite its much stronger FP32 throughput.
For a compute-bound affected kernel, FP64-to-FP32 has a theoretical same-device
ceiling near 2x on V100 and near 64x on RTX 5090; moving a V100 FP64 kernel to
RTX 5090 FP32 has a much larger arithmetic ceiling than moving it unchanged to
RTX 5090 FP64. These are not update speedups.

Even the impossible case of making the entire measured 72.4% loss/gradient
category free would cap the current 14.09-second update near 3.62x by Amdahl's
law. The attention promotion is only a subset of that category, so its
whole-update ceiling is lower and must be measured. FP64 loss/reconstruction
elementwise work may remain a later RTX 5090 bottleneck after attention is
repaired, but that is a second, separately admitted precision study.
