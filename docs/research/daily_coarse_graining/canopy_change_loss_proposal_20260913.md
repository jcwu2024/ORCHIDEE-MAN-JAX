# Canopy carbon stock-change supervision proposal, 2026-09-13

## Decision and scope

This is a proposal for the next declared fresh-initialization, full-network
daily-surrogate method. It does not change the current Attention architecture,
Teacher data, decoder parameterization, shared science extractors, or any
running/finished checkpoint. In particular, the 1,998-update checkpoint from
job `15048786` remains a diagnostic artifact and is not a source of weights for
this method.

The proposal adds one separately reported and separately weighted term to the
existing public one-day objective. It corrects a demonstrated loss-geometry
gap: a large land-area canopy organic-carbon stock drain can have a small
dimensionless outgoing-fraction error and be strongly diluted by the ordinary
next-state stock scale. It is not a claim of complete ecosystem-carbon
closure, and it does not infer that this stock drift is the cause of the GPP
recurrence error. The independent state-replacement work remains the evidence
route for that causal question.

This document specifies the loss proposal; execution authority comes from the
user's current instructions. The shared helper and optional one-day objective
integration are locally tested. Enable it with protocol
`canopy_carbon_stock_change = {weight: lambda, scale_source_pft_index: 13}`;
13 is the zero-based source index of PFT14. Absent/zero weight preserves the
previous objective. No trained weights have changed. The combined calibration
with litterpart supervision is specified in
`feedback_and_canopy_loss_amendment_20260913.md` and supersedes the older
canopy-only sequencing proposal below.

## Evidence fixed before this proposal

`canopy_inventory_supervision_audit_20260913.md` and
`outputs/trained-canopy-inventory-label-audit-v1.json` establish the following
on 240 selected T30 days from the fixed diagnostic checkpoint.

| Reading | Value |
| --- | ---: |
| Teacher PFT14 outgoing target | `0` on all 240 days |
| Teacher PFT14 incoming target mean | `4.54273245 gC m-2 day-1` |
| Teacher-reset predicted PFT14 outgoing-fraction mean | `0.0081325875` |
| Teacher-reset predicted incoming, all active slots | `0.159 gC m-2 day-1` |
| Teacher-reset predicted incoming, bare / vegetation slots | `0.065 / 0.094 gC m-2 day-1` |
| Teacher-reset canopy stock change | about `-275.061 gC m-2 day-1` |
| Current official PFT14 train/train canopy state-delta scale | `5.403926065530931 gC m-2 day-1` |
| Existing next-state canopy stock scale | `47559.68758 gC m-2` |

The reset quantity starts each day from the Teacher state. It is therefore a
conditional one-day error for that checkpoint and panel, not merely accumulated
free-run drift. The raw-v6 readback also verifies that the selected fast and
terminal `interception_storage` coordinates agree exactly, so the typed labels
are available without an inferred gross-flux split.

`interception_storage` is the carbon component of the STOMATE TF-DOC canopy
store, not interception water. In
`fortran_source/ORCHIDEE/src_stomate/stomate_soilcarbon.f90`,
`soilcarbon_leak`, lines 731-737 describe the store and lines 1497-1525 apply
its deposition/washoff transition. The transition loops over Fortran
`j=2:nvm`, excluding bare soil. The source balance adds this store directly,
without another `veget_max` factor, at lines 1245-1251 and 2326-2332. The
shared research extractor now follows that land-area convention. This loss
must use the same convention.

## Exact term

For sample/day `b`, let `x0`, `xT1`, and `xP1` be respectively the Teacher
start, Teacher next, and predicted next values of the carbon coordinate of
`slowproc_stomate_previous_step_state.interception_storage`. Let `V_b` be all
active compact PFT slots, including an active bare slot. These values are
already in `gC m-2` on the land-area basis. Define

```text
dT_b = sum_{p in V_b} (xT1[b,p] - x0[b,p])
dP_b = sum_{p in V_b} (xP1[b,p] - x0[b,p])
r_b  = (dP_b - dT_b) / s_canopy_change
L_canopy_change = mean_{b in valid} r_b^2
```

The denominator `s_canopy_change` has units `gC m-2 day-1`; the caller obtains
it at runtime from the frozen train/train `state_delta` statistics for this
coordinate (currently the official PFT14 value shown above). It is not a
rounded scientific constant embedded in the helper and is not a statistic
estimated from the selected T30 panel, a monitor, validation, or test data. The subtraction of
the common Teacher start is deliberately retained in the implementation even
though it algebraically cancels for a completely defined single day: it makes
the supervised physical quantity explicit and carries correctly to free
recurrence.

The term is **not** a loss on the independent `incoming_amount` and
`outgoing_fraction` decoders. Those typed targets and their current
`carbon.ok_leak_inventory_endpoints` process supervision remain in the base
loss. This term supervises their physically realized net stock consequence;
it is what makes a small fractional outgoing error visible at a large stock.

The formal one-day objective becomes

```text
L = L_base + lambda_canopy_change * L_canopy_change

L_base = w_flux * L_independent_daily_flux
       + w_state * L_one_day_state_anchor
       + w_process * L_necessary_process_supervision
```

`L_base` and its existing weights are unchanged. This proposal does not
replace state scales, retune every state, alter the process-label macro, or add
a second cover weighting. `lambda_canopy_change` is selected by the
train-only calibration below and then frozen in the new method manifest.

## Applicability, masks, and numerical behavior

Implementation must obtain the compact PFT axis and carbon coordinate through
the existing state/contract layout. It must not hard-code a flat state offset
or assume that PFT14 is a particular array index. Its PFT eligibility is

```text
coordinate_valid = active_pft_mask
                 AND start_defined
                 AND teacher_next_defined
                 AND predicted_next_defined
                 AND finite(start, teacher_next, predicted_next)
row_valid = any active coordinate AND every active coordinate is coordinate_valid
```

The per-row sum is evaluated only for `row_valid`; an invalid row contributes
zero numerator and zero count. The batch term is zero when its valid count is
zero. This produces a finite scalar and keeps the existing nonfinite/mask hard
counts responsible for rejecting a bad transition. It does not conceal a failed
model output as a valid target.

The TF-DOC transition excludes bare soil, but an active bare slot created by a
model is still part of the source-direct stored-carbon stock error. It remains
in this sum and is not clamped, silently reclassified, or made undefined by
this loss. The implementation must separately report its predicted/Teacher
change and residual, active-bare count, and nonzero Teacher/predicted canopy
start/end counts. This preserves the observed bare-slot mismatch as an
applicability diagnostic until a separately source-backed output rule is
proposed.

No PFT-fraction/cover multiplication is permitted in this reduction.
`interception_storage` is already land-area carbon. The term must require at
least one valid active slot per contributing sample and report the number of
slots and rows used. It must also report `dT`, `dP`, their residual in
physical units, normalized MSE, and the raw source-direct endpoint stock
error. The historical fraction-weighted quantity may remain a comparator in
reports but cannot drive this loss.

## Teacher-start versus free recurrence

The first fresh one-day admission and coefficient calibration use the Teacher
day-start state, exactly as the current one-day objective does. They answer:
given the correct start and same-day forcing, can the network avoid producing
the erroneous canopy stock change? They do not establish free-rollout
stability.

When a later declared 7/30-day stage is admitted, apply the same term at each
calendar day with the recurrent state:

```text
dP_t = sum_V (xP[t+1] - xP[t])
dT_t = sum_V (xT[t+1] - xT[t])
```

The initial `xP[0]` is the declared Teacher initial state, but subsequent
`xP[t]` is the model's carried state. Gradients flow through the complete
declared rollout using native checkpointed JAX AD; there is no daily Teacher
reset or `stop_gradient` in this term. Calendar-day validity masks are
applied before the time/window mean. A later recurrence contract must declare
its horizon mixing and reuse this exact shared extractor rather than rebuilding
an observation cotangent in a runner.

## Proposed relative-weight calibration

The relative coefficient cannot be selected from the diagnostic checkpoint or
the selected T30 panel. Before a formal run, declare a calibration manifest
that uses the existing prepared one-day train pool only:

```text
source pool: fresh_start_one_day_pool_v1
sampler:     seed 20260912, epoch 0
rows:        the first four 4-row microbatches (16 rows total)
parameters:  new fresh initialization from the final declared architecture seed
candidate lambda_canopy_change:
  [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
```

On each of the four microbatches, compute native JAX AD gradients separately
for `L_base` and unweighted `L_canopy_change`. For each candidate calculate

```text
q = ||lambda_canopy_change * grad(L_canopy_change)||_2
    / max(||grad(L_base)||_2, 1e-30)
```

The proposed selection recipe is the **largest** candidate whose four-batch median `q` is at least `0.10`
and whose four-batch maximum `q` is at most `0.50`; all loss values, gradient
leaves, and existing hard counts must be finite/zero. The upper bound limits a
single inventory term from replacing the base objective; the lower bound makes
the added supervision materially connected to the full-network update. If no
candidate passes, reject the proposed coefficient and revise the declared
candidate set or objective before formal training. This ladder is a concrete
proposal for review, not an already accepted numerical admission truth. Do not
tune scales or select a coefficient against validation/test science metrics.

This is a weighting calibration, not normalization fitting. The runtime-derived
frozen train/train scale remains fixed regardless of its outcome. Persist the 16
semantic day keys, initialization seed, selected coefficient, per-batch losses,
gradient norms, counts, and code/config identity. The selected coefficient is
then immutable for the subsequent fresh run; a new architecture, data split,
or loss change requires a new declared calibration.

## Required bounded pre-training check

After the local unit tests and the gradient calibration pass, run exactly two
complete AdamW updates from a fresh reinitialization, using the first two
32-row epoch-0 batches of the same train-only pool and the selected coefficient.
This is a bounded technical admission, not a training result and not a
checkpoint to resume. It must use the actual shared one-day objective and
ordinary native JAX AD, not a hand-built derivative of this scalar.

Its report must contain before/after `L_base`, `L_canopy_change`, total loss,
the physical canopy-change RMSE/bias, per-term gradient norms, global clipping
factor, peak memory, cold/hot update time, valid-row count, active/bare-slot
counts, and all existing physical/mask/budget hard counts. Admission requires
all values and gradients finite, no hard failure, nonzero valid canopy rows,
and proof that `L_canopy_change` reaches trainable parameters. It must also
report GPP, NPP, maintenance/growth/autotrophic respiration, and biomass-change
outcomes in physical units before/after the two updates. It is a resource/graph
and update-integrity check only; those two-update science readings do not
establish 7/30-day accuracy or ecosystem closure.

The immediate next small check is therefore a **local, synthetic no-update
unit/AD probe** of the shared extractor. Its purpose is to verify direct
land-area units, masks (including active bare slots), finite loss/gradients
with undefined NaNs, and the explicit stock-change formula. Only after it
passes is the bounded free-node, actual train-only graph probe above considered;
the prepared data are remote. No expensive new 7/30-day graph is justified
until the combined GPP-diagnosis result is ready.

## Implementation and acceptance checklist

The implementation belongs in one shared, architecture-independent helper
(path: `research/daily_coarse_graining/daily_operator/canopy_change_loss.py`)
and returns a named result containing `loss`, `count`, `teacher_delta`,
`predicted_delta`, `physical_residual`, and bare/applicability diagnostics. The fresh
one-day objective should consume that helper once; a future public rollout
objective should consume the same helper per day. Runners and science
extractors must report its returned values, not recreate its reduction.

Focused tests must cover:

1. Direct sum over all active slots with no PFT-fraction weighting.
2. Inclusion-and-separate-reporting of active bare slots.
3. Exact zero loss for matching stock changes, including nonzero common starts.
4. Runtime use of the supplied frozen scale and physical-unit residual conversion.
5. Defined/undefined and nonfinite masking, zero-valid behavior, and finite
   gradients for a valid synthetic transition.
6. One-day Teacher-start and two-day free-recurrence formulas agreeing with
   their explicit sums.

The new final-method protocol must name this term, the frozen scale, selected
coefficient, fresh initialization, full neural trainable scope, sampling,
stage transition to free recurrence, stopping rule, and checkpoint-selection
rule before any scientific training is claimed. Passing the bounded check
admits neither the old checkpoint nor a final model.

## Local implementation evidence

The helper and four focused tests are implemented. Together with the existing
source-direct area regression, five tests pass; Ruff and diff whitespace pass.
Root also evaluated the helper on the240 archived Teacher-reset T30 rows,
without running the model: physical canopy-change RMSE503.46417gC/m2/day,
bias-279.60340, normalized MSE8679.97234, bare contribution+0.065164.
The exact scale was read from frozen state_delta statistics. This reproduces
the previously measured physical error and demonstrates that the new term
exposes it. This diagnostic-panel calculation selects no coefficient and is
not a neural parameter-gradient, learning, or model acceptance claim.
