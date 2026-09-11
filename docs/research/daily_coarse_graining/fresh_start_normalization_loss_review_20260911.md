# Fresh-start normalization and loss review

Date: 2026-09-11

Status: completed independent review of the pre-applicability fresh-start
one-day run. This is a bounded diagnosis, not trained-model, all-PFT, or
all-branch acceptance.

## Conclusion

The bare-soil `gsmean` domination is an exact output-applicability and
normalization mismatch, not ordinary random-initialization noise. Compact PFT
output moments pool bare soil with PFT14. The near-zero fresh raw head therefore
decodes both slots near a shared positive center, while the destination state
anchor scores the fixed-zero bare-soil Teacher column with fallback scale 1.0.
The resulting normalized state residual is about 2768.

No evidence blocks the unchanged v2 settings. The clean next experiment is an
exact, trait-based bare-soil `gsmean` zero before state reconstruction, while
retaining the v1 seed, samples, statistics, loss weights, optimizer, clipping,
and 12-update budget. The coordinate should remain defined and supervised; do
not hide it with a loss mask, inflate its state scale, or lower the whole state
coefficient.

## Proven mechanism

1. `successor_fast_owner_decoder.py:89-170` constructs the added fast-owner
   fields with compact shapes and `full_pft_axis=False`. The audited `gsmean`
   field has a two-slot PFT axis, a nonnegative constraint, and a
   day-end-minus-day-start owner representation.

2. `successor_fast_owner_statistics.py:274-304` creates compact-PFT moments
   with `pft_indices=tuple(range(field.shape[field.pft_axis]))`. The policy is
   recorded as `shared_across_compact_active_slots` at lines 374-380. Therefore
   both compact slots contribute to one pooled population.

3. `training_statistics.py:200-258` pools count, mean, and second moment over
   those indices and copies the pooled mean and scale back to every selected
   slot. Bare-soil and PFT14 `gsmean` consequently use the same decoder center
   and scale. Generic surface activity is being treated as field
   applicability, although the two are not equivalent for stomatal conductance.

4. Fresh initialization loads no historical weights (`fresh_start.py:47-67`).
   State-output heads have zero bias and kernels scaled by `1e-3`
   (`target_oriented_attention.py:214-250`; zero-bias linear initialization is
   at `strong_baseline_architecture.py:704-708`). Their raw outputs begin close
   to zero.

5. The nonnegative transform constructs an inverse-softplus baseline from the
   fitted center (`output_transform.py:39-50,66-88`). Raw zero maps back to that
   center. Since bare-soil `gsmean` starts at zero, the pooled positive delta
   mean materializes as a positive bare-soil endpoint.

6. The decoder mask is generic PFT activity. In
   `strong_baseline_architecture.py:1350-1395`, a compact coordinate remains
   active when `context.pft_mask` is true. In the paper-case registry, compact
   slot 0 is source PFT1 bare soil and slot 1 is PFT14; both are active. No
   field-specific mask removes bare soil.

7. Process targets repeat that assumption. The structural mask is built only
   from `inputs.pft.active_pft_mask` (`successor_fast_owner_training.py:67-76`)
   and is intersected with Teacher validity at lines 96-117. The process loss
   then divides the zero bare-soil target error by the pooled output scale
   (`training_loss.py:33-100`), making the error comparatively cheap in that
   coordinate system rather than preventing it.

8. The state anchor uses a different scale. Base `next_state` statistics are
   fitted independently for every flattened state column
   (`markov_dataset.py:341-422`), with unresolved columns assigned scale 1.0
   (`markov_dataset.py:239-255`). `target_oriented_training.py:429-499` projects
   these column scales into the state groups. Bare-soil `gsmean`, which is
   fixed zero in the audited train/train Teacher data, therefore has scale 1.0;
   PFT14 has scale `360508.62272754736`.

The state eligibility amendment excludes named inventory and thermal leaves
from residual `other`, but correctly does not exclude `gsmean`
(`target_oriented_training.py:505-517` and
`target_oriented_state_anchor_amendment_v1.json`). Once the physical invariant
is enforced, keeping this coordinate eligible makes the loss detect future
regressions.

## Numerical evidence

The authoritative artifacts for this review are
`outputs/fresh-start-loss-balance-v1.json`,
`outputs/fresh-start-one-day-v2.json`, and
`outputs/gsmean-zero-counterfactual.json`.

| Initial sample | Bare prediction | Teacher | Bare scale | PFT14 prediction | PFT14 scale |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 2769.586976 | 0 | 1.0 | 2768.264963 | 360508.622728 |
| 1 | 2767.579087 | 0 | 1.0 | 2765.49 | 360508.622728 |
| 2 | 2768.924970 | 0 | 1.0 | 2767.890662 | 360508.622728 |

`gsmean` supplies 99.75-99.77% of the complete weighted objective and about
99.79% of the grouped state loss. Its 513-parameter output head has gradient
norm `758.447583`, versus full norm `758.451725`, or 99.99891% of the squared
full norm. After 12 clipped updates, the invalid bare-soil values remain about
2656-2659. Most of the apparent learning signal is therefore this one endpoint
moving toward zero.

The accounting-only counterfactual sets only exported state coordinate 10 to
zero and leaves every other value, scale, and mask unchanged:

| Sample | Original state loss | Counterfactual state loss |
| --- | ---: | ---: |
| 0 | 229.648048 | 0.484567 |
| 1 | 229.307810 | 0.486184 |
| 2 | 229.426814 | 0.470893 |

This proves that global state-loss down-weighting is unnecessary to remove the
primary domination. It is not a forward acceptance test and does not test
state reconstruction, derivatives, budgets, or checkpoint replay.

## Why existing gates and loss components did not stop it

The original hard checks cover finite state and process values, undeclared
defined-mask changes, discrete equality, water and carbon budget residuals,
negative reported stocks, and generic output bounds
(`fresh_start_training.py:77-95`). A large positive bare-soil `gsmean` is
finite, satisfies the nonnegative constraint, and occupies a catalog-active
slot. It therefore passes every declared check. The missing declaration was a
named, field-specific applicability invariant.

The state anchor computes, within each active group,

`0.5 * normalized_MSE + 0.5 * normalized_signed_mean_error**2`

and then averages the active groups equally
(`target_oriented_training.py:657-714,849-868`). The bias term is not duplicate
MSE: it penalizes coherent signed drift. One extreme coordinate makes the MSE
and signed-bias terms reinforce. Attribution must retain the signed cross term,
as `state_loss_attribution.py:37-74` does; per-coordinate bias contributions
can be negative even though total squared bias is nonnegative.

The residual `other` group has 2,392 eligible coordinates and receives one of
seven group shares. Its within-group averaging is still insufficient against
a normalized residual near 2769. Equal group weighting is a design choice to
reassess after the fix, not the source of this failure and not a reason to
change v2.

Known duplicated state views are already partly repaired: named inventory and
thermal leaves are excluded from `other` by
`target_oriented_state_anchor_amendment_v1.json`. Direct primary flux labels
also have zero weight in necessary-process supervision specifically to avoid
duplicating the independent flux objective
(`target_oriented_daily_operator_v1.json:161-167`). The fresh-start flux list
contains GPP, maintenance respiration, growth respiration, and heterotrophic
respiration; it does not add a separate direct NPP loss.

State-budget-essential endpoints intentionally appear in both process
supervision and next-state supervision. That is not automatically erroneous:
the views can constrain an intermediate process and its state consequence. For
`gsmean`, however, they use incompatible applicability and normalization
semantics. Correct that mismatch before judging whether two-view weighting
needs redesign.

## Bounded v2 requirements

Apply exact zero only to the named learned `gsmean` output where the canonical
PFT trait identifies bare soil, before physical/state reconstruction. Do not
assume a fixed array position. Preserve vegetation values and gradients,
inactive-slot behavior, PFT permutation behavior, and the historical
default-off path. The bare-soil output and its parameter derivative must both
be exactly zero, while the zero value remains defined and loss-eligible.

For causal comparison, retain the existing output and state statistics in this
probe. Re-fitting statistics may be appropriate in the later systemic audit,
but combining that change with the first applicability test would obscure
which correction removed the failure.

The v2 report should require:

- exact bare-soil zero in both named `gsmean` output and next-state coordinate
  at initialization, derivative points, and monitored updates;
- unchanged PFT14 zero-update `gsmean` relative to the pre-applicability run
  within the existing forward tolerance;
- a named applicability hard count plus all original finite, mask, discrete,
  constraint, water-budget, carbon-budget, and stock checks;
- initial total/state/process/flux losses, full and `gsmean`-head gradient
  norms, clipping norm, and all parameter-family gradient summaries;
- evidence that the shared `gsmean` head can still learn from PFT14;
- the original reverse/JVP/finite-difference checks and checkpoint replay; and
- a new ranking of remaining fields before any loss reweighting.

The counterfactual losses near 0.47-0.49 are an accounting expectation, not an
acceptance threshold. A lower initial loss caused by enforcing a physical
invariant is not learned improvement.

## Remaining risk and what is not proven

The next largest state fields are
`slowproc_stomate_previous_step_state.fuel_1hr`, `dead_leaves`, and
`fuel_1000hr`. In initial sample 0, their largest zero-Teacher, scale-1
coordinates are predicted at about 78.4, 64.7, and 57.3. They are compact
PFT-axis, nonnegative OK_LEAK outputs whose moments are also pooled across
compact slots. This proves that they share the same normalization/applicability
risk pattern.

It does **not** prove that those fields should be hard-zeroed. Their Fortran
reset, carry, partition, and bare-soil semantics have not been established by
this review. They are overlapping tracer/partition quantities, not independent
carbon stocks. Do not generalize the `gsmean` correction by name or shape.

After v2, build a source-backed applicability ledger for every learned
PFT-axis field. At minimum compare stable PFT identity and traits, Fortran
branch/reset/carry provenance, Teacher zero/defined frequency, decoder pooling
population and scale, process mask and scale, destination-state eligibility
and scale, and decoded initialization. Flag any structurally fixed or
inapplicable coordinate pooled with applicable PFTs. Shared learned parameters
can remain for equivariance, but physical materialization and masks must use
semantic applicability; final statistics should pool only physically
comparable populations.

This review proves the current mechanism only for the audited `gsmean`
coordinate and three selected train/train one-day PFT14 cases. It does not
establish all-PFT, all-branch, longer-horizon, validation, sealed-test, or final
training-method acceptance. It does not authorize Teacher regeneration,
objective redesign, analogous hard zeros, or paid training.
