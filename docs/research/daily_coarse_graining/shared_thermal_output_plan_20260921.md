# Shared thermal output scaling: bounded correction

Status: TRAIN audit supports the correction; implementation and local tests
pass. Actual forward/gradient/update admission remains pending.
This is not a new backbone or paid training plan.

## Mechanism and scope

The shared head currently routes direct signed-unit thermal tendencies through
`tanh(raw)`. The conservative updater adds this normalized tendency times the
declared thermal bound (100K for ptn). Loaded TRAIN output mean and scale are
not used in this branch. Order-one random logits can therefore produce tens
of kelvin of one-day error. The two saved admission rows show80-87K at a deep
soil coordinate whose TRAIN state-change scale is approximately0.000119K.

Proposed shared-head-only transform, for direct signed-unit thermal fields:

```
c = clip(TRAIN_mean, -1 + 1e-6, 1 - 1e-6)
s = existing positive TRAIN output scale
tendency = tanh(atanh(c) + raw * s / max(1-c*c, 1e-6))
```

At raw=0 the value is c and the derivative is s for audited interior centers.
It preserves the mathematical open interval(-1,1), the existing physical
updater, all output coordinates and state/parameter gradients. It does not
hard-clamp daily temperature changes to a narrow learned interval. It changes
learning coordinates, so gradients and learning dynamics require new admission.
The existing TRAIN fallback scales must be checked against actual applicability
before adoption; unobserved but active coordinates cannot silently acquire a
scientific scale from the generic unit fallback.

The free test01 audit reads the existing process-statistics NPZ: ptn has64
observed coordinates, all with finite interior centers and positive scales
1.186178330e-6..0.01136814890. Its other384 coordinates have count0/scale1
and are all outside the declared active-PFT layout; static gather and runtime
masks exclude them. Observed temp_sol/stempdiag/snowtemp/snowheat statistics
are also finite/interior with positive scales. A scale numerically equal to1
is not intrinsically invalid; here its fallback provenance and inactive masks
are established together. For ptn the center's physical increment is
-0.000253741..+0.000104816K; the smallest local physical slope is0.000118618K.

Keep the shared backbone, initialization seed, parameter tree, process/science
losses, coefficients, state masks and daily conservative/retained boundaries
unchanged. Duplicate ptn/tdeep and cgrnd state-loss ownership is documented as
a separate issue; do not remove terms in this comparison. Historical heads
retain their original transform. The old admission protocol must still select
direct tanh explicitly, so its saved gradient cannot be reused under this change.

## Matched comparison and gates

1. Read existing TRAIN metadata/arrays on a free CPU node: mean/scale ranges,
   finite/interior centers, count-zero/fallback coordinates and active masks.
   No Teacher regeneration or test-set calibration.
2. Local tests: center/local derivative, full bounded range, inactive zeros and
   gradients, inverse recovery of selected physical tendencies, and unchanged
   legacy transform. Exercise the canonical full-schema head.
3. On the same two TRAIN rows and fresh seed as094dfd2, compare canonical
   forward values and unchanged losses with saved original evidence. Require
   all hard counters zero and report physical thermal errors, field attribution
   and every carbon/science group. A smaller aggregate loss alone is insufficient.
4. If the thermal mechanism is confirmed in actual execution, run the revised
   one-day native reverse/JVP and two complete updates, with the existing24GiB
   memory bound and declared directional tolerances. Save completed evidence
   before further compilation. No inherited optimized weights.
5. Qualify the effective T7/T30 joint loss and resource use next, then prepare
   one bounded learning contract. No new paid run is authorized by this plan.

The old8004a11 saved-gradient recovery continues in an immutable checkout.
Its result only qualifies the old transform. Do not overwrite or pool its
directional/update evidence with the corrected transform.

Implementation selects `thermal_transform=train_centered` in
`shared_transformer_admission_v2.json`. The runner explicitly preserves
`direct_tanh` for v1, even from newer code. `--forward-only` stops after
canonical physical/loss/attribution reporting, before gradient compilation or
updates. The paired forward compares the same declared fresh initialization;
it does not inherit a trained checkpoint. Existing losses and masks are unchanged.

Local evidence: shared-head/recovery suite9 tests PASSED177.24s, including
canonical161-field physical output parity between original and optimized
decoder execution, thermal center/local Jacobian/mask and wide-range inversion.
After adding forward-only and v1-policy regression coverage, recovery suite
4 tests PASSED3.37s; focused thermal2 tests PASSED3.36s. Ruff/diff checks pass.
