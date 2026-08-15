# Repeated Active-Set Smoke, 2026-08-15

## Decision Before Execution

Freeze one five-total-step, zero-checkpoint smoke of repeated active-set
recalibration. This continues the admitted direct-direction mechanism; it
does not change `source_routed_process_attention_v2`, the v6 data, or the four
training objectives.

The smoke reconstructs the already accepted first norm-`0.002` direct update,
then attempts four new updates. This matches the five-step horizon of the
rejected fixed-parent-weight smoke while replacing only its invalid fixed
weights.

## Per-Update Mechanism

1. Start from the preceding update's positive simplex support. The first
   active-set update starts from the parent's 14 positive constraints.
2. Compute those component gradients separately and solve their minimum-norm
   simplex at the current parameters.
3. Run ten group JVPs to check all 40 relative directional derivatives.
4. Add any nonpositive constraint whose gradient is unavailable and re-solve,
   with at most three rounds, six additions, and 20 component reverse passes.
5. Try norms `0.002, 0.0015, 0.001, 0.0005` in descending order. Apply only
   the first step for which all 40 losses are nonincreasing within `1e-6`, all
   four component medians improve by at least `1e-4`, and hard/nonfinite
   counts are zero.
6. Stop before any update if no useful step passes.

The frozen contract is
`manifests/coarse_graining/successor_repeated_active_set_smoke_v1.json`,
SHA256 `2bef78c7d683de49250708a2a2b4d7016d11c0300d6c1cdf3d61dd8c4a1028c5`.

## Endpoint Evaluation

After the bounded updates, evaluate the same six-artifact non-test 7/30-day
subset used by the preceding mechanism smokes. The smoke separates two
claims:

- directional rollout support requires positive combined NPP/AGB/BGB/live
  biomass improvement, every primary aggregate no worse, at least `4/6`
  slice-horizon cells improved, and the established maintenance guards;
- the existing `0.5%` combined primary improvement remains a separate
  promotion-scale diagnostic and is not lowered.

Passing the mechanism and directional rollout gate does not promote a model.
It only permits review of whether a larger bounded active-set continuation is
worth running.

## Boundaries

- free `gln01` only; no paid job;
- four new direct updates maximum after one reconstructed update;
- same ten frozen train/train groups at each update;
- no AdamW, momentum, coordinate scaling, weight decay, or clipping;
- no checkpoint, recovery, or final parameter retention;
- no Teacher regeneration, v5 data, sealed test, 365-day rollout, or Gate D2.
