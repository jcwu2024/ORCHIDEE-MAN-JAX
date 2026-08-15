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

## Result

The free-`gln01` smoke completed at commit `b730731`. The report is
`runtime/outputs/training/successor-repeated-active-set-b730731/report.json`,
SHA256 `04c354748bed5abbd2640536dee4d1a50829e166dc4243cdf5ac8858b06613d7`.
The evidence is valid and reproduces the preceding accepted direction to
`4.96e-15` relative norm error.

The first new active-set update passes in one JVP round. It uses 14 component
reverse passes, retains a 13-constraint support, and applies norm `0.002`.
All 40 losses decrease. The component median relative changes are
`-1.17109e-4` for the one-day anchor, `-9.82175e-4` for NPP,
`-1.28702e-3` for AGB tendency, and `-3.33786e-4` for BGB tendency.

The following active-set direction also passes in one round. Its minimum
relative directional derivative is `0.0105908`; it uses 13 component reverse
passes, adds no constraint, and keeps the same 13-constraint support. The
finite-step gate, not the direction gate, stops the smoke:

- norms `0.002`, `0.0015`, and `0.001` regress at least one loss, with maximum
  relative increases `0.286978`, `0.287066`, and `0.287155`;
- norm `0.0005` decreases all 40 losses, with maximum change
  `-5.07888e-6`, but its anchor and BGB median decreases are only
  `2.85324e-5` and `8.06767e-5`, below the frozen `1e-4` useful-effect gate;
- the same `0.0005` step decreases NPP and AGB medians by `2.37723e-4` and
  `3.11891e-4`.

No attempted step has a hard failure or nonfinite value. The run applies one
of four planned new updates, takes `1683.38` seconds, and reaches
`13,826,391,552` peak device bytes.

## Rollout Interpretation

The retained two-total-update endpoint passes the separate directional
7/30-day gate. All `6/6` slice-horizon cells improve and the aggregate NPP,
AGB, BGB, and total-live RMS metrics are all lower. The combined primary
composite falls from `0.8465344173` to `0.8463801822`, a relative improvement
of `0.0182196%`. Maintenance RMS regresses by only `0.0979638%`, within the
frozen guards.

This is consistent positive rollout evidence, but it is far below the
unchanged `0.5%` promotion threshold. The formal classification is
`repeated_active_set_smoke_not_supported`: the four-update mechanism does not
close, no model is promoted, and no parameters are retained.

## Decision

Do not search smaller one-shot norms or continue the same per-step effect
gate. At the stopped state, `0.0005` is already safe across all 40 objectives,
while a smaller step cannot meet the missing anchor/BGB magnitude gate. The
next bounded mechanism may accumulate several safe `0.0005` microsteps,
recalibrating the active set after every microstep, and apply the `1e-4`
effect requirement to their cumulative macro-update. This changes the update
schedule only; it does not change the network architecture, data, objectives,
or promotion threshold.

## Boundaries

- free `gln01` only; no paid job;
- four new direct updates maximum after one reconstructed update;
- same ten frozen train/train groups at each update;
- no AdamW, momentum, coordinate scaling, weight decay, or clipping;
- no checkpoint, recovery, or final parameter retention;
- no Teacher regeneration, v5 data, sealed test, 365-day rollout, or Gate D2.
