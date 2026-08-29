# Distribution-Aware Teacher-GPP Causal Bridge

Date: 2026-08-29

Status: frozen before zero-training execution

Contract SHA256:
`ce2eefb1360bb5b6f49f727ab470d1faef3952a7649d411539d81ba3b28c5647`

## Question

The update-250 distribution-aware candidate repairs maintenance respiration,
but GPP, growth respiration, and NPP remain outside the 7/30-day absolute
gates. This audit asks whether the remaining daily carbon error is primarily
caused by the learned GPP boundary or requires changes to the shared
Attention/state representation.

This is a causal intervention, not a candidate architecture and not a
training run.

## Intervention

For every day in each matched rollout window:

1. assemble the candidate's ordinary day-start state, native forcing,
   parameters, traits, and static inputs;
2. execute the accepted update-250 Attention candidate and retain its
   maintenance and every learned output except `daily.fast_gpp`;
3. replace only `daily.fast_gpp` with the matched Teacher-v6 value;
4. rebuild `ProcessPrediction` from those learned coordinates;
5. execute the ordinary conservative update and canonical retained exact
   tail; and
6. carry the resulting model-predicted continuous and discrete state into the
   next day.

Replacing GPP only in the reported trajectory is forbidden because it would
not affect growth respiration, NPP, or the next-day carbon state.

## Matched Modes

The fixed panel is train/train updates `251, 358, 465, 572, 679, 786, 893,
1000`, with eight 7-day and four 30-day windows per group.

- Ordinary free rollout reuses the accepted update-250 report values.
- Teacher-GPP bridge reexecutes the same checkpoint and windows with only the
  pre-update GPP substitution.
- Teacher-day-start reexecutes the same checkpoint and windows with the full
  matched Teacher day-start state. It is a diagnostic control for the effect
  of removing all state-recurrence drift.

No validation or sealed-test value may be read. No Teacher data are generated,
no historical v5 row is used, and no optimizer or parameter step is allowed.

## Frozen Classification

The acceptance score for each key is
`0.5 * (normalized_rmse^2 + normalized_signed_bias^2)`.

`gpp_boundary_supported_as_next_scope` requires all of the following at both
7 and 30 days:

- aggregate GPP, maintenance, growth, total autotrophic respiration, NPP,
  AGB, BGB, total-live biomass, and ecosystem carbon pass their existing
  absolute gates;
- the median bridge-versus-free score reduction is at least `50%` for both
  growth respiration and NPP; and
- all physical hard-failure counts are zero.

`gpp_boundary_contributes_but_not_sufficient` requires at least one of the
four growth/NPP-by-horizon cells to improve by `20%`, no such cell to regress
by more than `5%`, and zero hard failures. This result does not authorize a
GPP-only architecture.

Every other valid outcome is
`shared_attention_or_state_scope_required`. After the audit, all three
classifications return to the user-required architecture discussion before
any network is built.

## Source Ownership

GPP is a direct learned daily output. Growth respiration and NPP are retained
exact outputs. The governing Fortran boundary is
`fortran_source/ORCHIDEE/src_stomate/stomate_npp.f90::npp_calc:314-530`, with
the identity at line 530:

```fortran
npp = gpp - resp_growth - resp_maint
```

The final surrogate remains a true daily operator. This diagnostic reconstructs
no 48 half-hour forcing records and executes no subdaily state scan.
