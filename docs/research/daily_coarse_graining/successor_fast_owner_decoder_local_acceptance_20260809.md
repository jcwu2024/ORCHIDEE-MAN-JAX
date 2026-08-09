# Successor Fast-Owner Decoder Local Acceptance (2026-08-09)

Status: **local decoder gate passed; formal v6 statistics scan not run**.

## Accepted Matched Boundary

The user approved extending the selected source-routed attention candidate and
fixed-graph control with the same 80 added field queries. The resulting schema
contains 181 fields and 48 equal-macro loss labels:

- the accepted 101 process fields remain unchanged;
- 79 fast-owner value fields add 839 scalar values;
- one `rveget` status field adds two compact-PFT logits;
- each arm has exactly 8,722,689 parameters;
- candidate and control have identical parameter-tree structure.

Only the interaction rule differs. The candidate uses content-dependent
attention along admitted source and process-graph edges. The control uses the
same parameters as fixed graph mixing. Inputs, queries, coordinate decoder,
normalization, constraints, losses, masks, parameter routing, and materializer
are shared.

## Decoder Semantics

The 65 fast-state value fields learn normalized day-end-minus-day-start-owner
targets. Every one has a unique same-shape canonical state slice. The only
cross-owner mapping is `final_diagnostics.temp_sol`, whose day-start owner is
`enerbil_previous_step_state.temp_sol`.

Undefined day-start values use a zero numerical base. This matters for
`rveget`: a transition from undefined to defined remains representable without
adding or subtracting the `1e20` sentinel. Its value loss is active only where
the Teacher day-end value is defined; its eligible compact-PFT slots receive a
separate stable binary cross-entropy status loss. The status BCE is averaged
with the `fast_owner.diffuco` value MSE before the 48 labels are averaged, so it
does not become a ninth added macro label.

The 14 daily-summary and OK_LEAK auxiliary fields learn normalized absolute
Teacher-native targets. At inference, the decoder converts every delta field
to its physical day-end endpoint before the existing fast-owner materializer.
The loss adapter converts those physical endpoints back to delta form only on
the host training path. Nonnegative and unit-interval constraints apply to the
physical endpoint, not to the signed delta.

Loss group, process-graph domain, and input source route are independent schema
attributes. Water and energy fields use their corresponding routes. DIFFUCO,
`control_inudate`, and `sechiba_finalize_state.leaf_ci` use
`carbon.fast_gpp`; `daily_interface.resp_maint_radia` uses
`carbon.fast_maintenance_respiration`; other carbon auxiliaries use
`carbon.other`.

## Local Gates

The local decoder packet passes:

- schema size, route/domain registry, 65 delta-owner slices, and 14 absolute
  targets;
- exact candidate/control parameter count and tree match;
- eager and JIT forward for both arms;
- 48-label value/BCE loss, JVP, and VJP with finite nonzero gradients;
- strict fitted-normalization registry checks, with unit normalization allowed
  only in explicitly named plumbing tests;
- compact-PFT inactive-slot zeros and inherited PFT permutation behavior;
- one-day conservative water/carbon budget residuals within `2e-12`;
- physical endpoint handoff through the complete 160-leaf materializer;
- inference source-graph exclusion of all host-only target, loss, and
  statistics modules.

The focused local commands currently pass 16 successor tests plus 22
training/materializer/contract tests. No Teacher row was generated, no v5 row
was read, no checkpoint was written, and no training update was executed.

## Added Statistics Path

`successor_fast_owner_statistics.py` vectorizes the same 80 added targets over
existing v6 shards. It fits mergeable count/mean/M2 moments for only the 79
value fields, records `rveget` defined/eligible counts, shares applicable
moments over compact active-PFT slots, and writes a hash-bound JSON/NPZ pair.
Local batch targets match the single-day adapter exactly; write, reload, merge,
181-field normalization, and 180-value loss-scale assembly pass.

The proposed Explore1000 scan reads only accepted v6 `state_trajectory` and
`fast_day_target` arrays. It does not read historical v5 or regenerate the
Teacher. The proposal is one `cnmix` node allocation with 8 CPU cores and a
2-hour limit, running eight one-core partitions. Worst-case charge is
`8 * 2 * CNY 0.07 = CNY 1.12`. It has not been submitted.

## Active Gate

The next active operation is the explicitly approved train/train statistics
scan and independent reload of its merged artifact. Model training remains
closed until that artifact passes identity, finite-moment, no-missing-column,
and `rveget` prevalence checks. Afterward, the next gate is a free `gln01`
no-checkpoint forward/loss/gradient probe before any bounded GPU fit. The
7/30-day rollout, 365-day run, long-held points, Gate D2, and product
acceptance remain closed.
