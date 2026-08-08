# First Neural Process-Head First-Pair Decision

Status: **closed after one matched seed by user-directed practical equivalence**.

Date: 2026-08-08.

## Completed Pair

Both arms used seed `20260807`, the accepted v6 release, the same 10,000-update
budget and the same 5,464,321-parameter tree.

| Arm | Explore1000 job | Best update | Selection score | Runtime | Checkpoint SHA256 |
| --- | --- | ---: | ---: | ---: | --- |
| `axial_process_operator_v1` | `14507225_0` | 8750 | 0.8219892971 | 54:08 | `6c05411a117919770c921dec442dd4c37d34a152864a5592383db9879a5eddae` |
| `factorized_process_control_v1` | `14508831_1` | 9000 | 0.8433795294 | 57:14 | `a764bebad7787b8dfe7482e8c00234080a16ddab108863d1b095f70dcfcb9d40` |

Candidate/control training-report SHA256 values are respectively
`8f2493ed8f43d70c19640cae27ae4181e0864ef94e5e826970db550f3a191e4f`
and
`25c54b215ff97d5779d6f1be7084a09d7eab5e1f91298dbcadb351951e5b18a5`.

Candidate score improvements relative to the control were `2.54%` overall,
`5.08%` temporal and `1.83%` spatial; the candidate regressed `4.41%` on the
joint slice. Final training losses were nearly identical and both runs were
stable.

These are the runner's macro-MSE selection scores. They are not the NRMSE
named in the original architecture contract, and that contract did not define
a three-seed aggregation rule. Therefore this pair cannot establish formal
equivalence, formal rejection under the complete protocol, or promotion.

## Decision

The controlling decision is:

```text
stop_for_practical_equivalence_after_first_matched_pair
```

- Seeds `20260808` and `20260809` will not be submitted.
- Neither arm is promoted as the next daily surrogate.
- The attention-versus-factorized comparison is closed; no marginal ranking
  or post-hoc threshold change will continue it.
- Both checkpoints and reports remain negative evidence.
- No 7/30-day rollout, 365-day screen, Gate D2, or product claim is made for
  either arm.

The scientific bottleneck is redirected from process-axis interaction toward
long-term biomass, productivity and conservative compartment transfers. The
active foundation is
`long_horizon_biomass_productivity_objective_v1.md`. A successor network must
be discussed before implementation.
