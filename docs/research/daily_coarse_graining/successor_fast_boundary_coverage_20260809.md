# Successor Fast-Boundary Coverage Audit

Status: blocking architecture/boundary evidence; no candidate promoted.

Date: 2026-08-09.

## Question

Does the selected successor's 101 named process outputs fully reconstruct the
canonical 2,855-column fast-day boundary before the retained exact daily tail?

This audit does not train a model, regenerate v6, read v5, reconstruct 48
forcing records, or execute a 48-step state scan.

## Trigger

The strict real-v6 lifecycle gate lost 37 defined state values on Day 2. Every
mismatch had the same direction: Teacher defined, successor undefined. The
affected compact PFT slot is the active vegetation slot and includes biomass,
`bm_to_litter`, LAI, vegetation, and related retained daily state.

Initialization-scale, nonnegative-baseline, part-aware carbon, and named-output
Teacher replacement probes did not remove the failure. Replacing all 101 named
outputs by same-row Teacher values still lost the same 37 state masks.

## Oracle Result

The all-Teacher named-output case reconstructs only `87/160` canonical
fast-target leaves within `atol=1e-8, rtol=1e-10`; 73 leaves remain nonexact.
It also leaves one `diffuco_previous_step_state.rveget` mask mismatch and the
GPP/maintenance daily-interface mask mismatches.

This proves that the failure lies downstream of the 101 named values, in the
mapping

```text
101 named process outputs
  -> ProcessPrediction
  -> retained boundary
  -> 2,855-column canonical fast target
  -> retained exact transition
```

It is not evidence against attention, the matched graph control, model width,
or training initialization.

## Contract Ledger

Commit `e705b68` adds
`scripts/dev/audit_successor_fast_boundary_coverage.py`. The audit explicitly
separates a same-owner residual baseline from numeric persistence. The v6
representation has 2,754 same-owner baseline columns and 101 mean-centered
columns, but the former are not constant over a day.

The current adapter assigns the 160 leaves as follows:

| Current source | Leaves | Columns | Verdict |
| --- | ---: | ---: | --- |
| Explicit conservative/named endpoint | 22 | 1,483 | represented |
| Source-exact/native-forcing derivation | 12 | 25 | represented |
| Daily integral written into instantaneous state | 14 | 113 | semantic collision |
| Day-start carry | 106 | 1,204 | requires owner audit |
| Day-start/constant daily derivation | 6 | 30 | requires exactness audit |

The 14 semantic collisions include `typed.energy.fluxsens`, `soilflx`, and
`soilflx_pft`, plus daily-integrated HYDROL transfer labels such as `wat_flux`,
`drainage_per_soil`, and `precip2ground`. These daily integrals cannot be
written into same-named instantaneous fast-state fields merely because the
names match.

## Existing-v6 Numeric Scan

The optional read-only scan ran on free `test01` over the earliest two shards
from each of the nine spatial/temporal split combinations: 18 annual shards
and 6,564 daily transitions. Among the 106 leaves currently carried from the
day start, 62 leaves totaling 1,007 columns actually changed within the day.
Examples include:

- driver albedo and coupling coefficients;
- `diffuco` canopy exchange, resistance, humidity, temperature, and `rveget`;
- `enerbil` and `thermosoil` thermal coefficients and mirror fields;
- non-inventory HYDROL profiles and diagnostics;
- nine non-inventory/derived OK_LEAK fields;
- `sechiba_finalize_state.leaf_ci`.

The report is
`runtime/outputs/training/successor-process-head-v2-canonical-e705b68/fast_boundary_coverage_report.json`
on Explore1000, SHA256
`57a2cc6db11605a1fc02e2b3d3e92d3fae4c55145d63ae3ddb7d0f2cfaade290`.

## Interpretation

Gate C2 requires non-inventory fast-boundary fields to remain explicit endpoint
reconstruction targets. The current successor schema implements inventory,
thermal, daily-interface, and typed process labels, but omits the canonical
non-inventory endpoint reconstruction layer. Synthetic plumbing did not expose
this because it did not require a real retained lifecycle to recover all v6
fast owners.

The repair is therefore an architecture/boundary decision, not a training
hyperparameter change. A valid design must:

1. preserve the 101 source-backed process outputs and their conservative uses;
2. choose one canonical owner for each required fast prognostic field;
3. reconstruct duplicate packet/restart mirrors deterministically;
4. represent the dynamic `rveget` value and defined-status transition
   explicitly;
5. remove every daily-integral/instantaneous-state name collision;
6. give the attention candidate and matched graph control the same added output
   boundary and comparable parameter budget.

No new training or rollout is admissible until all 101 Teacher process labels
plus the proposed owner reconstruction produce `160/160` exact fast leaves and
zero cold/ordinary/restart lifecycle mask failures.
