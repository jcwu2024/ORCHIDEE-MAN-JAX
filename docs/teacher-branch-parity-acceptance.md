# Teacher Branch Parity Acceptance

Acceptance date: 2026-08-02.

## Decision

Gate A is complete. The canonical Teacher implementation for all subsequent
research and PFT-extensibility work is the shared Teacher core at
`research/daily-coarse-graining` commit
`a7796635f892af51c0902763d4938ae9179839db`.

`main` commit `7333b46c0b38650fb6c9250582876137831657b8` remains the frozen
user-facing release baseline. It is not a second development authority. The
branches remain separate because the research branch also contains
experimental dataset and neural code; this decision does not merge that code
into `main`.

## Executed Matrix

The manifest-driven runner used detached worktrees at the two exact commits,
the same external assets and content hashes, ordinary Teacher mode, and
capture options left at their default `False` values.

| Case | Lifecycle | Landpoint | Result |
| --- | --- | --- | --- |
| `cold_smoke_001_071` | two-day cold start | `001.0-071.0` | exact within `1e-12` |
| `later_smoke_069_119` | reference start plus later day | `069.0-119.0` | exact within `1e-12` |
| `restart_smoke_319_057` | restart smoke | `319.0-057.0` | exact within `1e-12` |
| `cold_365_001_071` | 365-day cold start | `001.0-071.0` | exact within `1e-12` |
| `cold_365_069_119` | 365-day cold start | `069.0-119.0` | accepted source correction |
| `restart_365_319_057` | 365-day restart | `319.0-057.0` | exact within `1e-12` |
| `year_handoff_001_071` | 365-day cold 1961 to restart 1962 | `001.0-071.0` | exact within `1e-12` |

Every day compares full state by component and field, exact discrete values,
modelout fields, compact modelout, and final/restart state. The complete
matrix contains 421,243 compared array leaves. Six cases have no failed leaf.

## Source-Backed Disposition

The only raw difference is `cold_365_069_119`. Days 1-273 are identical. On
day 274, the research branch first applies the Fortran
`min_stomate=1e-8` correction to a negative PFT14 reserve stock. The old
`main` input bundle omitted that source constant and used the Python helper's
historical zero default.

The first differing day contains exactly seven continuous fields:

- reserve biomass and `RESERVE_M`;
- maintenance respiration and `MAINT_RESP`;
- daily NPP, `NPP`, and `NPP_model`.

The initial absolute change is at most `1.0e-8`; no discrete or defined-status
field differs. Later continuous differences are propagation from that state
change. The disposition is fail closed in
`configs/teacher_branch_parity.json`: it fixes the first day, exact first-day
field set, maximum initial magnitude, source owner, JAX owner, and the SHA256
of the passed source-extracted `stomate_npp_growth` Oracle. Any additional
first-day field, discrete mismatch, larger initial change, missing Oracle, or
Oracle hash drift fails the gate.

Fortran provenance is
`fortran_source/ORCHIDEE/src_stomate/stomate_npp.f90:499-517`. The literal
Fortran add-back can land one ULP above the intended threshold; both `-O0` and
`-O3` source extraction demonstrated that pathology. The canonical JAX owner
preserves the carbon-budget compensation while pinning the corrected stock to
the source comment's intended threshold. See
[`rollout_min_stomate_threshold_20260730.md`](research/daily_coarse_graining/rollout_min_stomate_threshold_20260730.md).

## Six Shared Files

| File | Accepted treatment |
| --- | --- |
| `ad_primitives.py` | custom derivative rules preserve forward primals |
| `coupled.py` | canonical source `min_stomate` wiring retained |
| `driver/orchestration.py` | capture outputs are default-off and ordinary output is unchanged |
| `stomate/carbon_kernels.py` | inactive-lane AD guards preserve primals; source threshold correction retained |
| `stomate/season.py` | inactive-lane safe divisions preserve forward primals |
| `stomate/soilcarbon_kernels.py` | DOC square-root derivative rule preserves the source primal |

## Reproduction

```bash
conda run -n ORCJAX python scripts/dev/run_teacher_branch_parity.py --group all
```

`--resume` may reuse a snapshot only when its commit, case, and runtime flags
match. Generated evidence remains under `outputs/branch_parity/` and is not
committed. Accepted local artifact SHA256 values are:

```text
manifest.json          fa54409d1312c045d85c2cfce94d0f9b0b05a19b80e7437fe81efd72b17f9199
comparison.json        a98af5346f770aa634bb374e87e27d0fe7c2ecde0a1119dd7466a7882e9dcbc4
field_comparisons.csv  4787e497065716a2c6bd5fb02a66de15215e43b2eb18b8c3c3775b00f20fdcf0
report.md              cf48981f1f266855fe98a5e3095c58daf8e81f6a6357820aa60aaae81393fb8d
```

The final targeted regression covered the parity runner, AD primitives,
source-backed carbon, season, soil-carbon, coupled state assembly, and
default-off OK_LEAK capture: `333 passed`. Ruff, `py_compile`, and
`git diff --check` also passed.

## Boundary

This acceptance closes unresolved Teacher differences before new PFT or
neural work. It does not itself add another PFT, validate physical-parameter
gradients, implement the true daily operator, or replace the final
669-landpoint Teacher release run. Those are named Gates B-F, not Gate A
cleanup left for a future contributor.
