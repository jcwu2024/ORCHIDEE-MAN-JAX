# Multi-Landpoint Validation Checkpoint 2026-07-11

## Scope

The canonical local reference contains all 669 independent paper landpoint
packages. A deterministic 12-point maximin sample spans coordinates, the four
calibrated PFT14 parameters, and archived AGB/BGB/GPP/NPP values.

The validation gate now compares:

- 12 underlying annual modelout history fields;
- 12 process diagnostics (`LAI`, total and four mangrove-root maintenance
  terms, four allocation terms, `GROWTH_RESP`, and `VCMAX`);
- four derived AGB/BGB/GPP/NPP values.

Tolerance remains `atol=2e-6`, `rtol=1e-7`. It has not been enlarged to hide
process differences.

## Current Evidence

Before the explicit-snow correction, eight of the 11 new sample runs passed
1961 at roughly `2.1e-7` to `4.3e-7`; the original calibration point passed at
`1.75e-7`. The three high-risk points were:

| Landpoint | 1961 max derived error | Classification |
| --- | ---: | --- |
| `069.0-119.0` | `2.435e-3` | dry, low-productivity |
| `215.0-119.0` | `3.113e-3` | extremely dry, low-productivity |
| `319.0-057.0` | `7.510e-4` | low-productivity with active snow |

The active explicit-snow execution gap at `319.0-057.0` is fixed: all 365
days now execute. A separate restart-binding defect that loaded point 001
state for selected points is also fixed. With correct 2010 restart binding,
`215.0-119.0` closes at `1.30e-7`; `069.0-119.0` and `319.0-057.0` residuals
fall to `2.31e-5` and `4.27e-5` in the four derived values.

Each selected point now uses its own `z1/used_run.def`. Re-running
`069.0-119.0` cold-start 1961 with that base leaves every annual delta
unchanged, so the remaining cold-start mismatch is not configuration mixing.

## First-Divergence Localization

For `069.0-119.0` cold-start 1961:

| Field | JAX | Fortran | Absolute error |
| --- | ---: | ---: | ---: |
| `GPP` | `0.506338576` | `0.505973458` | `3.651e-4` |
| `NPP` | `0.209375745` | `0.207943201` | `1.433e-3` |
| `MAINT_RESP` | `0.193493736` | `0.194604114` | `1.110e-3` |
| `GROWTH_RESP` | `0.103469094` | `0.103426129` | `4.297e-5` |
| `LAI` | `1.006204088` | `1.004396915` | `1.807e-3` |
| `VCMAX` | `67.942905743` | `67.942924500` | `1.876e-5` |

The four mangrove-specific added-root maintenance terms agree at `1e-9` or
zero. The mismatch therefore precedes annual biomass/modelout aggregation and
is concentrated in ordinary leaf/root/sap maintenance plus dry-path GPP.
`dyn_nroot_larix=y` is active; the JAX year-end root profile collapses to one
wet layer under this forcing. Source inspection confirms that dynamic `nroot`
is SAVE state, is updated sequentially across soil tiles, and feeds both
HYDROL stress and STOMATE root-temperature maintenance.

## Next Required Evidence

Annual STOMATE histories do not contain `nroot`, `humrel`, `vegstress`, or the
HYDROL moisture profile, so they cannot identify the first bad day in this
branch. The next server diagnostic should be narrow and limited to one failed
point (`069.0-119.0`): day-end values for `nroot`, `humrel`, `vegstress`,
selected layer moisture, `MAINT_RESP`, GPP, and NPP until the first mismatch.
No half-hour all-module trace and no multi-year run is needed.

Until that branch is numerically closed, the model must not be described as a
Fortran replacement for all paper PFT14 landpoints. The current supported
claim is nine representative cold-start points closed, explicit snow able to
execute, and selected restart binding closed.

## Compiled Acceptance Pilot

The acceptance policy is now fixed in
`docs/source_audits/paper_scientific_acceptance_policy.md`. The compiled path
runs first; strict mode is invoked only after a scientific failure.

Three-point cold-start 1961 pilot:

- `001.0-071.0` passes both scientific and strict numerical gates; maximum
  derived modelout error is `1.75e-7`.
- `069.0-119.0` fails the scientific gate primarily through NPP relative error
  `6.89e-3`; compiled and strict results agree to floating-point scale, so this
  is not a compiled-transition regression.
- `319.0-057.0` narrowly fails through NPP relative error `1.086e-3`; compiled
  and strict likewise agree.

The two failed points were then run for 2010 from their downloaded
landpoint-specific restart states. Both compiled runs pass the predeclared
scientific gate without strict fallback:

- `069.0-119.0`: maximum key-output normalized error `2.91e-4`;
- `319.0-057.0`: maximum key-output normalized error `3.33e-4`.

This establishes that the mature-state annual compiled calculation closes for
the two high-risk points, while the 1961 cold-start discrepancy remains a
real shared strict/compiled semantic residual to track across the full chain.
It does not prove that the residual disappears before every paper CSV target
year (`2008` for `069`, `2005` for `319`).

Runtime and recovery evidence:

- stable post-compile annual runtime is approximately `83-95 s` per point;
- first point in a fresh process includes about 80 seconds of compilation;
- `run_multiyear_modelout_lite.py --resume-checkpoints on` restored a complete
  real 2010 checkpoint and summary in `0.043 s` without rerunning the model;
- deterministic 12-point and complete 669-point selections are stored under
  `outputs/acceptance/`;
- the acceptance runner supports deterministic process shards while retaining
  one compiled executable across sequential points inside each shard.
