# Source-Backed Carbon-Stock Domain Projection

Date: 2026-07-26

## Question

Explain the single extreme batch in the rejected `on_policy_pushforward_v1`
run and determine whether it exposes a dynamic undefined-status defect, an
invalid recursive state, or a generic network explosion.

## Deterministic Reproduction

The fixed training RNG selected landpoint `295.0-113.0`, year 1966, and
terminal days 15, 57, 64, and 182. Before the fix, their one-step losses were:

```text
0.099742, 0.565832, 0.098369, 562889.222675
```

The Day-182 error was dominated by `carbon_32l`, `DOC`, and `deepC_peat`.
Teacher values reached about `7.27e8`, and the fast-day plus next-state
comparison contained 980 defined-status mismatches.

The terminal model state and clean Teacher state had identical
defined/undefined and discrete masks. Their `veget_max`, `everywhere`,
`pft_present`, `altmax`, and fixed cryoturbation controls also matched exactly.
The failure was therefore not a missing dynamic mask transition.

## Root Cause

The unconstrained residual network produced negative material stocks before
the same-state Teacher query:

| Field | Negative values on Day-182 terminal state | Width |
|---|---:|---:|
| `carbon_32l` | 80 | 192 |
| `DOC` | 383 | 896 |
| `deepC_peat` | 14 | 64 |

The real `soilcarbon_leak` and `PERMA_PEAT` processes then applied exponential
decomposition and sequential layer redistribution to an invalid state. The
first three samples produced NaNs in the carbon fields; Day 182 produced the
large finite artifact.

This is a source-domain violation. In
`fortran_source/ORCHIDEE/src_stomate/stomate_soilcarbon.f90`,
`soilcarbon_leak` lines 1348-1437 and 1605-2303 treat these arrays as material
stocks, while `cryoturbate_doc_POC` lines 2916-3091 explicitly diagnoses
negative `carbon_32l` and `DOC` as invalid.

## Fix

Commit `5414d8c` registers exactly the three source-nonnegative fast-day stock
families in the contract-derived target representation:

- `ok_leak.carbon_32l`
- `ok_leak.DOC`
- `ok_leak.deepC_peat`

Both NumPy inference and compiled JAX rollout project those physical values to
`>= 0` at the single reconstruction boundary before retained STOMATE receives
them. Signed fluxes and all other outputs are unchanged. Existing valid
positive values are unchanged, and source-defined undefined values are still
restored after the projection.

## Verification

The same checkpoint and exact four samples after the fix produced:

```text
0.099889, 0.094551, 0.097007, 0.116539
```

The extreme loss disappeared and defined-status mismatches fell from 980 to
zero. The maximum loss is now `0.116539`.

The frozen four-way seven-day baseline was also rerun under the promoted
physical reconstruction semantics:

| Split | Old Day-7 RMSE | Projected Day-7 RMSE |
|---|---:|---:|
| train/train | 0.145471 | 0.144765 |
| train/validation-year | 0.134233 | 0.133391 |
| validation-point/train-year | 0.519886 | 0.519286 |
| joint validation | 0.429058 | 0.428353 |

All four cases retain zero defined-status and discrete mismatch. The small
global improvement confirms that the projection fixes invalid off-manifold
states without changing the prior scientific diagnosis: spatial
generalization remains the dominant limitation, and seen-condition
`litterpart` still grows from `0.540939` to `3.651537` by Day 7.

## Evidence

Local reports:

```text
outputs/research/daily_coarse_graining/pushforward-outlier-5904b38/report.json
outputs/research/daily_coarse_graining/pushforward-outlier-5414d8c/report.json
outputs/research/daily_coarse_graining/four-way-projected-5414d8c/
```

Related unit, compiled-rollout, Ruff, and syntax checks passed: 29 tests.

## Decision

Keep `canonical_multistep_v1` as the accepted checkpoint baseline. The
`on_policy_pushforward_v1` candidate remains rejected; removing its numerical
outlier does not repair its spatial and named-state gate failures.

The next architecture experiment must combine:

1. process-aware state encoding;
2. persistent parameter/static-condition modulation rather than one flat
   condition bottleneck;
3. the promoted source-backed carbon-stock domain projection.

Only after that bounded architecture A/B should a source-selected spatial
Teacher-data pilot be generated.
