# Coupled GPP Direct Recalibration Smoke V1

## Question

The update-3 common-direction audit shows that the existing coupled-GPP
Attention candidate has a broad direct finite-step interval even though its
fixed-weight AdamW proposal fails. This smoke asks whether that geometry can
be recomputed and directly realized across three consecutive scheduled macro
updates, rather than only at one state.

The frozen machine contract is
`manifests/coarse_graining/target_oriented_coupled_gpp_direct_recalibration_smoke_v1.json`,
canonical SHA256
`f7f0efc11fb716ec8946f70539b06585cce0c106087cd63de79acb5db648e9cf`.

## State And Schedule

The runner loads the unchanged update-1 recovery from paid job `14903328` and
reconstructs accepted fresh-moment update 2 on artifacts
`326,935,542,913,897,836,887,863`. That state exists only in memory.

The new rule is then evaluated consecutively on scheduled updates 3, 4, and
5:

| Macro update | Train/train artifacts |
| ---: | --- |
| 3 | `968,750,987,728,953,253,646,937` |
| 4 | `481,672,900,294,998,668,433,535` |
| 5 | `965,351,272,363,660,670,548,740` |

Each update covers 1,664 direct-GPP teacher days, 256 one-day reachable-state
transitions, 64 seven-day windows, and 32 thirty-day windows. Validation and
sealed-test values remain unread.

## Recalibrated Direct Step

At every current state, the runner independently computes the four source
gradients, their `4x4` Gram matrix, and the minimum-norm simplex direction over
unit source gradients. Every source cosine must be at least `1e-4`.

The accepted geometry is realized directly, without AdamW first/second
moments or coordinatewise scaling. The fixed global parameter displacement is
`0.007758973351050004`, the full scale accepted by the update-3 audit. The
direction is recomputed after every accepted proposal; neither simplex weights
nor gradients are inherited across macro updates.

## Finite-Step Gate

Each proposed state is evaluated on all eight artifacts that generated its
gradient. Direct GPP, reachable one-day state, 7-day science, and 30-day
science must each be non-worsening at relative tolerance `1e-6`, and the
weighted objective must decrease strictly. All one-day, 7-day, and 30-day
physical hard counts must remain zero.

The runner stops at the first missing common direction or rejected finite
step. It does not search smaller scales within this smoke: the purpose is to
test the frozen full-step repeated rule, not to tune against each scheduled
group.

## Retention And Decisions

All proposals are discarded after reporting. The parent recovery, source
checkpoint, and fixed parameters must remain unchanged; no checkpoint is
written.

If updates 3-5 all pass, the next action is a separate bounded direct-
recalibration training contract and paid-resource review. It is not an
automatic paid launch. A missing common direction redirects the objective
hierarchy toward long-horizon-primary optimization with GPP/state protection.
A common direction with a failed full step triggers a state-local finite-step
review without changing the architecture.

Fresh validation, 365-day rollout, Gate D2, sealed test, and product
acceptance remain closed.

## Local Qualification

The frozen loader, exact schedule, zero-retention runner boundary, CLI, and
free-node launcher pass 36 focused and related coupled-GPP/reachability tests.
Ruff, Python compilation, Bash syntax, and whitespace checks pass. One free
`gln01` execution is the active gate.
