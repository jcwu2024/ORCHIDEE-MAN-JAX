# Target-Oriented Formal Update-1000 Result (2026-08-26)

## Status

Paid job `14674142` completed the exact update-251-to-1000 continuation and
the unchanged frozen 96-window validation panel. The science macro improved
from the update-250 value `0.6894834689` to `0.3408543729`, a further
`50.5638%` reduction. All physical hard gates pass and the preregistered 5%
pacing gate passes. The absolute 7/30-day gate still fails, so this checkpoint
is not promoted and 365-day, sealed-test, Gate D2, and product work remain
closed.

The central result is no longer a slow-carbon failure. Mean 30-day ecosystem
carbon NRMSE fell from `2.55920` to `0.04742`, and nonliving ecosystem carbon
fell from `2.55725` to `0.04726`; both pass all three model-selection slices.
The unresolved failures are direct productivity and respiration, especially
GPP, NPP, and maintenance/autotrophic respiration.

## Executable And Artifact Identity

- branch: `research/daily-coarse-graining`;
- executable commit: `433d2869bb1bf8f7657d7bd3bf18930a1c536b2d`;
- Slurm job: `14674142`;
- output directory:
  `runtime/outputs/training/target-oriented-training-formal-u1000-433d286`;
- report schema: `target_oriented_training_stage_report_v1`;
- report SHA256:
  `8759b4bd3a5e5c44a56b853e09e6c6206f3b1e1544bf4e3d7e6beacd976f414f`;
- update-1000 stage-checkpoint SHA256:
  `a566f9d95726984dc03cbf1f1a183dd0d1eae9e3479ce5055c64e571ef6a16c7`;
- analysis JSON SHA256:
  `b10b325826c2a64e88be35c0a8e2d5034e57bf0cb53a76351e818b21c2631e5e`;
- analysis Markdown SHA256:
  `bf79e64a887173f4c73f8010b39645a6b5981d83a5660a130f345bc918f2ce20`.

The report again declares `teacher_data_regenerated=false`,
`historical_v5_used=false`, and `sealed_test_read=false`. The continuation
restored the exact update-250 parameters and AdamW state and consumed only the
previously admitted v6 artifacts for absolute updates 251 through 1000.

## Execution And Training Integrity

Slurm reports `COMPLETED`, exit code `0:0`, elapsed time `08:12:26`, and node
`ibc13b03n02`. MaxRSS was `19,316,328 KiB`. All 750 continuation records are
present and contiguous. The stage checkpoint records exactly 1000 completed
updates.

There are no budget failures, discrete mismatches, nonfinite predictions,
nonfinite states, negative stocks, or unexpected state-mask mismatches.
Continuation clipping is `0/750`; cumulative clipping remains only `3/1000`
from the update-1-to-250 parent. Gradient clipping therefore did not limit
this stage.

## Frozen 96-Window Result

| Slice | Horizon | Update-250 macro | Update-1000 macro | Improvement | Passed keys |
| --- | ---: | ---: | ---: | ---: | ---: |
| joint | 7 | 0.65808 | 0.35815 | 45.58% | 6/13 |
| joint | 30 | 0.84713 | 0.42266 | 50.11% | 6/13 |
| spatial | 7 | 0.44090 | 0.19434 | 55.93% | 8/13 |
| spatial | 30 | 0.66517 | 0.32510 | 51.13% | 6/13 |
| temporal | 7 | 0.73593 | 0.43100 | 41.44% | 6/13 |
| temporal | 30 | 0.78969 | 0.31388 | 60.25% | 6/13 |

No cell passes all 13 keys, so `absolute_short_horizon_passed=false`.

| Quantity | 7-day mean NRMSE | Passed cells | 30-day mean NRMSE | Passed cells |
| --- | ---: | ---: | ---: | ---: |
| AGB | 0.00475 | 3/3 | 0.02044 | 3/3 |
| BGB | 0.00335 | 3/3 | 0.01063 | 3/3 |
| total live biomass | 0.00571 | 3/3 | 0.02193 | 3/3 |
| ecosystem carbon | 0.00998 | 3/3 | 0.04742 | 3/3 |
| nonliving ecosystem carbon | 0.01001 | 3/3 | 0.04726 | 3/3 |
| GPP | 0.69096 | 0/3 | 0.84249 | 0/3 |
| NPP | 1.02315 | 0/3 | 0.85922 | 0/3 |
| maintenance respiration | 0.37243 | 0/3 | 0.57437 | 0/3 |
| growth respiration | 0.36057 | 1/3 | 0.36242 | 0/3 |
| autotrophic respiration | 0.31724 | 1/3 | 0.58517 | 0/3 |

Heterotrophic respiration remains exact in all six cells through its retained
exact path. NEE and NEP remain failed derived productivity keys and track the
remaining NPP error.

## What Training Learned

The continuation decisively learned the previously missing slow-carbon state
transition. Compared with update 250, 30-day ecosystem and nonliving carbon
improve by `98.147%` and `98.152%`; their 7-day counterparts improve by about
`99.42%`. AGB, BGB, and total live biomass remain accurate in every cell.

The flux side did not make comparable progress. NPP improves only `5.60%` at
7 days and `0.132%` at 30 days. GPP improves `16.04%` and `12.54%` but remains
well above threshold. Seven-day maintenance and growth respiration worsen by
`3.88%` and `9.05%` respectively.

Training dynamics support this split. The 30-day stock MSE falls from
`0.480894` over updates 251-300 to `0.000988` over updates 951-1000, while the
30-day flux MSE remains noisy and ends at `0.235390`. At update 1000, the
shared-backbone gradient norms are `6.30695` for one-day/process,
`1.89030` for 7-day science, and `0.493817` for 30-day science. The
long-horizon paths are connected, and direct primary heads receive order-one
science gradients; slow-carbon connectivity is no longer the central
blocker.

## Gate Decision And Next Operation

The report classification is:

- `absolute_short_horizon_passed=false`;
- `relative_macro_improvement=0.5056380779` from update 250;
- `pacing_improvement_passed=true`;
- `hard_gates_passed=true`;
- `next_gate=review_update_1001_to_3000_continuation`.

That machine action admits review; it does not justify blind continuation.
The next active task is a zero-update, train-only flux-gradient variance audit
at the update-1000 checkpoint. It must measure the variance and pairwise cosine
of GPP, NPP, and respiration gradients across independent batches, with
particular attention to the four 30-day windows currently used per update. It
must compare the realized estimates from 4 versus 8 and 16 30-day windows
without changing the architecture or retaining an optimizer update.

Only if that audit finds a stable, actionable reduction in flux-gradient
noise should a bounded update-1001-to-3000 training design be proposed. If
gradient estimates are already stable, the result instead points to objective
or output-head expressivity and should prevent another expensive unchanged
continuation. No paid computation is authorized by this result document.
