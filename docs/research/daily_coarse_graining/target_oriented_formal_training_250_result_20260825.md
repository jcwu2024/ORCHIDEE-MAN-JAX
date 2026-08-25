# Target-Oriented Formal Update-250 Result (2026-08-25)

## Status

Paid job `14664525` completed all 250 retained updates and the complete frozen
96-window validation panel. The stage is physically valid and improves the
validation science macro by `28.4812%`, well above the preregistered `5%`
pacing threshold. It does not pass the absolute 7/30-day gate. This admits
review and implementation of the update-251-to-1000 continuation; it does not
promote the checkpoint or open 365-day, sealed-test, Gate D2, or product work.

## Executable And Data Identity

- branch: `research/daily-coarse-graining`;
- executable commit: `e363482a75cbec3d6eaaa46555cf7191f3f55350`;
- Slurm job: `14664525`;
- report schema: `target_oriented_training_stage_report_v1`;
- report SHA256:
  `aff1877e289d88740a7105c62d741f6cd75f622e44fbe0f9cf5f3fa387f02d79`;
- prepared update-1-to-250 manifest SHA256:
  `0b1e2c97419fe48ff9ce286173a2de25a20b5c038ae98b9c48aace33285dfcd3`;
- prepared 96-window validation manifest SHA256:
  `1c9081ef4d61da93dd94c8b0856549d570c09b99f61550ffef4b0452b6f5186a`;
- frozen validation panel SHA256:
  `f4952f5fa106aae107fcee6595a6b69897c67b0a54c3a17d3323abc7ed933e79`;
- target contract SHA256:
  `7c9857c0ca57ee65cfcf7514a216a217c3b1661f6cc869c6dfb85d3786bfce5c`;
- training contract SHA256:
  `91faf01123008a60c59038c074779a2a2b4553b82c3739475dd4829fe91ea25b`.

The report declares `teacher_data_regenerated=false`,
`historical_v5_used=false`, and `sealed_test_read=false`.

## Execution And Artifacts

Slurm reports `COMPLETED`, exit code `0:0`, and elapsed time `02:57:58` on
`ibc13b03n02`. The allocation was one V100, four CPUs, and 32 GiB host memory
with a five-hour limit. The log has no traceback, timeout, OOM, CUDA error,
kill, segmentation fault, or hard-gate failure.

The output directory contains both alternating recovery generations,
`stage-update-0250.pkl`, `best.pkl`, `progress.json`, and `report.json`.
The stage checkpoint has SHA256
`099d48c76277cafb04daf868c492200d0271fd6a4a2c6fa29ca6e4bdbe7989a5`;
the best-parameter checkpoint has SHA256
`5711cc40566243af9ae7d118a55ed290b3a26a14219451378bdca65bcbc927c6`.
Both record 250 completed updates and `promotion_eligible=false` because the
absolute science gate did not pass.

Peak live JAX device use was 10,772,760,320 bytes, below the V100 limit. Slurm
reported batch MaxRSS 19,590,408 KiB, below the requested host-memory limit.

## Training Behavior

The update history is complete, unique, and contiguous from 1 through 250.
Every update has zero budget failures, discrete mismatches, nonfinite
predictions, nonfinite states, negative stocks, and unexpected state-mask
mismatches. The 29,266 declared-dynamic mask mismatches are the accepted
report-only `rveget` status and are not hard failures.

The frozen clip threshold was `5,323,655.8449`. Only updates 5, 32, and 143
were clipped, giving clip fraction `0.012`. Update 143 was the single large
gradient outlier (`1.05055e8`, clip scale `0.050675`); the last 25 updates had
no clipping. Thus clipping is not chronically suppressing training.

The per-batch loss is not expected to be monotonic because each update uses a
different diverse effective batch. Its first-25 mean fell from `272.153` to a
last-25 mean of `10.082`; update 250 loss was `2.58422`. The gradient-norm
median was `8.87591e5`, and update 250 remained finite at `1.20504e5`.

At update 250, both long-horizon sources still had nonzero gradients in every
required shared and direct family. Compared with update 1's different batch,
the 7-day shared-encoder norm changed from `0.000605` to `0.054805` and the
30-day norm from `0.000346` to `0.015425`; direct primary-head norms remained
order one. These are conditioning diagnostics rather than a matched-batch
improvement measure, but they show that warmup did not disconnect the
long-horizon objectives from the backbone.

## Frozen 96-Window Result

The authoritative full-panel science macro changed from
`0.9640589738255110` at initialization to `0.6894834688713738` at update 250,
a relative improvement of `28.4811938%`.

| Slice | Horizon | Initial macro | Update-250 macro | Improvement | Passed keys |
| --- | ---: | ---: | ---: | ---: | ---: |
| joint | 7 | 1.0426 | 0.6581 | 36.88% | 4/13 |
| joint | 30 | 1.0986 | 0.8471 | 22.89% | 4/13 |
| spatial | 7 | 0.6641 | 0.4409 | 33.61% | 5/13 |
| spatial | 30 | 0.9536 | 0.6652 | 30.25% | 4/13 |
| temporal | 7 | 1.0044 | 0.7359 | 26.73% | 4/13 |
| temporal | 30 | 1.0211 | 0.7897 | 22.66% | 4/13 |

The 7-day absolute limits are NRMSE at most `0.35` and absolute normalized
bias at most `0.10`; the 30-day limits are `0.25` and `0.07`. A key passes
only when both limits pass.

### Joint

| Quantity | 7-day NRMSE init -> 250 | 7-day bias | 7-day pass | 30-day NRMSE init -> 250 | 30-day bias | 30-day pass |
| --- | ---: | ---: | :---: | ---: | ---: | :---: |
| AGB | 0.0103 -> 0.0048 | -0.0029 | yes | 0.0472 -> 0.0225 | -0.0101 | yes |
| autotrophic respiration | 0.5721 -> 0.3906 | -0.1989 | no | 0.8375 -> 0.9198 | -0.6342 | no |
| BGB | 0.0054 -> 0.0026 | -0.0010 | yes | 0.0216 -> 0.0108 | 0.0002 | yes |
| ecosystem carbon | 2.9152 -> 2.1930 | -1.6604 | no | 2.7012 -> 2.6904 | -2.1670 | no |
| GPP | 1.5202 -> 0.8322 | -0.4804 | no | 1.7049 -> 1.0951 | -0.6548 | no |
| growth respiration | 0.6302 -> 0.3214 | -0.1244 | no | 0.6285 -> 0.3839 | -0.1773 | no |
| heterotrophic respiration | 0.0000 -> 0.0000 | 0.0000 | yes | 0.0000 -> 0.0000 | 0.0000 | yes |
| maintenance respiration | 0.3012 -> 0.3054 | -0.1552 | no | 0.7182 -> 0.9875 | -0.6693 | no |
| NEE | 1.3758 -> 0.6792 | 0.4003 | no | 1.4343 -> 0.6448 | 0.2360 | no |
| NEP | 1.3758 -> 0.6792 | -0.4003 | no | 1.4343 -> 0.6448 | -0.2360 | no |
| nonliving ecosystem carbon | 2.9132 -> 2.1916 | -1.6593 | no | 2.6982 -> 2.6886 | -2.1654 | no |
| NPP | 1.9235 -> 0.9496 | -0.5596 | no | 2.0053 -> 0.9014 | -0.3299 | no |
| total live biomass | 0.0113 -> 0.0054 | -0.0030 | yes | 0.0501 -> 0.0234 | -0.0081 | yes |

### Spatial

| Quantity | 7-day NRMSE init -> 250 | 7-day bias | 7-day pass | 30-day NRMSE init -> 250 | 30-day bias | 30-day pass |
| --- | ---: | ---: | :---: | ---: | ---: | :---: |
| AGB | 0.0071 -> 0.0048 | -0.0035 | yes | 0.0395 -> 0.0210 | -0.0102 | yes |
| autotrophic respiration | 0.4517 -> 0.1576 | -0.0089 | yes | 0.8546 -> 0.5735 | -0.2291 | no |
| BGB | 0.0055 -> 0.0037 | -0.0027 | yes | 0.0214 -> 0.0073 | -0.0013 | yes |
| ecosystem carbon | 1.2997 -> 0.9705 | -0.7581 | no | 2.1728 -> 2.1642 | -1.3776 | no |
| GPP | 0.9588 -> 0.6125 | -0.4684 | no | 1.3686 -> 0.8859 | -0.3900 | no |
| growth respiration | 0.3479 -> 0.2220 | -0.1769 | no | 0.6670 -> 0.4076 | -0.1838 | no |
| heterotrophic respiration | 0.0000 -> 0.0000 | 0.0000 | yes | 0.0000 -> 0.0000 | 0.0000 | yes |
| maintenance respiration | 0.5432 -> 0.2514 | 0.1293 | no | 0.8363 -> 0.4380 | -0.1466 | no |
| NEE | 1.0923 -> 0.7455 | 0.5480 | no | 1.2426 -> 0.5787 | 0.2673 | no |
| NEP | 1.0923 -> 0.7455 | -0.5480 | no | 1.2426 -> 0.5787 | -0.2673 | no |
| nonliving ecosystem carbon | 1.2987 -> 0.9698 | -0.7575 | no | 2.1702 -> 2.1624 | -1.3765 | no |
| NPP | 1.5272 -> 1.0423 | -0.7662 | no | 1.7374 -> 0.8091 | -0.3737 | no |
| total live biomass | 0.0089 -> 0.0060 | -0.0044 | yes | 0.0437 -> 0.0209 | -0.0091 | yes |

### Temporal

| Quantity | 7-day NRMSE init -> 250 | 7-day bias | 7-day pass | 30-day NRMSE init -> 250 | 30-day bias | 30-day pass |
| --- | ---: | ---: | :---: | ---: | ---: | :---: |
| AGB | 0.0094 -> 0.0062 | -0.0028 | yes | 0.0428 -> 0.0220 | -0.0139 | yes |
| autotrophic respiration | 0.6234 -> 0.5429 | 0.0041 | no | 0.6046 -> 0.6050 | -0.4542 | no |
| BGB | 0.0054 -> 0.0042 | -0.0005 | yes | 0.0200 -> 0.0109 | 0.0005 | yes |
| ecosystem carbon | 2.7209 -> 1.9773 | -1.6454 | no | 2.8329 -> 2.8230 | -2.4022 | no |
| GPP | 1.3691 -> 1.0242 | -0.2776 | no | 1.5091 -> 0.9088 | -0.5852 | no |
| growth respiration | 0.6307 -> 0.4486 | -0.0996 | no | 0.7730 -> 0.4036 | -0.2486 | no |
| heterotrophic respiration | 0.0000 -> 0.0000 | 0.0000 | yes | 0.0000 -> 0.0000 | 0.0000 | yes |
| maintenance respiration | 0.5625 -> 0.5187 | 0.0845 | no | 0.2367 -> 0.5333 | -0.3826 | no |
| NEE | 1.2965 -> 0.9010 | 0.3328 | no | 1.2887 -> 0.6227 | 0.3069 | no |
| NEP | 1.2965 -> 0.9010 | -0.3328 | no | 1.2887 -> 0.6227 | -0.3069 | no |
| nonliving ecosystem carbon | 2.7191 -> 1.9761 | -1.6443 | no | 2.8298 -> 2.8208 | -2.4004 | no |
| NPP | 1.8127 -> 1.2597 | -0.4653 | no | 1.8017 -> 0.8706 | -0.4291 | no |
| total live biomass | 0.0107 -> 0.0073 | -0.0026 | yes | 0.0462 -> 0.0227 | -0.0111 | yes |

## Interpretation

The candidate has learned a useful signal. GPP improves by `25.19%` to
`45.26%` across cells; NPP improves by `30.51%` to `55.05%`; NEE and NEP
improve by `30.51%` to `55.05%`. AGB, BGB, and total live biomass pass every
7/30-day cell with small bias.

The result is nevertheless far from the scientific target. Ecosystem and
nonliving ecosystem carbon remain the dominant failures. Their 30-day errors
improve by less than `0.4%` in every slice and retain large negative biases.
GPP/NPP errors remain substantially above their absolute thresholds.
Maintenance respiration is inconsistent and worsens in joint 7/30-day and
temporal 30-day cells; joint 30-day autotrophic respiration also worsens.

Because update 250 is the end of linear warmup and the first point at the
peak learning rate, this stage does not establish a converged loss floor. The
large full-panel improvement, low clipping fraction, healthy physical gates,
and continued long-horizon gradient connectivity support continuing the
same architecture and frozen objective through update 1000. Changing the
architecture or loss now would discard the cleanest available test of whether
the model can improve after warmup.

## Gate Decision And Next Operation

The report classification is:

- `absolute_short_horizon_passed=false`;
- `relative_macro_improvement=0.2848119383`;
- `pacing_improvement_passed=true`;
- `status=rejected` for promotion;
- `next_gate=review_update_251_to_1000_continuation`.

The active task is therefore to implement and qualify a true continuation
runner that:

1. starts from the exact update-250 stage checkpoint, including AdamW state;
2. consumes only newly prepared v6 train/train updates 251 through 1000;
3. preserves the frozen architecture, objective, schedule, clip threshold,
   validation panel, and absolute gates;
4. writes bounded recovery plus the update-1000 stage checkpoint;
5. requires zero physical hard failures and at least another `5%` macro
   improvement from update 250 before any further training review.

The current runner is intentionally hard-coded to end at update 250, so it
must not be repurposed by merely placing the stage checkpoint in its recovery
directory. The continuation path requires an explicit checkpoint-handoff and
absolute-update regression before new batch preparation or paid submission.
