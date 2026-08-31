# Structured-GPP Broad Training Stage 1 Result

Date: 2026-08-31

Status: accepted; freeze a fresh independent 7/30-day science screen next

Slurm job: `14883081`

Executable commit: `f040679d1bc22f1dfe2d93ac0439edc86a804239`

Report SHA256:
`5cfab5a693be4c58fd2726420106f2844f93703e5ca8e2910268498154b2bee9`

Selected update-1000 checkpoint SHA256:
`3b173b8980c37b49db3d1547dddad2ab98ec00c733e2f564469490ee382d4ec0`

Final recovery SHA256:
`4cd8a249bb9c1f6c6ef1c65239c6ce5b72cb2e9f5b85414961c404c035dc975c`

## Execution

Job `14883081` completed with Slurm state `COMPLETED`, exit code `0:0`, and
elapsed time `02:30:39`. It executed all 1,000 updates and consumed the frozen
208,000 unique train/train days. Training did not stop early. Seven updates
were clipped, all during the first 100 updates; later gradients remained below
the frozen threshold.

At the published additive rates, the observed allocation corresponds to about
CNY 1.41 of CPU and CNY 5.52 of GPU time, or CNY 6.93 total before any
center-side billing granularity. JAX peak live memory was about 2.66 GiB.

## Fixed Train-Only Monitor

The aggregate direct-GPP monitor loss changed as follows:

| Optimizer update | Loss | Reduction from initial | Improved point-years |
| ---: | ---: | ---: | ---: |
| 0 | 0.82876149 | 0.00% | 0/128 |
| 250 | 0.40201390 | 51.49% | 112/128 |
| 500 | 0.18334220 | 77.88% | 122/128 |
| 1000 | 0.08593347 | 89.63% | 125/128 |

The selected aggregate normalized bias improved in magnitude from
`-0.43192778` to `-0.09223345`. The final checkpoint therefore passes all three
frozen monitor requirements: at least 5% loss reduction, at least 96 of 128
point-years improved, and no increase in absolute aggregate bias.

## Structural And Physical Gates

All physical hard counts were zero at updates 0, 250, 500, and 1,000:
nonfinite prediction and state, budget failure, discrete mismatch, negative
stock, and unexpected mask mismatch. The 19 declared-dynamic mask differences
remain the established reported nonfailure count.

All 12,879,363 specialist coordinates were trainable. All 93,504,164 retained
parent coordinates remained exactly unchanged. No Teacher data were
regenerated, no v5 data were used, and no validation, sealed-test, consumed
96-window, or rollout target values were read.

## Decision

The broad-data stage is accepted and selects update 1,000. This is strong
evidence that the former structured-GPP failure was substantially caused by
shallow repeated exposure rather than an inability of the unchanged Attention
mechanism to learn GPP.

This is not yet long-horizon or product acceptance. The selected checkpoint
remains non-promotable until a fresh, target-blind, independent 7/30-day screen
is frozen before values are read and passes GPP plus downstream science gates.
Do not reuse the consumed 96-window panel for tuning, and do not open 365-day,
Gate D2, sealed-test, or product acceptance yet.
