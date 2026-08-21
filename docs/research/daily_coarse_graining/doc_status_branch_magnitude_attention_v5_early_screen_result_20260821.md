# DOC Status And Branch-Magnitude Attention v5 Early-Screen Result

Date: 2026-08-21.

Status: execution completed normally; the frozen epoch-10 gate failed and the
candidate is rejected. No checkpoint was written.

## Execution

Approved paid job `14632388` ran commit
`e9fc0d816dcebf870d618e7afe3660a527792e0a` on node `ibc13b02n03`. It used one
V100, four CPUs, the frozen 364-day train/train point-year
`001.0-071.0`/1961, logical batch size 8, and exact size-2 gradient
microbatches. Slurm reports `COMPLETED`, exit code `0:0`, and elapsed time
`01:14:07` within the two-hour limit.

The report is
`runtime/outputs/training/doc-status-branch-magnitude-attention-v5-litter-10-epoch-e9fc0d8/report.json`
on Explore1000, with SHA256
`7acc06d30da713b177849793b9c332938040163a9e929b7a9688b525fcbac1ba`.
The run completed all 10 epochs and 460 updates. All observed gradients were
finite, first/final selected-block and selected-head coverage passed, counts
were unchanged, every prediction/state/budget/inventory hard count was zero,
and peak JAX bytes in use were `23,662,248,192` (`22.04 GiB`). It used only v6
data, regenerated no Teacher data, used no v5 rows, reconstructed no 48 forcing
records, executed no 48-step candidate scan, and wrote no checkpoint.

## Frozen Gate

The epoch-5 review passed. Litter/POC/DOC loss had fallen `74.5650%`; DOC
incoming and outgoing losses had fallen `24.3175%` and `18.5776%`, so neither
exceeded the frozen two-times-initial rejection threshold. All hard and finite
checks passed, and training continued to epoch 10.

The epoch-10 gate failed:

| Gate | Required | Observed | Result |
| --- | ---: | ---: | --- |
| litter/POC/DOC aggregate reduction | `>=90%` | `83.1178%` | fail |
| DOC incoming reduction | `>=50%` | `29.6589%` | fail |
| DOC outgoing reduction | `>=50%` | `26.1806%` | fail |
| free-to-adsorbed reduction | `>=0%` | `16.6932%` | pass |
| advective reduction | `>=0%` | `58.9182%` | pass |

This is an absolute-gate failure. V4 is mechanism attribution only and is not
used to weaken the verdict.

## Interpretation

V5 removes the catastrophic v4 endpoint trade: both incoming and outgoing
physical losses decrease, and both monitored transport fields improve. The
status and latent branches are trainable: status macro cross entropy,
incoming-coordinate smooth-L1, and outgoing-coordinate smooth-L1 fall
`75.4655%`, `78.4438%`, and `54.2036%`. The complete DOC composite falls
`71.6314%`.

The remaining bottleneck is the materialized physical endpoint. Its
status-macro physical MSE falls only `29.2932%`, closely matching the slow
incoming/outgoing physical-field reductions. The evidence therefore does not
reject the coordinate-aligned Attention backbone or show a dead status/latent
gradient path. It rejects this soft-status-times-magnitude materialization and
loss balance as a sufficiently capable formal endpoint head under the frozen
screen.

The aggregate training loss continued to decrease through epoch 10, but the
screen was deliberately frozen to prevent a weak candidate from consuming a
full 100-epoch run. No continuation is authorized.

## Decision

Do not run the V5 100-epoch gate, retain a V5 checkpoint, repeat V4/V5, or open
multi-point training and rollout. The next operation is a mechanism review
focused on converting the well-learned status/branch latents into accurate
physical endpoints without discarding the accepted coordinate-aligned source
bank. Any successor mechanism and its bounded absolute gate must be discussed
and frozen before implementation or GPU work. Sealed tests, Gate D2, and
product acceptance remain closed.
