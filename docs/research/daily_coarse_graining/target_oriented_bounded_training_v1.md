# Target-Oriented Bounded Training v1

Status: **frozen for local implementation and training-step preflight; paid
training is not yet authorized**.

Date: 2026-08-21.

## Purpose

This is the first formal training design for
`target_oriented_hierarchical_coordinate_attention_v1`. It is not a new
candidate-versus-control comparison. The candidate must pass the absolute
7/30-day science gates in the frozen target-oriented operator contract; small
relative gains over a weak historical network cannot promote it.

The machine contract is
`manifests/coarse_graining/target_oriented_bounded_training_v1.json`. It binds
the accepted Teacher-v6 dataset, candidate manifest, target objective, and GPU
admission evidence. Its SHA256 is
`91faf01123008a60c59038c074779a2a2b4553b82c3739475dd4829fe91ea25b`.

## Effective Update

Every optimizer update aggregates 28 mutually distinct point-years from 28
distinct landpoints:

| Objective input | Effective count | Point-years | Teacher days |
| --- | ---: | ---: | ---: |
| One-day state and process samples | 32 | 16, two days each | 32 |
| Seven-day science windows | 8 | 8 | 56 |
| Thirty-day science windows | 4 | 4 | 120 |
| Total | - | 28 | 208 |

All rows come from the accepted Teacher-v6 `train/train` split. The 28
point-years in one update share one runtime year and use 28 distinct
landpoints; eligible years rotate in one deterministic hash-ranked cycle.
This keeps the retained-tail executable scaffold static within a compiled
update while every sample carries its own dynamic retained-tail arrays.
Landpoint identity is not a network feature, and no point-year or landpoint
repeats within an update. The first annual row is excluded. This uses much
more of the available data per update than the earlier one-point-year
diagnostics without requiring a large tensor batch to reside in GPU memory at
once.

## Memory Strategy

The effective batch is divided into microbatches of four one-day samples, one
seven-day window, and one thirty-day window. Thirty-day scans use JAX
rematerialization: activations are recomputed during backpropagation instead
of retaining all 30 days. Component sufficient statistics are accumulated so
that MSE and signed-bias terms are exactly those of the full effective batch;
averaging per-microbatch bias losses is forbidden. All component gradients are
combined with the frozen `1.0/1.0/1.0/0.1` coefficients, followed by one
optimizer update.

This is single-GPU gradient accumulation, not data-parallel multi-GPU
training. The zero-update V100 admission measured only 2.49 GiB peak live
memory for one-day differentiation, but the actual full-objective step must be
measured before selecting a paid resource envelope. Admission requires at
most 28 GiB actual JAX `peak_bytes_in_use`, leaving headroom on a 32 GB V100.

## Optimizer

The frozen optimizer is AdamW with beta1 `0.9`, beta2 `0.95`, epsilon `1e-8`,
weight decay `0.01`, 250-update linear warmup to `1e-4`, and cosine decay to
`1e-5` by update 3000.

The global gradient clip is not guessed as `1.0`. Before the first retained
update, eight complete train/train effective batches measure the full
unclipped gradient norm. The 90th percentile becomes the frozen threshold.
Every update reports its unclipped norm, and every milestone reports the clip
fraction. Validation data cannot tune the threshold.

## Stages And Checkpoints

The maximum bounded budget is 3,000 updates, with validation at initialization
and updates 250, 1,000, and 3,000. At 208 Teacher days per update, the maximum
budget exposes the objective to 624,000 day-usages while repeatedly mixing
space, years, and horizons.

The 7/30-day panel contains 16 distinct-landpoint windows per horizon for each
temporal, spatial, and joint non-test slice. Promotion uses only the absolute
per-key RMSE and signed-bias gates in the target-oriented operator contract.
A 5% macro improvement between milestones is only a resource pacing rule: it
may authorize continuation but cannot promote a scientifically inaccurate
model.

The preflight retains no update and writes no checkpoint. Formal stages write
one recovery record every 50 updates, retain at most two recovery generations,
and keep a best checkpoint only when hard failures are zero. Stage-boundary
checkpoints allow paid runs to remain short and separately reviewable.

## Next Operation

The exact microbatch sufficient-statistics accumulator, multi-point-year
prepared reader, dynamic retained-tail day-input path, and clipped AdamW
proposal are implemented. The eight real-v6 batches and first free-`gln01`
preflight are complete. It measured about 8.15 GiB peak live memory and 35.65
seconds per hot effective step, but all eight gradients were NaN because the
one-day state anchor amplified one near-constant DOC coordinate and counted it
again through `other`.

The follow-up state-anchor amendment is frozen separately in
`target_oriented_state_anchor_amendment_v1.json`. It does not change this
batch, optimizer, milestone, or checkpoint design, so the prepared batches
remain valid. The active operation is one representative free-node component-
gradient probe under the amended loss, followed by the full eight-batch
preflight only if all three gradient sources are finite.

No Teacher regeneration, historical v5 access, paid submission, 365-day
rollout, long-held validation, Gate D2, sealed-test access, or product
acceptance is opened by this design.
