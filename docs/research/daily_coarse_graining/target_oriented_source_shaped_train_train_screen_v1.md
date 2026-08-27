# Target-Oriented Source-Shaped Train/Train Screen v1

Date: 2026-08-27

Status: frozen; all pre-training admissions accepted and the paid runner awaits authorization

Machine contract:
`manifests/coarse_graining/target_oriented_source_shaped_train_train_screen_v1.json`

Contract SHA256:
`a06beb815834142f44939022eea6ebd044e0a096b5d3c443754a9d87fb0f8cee`

## Purpose

This screen asks whether the source-shaped maintenance mechanism produces a
substantial 7/30-day improvement, not whether it narrowly outranks a weak
network. The unchanged update-1000 parent supplies scores on identical windows
only so the effect has a coordinate system. It is not a promoted control arm.

No Teacher data is generated. The screen reuses accepted v6 train/train batch
artifacts and does not evaluate validation or sealed-test values.

## Frozen Data Separation

The evaluation panel uses prepared updates `251, 358, 465, 572, 679, 786,
893, 1000`. Each group contains all eight 7-day and four 30-day windows from
its effective batch. These are the same groups already evaluated for the
immutable parent report, so parent execution is not repeated.

If training is later authorized, eight separate artifacts calibrate the
branch-gradient clip threshold and artifacts `252..351` supply 100 optimizer
updates. Neither set overlaps the eight evaluation groups. This is a bounded
train/train learnability screen, not validation.

## Ordered Execution

1. Evaluate the source-shaped zero-correction initialization on all eight
   groups with zero optimizer state and zero updates.
2. If outputs, states, budgets, and hard counts are valid, separately request
   authorization for a one-update free-GPU training admission.
3. The admitted paid stage may run at most 100 branch-only updates and evaluate
   the same panel only at initialization and completion.

The branch-only stage freezes all retained parent parameters. Only the two new
`768 -> GELU(256) -> 1 -> tanh` branches, 394,242 parameters in total, receive
AdamW updates. The full existing `1/1/1/0.1` effective-batch objective remains
unchanged. A fresh optimizer uses ten-update warmup to `3e-4`, cosine decay to
`1e-4` at update 100, and a train/train p90 branch-gradient clip calibrated
before the first update.

## Science Gate

At both 7 and 30 days, the median maintenance-respiration score must fall by at
least 20% relative to the immutable parent scores. Median score change must be
nonpositive for GPP, growth respiration, total autotrophic respiration, NPP,
AGB, BGB, total live biomass, and ecosystem carbon. No protected key in any
group may regress by more than 1%, and all hard/nonfinite counts must be zero.

Passing these relative gates without passing the absolute target-oriented
contract means `source_mechanism_supported_not_promoted`. It can justify a
separately designed broader training stage, but it cannot open validation,
365-day rollout, Gate D2, sealed test, or product acceptance.

## Authorization Boundary

The zero-update panel and one-update free-GPU admission are complete. The
admission created a fresh optimizer, applied one ephemeral branch-only update,
and discarded the proposal without writing a checkpoint. The paid 100-update
stage still requires new explicit user authorization after its direct
branch-autodiff resource envelope is known.

## Zero-Update Result

The complete eight-group panel passed execution at commit `250986f`; report
SHA256 is `6e7302fe...e094291`. With no training, maintenance scores improve by
`73.61%` at 7 days and `87.35%` at 30 days. The protection gate fails because
the fixed source correction is too large for several 7-day groups, including
median regressions in growth respiration, NPP, AGB, BGB, and total live
biomass. All hard and nonfinite counts are zero.

This is evidence for the new mechanism, not acceptance of its current output.
See `target_oriented_source_shaped_zero_panel_result_20260827.md`.

## One-Update Admission Result

The free admission passed at commit `0ed926b`; report SHA256 is
`c2743cbc...34ea59`. Eight calibration batches freeze p90 branch clip
`0.1973867`. Artifact 252 then produces raw branch norm `0.2293467`, uses clip
scale `0.86065`, changes both branches, and leaves all retained parameters
unchanged. Loss and all proposed parameters and optimizer values are finite;
hard failures are zero before and after. No checkpoint or recovery was
written.

The direct branch-subtree admission subsequently passed at commit `d965bea`;
report SHA256 is `5852d996...e58538`. It matches the extracted complete
gradient with zero failing coordinates, reduces hot effective-batch time from
`36.85 s` to `25.18 s`, and reduces JAX peak live memory from about 7.64 GiB
to 2.24 GiB. See
`target_oriented_source_shaped_branch_autodiff_admission_20260827.md`.

## Paid-Stage Admission

The 100-update runner, compact branch-only recovery/checkpoint, exact
`252..351` schedule, update-100 eight-group screen, and Slurm wrapper are
locally qualified at commit `9224398`. The frozen request is one `gnall` node,
eight CPU cores, one V100, 32 GiB, three hours, and CNY 8.28 worst case. It has
not been submitted. See
`target_oriented_source_shaped_branch_training_admission_20260827.md`.
