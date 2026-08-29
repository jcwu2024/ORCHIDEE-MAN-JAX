# Structured-GPP Batch-Four Joint Training v1

Date frozen: 2026-08-30

Machine contract:
`manifests/coarse_graining/target_oriented_structured_gpp_joint_training_v1.json`

Canonical contract SHA256:
`168d44bdf8e1094ab4d683edaf8ebf576ff7dd27053d754419956cce30798832`

## Purpose

The direct-GPP warm start established specialist capability, but the first
single-artifact joint AdamW proposal regressed its 30-day objective. The
subsequent heterogeneity audit and discarded finite-step preflight established
that four-artifact accumulation is the minimum tested effective batch that
consistently gives direct, 7-day, and 30-day aggregate descent. This contract
turns that result into one bounded training stage. It does not revisit the
structured-GPP Transformer architecture and does not promote any checkpoint.

## Frozen Parent And Trainable Boundary

- Load the accepted update-250 structured-GPP specialist and its inherited
  AdamW state.
- Train only the `12,879,363` structured-GPP specialist parameters.
- Keep all `93,504,164` retained parent parameters outside the optimizer and
  exactly unchanged.
- Continue AdamW's absolute step from 250, with learning rate `4e-6`, frozen
  beta/epsilon/weight-decay values, and global norm clip
  `46.802861793665286`.
- Do not regenerate v6 Teacher data, join v5 data, reconstruct half-hourly
  forcing, or execute a 48-step candidate scan.

## Data And Update Semantics

The candidate pool is the 250 train/train prepared artifacts from 512 through
763, excluding the train-only evaluation artifacts 572 and 679. A PCG64
generator with seed `20260830` is advanced continuously across four epochs.
Each epoch shuffles the 250 candidates, drops the final two, and forms 62
ordered groups of four. The complete materialized schedule has SHA256
`770bae619e427820df2c6ad71c48bab3180a33bc1a1784f37e04680a7a455f70`.
Every candidate is used at least once over the bounded stage.

For each four-artifact group, compute direct-GPP, 7-day, and 30-day gradients
for each artifact at the same current parameter state. Arithmetic-mean each
source across the four artifacts, then form exactly one combined gradient:

```text
g = mean(g_direct) + 5.065197169020285 * mean(g_7day)
                      + 4.594433600967992 * mean(g_30day)
```

Apply one AdamW update to that combined gradient. Thus four epochs contain at
most 248 optimizer updates and 992 artifact presentations, ending at absolute
optimizer step 498.

## Train-Only Epoch Gate

Before training and after every complete epoch, evaluate artifacts
`251, 358, 465, 572, 679, 786, 893, 1000`. These are train/train artifacts
disjoint from the joint-update pool. Continue only when the arithmetic mean of
each of direct GPP, 7-day science, and 30-day science strictly decreases from
the preceding panel. Individual artifacts are reported but not thresholded.

A passing epoch writes a non-promotable specialist checkpoint. A failed epoch
writes no checkpoint, stops the stage, and cannot replace the last passing
checkpoint. If epoch 1 fails, the selected state remains the accepted
update-250 warm start. Training and checkpoint selection do not evaluate
validation values or sealed-test data.

## Execution And Recovery

Write an atomic recovery after every ten joint updates and every epoch
boundary. Recovery is valid only for the identical git commit, contract,
schedule, parent assets, warm-start checkpoint, and output directory. On
resume, the joint-record prefix, epoch panels, passing checkpoints, checkpoint
files, and absolute AdamW step must agree before another update is executed.

Every executed group requires finite full-coverage source and combined
gradients. Every AdamW proposal must contain finite specialist and optimizer
trees, make a nonzero parameter change, and pass the post-update one-day hard
gate. All pre-update 1/7/30-day and epoch-panel physical hard counts must be
zero. Any failure stops execution without selecting the failed state.

## Paid Envelope And Next Gate

The prepared launcher requests one `gnall` node, eight CPU cores, one V100,
32 GiB host memory, and 10 hours. At additive published rates, its maximum
charge is CPU `8 * 10 * 0.07 = CNY 5.60` plus GPU
`1 * 10 * 2.20 = CNY 22.00`, or `CNY 27.60` total. The launcher reuses the
accepted structured-GPP joint JAX compilation cache. Submission still requires
explicit approval of the exact commit and command.

If at least one epoch passes, the last passing checkpoint advances to the
already frozen non-test 96-window 7/30-day screen. If no epoch passes, stop and
review the objective or optimizer. Neither outcome authorizes 365-day rollout,
Gate D2, sealed test, or product acceptance.
