# Attention Mixed-Point Annealed Continuation Protocol

Date: 2026-08-16

Status: **frozen for one checkpoint-producing single-GPU continuation**.

## Decision

Stop post-training gradient projection. The accepted update-1000 Attention
candidate will receive one conventional continuation stage that changes data
diversity and the effective learning-rate horizon, while keeping the network,
Teacher release, process loss, conservative operator, and retained exact tail
unchanged.

This stage is justified by two accepted observations:

- four-point-year mixed sampling improved the Attention primary composite by
  `0.73%` from update 1000 and improved all six 7/30-day cells after 100
  updates;
- the staged-v3 cosine schedule spans 10,000 updates, so at update 1000 the
  learning rate is still about `99.3%` of its peak. The completed fit has not
  experienced a meaningful annealing phase.

The earlier mixed-sampling gate rejected a universal policy because graph
control was `0.25%` better under single sampling. That result remains valid,
but it no longer blocks candidate-specific training. The graph control is not
continued in this stage and no new architecture claim is made.

## Frozen Training

The machine contract is
`manifests/coarse_graining/successor_attention_mixed_annealed_continuation_v1.json`,
SHA256 `00f7e598...337ba`.

- Parent: accepted update-1000 parameters and AdamW state.
- Architecture: `source_routed_process_attention_v2`, 8,722,689 parameters.
- Existing input: the accepted prepared updates `1001-2000`; no Teacher run.
- Coverage: 1,000 unique train/train point-years, used once each.
- Updates: 250; each contains four point-years, eight one-day anchors and one
  30-day rollout window per point-year.
- Effective batch: 32 one-day anchors and four 30-day windows.
- Objective: unchanged staged-v3 one-day process anchor plus 1/7/30-day
  long-horizon objective.
- Optimizer: inherited AdamW moments and global norm clip `1.0`.
- Schedule: cosine decay from the staged-v3 update-1001 learning rate to
  `3e-5` over these 250 updates.
- State: atomically recoverable every 50 updates and retained at completion.

There is no generic synthetic data augmentation. Perturbing state, forcing,
or physical parameters without matching Teacher labels remains invalid.

## Acceptance

The unchanged 48-window non-test panel is evaluated once after training.
Admission requires:

- zero numerical, state, mask, discrete, stock, and budget hard failures;
- at least `0.5%` improvement in the joint NPP/AGB/BGB/total-live composite;
- no aggregate regression in any of those four primary quantities;
- improvement in at least four of six temporal/spatial/joint by 7/30-day
  cells;
- maintenance-respiration aggregate regression at most `20%` and every cell
  at most `30%`.

Gradient clipping is reported, not used alone as a pass/fail gate. The final
state is retained even if the scientific gate fails, but it is promotion
eligible only on a pass. A pass opens a non-test 365-day screen. A failure
ends this training recipe and triggers an explicit user discussion of model
capacity or backbone; it does not reopen parameter projection.

## Boundaries

- No v6 Teacher regeneration and no historical v5 data.
- No validation-weight tuning and no sealed-test access.
- No 48-record forcing reconstruction or 48-step candidate state scan.
- No Attention architecture change.
- Execution may use free `gln01` or one `gnall` GPU. Every paid submission
  still requires explicit approval of its resources, wall time, command,
  working directory, and worst-case charge.
