# Coupled GPP Conditioned Attention Discarded Update V1

Date: 2026-09-06

Status: frozen and locally qualified before any conditioned update-3 gradient,
finite-step, or fixed-monitor value is evaluated

Machine contract:
`manifests/coarse_graining/target_oriented_coupled_gpp_conditioned_attention_discarded_update_v1.json`

Canonical contract SHA256:
`b29e92b5696b0d4f980b33f1b802eb8f014aa61f7d9e49e1ca82134fcb97d27d`

## Question

The Q/K-conditioned Attention candidate removes the diagnosed Day-5 GPP
score jump and passes real-v6 one-day forward, gradient, state, and budget
checks. Its common conditioned endpoint is not an accuracy result. This
preflight asks whether the candidate's actual initialization admits one
bounded train-only parameter displacement that improves direct GPP, reachable
state, and 7/30-day science, and whether that displacement transfers to a
fixed non-test monitor.

This is not persistent training. It creates no optimizer, applies no logical
update, and discards every parameter proposal.

## Exact Starting State

The runner loads the identity-exact recovery from the artifact-aware
persistent run. The recovery contains only accepted logical update 2; its
rejected update-3 proposal is absent. The required parent identities are:

- report SHA256 `ded2f4a59fbcd8fafd84d7a64e597017911f1222f5f748170fc82dec97d7aa3a`;
- recovery SHA256 `da019556821239cbf50bd67372002f4217bb10f70eafef55375c9c0c31de857c`;
- completed logical update `2` and selected scale history `(0.5,)`.

The recovered parent trainable tree is then converted to the conditioned
candidate by attaching 16 learnable per-block/per-head temperatures initialized
at `sqrt(48)`. Earlier bootstrap parameters are not used as the candidate
start. The resulting candidate has `109,464,375` total parameters and
`15,960,211` first-stage trainable parameters.

The process head still newly controls only `daily.fast_gpp`. It remains bound
through the canonical daily operator, conservative updater, and retained exact
tail. Inference reconstructs no half-hour forcing and performs no 48-step
state scan.

## Train-Only Evidence

All data come from the accepted Teacher-v6 `train/train` release. No Teacher
data are regenerated and no historical v5 row is read.

Before any gradient is measured, the runner reports the initial 7/30-day
science panel on scheduled group 3. Direction construction then uses five
frozen groups: scheduled group 3 plus guard groups 15-18. They contain 40
distinct point-year artifacts and 208 Teacher days per direct-GPP artifact.

The common direction has 43 independent constraints:

- 40 individually unit-normalized direct-GPP artifact gradients;
- scheduled-group reachable one-day state loss;
- scheduled-group 7-day science loss;
- scheduled-group 30-day science loss.

The minimum-norm simplex direction must have cosine at least `1e-4` with every
constraint. AdamW moments and coordinate-wise optimizer transforms are not
used. The 16 temperature coordinates participate in the same gradient tree as
the inherited Q/K/V/output parameters.

## Architecture-Aware Step Calibration

The old parent displacement `0.007758973...` is not inherited. For each
constraint `i`, define

```text
relative_slope_i = gradient_norm_i * direction_cosine_i / max(loss_i, 1e-12)
```

The full candidate displacement is

```text
min(
    0.02 / max(relative_slope_i),
    0.01,
    1e-5 * trainable_parameter_l2,
)
```

It is usable only at or above `1e-8`. The runner evaluates descending
multipliers `1, 1/2, 1/4, 1/8, 1/16, 1/32, 1/64` and stops at the first
train-only passing scale.

A train-only scale passes only when:

- all 40 direct-GPP artifacts stay within `1e-6` relative regression;
- their combined direct-GPP mean strictly improves;
- scheduled state, 7-day, 30-day, weighted, and physical gates pass;
- the nine-key 7/30-day science macro strictly improves;
- GPP, NPP, AGB, BGB, and total-live loss regress by at most `0.1%` per horizon;
- respiration and ecosystem-carbon loss regress by at most `1%` per horizon.

The absolute promotion thresholds are also reported: normalized RMSE/bias at
most `0.35/0.10` for 7 days and `0.25/0.07` for 30 days on all nine science
keys. A single discarded step is not required to reach those final thresholds.

## Fixed Monitor

Only after a train-only scale is selected may the runner read monitor updates
`251, 358, 465, 572, 679, 786, 893, 1000`. These artifacts are disjoint from
the 40 direction/selection artifacts and never choose a direction or scale.

Each monitor artifact is loaded exactly once. Its initial and candidate states
are evaluated consecutively as a pair before the artifact is released, keeping
the host-memory peak bounded. Transfer requires strict direct-GPP mean and
nine-key science-macro improvement, at least `6/8` nonregressing direct-GPP
artifacts, the same per-key regression caps, and zero physical hard failures.

## Retention And Decision Boundary

The run writes only one JSON report. It does not write parameters, checkpoint,
optimizer state, or recovery, and it verifies that the parent recovery remains
unchanged.

- A passing monitor with or without immediate absolute-science acceptance may
  open a separately frozen bounded conditioned-Attention training smoke.
- A monitor failure requires transfer attribution before training.
- No 43-constraint common direction requires objective-geometry review.
- No train-only finite step requires step-conditioning review.

Paid training, fresh validation, 365-day rollout, Gate D2, sealed-test access,
and product acceptance remain closed regardless of this preflight's outcome.
One free `gln01` run with a 180-minute timeout is authorized.

## Local Qualification

The frozen loader, architecture-aware trainable replacement, step calibration,
science and monitor gates, one-load paired monitor, CLI argument wiring, and
no-retention boundary pass 67 focused and helper-regression tests. Ruff,
Python compilation, Bash syntax, contract loading, and whitespace checks pass.
The free real-v6 execution is the active gate.
