# Coupled-GPP Bounded Training Result

Date: 2026-09-02

Status: completed scientific stop after one retained update; no checkpoint

Explore1000 job `14903328` ran the frozen bounded-training contract at commit
`6d1864b2d7c2c95b176c1995d8f59ebd03d92d8a`. The runner completed normally,
wrote its final report and report digest, and stopped through the declared
finite-step science gate rather than an engineering exception. Total runner
time was `1,717.025` seconds. Report SHA256 is
`bf83501782f0e6815a5bd408b203f15c8bcceefc3effb9cdda713aefd58be39c`.

The first scheduled macro update was retained. It improved direct GPP,
reachable one-day state, 7-day science, 30-day science, and the weighted
objective by `1.60249%`, `10.41690%`, `1.02810%`, `1.32569%`, and `2.52874%`.
Its combined gradient norm was `3.48236`, below the frozen `12.80819` clip
threshold, so the clip scale was `1.0`. All 15,960,195 trainable gradient
values were finite and all 190 leaves were nonzero.

The second scheduled proposal was evaluated and rejected. Its weighted
objective still improved `0.686293%`; reachable state, 7-day science, and
30-day science improved `6.17157%`, `1.00842%`, and `0.550934%`. Direct GPP,
however, regressed `2.63633%`, violating the frozen rule that every one of the
four macro source losses must not increase. The proposal was therefore
discarded without advancing the retained optimizer state.

This is not a clipping or numerical failure. The second raw combined gradient
norm was `2.40601` and its clip scale was `1.0`; all gradients, optimizer
values, proposals, next-day states, and physical checks were finite. The raw
direct-GPP gradient had positive cosine `0.245048` with the combined gradient,
while the reachable-state cosine was `-0.0707631`. Because update 2 used the
inherited AdamW moments, the evidence does not yet distinguish optimizer-step
rotation from finite-step curvature. That distinction must be made before
changing the objective or architecture.

All water/carbon and other true physical hard-failure counts remain zero. The
93,504,164 fixed parameters retain their original arrays and values, and the
source structured-GPP checkpoint is unchanged. No v5 data, regenerated
Teacher data, fresh validation target, sealed test, reconstructed 48-record
forcing, or subdaily scan was used.

The retained update-1 state is stored only as atomic recovery on the server.
The stage did not reach update 4, so no milestone monitor ran and no checkpoint
was written or selected. The recovery is diagnostic continuation state, not a
science checkpoint and not promotion eligible.

The active gate is a zero-training update-2 optimizer-geometry audit loaded
from the exact recovery. It must recompute the same update-2 macro gradient,
compare the raw combined direction with the inherited AdamW displacement, and
evaluate a frozen bounded scale grid without retaining any proposal. It may
include a preregistered fresh-moment comparator solely to distinguish moment
rotation from curvature. Fresh 96-window validation, continued training,
365-day rollout, Gate D2, sealed test, and product acceptance remain closed.
