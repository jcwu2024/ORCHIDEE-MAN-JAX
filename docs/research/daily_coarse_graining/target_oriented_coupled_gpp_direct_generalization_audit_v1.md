# Direct-GPP Generalization Attribution Audit

Date: 2026-09-03

Status: frozen and locally qualified before any attribution values are evaluated

Contract SHA256:
`017b8a1de69260970cb78f500afc8984cb86afb2a57cfb53bdb111e1fd55ff5f`

## Question

Persistent job `14911228` improved the fixed-monitor reachable state,
7/30-day science, and cumulative GPP by approximately `15-34%`, but direct
GPP on Teacher states worsened `24.1562%`. Every accepted update nevertheless
improved direct GPP on its own eight-artifact training group.

The next decision cannot be made from the two endpoint monitors. This audit
locates the first adjacent direct-GPP monitor regression and measures the
training-versus-monitor direction geometry at that state. It does not train a
new model, tune a coefficient, change the Attention architecture, or select a
checkpoint.

## Frozen Trajectory Replay

The audit reconstructs the accepted state after update 7 in one pass from the
bounded-training update-1 recovery. Update 2 uses the accepted fresh-AdamW
mechanism once; updates 3 through 7 recompute the common direction once and
apply the already accepted scales `1, 1, 1/4, 1, 1/2`. It does not recursively
rerun the historical update-5 scale audit or scale-selecting smoke. It then
deterministically recomputes the four-source minimum-norm direction for
persistent updates 1 through 10 on the original frozen schedule. The selected
scales are read from the accepted result and fixed to:

```text
1, 1/2, 1/2, 1, 1/2, 1/2, 1, 1, 1, 1
```

Scale search and its repeated finite-step rollouts are not rerun. After each
in-memory update, the audit evaluates only direct GPP on the fixed eight-
artifact train/train monitor. The first strict adjacent increase in mean loss
is the attribution state. The accepted persistent result remains the identity
and direction reference, but independent GPU replay losses are report-only.
The bootstrap monitor must remain within `1%` of that result, while exact
`rtol=1e-6, atol=1e-9` comparisons are reserved for values recomputed inside
the same replay.

The persistent `recovery.pkl` is not read. The replay writes no parameters,
optimizer state, recovery, or checkpoint.

## First-Regression Attribution

At the state immediately before the first adjacent regression, the audit
computes direct-GPP gradients separately for:

- the eight artifacts used by that persistent update;
- the eight fixed monitor artifacts;
- the arithmetic mean of each group; and
- the parameter-descent direction actually applied by the four-source update.

It records the complete cosine matrix, within-group and cross-group cosine
summaries, and the direct-GPP finite loss change for every artifact. Signs have
a direct interpretation: a positive cosine with the applied descent direction
predicts lower direct-GPP loss to first order; a negative cosine predicts
higher loss.

The report uses only sign-based attribution:

1. A nonpositive training-mean versus monitor-mean cosine identifies a
   scheduled-batch/monitor direction mismatch.
2. A positive training-monitor cosine but nonpositive monitor cosine with the
   applied four-source step identifies rotation caused by the multi-source
   direction.
3. A positive monitor cosine with the applied direction followed by a finite
   regression identifies finite-step curvature or nonlinearity.
4. Negative per-artifact pairs quantify point-year direction conflict.

One 8-versus-8 panel cannot statistically separate finite-sample variance
from population heterogeneity. The audit therefore reports that limitation
explicitly instead of assigning an unsupported certainty. Its evidence is
enough to choose the next bounded repair: larger/multiple training groups,
robust group constraints, or a revised loss/acceptance hierarchy.

## Acceptance And Execution Boundary

The audit succeeds only if all ten updates replay, the first adjacent
regression is found, all 16 artifact gradients are finite and nonzero, the
per-artifact means agree with the same-run trajectory, and the replay agrees
with job `14911228` that direct GPP worsens over the ten updates. This is an
engineering/evidence-completeness gate, not a model-promotion gate.

The first free execution at commit `06c765a` completed update-7
reconstruction and its initial eight-artifact monitor, then stopped before
persistent update 1 because the original cross-run `rtol=1e-6` gate was too
strict. Two already accepted independent initial monitors differ by
`0.125583%`, proving that bit-level cross-run equality is not a valid identity
condition. The amendment above removes only that false precision; schedule,
scale, same-run, gradient, and scientific-direction gates remain binding. No
report or parameter state was written by the stopped run.

That execution also exposed an independent performance defect: the old runner
recursively repeated the historical scale searches while constructing the
update-7 state, taking almost three hours before the actual attribution began.
The amended runner preserves the accepted update mechanisms and fixed scales
but constructs that state in a single pass. No timeout increase substitutes
for this correction.

Only the accepted v6 train/train prepared artifacts and the existing non-test
monitor are read. Teacher generation, historical v5 data, fresh validation,
sealed test, 365-day rollout, Gate D2, 48-record forcing reconstruction, and a
48-step state scan remain forbidden.

One free `gln01` execution is allowed with a 180-minute timeout. Paid
partitions are not authorized by this contract. Local qualification consists
of contract/hash validation, pure trajectory/cosine/classification tests,
runner CLI import, Ruff/format checks, and Bash syntax validation.

## Implementation

- Contract:
  `manifests/coarse_graining/target_oriented_coupled_gpp_direct_generalization_audit_v1.json`
- Contract logic:
  `research/daily_coarse_graining/daily_operator/target_oriented_coupled_gpp_direct_generalization.py`
- Runner:
  `scripts/hpc/audit_target_oriented_coupled_gpp_direct_generalization.py`
- Free-node launcher:
  `scripts/hpc/run_target_oriented_coupled_gpp_direct_generalization.sh`

No training continuation or architecture decision is opened until the audit
result is interpreted.
