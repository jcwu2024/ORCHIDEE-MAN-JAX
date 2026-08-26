# Target-Oriented Flux Residual-Tail And Sampling-Unit Audit v1

Date: 2026-08-26

Status: frozen before zero-training implementation and execution

Machine contract:
`manifests/coarse_graining/target_oriented_flux_residual_tail_audit_v1.json`

Contract SHA256:
`bafcb2daa0bcfc0e7108022b44d8af4741d87a1c5e73e371104e5985819bb763`

## Question

Why does the raw component-resolved flux objective produce a 7-day gradient
norm range from `0.515` to `1713.736`, and why does its valid common
first-order direction fail at every tested finite step?

The audit distinguishes four mechanisms:

1. a few high-loss windows dominate the squared objective;
2. an entire runtime-year/landpoint group represents a difficult regime;
3. gradients are amplified by the model Jacobian even when losses are not
   extreme;
4. window gradients conflict diffusely, so tail reweighting alone cannot
   repair the objective.

This is a zero-training diagnosis. It does not choose a new loss during
execution and cannot promote a model.

## Frozen Data Boundary

Use the exact update-1000 Attention checkpoint and the same train/train
prepared updates used by the completed geometry and virtual-step audits:

```text
251, 358, 465, 572, 679, 786, 893, 1000
```

This yields 64 independent 7-day windows and 32 independent 30-day windows.
Every result retains its update, runtime year, landpoint, and start row. No
Teacher data are regenerated, no historical v5 or validation values are
loaded, and sealed test remains unread.

## Window Measurements

For every window, run one exact free daily rollout and report the frozen-scale
score

```text
0.5 * normalized MSE + 0.5 * normalized signed bias squared
```

for GPP, maintenance respiration, and growth respiration. Heterotrophic
respiration remains an exact retained-path guard. Autotrophic respiration,
NPP, NEP, and NEE are report-only derived keys.

Also compute one standalone component-resolved window-flux gradient over all
`83,755,173` parameters. This is deliberately different from merely reading
loss values: a large gradient with an ordinary loss indicates Jacobian
amplification rather than squared-error tail domination.

Within each update and horizon, form an exact float64 Gram matrix over the
eight or four standalone window gradients. Report gradient norms, pairwise
cosines, the equal-window direction, and the directions implied by every
frozen bounded-influence candidate. No full gradient stack is kept on the GPU
after its group Gram is complete.

## Concentration Measures

For nonnegative loss scores and gradient norms, report:

- top-1, top-5, and top-10 contribution shares;
- effective sample size `(sum x)^2 / sum(x^2)` and its fraction of the
  available windows;
- group contribution shares;
- within-group fraction above the global median;
- sensitivity proxy `gradient norm / sqrt(window score)`;
- loss rank and gradient-norm rank of every window.

The preregistered localized-tail thresholds are top-1 share at least `0.2`,
top-5 share at least `0.5`, or effective-sample-size fraction at most `0.4`.
A broad group requires at least `0.25` of total loss, no single internal
window above `0.3` of that group, and at least `0.75` of its windows above the
global median.

## Candidate Influence Functions

The audit compares equal-window averaging with two smooth bounded-influence
families. For window score `s` and scale `tau`:

```text
soft-L1: rho = 2*tau*(sqrt(1 + s/tau) - 1)
          derivative weight = 1/sqrt(1 + s/tau)

log1p:   rho = tau*log(1 + s/tau)
          derivative weight = 1/(1 + s/tau)
```

For each horizon, `tau` is the positive train/train median window score times
`1`, `4`, or `16`. All derivative weights are normalized to unit sum before
Gram evaluation. These six candidates are fixed before execution; the audit
cannot add another threshold after reading results.

These are diagnostic influence directions, not retained parameter updates.
The later robust-loss contract may select at most one family and scale if the
evidence supports it.

## Decision Rule

- Localized loss and gradient tails admit one bounded window-influence
  candidate to a later zero-update geometry gate.
- A broad difficult group admits an explicit group-stratified sampling
  candidate instead of calling the group an outlier.
- A gradient tail without a loss tail closes robust-loss-only repair and
  requires Jacobian-conditioning diagnosis.
- Diffuse losses with conflicting window gradients close tail reweighting as
  a sufficient mechanism and require objective decomposition.

Engineering acceptance requires all 96 window scores and full-parameter
gradients to be finite, complete group Grams, and zero physical hard failures.
No optimizer, parameter step, checkpoint, recovery, paid job, 365-day
rollout, Gate D2, or product gate is authorized.
