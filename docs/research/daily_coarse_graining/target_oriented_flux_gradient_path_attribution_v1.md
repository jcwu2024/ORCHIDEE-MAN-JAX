# Target-Oriented Flux Gradient-Path Attribution v1

Date: 2026-08-26

Status: frozen before zero-training execution

Machine contract:
`manifests/coarse_graining/target_oriented_flux_gradient_path_attribution_v1.json`

Contract SHA256:
`d683b87ae2658c4aeb5658cf42ab0b55f6e9cef94cd18af5419a376e3a821900`

## Question

Why do two maintenance-respiration windows with ordinary losses produce
full-gradient norms of `22489.30` at 7 days and `53.87` at 30 days, while
windows with closely matched maintenance losses produce norms of only `5.57`
and `2.10`?

The gate separates three mechanisms without changing the model:

1. a specific GPP, maintenance, or growth-respiration objective is dominant;
2. a primary/state output path or shared Attention family has an extreme local
   Jacobian;
3. repeated free state transitions amplify an otherwise ordinary daily
   Jacobian across the 7/30-day rollout.

## Frozen Windows

Use the exact update-1000 Attention checkpoint, accepted prepared train/train
manifest, and accepted residual-tail report. The four frozen windows are:

| Role | Horizon | Update | Landpoint | Row | Maintenance score | Full gradient norm |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| Extreme | 7 | 465 | `081.0-107.0` | 90 | 0.276469 | 22489.30 |
| Matched control | 7 | 1000 | `305.0-103.0` | 337 | 0.358222 | 5.57 |
| Extreme | 30 | 465 | `299.0-095.0` | 207 | 0.276488 | 53.87 |
| Matched control | 30 | 572 | `167.0-099.0` | 95 | 0.273844 | 2.10 |

The 7-day control also approximately matches the aggregate score. The 30-day
control tightly matches the maintenance score. They are controls for the
suspected mechanism, not candidate models and not promotion baselines.

## Two State Paths

For every window and each of GPP, maintenance respiration, and growth
respiration, compute the same normalized MSE-plus-squared-bias field score in
two ways:

- **free rollout**: the accepted daily operator feeds each predicted next
  state into the following day;
- **Teacher-state daily evaluation**: the same daily operator, parameters,
  forcing, retained exact tail, and target are used, but each day begins from
  that day's prepared Teacher state.

The second path is diagnostic teacher forcing. It is not training and is not
allowed in final inference. A large free-to-Teacher-state gradient ratio
isolates cross-day recurrence because the one-day operator and physical output
definition are unchanged.

There are exactly `4 x 3 x 2 = 24` full gradients. Each must cover all
`83,755,173` parameters and be finite and nonzero. The free field gradients
must reconstruct the original aggregate gradient with weight `0.125` per
learned field; retained exact heterotrophic respiration has the fourth
`0.125` weight and zero learned gradient.

## Parameter Attribution

Every gradient is partitioned twice:

- five complete coarse families: base/source encoder, shared science
  backbone, primary path, state path, and auxiliary path;
- the 16 exact `target_oriented` parameter keys plus the base/source encoder.

Primary and state final-head norms are additionally reported by learned field.
The audit reports exact field-field and free-versus-Teacher-state Gram
geometry. It does not infer a mechanism from parameter count alone.

## Bounded Interpretation

The preregistered diagnostic thresholds are:

- extreme-to-control gradient ratio at least `10`;
- free-to-Teacher-state ratio at least `10` for recurrence dominance;
- free-to-Teacher-state ratio at most `2` for a local-path classification;
- dominant-field cosine to the reconstructed aggregate at least `0.9`;
- one parameter family is called concentrated only above `80%` of squared
  norm.

These thresholds select the next diagnostic or design discussion; they cannot
promote a model or authorize training. A local primary/state-path result must
pause for architecture discussion before any network edit. A recurrence
result opens recurrence/state-Jacobian conditioning design. A shared-backbone
result opens a zero-training bounded family-conditioning comparison. Mixed
evidence retains the architecture and requires another focused diagnosis.

## Execution Boundary

The executable is
`scripts/hpc/audit_target_oriented_flux_gradient_path.py`; the free-node
launcher is
`scripts/hpc/run_target_oriented_flux_gradient_path_attribution.sh`.

No optimizer, parameter step, checkpoint, recovery, Teacher generation,
historical v5 read, validation-value read, sealed-test read, architecture
edit, paid job, 365-day rollout, Gate D2, or product gate is authorized.
