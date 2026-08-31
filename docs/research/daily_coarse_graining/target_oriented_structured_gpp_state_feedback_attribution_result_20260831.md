# Structured-GPP State-Feedback Attribution Result

Date: 2026-08-31

Status: accepted zero-training train-only scope result; architecture discussion
is the next gate

Executable commit:
`15171cd433288085afb60e3a90f9d7705b9650c3`

Attribution contract SHA256:
`858b59077d4a9dbdc7ecaeb06345725bd94c2a570f1b78d258d8dee480bc69ed`

Report SHA256:
`1b2d1820a48b17ae0297bacc5e0cdf2b00e3290c3a5b9c135a98417a41c2a19b`

## Execution

The direct free-`gln01` run completed in `2010.86 s` (`33.51 min`) with
status `passed`. It loaded the accepted broad update-1000 structured-GPP
checkpoint and evaluated prepared `train/train` updates `301`, `601`, and
`901`. The evaluated model contains `106,383,527` parameters, but the audit
created no optimizer and changed none of them.

The report is bound to broad-training report SHA256
`5cfab5a693be4c58fd2726420106f2844f93703e5ca8e2910268498154b2bee9`,
selected checkpoint SHA256
`3b173b8980c37b49db3d1547dddad2ab98ec00c733e2f564469490ee382d4ec0`,
and prepared-manifest SHA256
`8205140ddeae34967a287d41d204318659a84a8574de954e9b0be14f8998fc31`.

The first free, full-Teacher, and component executions paid the separate JAX
compilation costs. Once compiled, each update/mode evaluation took about
`11.5-11.7 s`. No paid Slurm allocation was used.

## Integrity And Scope

| Check | Result |
| --- | ---: |
| Acceptance statistics finite and nonempty | passed |
| Physical hard failures | `0` |
| Training updates / parameter steps | `0 / 0` |
| Optimizer or checkpoint created | no / no |
| Teacher data regenerated | no |
| Historical v5 read | no |
| Fresh validation values or sealed test read | no / no |
| Reconstructed half-hour forcing records | `0` |
| Subdaily state-scan steps | `0` |

The six source components cover these real v6 day-start coordinates:

| Component | Source owner | Coordinates |
| --- | --- | ---: |
| canopy | `diffuco_previous_step_state` | `52` |
| energy | `enerbil_previous_step_state` | `141` |
| water | `hydrol_previous_step_state` | `622` |
| thermal | `thermosoil_previous_step_state` | `512` |
| carbon | `slowproc_stomate_previous_step_state` | `2,386` |
| driver | `driver_previous_step_state` | `11` |

## Cumulative-GPP Gap Capture

The frozen capture fraction measures how much of the ordinary-free to
full-Teacher cumulative-GPP score gap is removed by carrying only one Teacher
state component. The materiality threshold is a median capture of at least
`10%`, positive capture in at least two of three groups, at either horizon.

| Component | 7-day median capture | 30-day median capture | Material |
| --- | ---: | ---: | --- |
| carbon | `96.0598%` | `81.9454%` | yes |
| water | `9.1288%` | `5.1239%` | no |
| energy | `0.0492%` | `0.3574%` | no |
| canopy | `-0.1916%` | `-0.0973%` | no |
| thermal | `-0.6293%` | `-1.0273%` | no |
| driver | `-0.0009%` | `-0.0005%` | no |

Carbon capture is positive in all three updates at both horizons:

| Horizon | Update | Free score | Full-Teacher score | Carbon-carry score | Capture |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | `301` | `1.390266` | `0.002848` | `0.048861` | `96.6835%` |
| 7 | `601` | `0.914327` | `0.003332` | `0.039227` | `96.0598%` |
| 7 | `901` | `0.590125` | `0.027605` | `0.211886` | `67.2402%` |
| 30 | `301` | `0.648748` | `0.001161` | `0.118081` | `81.9454%` |
| 30 | `601` | `1.276762` | `0.000345` | `0.004142` | `99.7025%` |
| 30 | `901` | `0.850810` | `0.016530` | `0.762948` | `10.5314%` |

The formal classification is
`material_source_components_identified`. The selected first scope is
`state_carbon`, owned by `slowproc_stomate_previous_step_state`.

## Interpretation

The current GPP failure is not a generic recurrence problem spread evenly
across the state. On this frozen train-only panel, the slow-process carbon
state is the dominant carrier from one day's prediction into later GPP.
Canopy, energy, thermal, and driver carry provide no material repair.

Water is not selected by the frozen threshold, but it is positive in all
three 7-day groups and has a `9.13%` median capture, narrowly below `10%`.
It therefore remains a monitored secondary path rather than being declared
irrelevant. The low `10.53%` 30-day carbon capture for update `901` also means
that carbon alone does not explain every scenario or possible interaction.

This result does not authorize direct prediction or replacement of retained
exact STOMATE state. The architecture must preserve the canonical retained
tail and use the selected source scope to improve the process quantities and
state-conditioned GPP mapping that feed it.

## Decision And Next Gate

The broad update-1000 checkpoint remains rejected for promotion. Unchanged
direct-GPP continuation is closed.

Before implementation or training, freeze one coupled design that:

1. retains the existing day-local coordinate-Attention GPP specialist;
2. opens only source-backed paths associated with
   `slowproc_stomate_previous_step_state` in its first version;
3. keeps water as a reported diagnostic and protects it from regression;
4. supervises direct daily GPP and the next-day carbon state, then applies
   7/30-day GPP, respiration, NPP, and carbon-stock objectives through the
   canonical constrained update and retained exact tail;
5. names a matched mechanism control, bounded admission, and stop rule.

The fresh 96-window panel remains evaluation-only. Paid training, 365-day
rollout, Gate D2, sealed test, and product acceptance remain closed.
