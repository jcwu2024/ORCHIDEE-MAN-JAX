# Coupled GPP Multi-Group Direction Result (2026-09-04)

## Outcome

The bounded free-`gln01` audit completed at commit `c5c0a56` in
`4,874.724` seconds. Its classification is
`multi_group_direction_fails_monitor`: an eight-constraint common first-order
direction exists, and all six frozen train-only scales pass, but direct-GPP
mean loss on the fixed monitor increases by `0.4731%` at the selected scale
`1/2`.

The server report is
`runtime/outputs/diagnostics/target-oriented-coupled-gpp-multi-group-direction-c5c0a56.json`,
SHA256
`cf10f5e27da8f0c50f582b14e7513b6ec87a913b2121dd52584a872672663577`.
The hash identifies the immutable report; it is not a cross-GPU numerical
equality gate.

## Common-Direction Geometry

The minimum-norm simplex solver finds a direction with normalized directional
derivative `0.1504357081` for every one of the eight constraints, well above
the frozen `1e-4` minimum. All eight constraints have positive support:

| Constraint | Normalized simplex weight |
| --- | ---: |
| Scheduled direct GPP | `20.8454%` |
| Guard 1 direct GPP | `1.0279%` |
| Guard 2 direct GPP | `8.1395%` |
| Guard 3 direct GPP | `3.9541%` |
| Guard 4 direct GPP | `10.2791%` |
| Reachable one-day state | `45.6699%` |
| 7-day science | `2.5154%` |
| 30-day science | `7.5688%` |

The five train-only direct-GPP group gradients are mutually aligned, with
pairwise cosines from `0.724643` to `0.967555`. Their conflict with reachable
state remains strong: the corresponding cosines range from `-0.694163` to
`-0.933249`. The solver can balance those group means, so group-mean
first-order incompatibility is not the blocker.

## Train-Only Scale Selection

The common direction was evaluated at all six frozen scales. Negative
percentages are improvements. The scheduled columns are direct GPP,
reachable one-day state, 7-day science, and 30-day science. Every row passes
the scheduled physical hard gates.

| Scale | Scheduled GPP | State | 7 day | 30 day | Guard 1 | Guard 2 | Guard 3 | Guard 4 | Combined guard | Pass |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `1/2` | `-5.5415%` | `-4.0880%` | `-0.6678%` | `-2.1346%` | `-0.6024%` | `-3.1333%` | `-1.8701%` | `-2.4597%` | `-1.9772%` | yes |
| `1/4` | `-3.8872%` | `-2.3150%` | `-0.2881%` | `-0.9223%` | `-0.7400%` | `-2.2518%` | `-1.8236%` | `-2.0788%` | `-1.6787%` | yes |
| `1/8` | `-2.2000%` | `-1.2003%` | `-0.3242%` | `-0.4159%` | `-0.4545%` | `-1.2704%` | `-1.1185%` | `-1.2072%` | `-0.9841%` | yes |
| `1/16` | `-1.1613%` | `-0.6070%` | `-0.2087%` | `-0.2499%` | `-0.2454%` | `-0.6671%` | `-0.6086%` | `-0.6395%` | `-0.5244%` | yes |
| `1/32` | `-0.5957%` | `-0.3052%` | `-0.0573%` | `-0.1297%` | `-0.1269%` | `-0.3410%` | `-0.3163%` | `-0.3279%` | `-0.2698%` | yes |
| `1/64` | `-0.3015%` | `-0.1530%` | `-0.0407%` | `-0.0678%` | `-0.0645%` | `-0.1723%` | `-0.1611%` | `-0.1659%` | `-0.1367%` | yes |

Because every scale passes, the preregistered selector chooses the largest
scale, `1/2`. The fixed monitor was not read until this train-only selection
was complete.

## Fixed-Monitor Transfer

At scale `1/2`, fixed-monitor direct-GPP mean loss changes from
`0.0628461098` to `0.0631434307`, or `+0.473094%`. Four of eight artifacts
improve and four regress:

| Artifact update | Relative change |
| ---: | ---: |
| `251` | `-5.0275%` |
| `358` | `+20.2120%` |
| `465` | `+10.3262%` |
| `572` | `-23.5330%` |
| `679` | `-2.9033%` |
| `786` | `+27.3672%` |
| `893` | `-11.6256%` |
| `1000` | `+2.5015%` |

The predecessor audit independently reconstructed the same logical
pre-update-2 state and reported a slightly different monitor baseline. The
contracts do not require separate GPU reconstructions to be numerically
identical. Therefore the predecessor's `+0.5107%` and this run's `+0.4731%`
must not be ranked as a paired improvement; each within-run before/after
comparison independently fails its transfer gate.

## Integrity And Runtime

The parent recovery, predecessor result, and all `93,504,164` fixed
parameters remain unchanged. No optimizer update, parameter proposal,
recovery, checkpoint, or trainable parameter is written. The audit uses only
accepted v6 train/train artifacts and does not read fresh validation or sealed
test data, regenerate Teacher data, join v5 data, reconstruct 48 forcing
records, or execute a 48-step state scan.

Reported JAX allocator peak is `4,098,246,656` bytes. The run completed with
all engineering and scheduled physical gates passing.

## Decision

Close group-mean multi-group GPP direction construction as the next training
rule. It solves the five observed train-only group means but does not transfer
to the independent monitor, so a three-update smoke is not opened and the
Attention architecture is not rejected.

The next gate is the frozen artifact-aware zero-training direction audit in
[`target_oriented_coupled_gpp_artifact_aware_direction_audit_v1.md`](target_oriented_coupled_gpp_artifact_aware_direction_audit_v1.md).
It exposes all 40 train-only artifact gradients to direction construction,
retains train-only scale selection, and evaluates the fixed monitor only after
selection. No persistent training or architecture change is justified until
that audit shows independent transfer.
