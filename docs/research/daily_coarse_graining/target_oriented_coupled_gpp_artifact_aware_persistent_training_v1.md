# Coupled GPP Artifact-Aware Persistent Training V1

## Question

The accepted continuity smoke showed that the current Attention candidate can
take three consecutive artifact-aware updates while improving the held-out
train-split monitor. It also showed that reconstructing the same nominal
starting state in a new process changes losses and selected scales enough to
make repeated bootstrap an unsuitable training boundary.

This stage asks whether the accepted mechanism remains useful for six bounded
updates when every continuation starts from exactly serialized parameters.
It does not change the network architecture, regenerate Teacher data, use v5,
or expose fresh validation or sealed-test values.

The frozen machine contract is
`manifests/coarse_graining/target_oriented_coupled_gpp_artifact_aware_persistent_training_v1.json`,
canonical SHA256
`af7b3dbf64b70b50133ac76728413a56389f69afd50223000ac32ee76f01aa02`.

## Exact Starting State

On a fresh run, the runner reconstructs the historical state once and replays
accepted persistent update 1. Before reading the monitor, it writes the
resulting 15,960,195 trainable parameters to immutable `start.pkl`, reloads
them, and requires exact array identity. Fixed parameters and optimizer state
are not serialized.

If `start.pkl` already exists, the runner initializes only the model context
and compiled functions. It loads either `start.pkl` or the latest
identity-matched `recovery.pkl` and performs no historical update replay.
This makes an interrupted execution a continuation of one numerical
trajectory rather than another approximate reconstruction.

## Train-Only Sequence

Logical updates 2 through 7 each use 40 distinct train-only point-year
artifacts. The six panels contain 240 unique artifacts and remain disjoint
from the fixed eight-artifact monitor.

| Logical update | Scheduled group | Four additional groups |
| ---: | ---: | --- |
| 2 | 2 | 11, 12, 13, 14 |
| 3 | 3 | 15, 16, 17, 18 |
| 4 | 4 | 19, 20, 21, 22 |
| 5 | 5 | 23, 24, 25, 26 |
| 6 | 6 | 27, 28, 29, 30 |
| 7 | 7 | 31, 32, 33, 34 |

At every update, the runner recomputes 43 unit gradients: 40 individual
direct-GPP artifacts plus scheduled reachable-state, 7-day, and 30-day
objectives. It solves the minimum-norm simplex again, applies the fixed full
displacement norm `0.007758973351050004`, and searches
`1/2, 1/4, 1/8, 1/16, 1/32, 1/64`. Selection uses train-only values and the
existing physical gates. The monitor never selects a direction or scale.

## Persistence And Monitor Gates

The runner writes atomic `recovery.pkl` after the initial monitor and after
every accepted update. Recovery stores the trainable parameter tree, exact
identity, retained update records, initial monitor, and completed milestone
monitors. It stores neither fixed parameters nor optimizer state.

The fixed train-split monitor is evaluated after updates 4 and 7. The update-4
gate permits at most 10% regression in mean direct GPP and the primary science
macro, at most 25% for any primary key/horizon, and requires all physical
checks. The final gate requires strict improvement in mean direct GPP and the
primary science macro, non-regression of 7-day and 30-day GPP, at most 1%
regression in every non-GPP source mean, at most 10% regression in any primary
key/horizon, and all physical checks.

Primary science covers GPP, maintenance respiration, growth respiration,
heterotrophic respiration, autotrophic respiration, NPP, aboveground biomass,
belowground biomass, and total live biomass at both 7 and 30 days. Only a
passing update-7 gate can create immutable `checkpoint-update-0007.pkl`; that
checkpoint is diagnostic and not yet promotion eligible.

## Execution Boundary

One free `gln01` execution is allowed with a 240-minute timeout. A timeout is
an expected resumable interruption: a later run continues from the latest
atomic recovery. Paid partitions, fresh validation, 365-day rollout, Gate D2,
sealed test, and product acceptance remain closed.

The runner performs no reconstruction of 48 half-hour forcing records and no
48-step state scan. Final inference architecture remains the existing daily
Attention process head bound through the canonical daily-operator boundary.

## Implementation

- Contract logic and gates:
  `research/daily_coarse_graining/daily_operator/target_oriented_coupled_gpp_artifact_aware_persistent_training.py`
- Parameter-only persistence:
  `research/daily_coarse_graining/daily_operator/target_oriented_coupled_gpp_artifact_aware_persistence.py`
- Runner:
  `scripts/hpc/run_target_oriented_coupled_gpp_artifact_aware_persistent_training.py`
- Free-node wrapper:
  `scripts/hpc/run_target_oriented_coupled_gpp_artifact_aware_persistent_training.sh`

All 79 coupled-GPP unit tests pass, including the new contract, gate,
persistence, resume-routing, parser, and launcher coverage. Ruff, Python
compilation, and shell syntax checks also pass. The next gate is one free-
`gln01` execution from a clean synchronized commit.
