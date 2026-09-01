# Coupled-GPP Bounded Training Contract

Date: 2026-09-01

Status: frozen before runner qualification and paid resource review

The accepted half-rate proposal proves one finite fresh-AdamW update at
`2e-6`. It does not by itself authorize an open-ended fit. This contract
freezes the first retained stage without reading validation targets or
changing the coupled GPP/carbon-state Attention architecture.

## Data Separation

The existing 1,000 prepared train/train artifacts are reused unchanged. The
eight objective-calibration artifacts are excluded. A fixed train-only monitor
uses artifacts `251, 358, 465, 572, 679, 786, 893, 1000`; these artifacts are
excluded from optimizer updates and are never used to change coefficients or
learning rate.

The remaining 984 eligible artifacts are permuted once with NumPy `PCG64`
seed `20260901`. The first 64 entries are grouped without replacement into
eight ordered macro batches of eight artifacts. The materialized schedule
SHA256 is `71928c65...5f160e`. Selection uses artifact identity only and was
frozen before any monitor value was evaluated.

Each retained optimizer update therefore covers:

- 1,664 direct-GPP Teacher days;
- 256 one-day reachable-state transitions;
- 64 seven-day free-rollout windows;
- 32 thirty-day free-rollout windows.

No v6 Teacher row is regenerated, no v5 row is read, and no fresh validation
or sealed-test artifact is opened.

## Objective And Optimizer

The four fixed coefficients remain direct GPP `1.0`, reachable state
`0.6524275555272667`, seven-day science `0.4072260558556481`, and thirty-day
science `0.28249951035415655`. Gradients and losses are arithmetic means over
the eight artifacts. They are not recalibrated during this stage.

Both `target_oriented.structured_gpp` and
`target_oriented.coupled_gpp_carbon_feedback` start with one fresh AdamW state.
The learning rate is constant `2e-6`; beta values, epsilon, weight decay, and
the accepted p90 clip threshold `12.808193184117403` remain unchanged.

## Finite-Step Gate

Every proposal is evaluated on its complete eight-artifact macro batch before
it is retained. All gradients, parameters, and optimizer values must be
finite; both trainable families must change; and all one-, seven-, and
thirty-day physical hard counts must remain zero. The weighted objective must
strictly decrease, and no individual source may increase by more than `1e-6`
relative. A failed proposal is discarded and stops the stage without
advancing the retained optimizer.

The `1e-6` allowance covers numerical realization only. It is not a science
regression budget. Checkpoint selection uses the stricter monitor rule below.

## Milestones And Ownership

The initial monitor is evaluated before training. The same disjoint monitor is
evaluated after updates 4 and 8. Relative to the preceding monitor, all four
source losses must be non-increasing, the weighted objective must strictly
decrease, and every physical hard count must remain zero. Only a passing
milestone writes a non-promotable stage checkpoint. The selected checkpoint
is the last consecutive passing milestone.

Recovery is written atomically after every accepted update and contains only
the two trainable parameter families, their fresh AdamW state, accepted update
records, and train-only monitor records. The 93,504,164 fixed parameters are
reconstructed from the accepted parent and must remain the same JAX array
objects. The source structured-GPP checkpoint must remain hash-identical.

The stage has at most eight optimizer updates and is not promotion evidence.
Even a passing checkpoint only opens a separately frozen fresh 96-window
7/30-day screen. The 365-day rollout, long held points, Gate D2, sealed test,
and product acceptance remain closed.

## Execution Boundary

The machine contract is
`manifests/coarse_graining/target_oriented_coupled_gpp_bounded_training_v1.json`
at SHA256 `56652448...81f1e`. The schedule and stop-rule owner is
`daily_operator/target_oriented_coupled_gpp_bounded_training.py`; atomic state
ownership is in `target_oriented_coupled_gpp_checkpoint.py`; and the executable
runner is `scripts/hpc/run_target_oriented_coupled_gpp_bounded_training.py`.

Before paid execution, `--admission-only` must run on free `gln01`. It executes
the same initial monitor and first full macro proposal, round-trips a temporary
recovery, then discards the proposal and deletes the temporary recovery. It
cannot write a retained training checkpoint.
