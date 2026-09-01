# Coupled-GPP Optimizer Proposal Preflight V1

Date: 2026-09-01

Status: frozen before one discarded train-only optimizer proposal

## Purpose

The accepted objective calibration fixed four loss coefficients but exposed
artifact-level gradient conflict. This preflight tests whether one actual
finite optimizer step preserves the intended macro objective geometry before
any training or checkpoint is authorized.

## Effective Batch

Prepared train-only artifacts `352-357`, `359`, and `360` form one macro
effective batch. Each source gradient is the arithmetic mean of its eight
artifact gradients. Each before/after source loss is likewise the arithmetic
mean of the eight artifact losses. No artifact is selected or excluded after
seeing its gradient geometry.

The effective batch contains 1,664 direct-GPP Teacher days, 256 one-day state
transitions, 64 seven-day windows, and 32 thirty-day windows. It uses no fresh
validation values and no sealed-test data.

## Objective And Optimizer

The frozen coefficients are direct GPP `1.0`, reachable state
`0.6524275555272667`, 7-day science `0.4072260558556481`, and 30-day science
`0.28249951035415655`. The global gradient clip is
`12.808193184117403`.

The proposal uses one fresh AdamW state over both trainable families. The
structured-GPP checkpoint has optimizer history, but the new coupled adapter
does not; splicing inherited moments into only one family would impose an
uncontrolled family-dependent preconditioner. The learning rate is the
already accepted broad structured-GPP rate `4e-6`; no learning-rate search is
performed.

## Acceptance

The macro gradients, combined gradient, parameter proposal, and optimizer
state must be finite with exact parameter coverage. Every trainable leaf and
both trainable families must change. Every source gradient must have strictly
positive cosine with the realized negative parameter step. Direct GPP,
reachable state, 7-day science, 30-day science, and their weighted total must
all strictly decrease on the macro effective batch.

All pre/post prediction, state, unexpected-mask, discrete-state, budget, and
negative-stock hard counts must be zero. Declared-dynamic `rveget` counts stay
reported but are not hard failures. All 93,504,164 fixed parameters and the
source checkpoint must remain unchanged.

The optimizer and proposed parameters are discarded after evaluation. The run
writes one JSON report and no checkpoint or recovery file. Passing opens only
target-blind training/monitor schedule design and a separately frozen bounded
training contract. Failure stops before training and returns to optimizer
realization or objective-conflict design.

The machine contract is
`manifests/coarse_graining/target_oriented_coupled_gpp_optimizer_preflight_v1.json`,
canonical SHA256
`7932efa132780c1c687434980a05fab17d14b4b73cc81926d9eaab92640aa3cf`.
