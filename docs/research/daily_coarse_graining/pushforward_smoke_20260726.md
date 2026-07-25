# On-policy Pushforward Smoke

Date: 2026-07-26

## Purpose

Prove that the accepted one-step checkpoint can:

1. freely advance through a detached model-generated prefix;
2. query the exact JAX Teacher at that terminal state; and
3. update only the final matched neural transition with finite gradients.

This is an implementation gate, not scientific acceptance of the candidate.

## Evidence

- Research commit: `c3e2d79f986ae746b89eb5a9de21377f51fb8590`
- Slurm job: `14380290`
- State: `COMPLETED`, exit `0:0`
- Elapsed: 6:33
- Resources: 4 CPUs and one V100
- Peak RSS: 5,207,864 KiB
- Prefix: 3 days, explicitly stop-gradient
- Batch size: 1
- Final matched updates: 1
- Initialization: accepted v5 one-step checkpoint
- Output:
  `runtime/outputs/smoke/pushforward-prefix3-c3e2d79`

Results:

- loss: `0.08365421669078411`
- gradient norm: `0.13334020711655528`
- Teacher queries: 1
- Teacher query/compile time: `277.5144` seconds
- complete stage time: `372.6727` seconds
- report SHA256:
  `dada09b5c0aba0c4ced320729a45081d7c7f8a61888dc9e8a6a696f9fdb9fcce`
- checkpoint SHA256:
  `ac3dc6384ff9e5505bec33a0d341532bbb634a586d8af56ba3910adadd7ce112`

Estimated actual charge is about CNY 0.27: CNY 0.03 for four allocated
CPUs plus CNY 0.24 for one V100 over 6:33.

## Decision

The implementation gate passes. The next and only training experiment for
this method is one equal-update candidate against the frozen
`canonical_multistep_v1` baseline:

- initialize from the same accepted one-step checkpoint;
- use batch size 4 and 192 gradient updates, matching the baseline;
- use `0:48,1:48,3:48,7:48` detached-prefix stages;
- prefix 0 keeps clean archived Teacher pairs;
- nonzero prefixes query the exact Teacher at model-visited states;
- evaluate the resulting checkpoint on the frozen four seven-day windows;
- require improved train/train global error and named carbon-state errors,
  with no defined-status or discrete regression.

If that gate fails, reject `on_policy_pushforward_v1`. Do not follow it with
an unbounded hyperparameter search or more Teacher data from the same points.
