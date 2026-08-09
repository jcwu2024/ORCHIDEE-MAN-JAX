# Successor Bounded-Small-Fit Runner Contract V1

Date: 2026-08-09.

Status: frozen for local and free-`gln01` preflight; no paid fit is authorized.

## Purpose

This contract closes the execution details that were intentionally outside the
accepted successor architecture contract. It does not reopen the attention
candidate, graph control, 181-field output schema, fast-owner reconstruction,
or v6 data decisions.

The machine authority is
`manifests/coarse_graining/successor_small_fit_runner_v1.json`, SHA256
`8317fc74b6ba921b0ed60e774743884ddc81b233c75ce1823f3246211d1640de`.
It binds the accepted candidate, fast-owner, v6 release, process statistics,
and added fast-owner statistics by hash.

## Matched Sampling

- Both arms use seed `20260809` and the byte-identical schedule.
- Every update uses exactly one train/train annual shard.
- Annual shards are ordered deterministically with landpoint-balanced rounds;
  identity controls sampling only and is never a model feature.
- The one-day anchor contains 16 distinct rows.
- The rollout batch contains two distinct, real, contiguous 30-day windows.
- The first row of every annual shard is excluded from training. This keeps
  every training scan on the accepted normalized ordinary-day retained-tail
  lifecycle. Cold/restart first days remain separate lifecycle acceptance
  cases rather than being silently normalized inside a scan.
- Test rows are sealed and validation rows cannot tune the loss coefficients.

The 30-day trajectory supplies the frozen 1/7/30-day cumulative GPP, NPP,
maintenance/growth/heterotrophic respiration and terminal biomass/carbon
objective. It does not reconstruct 48 half-hour forcing records or execute a
48-step state scan.

## Retained-Tail Runtime

One canonical retained-tail adapter is built per selected landpoint-year. The
two rollout windows in an update share that annual-shard runtime, while native
five-record forcing, day-start state, day number and routed parameters remain
dynamic inputs.

Before any neural update, the probe replays the Teacher named outputs on the
first day of both sampled windows. It requires exact defined status, undefined
representation and discrete state, existing eager/JIT numerical tolerances on
defined state, and water/carbon residuals within `2e-12`. This gate is intended
to catch a landpoint, year or lifecycle closure mismatch directly.

## Objective And Optimizer

The one-day 48-label process loss retains coefficient one. A single
train/train candidate preflight measures gradient norms for the four
long-horizon components and resolves coefficients inside `[0.001, 10]` for
target ratios `0.10`, `0.15`, `0.10`, and `0.15`. Those same resolved numbers
are used unchanged by the graph control.

The bounded optimizer is AdamW with:

- linear warmup for 50 updates to `3e-4`;
- cosine decay to `3e-5` at update 500;
- beta1 `0.9`, beta2 `0.999`, epsilon `1e-8`;
- weight decay `1e-4`;
- global pre-update gradient clipping at norm `1.0`;
- at most 500 updates per arm and no checkpoint.

## Promotion Gates

The free-`gln01` probe is limited to one update per arm and writes only a JSON
report. It must show finite pre/post loss, finite nonzero gradients, finite
updated parameters, zero rollout hard failures, and one completed optimizer
step for both arms.

Only a passing probe may be used to estimate the paid resource envelope. Any
`gnall` submission still requires a new approval block with resources, wall
time, working directory, command and worst-case cost. The 7/30-day screen,
365-day rollout, long-held points, Gate D2 and product acceptance remain
closed.
