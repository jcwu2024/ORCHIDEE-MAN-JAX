# Current Project Status

Last updated: 2026-07-22.

This page is the current status authority. Dated files under
`docs/source_audits/` and `docs/research/` are evidence snapshots and may
describe an earlier gate without being rewritten.

## Stable Teacher

The `main` branch is the stable, user-facing PFT14 Teacher implementation. It
contains the full half-hour ORCHIDEE-MAN path, daily STOMATE processing,
restart handoff, annual modelout, compiled complete-day blocks, and the
production CLI.

Current evidence:

- cold-start, restart, 365-day, and cross-landpoint executable-reuse gates
  pass;
- seven previously unused landpoints completed 1961-2010 validation;
- their largest 2010 AGB/BGB/GPP/NPP relative error is `8.5e-5`;
- the full 669-landpoint acceptance run remains a release gate.

The project must not claim equivalence for other PFTs or unsupported
ORCHIDEE configurations.

## Daily Coarse-Graining Research

The `research/daily-coarse-graining` branch contains the complete Teacher plus
research-only capture, dataset, and neural-surrogate code. The production
Teacher defaults are unchanged because all capture hooks default to off.

Completed infrastructure:

- fixed daily-boundary and parameter-conditioning contracts;
- pure-JAX gradient and checkpoint plumbing;
- compiled Teacher training capture with reduced host transfers;
- restartable, atomic landpoint-year NPZ shard generation with provenance,
  frozen spatial/temporal splits, resume, and aggregate validation;
- Linux CPU and V100 compatibility tests.

Current scientific status:

- the neural path is technically connected but is not a validated daily
  surrogate;
- no user-facing `daily-surrogate` run mode exists;
- no free-running 7-, 30-, 365-day, or 50-year neural rollout has passed;
- generated labels remain `provisional_teacher` until the 669-point Teacher
  acceptance gate is complete.

## CPU and GPU Decision

One-landpoint V100 Teacher capture reached about `0.263 s/day` hot for a
28-day block, with about `922.6 s` cold compilation. This does not establish
that GPU is faster than CPU because the retained CPU measurement was not a
matching same-process hot benchmark. It does establish that one landpoint
does not provide enough parallelism to justify GPU Teacher generation.

Current policy:

- use persistent CPU workers for Teacher shard generation unless a measured
  batched-landpoint GPU implementation changes the result;
- use GPU for neural-network training;
- keep inference backend-neutral and benchmark CPU latency versus batched GPU
  throughput after the network architecture is frozen;
- do not describe the research branch as a delivered GPU model.

## Next Bounded Milestone

Before large paid generation or training:

1. run a same-process CPU cold/hot Teacher capture benchmark on explicitly
   allocated Slurm compute resources; shared test nodes are for correctness
   smoke tests and must not be used for accepted performance numbers;
2. freeze a bounded pilot plan with explicit spatial and temporal holdouts;
3. estimate worker memory, hot landpoint-year time, storage, and worst-case
   cluster cost;
4. generate only the approved pilot shards;
5. train and evaluate one-step accuracy and free rollout before scaling.

Do not generate all 669 x 50 landpoint-years merely to discover whether the
surrogate architecture can learn the daily transition.
