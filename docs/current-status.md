# Current Project Status

Last updated: 2026-07-24.

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

- Daily Fast-Day Teacher Contract v4:
  `S[d] (3,854 continuous values plus 12 exact discrete values) + native
  6-hour forcing[d] + P -> B_fast[d] (2,815 values)`, followed by retained
  source-backed daily season/STOMATE to produce `S[d+1]`;
- only the true cross-day subset of `sechiba_finalize_state`, including
  `leaf_ci`, is carried in `S[d]`; the 93-field finalize packet is no longer a
  duplicated learned output;
- canonical PFT14 state trajectories with PFT1/PFT14 axis compaction,
  exact discrete state, deterministic mirror reconstruction and year-start
  `nroot` handling;
- exact native-forcing window reconstruction of all 48 Teacher input steps;
- named condition slices for 84 parameter values, 735 physical/static
  landpoint values, annual CO2, year, and day index;
- a machine-audited ownership ledger covering every compiled complete-day
  Teacher argument;
- a hash-verifying training reader that enforces frozen splits, excludes
  landpoint identity, computes train-only streaming statistics, excludes
  NaN/inf and ORCHIDEE `+/-1e20` sentinels, and uses bounded prefetch;
- a persistence-centered residual target representation with exact
  restoration of persistent undefined values and a dedicated classifier for
  source-dynamic `diffuco.rveget` defined/undefined transitions;
- pure-JAX gradient and checkpoint plumbing;
- compiled Teacher training capture with reduced host transfers;
- restartable, atomic landpoint-year NPZ shard generation with provenance,
  frozen spatial/temporal splits, resume, and aggregate validation;
- a frozen 669-point production specification and a first five-point,
  1961-2010 architecture-development subset with two train, two validation,
  and one test landpoint;
- Linux CPU and V100 compatibility tests.

Current scientific status:

- the neural path is technically connected but is not a validated daily
  surrogate;
- no user-facing `daily-surrogate` run mode exists;
- no free-running 7-, 30-, 365-day, or 50-year neural rollout has passed;
- generated labels remain `provisional_teacher` until the 669-point Teacher
  acceptance gate is complete.
- the first five-epoch V100 run is rejected as architecture evidence because
  it used contract v3 and statistics v1, which treated `rveget=1e20` as an
  ordinary regression value and duplicated finalize mirrors in `B_fast`;
- the accepted five-point v3 Teacher dataset remains provenance evidence, but
  it and its v1 statistics/checkpoint must not be reused for v4 training.
- old v1/v2 shards are audit evidence only and must not be expanded into the
  current v3 dataset.

## CPU and GPU Decision

One-landpoint V100 Teacher capture reached about `0.263 s/day` hot for a
28-day block, with about `922.6 s` cold compilation. This does not establish
that GPU is faster than CPU because the retained CPU measurement was not a
matching same-process hot benchmark. It does establish that one landpoint
does not provide enough parallelism to justify GPU Teacher generation.

Current policy:

- use persistent CPU workers for Teacher shard generation unless a measured
  batched-landpoint GPU implementation changes the result;
- use the accepted seven-day complete-day block for CPU Teacher generation;
  the 28-day capture result is a historical V100-specific experiment;
- use GPU for neural-network training;
- keep inference backend-neutral and benchmark CPU latency versus batched GPU
  throughput after the network architecture is frozen;
- do not describe the research branch as a delivered GPU model.

The project-owned Explore1000 `orcjax_gpu` environment passed its real-V100
runtime gate on 2026-07-23. JAX 0.4.38 selected backend `gpu` and executed a
finite JIT loss and gradient after explicit PJRT plugin discovery. The
canonical verification and training entry points now share that fail-fast
initialization and reject CPU fallback. This validates the training runtime;
it is not evidence that the neural surrogate itself is scientifically valid.

## Repository and Release Architecture

The intended user-facing release is one codebase with two independent runtime
choices:

- execution backend: `cpu` or `gpu`;
- transition implementation: `teacher_half_hour` or `neural_daily`.

CPU and GPU support must not become permanent model forks. They share the
same state, forcing, parameter, restart, output, and acceptance contracts;
only batching, compilation, and device execution may differ. The stable
`main` branch contains accepted user-facing implementations. The
`research/daily-coarse-graining` branch remains a reproducible development
line for dataset generation, network training, experiments, and promotion
evidence, not a separate product.

A released neural checkpoint cannot be the sole training artifact. Its
provenance must bind the Teacher commit, dataset and split manifests, state
and input schema hashes, train-only normalization statistics, architecture,
optimizer and schedule, random seeds, environment lock, checkpoint hash, and
rollout acceptance report. Reproduction code and compact manifests belong in
Git; large datasets and weights remain external assets with stable identifiers
and SHA256 hashes. Accepted training and inference code is merged into
`main`, while ongoing experiments continue on the research branch.

## Teacher Production Evidence

Explore1000 admission measurements established the current production policy:

- a first eight-day landpoint compiled and captured in 1,406.85 seconds;
- a second landpoint in the same process captured in 3.35 seconds and took
  40.15 seconds including preparation, proving executable reuse across
  equal-shape PFT14 landpoints;
- a cold 1961 full year took 2,395.32 seconds;
- the following in-process 1962 hot year took 102.30 seconds to capture and
  139.20 seconds including compression, hashing, checkpoints, and shared IO;
- peak RSS reached about 29.4 GiB, so the initial total concurrency cap is five
  persistent workers per approximately 187 GiB node;
- persistent compilation cache did not eliminate compilation in a new Slurm
  process, so production relies on same-process reuse rather than cross-job
  cache reuse.

The first bounded production target was the frozen five-point, 1961-2010 v3
dataset: 250 landpoint-year shards and 91,245 daily transitions. Its first
submission was rejected before acceptance because the production planner
incorrectly inserted Gregorian leap days into the paper model's fixed 365-day
noleap calendar. The planner, plan validator, and training calendar features
now enforce noleap semantics. The clean replacement completed and passed its
aggregate gate on 2026-07-23: all five 1961-2010 chains, 250 shards, 91,245
transitions, source/input hashes, checkpoint links, splits, and the single
Markov contract are valid. Its operational evidence is recorded in
[`research/daily_coarse_graining/HANDOFF.md`](research/daily_coarse_graining/HANDOFF.md).

## Next Bounded Milestone

Regenerate the same bounded five-point trajectories with contract v4, fit
sentinel-aware v2 train-only statistics, rerun the inexpensive five-epoch
parameter-conditioned `B_fast` experiment, and evaluate held-out one-step
errors before attempting a seven-day free rollout through retained daily
season/STOMATE. Do not generate all
669 x 50 landpoint-years merely to discover whether the surrogate architecture
can learn the daily transition.
