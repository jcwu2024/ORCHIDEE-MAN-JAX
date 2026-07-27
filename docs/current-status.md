# Current Project Status

Last updated: 2026-07-27.

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

- Daily Fast-Day Teacher Contract v5:
  `S[d] (3,854 continuous values plus 12 exact discrete values) + native
  6-hour forcing[d] + P -> B_fast[d] (2,855 values)`, followed by retained
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
- a hash-bound post-generation acceptance gate that requires a complete v4
  aggregate, audits target representation, fits sentinel-aware train-only
  statistics, and passes a real-batch finite forward/loss/gradient smoke
  before GPU training;
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
- the contract-v4 nine-point V100 baselines established
  nontrivial one-step learnability: its equal-split validation score was
  0.812 versus 1.064 for the matched persistence baseline, but spatial and
  joint scores remained 0.855 and 1.438;
- that run exposed an absolute-state `rveget` undefined classifier which
  damaged a stronger persistence prior; the accepted follow-up predicts only
  defined/undefined flips and reduced temporal/spatial/joint classification
  errors to `18/0/0`;
- a fresh 15-epoch run reached its best equal-split score of `0.772` at epoch
  10, improving temporal/spatial/joint scores over matched persistence by
  `67.0%/28.0%/18.0%`; additional epochs did not resolve the dominant spatial
  DIFFUCO/ENERBIL and HYDROL errors;
- no user-facing `daily-surrogate` run mode exists;
- the first seven-day v4 free rollout reached `0.921` normalized state RMSE,
  while teacher-forced daily error stayed near `0.11-0.15`; this identifies
  recursive state-distribution drift rather than a retained-tail handoff error;
- contract v4 was then found to omit the fast-owned cross-day `leaf_ci` output.
  Contract v5 appends it losslessly from stored `S[d+1]`, so no Teacher rerun
  is required, but all v4 statistics and checkpoints are superseded;
- the pure-JAX retained-tail transition passes a real Day 2 forward and
  reverse-mode gate: maximum next-state error `5.68e-14`, zero mask/discrete
  mismatch, and finite gradients for all 2,847 defined target values;
- a real three-day `lax.scan` gate also passes: maximum trajectory error
  `1.71e-13`, zero mask/discrete mismatch, and finite gradients for all 8,541
  defined day-target values;
- the complete nine-point v4 asset has been losslessly migrated to v5 without
  rerunning Teacher: 450/450 landpoint-year shards and about 1.4 GiB passed
  source/output hash verification;
- the v5 dataset acceptance gate passed on 96,354 train/train transitions with
  zero target-representation persistence mismatches and finite forward, loss,
  and gradients for the 1,963,369-parameter canonical model;
- a real V100 one-window multistep smoke passed at research commit `f68ea59`:
  one horizon-1 update completed in 104.84 seconds including first compile,
  with loss `0.0495625`, gradient norm `0.389538`, and a finite 31.4 MB
  checkpoint; this also closed canonical generation-plan hashing and external
  server data-root wiring in the multistep launchers;
- the first complete v5 V100 curriculum finished on `gln01` in 35:03. Its
  ten-epoch one-step initialization selected epoch 10 at validation score
  `0.680439`; the subsequent `1:64,3:64,7:64` stages remained finite;
- on the joint spatial-and-temporal validation window at landpoint
  `215.0-119.0`, year 2005, days 100-106, the seven-day free-rollout final
  state RMSE improved from `0.473270` for the one-step checkpoint to
  `0.429058` after multistep training, with zero defined-status or discrete
  mismatches. Teacher-state feedback kept all daily RMSE values below
  `0.086930` and ended at `0.068236`, so recursive state-distribution drift,
  rather than an invalid retained-tail handoff, remains the dominant error;
- a complete field-level drift diagnostic at research commit `18471aa`
  reproduced every daily rollout RMSE exactly. The current multistep state
  loss is uniform over 3,854 scalar values: slowproc/STOMATE owns 2,386 values
  and 61.91% nominal weight, but already contributes 76.89% of the first-day
  realized Huber state loss. Biomass alone contributes 36.39% on Day 1 and its
  normalized RMSE grows from `0.660380` to `4.841768` by Day 7; growth and
  maintenance respiration, NPP, litter partitioning, and surface energy state
  are also early divergence leaders. This rules out the simplistic diagnosis
  that important carbon state is merely invisible to the loss. The next
  objective must combine process-balanced weighting with explicit recursive
  stability/state-change supervision, then be tested under a controlled A/B;
- a four-way split diagnosis at research commit `825935b` compared
  train/train, train/validation-year, validation-point/train-year, and joint
  validation rollouts on the same Day 100-106 window. Day-7 RMSE was
  `0.145471`, `0.134233`, `0.519886`, and `0.429058`, respectively. The
  controlled temporal holdout did not degrade the training landpoint, while
  the spatial holdout failed even in a training year. Spatial-condition
  coverage is therefore a demonstrated limitation of the current six
  training landpoints. Recursive objective failure also remains: train/train
  `litterpart` grows from `0.541` to `3.652`, proving that the global `<=0.29`
  gate can hide a scientifically important low-dimensional failure. Do not
  add more years of the same points; first run one process-balanced/state-
  increment objective A/B, then add a bounded, source-selected spatial pilot;
- the bounded `process_increment_v2` equal-budget A/B completed and is
  rejected. On train/train Day 7, global normalized state RMSE changed from
  `0.145471` to `0.146456`, biomass from `0.014259` to `0.016716`, and
  litterpart from `3.651536` to `4.008877`. The candidate retained zero
  defined-status and discrete mismatches but failed all required seen-condition
  improvement gates. `canonical_multistep_v1` remains the baseline; do not
  continue tuning process/increment loss weights;
- the counterfactual Teacher-at-model-state gate passed at research commit
  `87e1132`: exact Teacher re-entry completed at model-generated Days 2, 4,
  and 7 with zero defined-status or discrete mismatch. Mean matched operator
  error was `0.085390`, mean state sensitivity was `0.073494`, and their ratio
  was `1.161865`. This supports stop-gradient pushforward with same-state
  on-policy Teacher labels;
- the real detached-prefix pushforward smoke passed at research commit
  `c3e2d79`. After a three-day model prefix, one exact same-state Teacher query
  and final-transition update produced finite loss `0.0836542`, gradient norm
  `0.1333402`, and a finite checkpoint;
- the corresponding 192-update, batch-4 `on_policy_pushforward_v1` A/B is
  complete and rejected. On train/train Day 7, global RMSE improved from
  `0.145471` to `0.092764` and litterpart from `3.651536` to `0.854632`, but
  NPP and growth respiration regressed about 9.6%. Spatial-validation global
  RMSE worsened from `0.519886` to `0.550696` and joint validation from
  `0.429058` to `0.459871`; spatial litterpart also worsened by more than 60%.
  All mask/discrete gates passed, but the predeclared named-state and spatial
  gates did not. Keep `canonical_multistep_v1`; do not tune this candidate or
  inspect the sealed test split;
- deterministic replay of that run's single extreme batch proved a separate
  physical-domain defect in neural rollout reconstruction: the unconstrained
  network produced negative `carbon_32l`, `DOC`, and `deepC_peat` stocks before
  exact Teacher re-entry, causing NaNs and a `7.27e8` soil-carbon artifact.
  Commit `5414d8c` now applies the source-backed nonnegative-stock projection
  at the shared NumPy/JAX physical reconstruction boundary. The exact four
  replayed losses changed from `0.0997/0.5658/0.0984/562889` to
  `0.0999/0.0946/0.0970/0.1165`, with defined-status mismatches falling from
  980 to zero;
- the frozen four-way seven-day baseline was rerun after that semantic fix.
  Day-7 RMSE is now `0.144765`, `0.133391`, `0.519286`, and `0.428353` for
  train/train, train/validation-year, validation-point/train-year, and joint
  validation. The small changes leave the scientific diagnosis intact:
  spatial generalization remains dominant and train/train `litterpart` still
  reaches `3.651537`;
- the matched `structured_process_film_v1` architecture A/B is complete and
  rejected. Relative to an identically continued flat control, structured
  Day-7 RMSE improves 4.36% and 4.24% on the two spatial-validation cases but
  regresses 7.68% and 11.32% at the seen landpoint. Its maximum useful
  parameter/static validation permutation effect is only `5.26e-5`, below the
  predeclared `1e-4` condition-use threshold; parameter permutation slightly
  improves loss, showing a wrongly learned association rather than a dead
  channel. The flat continuation itself worsens all four frozen global scores,
  so neither checkpoint is promoted. See
  `structured_architecture_ab_result_20260726.md`;
- the reusable 669-point Teacher data-product admission now passes at the
  contract level. The versioned machine gate binds the frozen 535/67/67
  spatial split and 1961-2010 temporal split to Markov Contract v5, verifies
  all state/target/diagnostic/forcing/condition inventories, and computes
  33,450 point-years with 12,208,581 transitions. Its final mode additionally
  requires all 33,450 aggregate shards and exact split assignments. The next
  data operation is therefore full 669-point baseline generation after
  staging/plan/resource approval, not another architecture-specific spatial
  pilot. Parameter perturbations and new forcing products remain separate
  parent-bound extension datasets;
- the first real bounded 669 production probe generated and hash-verified the
  complete `001.0-071.0:1961` v5 shard at contract SHA256
  `6813065b...9d442`. It took 3084.5 seconds for the entry and reached about
  29.5 GiB peak process RSS. Slurm's exit code 2 was an orchestration defect:
  the old CLI reported an intentionally partial `--max-new-entries` worker as
  failed after successfully writing its resumable manifest. That exit
  semantics now has a regression fix. The `07fe2f9` probe remains immutable
  admission evidence; full production must use one fresh output root bound to
  the accepted post-fix commit rather than mixing commit identities;
- the provisional seven-day free-rollout gate of `<=0.29` therefore did not
  pass. No v5 30-, 365-day, or 50-year neural rollout has passed;
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

Freeze the post-probe Teacher-generation commit, rebuild the canonical
669-point plan with a fresh output root, and repeat one real shard to bind that
provenance identity. After admission, generate the full baseline in bounded
five-worker CPU waves and run the final 33,450-shard data-product gate before
reopening neural architecture or objective work.
