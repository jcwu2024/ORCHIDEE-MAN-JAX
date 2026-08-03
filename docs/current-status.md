# Current Project Status

Last updated: 2026-08-03.

This is the detailed status and evidence authority, not the shortest
onboarding page. Current work order is in [`NEXT_STEPS.md`](NEXT_STEPS.md),
code ownership is in [`CODE_MAP.md`](CODE_MAP.md), and document status is in
[`DOCUMENT_STATUS.md`](DOCUMENT_STATUS.md).

This page is the current status authority. Dated files under
`docs/source_audits/` and `docs/research/` are evidence snapshots and may
describe an earlier gate without being rewritten.

## Stable Teacher

The canonical user-facing PFT14 Teacher core is on `main` at `7eb41be`.
`research/daily-coarse-graining` merged that commit at `dda3627`; its complete
`jax_orchidee/` tree is byte-identical to `main`. The branches remain separate
only because the research branch also contains experimental dataset and
neural code.

Gate A cross-branch parity is complete. Seven detached-worktree cases cover
cold start, later day, restart, two 365-day cold runs, a 365-day restart run,
three landpoints, and a true 1961-to-1962 year handoff. Six cases have no
differing state or modelout leaf. One low-stock case activates the research
branch's source `min_stomate` correction on day 274; its first fields,
magnitude, continuous-only behavior, and source-extracted Fortran Oracle are
hash-bound in the historical admission evidence. The accepted six-file core
was then synchronized into `main`, so the current drift gate requires raw
parity and has no source-correction exception. See
[`teacher-branch-parity-acceptance.md`](teacher-branch-parity-acceptance.md).

Current evidence:

- cold-start, restart, 365-day, and cross-landpoint executable-reuse gates
  pass;
- seven previously unused landpoints completed 1961-2010 validation;
- their largest 2010 AGB/BGB/GPP/NPP relative error is `8.5e-5`;
- the full 669-landpoint acceptance run remains a release gate.

The project must not claim equivalence for other PFTs or unsupported
ORCHIDEE configurations.

Gate B1 now has a source-backed foundation. The versioned PFT catalog records
stable semantic IDs, canonical Fortran PFT/MTC identities, traits, parameter
row ownership, and explicit process capabilities. Runtime parameter loading,
modelout selection, state checkpoints, and newly created SECHIBA/STOMATE
restart files retain this identity. Restart reads now parse stable metadata or
require an explicit legacy paper layout, then remap all declared PFT axes by
stable ID. The paper layout still activates only bare soil and
`mangrove_pft14`; PFT2-PFT13 are `structural_only` and fail closed if activated.
Remaining B1 work is removal of positional selectors at generic orchestration
boundaries and a complete compact-layout cold/restart lifecycle proof.

## Daily Coarse-Graining Research

The `research/daily-coarse-graining` branch contains the complete Teacher plus
research-only capture, dataset, and neural-surrogate code. The production
Teacher defaults are unchanged because all capture hooks default to off.

### Active architecture decision

The final surrogate target is now a true conservative daily process operator:

```text
day-start state + native 6-hour forcing + parameters/static conditions
  -> daily fluxes, transfer fractions, and bounded tendencies
  -> one constrained water/carbon/energy update
  -> retained source-backed daily processes
  -> next-day state
```

Final inference must not reconstruct 48 interpolated forcing steps or execute
a 48-step state scan. The accepted 96-day `OK_LEAK` driver capture and exact
48-step replay remain Teacher/Oracle evidence, not the final neural boundary.
Direct neural ownership of `carbon_32l`, DOC, or `deepC_peat` remains rejected.

PFT14 is the current training and acceptance scope, but the architecture must
use shared PFT-axis processing conditioned on source-backed traits, parameters,
fractions, and masks. Supporting another PFT later will require Teacher branch
coverage and training data, but must not require redesigning the neural
interface. See
[`research/daily_coarse_graining/conservative_daily_process_operator_v1.md`](research/daily_coarse_graining/conservative_daily_process_operator_v1.md).

No new GPU training is authorized before parameter ownership, daily flux
labels, existing-shard availability, and a true-label non-neural constrained
replay are frozen and pass their gates.

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
- the post-fix admission at commit `7397d1e` passed: Slurm task `14390213_0`
  completed with exit code zero in 42:43, the entry took 2381.0 seconds and
  reached about 29.5 GiB peak process RSS. Its NPZ SHA256 exactly matches the
  original probe, proving the orchestration-only change did not alter
  scientific arrays. This five-worker output remains immutable admission
  evidence and is not mixed into the later 100-worker aggregate;
- the accepted full-production topology is now 20 `cnall` nodes with five
  one-CPU workers per node, for 100 concurrent workers. The launcher commit is
  `74bd4eb`, while the separately validated Teacher worktree remains fixed at
  `7397d1e`. Five-minute within-node startup staggering reduces overlapping
  cold-compilation memory peaks. The validated `w100` plan SHA256 is
  `9ba2caf5...5fa17e`; 69 workers own 350 point-years each and 31 own 300 each;
- the first two-node admission job `14391204` exposed default `srun`
  first-task wait semantics, not a Teacher failure. The explicit `--wait=0`
  fix was then accepted by rerun `14391639`: all ten ranks exited zero in
  55:46, all 11 accumulated point-years and hashes passed, no active lock or
  manifest error remained, and worker 0 closed the 1961-to-1962 restart chain.
  Cold workers peaked near 31 GB RSS and the resumed worker near 26.2 GB;
- consecutive-entry calibration `14391864` passed in 40:03. Its first fresh
  process entry took 1743.348 seconds, while the next three years in the same
  process took 158.455, 158.915, and 160.493 seconds. Mean steady-state
  production is therefore 159.288 seconds per point-year, including capture,
  compression, hashes, and checkpoint output, or about 0.436 seconds per
  simulated day. The post-run hash audit found 15 valid point-years and no
  active lock or manifest error;
- production recovery and consumption are prepared: a progress auditor
  identifies running/partial/missing/invalid workers, arbitrary worker IDs can
  be resumed without changing the canonical 100-worker assignment, and final
  aggregation now requires complete worker inventories, no active lock, all
  shard/metadata/checkpoint hashes, and exact checkpoint chains;
- the complete-data training protocol is frozen at
  `manifests/coarse_graining/daily_teacher_669_training_protocol.json`.
  It specifies exact-once balanced landpoint/year/day streaming with at most
  eight open shards, train/train-only normalization, four non-test
  model-selection slices, sealed final testing, and one-step through
  complete-chain promotion gates. A local real-shard streaming preflight
  passed; it is an I/O gate, not neural accuracy evidence;
- the literature-backed post-admission architecture decision is frozen in
  `long_rollout_architecture_review_20260727.md`: first compare a matched flat
  control with a process- and axis-aware operator, then add mixed-horizon and
  explicit tendency-bias training only if the architecture passes. Long-term
  acceptance targets bounded physical drift and unbiased budgets, not an
  impossible promise of mathematically zero accumulated error;
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

Physical-parameter gradient validation is a separate, conditional promotion
gate. The current rollout experiment validates gradients with respect to
network weights and cross-day state; it does not establish that neural AD with
respect to `alloc_min`, `residence_time`, `vcmax25`, or `maint_resp_slope`
matches the Teacher/Fortran parameter response. This is not a prerequisite for
forward surrogate training or rollout acceptance. It becomes mandatory before
using the neural model for gradient-based parameter calibration, reporting
parameter sensitivities or Jacobians, or calling those physical-parameter
gradients scientifically validated. That gate must use a parent-bound
controlled-parameter-perturbation Teacher dataset, freeze parameter ranges and
held-out combinations before evaluation, compare neural AD against
Teacher/Fortran central finite differences at process, one-day, multiday, and
cross-year scales, and report nonsmooth or inactive branches separately.

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

## Historical Neural Milestones

The material below records completed v3-v5 dataset and rejected-model
evidence. It is not the active work queue. The active milestone is the
parameter/flux contract and non-neural daily replay described above.

The full 669-point baseline is generated and admitted. Finalization job
`14400343` completed in 4:08:54 with exit code zero. Its fail-closed gates
accepted all 33,450 point-years and 12,208,581 transitions, found zero target
persistence mismatches, fitted statistics from the 8,591,565 train/train
samples only, and passed a finite real-batch JAX forward/loss/gradient smoke.
The dataset manifest SHA256 is
`89ea2f24bad8f4b39129937f4106285dd53936c35d8dc610810f865b02d4e508`;
The statistics JSON SHA256 is
`c9ed1c6ea4cfca6c0da9504a36d440a2b388375f01acf8bc78edc7645a4ffb37`;
the statistics NPZ SHA256 is
`33865286021c39bbdb6511a115da55143789e857c31e701c7417ff1a9c50cdbd`;
the acceptance report SHA256 is
`0b13da5c7f8664f9b3b8fe70f6b8dc8e3aca513ecbf15d4b3d879a07d663b596`.

The matched `canonical_flat_v1` versus `axis_process_coupled_v1` architecture
A/B is complete and passed. Both arms used the same admitted samples, update
budget, optimizer, schedule, seed, physical reconstruction, and canonical
objective. Teacher shards were not regenerated and the sealed test split was
not inspected.

The architecture screen was frozen in
`canonical_669_axis_process_architecture_ab.json`, canonical SHA256
`39b94a28e51dba67017ac3f06973b7f4fbbfa4a560b249eec49420f4baed8099`.
It uses one exact-once train/train epoch, batch size 256, learning rate
`1e-4`, seed `20260728`, full temporal/spatial/joint one-step validation, and
no test samples. The production trainer now supports both registered
architectures through the same loss and balanced reader, performs one shared
asset verification for the two-arm run, and avoids per-batch host loss
synchronization.

The prepared real-shard V100 compile/update smoke passed on `gln01` at commit
`0992178`. It hash-verified the train/train shard `001.0-071.0:1961`, used
zero-based sample indices 0-255 as one batch of 256, and consumed no
validation or test sample.
Both arms retained float32 parameters and finite losses. Flat compile/hot
updates took `2.2506/0.00862 s`; axis/process compile/hot updates took
`8.7199/0.00757 s`. The report SHA256 is
`3975eb93000b355b51c565f421f04fdc19342032360c1610e1823dce607f3d16`.

Full job `14403674` completed in `03:00:49` with exit code zero from clean
commit `4c1fe0c`. The final report SHA256 is
`1995fb7f9284bde56c8ac5a9cd411d1bf08fc94f795a0ab1fdad69b42814c9a7`.
Axis/process passed every predeclared gate and produced decision
`advance_axis_process_to_rollout_stability_experiment`. Its
temporal/spatial/joint score ratios relative to flat are
`0.943530/0.960916/0.952866`; the maximum process-family ratio is `0.990947`,
the parameter ratio is `0.995249`, and test evaluation is false. The selected
axis/process checkpoint SHA256 is
`a119999606b063ac9f9ed47e4d1bb664cdfe32b9459e7e7ee24e96a0e944db2e`.

The production screen now performs the full acceptance/hash preflight once,
runs the flat and axis/process arms concurrently on two isolated V100 devices,
and then applies the unchanged aggregate gates. The original single-process
runner remains a sequential fallback. A real-shard two-process smoke passed on
both free `gln01` V100s at commit `ef3ddda`: both reports retained float32
parameters and reproduced identical compile/hot losses for both architectures.
The GPU 0/GPU 1 report SHA256 values are
`ec66dac4be68cc0afc12dac5cd59efc2b3b6932377c999fa6ccc6665ed9a680a`
and
`4001937f2d22916755d9646cef7227cdb26496b5a6375246d8349070a0322351`.
The launcher is restart-safe under the same output root: an existing preflight
must match every frozen input exactly, completed arm checkpoints/reports are
resumed without another epoch, incomplete arms rerun, and worker logs append.
The 12-hour Slurm limit is retained as a safety ceiling.

The conditional post-architecture rollout-stability experiment was frozen
before the parent result in
[`canonical_669_rollout_stability_protocol.json`](../manifests/coarse_graining/canonical_669_rollout_stability_protocol.json),
Its v1 canonical SHA256 was
`370011f6d8edc447bd0ebf037249df1a18f3bfb30071204001366a2aecde310b`.
Its auditor binds the admitted 669-point assets and architecture-screen hash,
keeps the test split sealed, and rejects drift in the matched arms,
`1/3/7/30`-day sampling, loss inventory, train-only coefficient calibration,
hard constraints, screening thresholds, promotion ladder, or stop rules.
Experiment B runs only if the parent decision is exactly
`advance_axis_process_to_rollout_stability_experiment`; a rejected architecture
returns to a named architecture hypothesis rather than silently applying the
stability objective to the flat model.

The Experiment B training implementation and real-shard GPU gate now pass.
Horizon `1/3/7/30` protocol shapes completed on a V100 with two distinct train
landpoints sharing one executable, finite first and second updates, zero hard
counts, and exact checkpoint resume. Horizon-7 and rematerialized horizon-30
hot updates take about `0.10 s` and `0.20 s`, with peak allocator use below
`0.55 GiB`.

Host preparation was then reduced without changing any measured loss,
component value, gradient norm, or hard count. Each shard is loaded once,
runtime/static state is cached by landpoint, and the five forcing leaves
actually consumed after the learned fast-day boundary are reconstructed in
one vectorized batch operation. Steady forcing preparation fell from seconds
to about `0.015 s`; a complete warm-cache horizon-7 preparation is `0.245 s`.
A 16-landpoint cache gate measured about `7.8 MiB` RSS growth per added
landpoint, making the 535-train-landpoint cache safe on a gnall node. Details
are in
[`rollout_stability_smoke_20260729.md`](research/daily_coarse_graining/rollout_stability_smoke_20260729.md).

Preflight, calibration, execution, and arm checkpoint identities now bind the
training Git HEAD in addition to protocol, data, schedule, model, and
environment.

The first paid Experiment B attempt, job `14410820`, failed after `00:07:47`
during train-only coefficient-calibration ordinal 4, before either training
arm started and before any checkpoint was written. The exact batch was
landpoint `087.0-105.0`, year 1961, horizon 1, batch size 256. It contained
three and only three defined-status errors, all in the PFT14 column of
`diffuco_previous_step_state.rveget`, on Days 2, 294, and 358. There were zero
discrete mismatches, nonfinite defined values, or negative source-constrained
carbon stocks. The sealed test split was not used. The diagnostic SHA256 is
`f24eeef2f937c2f828b4b7449cdadbaa5d60ef8cf581d186564eb622e6751a2e`.

This exposed a protocol contradiction rather than a new retained-tail defect:
`rveget` defined/undefined is an explicitly declared learned classification
target, but v1 also treated every classification mistake as an exact-zero
optimizer veto. Protocol v2, canonical SHA256
`a0ecfd4c89ff3c5691f153168f3ba80939582774689b92ff04b0cf77e699cb40`,
separates declared dynamic `rveget` status errors from unexpected structural
status errors. Declared errors remain supervised, counted, reported, and
screened against the matched control; they no longer veto an optimizer update.
Unexpected status errors, discrete mismatches, nonfinite defined values, and
negative source-constrained carbon stocks remain exact hard failures.

The required free `gln01` gate passed at commit `13f09c9`. It reproduced
calibration ordinal 4 with exactly three declared dynamic errors, zero
unexpected/discrete/nonfinite/negative-stock hard counts, finite loss and
gradient norm, and `update_applied=true`. The sealed test split was not used.
The smoke report SHA256 is
`66dc952abef5804eabe43ca8a9e9144828caba9649452557ded274f2bf8146a5`.

A later paid attempt reached mixed-horizon update 226 before failing on two
rollout samples with nonfinite gradients. Source-owner isolation proved that
the failure came from the NPP negative-stock add-back landing one ULP above
`min_stomate`, which activated a strict leaf-fraction division gate. The same
pathology was reproduced from the original Fortran procedure under both
`gfortran -O0` and `-O3`. Commit `f9554c6` preserves the source carbon budget
while pinning the corrected stock exactly to the intended threshold.

After that stabilization, both failing samples had finite gradients, the
complete update-226 audit passed 64/64 rollout samples with all hard counts
zero, and the candidate update was applied. A resumable sequential prefix gate
then passed all 256 mixed-horizon updates at commit `452b743`, including an
exact checkpoint restart at update 128 and coverage of horizons 1, 3, 7, and
30. The sealed test split remains unused.

Formal Experiment B rerun `14419242` completed the full 8,192-update control
but failed closed at candidate update 302. Exact checkpoint replay and
horizon-prefix bisection localized the first bad transition to 1973 Day 355.
The forward model remained finite; reverse-mode AD failed in the STOMATE
leaf-age turnover quotient at a `3.7058e-311` leaf stock. Commit `90113c0`
preserves the source-equivalent quotient primal and masks only quotient
tangents below the float64 representable-gradient boundary. The complete
update-302 audit then passed with zero bad samples, finite independent
component gradients, all hard counts zero, and an applied update.

An identity-checked diagnostic checkpoint fork at commit `b8c8dee` resumed the
formal update-256 state and passed every mixed-horizon update through 1024.
All 768 post-fix updates were applied; horizon counts were `431/317/182/94`
for 1/3/7/30 days, all exact hard counts stayed zero, and the sealed test split
was not used. The failed formal root remains rejected. See
[`rollout_turnover_ratio_gradient_20260730.md`](research/daily_coarse_graining/rollout_turnover_ratio_gradient_20260730.md).

The clean formal Experiment B rerun is complete. Slurm job `14425151` ran on
one V100 from training commit
`d6e43cc79f590e494f79fe54adba4ac2c28779b6`, completed in `04:16:53`, and
returned exit code zero. Both matched arms completed all 8,192 updates with
identical horizon counts `3277/2458/1638/819` for 1/3/7/30 days and did not
access the sealed test split. The immutable output root is
`runtime/outputs/training/canonical-669-rollout-stability-v2-d6e43cc`.
Control/candidate checkpoint SHA256 values are
`8a099c85f8c869398ffe2203b0fa1d558735b0f819ac606d112d452db2fb2ec1`
and
`4f36bb39839f8fba7a9cbd0283610bced3bd45c0688891bcdb4d051917320300`;
their training-report SHA256 values are
`6ca7985edd5b5b61beecd534a7f341a7173a6ef549b39ddccca3b93be2237069`
and
`ffd25be74ab739d15c45f66d04fc79d304e4877821b91f23f12d85da5749184c`.
This proves successful matched training, not promotion.

Commit `a9d2390` adds the fail-closed full-data screening evaluator. Local
validation passes 64 tests, Ruff, `py_compile`, shell syntax, and diff checks.
Real V100 smokes cover temporal, spatial, and joint model-selection shards,
large batches `512/512/256`, and a bit-exact 30-day versus `15+15`
restart-split probe with maximum difference zero. The evaluator covers every
valid 1/7/30-day teacher-forced and free-rollout window, the complete state,
process families, tendency bias, 12 named science fields, dynamic status and
hard constraints; incomplete evidence cannot be classified.

The formal matched screen completed as Slurm job `14430377` in `04:21:26`
with exit code zero. All 4,754 point-year references and every valid
teacher-forced/free 1/7/30-day window were evaluated; the report SHA256 is
`1387448b0b0831661daaaffc8985ef22602540e0c7e91fea7bfe96c085828bb0`.
The candidate is rejected: 114 of 223 relative gates fail. All one-step global
no-regression gates and all three 30-day global rollout gates pass, with
30-day temporal/spatial/joint error ratios `0.9181/0.7539/0.8047`. However,
none of the 7-day global gates reaches the required 5% improvement, 46/54
tendency-bias gates fail, and 57/108 named science-field gates fail. NPP,
growth and maintenance respiration, biomass, LAI, `carbon_32l`, and
`deepC_peat` are the repeated failures.

All exact hard and structural constraints pass, including zero unexpected
status/discrete/nonfinite/negative-stock errors and a bit-exact 30-day
`15+15` restart split. This is therefore an objective/architecture failure,
not a numerical-integrity or orchestration failure. Do not run confirmation
seeds, inspect the sealed test split, or begin 365-day validation for this
candidate. See
[`rollout_stability_screening_20260731.md`](research/daily_coarse_graining/rollout_stability_screening_20260731.md).

Experiment C is now frozen as a bounded causal-carbon hypothesis, but has not
yet run on a real GPU shard. It keeps the Experiment B one-step control as a
frozen 1,954,041-parameter parent and attaches an exact-zero
76,877-parameter adapter. The adapter can alter only 141 PFT14 columns:
same-day GPP, 12-part maintenance respiration, 32-layer carbon, and deep peat
carbon. The other 2,714 fast-day columns and the dynamic undefined head remain
parent-owned and bit-exact. The objective directly supervises this causal
`B_fast` interface, then separately supervises downstream carbon state,
signed flux bias, and stock-tendency bias; reset day-end `gpp_daily` is
explicitly excluded as a proxy for same-day GPP. DOC and litter are
no-regression guards.

The protocol SHA256 is
`35ede512bbc5f55881317cb700d7512d71993391c65f36ed8a5de087c4d4c3e7`.
Local architecture/objective/protocol regression and a complete synthetic
two-day retained-transition reverse-mode path pass. Paid matched training is
forbidden until an eight-update, real train/train-shard V100 feasibility gate
passes all 1/3/7-day, frozen-parent, protected-column, gradient, and hard-state
checks. This preparation does not establish that Experiment C improves
rollout quality. See
[`causal_carbon_adapter_experiment_20260731.md`](research/daily_coarse_graining/causal_carbon_adapter_experiment_20260731.md).

The Experiment C real-shard feasibility gate subsequently passed at commit
`79258fd`. Eight matched train/train updates covered 1/3/7-day horizons on a
free `gln01` V100. Every update was applied with finite gradients; all
unexpected/dynamic status, discrete, nonfinite, and negative-stock counts were
zero. The frozen 1,954,041-parameter parent stayed outside the optimizer, all
2,714 protected columns and the dynamic undefined head remained bit-exact,
and the maximum protected-column difference was `0.0`. Report SHA256 is
`ca8cacdccb42f20b94b5cd64796118cb170f7fbc7813c036cbc6b1edc8aa7fa8`.
The smoke checkpoints are not training parents. Formal train-only
independent-component gradient calibration must precede a fresh matched A/B
from the exact-zero adapter. See
[`causal_carbon_adapter_feasibility_20260731.md`](research/daily_coarse_graining/causal_carbon_adapter_feasibility_20260731.md).

The formal Experiment C lifecycle completed as paid job `14434127` at training
commit `8d24fd0`. It ran for `02:33:11` on one `gnall` V100 and exited zero.
All 48 deterministic train/train calibration records and both matched
4,096-update arms completed. Each arm has exact horizon counts
`2048/1229/819` for 1/3/7 days. Both final parent-invariance checks report
bit-exact protected columns and dynamic undefined head with maximum protected
absolute difference `0.0`; neither arm used sealed test data. Evidence hashes:

```text
control checkpoint:
4ee68de505f0163a62231e0310369882d46a0b72663b87d2a394c0e4ffeef720
control training report:
7400144a2f7143d5591e7a1bf2b49cfcd0c50d55385cee71dcd481c554e2170c
candidate checkpoint:
ea7193453e279bd94eb55984c629a0f530d0275466c1cd0daccf142754497892
candidate training report:
0a358a32f9b156c2d2434a0d11b6e0a8a35ceae47f3f6f0fdd7f3b78d7dc0eaf
```

This closes formal training execution only; it does not establish that the
candidate improves model-selection rollout quality.

The post-training screen was frozen before any candidate validation output was
available. It evaluates all temporal, spatial, and joint model-selection
windows at Day 1 teacher-forced and Day 7/30 free rollout. Classification
requires all 165 relative, 108 exact-zero, and six structural gates to pass;
the sealed test remains unread. After training, run a one-shard-per-slice free
`gln01` smoke before requesting the paid all-sample screen. The smoke
subsequently completed on GPU 1 in `286.46` seconds at evaluation commit
`375b8d2`. It covered one complete temporal, spatial, and joint shard for both
arms, with all unexpected-status, discrete, nonfinite, and negative-stock
counts zero. Both restart-split probes passed, and sealed test data remained
unread. The top-level smoke report SHA256 is
`f05d88d97ec93152487f22697e50361de705a6e1223964971bdfccdcd777b7cb`.
As preregistered, this limited smoke is not promotion-eligible; it authorizes
the paid all-sample screen.

The first paid submission, job `14436672`, exited after two seconds before
starting the evaluator because the Slurm wrapper directly executed a
non-executable checkout file. Commit `7091e1d` changes only that invocation to
explicit `bash` and adds a regression test; it does not alter the evaluator or
frozen gates. Replacement job `14436716` completed all 4,754 references in
`04:25:50` with exit code zero from the immutable `7091e1d` checkout.

Experiment C is rejected with decision
`stop_and_attribute_declared_screening_failure`. All 108 hard and six
structural gates passed, but 73/165 scientific relative gates failed:
11/12 causal-interface, 46/63 primary-state, 3/27 flux-bias, and 13/36
stock-tendency gates. All nine global terminal-state and all 18 litter/DOC
guard gates passed. The dominant failure is direct damage to the carbon-stock
boundary: `carbon_32l` interface error worsened by `7.33-9.32x` and
`deepC_peat` by `5.66-7.45x` across all three model-selection slices. The
matched report SHA256 is
`2b636981a5aeb73479c354558c09ed99cf398559c7f315bd56774f7f9822b5dd`.
Do not run confirmation seeds, 365-day validation, complete-chain validation,
or sealed test for this candidate. See
[`causal_carbon_adapter_screening_20260731.md`](research/daily_coarse_graining/causal_carbon_adapter_screening_20260731.md)
and
[`causal_carbon_adapter_screening_protocol_20260731.md`](research/daily_coarse_graining/causal_carbon_adapter_screening_protocol_20260731.md).

A single post-rejection attribution is now frozen before any successor is
built. It evaluates the rejected checkpoint with only
`carbon_stock_interface` set to exact zero while retaining the trained flux
adapter. The one-shard-per-slice smoke must reproduce the prior control,
recover `carbon_32l` and `deepC_peat`, retain at least 24/27 flux-bias gates,
and keep every hard and litter/DOC guard gate. It does not retrain or revive
Experiment C. See
[`causal_carbon_adapter_stock_ablation_protocol_20260731.md`](research/daily_coarse_graining/causal_carbon_adapter_stock_ablation_protocol_20260731.md).

That one-shard-per-slice ablation is complete at commit `7c53f79` and fails
its predeclared gate. It exactly preserves the original candidate's direct
flux interface, and its 27 downstream flux-bias metrics change by only about
`-0.23%` to `+0.31%`. It strongly repairs temporal and joint carbon-stock
error, but spatial `carbon_32l` and `deepC_peat` remain `13.71x` and `16.20x`
worse than the matched control. Thus neither independent stock residuals nor
an unconditional return to parent-owned stocks is acceptable. The next
candidate requires source-backed learned transfers with a conservative stock
update. No paid training is authorized yet. See
[`causal_carbon_adapter_stock_ablation_result_20260731.md`](research/daily_coarse_graining/causal_carbon_adapter_stock_ablation_result_20260731.md).

The subsequent source audit closes the ownership ambiguity. Contract v5's
1,376 compact `OK_LEAK` values contain 1,222 independent inventory values,
64 derived `deepC_peat` values, and 90 ratio or overlapping tracer/partition
values. `deepC_peat` is reset from `carbon_32l` before decomposition and must
not be a neural stock owner. Existing shards provide endpoint supervision but
not aggregate transfer labels or the 13 half-hour driver series required by
the exact source-backed scan. At that historical decision point, a bounded
successor predicted those drivers and executed the existing 48-step
`_paper_compiled_ok_leak_fold`. Its capture and replay evidence remains valid,
but the 2026-08-02 true-daily decision supersedes it as the final surrogate
design. See
[`carbon_budget_ownership_audit_20260731.md`](research/daily_coarse_graining/carbon_budget_ownership_audit_20260731.md).

The bounded train-only auxiliary asset and persisted replay gate are now
complete. The accepted plan selects 16 days from each of six training
landpoints across years and process conditions. All `96/96` target-day
captures pass the exact capture interface with no failed record; the
hash-bound reader verifies every source shard, state/target row, report, and
driver NPZ. Replaying each persisted 13-driver sequence through the current
exact 48-step scan reproduces all 14 `ok_leak.*` endpoints within `1e-12` for
`96/96` days. The global maximum absolute error is `4.5401182813264995e-14`;
six records are fully bit-exact and 77 have 13/14 bit-exact endpoints. The
remaining differences are float64 evaluation-order noise below the frozen
tolerance, not relaxed acceptance. The sealed test split was not used.

This closes capture-interface and persisted-scan replay sufficiency. It does
not assert equivalence of a reconstructed full outer seven-day Teacher block:
historical shards were produced in seven-day compiled blocks, and one-day
outer replay is not the accepted verification boundary. The next action is to
close conservation, feasible-perturbation/nonnegative-stock, real
forward/reverse finite-gradient, and bounded tiny-fit parent-no-regression
micro-gates before any paid neural A/B. See
[`ok_leak_auxiliary_capture_replay_20260801.md`](research/daily_coarse_graining/ok_leak_auxiliary_capture_replay_20260801.md).

The source-backed combined carbon-inventory diagnostic is now connected to a
diagnostic-only full-step output of the same 48-step scan. The production
return boundary remains unchanged. A real persisted-driver smoke for
`001.0-071.0`, 1961 Day 2 passes all 48 steps: all values are finite, all five
independent stock families are nonnegative, the diagnostic final carry is
bit-exact to ordinary replay, and maximum absolute closure is `2.9293e-9`.
The closure policy uses the source `min_stomate=1e-8` verdict from
`constantes_var.f90`; relative closure (`1.1884e-8` at the worst low-stock
step) is retained as a diagnostic and is not an invented source gate. Job
`14446767` completed in `00:06:38` with exit code zero. Feasible perturbation,
real forward/reverse gradient, and bounded tiny-fit gates remain.

A deterministic bounded selection tool is now ready for the existing
server-side v5 shards. It admits only the six spatial-train/temporal-train
landpoints and selects 16 days per point: cold-start anchor, low/high extrema
of six reachable dynamic process metrics and rank-space farthest fill.
`shumdiag_peat` and `runoff2peat` remain exact-zero contract fields because
the audited HYDROL-local SAVE branch is statically
`peat_hydro=F/branch_peat=F`; changing forcing or landpoint cannot activate
it. `PERMA_PEAT` carbon redistribution remains in the exact scan. Static
`fpeat` is report-only. The intended 96
days require about 44.58 MiB of uncompressed driver labels. Source shard,
day-start state, and endpoint hashes are mandatory, and sealed references are
filtered before file access. The first real plan
`500979ab...d9166a3` is rejected because its old static-`fpeat` proxy was zero.
The second `f7967d70...09d0f92` plan is also rejected because it required an
unreachable peat-hydrology branch and ranked tied zeros by date. Neither is a
capture asset. The accepted replacement is frozen at
`manifests/coarse_graining/ok_leak_auxiliary_capture_96day_v1.json`, canonical
SHA256 `8ebe5345...3b922c7`. It passes all 19 independent checks over 96 unique
train-only days, including exact selected-row state and target hashes. The
complete capture and replay result is recorded in
[`ok_leak_auxiliary_capture_replay_20260801.md`](research/daily_coarse_graining/ok_leak_auxiliary_capture_replay_20260801.md).
See also
[`ok_leak_auxiliary_capture_selection_protocol_20260731.md`](research/daily_coarse_graining/ok_leak_auxiliary_capture_selection_protocol_20260731.md).

The restartable one-process capture and persisted-replay runners are now
validated on all 96 accepted records. They verify the source generation plan,
stage each day atomically, and resume only hash-identical passed records.
Capture-interface failures and persisted-replay failures are both zero.
