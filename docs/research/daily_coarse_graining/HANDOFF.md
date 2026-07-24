# Daily Coarse-Graining Handoff

Snapshot date: 2026-07-24

This is the single operational handoff page for the daily coarse-graining
research branch. Read this page before dated experiment reports. Stable
scientific and release facts remain authoritative in
[`../../current-status.md`](../../current-status.md).

## Five-Minute Startup

1. Check out `research/daily-coarse-graining` and confirm a clean worktree.
2. Read this page, then
   [`daily_markov_contract_v5.md`](daily_markov_contract_v5.md) and
   [`teacher_dataset_generation.md`](teacher_dataset_generation.md).
3. Use the temporary `cln02` login at `192.168.11.2` until the user confirms
   that `cln01` has been restored.
4. Check the active ten-point v4 recovery state below. Do not rerun completed
   shards.
5. After aggregation, run the canonical dataset-acceptance command before any
   GPU training.

```bash
git status --short --branch
git log -1 --oneline
```

The ten-point v4 production code is fixed at commit `1f19ed7`; later research
commits do not alter its process image or outputs. Any recovery of point 319
must use a dedicated clean worktree at `1f19ed7`, not the advancing neural
worktree. Any future generation run must record its own exact Teacher commit.

## Current Architecture

The learned operator replaces exactly this block:

```text
48 half-hour SECHIBA/HYDROL/THERMOSOIL/DIFFUCO/ENERBIL transitions
+ daily accumulation and maintenance
+ 48 ordered OK_LEAK transitions
```

Its contract is:

```text
S[d] (3,854 continuous + 12 exact discrete values)
  + native 6-hour forcing[d] + parameters/static conditions
  -> B_fast[d] (2,855 values)
B_fast[d] + S[d]
  -> retained source-backed daily season/STOMATE
  -> S[d+1]
```

`B_fast` is a complete same-day interface, not persistent state. `S` is the
cross-day Markov state. Contract v5 carries only the true cross-day subset of
the 93-field SECHIBA finalize packet and learns only its fast-owned `leaf_ci`
output, not the duplicated finalize packet. For a 1961 cold start, the real Teacher executes Day 1
to construct canonical `S[1]`; the first training sample is Day 2. The shard
schema is `daily_teacher_markov_year_v5` and uses lossless NPZ compression.

Contract v4 is superseded. A real 1962 Day 2 differentiable retained-tail gate
proved that v4 could not advance `leaf_ci`; all other next-state leaves closed.
V5 appends 40 compact `leaf_ci` values from existing `S[d+1]`, so the complete
nine-point v4 dataset can be migrated without rerunning Teacher. The v5 gate
passes at `5.68e-14` maximum state error with zero mask/discrete mismatches and
finite gradients for all 2,847 defined target values.
The three-day real retained-tail scan also passes at `1.71e-13` maximum error
with 8,541/8,541 finite defined target gradients.
The reverse compile still emits the nonfatal XLA algebraic-simplifier 50-run
warning and then completes; keep it as a runtime-version A/B item.

The intended final repository has independent runtime choices for execution
backend (`cpu` or `gpu`) and transition implementation
(`teacher_half_hour` or, after acceptance, `neural_daily`). The research
branch is temporary development history, not a separate product.

## Active Ten-Point v4 Production

The frozen ten-point v4 run uses plan
`runtime/plans/teacher_initial_10point_v4_1f19ed7.json`, plan SHA256
`b32eef75aafc6793c886bcec5858b4f0b343b8e510d12fb744518d35030d84af`,
four persistent one-CPU workers, and output root
`runtime/outputs/training/pft14-daily-teacher-initial-10point-1961-2010-v4-1f19ed7`.

Worker array `14365952` completed 459 of 500 landpoint-year entries. Workers
1-3 completed. Worker 0 timed out after completing all years for points 001
and 281 plus point 319 through 1969. Its residual internal job `14365953`
was finally removed on 2026-07-24; `squeue` no longer recognizes it and its
worker steps are recorded as cancelled. The only unfinished work is point 319
for 1970-2010, 41 entries. The obsolete aggregate `14365958` cannot pass its
`afterok` dependency.

After `CG` clears, read and formally recover the existing worker-0 lock, then
resume worker index 0 with worker count 4, the same plan/output root, and
`ibc11b02n13` excluded. Do not rerun any completed shard. Submit a fresh
aggregate only after the resumed worker completes.

Neural architecture work does not need to wait for that recovery. A
provisional nine-point dataset may be materialized with `teacher_shards
aggregate-subset`, selecting the nine complete 1961-2010 landpoint chains and
excluding point `319.0-057.0` entirely. It must use a distinct dataset ID,
statistics asset, acceptance report, experiment directory, and checkpoint.
The partial 1961-1969 point-319 chain must not enter this dataset. Results are
architecture-development evidence only and cannot replace the frozen
ten-point v4 experiment.

The provisional nine-point v4 dataset, flip-classifier A/B, and fresh 15-epoch
V100 run are complete historical architecture evidence. See
[`nine_point_v4_neural_baseline_20260724.md`](nine_point_v4_neural_baseline_20260724.md).
The accepted flip classifier restored persistence-level defined/undefined
accuracy. The 15-epoch run reached its best equal-split score of `0.772370` at
epoch 10 versus `1.064329` for matched persistence. Temporal/spatial/joint
improvements were `67.0%/28.0%/18.0%`. Longer training did not resolve the
DIFFUCO/ENERBIL and HYDROL spatial errors. Its seven-day free rollout reached
`0.921` RMSE and is superseded by the v5 contract correction. Do not evaluate
the sealed test split or merely add more v4 epochs.

The nine complete chains were migrated losslessly to contract v5 at
`runtime/outputs/training/pft14-daily-teacher-9point-1961-2010-v5-4c0f886`.
Migration job `14371675` produced 450/450 shards; provenance repair job
`14371681` rebound the source plan, subset, source-manifest hash, and migration
snapshot without rewriting shard arrays. Acceptance job `14371682` completed
in 1:37 with about 203 MiB peak RSS. Its assets are under
`acceptance-v5-5a16e78`: 96,354 train/train samples, zero target persistence
mismatches, and finite real-batch forward/loss/gradients. The earlier
acceptance attempt `14371680` is rejected because its migrated manifest lacked
the inherited generation-plan hash.

## Accepted Historical Production Run

The bounded architecture-development dataset is frozen by
[`../../../manifests/coarse_graining/daily_teacher_initial_5point_1961_2010.json`](../../../manifests/coarse_graining/daily_teacher_initial_5point_1961_2010.json):

- 5 landpoints: 2 train, 2 validation, and 1 test;
- 1961-2010 complete trajectories;
- 250 atomic landpoint-year shards;
- 91,245 daily transitions after excluding cold-start Day 1 at each point.

The first submission (`14362790`/`14362791`, commit `23046a3`) was cancelled
and rejected before acceptance. Its plan incorrectly assigned 366 days to
Gregorian leap years even though the paper forcing and Teacher lifecycle use a
fixed 365-day noleap calendar. Those outputs are diagnostic evidence only.

Replacement Explore1000 jobs use the noleap fix at commit `3afe92f`:

- worker array: `14363439`, two persistent one-CPU workers on `cnall`;
- after-success aggregate: `14363441` on `cnmix`;
- plan: `runtime/plans/teacher_initial_5point_noleap_3afe92f.json`;
- output: `runtime/outputs/training/pft14-daily-teacher-initial-5point-1961-2010-noleap-3afe92f`;
- plan SHA256: `353f517ddc84a46723a71de11b74a557af030fe367314689b8a6b6fb8e8e68b9`;
- validated plan: 5 landpoints, 50 years, 250 entries, all exactly 365
  days, worker loads 150 and 100 entries;
- final state: both workers and the aggregate completed with exit code zero;
- worker elapsed times: 4:47:03 and 3:32:02; peak RSS about 22.4 and 20.8 GiB;
- aggregate elapsed time: 1:10;
- accepted asset: five complete 1961-2010 chains, 250 shards, 91,245
  transitions, one Markov contract, and fixed 365-day noleap semantics;
- output footprint: about 1.2 GiB, including an 84 MiB legacy v3 manifest;
- requested worker limit: 8 hours; approved worst-case cost: about CNY 1.13.

The accepted jobs emitted worker-v2 and dataset-v3 manifests from their fixed
snapshot. The branch now also contains a streamed canonical `B_fast` trainer
and compact worker-v3/dataset-v4 manifests. The new reader and contract loader
explicitly support both dataset v3 and v4. However, the scientific v4 target
changed after the first neural experiment exposed sentinel and finalize-state
contract defects. The v3 five-point run remains valid Teacher/provenance
evidence but is not a valid input to the revised trainer.

The historical five-point run was monitored from `cln01`; no computation ran
on the login node:

```bash
/rmprog/slurm/v22.05.7/bin/squeue -j 14363439,14363441 \
  -o '%i %P %j %T %M %l %R'
```

Project runtime files must remain under:

```text
/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime
```

The two worker logs are expected at
`runtime/logs/teacher_14363439_0.txt` and
`runtime/logs/teacher_14363439_1.txt`. The aggregate log is expected at
`runtime/logs/teacher_aggregate_14363441.txt`. Read the `plan=` line in a
worker log for the exact generated plan path instead of guessing it.

## Completion Gate

This gate passed on 2026-07-23. Slurm `COMPLETED` alone was not used as
acceptance: aggregate job `14363441` reopened every worker manifest and shard,
verified hashes and schema consistency, rejected missing or duplicate entries,
and emitted the accepted dataset manifest.

After the jobs finish:

```bash
SLURM=/rmprog/slurm/v22.05.7/bin
$SLURM/sacct -j 14363439,14363441 \
  --format=JobID,State,Elapsed,ExitCode,MaxRSS,NodeList
grep -h '^plan=' runtime/logs/teacher_14363439_*.txt
tail -n 80 runtime/logs/teacher_aggregate_14363441.txt
```

Then rerun the same aggregate validator explicitly if an independent check is
needed, using the exact plan path printed by the worker:

```bash
.venvs/orcjax_cpu/bin/python \
  -m research.daily_coarse_graining.teacher_shards aggregate \
  --plan "$TEACHER_PLAN" --worker-count 2
```

Acceptance requires all of the following:

- 5 complete landpoint chains and 250 unique shards;
- 91,245 transitions with the frozen spatial and temporal split;
- every NPZ, checkpoint, plan, contract, and source-asset hash valid;
- one Teacher commit and one `daily_teacher_markov_year_v3` schema throughout;
- no partial worker manifest, stale lock, duplicate entry, or missing year;
- recorded per-worker elapsed time and peak RSS for production planning.

If a worker times out, do not delete `generation.lock`. Confirm the owning job
is terminal and use the strict `teacher_shards recover-lock` workflow described
in [`teacher_dataset_generation.md`](teacher_dataset_generation.md), then
resume only the incomplete worker assignment.

## Next Single Milestone

1. run one bounded V100 curriculum with matching v5 one-step initialization
   and fixed-shape 1/3/7-day batches through the retained tail;
2. run a validation-only seven-day free rollout from the resulting checkpoint;
3. require both one-step `B_fast` and canonical next-state supervision;
4. require seven-day free-rollout final RMSE near the teacher-forced envelope
   (provisional gate `<=0.29`) with zero mask/discrete mismatch;
5. only then decide whether more Teacher landpoints or a revised network are
   justified.

Do not start all 669 x 50 years before this bounded learnability and rollout
gate. The ten-point dataset tests architecture development; it cannot by
itself establish global spatial generalization.

The prepared v4 subset is
[`../../../manifests/coarse_graining/daily_teacher_initial_10point_1961_2010_v4.json`](../../../manifests/coarse_graining/daily_teacher_initial_10point_1961_2010_v4.json).
It preserves the frozen 669-point parent split and contains six train, two
validation, and two test landpoints. The original five are retained; the five
additions are `069.0-119.0`, `281.0-095.0`, `283.0-091.0`, `295.0-113.0`, and
`333.0-057.0`, selected to add dry/low-productivity, high-productivity,
high-maintenance, low-Vcmax/short-residence, and geographic diversity.

The v4 production run was submitted on 2026-07-24 from fixed Teacher commit
`1f19ed7`:

- worker array: `14365952`, four one-CPU tasks on `cnall`, array concurrency 4;
- aggregate: `14365958` on `cnmix`, dependency `afterok:14365952`;
- plan: `runtime/plans/teacher_initial_10point_v4_1f19ed7.json`;
- plan SHA256: `b32eef75aafc6793c886bcec5858b4f0b343b8e510d12fb744518d35030d84af`;
- output: `runtime/outputs/training/pft14-daily-teacher-initial-10point-1961-2010-v4-1f19ed7`;
- worker loads: 150, 150, 100, and 100 landpoint-years;
- requested worker wall time: 7 hours; aggregate wall time: 20 minutes;
- worst-case charge: approximately CNY 1.98, with no GPU allocation.

Do not submit replacement jobs while this array is active. Completion requires
the aggregate manifest plus the v4 target-representation audit; Slurm
`COMPLETED` alone is insufficient.

The Explore1000 GPU runtime prerequisite is complete at commit `c566538`:

- project environment: `.venvs/orcjax_gpu`;
- JAX/JAXLIB/CUDA plugin: 0.4.38 with the pinned CUDA 12.1 compatibility set;
- real `gln01` V100 smoke: backend `gpu`, device `cuda:0`, finite JIT loss and
  gradient;
- formal entry points: `scripts.hpc.verify_orcjax_gpu` and
  `scripts.hpc.run_canonical_gpu_train`;
- train-only statistics job `14365346` completed in 68 seconds on one `cnall`
  CPU with exit code zero;
- obsolete v1 statistics: `runtime/outputs/training/canonical-initial-5point-1961-2010/training_statistics.json`;
- statistics JSON SHA256:
  `1041c1231fb6c9bb5905e0a9403aefc03666ba745390539cf0ec644531be6885`;
- statistics NPZ SHA256:
  `584cee4ffbf1d05e79f3e66b3a7998d9d94046539eb8583e3332a5be16f5773d`;
- consumer-level validation passed for dataset identity, contract identity,
  NPZ hash, finite means/variances/scales, positive scales, 32,118 train/train
  samples, and 88 source shards;
- first GPU training job `14365446` completed in 5:40 with about 3.1 GiB peak
  device memory, but is rejected for promotion: contract v3 included finalize
  mirrors and statistics v1 treated ORCHIDEE sentinels as finite data;
- its checkpoint and statistics must not be resumed or reused with v4.

GPU training now requires the matching passed `acceptance_report.json`; the
launcher refuses a missing or hash-mismatched report. Do not bypass the GPU
launcher with a direct import of `canonical_training_run`: in this container,
JAX plugin auto-discovery can otherwise initialize CPU first. The next GPU
operation comes only after the v4 ten-point dataset and v2 statistics are
generated and accepted.

## Decisions Not To Reopen

- Do not redefine the label as the final daily state. The accepted target is
  the complete `B_fast` boundary, while `S[d+1]` remains rollout supervision.
- Do not store all 48 interpolated forcing steps. Store native 6-hour forcing
  and reconstruct deterministic preprocessing.
- Do not invent a cold-start `S[0]`; bootstrap canonical `S[1]` with Teacher
  Day 1.
- Do not generate v1 or v2 shards. They are historical evidence only.
- Do not use landpoint ID as a neural feature or leak validation/test samples
  into normalization statistics.
- Do not assume one-landpoint Teacher generation benefits from a GPU. Current
  production uses persistent CPU workers; GPU is reserved for batched neural
  training unless a future workload-level A/B proves otherwise.
- Do not run one process per year or landpoint. Same-process executable reuse
  is the accepted performance mechanism.
- Do not restart PFT14 source-branch equivalence work merely because a neural
  experiment fails. First distinguish Teacher-data integrity, supervised
  optimization, and autoregressive rollout error.

## Authority Map

| Question | Authoritative file |
| --- | --- |
| Overall Teacher/release status | [`../../current-status.md`](../../current-status.md) |
| Current operational task | this handoff page |
| Daily state and target semantics | [`daily_markov_contract_v5.md`](daily_markov_contract_v5.md) |
| Dataset production and recovery | [`teacher_dataset_generation.md`](teacher_dataset_generation.md) |
| Research quality gates | [`development_standard.md`](development_standard.md) |
| Frozen five-point batch | [`../../../manifests/coarse_graining/daily_teacher_initial_5point_1961_2010.json`](../../../manifests/coarse_graining/daily_teacher_initial_5point_1961_2010.json) |
| Frozen ten-point v4 batch | [`../../../manifests/coarse_graining/daily_teacher_initial_10point_1961_2010_v4.json`](../../../manifests/coarse_graining/daily_teacher_initial_10point_1961_2010_v4.json) |
| Frozen 669-point population/splits | [`../../../manifests/coarse_graining/daily_teacher_production_669.json`](../../../manifests/coarse_graining/daily_teacher_production_669.json) |
| Explore1000 operations | [`../../deployment-explore1000.md`](../../deployment-explore1000.md) |

Dated reports in this directory preserve evidence under their original
conditions. They are not current roadmaps and must not override this page or
`docs/current-status.md`.
