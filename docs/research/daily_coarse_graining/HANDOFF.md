# Daily Coarse-Graining Handoff

Snapshot date: 2026-07-23

This is the single operational handoff page for the daily coarse-graining
research branch. Read this page before dated experiment reports. Stable
scientific and release facts remain authoritative in
[`../../current-status.md`](../../current-status.md).

## Five-Minute Startup

1. Check out `research/daily-coarse-graining` and confirm a clean worktree.
2. Read this page, then
   [`daily_markov_contract_v3.md`](daily_markov_contract_v3.md) and
   [`teacher_dataset_generation.md`](teacher_dataset_generation.md).
3. Check the active Explore1000 jobs listed below. Do not submit a replacement
   while the current worker array is running.
4. When aggregation succeeds, validate the resulting dataset before changing
   the neural model or launching more Teacher generation.

```bash
git status --short --branch
git log -1 --oneline
```

The implementation snapshot from which the current production jobs were
submitted is commit `23046a3`. Documentation-only commits after that snapshot
do not invalidate generated Teacher data; any model, contract, capture, or
shard-code change does.

## Current Architecture

The learned operator replaces exactly this block:

```text
48 half-hour SECHIBA/HYDROL/THERMOSOIL/DIFFUCO/ENERBIL transitions
+ daily accumulation and maintenance
+ 48 ordered OK_LEAK transitions
```

Its contract is:

```text
S[d] (3,724 values) + native 6-hour forcing[d] + parameters/static conditions
  -> B_fast[d] (6,758 values)
B_fast[d] + S[d]
  -> retained source-backed daily season/STOMATE
  -> S[d+1] (3,724 values)
```

`B_fast` is a complete same-day interface, not persistent state. `S` is the
cross-day Markov state. For a 1961 cold start, the real Teacher executes Day 1
to construct canonical `S[1]`; the first training sample is Day 2. The shard
schema is `daily_teacher_markov_year_v3` and uses lossless NPZ compression.

The intended final repository has independent runtime choices for execution
backend (`cpu` or `gpu`) and transition implementation
(`teacher_half_hour` or, after acceptance, `neural_daily`). The research
branch is temporary development history, not a separate product.

## Current Production Run

The bounded architecture-development dataset is frozen by
[`../../../manifests/coarse_graining/daily_teacher_initial_5point_1961_2010.json`](../../../manifests/coarse_graining/daily_teacher_initial_5point_1961_2010.json):

- 5 landpoints: 2 train, 2 validation, and 1 test;
- 1961-2010 complete trajectories;
- 250 atomic landpoint-year shards;
- 91,305 daily transitions after excluding cold-start Day 1 at each point.

Explore1000 jobs submitted from commit `23046a3`:

- worker array: `14362790`, two persistent CPU workers;
- after-success aggregate: `14362791`;
- last observed state: both workers running; aggregate waiting on dependency;
- requested worker limit: 8 hours; approved worst-case cost: about CNY 1.13.

Monitor from `cln01`; never compute on the login node:

```bash
/rmprog/slurm/v22.05.7/bin/squeue -j 14362790,14362791 \
  -o '%i %P %j %T %M %l %R'
```

Project runtime files must remain under:

```text
/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime
```

The two worker logs are expected at
`runtime/logs/teacher_14362790_0.txt` and
`runtime/logs/teacher_14362790_1.txt`. The aggregate log is expected at
`runtime/logs/teacher_aggregate_14362791.txt`. Read the `plan=` line in a
worker log for the exact generated plan path instead of guessing it.

## Completion Gate

Do not treat Slurm `COMPLETED` alone as dataset acceptance. The aggregate job
must finish successfully; it reopens every worker manifest and shard, verifies
hashes and schema consistency, rejects missing or duplicate entries, and emits
the dataset manifest.

After the jobs finish:

```bash
SLURM=/rmprog/slurm/v22.05.7/bin
$SLURM/sacct -j 14362790,14362791 \
  --format=JobID,State,Elapsed,ExitCode,MaxRSS,NodeList
grep -h '^plan=' runtime/logs/teacher_14362790_*.txt
tail -n 80 runtime/logs/teacher_aggregate_14362791.txt
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
- 91,305 transitions with the frozen spatial and temporal split;
- every NPZ, checkpoint, plan, contract, and source-asset hash valid;
- one Teacher commit and one `daily_teacher_markov_year_v3` schema throughout;
- no partial worker manifest, stale lock, duplicate entry, or missing year;
- recorded per-worker elapsed time and peak RSS for production planning.

If a worker times out, do not delete `generation.lock`. Confirm the owning job
is terminal and use the strict `teacher_shards recover-lock` workflow described
in [`teacher_dataset_generation.md`](teacher_dataset_generation.md), then
resume only the incomplete worker assignment.

## Next Single Milestone

After the five-point dataset passes the completion gate:

1. fit train-only normalization statistics with the hash-verifying v3 reader;
2. train the first parameter-conditioned network to predict `B_fast`;
3. evaluate train/validation/test one-step errors by field group;
4. run 7-day free rollout through the retained daily season/STOMATE tail;
5. only after those gates, decide whether to expand Teacher data or revise the
   network architecture.

Do not start all 669 x 50 years before this bounded learnability and rollout
gate. The five-point dataset tests architecture development; it cannot by
itself establish global spatial generalization.

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
| Daily state and target semantics | [`daily_markov_contract_v3.md`](daily_markov_contract_v3.md) |
| Dataset production and recovery | [`teacher_dataset_generation.md`](teacher_dataset_generation.md) |
| Research quality gates | [`development_standard.md`](development_standard.md) |
| Frozen five-point batch | [`../../../manifests/coarse_graining/daily_teacher_initial_5point_1961_2010.json`](../../../manifests/coarse_graining/daily_teacher_initial_5point_1961_2010.json) |
| Frozen 669-point population/splits | [`../../../manifests/coarse_graining/daily_teacher_production_669.json`](../../../manifests/coarse_graining/daily_teacher_production_669.json) |
| Explore1000 operations | [`../../deployment-explore1000.md`](../../deployment-explore1000.md) |

Dated reports in this directory preserve evidence under their original
conditions. They are not current roadmaps and must not override this page or
`docs/current-status.md`.
