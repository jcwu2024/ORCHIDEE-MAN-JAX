# Teacher dataset generation

`research.daily_coarse_graining.teacher_shards` is the production-oriented
capture entry point for daily coarse-graining research. It does not change the
Teacher model. It turns complete compiled-day boundaries into restartable
landpoint-year shards.

## Contract

- The input JSON plan freezes both spatial and temporal splits before capture.
- One landpoint always belongs to one spatial split; one year always belongs to
  one temporal split. The validator rejects leakage.
- Worker assignment deterministically balances complete landpoint chains, so
  all years of one landpoint stay in one persistent process while worker loads
  remain as even as possible.
- Each landpoint-year is one atomic, uncompressed NPZ shard. There are no daily
  files and no per-array files.
- Each completed shard has SHA256 provenance, a year-end state checkpoint, the
  complete boundary schema, input hashes, and exact discrete state arrays.
- Interrupted workers skip only shards whose metadata, Teacher commit, plan
  hash, NPZ hash, and checkpoint hash all still match.
- Worker manifests are independent. Aggregation fails on missing, duplicate,
  corrupt, schema-drifting, or mixed-commit shards.
- Generated data remains `provisional_teacher` until the 669-landpoint Teacher
  acceptance gate is complete.
- A 1961 chain begins with `initialization_mode=cold_start_bootstrap`: the real
  Teacher executes Day 1 to create canonical `S[1]`, and capture starts at
  Day 2. Day 1 is never represented as a normal Markov sample with an invented
  state. Its year-end state is checked exactly against the staged accepted
  1961 checkpoint.
- Later first entries use `initialization_mode=year_start_checkpoint`; all
  following years consume the previous in-process checkpoint and preserve the
  normal restart-year Day 1 transition.

The example plan at
`manifests/daily_coarse_teacher_plan.example.json` documents the schema only.
Its asset paths must be replaced by the actual transferred state caches and
reference directories before generation.

## Commands

Validate a frozen plan without running the model:

```bash
python -m research.daily_coarse_graining.teacher_shards validate \
  --plan manifests/daily_coarse_teacher_plan.json \
  --worker-count 8
```

Run one persistent worker. Different worker indices may run concurrently; they
never write the same directory:

```bash
python -m research.daily_coarse_graining.teacher_shards generate \
  --plan manifests/daily_coarse_teacher_plan.json \
  --worker-index 0 \
  --worker-count 8
```

After every worker is complete, verify hashes and create the dataset manifest:

```bash
python -m research.daily_coarse_graining.teacher_shards aggregate \
  --plan manifests/daily_coarse_teacher_plan.json \
  --worker-count 8
```

For a bounded smoke run, `generate --max-entries 1` processes only the first
entry assigned to that worker. A partial worker manifest intentionally cannot
pass final aggregation.

## Stored arrays

The production schema is `daily_teacher_markov_year_v2`. Each shard stores:

- one canonical continuous state trajectory `state_trajectory[0:T+1]`;
- exact discrete trajectories under `state_discrete__*` with their original dtype;
- five native 6-hour source records per day in `forcing_native`, plus their
  cyclic source indices;
- run-def-controlled parameters and static landpoint conditions, without a
  landpoint-ID feature;
- one named annual-exogenous condition vector (currently atmospheric CO2) and
  a scalar source year;
- daily diagnostics `Y[d]` and one-based `day_index`.

The neural transition is therefore trained on
`S[d] + native_forcing[d] + P -> S[d+1] + Y[d]`. The shard does not store the
48-step interpolated forcing, separate day-start/day-end copies, or finite
masks. Interpolation, precipitation spreading, solar redistribution, unit
conversion, annual CO2, salinity and tide assembly remain deterministic
preprocessing. Finite masks are derived with `isfinite()` after loading.

For a cold-start 1961 shard, `day_index` is `2..365`, `transition_count` is
364, and `state_trajectory[0]` is canonical Day 1 end. For ordinary/restart
years, `day_index` begins at 1 and the shard has 365 or 366 transitions.

`research.daily_coarse_graining.markov_dataset` is the training-side reader.
It verifies dataset/shard hashes, enforces frozen spatial and temporal splits,
and collates numerical batches without exposing landpoint identity as a model
feature. The complete-day argument ownership ledger is
`manifests/coarse_graining/daily_markov_input_ownership_v2.json`.

`fit_training_statistics` streams only shards whose spatial and temporal
splits are both `train`. It computes finite-only count, mean, population
variance and scale per feature column without materializing the full dataset.
The hash-linked `daily_teacher_training_statistics_v1` JSON/NPZ asset records
the Teacher commit, Markov contract, source shard hashes and the number of
never, once and conditionally finite columns. Columns with zero finite values
use mean zero and scale one; columns with one finite value use that value and
scale one. `normalize_finite` maps undefined entries to normalized zero and
returns their explicit boolean mask. Validation and test shards must never
contribute normalization statistics.

The first real v2 smoke shard (`103.0-095.0`, 1962 Day 1) contains one
source-defined non-finite state column: the bare-soil/PFT1 slot of
`diffuco_previous_step_state.roughheight_pft`. CONDVEG intentionally assigns
roughness height only to vegetated PFT slots, and the existing source-backed
CONDVEG regression requires this slot to remain NaN. The PFT14 slot is finite
(`10.2` in this smoke). This slot must remain masked; it is not evidence of a
Teacher instability and must not be replaced with an invented physical value.

Training input prefetch is bounded by an explicit batch count. Benchmark it
against a verified local dataset without copying all samples into memory:

```bash
python -m research.daily_coarse_graining.benchmark_markov_dataset \
  outputs/training/local-markov-smoke-v2/dataset_manifest.json \
  --repeats 256 --batch-size 32 --prefetch 2
```

On the 1962 Day 1 Windows smoke, 256 synthetic replays produced 8 batches at
about 254 samples/s; the largest collated batch was 3.49 MB and the measured
`tracemalloc` peak was 9.89 MB. This is a loader-memory gate, not a model
training-throughput claim.

The schema and its field provenance are implemented in
`research/daily_coarse_graining/daily_markov_contract.py`. Existing v1 shards
remain historical audit evidence, but no new pilot may be generated with v1.

Normalization, finite masking, and train-time dtype conversion happen after
split selection. They are not baked into Teacher shards. Non-finite targets
are masked rather than silently sanitized into scientific values.

## Resource policy

The V100 single-landpoint benchmark showed insufficient GPU parallelism. Use
CPU persistent workers for Teacher generation and reserve GPUs for batched
neural-network training. A shared test node may validate only correctness and
compatibility; its wall time is not an accepted performance result. Benchmark
one worker on explicitly allocated Slurm compute resources, measure cold and
in-process hot time, CPU affinity, peak memory, and landpoint-year storage,
then present the requested cores, finite time limit, and worst-case charge for
approval before scaling. CPU generation uses seven-day compiled blocks, which
match the accepted complete-day Teacher path. The historical 28-day capture
result remains a GPU-specific throughput experiment rather than the CPU
production default.

## Explore1000 environments

Use project-scoped uv environments:

- CPU: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_cpu`
- GPU: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/orcjax_gpu`

Do not create new generic names such as `jc_gpu`. Historical `orcj_gpu` and
`orcj_gpu_compat` environments are legacy validation assets under
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/.venvs/legacy/`; keep them read-only
until the canonical `orcjax_gpu` environment reproduces their accepted
results.

Explore1000 CPU nodes run CentOS 7. The general project `uv.lock` may resolve
new wheels that require GLIBC 2.27, so CPU Teacher generation uses the pinned
compatibility profile in `scripts/hpc/requirements-orcjax-cpu.txt`. Create or
reconcile it with:

```bash
bash scripts/hpc/bootstrap_orcjax_cpu.sh
```

Run the bootstrap command on `cln01`, which has package-index access. CPU
compute and test nodes have no DNS access; they use the resulting shared
environment read-only rather than resolving dependencies themselves.

This remains uv-managed: Conda is not used to resolve or install project
packages. The profile pins JAX 0.4.38 to match the accepted GPU compatibility
runtime and pins manylinux2014-compatible numerical wheels for CPU nodes.
