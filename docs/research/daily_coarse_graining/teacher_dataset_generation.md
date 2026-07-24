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
- Each landpoint-year is one atomic, losslessly compressed NPZ shard. There are no daily
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
  state. Its year-end state is checked against the staged current 1961
  checkpoint with exact structure/discretes and explicit float64
  `rtol=1e-12`, `atol=1e-12` diagnostics.
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

## Production planning

`research.daily_coarse_graining.teacher_production` converts the complete paper
landpoint inventory into a portable, frozen production specification. Spatial
holdouts are selected by deterministic maximin coverage of location and the
four calibrated PFT14 parameters; archived model outputs are deliberately not
used to choose the split. The explicit assignments are stored in the spec and
never recomputed while generating data.

Unlike the stricter 12-point acceptance pilot, production does not require a
prebuilt 1961 year-end checkpoint for every landpoint. Every chain runs the
source-backed 1961 Day 1 cold start, stores canonical `S[1]`, and begins normal
training transitions at Day 2. An external checkpoint is optional acceptance
evidence, not an input dependency.

Freeze the 669-point, 1961--2010 specification once:

```bash
python -m research.daily_coarse_graining.teacher_production freeze \
  --population-manifest manifests/landpoints_669.json \
  --output manifests/coarse_graining/daily_teacher_production_669.json \
  --dataset-id pft14-daily-teacher-669-1961-2010
```

Use `teacher_production subset` to create bounded admission or training batches
from that parent spec. A subset inherits the exact parent spatial assignments
and temporal ranges and records the canonical parent-spec hash; it must not
recompute splits on the smaller population. The first complete-trajectory
batch is frozen in
`manifests/coarse_graining/daily_teacher_initial_5point_1961_2010.json`: two
train landpoints, two validation landpoints, one test landpoint, 50 years per
point, and 250 atomic point-year shards. It is an architecture-development
dataset, not the final spatial-generalization claim.

Inventory and stage only the six small source-truth files needed by each
landpoint, then build the executable plan on the target machine so all runtime
paths are native to that machine:

```bash
python -m research.daily_coarse_graining.teacher_production inventory \
  --spec manifests/coarse_graining/daily_teacher_production_669.json \
  --reference-root reference
python -m research.daily_coarse_graining.teacher_production stage \
  --spec manifests/coarse_graining/daily_teacher_production_669.json \
  --reference-root reference --destination /absolute/runtime/assets/teacher-669
python -m research.daily_coarse_graining.teacher_production plan \
  --spec manifests/coarse_graining/daily_teacher_production_669.json \
  --asset-root /absolute/runtime/assets/teacher-669 \
  --teacher-config configs/orchidee_man_250919.yaml \
  --output-root /absolute/runtime/outputs/training/pft14-teacher-669 \
  --plan-path /absolute/runtime/plans/pft14-teacher-669.json
```

The plan is consumed by `scripts/hpc/slurm_teacher_production_worker.sh` as a
bounded Slurm array. A worker owns complete landpoint chains and remains alive
across landpoints so compatible JAX executables can be reused. Submit
`scripts/hpc/slurm_teacher_production_aggregate.sh` with an `afterok`
dependency on the complete array. The worker count and array concurrency must
be selected from an allocated-node peak-RSS and cross-landpoint compile-reuse
probe, not guessed from logical core count.

If Slurm kills a worker at its wall-time limit, the exclusive lock may remain
because the process cannot execute its cleanup handler. Confirm that the owner
job is terminal, read the recorded host/PID/job identity, and use the explicit
`teacher_shards recover-lock` command before resubmission. Recovery requires an
exact owner match and preserves the stale lock as an audit file; never delete
or overwrite `generation.lock` blindly.

## Stored arrays

The production schema is `daily_teacher_markov_year_v4`. Each shard stores:

- one canonical continuous state trajectory `state_trajectory[0:T+1]`;
- one compact `fast_day_target[d]` containing the complete output of the
  replaced 48-step SECHIBA/accumulation/maintenance/OK_LEAK block;
- exact discrete trajectories under `state_discrete__*` with their original dtype;
- five native 6-hour source records per day in `forcing_native`, plus their
  cyclic source indices;
- run-def-controlled parameters and static landpoint conditions, without a
  landpoint-ID feature;
- one named annual-exogenous condition vector (currently atmospheric CO2) and
  a scalar source year;
- daily diagnostics `Y[d]` and one-based `day_index`.

The neural operator is therefore trained on
`S[d] + native_forcing[d] + P -> B_fast[d]`. The retained source-backed daily
season/STOMATE tail consumes `B_fast[d]` and writes `S[d+1]`; the stored state
trajectory provides rollout supervision and continuity checks. The shard does not store the
48-step interpolated forcing, separate day-start/day-end copies, or finite
masks. Interpolation, precipitation spreading, solar redistribution, unit
conversion, annual CO2, salinity and tide assembly remain deterministic
preprocessing. Validity masks are derived after loading. Values are valid only
when finite and `abs(value) < 5e19`; this excludes the ORCHIDEE `+/-1e20`
undefined sentinels as well as NaN and infinity.

For a cold-start 1961 shard, `day_index` is `2..365`, `transition_count` is
364, and `state_trajectory[0]` is canonical Day 1 end. For ordinary/restart
years, `day_index` begins at 1 and the shard has exactly 365 transitions. The
paper forcing and Teacher lifecycle use a fixed 365-day noleap calendar;
Gregorian leap days must never be inserted.

`research.daily_coarse_graining.markov_dataset` is the training-side reader.
It verifies dataset/shard hashes, enforces frozen spatial and temporal splits,
and collates numerical batches without exposing landpoint identity as a model
feature. The complete-day argument ownership ledger is
`manifests/coarse_graining/daily_markov_input_ownership_v2.json`.

New production uses `daily_teacher_worker_manifest_v3` and
`daily_teacher_dataset_manifest_v4`. The complete per-year metadata remains
next to its NPZ shard and is hash-addressed by the compact worker record. The
dataset manifest stores only shard/metadata/checkpoint paths and hashes,
frozen splits, checkpoint-chain links, and generation input hashes; the full
Markov contract is stored once at dataset level. This avoids copying roughly
0.3 MB of metadata into both aggregate layers for every landpoint-year. The
reader remains compatible with existing `daily_teacher_dataset_manifest_v3`
assets, including the active five-point noleap production run.

`fit_training_statistics` streams only shards whose spatial and temporal
splits are both `train`. It computes finite-only count, mean, population
variance and scale per feature column without materializing the full dataset.
The hash-linked `daily_teacher_training_statistics_v2` JSON/NPZ asset records
the Teacher commit, Markov contract, source shard hashes and the number of
never, once and conditionally finite columns. Columns with zero finite values
use mean zero and scale one; columns with one finite value use that value and
scale one. `normalize_finite` maps undefined entries to normalized zero and
returns their explicit boolean mask. Validation and test shards must never
contribute normalization statistics.

After `teacher_shards aggregate` succeeds, use the single acceptance command
instead of running statistics and representation checks independently:

```bash
python -m research.daily_coarse_graining.canonical_training_run \
  accept-dataset \
  --dataset "$DATASET_ROOT/dataset_manifest.json" \
  --output-dir "$ACCEPTANCE_ROOT"
```

This command requires a complete v4 aggregate, reopens and hashes every shard,
checks split coverage, audits the v4 persistence/sentinel target
representation, fits train/train-only v2 statistics, and executes a real-batch
finite forward/loss/gradient smoke. It writes `training_statistics.json`,
`training_statistics.npz`, and `acceptance_report.json`. The GPU training CLI
requires `--acceptance` and verifies that the report, dataset manifest, and
statistics still have the accepted hashes and identity.

The first real v2 smoke shard (`103.0-095.0`, 1962 Day 1) remains historical evidence and contains one
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
`research/daily_coarse_graining/daily_markov_contract.py`. Existing v1/v2 shards
remain historical audit evidence, but no new pilot may be generated with them.

The same contract now provides the inverse learned-output transformation:
`reconstruct_fast_day_target` inflates a compact `B_fast` vector into the full
SECHIBA state fields, daily interface, OK_LEAK updates, and final diagnostics
consumed by the retained daily tail. It restores active PFT1/PFT14 values from
the prediction while preserving source-defined inactive-PFT and discrete
state from the current-day template. A contract roundtrip test requires
`extract -> reconstruct -> extract` to be exact. Canonical validation reports
both family-balanced normalized errors and per-leaf physical-scale RMSE, MAE,
and maximum absolute error.

Normalization, defined-value masking, and train-time dtype conversion happen
after split selection. They are not baked into Teacher shards. NaN, infinity,
and ORCHIDEE `+/-1e20` sentinels are masked rather than silently sanitized into
scientific values.

## Resource policy

The first v3 compact-projection A/B on the local CPU used a cold-start Day 1
and ten Day 2+ transitions. The compact target width was 6,758. Daily state,
target, diagnostics, discrete state, and final state all passed the explicit
float64/exact comparison. First compilation plus capture took 146.94 seconds;
two same-process hot captures took 1.65 and 1.75 seconds, or about 0.17 seconds
per transition. Large production must therefore use persistent workers and
reuse the compiled seven-day executable across years and compatible
landpoints. A process-per-landpoint design that recompiles for every point is
not accepted.

The first complete local 1961 v3 shard contained 364 training transitions,
3,724 state columns, 6,758 fast-day target columns, and 90 diagnostic columns.
Its arrays occupied 30,969,096 bytes before compression. Deflate compression
reduced the NPZ from 30,973,822 to 3,505,248 bytes (11.3%) in 0.19 seconds, so
production shards use `np.savez_compressed` without changing float64 values.

The immediately following 1962 shard in the same Windows process took 33.15
seconds end to end, including 31.30 seconds of capture, and added no compiled
cache entries. This proves cross-year executable reuse for one landpoint. It
does not by itself prove cross-landpoint reuse or establish Linux peak memory;
those are explicit production admission gates.

The Explore1000 cross-landpoint admission probe at Teacher commit `adebf96`
used one persistent process, a fresh compilation cache, seven-day blocks, and
two eight-day 1961 cold-start chains (`001.0-071.0` and the explicit-snow
holdout `319.0-057.0`). On allocated node `ibc12b04n31`:

- the first point took 150.59 seconds to prepare, 1,256.11 seconds to compile
  and capture, and 1,406.85 seconds in total;
- the second point took 36.74 seconds to prepare, 3.35 seconds to capture, and
  40.15 seconds in total;
- the in-memory compiled cache remained exactly `later_day_block=1` and
  `sechiba_scan=1` across the second point, proving cross-landpoint executable
  reuse for equal-shape PFT14 chains;
- `/usr/bin/time -v` reported a 27,395,160 kB maximum RSS (about 26.1 GiB);
- the after-success aggregation verified both shards and emitted a complete
  dataset manifest.

The follow-up complete-year probe used the same cache and one 1961--1962 chain.
The cold 1961 entry took 2,395.32 seconds in total. The following in-process
1962 hot year took 102.30 seconds to capture and 139.20 seconds including shard
assembly, lossless compression, hashing, checkpointing, and shared-filesystem
IO. It added no in-memory compiled cache entries. The complete two-year
dataset aggregated successfully. Peak RSS rose to 30,852,280 kB (about 29.4
GiB), and a new Slurm process still incurred compilation despite pointing at
the prior persistent cache. Production may rely on same-process reuse across
years and landpoints, but not on cross-job cache reuse.

The first conservative production cap is therefore five concurrent workers in
total, not one worker per advertised CPU core. Five worst-observed processes
occupy about 147 GiB and leave useful headroom on a CPU node with approximately
187 GiB OS-visible memory even if Slurm places all five on one node. Raise this
cap only after a longer multi-point probe demonstrates a lower stable peak.
Worker count controls chain assignment; array concurrency must never exceed
the accepted memory cap.

Array workers use separate XLA cache subdirectories. Same-process executable
reuse is the accepted acceleration mechanism; workers must not contend for or
silently depend on a shared writable compilation-cache directory.

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
