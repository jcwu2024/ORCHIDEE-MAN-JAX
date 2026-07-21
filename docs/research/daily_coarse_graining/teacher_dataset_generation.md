# Teacher dataset generation

`research.daily_coarse_graining.teacher_shards` is the production-oriented
capture entry point for daily coarse-graining research. It does not change the
Teacher model. It turns complete compiled-day boundaries into restartable
landpoint-year shards.

## Contract

- The input JSON plan freezes both spatial and temporal splits before capture.
- One landpoint always belongs to one spatial split; one year always belongs to
  one temporal split. The validator rejects leakage.
- Worker assignment hashes the landpoint ID, so all years of one landpoint stay
  in one persistent process and can reuse compiled executables and year-end
  state.
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

Each shard stores:

- raw float64 day-start state and its finite mask;
- all 48 half-hour forcing rows and their finite mask;
- run-def-controlled parameter conditioning;
- static landpoint physics without a landpoint-ID feature;
- calendar conditioning;
- the complete Teacher daily boundary target and finite mask;
- all discrete state arrays with their original integer/logical dtype.

Normalization, target sanitization, and train-time dtype conversion happen
after split selection. They are not baked into Teacher shards.

## Resource policy

The V100 single-landpoint benchmark showed insufficient GPU parallelism. Use
CPU persistent workers for Teacher generation and reserve GPUs for batched
neural-network training. Before a paid Explore1000 submission, benchmark one
worker on a test node, measure peak memory and landpoint-year wall time, then
present the requested cores, finite time limit, and worst-case charge for
approval.

## Explore1000 environments

Use project-scoped uv environments:

- CPU: `/WORK/liwei_work/jcwu/.venvs/orcjax_cpu`
- GPU: `/WORK/liwei_work/jcwu/.venvs/orcjax_gpu`

Do not create new generic names such as `jc_gpu`. Historical `jc_gpu`,
`orcj_gpu`, and `orcj_gpu_compat` environments are legacy validation assets;
keep them read-only until the canonical `orcjax_gpu` environment reproduces
their accepted results.

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
