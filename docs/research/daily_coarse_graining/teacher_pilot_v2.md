# Daily Teacher Pilot v2

## Purpose

This pilot tests whether a daily neural transition can learn complete PFT14
trajectories without committing to all 669 landpoints. It is not a Teacher
equivalence gate and it does not change Teacher science.

The frozen specification is
`manifests/coarse_graining/daily_teacher_pilot_v2.json`:

- 12 representative landpoints;
- 8 spatial train, 2 spatial validation, and 2 spatial test landpoints;
- complete 1962-2010 trajectories, starting from each point's accepted 1961
  year-end checkpoint;
- 1962-2004 train, 2005-2007 temporal validation, and 2008-2010 temporal test;
- 588 point-years and 214,764 daily transitions in total;
- 125,648 samples in the strict spatial-train/temporal-train intersection.

The dry low-productivity point `069.0-119.0` and explicit-snow point
`319.0-057.0` are spatial test holdouts. Their samples cannot contribute
normalization, architecture selection, or model fitting.

## Bounded Generation

Production generation consumes one compiled block at a time. Heavy Teacher
day records are converted immediately to the 3,724-column Markov state,
discrete state, and 90 diagnostics, then released. The retained annual arrays
are about 11.3 MB per point-year before NPZ compression. The full pilot is
therefore about 6.6 GB uncompressed, not the old multi-terabyte v1 estimate.

Before generating the pilot, run exactly one 001/1962 resource probe with
`scripts/hpc/slurm_teacher_v2_resource_probe.sh`. It measures:

- cold compile plus 365-day capture time;
- process peak RSS after block-streaming conversion;
- final compressed and uncompressed shard sizes;
- checkpoint size and hash-valid aggregation.

Do not extrapolate production resources from the old v1 annual capture. That
path retained all full day records and reached about 26.85 GB peak RSS.

## Minimal Assets

Only the four start/restart NetCDF files, `used_run.def`, and the accepted 1961
JAX checkpoint are needed per point. The local inventory passed for all 12
points and totals 14,598,446 bytes.

Create the portable package locally:

```powershell
conda run -n ORCJAX python -m research.daily_coarse_graining.teacher_pilot `
  --spec manifests/coarse_graining/daily_teacher_pilot_v2.json stage `
  --reference-root . `
  --checkpoint-root outputs/acceptance/compiled_1961_12point_wetdiaglong_fix `
  --destination outputs/transfer/daily_teacher_pilot_v2
```

After transfer, verify it on Explore1000 without running the model:

```bash
cd /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX
.venvs/orcjax_cpu/bin/python -m research.daily_coarse_graining.teacher_pilot \
  --spec manifests/coarse_graining/daily_teacher_pilot_v2.json verify \
  --asset-root runtime/assets/daily_teacher_pilot_v2
```

Build the final generation plan only after the resource probe passes:

```bash
.venvs/orcjax_cpu/bin/python -m research.daily_coarse_graining.teacher_pilot \
  --spec manifests/coarse_graining/daily_teacher_pilot_v2.json plan \
  --asset-root runtime/assets/daily_teacher_pilot_v2 \
  --teacher-config configs/orchidee_man_250919.yaml \
  --output-root runtime/outputs/training/pft14-daily-teacher-pilot-v2 \
  --plan-path runtime/outputs/training/pft14-daily-teacher-pilot-v2/generation_plan.json
```

## Promotion Gates

1. The one-point resource probe must finish within its finite allocation, use
   `bounded_compiled_blocks`, aggregate successfully, and materially reduce
   the old annual peak RSS.
2. Resource count, wall time, worker count, expected storage, and worst-case
   charge are recalculated from that result before any multi-point `sbatch`.
3. Pilot generation must preserve one persistent process per landpoint chain,
   assign complete landpoint chains deterministically with balanced worker
   loads, and preserve exact source/contract/checkpoint hashes and frozen
   splits.
4. Train-only statistics are generated from spatial-train/temporal-train
   shards. No validation or test data may influence normalization.
5. One-step, 7-day, 30-day, and 365-day free rollout gates run before any
   larger Teacher dataset or 50-year surrogate claim.

The 12-point pilot can be stopped after any failed gate. It is not an
automatic authorization to generate all 588 point-years.
