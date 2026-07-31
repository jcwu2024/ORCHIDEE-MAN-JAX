# Bounded OK_LEAK Auxiliary-Capture Selection Protocol

Date: 2026-07-31

## Purpose

Select a small but deliberately diverse set of real train-only days before
capturing the 13 half-hour driver series required by the exact 48-step
`OK_LEAK` scan. This protocol prevents an arbitrary one-point smoke from
becoming training evidence and prevents a premature 669-point recapture.

The selector is implemented, unit tested, and ready to run against the
existing server-side v5 dataset. The real 96-day plan has not yet been
materialized or frozen.

## Admitted Source

- dataset: `pft14-daily-teacher-9point-1961-2010-v5-4c0f886`
- server root:
  `runtime/outputs/training/pft14-daily-teacher-9point-1961-2010-v5-4c0f886`
- admitted split: spatial `train` and temporal `train` only
- expected source scope: six landpoints, 264 landpoint-year shards, 96,354
  candidate days
- validation, spatial-test, temporal-test, and joint sealed-test shards are
  filtered from manifest metadata before any shard path is opened

Every admitted shard is verified against its manifest SHA256 before its
metrics are read. The output plan binds the dataset manifest, Markov contract,
Teacher commit, every source shard, and every selected day-start state and
fast-day target by SHA256.

## Metrics

For each candidate day the selector reads seven source-backed condition
metrics:

1. `soil_wetness`: mean day-start `hydrol.soil_mc`;
2. `soil_temperature_k`: mean Teacher `daily_interface.tsoil_daily`;
3. `pft14_gpp_daily`: Teacher daily GPP for PFT14 (zero-based source index
   13);
4. `soil_carbon_proxy`: sum of day-start `stomate.soilc_total`, used only for
   diversity ranking and not claimed as the full scientific carbon inventory;
5. `precip_daily`: Teacher daily precipitation;
6. `hydrologic_export`: endpoint runoff plus drainage across soil tiles;
7. `peat_fraction`: day-start `stomate.fpeat`.

Nonfinite values and ORCHIDEE `+/-1e20` sentinels are excluded. A candidate
set with an invalid metric row is rejected rather than silently imputed.

## Deterministic Selection

Select exactly 16 days from each of the six train landpoints, for 96 days in
total:

1. retain the earliest available train day, normally 1961 Day 2 after the
   canonical cold-start day;
2. retain the minimum and maximum of each of the six dynamic metrics within
   that landpoint;
3. merge duplicate anchors and fill the remaining slots by deterministic
   farthest-point selection in per-landpoint metric-rank plus time-rank space;
4. break all ties by landpoint, year, and day order.

This gives every train landpoint equal representation, preserves cold-start
and named process extremes, and adds multidimensional interior coverage. Peat
fraction is a condition metric and is covered across landpoints; it is not
used as a within-landpoint dynamic axis because it is normally static.

At the measured 486,912 uncompressed bytes per selected day, 96 captures
require 46,743,552 bytes, or 44.58 MiB, before NPZ compression. This is small
enough for a bounded architecture gate and does not justify all-day capture.

## Command

Run from the repository root where the server-side v5 shards exist:

```bash
PYTHONPATH="$PWD" \
python -m scripts.dev.plan_ok_leak_driver_capture \
  --dataset-manifest runtime/outputs/training/pft14-daily-teacher-9point-1961-2010-v5-4c0f886/dataset_manifest.json \
  --dataset-root runtime/outputs/training/pft14-daily-teacher-9point-1961-2010-v5-4c0f886 \
  --samples-per-landpoint 16 \
  --output runtime/plans/ok_leak_auxiliary_capture_96day.json
```

Do not capture drivers until the generated plan is reviewed for all expected
landpoints, anchor reasons, year/day ranges, source hashes, selected count,
and `sealed_test_used: false`, then frozen by its `plan_sha256`.

## Next Implementation Boundary

After the real plan is accepted, add one batch capture runner which reuses
compiled Teacher executables across selected days and writes one atomic,
restartable asset per planned day. It must invoke the existing capture hook;
it must not duplicate formulas or launch one Python/compiler process per day.
Only after those assets pass exact replay may the conservation, feasible
perturbation, reverse-gradient, and tiny parent-no-regression gates proceed.
