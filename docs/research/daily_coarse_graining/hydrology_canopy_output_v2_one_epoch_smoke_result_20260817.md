# Hydrology And Canopy Output v2 One-epoch Smoke Result

Date: 2026-08-17.

Status: accepted. The formal 100-epoch one-point memorization gate is prepared
but requires explicit paid-job approval.

## Execution

The free `gln01` smoke ran at commit
`ba7f5d96bdcef161ff4c26d64e44a9272389af26`. It used all 364 valid train/train
days from `001.0-071.0`, 1961, exactly once. Batch size 8 produced 46 finite
AdamW updates. No checkpoint was written.

The report is
`outputs/research/daily-coarse-graining/mechanistic_output_v2_one_epoch_training_smoke_20260817.json`.
Its SHA256 is
`61de8c4c8df98c0da4e52bfda1074b4c4a6bfd0e9b3ddb93280bb9a82a48b5c0`.

## Result

The smoke passes its preregistered execution gate. Complete process loss falls
from `4.5707906` to `4.2333845`, a `7.3818%` reduction. All seven experts
improve: canopy GPP `23.21%`, daily interface/finalize `0.181%`, hydrology/snow
`15.69%`, litter/POC/DOC `27.86%`, maintenance respiration `55.89%`, soil
thermal `38.49%`, and surface energy `19.40%`.

GPP, maintenance, and NPP normalized RMSE improve from `0.3963`, `0.6725`, and
`1.0865` to `0.1390`, `0.2301`, and `0.1253`. Final AGB and BGB tendency NRMSE
are `5.55e-5` and `1.12e-4`. Every prediction, defined next-state, water/carbon
budget, and nonnegative-inventory hard count is zero. Initial/final loss masks
and counts are identical.

Gradient norms remain finite across all updates: minimum `1.4241`, median
`5.0313`, and maximum `35.8052`. This does not show the extreme early gradient
spike observed in the v1 attribution reproduction.

Elapsed time is `1,279.73 s`. The first `jit_train_step` compilation takes
`215.79 s`; the complete epoch record takes `441.17 s`. JAX reports 8.012 GB
peak bytes in use and a 25.560 GB allocator limit.

## Formal One-point Gate

The next gate uses the same candidate, seed, all 364 train/train days, batch
size 8, unclipped AdamW schedule, and no checkpoint for 100 epochs and 4,600
updates. It has no matched control. Acceptance requires:

- at least `99%` complete process-loss reduction;
- at least `95%` reduction for each of the seven independent experts;
- GPP, maintenance, NPP, AGB tendency, and BGB tendency NRMSE no larger than
  `0.05`;
- zero prediction/state/budget/inventory hard counts and all finite gradients.

The paid envelope is one `gnall` node, four allocated CPU cores, one V100, and
a two-hour wall limit. Worst-case additive charge is CNY `0.56` CPU plus CNY
`4.40` GPU, totaling CNY `4.96`. The launcher is
`scripts/hpc/slurm_mechanistic_output_v2_one_point.sh` and reuses the accepted
smoke compilation cache.

A pass opens design of the 64-point-year gate. It does not directly open
full-v6 training, 7/30/365-day rollout, Gate D2, sealed-test, or product
acceptance.
