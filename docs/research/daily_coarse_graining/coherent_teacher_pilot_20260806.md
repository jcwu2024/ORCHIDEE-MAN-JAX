# Coherent Teacher Six-Point Pilot Evidence

Status: **accepted bounded production evidence**.

Date: 2026-08-06.

## Identity

- Git commit: `2eb63842aefc5a685451e502815039bdc24b0ae1`.
- Teacher release: `pft14-daily-teacher-coherent-v1`.
- Release SHA256:
  `843d559b1dd1b5dbc494b22f478ded1b1cfc474716753f43021851c0428b6f7d`.
- Production plan SHA256:
  `8c6a4679d44b9f2430bef804ca3b7e5bac2af4916a7dd61d2defa4bf9aff90d1`.
- Slurm job: `14486923`, `cnall`, two nodes, six one-CPU workers.
- Dataset manifest SHA256:
  `a7e44cc536a1c1bae4f707a5097d379c8019fe1e1fbe146fad48084ef0042d88`.

The server evidence is under
`runtime/outputs/training/pft14-daily-teacher-coherent-v1-pilot/`; generated
data and logs are external runtime assets and are not committed to Git.

## Acceptance Result

All six workers exited zero. Unified aggregation and the formal coherent
reader reopened and hash-verified all six shards. The accepted inventory is:

- six landpoints and six complete 1961 point-years;
- 2,184 transitions, exactly 364 per landpoint;
- one Markov contract and one coherent typed contract;
- complete base shards, typed shards, exact masks, and year-end checkpoints;
- valid day inventories, serialization hashes, and checkpoint metadata;
- no source, release, plan, or contract drift.

This accepts the coherent packaging and production transaction. It does not
yet measure a later hot year or prove a real cross-year checkpoint chain,
because every pilot worker ran only a 1961 cold-start year.

## Resource Measurements

| Measurement | Observed value |
| --- | ---: |
| Slurm elapsed | `01:41:21` |
| Worker scientific generation | 2,505-2,721 s per point-year |
| Teacher capture | 2,184-2,306 s per point-year |
| Maximum process RSS | 33,694,593,024 bytes (31.4 GiB) |
| Base shards | 18,661,790 bytes |
| Typed shards | 8,280,262 bytes |
| Year-end checkpoints | 1,279,542 bytes |
| Total core assets | 28,221,594 bytes |

The workers received only 39-61% effective CPU time on the selected nodes, so
their wall times include substantial node contention. The allocation cost was
approximately CNY 0.71; the approved two-hour ceiling was CNY 0.84.

At the observed cold-year compression rate, 669 landpoints times 50 years is
approximately 146.5 GiB for base, typed, and checkpoint assets. This is a
provisional storage estimate, not a production resource request: later years
may compress differently, and cold-start compile cost must not be multiplied
as if every year used a fresh process.

## Next Bounded Gate

The pilot accepted the co-generation transaction under the predecessor v1
release and v5 manifest packaging. The formal product was then frozen as
release `pft14-daily-teacher-v6`, manifest v6, and typed contract v3 without
changing the Teacher tree, Markov contract, label fields, or numerical arrays.

Do not create a disposable two-year calibration. Launch the first resumable
worker batch from the formal 669-landpoint, 1961-2010 v6 plan. Its acceptance
requires:

1. 1962 starts from the exact 1961 checkpoint and both entries aggregate;
2. compilation remains in the same process and the 1962 hot-year timing is
   reported separately from 1961 cold start;
3. base, typed, checkpoint, RSS, and wall-time measurements are recorded;
4. the formal coherent reader verifies the first cross-year segment and the
   complete retained landpoint chain.

Those rows are part of the formal dataset. Later batches resume the same plan
and release rather than regenerating or relabeling them.
