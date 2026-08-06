# Teacher v6 Reuse Benchmark

Status: **accepted for production sizing**.

## Run Identity

- Explore1000 job: `14490208`
- Git commit: `19087b171564f44b49eb96fa1217169deca5d57a`
- Node: `ibc12b03n21`
- Resources: one `cnall` node, five one-CPU workers
- Wall-time limit: 2 hours
- Actual elapsed: `00:52:49`
- Exit state: all five workers `COMPLETED`, exit code zero
- Plan SHA256: `aeda638cce2468533f819b5bb7ab189dbed4e07c82035e3a0fb44615801c0d68`

The plan contains ten representative landpoints and years 1961-1962. Each
worker processes two complete two-year landpoint chains in one Python process.
This separates first-process compilation, same-landpoint hot execution, and
execution after switching landpoints.

## Acceptance

The post-run progress audit reopened and hash-verified every generated asset.
It accepted `20/20` point-years and `5/5` complete workers, with no missing
entries, stale locks, or recovery candidates. Unified aggregation produced a
20-shard `daily_teacher_dataset_manifest_v6` dataset manifest.

## Timing Results

| Position in worker | Meaning | Mean capture | Mean total entry |
| --- | --- | ---: | ---: |
| first point, 1961 | cold process and compilation | 2057.60 s | 2148.84 s |
| first point, 1962 | same-point hot year | 82.73 s | 90.82 s |
| second point, 1961 | switched-point reuse | 91.07 s | 122.41 s |
| second point, 1962 | switched-point hot year | 81.97 s | 85.54 s |

The compiled-cache counts changed from zero to three entries during the first
cold year and remained exactly unchanged for every later entry, including
after switching landpoints. Landpoint changes therefore reuse the same
compiled executable in one worker. They do not require recompilation.

The representative steady hot capture rate is about 82 seconds per
point-year. A new landpoint adds about 30-37 seconds of one-time context and
state preparation before its first hot capture.

## Memory And Production Shape

`/usr/bin/time -v` reported 32,677,812-32,695,768 KiB peak RSS per worker,
about 31.2 GiB. Five staggered workers completed together on the approximately
187.6 GiB node without swapping or OOM. The measured five-worker aggregate
peak envelope is about 156 GiB, leaving about 31 GiB for the OS and variation.

The accepted production shape is therefore five workers per CPU node with a
180-second local startup stagger. Four workers remain a fallback for a node
with abnormal external load; they are not the default.

For the frozen 669-landpoint, 1961-2010 plan, 100 workers on 20 nodes imply
about 334-335 point-years per worker. Extrapolating the measured cold, point
switch, and hot timings gives about 858 allocated core-hours and 8.6-10 hours
of wall time. At CNY 0.07 per core-hour, expected CPU charge is about CNY 60.
A 12-hour allocation would cap the charge at CNY 84. This is a production
sizing estimate, not a substitute for final dataset aggregation and admission.

## Decision

Use one long-lived Python process per worker and assign whole landpoint chains
to it. Do not launch one fresh process per point-year or per landpoint. The
formal global job may proceed only against the frozen v6 plan and release and
must retain resumable worker manifests, checkpoint-chain validation, final
hash verification, aggregation, and dataset admission.
