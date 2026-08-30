# Structured-GPP Direct-Data Coverage Audit Result

Date: 2026-08-30

Status: accepted zero-target-read coverage audit; broad streaming throughput
is the next gate

Executable commit:
`db9763bda78a480f357762b6a3f69891fc7a8b5f`

Report SHA256:
`a53330a621cd63cd863798caea012a1517ef8552205839cb6f7f93aafff0528b`

## Execution

The audit completed on free `test01` in `69.93 s`. It loaded the accepted v6
index and rebuilt the exact 1,000-update schedule, warm-start range, and
materialized joint-training epoch schedule. It read no Teacher target values,
performed no optimizer update or checkpoint write, regenerated no data, and
used neither v5, validation, nor sealed-test data.

## Exact Exposure

The accepted train/train release has 8,591,565 transitions in 23,540
point-years, 535 landpoints, and 44 eligible years. Excluding the first
lifecycle row of each point-year leaves 8,568,025 eligible direct-training
days.

| Stage | Artifacts | Presentations | Unique days | Unique point-years |
| --- | ---: | ---: | ---: | ---: |
| Direct-GPP warm start | 250 | 52,000 days | 51,863 | 6,184 |
| Joint training | 250 unique / 992 presented | 206,336 days | 51,878 | 6,215 |
| Combined | 500 unique | 258,336 days | 103,399 | 10,761 |

Only 342 direct days overlap across the two disjoint artifact ranges. The
combined unique days are `1.203494%` of all accepted train/train transitions,
or `1.206801%` of first-row-eligible transitions. All 535 landpoints and all
44 years appear, so the weakness is not absence of global IDs; it is shallow
day-level exposure within a 12,879,363-parameter specialist.

The combined unique exposure by original objective source is 15,988 one-day
days, 27,966 seven-day-window days, and 59,728 thirty-day-window days. These
sets can overlap because the same day may be selected by different stages or
source types.

## Train-Only Monitor

A target-blind SHA256 ranking selected 128 point-years on 128 distinct
train/train landpoints that are completely disjoint from the 103,399 prior
exposure days and their 10,761 point-years. Sixteen monitor rows per point-year
give 2,048 fixed direct-GPP monitor days. Reserving each selected point-year in
full removes 46,589 eligible days (`0.543754%`) and leaves 8,521,436 eligible
days for future broad direct training.

The report records every point-year identity and row. This is a proposed
train-only monitoring pool; its identity must be frozen before it is used for
stopping or model selection.

## Throughput Bound And Next Gate

The accepted joint-training report contains 992 measured 208-day direct-GPP
computations. Median direct forward and gradient times are `0.6993 s` and
`1.3191 s`, or about 103 Teacher days/s for the existing exact two-pass
sufficient-statistics path before raw-shard loading and assembly. At that
compute-only rate, one million unique days is roughly 2.70 hours and the
8.52-million-day post-reservation pool is roughly 23 hours on one V100.

Do not treat that extrapolation as a paid-training estimate yet. The next
operation is a direct-only, point-year-local streaming prototype that measures
raw v6 shard read, assembly, host-to-device transfer, forward, gradient, and
peak bytes for representative batch sizes. It must avoid the retained-tail and
7/30 rollout payloads that made the old prepared artifacts inappropriate for
full-epoch direct supervision.
