# Structured-GPP Fresh Independent 7/30-Day Screen V1

Date: 2026-08-31

Status: frozen before fresh-panel target values are read

Contract SHA256:
`894a0b2dc329179566e624e7c04b3a63522588aed34feb8dc00e3187eb73e6a7`

## Purpose

Broad-data Stage 1 established that the unchanged structured-GPP Attention
specialist can learn direct daily GPP from 208,000 unique train/train days. It
did not establish out-of-sample or free-rollout science quality. This screen
is the first independent test of the selected update-1000 checkpoint.

## Fresh Panel

The screen keeps the established temporal, spatial, and joint non-test slices,
with 16 windows in every 7-day and 30-day cell (96 windows total). Selection
uses only shard identity and day-count metadata under seed `20260831`. Every
point-year in the consumed panel `f4952f5f...33e79` is excluded from the
corresponding slice before ranking, and landpoints are distinct within each
cell. No target value may be read while the panel is selected.

The first window in each of the six cells forms an execution admission. It
cannot promote the checkpoint. The full 96-window panel is evaluated only
after admission passes.

## Measurements And Gate

Every selected day receives a Teacher day-start direct-GPP prediction. Its
area-weighted daily normalized RMSE and signed bias are reported to distinguish
day-local mapping quality from recurrence. They are report-only because the
accepted product target is free-rollout science and daily errors need not have
the same cancellation behavior as 7/30-day totals.

Promotion requires all 13 existing absolute keys to pass in all six
slice/horizon cells: GPP, maintenance/growth/autotrophic/heterotrophic
respiration, NPP, NEP, NEE, AGB, BGB, total live biomass, nonliving ecosystem
carbon, and ecosystem carbon. The limits remain the already frozen `0.35/0.10`
at seven days and `0.25/0.07` at thirty days for normalized RMSE/absolute
normalized bias. All prediction, state, mask, discrete, budget, and negative
stock hard-failure counts must be zero, and the direct-GPP metrics must be
finite and nonempty.

This is an absolute screen. No control or comparison against a weak parent is
required for acceptance.

## Boundary

The selected specialist is loaded into the unchanged coordinate-aligned
Attention parent and distribution-aware maintenance model. Evaluation uses
the canonical conservative daily operator and retained exact tail. It performs
no optimizer update, writes no checkpoint, regenerates no Teacher data, reads
no v5 or sealed-test data, reconstructs no 48 half-hour forcing records, and
executes no 48-step state scan.

A pass opens only a separately frozen 365-day and long-held-point plan. A
failure stops before 365 days, Gate D2, sealed test, and product acceptance.
