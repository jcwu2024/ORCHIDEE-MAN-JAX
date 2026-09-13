# Broader one-day data: declared preparation and real-data pilot

Status: user authorized expansion preparation; local implementation complete,
actual metadata/pilot results pending. No new paid allocation. Paid training
15061139 continues unchanged in its frozen checkout.

## Sampling and scientific scope

Keep all63707 unique admitted TRAIN days. At each of the17677 existing
training point-years, add16 different dates using seed20260913. Remove all
existing pool occurrences, all64 fixed monitor days and all296 developmental
rollout days before selection. Split remaining chronologically ordered row
indices into16 equal-size strata and draw one per stratum with a reproducible
point-year RNG. Do not equate row offsets with calendar day indices; retain
actual payload day indices. Use the existing conservative first-row exclusion.

Expected new282832 days and combined346539 unique days (about4.04% of8568025
eligible TRAIN days), spanning the same535 points/44 years. This improves
within-year coverage while preserving the broad base. It does not claim all
23540 training point-years, extreme-regime balance, or final-data sufficiency.
The next expansion can include new point-years after measuring this route.
No Teacher regeneration, v5 mixing, or held-out target access is allowed.

This is data preparation, not a new training run or a changed current run.
No loss/architecture/normalization/AD boundary is changed. Full-data fitting
does not wait for every small-pool science metric to pass. The next training
contract will explicitly budget unique exposure, updates, initialization and
checkpoint selection after current learning/rollout evidence. A346539-row
deduplicated epoch would require10829 full batches of32 plus a final11 rows;
its final-batch policy must be declared before training. Merely increasing
the pool with unchanged1998 updates would still consume only63936 draws.

## Data route

Prepared7/30-day arrays lack full process targets, so their539171-date union
cannot be counted as ready one-day data. Load each selected existing v6
parent and typed shard once, assemble its selected rows through the existing
DailyProcessTrainingAdapter and181-field fast-owner target builder. Reuse
retained day_inputs from an exact non-monitor occurrence of the same
point-year in an accepted prepared artifact. No cross-year assumption or
Teacher stepping is required. Canonical retained-tail binding supplies each
row's native forcing during actual model execution.

Write one-day-only chunks grouped by point-year in the existing prepared
artifact container (seven_day/thirty_day=None). This explicitly identifies
them as one-day artifacts; do not pass them to historical multi-day runners.
Preserve base rows once and include new dates in the same chunk. The existing
PreparedOneDayPool reader can read this payload. A complete manifest/catalogue
and final training integration are only accepted after full preparation;
four pilot chunks are not a complete expanded training pool.

## Bounded admission

Sol(high) independently audited full-loss label requirements and exclusions;
Terra(high) implemented the raw materializer using existing source helpers.
Local tests cover real row order/mask/dtype ownership with small mocks,
train-only/lifecycle restrictions and deterministic disjoint selection.

Run metadata selection on free test02, then four point-years with entry IDs
1/5893/11785/17677 under a15-minute CPU cap. Before execution exercise both
launcher branches with actual Bash --dry-run. Each pilot chunk compares one
existing row exactly against its accepted prepared counterpart across all
six one-day fields, then writes new dates and measures raw/read/assembly,
compression, size and memory. Read the saved payload with the unchanged
training reader and verify exact roundtrip. No neural forward or gradient
is needed for this data-only pilot; training admission for a future modified
batch/runner will be scoped to that actual implementation.

Use measured time/storage to size full preparation, accounting for shared IO
and parallel memory. Full paid CPU preparation requires a concrete resource,
time, command and cost proposal. Do not launch hours of unbounded processing
on free test nodes. Running paid training checkout stays immutable.
