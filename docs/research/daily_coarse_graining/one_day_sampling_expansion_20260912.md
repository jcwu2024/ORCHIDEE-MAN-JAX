# Prepared one-day pool sampling expansion

Date: 2026-09-12.

This note defines the metadata-only sampler for broader one-day training data.
It implements the sampling part of the approved infrastructure work; it does
not start training, generate Teacher data, or admit a final training pool.

## Scope and integration boundary

`daily_operator/one_day_sampling.py` plans batches over one-day rows that have
already been prepared and accepted. It does not open or decompress prepared
packages. The data-loading layer remains responsible for validating manifest,
contract, schedule, artifact, and shard identities and for gathering the
planned `(artifact_id, batch_row)` values from `artifact.batch.one_day`.

`OneDayArtifact.from_schedule_entry(...)` reconstructs the preparer's flattened
row order directly from `components["one_day"]`: record order first, then each
record's `start_rows` order. Every flattened row retains:

- artifact ID and flattened batch-row index;
- source record and within-record row indices;
- landpoint, year, and source start row;
- parent and typed shard hashes when present; and
- an optional prepared day index supplied by the caller.

This mapping lets the root integration verify identities from the v6 index and
the accepted data contract before package materialization. It does not infer
missing identities from array contents. In particular, it does not infer a
calendar date: `start_row` is the physical-day identity within a point/year,
and `prepared_day_index` is recorded only when supplied from the payload.

## Public sampling API

`OneDaySampler` accepts admitted `OneDayArtifact` metadata plus a seed, batch
size, microbatch size, maximum artifacts per batch, and optional exclusions.
Its epoch scope is explicitly named `prepared_one_day_pool`. It must not be
reported as a full-v6 epoch.

`iter_epoch(epoch)` yields immutable `OneDayBatch` plans. Each plan contains
the exact ordered rows, artifact IDs, draw offset, microbatch slices, pool-wide
counts, and a compact next cursor. The loader can materialize those exact rows
after opening only the referenced artifacts.

## Deterministic ordering and bounded mixing

Before shuffling, artifacts are put in canonical artifact-ID order: ascending
integer IDs followed by lexicographically ordered string IDs. Rows within each
artifact are put in ascending flattened batch-row order. For each epoch, one
NumPy `Generator` backed by `PCG64` is initialized from a `SeedSequence` whose
entropy is the `(seed, epoch)` pair. It draws one artifact permutation and then
one row permutation for each artifact in canonical order. This conventional
seeded permutation is reproducible for the fixed environment and independent
of the caller's artifact sequence.

The sampler then opens a logical active set of at most
`max_artifacts_per_batch` artifacts and draws from them round-robin. Exhausted
artifacts are replaced only at the next batch boundary. Consequently, if the
current active set runs out before `batch_size` is reached, that
constraint-limited partial batch is emitted. A later artifact is not pulled
into the same batch merely to fill it.

Every included prepared row occurrence is emitted exactly once per prepared-
pool epoch. Final and constraint-limited remainders are preserved. Neither
batches nor microbatches are padded, duplicated, or silently dropped.

## Exclusion identities

Two exclusion levels are intentionally separate:

- `PreparedRowKey(artifact_id, batch_row)` removes one exact prepared-row
  occurrence. Another artifact containing the same Teacher day remains
  eligible.
- `DayKey(landpoint_id, year, start_row)` removes every occurrence of that
  semantic Teacher day across all admitted artifacts. This is the required
  boundary for preventing physical-day overlap with fixed train-only monitors
  or other holdouts.

Exclusions are applied before epoch ordering. Unknown exclusion identities are
rejected, as is an exclusion set that removes the complete pool. Root still
owns the audit that constructs the exclusion sets and verifies that the final
combined pool is train-only.

## Draw and coverage reporting

A row draw is one exact prepared occurrence. A unique day is one distinct
`(landpoint_id, year, start_row)` identity. Because the same Teacher day can
occur in multiple packages, these counts can differ. Pool, batch, and
microbatch metadata report them separately; batch metadata also records the
cumulative unique-day count through that batch.

The downloaded extension source manifest lists 2,000 packages, updates
1001-3000. At 32 prepared one-day rows per package, 64,000 is the maximum
extension row-draw count before admission and exclusions. It is not yet the
final combined-pool count: root must discover the existing and extension
metadata, verify their identities, apply exclusions, and report the admitted
result. The previously reported 539,171 rollout-day coverage count is a
different quantity and must not be used as one-day training coverage.

## Resume behavior

`OneDaySamplingCursor(epoch, batch_offset)` is sufficient to resume. The full
ordering is re-derived from the fixed sampler configuration, admitted artifact
metadata, seed, and epoch; no serialized full schedule is required. After a
batch is durably consumed, callers persist that batch's `next_cursor`. Resuming
from it yields the exact suffix of the original epoch. A cursor at the epoch
end yields no batches, while an offset beyond the derived epoch is rejected.

Reproducible resume therefore depends on preserving the admitted pool and
exclusions as part of the surrounding run contract. The cursor alone is not a
claim that two different pool definitions are interchangeable.

## Claims not made

The implementation and unit tests establish deterministic sampling semantics,
identity mapping, exclusion behavior, bounded package mixing, remainder
preservation, metadata accounting, and schedule-free resume. They make no
claim about training success, throughput, memory admission, scientific
accuracy, held-out performance, or full-v6 branch or data coverage.
