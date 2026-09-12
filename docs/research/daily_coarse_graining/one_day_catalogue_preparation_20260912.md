# Existing one-day pool: catalogue and read-path preparation

Freeze this metadata/readiness task before execution. Source is the accepted
2000-package extension, update IDs1001-3000, under
`conditioned-coverage-expanded-14992138/prepared_manifest.json`. Do not regenerate
Teacher data or recalculate its statistics. Reconstruct the original3000-entry
data schedule from the v6 train/train index and original data contract and
require it to match the accepted schedule identity. Read only selected payloads
to verify metadata mapping and application-cache behavior.

Use the sampler's conventional NumPy PCG64/SeedSequence epoch scheme. Seed
20260912, requested batch32, microbatch4, at most4 packages per batch. Select
64 unique Teacher-row monitor identities using seed20260913 and exclude every
matching occurrence from training. These are fixed train-only development
monitors, not sealed validation or a generalization claim. Persist compact
source schedule entries, monitor identities and sampler settings, not a huge
per-update expanded schedule. Do not infer calendar day from row index; the
physical day key is point/year/row, with actual day index available on payload
materialization.

Report exact remaining row draws, unique Teacher rows, locations/point-years/
years, epoch batch/remainder counts, and a first-batch preview. A prepared-pool
epoch is not a full-v6 epoch;64000 row occurrences is the pre-exclusion ceiling
for this extension's one-day part, not the539171-day whole-window coverage.

The bounded reader keeps only one-day views in a four-package LRU cache, while
preserving exact targets/dtypes and row order. It may decompress a full existing
artifact to extract that view but does not retain unused7/30-day arrays in the
cache. Measure an application-cache miss followed by a cache-hit stack and
compare values exactly. Shared filesystem cache means this is not a cold-disk
benchmark. No parameter/model computation occurs.

Run locally validated sampling/cache primitives on free test02, one process,
15-minute cap. This produces a ready data catalogue and IO evidence. Batch4
gradient memory and throughput still require a subsequent bounded GPU probe
before fixing a larger training budget. No training or paid submission here.
