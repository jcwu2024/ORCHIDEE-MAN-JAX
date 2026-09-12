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

## Completed catalogue and IO pilot

Free test02 completed at e3aa80f in149.20s. Original data schedule identity
reproduces, selected payload schedules agree, and cached/uncached batch values
match exactly. Result:

- 2000 source packages,64000 one-day row occurrences before exclusions;
-64 unique monitor days excluded from all matching training occurrences;
-63936 remaining draws,63707 unique Teacher days,535 locations,
 17677 point-years, all44 years1961-2004;
-1998 full32-row batches per prepared-pool epoch, no remainder for this pool;
-first batch reads4 packages (13.45MB compressed), with73.51MB one-day cache;
-application-cache-miss stack0.536s, cache-hit stack0.038s on a shared CPU node.

Files: `runtime/outputs/diagnostics/one-day-catalogue-v1.json` and
`one-day-catalogue-v1.catalogue.json.gz`; the JSON report is copied to local
outputs. These are available rows, not consumed training data. The next
bounded resource test uses the first four declared batch rows before a larger
training budget is proposed.
