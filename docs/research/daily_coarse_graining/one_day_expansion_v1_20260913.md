# Broader one-day data: declared preparation and real-data pilot

Status: user authorized expansion preparation; local implementation and
actual metadata/four-point-year pilot and parallel workers pass. Final merge
integration tests pass. Full CPU preparation proposed, not submitted. Paid training
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
checkpoint selection after current learning/rollout evidence. A fully packed
346539-row epoch would require10829 full batches of32 plus a final11 rows,
but the current sampler does NOT pack across exhausted artifact groups:
the actual metadata audit emits13258 updates,8838 full and4420 partial,
with microbatch sizes1/2/3/4. A future training-level carry layer should pack
consecutive rows into fixed32-row updates while preserving declared sampling
and checkpoint cursor semantics; its final11-row policy must be declared
and tested before expanded GPU training. Do not submit the existing runner
unchanged and assume10830 updates or only one remainder. Merely increasing
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

## Actual free-node evidence

At0a33541, test02 metadata selection passes in16.220s:63707 base+282832 new
=346539 unique TRAIN days;17677 point-years,535 locations,all44 years;
eligible fraction4.04456%; monitor64 and rollout296 excluded.
Report: `runtime/outputs/diagnostics/one-day-expansion-plan-v1.json`, copied
locally to `outputs/one-day-expansion-plan-v1.json`.

The four-point-year pilot passes in55.352s, exit0;4x18=72 rows,64 new.
All six one-day fields on the four reference rows match the old prepared
payload exactly, including dtypes/masks; persisted training-reader readback
also matches. Each parent+typed shard pair is4.47--4.54MB compressed;
each18-row chunk is0.455--0.460MB compressed/10.337MB uncompressed.
Per-point-year total times5.04--5.15s, assembly/read about4.90--4.91s;
peak host1.377GiB; final persisted read0.031s. This is real data admission,
not neural scientific acceptance. Runtime report:
`runtime/outputs/training/one-day-expansion-pilot-v1/prepared_manifest.json`,
local `outputs/one-day-expansion-pilot-v1.json`.

Measured extrapolation: about24.9 serial CPU hours, ideally46.7 minutes
with32 workers, before shared-IO/row-count effects; compressed output about
8.2GiB. These four point-years are a pilot, not a throughput guarantee.
Proposed production allocation:cnall,4 nodes x8 single-CPU workers,32CPUs,
no GPU,2h cap (110m workers+8m merge). Worst-case CPU cost
4*8*2*0.07=CNY4.48. No CPU --mem directive because of documented scheduler
memory metadata; measured8-worker data memory is well below node capacity.
The job is preparation only and remains unapproved/unsubmitted.

Working directory: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/one-day-expansion-v1`.
Proposed command: `/rmprog/slurm/v22.05.7/bin/sbatch scripts/hpc/slurm_one_day_expansion.sh`.
Output root: `runtime/outputs/training/one-day-expansion-<jobid>`.
Worker chunks are preserved independently; merge must verify completed range
coverage and retain/remap the same64 monitor payloads before reporting a
complete reader-compatible catalogue. Do not overwrite failed/partial chunks
or resubmit all preparation merely because a later merge needs correction.

At c6f51aa, parallel free test02/test03 workers prepare entries2--3/4--5,
respectively. Both exit0/status passed/stage completed with exact saved-reader
roundtrip, taking45.002s/84.025s including node-dependent bootstrap. Their
four point-years take5.04--6.13s each,38/40 rows,64 new total; peaks1.404GiB.
Together with the first pilot,8 different point-years/150 rows/128 new dates
are materialized. Local reports: outputs/one-day-expansion-parallel-worker0.json
and outputs/one-day-expansion-parallel-worker1.json. Free tests have finished.

The merge step validates completed chunk coverage and per-entry schedule
identities, remaps copied monitor artifacts beyond the new training IDs,
and produces a catalogue usable by the existing reader/sampler. Synthetic
compressed-artifact tests exercise its complete path, missing-chunk rejection,
monitor values/dtypes and actual sampling histograms. Root's combined new
data/merge/reader suite passes13 tests; earlier materializer/prepared-batch
compatibility suite passes17. Actual Bash expansions for pilot, last worker
range17125--17677 and production srun/merge command pass. No GPU preflight
or training restart is required for these data-only preparation changes.

Final execution checkout is fd47de3, observed after fast-forward deployment;
the merge entry point's actual test02 CLI import/help passes. The optional
attempt to repeat its local pytest suite on test02 found no installed pytest;
no environment change was made. Local compressed-artifact merge tests remain
passed, and actual raw/persisted data pilots are recorded above. This does
not block the bounded CPU preparation proposal. Next action is explicit
approval of the2h/CNY4.48 CPU job, then submit once from cln01. A successful
job must finish worker coverage and merge_report.json status=passed/stage=completed
before the expanded pool is described as fully prepared.
