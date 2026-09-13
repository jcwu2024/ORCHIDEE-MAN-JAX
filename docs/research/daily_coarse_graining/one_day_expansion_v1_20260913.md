# Broader one-day data: declared preparation and real-data pilot

## Completion audit and recovery

Job15061225 ended FAILED/exit9:0 after00:52:24. Cluster configuration is
WaitTime=50 seconds. Four workers10/11/20/23 completed normally; the srun
launcher, which omitted --wait=0, killed the other workers50 seconds after
the first normal exits. Responsibility is the launcher configuration; this
is neither a data-validity failure nor a neural/gradient result.

Free test02 inventory validates committed file existence/size and manifest
counts:15545/17677 point-years,304786/346539 rows saved (87.95%);2132
point-years/41753 rows remain. Compressed7742125326 bytes, uncompressed
175037380656 bytes, summed preparation96200.829 seconds. Every saved old-row
comparison was exact. No merge_report.json exists, so full-pool admission is
pending. Estimated actual CPU charge32*(52.4/60)*0.07=CNY1.9563.
Evidence: outputs/one-day-expansion-15061225-inventory.json.

Execution commit9e8b1a0 fixes both launchers with --wait=0 and adds explicit
--resume. Resume preserves committed files, validates the declared schedule
prefix and file metadata, skips their raw reads and prepares only missing
entries. A completed worker returns without model bootstrap. Original
source provenance is retained alongside the new invocation commit. No bulk
hashing or neural revalidation is introduced. Local focused suite19 tests
passes. Deployment is a separate checkout; actual Bash launcher and recovery
worker --dry-run expansions pass on the target environment.

Actual test02 interrupted-resume fixture at9e8b1a0 passes in132.877 seconds
for this invocation: copied one previously accepted chunk into a new fixture,
resumed the missing second chunk, and verified saved-reader equality and
that the committed file's modification time did not change. One entry reused,
one newly prepared; source pilot/production files untouched. Local evidence:
outputs/one-day-expansion-resume-smoke-v1.json. This admits the resume path,
not a whole-job throughput guarantee; free-node time varied from prior pilots.

Concrete recovery proposal (unsubmitted): cnall,2 nodes x4 single-CPU tasks,
8 CPUs,noGPU,1h limit,CNY0.56 maximum.28 unfinished old worker ranges are
balanced across8 ranks; estimated longest rank41.4min including bootstrap
allowances, subject to node/IO variation. Worker cap50min, final merge8min.
Working directory:
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/one-day-expansion-recovery-v1`.
Command:
`/rmprog/slurm/v22.05.7/bin/sbatch scripts/hpc/slurm_one_day_expansion_recovery.sh`.
The recovery resumes original15061225 worker output paths and completes the
existing merge/catalogue. New paid resources/time require user approval;
the earlier approval applied to the original2h job. Do not resubmit all rows.

## Reusable full TRAIN cache recommendation

The user's proposal is sound: prepare immutable reusable data once, then
sample day/window indices for each run. Full eligible TRAIN has8568025 days,
535 sites,44 years1961--2004 and23540 point-years. This differs from the
12208581 transitions in the full v6 release. Preserve train/development/
validation/test ownership and fixed64 monitor/296 rollout-day exclusions;
windows must respect those exclusions, continuity and owner boundaries.

Keep canonical inputs, targets, masks, actual dates and exact retained inputs
independent of batch size, epochs, loss coefficients and model checkpoints.
Do not regenerate v6, mix v5, or cache model predictions as Teacher truth.
Changing physically required fields may still require a cache schema update;
ordinary new draws or loss-weight changes should not rebuild the data.
Full disk preparation is distinct from loading everything into memory or
committing to a full training run. Load only the active batches/windows.

Linear scaling from the304786 saved rows gives202.70GiB compressed and
751.21 CPU-hours (CNY52.58 at0.07/CPU-hour). These are rough extrapolations,
not a requested budget or production quote. Full point-year raw IO is
amortized, contiguous compression changes, and repeated retained/static
inputs can be shared. The current gzip point-year payload may require
decompressing too much per random draw. Before selecting full-cache format
and production resources, measure bounded complete-point-year preparation,
peak memory, compressed size and actual random32-row/window read throughput.
Check exact retained-owner coverage for all23540 point-years; existing
expansion covers17677. Prefer chunked row access and explicit chronological
indices, with recovery at chunk boundaries. No full-cache paid run approved.

Expanded training separately needs fixed32-row packing, a reproducible
shuffle/cursor and an explicit final11-row policy: old grouped sampling
produces13258 updates/4420 partial batches, not10830 fixed packed updates.
Data availability must be reported separately from actual consumed coverage.

## Original preparation record

Status: CPU15061225 ended FAILED from launcher WaitTime behavior.304786 of
346539 rows are preserved; recovery prepared at9e8b1a0, not submitted.
Paid training15061139 continues unchanged in its frozen checkout.

Execution update: user approved the exact2h/CNY4.48 proposal below. Submitted
from cln01 at2026-09-13 19:04:41 scheduler time; originally ran on ibc11b02n20
and ibc13b07n26--28, cnall4nodes/32CPU/8 per node/noGPU. Frozen codefd47de3
in runtime/checkouts/one-day-expansion-v1. Output root:
`runtime/outputs/training/one-day-expansion-15061225`; scheduler log:
`runtime/outputs/diagnostics/one-day-expansion-15061225.log`.
This submission does not establish completed data preparation. Next read
worker progress and require the complete final merge acceptance.

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
The job is preparation only; subsequent approval/submission is recorded above.

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
not block the bounded CPU preparation proposal. Approval and submission are
now recorded above. A successful
job must finish worker coverage and merge_report.json status=passed/stage=completed
before the expanded pool is described as fully prepared.
