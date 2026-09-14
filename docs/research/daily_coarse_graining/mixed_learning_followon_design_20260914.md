# Mixed Learning Follow-On Design

Operational update: the executable200-update method is now implemented in
`mixed_learning_protocol_20260914.md`; use that document and NEXT_STEPS for
current tests/launch status. Implementation-pending wording below is history.

T30 prerequisite is now completed: actual native/JVP and two mixed updates
pass. See `mixed_physical_recurrent_t30_results_20260914.md`. Measured live
peak16.801GiB and second complete update20.76s support preparing the bounded
learning resource envelope; they do not establish convergence or eliminate
startup/IO/evaluation costs. Persistent learning implementation and a concrete
resource request are now next, rather than another two-update admission.

Metadata preparation update: implementation a9c2d03 passed13 focused tests and
completed on free test03 in124.9s. Report/index:
`runtime/outputs/diagnostics/mixed-learning-200-recurrent-schedule-v1.json`
(local copy under `outputs/`). It contains400 unique semantic windows,
267 sites,44 years,395 point-years and7387 unique Teacher days. The7400 day
occurrences include13 cross-update repeats; no same-update overlap. It excludes
the64 monitor and296 development dates. The single JSON is1535205 bytes;
there are zero Teacher payload reads/copies and no new per-window files.
Available unique candidates after exclusions/alias dedup:15968 T7 and7988 T30.
The preparer removed20 T7 and2 T30 duplicate semantic references. This admits
the metadata schedule only; a persistent learning runner and resource request
still need implementation and the T30 actual-update result.

Date: 2026-09-14. Status: design audit only; implementation, executable
manifest, resource envelope, and launch remain pending the active T30 mixed
admission. This is a bounded learning run, not the final reproducible training
method, an epoch, independent validation, or a reason to inherit exploratory
weights.

## Preconditions and scope

Use the active `mixed_physical_recurrent_t30_admission_20260914.md` gate
first. It must complete its two fresh T30 updates, physical/defined-value
checks, first-window native-gradient preservation, and two JVP directions. Its
measured compilation time, compiled/live/reserved memory, and warm complete
update time determine whether a 200-update request can be formed. Do not use
the prior T7 `16.803 GiB` live peak as a T30 capacity claim.

The proposed run starts from the declared fresh seed `20260911`, trains all
`109,464,375` Attention parameters, and creates a new AdamW state. It retains
the current constrained daily update, physical-feedback-v2 daily objective,
canonical rollout, checkpointed native AD, recurrent science loss/scales, and
shared physical/science extractors. It does not modify architecture, AD
boundaries, Teacher data, or the existing 16-window development screen.

The budget is exactly 200 optimizer updates. It consumes 64 one-day examples
per update in eight microbatches of eight. The broad pool is the accepted
346,539-row TRAIN catalogue, retaining its fixed 64-day train-domain monitor;
the current `fixed_replay_pad` one-day sampler is the appropriate existing
fixed-shape path. Thus the nominal one-day exposure is 12,800 draws; report
unique Teacher-day exposure and any replay padding rather than treating that
number as an epoch or an all-pool pass.

## Objective and update

Each update has one T7 and one T30 TRAIN trajectory. Let `D` be the mean of
the 64 daily examples, and `R7` and `R30` be the existing complete-window
recurrent-science scalars, each reduced over its own one-window sample. Freeze
the mixed scalar as:

```
L = 1.0 * D + 1.0 * (0.5 * R7 + 0.5 * R30)
```

This keeps daily and recurrent families equally weighted as the admitted
mixed updates, while preventing the selected horizon from changing recurrent
weight. It is not a per-day reduction and it does not retune the declared
within-recurrent flux/stock coefficients. Accumulate the eight daily
microbatch gradients to their mean; form the T7 and T30 gradients separately;
apply the equation above; then perform exactly one global clip and one AdamW
step. Do not step, clip, or update optimizer moments per microbatch or per
horizon.

Reuse the admitted daily/T7/T30 native-AD evidence and the tested linear
gradient-combination helper. Merely changing the sampled horizon mixture does
not require another full-model JVP compilation. Recheck only a consequential
change to the loss, state, or AD boundary; validate ordinary complete updates
within the learning run and preserve actual results.

## Reusable data and scheduling changes

`PreparedOneDayPool` and `OneDaySampler` already accept explicit row identities
and resumable epoch/batch cursors. Reuse them through `pool_plan`, changing
only the new protocol's sampling values to batch64/micro8 and
`fixed_replay_pad`. Before freezing that protocol, run its metadata-only
batch64 packing check against the actual catalogue and record the required
`max_artifacts_per_batch`; do not assume the existing batch32 cap remains
sufficient. Keep the catalogue source identity, exact daily row order, and
sampler cursor in every checkpoint/report.

The necessary recurrent change is narrow: separate the general prepared-window
reader from `scripts/hpc/trained_screen_common.py::WindowPool`'s present
16-window development-selection validation. The reader should accept a source
manifest plus an explicit metadata row and retain the existing `case()` checks:
artifact identity, component/horizon match, consecutive native day indices,
and retained-input binding. `WindowPool` can remain the strict development
reader; the training reader must not weaken `validate_selection` merely to
accept more training windows.

Build one immutable metadata-only 200-update recurrent schedule from the
existing `prepare_recurrent_training_probe.py` candidate route:

1. Reconstruct the exact train/train schedule from the existing prepared source
   manifest and retain its identity and candidate-source metadata. Do not
   create, copy, or materialize trajectory payloads.
2. Exclude all 64 fixed monitor days and all 296 development-window days before
   candidate selection. Keep the resulting exclusions and their source paths in
   the schedule. The one-day TRAIN rows remain eligible, as in the admitted
   recurrent selection.
3. With one declared seed, make a deterministic shuffled candidate stream for
   each horizon. Assign the first feasible unused T7 and T30 windows to each
   update. Require no repeated window identity and no T7/T30 day overlap within
   an update. Persist all 400 chosen metadata rows, per-update order, source
   identities, and a schedule hash in one JSON file.
4. Before a launch, preflight and report available/selected candidate counts,
   unique point-years, and unique recurrent Teacher days. These counts are not
   assumed here. If the existing source cannot satisfy the declared 200 pairs
   under these rules, fail before model bootstrap; do not silently recycle the
   16 probe windows. Cross-update day overlap may be reported but must not be
   hidden; global day-disjointness should be imposed only if that preflight
   demonstrates it is feasible without reducing the declared schedule.

The current candidate source records 2,000 prepared artifacts, but that is not
evidence of the eligible window count after exclusions. The schedule, not a
new prepared recurrent cache, is the persistence boundary. This avoids
per-window/per-update files and reuses the existing compressed artifacts.

## Checkpoints, monitoring, and decisions

Write atomic full recovery checkpoints at updates 0, 50, 100, and 200. Each
contains parameters, AdamW state, completed-update count, daily sampler cursor,
recurrent schedule cursor, method/protocol identities, and the fresh-init
identity. A rolling recovery file may be overwritten only for crash recovery;
the four milestone files are retained. Do not select weights from a later
exploratory checkpoint, and do not save one checkpoint or one recurrent data
artifact per update.

At 0/50/100/200, evaluate the unchanged excluded-64 one-day monitor and the
unchanged eight-T7/eight-T30 development panel using the shared extractor and
existing full physical report. Also report the current run's fixed first
training daily batch and a predeclared fixed recurrent TRAIN panel, both
separate from the rotating update data. Record loss families, GPP/NPP,
maintenance/growth/total autotrophic respiration, heterotrophic respiration
with zero-target totals flagged, AGB/BGB, canopy stock change, nonliving and
ecosystem carbon, plus all hard-count and coverage fields.

Immediate *technical* stop conditions are nonfinite loss/gradient/parameters
or optimizer state, invalid clip, any established physical/defined/discrete or
budget hard count, missing required metric population, source-identity/cursor
mismatch, or T30 resource failure. These are validity failures, not early
scientific conclusions.

There is intentionally no outcome-chosen stop at update 50 or 100 for a small
loss change, clipping, or a single bad science key: those are measured learning
signals. Complete 200 updates unless a technical stop occurs. At each milestone
classify the development panel separately as physical validity and the existing
13-key absolute science screen (T7 limits `<=0.35` NRMSE/`<=0.10` normalized
bias; T30 `<=0.25`/`<=0.07`). A failure is scientific evidence and a reason
not to promote the checkpoint; it is not a license to alter the budget,
weights, windows, or architecture mid-run. Canopy/litterpart diagnostics do
not compensate for primary-key failures.

Even a complete passing development screen is train-domain, day-disjoint
development evidence only. This run demonstrates or fails to demonstrate
bounded learning. It cannot claim independent validation, full-pool coverage,
full-epoch convergence, 365-day behavior, or the final paper method. A formal
route still needs a separately declared initialization, sampling/split policy,
stage transitions, stopping/checkpoint-selection rule, and independent
non-test validation.

## Resource decision after T30

No paid resource request follows from this note. After the active T30 result,
use its actual peak live and reserved memory, compiled executable memory,
initial compilation/setup time, and at least its warm complete-update timing to
size the 200-update envelope. The estimate must add explicit milestone
evaluation/checkpoint time rather than extrapolating the T7-only warm sample.
It must name the actual GPU/node, wall limit, runtime checkout, command,
working directory, and worst-case cost before any paid submission. If T30 does
not fit the available V100 envelope or the measured 200-update estimate cannot
fit the proposed limit, report a resource-limited admission and revise the
declared budget only in a new contract.

## Implementation map and risks

Minimal implementation ownership:

- `scripts/hpc/probe_mixed_physical_recurrent.py`: retain it for the active
  T30 admission only; do not stretch its fixed four-window/two-update branch
  into the learning runner.
- `research/daily_coarse_graining/daily_operator/prepared_one_day.py` and
  `one_day_sampling.py`: reuse explicit reads, fixed packing, metadata and
  cursor semantics; add no data format.
- `research/daily_coarse_graining/daily_operator/forward_screen_selection.py`
  and `scripts/hpc/prepare_recurrent_training_probe.py`: reuse candidate
  reconstruction and exclusion logic to build the immutable rotating schedule.
- `scripts/hpc/trained_screen_common.py`: factor or add a training-specific
  window reader without weakening the strict 16-window development contract.
- A new mixed-learning runner/protocol should own the one combined gradient,
  two cursors, atomic milestones, report, and final bounded classification.

Principal risks are T30 memory/compile cost, host decompression/cache churn
across many artifacts, an insufficient eligible metadata pool after exclusions,
accidental reuse of development windows, resumption that advances only one
sampler, and interpreting zero Teacher respiration as learned behavior. The
preflight/schedule identity, paired checkpoint cursor validation, physical
fail-closed reporting, and T30-based resource decision address these without
changing the model.
