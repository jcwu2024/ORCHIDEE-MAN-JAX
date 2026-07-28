# Daily Coarse-Graining Handoff

Snapshot date: 2026-07-28

This is the single operational handoff page for the daily coarse-graining
research branch. Read this page before dated experiment reports. Stable
scientific and release facts remain authoritative in
[`../../current-status.md`](../../current-status.md).

## Five-Minute Startup

1. Check out `research/daily-coarse-graining` and confirm a clean worktree.
2. Read this page, then
   [`daily_markov_contract_v5.md`](daily_markov_contract_v5.md) and
   [`teacher_dataset_generation.md`](teacher_dataset_generation.md).
3. Connect through the restored `cln01` alias and submit computation only
   through Slurm.
4. Check the 669-point production admission state below. Do not mix Teacher
   commits within one aggregate.
5. After full aggregation, run the production data-product admission command
   before any GPU training.

```bash
git status --short --branch
git log -1 --oneline
```

The ten-point v4 production code is fixed at commit `1f19ed7`; later research
commits do not alter its process image or outputs. Any recovery of point 319
must use a dedicated clean worktree at `1f19ed7`, not the advancing neural
worktree. Any future generation run must record its own exact Teacher commit.

## Current Architecture

The learned operator replaces exactly this block:

```text
48 half-hour SECHIBA/HYDROL/THERMOSOIL/DIFFUCO/ENERBIL transitions
+ daily accumulation and maintenance
+ 48 ordered OK_LEAK transitions
```

Its contract is:

```text
S[d] (3,854 continuous + 12 exact discrete values)
  + native 6-hour forcing[d] + parameters/static conditions
  -> B_fast[d] (2,855 values)
B_fast[d] + S[d]
  -> retained source-backed daily season/STOMATE
  -> S[d+1]
```

`B_fast` is a complete same-day interface, not persistent state. `S` is the
cross-day Markov state. Contract v5 carries only the true cross-day subset of
the 93-field SECHIBA finalize packet and learns only its fast-owned `leaf_ci`
output, not the duplicated finalize packet. For a 1961 cold start, the real Teacher executes Day 1
to construct canonical `S[1]`; the first training sample is Day 2. The shard
schema is `daily_teacher_markov_year_v5` and uses lossless NPZ compression.

Contract v4 is superseded. A real 1962 Day 2 differentiable retained-tail gate
proved that v4 could not advance `leaf_ci`; all other next-state leaves closed.
V5 appends 40 compact `leaf_ci` values from existing `S[d+1]`, so the complete
nine-point v4 dataset can be migrated without rerunning Teacher. The v5 gate
passes at `5.68e-14` maximum state error with zero mask/discrete mismatches and
finite gradients for all 2,847 defined target values.
The three-day real retained-tail scan also passes at `1.71e-13` maximum error
with 8,541/8,541 finite defined target gradients.
The reverse compile still emits the nonfatal XLA algebraic-simplifier 50-run
warning and then completes; keep it as a runtime-version A/B item.

The intended final repository has independent runtime choices for execution
backend (`cpu` or `gpu`) and transition implementation
(`teacher_half_hour` or, after acceptance, `neural_daily`). The research
branch is temporary development history, not a separate product.

The next matched architecture candidate, `axis_process_coupled_v1`, is now
implemented but untrained. It derives exhaustive state-axis partitions and
eight target families from Contract v5, uses eight source process tokens and
two process-coupling blocks, and carries no hidden cross-day memory outside
canonical `S[d]`. Its real Contract v5 layout SHA256 is
`bbe5e694f9bc2e8e4038f955cdbc967464b27bae883e95048ad092d1e734d24d`.
The candidate has 1,954,041 parameters versus 1,963,369 for the flat control,
and its local forward/reverse, checkpoint, restart, masking, conditioning, and
multiday-scan gates pass. This is implementation evidence only; no accuracy or
promotion claim has been made.

## Active Ten-Point v4 Production

The frozen ten-point v4 run uses plan
`runtime/plans/teacher_initial_10point_v4_1f19ed7.json`, plan SHA256
`b32eef75aafc6793c886bcec5858b4f0b343b8e510d12fb744518d35030d84af`,
four persistent one-CPU workers, and output root
`runtime/outputs/training/pft14-daily-teacher-initial-10point-1961-2010-v4-1f19ed7`.

Worker array `14365952` completed 459 of 500 landpoint-year entries. Workers
1-3 completed. Worker 0 timed out after completing all years for points 001
and 281 plus point 319 through 1969. Its residual internal job `14365953`
was finally removed on 2026-07-24; `squeue` no longer recognizes it and its
worker steps are recorded as cancelled. The only unfinished work is point 319
for 1970-2010, 41 entries. The obsolete aggregate `14365958` cannot pass its
`afterok` dependency.

After `CG` clears, read and formally recover the existing worker-0 lock, then
resume worker index 0 with worker count 4, the same plan/output root, and
`ibc11b02n13` excluded. Do not rerun any completed shard. Submit a fresh
aggregate only after the resumed worker completes.

Neural architecture work does not need to wait for that recovery. A
provisional nine-point dataset may be materialized with `teacher_shards
aggregate-subset`, selecting the nine complete 1961-2010 landpoint chains and
excluding point `319.0-057.0` entirely. It must use a distinct dataset ID,
statistics asset, acceptance report, experiment directory, and checkpoint.
The partial 1961-1969 point-319 chain must not enter this dataset. Results are
architecture-development evidence only and cannot replace the frozen
ten-point v4 experiment.

The provisional nine-point v4 dataset, flip-classifier A/B, and fresh 15-epoch
V100 run are complete historical architecture evidence. See
[`nine_point_v4_neural_baseline_20260724.md`](nine_point_v4_neural_baseline_20260724.md).
The accepted flip classifier restored persistence-level defined/undefined
accuracy. The 15-epoch run reached its best equal-split score of `0.772370` at
epoch 10 versus `1.064329` for matched persistence. Temporal/spatial/joint
improvements were `67.0%/28.0%/18.0%`. Longer training did not resolve the
DIFFUCO/ENERBIL and HYDROL spatial errors. Its seven-day free rollout reached
`0.921` RMSE and is superseded by the v5 contract correction. Do not evaluate
the sealed test split or merely add more v4 epochs.

The nine complete chains were migrated losslessly to contract v5 at
`runtime/outputs/training/pft14-daily-teacher-9point-1961-2010-v5-4c0f886`.
Migration job `14371675` produced 450/450 shards; provenance repair job
`14371681` rebound the source plan, subset, source-manifest hash, and migration
snapshot without rewriting shard arrays. Acceptance job `14371682` completed
in 1:37 with about 203 MiB peak RSS. Its assets are under
`acceptance-v5-5a16e78`: 96,354 train/train samples, zero target persistence
mismatches, and finite real-batch forward/loss/gradients. The earlier
acceptance attempt `14371680` is rejected because its migrated manifest lacked
the inherited generation-plan hash.

The first real multistep V100 smoke passed on `gln01` at research commit
`f68ea595dc244419736c6ef412f64100ab792ade`. It used one horizon-1 update and
batch size 1, completed its compiled stage in 104.84 seconds, reported loss
`0.04956254117421305` and gradient norm `0.38953763246536255`, and wrote a
finite checkpoint. Evidence is under
`runtime/outputs/smoke/multistep-gln01-smoke-f68ea59`; the report SHA256 is
`a97cbc9cd369336264e19243150c0089476de936e4a6df0782677fe36e2828cb` and the
best-checkpoint SHA256 is
`e0b0b6b05f04487d0d115174c2de337d52ca20d0ede3fa1cca157b8986cf7a55`.

This smoke exposed and closed two launcher defects before paid training:
multistep plan verification now uses the canonical JSON hash recorded by the
dataset manifest, and both GPU launchers pass the external runtime data roots
into the container. The superseded Slurm smoke `14371998` was cancelled before
allocation and is not evidence.

The complete v5 curriculum then ran directly on the shared `gln01` test GPU
from the accepted `23a1851` snapshot and finished successfully in 35:03. The
one-step phase selected epoch 10 with validation score `0.6804387148783407`.
The multistep phases completed 64 updates each at horizons 1, 3, and 7 with
finite losses and gradients; their mean losses were `0.0119944`, `0.0178953`,
and `0.0231536`. The final checkpoint SHA256 is
`79728593fa78f45f0c77dbe219f983114f76fe39bc0b63443c4af11e612b5bc2`.
Assets are under
`runtime/outputs/experiments/canonical-v5-curriculum-gln01-23a1851`.

Three joint-validation rollouts at landpoint `215.0-119.0`, year 2005, days
100-106 established the current failure mode:

- one-step checkpoint, free feedback: final normalized RMSE `0.473270`;
- multistep checkpoint, free feedback: final normalized RMSE `0.429058`;
- multistep checkpoint, Teacher-state feedback: final RMSE `0.068236`, maximum
  daily RMSE `0.086929`;
- all three had zero defined-status and discrete mismatches.

Multistep optimization therefore improved free rollout by about 9.3%, but the
provisional `<=0.29` gate did not pass. The retained-tail handoff remains
numerically valid; recursive distribution drift is the next optimization
target. The sealed test split remains untouched. The duplicate Slurm job
`14372761` was manually cancelled and must not be resubmitted.

A field-complete rerun at research commit `18471aa` reproduced all seven daily
RMSE values exactly and generated `drift-diagnostic-18471aa.json` and `.md` in
the experiment validation directory. The current state loss gives each of the
3,854 continuous scalar values equal nominal weight. Slowproc/STOMATE therefore
receives 61.91% nominal weight by width and contributes 76.89% of the realized
Day-1 Huber state loss. Day-1 leaders are growth respiration (`0.970347`), NPP
(`0.969258`), maintenance respiration (`0.886929`), biomass (`0.660380`), and
litter partitioning (`0.406447`); biomass alone contributes 36.39% of Day-1
state loss and reaches `4.841768` normalized RMSE on Day 7. Thermosoil and
surface-energy fields then become larger contributors as the feedback error
propagates. Do not respond by blindly increasing biomass weight: it is already
the largest realized loss term. Design the next objective around explicit
process-family balance, state-change/slow-state supervision, and recursive
stability, then compare it with the frozen baseline under equal compute.

The follow-up four-way split diagnosis at commit `825935b` is recorded in
[`four_way_rollout_diagnosis_20260725.md`](four_way_rollout_diagnosis_20260725.md).
Day-7 global RMSE for train/train, train/validation-year,
validation-point/train-year, and joint validation was `0.145471`, `0.134233`,
`0.519886`, and `0.429058`. This separates two real limitations: spatial
generalization is the dominant global failure, while recursive carbon-state
drift remains visible even on seen conditions (`litterpart` reaches `3.652`
normalized RMSE). Do not generate more years of the same points. Before a
spatial data expansion, complete one equal-budget objective A/B on the current
dataset and require named slow-state improvement; prepare a small diverse
landpoint selection independently from parameters/static data/forcing
climatology, then generate it only after the seen-condition objective gate.

The corresponding bounded `process_increment_v2` objective A/B is complete
and rejected. Its V100 curriculum remained finite, but the frozen four-way
rollout gate did not improve. On train/train Day 7, global normalized state
RMSE changed from `0.145471` to `0.146456`, biomass from `0.014259` to
`0.016716`, and litterpart from `3.651536` to `4.008877`. Train/validation
global RMSE worsened from `0.134233` to `0.152119`. All defined-status and
discrete mismatches remained zero. Do not tune this objective further or
change the default from `canonical_multistep_v1`.

The training-method review is recorded in
[`autoregressive_training_review_20260725.md`](autoregressive_training_review_20260725.md).
The counterfactual Teacher re-entry gate is complete and recorded in
[`counterfactual_teacher_reentry_20260725.md`](counterfactual_teacher_reentry_20260725.md).
Job `14380170` completed in 12:08. Exact Teacher queries at model-generated
Days 2, 4, and 7 had mean same-state neural operator error `0.085390`, mean
Teacher state sensitivity `0.073494`, and zero defined-status/discrete
mismatches. Teacher re-entry is valid; do not add a physical projection as the
primary next experiment.

The detached-prefix implementation gate passed at commit `c3e2d79`; see
[`pushforward_smoke_20260726.md`](pushforward_smoke_20260726.md). Job
`14380290` completed a three-day detached prefix, one exact same-state Teacher
query, and one final-transition update in 6:33. Loss was `0.0836542`, gradient
norm was `0.1333402`, and the 45 MiB checkpoint is finite.

The equal-update candidate and frozen four-way validation are complete; see
[`pushforward_objective_ab_20260726.md`](pushforward_objective_ab_20260726.md).
Job `14383581` trained 192 batch-4 updates and job `14384100` evaluated the
four frozen windows. Seen-condition global and litterpart errors improved
strongly, but NPP/growth respiration regressed about 9.6%, validation-year
biomass regressed 10.6%, and spatial global/litterpart errors worsened. The
candidate fails the predeclared gate and is rejected. Keep
`canonical_multistep_v1`; do not tune this objective, inspect the sealed test
split, or generate more same-point years.

The rejected pushforward run's extreme batch has now been fully attributed and
fixed at the shared inference boundary; see
[`physical_domain_projection_20260726.md`](physical_domain_projection_20260726.md).
The network had produced negative `carbon_32l`, `DOC`, and `deepC_peat`
material stocks, and the exact `soilcarbon_leak/PERMA_PEAT` re-entry converted
that invalid state into NaNs and a `7.27e8` artifact. Commit `5414d8c` applies
the source-backed nonnegative-stock projection before retained STOMATE in both
NumPy and compiled JAX paths. The exact four-sample maximum loss fell from
`562889.22` to `0.11654`, and 980 defined-status mismatches fell to zero.

The frozen four-way baseline was rerun with this promoted reconstruction
semantics. Day-7 RMSE is `0.144765`, `0.133391`, `0.519286`, and `0.428353`;
all mask/discrete gates pass. These small changes do not alter the diagnosis:
spatial generalization is still the dominant limitation, while train/train
`litterpart` still reaches `3.651537`. Do not reopen dynamic undefined-mask
classification for these carbon fields. The next bounded candidate must use
process-aware state encoding, persistent structured condition modulation, and
the promoted stock-domain projection before any spatial-data pilot.

That bounded architecture is now implemented as
`structured_process_film_v1`; see
[`structured_conditioned_architecture_20260726.md`](structured_conditioned_architecture_20260726.md).
It wraps the complete accepted flat trunk with eight source-derived state
adapters and parameter/static/annual condition FiLM at both fusion layers.
Zero initialization reproduces the flat checkpoint elementwise on the real v5
dimensions. Architecture identity, group hash, checkpoint upgrade, training,
rollout, condition permutation, and counterfactual diagnostics are wired and
the local forward/reverse gates pass.

The matched two-arm GPU continuation A/B is complete and recorded in
[`structured_architecture_ab_result_20260726.md`](structured_architecture_ab_result_20260726.md).
Job `14386641` completed in 1:03:27 with all expected evidence. The structured
arm improves the two spatial cases by about 4.3% relative to the matched flat
continuation, but regresses seen-condition global error by 7.7%-11.3% and does
not pass the condition-use threshold. The flat continuation also worsens every
frozen global score. Reject both continuation checkpoints and keep the
original checkpoint with SHA256 `79728593...b5bc2` as the baseline.

The contract-level 669-point Teacher data-product admission now passes; see
[`teacher_669_data_product_admission_20260726.md`](teacher_669_data_product_admission_20260726.md).
The machine policy binds the frozen production spec to Markov Contract v5 and
reports 33,450 point-years, 12,208,581 transitions, about 104-162 GiB, and an
estimated 67-91 CNY CPU cost. The next operation is to stage and verify the
complete source assets, build the canonical generation plan, and present the
paid Slurm request. The accepted staging root is
`runtime/assets/teacher_669_a403cd3`; all 4,014 managed files and
1,005,163,134 logical bytes verify. The generation plan is
`runtime/plans/teacher_669_1961_2010_v5_a403cd3.json`, with canonical SHA256
`8fcd6b84703311e2f848bd3c3ee2e43988240b69a73d91c16ded2a0b710d7f8b`,
33,450 entries, block size 7, and five-worker loads of
`6700/6700/6700/6700/6650`. Bind the worker launcher to the clean v5 worktree
and use bounded `TEACHER_MAX_NEW_ENTRIES` waves so jobs exit normally rather
than timing out. Do not create another architecture-specific spatial pilot,
inspect sealed test outputs, add epochs to either rejected arm, or tune FiLM
widths. After aggregation, run admission with
`--require-production-dataset`; only that mode can promote the full inventory.

The baseline contains one archived calibrated parameter tuple per landpoint.
Any controlled parameter perturbations or alternate forcing products must be
separate parent-hash-bound extension datasets using training landpoints; they
must not rewrite the immutable 669 baseline.

### 669 Production Admission Probe

The first real bounded production shard was generated by array task
`14387506_0` on `cnmix` from clean commit `07fe2f9`, worker 0 of 5, with
`TEACHER_MAX_NEW_ENTRIES=1`. The scientific data operation succeeded:

- entry `001.0-071.0:1961` is complete under the cold-start contract;
- the v5 Markov contract SHA256 is
  `6813065b51e95a2ec6663d8bc92f5336148704aea502632a05fd51d7e2e9d442`;
- the shard contains 364 Day 2-365 transitions, continuous state
  `(365, 3854)`, fast target `(364, 2855)`, and native forcing
  `(364, 5, 9)`;
- NPZ, metadata, and checkpoint SHA256 values all match their worker-manifest
  records;
- model preparation took 151.1 seconds, capture took 2932.9 seconds, and the
  complete entry took 3084.5 seconds;
- peak process RSS recorded by the generator was about 29.5 GiB; the
  compressed NPZ is 3.13 MB versus 19.99 MB of uncompressed arrays.

Slurm recorded the task as `FAILED` with exit code 2 after 54:25 because the
old CLI treated an intentionally incomplete bounded worker assignment as an
error even after atomically writing the valid resumable manifest. The
scientific shard did not fail. The CLI now treats successful
`--max-new-entries` termination as exit code zero and has a regression test.

Do not mix the `07fe2f9` probe shard with shards produced by the post-probe
commit. Preserve it as admission evidence. Build a fresh canonical plan/output
root at the accepted post-fix commit before starting production waves; source
assets do not need to be retransferred.

That post-fix admission passed on 2026-07-27. Array task `14390213_0` ran from
clean commit `7397d1e` and canonical plan SHA256
`26c90c58f9a099e485c9cd4779b936e2f83d525f6f2b7a4908813b2c4fe56475`.
It completed with Slurm state `COMPLETED`, exit code zero, and elapsed time
42:43. The entry took 2381.0 seconds and reached about 29.5 GiB peak process
RSS. Its NPZ SHA256 is exactly the same as the original probe,
`25d704e7315f5fc99c8b6f551f8c90ff70f9b6ebb0d6d66fec96696d412ee363`,
confirming that the orchestration-only fix did not change the scientific
arrays. This five-worker output remains immutable admission evidence.

The production topology was subsequently expanded to 100 workers at the
user's direction: 20 `cnall` nodes, five one-CPU workers per node. The
orchestration-only launcher is frozen at commit `74bd4eb`; every rank
fail-closes unless the separate Teacher worktree is clean at `7397d1e`.
Workers on one node start five minutes apart by local rank to avoid overlapping
five approximately 29.5 GiB cold-compilation peaks. Per-rank logs and XLA
caches are isolated.

The dedicated plan is
`runtime/plans/teacher_669_1961_2010_v5_7397d1e_w100.json`, with canonical
SHA256 `9ba2caf5e4110f25c24908fddc013f2aecc1dc8fe5f0cfe48bdf54ad425fa17e`.
It assigns 350 point-years to workers 0-68 and 300 point-years to workers
69-99, totaling 33,450. Its fresh output root ends in
`v5-7397d1e-w100`. Before full production, run the prepared two-node,
ten-worker, one-new-entry-per-worker admission. The admitted five-worker
output remains evidence and must not be mixed into the 100-worker aggregate.

That bounded admission job `14391204` ended `FAILED` after 38:46. Worker 0
completed `001.0-071.0:1961`, wrote a valid v3 manifest, reproduced the
accepted shard SHA256 `25d704e7...2ee363`, and exited zero. Its entry took
1900.5 seconds and reached 30,872,352 KiB peak RSS. The other nine workers had
started valid entries but were killed before writing shards.

This was an orchestration failure, not a Teacher or memory failure. Slurm
`srun` waits only about 60 seconds after the first task exits before
terminating remaining tasks. The local-rank startup staggering made worker 0
finish first while the other ranks still had legitimate work. The launcher
now passes `--wait=0`, which means unlimited wait for all ranks on the deployed
Slurm version, while retaining `--kill-on-bad-exit=0`. Before resubmission,
recover only the nine terminal-job locks through the strict documented
workflow and transfer/fast-forward the launcher fix. Preserve worker 0's
accepted shard.

The corrected admission `14391639` subsequently passed on 2026-07-27. Slurm
reported `COMPLETED`, exit code zero, and elapsed time 55:46. All ten ranks
exited zero; the hash-verifying progress audit found 11 valid point-years,
ten partial worker manifests, 90 workers not yet started, no active locks, and
no errors. Worker 0 reused its accepted 1961 shard and completed 1962, closing
the first restart chain. The other nine ranks completed their 1961 cold
starts. Cold-process peak RSS was about 31 GB and the resumed worker peaked at
26,190,848 KiB. The last staggered worker exited normally after earlier ranks,
which directly accepts the `srun --wait=0` fix.

Do not derive the full-production cost from these one-new-entry process
timings. Worker 0's resumed entry still started a fresh Python/JAX process and
compiled its in-memory transition cache. Full workers run 300-350 entries in
one process. First measure a small consecutive-entry worker wave so cold,
second-year, and steady-state point-year timings can be separated, then freeze
the full wall time and worst-case charge.

The consecutive-entry calibration is now accepted. Single-worker `cnmix` job
`14391864` completed with exit code zero in 40:03 and extended worker 0 through
1966. The first new entry in the fresh process took 1743.348 seconds. With the
compiled transition resident, 1964, 1965, and 1966 then took 158.455, 158.915,
and 160.493 seconds, respectively. Their mean is 159.288 seconds per
point-year, including array assembly, compression, hashes, and checkpoint
writing; mean capture alone is 126.767 seconds. The hash-verifying progress
audit reports 15 completed entries, ten partial workers, 90 not yet started,
no active lock, and no error.

At that measured rate, a 350-entry worker needs about 16 hours including its
fresh-process compilation. The five-local-rank startup stagger adds at most 20
minutes. The full 20-node, 100-worker request should therefore use a 20-hour
hard limit. Projected actual use is about 1,530 core-hours, or CNY 107 at CNY
0.07/core-hour; the scheduler hard-limit exposure is
`100 * 20 * 0.07 = CNY 140`.

Production control and consumption preparation is complete:

- `teacher_shards progress` audits all 100 worker directories and lists only
  inactive partial/missing worker IDs as sparse recovery candidates;
- `TEACHER_WORKER_INDICES=3,27,81` maps three Slurm ranks to exactly those
  canonical workers; duplicate or out-of-range lists fail before launch;
- full aggregation requires v3 worker manifests whose assigned, completed,
  and shard inventories agree, no generation lock, one Teacher commit, all
  hashes, and exact checkpoint chains;
- `slurm_teacher_production_finalize.sh` gates aggregate, complete 669
  admission, train-only statistics, target audit, and finite model smoke;
- `daily_teacher_669_training_protocol.json` freezes balanced streaming,
  split usage, sealed final testing, and rollout promotion gates;
- the bounded reader preflight passed on a real local Teacher shard without
  training a network.

Full production generation is now complete. All 100 workers and all 33,450
landpoint-year entries are complete with no active locks or worker errors.
Finalization job `14399470` passed the complete-worker gate and wrote the
53,029,341-byte aggregate
`runtime/outputs/training/pft14-daily-teacher-669-1961-2010-v5-7397d1e-w100/dataset_manifest.json`.
It then failed after 1:06:19, at about 24.1 GiB peak RSS, before statistics or
neural acceptance. The failure was an admission-reader schema defect:
`teacher_data_product_admission` required a per-shard
`markov_contract_sha256` even though dataset manifest v4 deliberately stores
the already cross-shard-verified contract hash once at manifest top level.
The aggregator had already read and hash-verified every shard metadata file
and proved that all shard contracts agree.

The local fix makes production admission consume the real dataset-v4 schema
and retains the top-level contract metadata/hash gate. Its focused aggregation
and admission regressions pass. Do not rerun Teacher generation or rewrite
the aggregate. Transfer a clean launcher snapshot containing the fix, rerun
finalization into a new output directory, and require
`production_admission.json`, train-only statistics, and
`acceptance_report.json` before freezing or launching the architecture A/B.

That rerun is complete and accepted. Finalization job `14400343` ran from
clean commit `90f751e`, completed in 4:08:54 with exit code zero, and peaked
at 24,110,968 KiB RSS. The accepted directory is
`runtime/outputs/acceptance/pft14-daily-teacher-669-1961-2010-v5-7397d1e-w100-final-20260728-r2`.
Its gates report:

- 669 landpoints, 50 years, and 33,450 hash-verified shards;
- 12,208,581 total transitions and 8,591,565 train/train samples;
- zero persistence mismatches across all 2,855 fast-day target columns;
- 103,153,433 explicitly audited undefined target values;
- train-only statistics JSON SHA256
  `c9ed1c6ea4cfca6c0da9504a36d440a2b388375f01acf8bc78edc7645a4ffb37`;
- statistics NPZ SHA256
  `33865286021c39bbdb6511a115da55143789e857c31e701c7417ff1a9c50cdbd`;
- a finite batch-8 flat-model CPU smoke with loss `0.01924320124089718`,
  gradient norm `0.10434712955999045`, and all 16 gradient leaves finite.

The complete 669 Teacher labels are now admitted for training. The next
operation is to freeze the matched architecture-only A/B manifest. Do not
reuse the nine-point statistics or checkpoint, add mixed-horizon objectives
to this first comparison, or inspect the sealed test split.

That architecture-only screen is now frozen at
`manifests/coarse_graining/canonical_669_axis_process_architecture_ab.json`,
canonical SHA256
`39b94a28e51dba67017ac3f06973b7f4fbbfa4a560b249eec49420f4baed8099`.
Both arms start from scratch with seed `20260728`, consume all 8,591,565
train/train samples exactly once in the same balanced order, use batch size
256 and learning rate `1e-4`, and evaluate every temporal, spatial, and joint
one-step validation sample. The test split remains sealed.

The one-step production trainer now dispatches through the common architecture
registry, binds checkpoint identity to architecture and training-protocol
hashes, uses the bounded eight-shard reader with two prefetched batches, and
accumulates loss on device rather than synchronizing the host after every
update. The two-arm orchestrator verifies all dataset assets once, runs both
arms sequentially in one GPU process, enforces exact sample counts and the
5% parameter budget, and applies the predeclared screening gates. This screen
does not include retained-tail rollout training or alter accepted Teacher
semantics.

Before the paid full screen, run
`canonical_architecture_ab_real_shard_smoke_v1` on `gln01`. It must hash one
real train/train shard, execute batch-256 compile and hot updates for both
architectures, report finite losses and parameter leaves, and consume no
validation or test sample.

## Accepted Historical Production Run

The bounded architecture-development dataset is frozen by
[`../../../manifests/coarse_graining/daily_teacher_initial_5point_1961_2010.json`](../../../manifests/coarse_graining/daily_teacher_initial_5point_1961_2010.json):

- 5 landpoints: 2 train, 2 validation, and 1 test;
- 1961-2010 complete trajectories;
- 250 atomic landpoint-year shards;
- 91,245 daily transitions after excluding cold-start Day 1 at each point.

The first submission (`14362790`/`14362791`, commit `23046a3`) was cancelled
and rejected before acceptance. Its plan incorrectly assigned 366 days to
Gregorian leap years even though the paper forcing and Teacher lifecycle use a
fixed 365-day noleap calendar. Those outputs are diagnostic evidence only.

Replacement Explore1000 jobs use the noleap fix at commit `3afe92f`:

- worker array: `14363439`, two persistent one-CPU workers on `cnall`;
- after-success aggregate: `14363441` on `cnmix`;
- plan: `runtime/plans/teacher_initial_5point_noleap_3afe92f.json`;
- output: `runtime/outputs/training/pft14-daily-teacher-initial-5point-1961-2010-noleap-3afe92f`;
- plan SHA256: `353f517ddc84a46723a71de11b74a557af030fe367314689b8a6b6fb8e8e68b9`;
- validated plan: 5 landpoints, 50 years, 250 entries, all exactly 365
  days, worker loads 150 and 100 entries;
- final state: both workers and the aggregate completed with exit code zero;
- worker elapsed times: 4:47:03 and 3:32:02; peak RSS about 22.4 and 20.8 GiB;
- aggregate elapsed time: 1:10;
- accepted asset: five complete 1961-2010 chains, 250 shards, 91,245
  transitions, one Markov contract, and fixed 365-day noleap semantics;
- output footprint: about 1.2 GiB, including an 84 MiB legacy v3 manifest;
- requested worker limit: 8 hours; approved worst-case cost: about CNY 1.13.

The accepted jobs emitted worker-v2 and dataset-v3 manifests from their fixed
snapshot. The branch now also contains a streamed canonical `B_fast` trainer
and compact worker-v3/dataset-v4 manifests. The new reader and contract loader
explicitly support both dataset v3 and v4. However, the scientific v4 target
changed after the first neural experiment exposed sentinel and finalize-state
contract defects. The v3 five-point run remains valid Teacher/provenance
evidence but is not a valid input to the revised trainer.

The historical five-point run was monitored from `cln01`; no computation ran
on the login node:

```bash
/rmprog/slurm/v22.05.7/bin/squeue -j 14363439,14363441 \
  -o '%i %P %j %T %M %l %R'
```

Project runtime files must remain under:

```text
/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime
```

The two worker logs are expected at
`runtime/logs/teacher_14363439_0.txt` and
`runtime/logs/teacher_14363439_1.txt`. The aggregate log is expected at
`runtime/logs/teacher_aggregate_14363441.txt`. Read the `plan=` line in a
worker log for the exact generated plan path instead of guessing it.

## Completion Gate

This gate passed on 2026-07-23. Slurm `COMPLETED` alone was not used as
acceptance: aggregate job `14363441` reopened every worker manifest and shard,
verified hashes and schema consistency, rejected missing or duplicate entries,
and emitted the accepted dataset manifest.

After the jobs finish:

```bash
SLURM=/rmprog/slurm/v22.05.7/bin
$SLURM/sacct -j 14363439,14363441 \
  --format=JobID,State,Elapsed,ExitCode,MaxRSS,NodeList
grep -h '^plan=' runtime/logs/teacher_14363439_*.txt
tail -n 80 runtime/logs/teacher_aggregate_14363441.txt
```

Then rerun the same aggregate validator explicitly if an independent check is
needed, using the exact plan path printed by the worker:

```bash
.venvs/orcjax_cpu/bin/python \
  -m research.daily_coarse_graining.teacher_shards aggregate \
  --plan "$TEACHER_PLAN" --worker-count 2
```

Acceptance requires all of the following:

- 5 complete landpoint chains and 250 unique shards;
- 91,245 transitions with the frozen spatial and temporal split;
- every NPZ, checkpoint, plan, contract, and source-asset hash valid;
- one Teacher commit and one `daily_teacher_markov_year_v3` schema throughout;
- no partial worker manifest, stale lock, duplicate entry, or missing year;
- recorded per-worker elapsed time and peak RSS for production planning.

If a worker times out, do not delete `generation.lock`. Confirm the owning job
is terminal and use the strict `teacher_shards recover-lock` workflow described
in [`teacher_dataset_generation.md`](teacher_dataset_generation.md), then
resume only the incomplete worker assignment.

## Next Single Milestone

Request explicit resource and cost approval, then submit the full 20-node,
100-worker production run with a 20-hour hard limit. It must reuse the same
output root, plan, Teacher commit, launcher, and 15 accepted shards. After
completion, use sparse recovery only for named incomplete workers, then run
the fail-closed finalization pipeline. Do not train on the partial production
dataset, inspect sealed test outputs, or create another small spatial pilot.

The architecture and rollout-stability decision is frozen in
[`long_rollout_architecture_review_20260727.md`](long_rollout_architecture_review_20260727.md).
After complete data admission, compare a matched flat control with the
process- and axis-aware candidate before changing the rollout objective.

The prepared v4 subset is
[`../../../manifests/coarse_graining/daily_teacher_initial_10point_1961_2010_v4.json`](../../../manifests/coarse_graining/daily_teacher_initial_10point_1961_2010_v4.json).
It preserves the frozen 669-point parent split and contains six train, two
validation, and two test landpoints. The original five are retained; the five
additions are `069.0-119.0`, `281.0-095.0`, `283.0-091.0`, `295.0-113.0`, and
`333.0-057.0`, selected to add dry/low-productivity, high-productivity,
high-maintenance, low-Vcmax/short-residence, and geographic diversity.

The v4 production run was submitted on 2026-07-24 from fixed Teacher commit
`1f19ed7`:

- worker array: `14365952`, four one-CPU tasks on `cnall`, array concurrency 4;
- aggregate: `14365958` on `cnmix`, dependency `afterok:14365952`;
- plan: `runtime/plans/teacher_initial_10point_v4_1f19ed7.json`;
- plan SHA256: `b32eef75aafc6793c886bcec5858b4f0b343b8e510d12fb744518d35030d84af`;
- output: `runtime/outputs/training/pft14-daily-teacher-initial-10point-1961-2010-v4-1f19ed7`;
- worker loads: 150, 150, 100, and 100 landpoint-years;
- requested worker wall time: 7 hours; aggregate wall time: 20 minutes;
- worst-case charge: approximately CNY 1.98, with no GPU allocation.

Do not submit replacement jobs while this array is active. Completion requires
the aggregate manifest plus the v4 target-representation audit; Slurm
`COMPLETED` alone is insufficient.

The Explore1000 GPU runtime prerequisite is complete at commit `c566538`:

- project environment: `.venvs/orcjax_gpu`;
- JAX/JAXLIB/CUDA plugin: 0.4.38 with the pinned CUDA 12.1 compatibility set;
- real `gln01` V100 smoke: backend `gpu`, device `cuda:0`, finite JIT loss and
  gradient;
- formal entry points: `scripts.hpc.verify_orcjax_gpu` and
  `scripts.hpc.run_canonical_gpu_train`;
- train-only statistics job `14365346` completed in 68 seconds on one `cnall`
  CPU with exit code zero;
- obsolete v1 statistics: `runtime/outputs/training/canonical-initial-5point-1961-2010/training_statistics.json`;
- statistics JSON SHA256:
  `1041c1231fb6c9bb5905e0a9403aefc03666ba745390539cf0ec644531be6885`;
- statistics NPZ SHA256:
  `584cee4ffbf1d05e79f3e66b3a7998d9d94046539eb8583e3332a5be16f5773d`;
- consumer-level validation passed for dataset identity, contract identity,
  NPZ hash, finite means/variances/scales, positive scales, 32,118 train/train
  samples, and 88 source shards;
- first GPU training job `14365446` completed in 5:40 with about 3.1 GiB peak
  device memory, but is rejected for promotion: contract v3 included finalize
  mirrors and statistics v1 treated ORCHIDEE sentinels as finite data;
- its checkpoint and statistics must not be resumed or reused with v4.

GPU training now requires the matching passed `acceptance_report.json`; the
launcher refuses a missing or hash-mismatched report. Do not bypass the GPU
launcher with a direct import of `canonical_training_run`: in this container,
JAX plugin auto-discovery can otherwise initialize CPU first. The next GPU
operation comes only after the v4 ten-point dataset and v2 statistics are
generated and accepted.

## Decisions Not To Reopen

- Do not redefine the label as the final daily state. The accepted target is
  the complete `B_fast` boundary, while `S[d+1]` remains rollout supervision.
- Do not store all 48 interpolated forcing steps. Store native 6-hour forcing
  and reconstruct deterministic preprocessing.
- Do not invent a cold-start `S[0]`; bootstrap canonical `S[1]` with Teacher
  Day 1.
- Do not generate v1 or v2 shards. They are historical evidence only.
- Do not use landpoint ID as a neural feature or leak validation/test samples
  into normalization statistics.
- Do not assume one-landpoint Teacher generation benefits from a GPU. Current
  production uses persistent CPU workers; GPU is reserved for batched neural
  training unless a future workload-level A/B proves otherwise.
- Do not run one process per year or landpoint. Same-process executable reuse
  is the accepted performance mechanism.
- Do not restart PFT14 source-branch equivalence work merely because a neural
  experiment fails. First distinguish Teacher-data integrity, supervised
  optimization, and autoregressive rollout error.

## Authority Map

| Question | Authoritative file |
| --- | --- |
| Overall Teacher/release status | [`../../current-status.md`](../../current-status.md) |
| Current operational task | this handoff page |
| Daily state and target semantics | [`daily_markov_contract_v5.md`](daily_markov_contract_v5.md) |
| Dataset production and recovery | [`teacher_dataset_generation.md`](teacher_dataset_generation.md) |
| Research quality gates | [`development_standard.md`](development_standard.md) |
| Frozen five-point batch | [`../../../manifests/coarse_graining/daily_teacher_initial_5point_1961_2010.json`](../../../manifests/coarse_graining/daily_teacher_initial_5point_1961_2010.json) |
| Frozen ten-point v4 batch | [`../../../manifests/coarse_graining/daily_teacher_initial_10point_1961_2010_v4.json`](../../../manifests/coarse_graining/daily_teacher_initial_10point_1961_2010_v4.json) |
| Frozen 669-point population/splits | [`../../../manifests/coarse_graining/daily_teacher_production_669.json`](../../../manifests/coarse_graining/daily_teacher_production_669.json) |
| Explore1000 operations | [`../../deployment-explore1000.md`](../../deployment-explore1000.md) |

Dated reports in this directory preserve evidence under their original
conditions. They are not current roadmaps and must not override this page or
`docs/current-status.md`.
