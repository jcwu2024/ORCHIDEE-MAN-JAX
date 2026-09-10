# Corrected Attention: second bounded continuation and independent evaluation

Status: job **14984032** completed all 20 updates on 2026-09-10; paired
relative transfer passes, absolute accuracy remains open. See the
[result and physical-fit interpretation](conditioned_continuation_v2_result_20260910.md).
This plan was fixed before new validation target arrays or
model scores are inspected. It follows the accepted ten-update result under
job 14979295 and keeps its Attention architecture, 109,464,375 total / 15,960,211
trainable parameters, physical boundary correction and 43-source optimizer.

## Training

Start from accepted logical-update-14 `recovery.pkl` in
`runtime/outputs/training/conditioned-snow-free-continuation-14979295/`.
Run 20 additional updates, logical 15--34. Keep per-step train-only selection,
full 7/30-day streamed derivatives, 2% response calibration and scales 1..1/64.
Keep local and stage-cumulative development-monitor gates, and additionally
preserve cumulative transfer from the original corrected update-4 monitor.
The independent panel never chooses a direction, scale or checkpoint.

The materialized schedule is
`manifests/coarse_graining/conditioned_snow_free_continuation_v2.json`.
It draws from the existing prepared train-only pool using seed 20260910,
deterministic reshuffled pool passes and blocks of 40 distinct artifacts,
each split into five groups of eight. Existing schedule-generation code handles
the pass boundary without repetitions within a block. There are 800 exposures,
678 distinct artifact IDs; the repeat uses are explicit. No overlap with the
eight fixed development-monitor artifact IDs, no new Teacher production or v5.
This is not a new epoch over all v6 data and is not 678 distinct locations.

Per update: 8,320 direct day positions, 256 state anchors, 64 seven-day and
32 thirty-day windows. Full run: 166,400 direct positions, 5,120 anchors and
28,160 recurrent day positions. These are overlapping sample uses.
Summaries at new updates 5/10/20; each accepted step has its own recovery.
New run state is separate from the parent. Pending-monitor resume and scientific
stopping rules remain unchanged; no automatic paid resubmission is allowed.

## Independent validation

Prepare 24 windows from the accepted v6 validation splits: temporal (train
sites, validation years), spatial (validation sites, train years), and joint
(validation sites and years). Each stratum has four 7-day and four 30-day
windows; each cell has distinct locations and all 24 have distinct point-years.
Rank using seed 20260910 and metadata only, with first-day exclusion as in the
existing validation protocol. Exclude the union of point-years in all three
known prepared historical panels: target-oriented-validation-9b82b5a,
structured-gpp-fresh-validation-c3ea51a, successor-staged-validation-a2c2638.
No sealed test split is read. This is independent of training and these known
development panels; it is a small validation sample, not sealed final testing
or proof of globally representative long-horizon accuracy.

Evaluate the parent once and the final accepted continuation once, using the
existing canonical direct-GPP plus rollout evaluator with the corrected head.
The free preflight evaluates the parent; paid startup independently loads the
same parent and reproduces the fixed development monitor. It also evaluates
the independent baseline using the same run's evaluator before updating weights.
After training, compare to that baseline; no evaluation-driven training retry.

Independent relative admission requires overall science macro normalized RMSE
to decrease, every stratum/horizon macro regression to remain <= 2%, identical
paired counts, and zero physical failures. All key RMSE/bias values and the
direct Teacher-state GPP metric remain reported, including absolute gates.
This independent macro is the existing acceptance normalized-RMSE composite,
not the development monitor's MSE/bias loss. Do not mix their percentages.
Absolute acceptance is still required for model promotion, not merely retaining
a training checkpoint. If relative transfer fails, preserve the final accepted
training checkpoint as diagnostic and review generalization before extending.

## Engineering checks and resource proposal

Progress reports now write bounded small status files at most once per 15s;
complete reports remain saved at lifecycle boundaries, along with unchanged
pending and accepted model checkpoints. CPU-only replay on free test02 measured
the old event write at 0.0616s with one retained update and 0.4603s with nine,
versus 0.00542s/0.00244s for the new compact event write even without throttling.
Nine-update full report: 8,227,452 bytes; compact progress: 184 bytes. This
demonstrates an avoidable reporting cost; it is not a new GPU speed benchmark.

Before paid submission: local lifecycle/CLI/split checks; completed free CPU
preparation and full artifact readability; free GPU parent reload, unchanged
development monitor, and the independent baseline's forward/state/budget check.
No duplicate derivative audit or extra trained step is required for this
reporting/scheduling change. Measure actual writing time and hot update time in
the longer run. The last ten-step report suggests roughly 5--6.5h including
startup and evaluation, subject to first-use compilation and contention.

Proposed command, from `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX` on gln01:
`sbatch scripts/hpc/slurm_conditioned_snow_free_continuation_v2.sh`, with the
tested `EXPECTED_GIT_HEAD` set in the environment. Resources: gnall, one V100
32GB, one task, 8 Slurm CPUs, 32GB host memory, 8h maximum (470m inner timeout),
no requeue. Additive maximum: GPU CNY 17.60 + CPU CNY 4.48 = **CNY 22.08**.
The execution commit, command, free-check results and cost were presented;
the user authorized continuation after discussing final-method reproducibility.
See the admission record for submission provenance. This remains exploratory
training; final reproducible-method acceptance is a separate requirement.

## Completed preparation evidence

Execution code is `b6929153fc028330cbfc347f12565043a85e7aa6` on both checkouts.
The 38-test local regression set passes, including the full mocked 20-update
lifecycle, completed resume, real update-14 recovery loading, CLI parsing,
independent paired-count checks, and existing streamed-gradient numerical tests.

Free test01 prepared all 24 independent windows in 783.53 seconds, using
35,528,315 bytes. The selected panel contains 24 distinct locations and 24
distinct point-years, excluding 215 point-years from the three historical
panels. Preparation reads existing v6 data and does not run the Teacher again.
Local evidence is `outputs/conditioned-independent-validation-panel-20260909.json`
and `outputs/conditioned-independent-validation-prepared-20260909.json`.
The reporting benchmark is `outputs/conditioned-reporting-db6ff1f.json`.

The zero-update free gln01 GPU-1 admission passed at this execution commit.
It has restored the accepted update-14 tree, checked 686 existing training and
monitor files for readability, and reproduced the fixed eight-artifact monitor
at 534.5 seconds. All 24 independent windows completed, with zero physical
hard failures and zero trained updates, in 1162.95 seconds overall. Absolute
science accuracy still fails; independent transfer cannot be measured until
the paired post-training evaluation. See
[the admission record](conditioned_continuation_v2_admission_20260909.md).
Report:
`runtime/outputs/diagnostics/conditioned-v2-preflight-b692915/report.json`.
