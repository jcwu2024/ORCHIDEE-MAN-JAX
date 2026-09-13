# Current Project Status

Last updated: 2026-09-13.

This is the detailed status and evidence authority, not the shortest
onboarding page. Current work order is in [`NEXT_STEPS.md`](NEXT_STEPS.md),
code ownership is in [`CODE_MAP.md`](CODE_MAP.md), and document status is in
[`DOCUMENT_STATUS.md`](DOCUMENT_STATUS.md).

This page is the current status authority. Dated files under
`docs/source_audits/` and `docs/research/` are evidence snapshots and may
describe an earlier gate without being rewritten.

## Active Daily-Coarse-Graining Gate

CPU recovery15061586 COMPLETED00:32:48,exit0:0; estimated CPU chargeCNY0.3061.
Merge passes in20.58s:346539 distinct TRAIN days,17677 point-year payloads,
64 disjoint monitor days in4 artifacts, persisted mixed first batch read.
No Teacher regeneration/training updates. Combined catalogue is admitted;
fixed32 packing remains required before expanded GPU training. Local evidence:
outputs/one-day-expansion-15061225-merge.json.

Latest forward results:500 and1998 completed16 windows each,358.3/355.2s,
exit0; physical gates pass. Final1998 T7 passes all13 science keys and T30
passes11/13, failing only ecosystem/nonliving carbon. Earlier aggregate
"7/30 screen failed" wording must not be interpreted as both horizons or
all fluxes failing. T30 final GPP daily RMSE0.20125, NPP0.13638 gC/m2/day;
GPP cumulative RMSE5.1583 gC/m2 (500-step28.4665). AGB/BGB increment RMSE
2.9695/0.9921 gC/m2; nonliving increment bias-9986.91 remains severe.
Proposed next route couples physical loss-balance correction and a declared
mixed one-day/recurrent T7/T30 objective, with broader TRAIN cache preparation
in parallel. Native checkpointed AD is preferred; unchanged historic AD issues
are not the main gate. No new training has been implemented/submitted here.

GPU15061139 is now COMPLETED,1998/1998 updates,03:04:49,exit0:0. All updates
finite; fixed train and excluded64-day total/flux learning screens pass.
Consumed63707 unique days/63936 draws, one epoch of the existing pool.
Peak actual device allocation10674317824 bytes (9.941GiB), reserved pool
22151763456 bytes (20.631GiB). Estimated additive cost3.08028*2.76=CNY8.50.
Local evidence outputs/feedback-pool-15061139-report.json. All five declared
snapshots are saved/roundtrip-validated. No long-horizon acceptance yet.
Free gln01 T7/T30 evaluation launched for500/GPU0/Python98744 and1998/GPU1/
Python98816 using unchanged5502ce3;75min cap each. Both logs reached bootstrap.
Output prefix runtime/outputs/diagnostics/feedback-pool-15061139-forward-
with500/1998 suffixes. These are day-disjoint development evaluations, not
sealed spatial/temporal validation; no new paid submission.

Physical one-day review is mixed: excluded64 GPP/NPP/autotrophic respiration
RMSE improves from3.065/2.613/0.521 to0.577/0.413/0.171 gC/m2/day. Canopy
stock-change RMSE remains667.62, predicted mean-476.573 versus Teacher+7.720.
Litterpart physical RMSE worsens0.50005 ->0.70462 despite weighted loss decline;
all four predicted coordinates approach0.9964, including two target-zero ones.
This is demonstrated physical/objective imbalance, not a claimed AD failure.
Complete pending rollouts and inspect coordinate weighting before new training.

CPU expansion15061225 FAILED after00:52:24, exit9:0. Missing srun --wait=0
allowed cluster WaitTime=50s to kill remaining workers after normal first
exits. This is a launcher failure, not invalid Teacher data. Saved304786 of
346539 rows/15545 of17677 point-years;41753 rows/2132 point-years remain.
Saved compressed bytes7742125326; all recorded old-row comparisons exact.
Inventory: outputs/one-day-expansion-15061225-inventory.json. No final merge.
Original checkoutfd47de3 and worker files are preserved. Recovery9e8b1a0
uses explicit resume and --wait=0; proposed cnall2nodes*4CPU/noGPU/1h,
CNY0.56 maximum, now approved/submitted as15061586. Started20:35:26 scheduler
time onibc13b07n01--02; fixed recovery code9e8b1a0. See dated expansion report.
GPU15061139 completed; latest evaluation state is recorded above.

User proposes preparing all training data once. Recommended next data route
is an indexed reusable TRAIN cache,8568025 eligible days/23540 point-years,
preserving held-out/monitor exclusions and chronology for window sampling.
Sparse-row extrapolation suggests203GiB compressed/751 CPU-hours/CNY52.58;
full-point-year amortization, compression and actual random-read throughput
must be measured before a production quote. No full-cache job is approved.
User reports a10-million-file storage expansion ceiling; use shared multi-row
container files and consolidated indices. Expansion root read-only count was
15609 files just after recovery start (includes metadata and incomplete files,
not an accepted-row count); final data files17677, full TRAIN point-years23540.
This is not an account-wide quota inventory. Reuse the pool between runs.

Parallel data-expansion preparation: selected346539 unique TRAIN dates
(63707 existing+282832 new) across17677 point-years/535 points/44 years.
Actual free CPU pilot and two workers passed8 point-years/150 rows,128 new,
exact old-row payload equality and persisted reader readback. Full preparation
is not yet complete. Approved cnall4nodes*8CPU/2h, CNY4.48 max, job15061225;
isolated checkout one-day-expansion-v1. See
`research/daily_coarse_graining/one_day_expansion_v1_20260913.md`.
Current grouped sampler would produce13258 updates/4420 partial batches;
future expanded GPU training needs an explicit fixed32-row packing/cursor
and final-remainder policy. This is batch engineering, not failed gradients.

User approved the6h/CNY16.56 proposal. Job15061139 was submitted on gln01
and started on ibc13b03n01 at2026-09-13 17:39:37 scheduler time; observed
COMPLETED1998 updates,1GPU/8CPU/32GiB,6h cap. Fixed code5502ce3 in
runtime/checkouts/feedback-pool-v1; keep immutable. Log:
runtime/outputs/diagnostics/feedback-pool-15061139.log; report:
runtime/outputs/training/fresh-feedback-pool-15061139/report.json.
Next read the running free500/final1998 developmental rollouts.
Broader-data planning is next in parallel; the current63707 unique days
represent0.74354% of8568025 eligible TRAIN days (0.52182% of12208581 total
v6 transitions). Site/year presence does not establish dense date coverage.
See NEXT_STEPS for the proposed expansion trigger and actual-exposure rule.

The next fixed fresh-start pool epoch is specified and implemented with
unchanged calibrated losses,1998 full-network updates and retained milestones.
Local training/recovery and intermediate-checkpoint/feedback-report tests
pass. Actual target Bash expansions and metadata checks pass; test02 real-data
gradient/optimizer/monitor shapes pass224.161s at5502ce3, zero updates,
host peak4659789824 bytes. TRAIN63707 unique days, excluded monitor64 days
and disjoint rollout296 days/16 windows are verified; old-control sampling
and source match. Local copies: outputs/feedback-pool-v1-plan.json and
outputs/feedback-pool-v1-cpu-shape.json. Preparation is complete; next is
the now-approved gnall1GPU/8CPU/32GiB/6h,CNY16.56 maximum submission from
runtime/checkouts/feedback-pool-v1 at5502ce3, recorded as job15061139 above. Plan:
`research/daily_coarse_graining/feedback_loss_pool_learning_v1_20260913.md`.

Two-update admission completed on free GPU0 at497fe80 in2050.326s, exit0;
PID65767/Python66443 are absent. Both native JVP checks, both full-network
AdamW updates, all finite and physical hard gates pass. Fixed64 TRAIN-panel
GPP/NPP RMSE decreases0.119%/0.315%, AGB/BGB change RMSE decreases0.305%/0.369%,
canopy/litterpart RMSE decreases0.191%/0.442%; total autotrophic respiration
RMSE increases0.706%. This does not establish scientific or long-horizon fit.
Next freeze a bounded learning/retained-checkpoint protocol and prepare7/30
evaluation using the unchanged architecture/scales and declared loss weights.
Detailed report: `research/daily_coarse_graining/feedback_loss_two_update_results_20260913.md`.
No active test, checkpoint from this probe or paid allocation.

Update: the zero-update loss calibration below has completed at ac5aa51 in
1063.644s, exit0, four batches, finite losses/gradients and zero hard counts.
Root declared trial coefficients base1/canopy3e-9/litterpart2.5e-5 from its
train-only gradient balance. Two-update native-AD/JVP runner and later7/30
acceptance plan are prepared, with local tests passing; real deployment
admission is next. No new trained model or paid job exists. See
`research/daily_coarse_graining/feedback_loss_calibration_results_20260913.md`.

Leaf follow-up completed at a952bb1 in646.28s, all16 windows, exit0, all controls
valid and hard counts zero. Litterpart alone gives95.1045% T30 excess-MSE rescue
and78.6935% reciprocal damage; rescue RMSE0.216585 and bias+1.1335%, versus
free RMSE0.871443/bias-31.5805%. This remains fixed-weight input mediation,
not trained-model acceptance. Actual input scale0.00789091 versus output/state
loss scales1 permits roughly40-standard-deviation input errors at typical
predicted reset values0.67 instead of Teacher1. Full litter+DOC rescue98.1422%
and damage94.2508% confirm an additional interaction; they do not sum additively.

Next: optional input-scale litterpart MSE and source-direct canopy stock-change
supervision, followed by fresh training-only gradient-balance calibration and
bounded optimizer admission. Keep Attention and input normalization unchanged.
See `research/daily_coarse_graining/feedback_and_canopy_loss_amendment_20260913.md`.
The first calibration is zero-update on a free GPU; no new paid training.
Optional shared losses and local tests are complete. Actual batch4 CPU shapes
pass151.33s at ac5aa51; the free GPU0 runtime-preflight/calibration chain is
launched as PID61046 in `runtime/checkouts/trained-feedback-v1`,60m calibration
cap. Output prefix `runtime/outputs/diagnostics/feedback-loss-calibration-v1`.
No new weights exist; read the gradient components before choosing coefficients.

Preceding group-level evidence:

Group intervention is now completed:905.16s,16/16 windows, all controls valid
and all32 arms' hard counts zero. T30 litter_metadata teacher replacement
removes95.314% of excess GPP MSE; reciprocal replacement restores78.301%.
Rescue improves7/8 windows, damage worsens8/8. GPP RMSE0.87144 to0.21293,
bias-31.58% to+1.34%. This is a same-day hybrid-input diagnostic, not a repaired
rollout. DOC rescue43.46% remains relevant to interactions; canopy stock
rescue0.126% separates that inventory problem from dominant GPP mediation.
The nine-leaf/family diagnosis is completed above. No architecture change or
paid training. Older launch logs below are history.

Latest operations: a136fda free GPU0 state-intervention chain PID57120 is
running under runtime/checkouts/trained-feedback-v1 (75m diagnostic cap;
72.1s runtime preflight passed). All16 actual-data CPU shapes passed100.8s.
The initial archive identity comparison incorrectly rejected matching NaNs;
independent no-model inspection found identical finite values and matching
NaN positions in all16 windows. equal_nan=True fixes comparison only.

The test02 canopy raw-v6 readback completes all240 T30 days: start, fast/end
and archive coordinates match exactly. Teacher outgoing targets all zero,
predicted positive-start outgoing mean0.00813259 and MSE7.5157e-5; incoming
normalized field MSE1.07546, Teacher incoming mean4.54273gC/m2/day.
Global train/train state_delta scale5.40393 versus stock scale47559.69
supports a separately weighted stock-change objective proposal. This does not
yet prove a repaired training method or that canopy drift mediates GPP loss.
Sol/Terra supplied initial implementations; root completed integration and
raw endpoint readback after subagent provider interruptions.

Current priority: use native checkpointed AD for the next specified training
objective and verify that objective plus a few complete updates. Gradient
infrastructure is no longer the main task; do not reopen passed unchanged
probes. Scientific work now isolates GPP state feedback and canopy label/loss
geometry with Sol(high)/Terra(high) assistance, without architecture changes.

Source correction: interception_storage is already land-area organic carbon.
The research stock extractor and host reducer now remove its second cover
factor, following soilcarbon_leak pool_start/end. Historical reports below
retain their original weighted convention. Teacher PFT14 cover is exactly1
on the selected T30 panel, so the large canopy decline remains real. Archived
Teacher-reset fast change averages-275.061gC/m2/day against Teacher+4.543;
predicted incoming/outgoing average0.159/275.219. Its frozen PFT14 state scale
is47559.69gC/m2. Typed-label and same-day GPP mediation checks are the next
bounded free-node operations. See the three new active20260913 documents.

Latest completion: both free chains finished with exit0 and absent launcher
PIDs. Native T30 directional gate passes: reverse/JVP projection errors0.1190%
and0.0637%, within declared1% tolerance, but larger than the earlier T7 errors.
All109,464,375 parameter gradients finite; reverse norm2.29535, active peak
2.689GiB. Reverse2393.4s (compilation2154.5s), verify680.1s. Scope remains the
single selected trained window/scalar/two parameter directions, not FD/D2.

Forward attribution16/16 completes468.2s with zero reported science-invalid or
hard-failure counts. T30 daily Teacher-reset GPP/NPP RMSE0.10116/0.06116 versus
free0.87144/0.61604. Cumulative reset biases-0.179%/+1.110%; free biases
-31.581%/-68.727%. Reset fitting is markedly better, although the different
T7 panel retains reset RMSE0.40614/0.30269; this is not global fit acceptance.
Canopy interception organic-carbon endpoint error11203.73gC/m2 explains
92.972% of nonliving error12050.67gC/m2. Its mean predicted30-day fast change
is-11067.45gC/m2; retained-tail change is zero, as for the other four inventories.
This locates the carbon error, not yet the causal GPP feedback state.

Canonical saved GPP reproduces exactly; independent carbon-pool sums agree
with shared extraction within2.91e-11gC/m2. Daily diagnostic replay is not
bitwise/state-allclose equivalent: raw-state max discrepancy0.05011, pool max
0.004708gC/m2 and raw GPP max3.47e-6. These are too small to explain measured
large stock losses, but do not certify all-state replay equality. Next use saved
states for targeted feedback attribution and canopy label/loss audit; no new
architecture or paid training. The following execution paragraphs are history.

Current execution after user approval: native T30 parameter AD is running on
free GPU0, PID100003, main checkout688e14c. Runtime preflight passes69.50s;
the real T30 forward passes hard checks in232.6s and reverse compilation is
active. GPU1 forward attribution launched as PID105072 at isolated a16ea78
`runtime/checkouts/trained-forward-attribution-v1`. Real CPU diagnostic shapes
and independent pool extraction passed199.3s at fcbb035; subsequent reporting
changes preserve the numerical graph. Three local accounting tests, actual
shell dry-run and independent code/source review pass. No new training or paid
job. Keep both running checkouts fixed. Read the T30 reverse/verify reports and
`trained-forward-attribution-v1.json` before further execution.

Source review finds no obvious carbon stock-unit/double-counting error. The
neural inventory representation labels decreases as external outgoing and
increases as external incoming, with no internal transfer. Thus its zero budget
residual is not ecosystem closure tied to GPP/Ra/Rh. The numerical attribution
will measure actual incoming/outgoing per pool, fast versus retained changes,
and Teacher-reset versus free-running GPP before assigning a causal mechanism.

Completion update (2026-09-13): both c76c553 chains below have finished with
exit0; their launcher PIDs are absent. Forward:16/16 windows,708.6s, complete
metric populations and zero physical hard failures, but neither science
horizon passes. Mean30-day Teacher/model cumulative GPP52.356/35.822,
NPP17.243/5.392 and autotrophic respiration35.114/30.430 gC/m2. Mean AGB
change+1.173/-7.423 and BGB+0.153/-2.990 gC/m2. Reported nonliving-carbon
change+125.270/-11925.404 gC/m2 needs constituent/extractor attribution before
assigning a mechanism. Stock-level acceptance alone is misleading here.
The panel is disjoint from training dates, not an independent-site/year test.

Native trained T7 parameter reverse/JVP directional gate passes: all109,464,375
gradient entries finite, norm7.55134; two projection absolute errors7.472e-6
and1.471e-6 (relative5.073e-6 and1.430e-6). Reverse/JVP stages take2180.8/
667.0s, including compilation; reverse active GPU peak2.683GiB versus20.631GiB
reserved pool. This is scoped evidence, not full multi-day AD admission or
finite-difference certification. Next: attribute carbon/GPP forward errors and
extend native parameter-AD evidence to the intended T30 objective. No new paid
job is authorized. The following launch descriptions are historical.

Current2026-09-13 execution: the final update1998 checkpoint is under a
train-domain, training-day-disjoint forward screen (8 T7+8 T30 windows,
14 locations/7 years/296 days). Selection excludes63,771 distinct one-day
training/monitor days and precedes model evaluation. No sealed split is used.
The metadata selection passed in148.18s and real CPU forward/metric-payload
tracing passed in141.17s. The c76c553 reporting follow-up enforces complete
panel populations and no filtering-away of defined nonfinite results;11 local
selection/layout/metric tests pass. Free GPU1 runs forward, launcher PID4617.

The first selected T7 window (artifact1887/window5, point277.0-091.0/year1995,
start-row73) is also the fixed native-checkpointed parameter-AD probe, with
two predeclared exact +/-1 Rademacher directions. CPU combined AD tracing hit
its15-minute limit without a final result; actual GPU lowering is the next
shape/memory evidence after exact-runtime interface checks. Free GPU0 runs
native reverse then JVP on success, PID6852. The source checkout is c76c553
for both chains and remains fixed. Reports/logs share the prefix
`runtime/outputs/diagnostics/trained-forward-ad-v1-`. See
`trained_forward_and_native_ad_20260913.md` for bounds and claim limits.
No parameters are updated and no paid job is submitted by either chain.

Latest completion: paid15048786 completed1998/1998 updates with exit0 in
02:24:59, consuming63,936 draws/63,707 unique days. The bounded learning screen
passes. Final excluded-day RMSE is GPP0.59343, NPP0.42091, maintenance0.05530,
growth0.16191, autotrophic0.18036 gC/m2/day; AGB/BGB one-day changes0.30753/
0.12010 gC/m2. The64-date panel spans44 locations/4 years, not independent
sites or a free-running trajectory. Late1500->1998 mild regression and negative
bias remain material. The final checkpoint is retained by the declared rule.

Both free diagnostics finished without process exceptions. Native checkpointed
AD uses2.27GiB estimated compiled buffers but fails the old direct-reference
comparison (46 state-coordinate failures; parameter leaf failures despite no
parameter-coordinate failures). At reserve coordinate2250, checkpointed
1470.28568 and JVP1470.29254 agree much more closely than either agrees with
direct reverse1469.15114. Along the parameter-disagreement direction,
checkpointed0.02122033 is also closest to JVP0.02122122. This supports further
verification of native checkpointed execution, not immediate full AD admission.
FD has no reliable reserve-coordinate plateau.

Original directional report projection/replay NaNs were diagnostic masking
defects on fixed undefined states. The code is repaired and6 tests pass.
Projection recovery at7577899 completed in379.49s using regenerated initial
parameters and saved gradients/epsilons, with zero rollout/JVP/gradient
evaluations; `public-rollout-T2-v2-projections-recovered.json` is finite and
retains the original FD/JVP observations. Original raw reports are unchanged.
Replay's old global norms/counts remain excluded from conclusions where NaNs
contaminated them. Full results:
`fresh_one_day_epoch_and_gradient_results_20260912.md` in the research docs.
No training or diagnostics remain running from these jobs. No new paid job
has been authorized. Next priority is a disjoint7/30-day forward screen of the
trained final checkpoint plus native-checkpointed numerical verification.

The execution notes below preserve the chronology of those completed runs.

Current execution: paid one-day job15048786 continues, while two independent
free-GPU diagnostics investigate the saved T2 mismatch. GPU0 checkpointed
comparison runs at6b69128/PID38864; GPU1 replay plus joint-JVP/FD runs at
isolated4047130/PID44184. Logs/reports:
`runtime/outputs/diagnostics/public-rollout-T2-v2-{checkpointed,directions}-diagnosis.*`.
The native/streamed acceptance gate is still unresolved. Neither diagnostic
updates parameters or changes model/input-normalization rules.
The first two directions launch attempts stopped after the small runtime gate,
before model computation: Bash4.2 rejects a nounset empty argument array, and
the old starter invoked the main-checkout launcher despite a different child
CODE_ROOT. The argument array is now always nonempty and the starter invokes
the absolute launcher inside CODE_ROOT. Both command branches pass actual
shell `--dry-run` expansion. Previous logs are preserved with attempt1/attempt2.

The actual GPU runtime matrix passes in59.73s on JAX0.4.38/NumPy2.4.6;
local JAX0.10.1 alone did not cover the earlier scalar-float0 AOT failure.
The gate now runs before expensive diagnostics. The new joint-JVP actual-data
CPU shape trace passed at6b69128 in220.62s. It measured the dominant coordinate
as PFT14 reserve carbon biomass[0,1,7,0]: value9.98185347e-9,
mean4.24204611e-9, scale4.66846019e-9, float32 standardized1.2294861078 versus
float64 arithmetic1.2294861953. This alone does not prove a normalization bug.
Source review confirms reserve is a prognostic state, with negative-only repair
to1e-8 and subsequent possible positive depletion; it must not be zeroed.
At its fitted input scale, the native derivative is6.85867e-6 and the native/
streamed difference7.22169e-9 per standard-deviation perturbation. This change
of units is diagnostic only and does not amend acceptance. Provenance is in
`reserve_carbon_gradient_case_20260912.md`.

Paid training's update500 fixed excluded-day panel improves GPP RMSE
3.06484->0.67214, NPP2.61346->0.47197, autotrophic respiration
0.52105->0.21476 gC/m2/day, and AGB one-day-change RMSE2.01949->0.35810 gC/m2.
Training has continued beyond950 updates at the latest read. The panel excludes
these days from updates but shares the train-domain locations/years; no
independent-space/time or long-rollout scientific claim follows.

The chronological evidence below preserves earlier completed/failed attempts.

Latest: v3 completed all12 updates in757.37s with convergence/physical/replay
checks passing and20.9% lower three-sample flux loss. Fraction outputs briefly
overshot at update1 and recovered by4; this is not final-model acceptance.
The current authorized work is parallel one-day data expansion and public
multi-day AD infrastructure. Native/checkpointed/streamed local recurrence
tests pass, including initial-state and PFT-fraction routes. A real two-day
full-network comparison is being prepared. The2000-package one-day catalogue
completed on free test02 in149.20s:63,707 unique Teacher days,535 points,
17,677 point-years and44 years, excluding64 monitor days. This is availability,
not actual training coverage. The corrected real two-day CPU direct/checkpointed
gradient shape probe passed at b7fd1bc in477.36s. Review identified missing
Teacher GPP validity and physical/coordinate comparison coverage in the probe;
those are corrected at4410c41. Seven probe/canonical/batch tests and21 public/
legacy-streamed regressions pass; batch sampling/cache tests also pass.
The batch4 real CPU shape gate passed at a49643f in662.2s. Its free GPU0
probe completed at4410c41 in2144.28s: all physical checks and all109,464,375
gradient-coordinate comparisons pass, gradient L2 difference6.61e-7. Actual
peak7.85GiB versus20.63GiB pool; hot batch4 gradient0.400s. Two separate
gradient compilations consumed957.90/953.13s. Report
`runtime/outputs/diagnostics/one-day-batch4-v1.json`.
The updated public T2 CPU shape gate passed on test02 at4410c41 in275.3s;
report `runtime/outputs/diagnostics/public-rollout-shapes-v3.json`.
The public T2 numerical comparison started on free gln01 GPU1 at13:10 CST,
launcher PID110317,90-minute cap; report
`runtime/outputs/diagnostics/public-rollout-T2-v1.json`. It has now exited at
the configured90-minute boundary; PIDs110317/110360 are absent. The report
still says running/compile_checkpointed because timeout produced no final JSON.
Direct AD completed at4126.9s: all109,464,375 parameter and3854 initial-state
gradients finite; every hard count zero; loss0.00384148725; compiled buffer
estimate5,699,182,032 bytes (5.31GiB). The checkpointed/streamed comparison and
JVP/FD did not finish. This is an incomplete timing-limited probe, not a
derivative mismatch or multi-day acceptance. Direct compilation alone took
about62.8minutes. The approved staged completion is now implemented and
independently reviewed;7 focused tests pass. It is running on free gln01 GPU1
at4d128c3, launcher PID19594, using separate core/directional/checkpointed
stages and durable gradient/epsilon artifacts. Limits90/60/90minutes are stop
limits, not duration estimates; failed/limited stages stop the chain.
Log `runtime/outputs/diagnostics/public-rollout-T2-v2-pipeline.log`; separate
stage reports use `public-rollout-T2-v2-{core,derivatives,checkpointed}.json`.
The first staged attempt saved reference.pkl (418MB), with reference compile
592.995s and evaluation0.806s, then failed at1770.9s on scalar numpy.void
cotangents in the JAX0.4.38 AOT call. The array-preserving float0 slice fix at
8e522ce passes17 local tests plus an exact-runtime tiny GPU reproduction and
repaired streamed/direct comparison. Report `rollout-float0-compatibility-v1.json`.
The chain resumed on free GPU1 at8e522ce, PID31604, reusing the saved reference;
original attempt report/log are preserved with `-attempt1`. Cross-method
core completed in326.3s with matching loss and finite gradients, but the
comparison is rejected. Four parameter and49 initial-state coordinates exceed
the declared tolerance; parameter leaf L2 gates also identify distributed
differences. Overall relative L2 error9.3602e-5/1.05105e-3 respectively.
The chain stopped before directional/checkpointed checks; PID31604 is absent.
Saved-array CPU inspection identifies the four parameter coordinates in
expert_blocks/5/0/attention/query/bias (indices193/216/219/232, absolute error
about1.09e-6--1.23e-6). Full reference/streamed gradients are durable; inspect
recurrence/precision and targeted directional checks before drawing a cause
conclusion. This is not yet proof of either harmless rounding or an incorrect
adjoint. The separate single-day direct-AD training remains independent.
The user asked to advance independent work instead of waiting. The broad
one-day runner, weighted gradient accumulation, atomic full-state checkpoint
and fixed physical monitor summaries are implemented. Eleven local tests pass,
including the actual runner's interrupted/resumed AdamW equality and sample
cursor behavior. The real CPU gradient/optimizer/monitor tracing passed at
fa05bc1 in277.7s. Recovery metadata/count/report-order fixes and host-memory
reporting at12bac94 preserve that numerical graph; focused tests pass.
The free GPU0 two-update technical smoke completed at12bac94 in455.40s with
exit0, both updates and technical gate passing. Second full32-row update3.956s,
peak active GPU9.49GiB and host8.14GiB. The two-update fixed-panel loss screen
also passes, without implying one-epoch learning or scientific acceptance.
Report `runtime/outputs/diagnostics/one-day-pool-smoke-v1.json`.
Server clock was about
8m16s behind the local workstation; prefer elapsed durations to mixed clocks.
The user approved fresh one1998-update prepared-pool-epoch training,
batch32/microbatch4, one V100/8CPUs/32GiB/4h, CNY11.04 maximum (CPU2.24+GPU8.80).
Job15048786 was submitted from gln01 and started immediately on ibc13b02n01,
server2026-09-12 15:46:03. scontrol confirms the approved resources and time.
Frozen code:12bac94 in `runtime/checkouts/one-day-pool-v1`. Startup log confirms
that checkout and catalogue phase. Report:
`runtime/outputs/training/fresh-one-day-pool-15048786/report.json`; log:
`runtime/outputs/diagnostics/one-day-pool-15048786.log`.
This is the approved one-day training job, not multi-day training or final
model acceptance. See `fresh_start_one_day_pool_v1_20260912.md`.
At the first40-update report, updates2--40 average4.113s each (3.883--6.656s),
after about381s of initialization/compilation/initial monitoring. This supports
roughly2.3hours of update arithmetic for the current1998-update pool. Naively
scaling the unchanged single-V100 method to8,568,025 eligible days would take
about306hours per full-data epoch, before extra IO/monitor costs. Full-data
training therefore needs throughput/batch/multi-device measurements rather
than blindly multiplying the present run or its4-hour stop limit.
See the linked current plan in `NEXT_STEPS.md`; prior chronological entries
below retain their original scope.

Current step: v3 convergence-admitted fresh-start learning. The seven-scale
audit completed in2266.39s; reverse/JVP/FD agree at3e-4,1e-4,3e-5,1e-5 with
all masks/discrete/hard counts valid. Large-step differences are dominated by
nonlinear state/process responses, including deeper lignin fractions moving
near1 on the negative perturbation. No model update occurred. Next execute
[`v3 bounded learning`](research/daily_coarse_graining/fresh_start_convergence_learning_v3_20260912.md)
with unchanged model/data/loss/optimizer, convergence-based admission and
explicit fraction monitoring. Paid job15005758 remains canceled.
The v3 integration passes seven focused tests, independent review and real-data
CPU shape tracing (208.4s). It is running free on gln01 GPU1 at0a3495e,
PID11826, with60-minute cap. Active report:
`runtime/outputs/diagnostics/fresh-start-learning-v3.json`. Training results
remain pending; admission and12 updates are in the same process.

Latest decision (2026-09-11): the paper method should start from fresh neural
initialization of the consolidated Attention architecture, with full-network
optimization as the target. Existing exploratory checkpoints are diagnostic
assets, not a required final weight source. The historical 15.96M-only scope
must not become permanent merely because other modules were pretrained.
`AGENTS.md` records the decision. Next build the clean initialization/training
entry point and bounded all-parameter numerical/resource/learning admission in
[`fresh_start_training_route_20260911.md`](research/daily_coarse_graining/fresh_start_training_route_20260911.md).

The fresh-start prototype is implemented and local actual full-network
initialization/process-loss/gradient/bound-day tests pass (336.45s). The
combined real-data science/state/process objective failed on its first trace
at cdf7eb3 after 1064.4s, zero updates: PFT fraction/mask lacked a single-day
axis. cb3c454 fixes that caller and passes a focused physical-extractor
regression. Free test02 complete-loss shape preflight passed in 218.69s, report
`runtime/outputs/diagnostics/fresh-start-one-day-shapes-v2.json`. The corrected
free gln01 GPU 1 run at cb3c454 completed all 12 updates in 2106.65s; report
`runtime/outputs/diagnostics/fresh-start-one-day-v2.json`. Its single-day
reverse/JVP/FD and hard gates pass, as does checkpoint replay. Mean flux loss
falls 16.14%, with mixed individual physical errors; all updates clip strongly
and state heads dominate the gradient. Next perform the fixed-sample
[`loss-balance audit`](research/daily_coarse_graining/fresh_start_loss_balance_audit_20260911.md)
before writing any training amendment. The original failure report is
`runtime/outputs/diagnostics/fresh-start-one-day-v1.json`. This is three fixed
train samples, not a full training epoch, final protocol or generalization test.

The field-level loss audit completed at 9a6edb6 in 572.14s, no updates. Report:
`runtime/outputs/diagnostics/fresh-start-loss-balance-v1.json`. It localizes the
dominance to bare-soil gsmean (~2769 versus Teacher0), 99.75-99.77% of total
loss and 99.99891% of squared gradient norm. All matched primals/hard checks
pass. Source-backed gsmean applicability is the next correction, keeping DOC
conditioning and fresh full-network initialization. Three explicitly authorized
Sol/max agents handle source audit, code fix and independent review; root owns
the v2 protocol and integration. No paid job or new trained checkpoint yet.
All three delegated tasks are complete and the correction is integrated.
Six focused tests, real-data CPU shape checks and actual CPU numerical forward
pass. Initial mean state loss falls to 0.480548 through exact bare-soil gsmean
materialization, while initial physical fluxes/stocks match the earlier run
within tolerance; all hard counts are zero. This is not new training.
The v2 GPU gradient/12-update screen remains pending because both free GPUs
were occupied at the last check. Fuel/dead-leaf source ownership forbids a
blanket zero-by-name extension; see the source audit linked from the v2 protocol.
The latest resource check also finds no paid GPU available (all ten gnall
nodes report GresUsed8/8, including the CPU-idle node). The user permits
immediate paid submission only when a GPU is available and requests discussion
before queueing otherwise. A one-V100/eight-CPU/one-hour Slurm option is ready,
maximum CNY2.76. The user approved queueing and job 15005758 is now submitted
at executable checkout 85e6b39. Initial state PENDING/Priority; scheduler estimate
2026-09-13 12:51:52, subject to change. No training has started. Track its
job-specific log and `runtime/outputs/training/fresh-start-gsmean-v2-15005758/report.json`.
The next check found both free GPUs empty. Root held paid job15005758
(PENDING/JobHeldUser, not canceled) and launched the same v2 probe free on GPU1,
source85e6b39, PID107826. Active report is now
`runtime/outputs/diagnostics/fresh-start-gsmean-v2-free.json`, with60-minute cap.
The paid job will not start while held. No bigger training budget or additional
job was submitted; future paid work should amortize queue/compile time across
declared admission and substantive training stages.
The free v2 run completed with an admission stop after2005.78s, zero updates:
gradient norm0.63538, finite full tree, matching reverse/JVP and FD at3e-4,
but discrepant FD at1e-3. Investigate fixed-scale convergence and mask/branch
changes before amending admission. Paid job15005758 was explicitly canceled
by user request; scheduler confirms CANCELLED and RunTime00:00:00.
The active work is
[`fresh-start FD convergence`](research/daily_coarse_graining/fresh_start_fd_convergence_20260911.md).
Its five local tests and CPU shape preflight pass. Free GPU1 run at35e608a,
PID908, is active with a45-minute cap. Report:
`runtime/outputs/diagnostics/fresh-start-fd-convergence-v1.json`. No model
updates, new checkpoints or paid jobs in this diagnostic.

The all-state and DOC-only T=16 input-tail tests completed in 2459.7s and
1838.3s on free gln01. Both pass the three tested numerical/mask/physical-hard
gates but fail old-weight science regression gates. DOC-only improves the
three cases' NPP/AGB/BGB absolute errors while maintenance worsens; flux-error
compensation is possible. Neither is an accepted trained correction or a
real-model derivative validation. Both jobs have ended. Result paths and
physical-unit evidence are in
[`conditioned_doc_tail_v1_20260911.md`](research/daily_coarse_graining/conditioned_doc_tail_v1_20260911.md)
and the linked all-state report. Earlier records below retain their historical
meaning; no paid continuation or final method has been accepted.

The rejected mechanistic-output v2 isolated diagnostic remains the comparison
point: litter/POC/DOC fell `91.8014%` and passed only `16/28` dominant fields.
The user-approved `hierarchical_coordinate_query_attention_v3` repaired the
coordinate mapping and completed its bounded 100-epoch run under job
`14619803`. Loss fell `98.5804%`, all hard and gradient-coverage gates passed,
and `25/28` dominant fields passed. The result is a substantial improvement
but misses the frozen `99%` aggregate gate. DOC incoming amount
(`-0.3085%`), free-to-adsorbed transfer (`83.2158%`), and advective transfer
(`93.5753%`) remain unresolved, so no checkpoint was retained.

A zero-training audit on the same 364 frozen v6 days shows that all three
failed targets are active on every day, although only `9.88%` to `29.42%` of
their coordinate cells are meaningfully nonzero. Their daily normalized
intensities are strongly related: absolute daily-mean correlations are
`0.9597` for incoming versus adsorption and `0.8892` for incoming versus
advection. Per-coordinate constant means still leave `54.49%` to `91.00%` of
zero-predictor squared error. The likely remaining bottleneck is therefore
the model's pooled handoff between DOC stages: it does not provide
coordinate-aligned predicted transfers to the inventory endpoint decoder.

The user approved `coordinate_coupled_doc_transition_attention_v4`, and its
local foundation is complete. All 896 DOC inventory coordinates receive five
upstream reaction/transport tokens aligned by physical PFT identity, layer,
pool, and element. One shared signed transition decoder replaces the two
independent incoming/outgoing heads and materializes mutually exclusive
conservative endpoints. The model has `59,971,257` parameters and preserves
the unchanged daily boundary, conservative updater, retained tail, and all
non-DOC paths. The focused v4, runner, unchanged v3, and unchanged mechanistic
suites pass `48` tests in isolated processes; static checks also pass.

Paid job `14627588` completed the v4 100-epoch, 4,600-update diagnostic
normally in `05:30:23`. Aggregate litter/POC/DOC loss fell `99.1353%`, which
passes the aggregate gate for the first time, and all execution, gradient,
state, budget, inventory, count, and finite-value gates passed. The candidate
still fails the frozen field gate at `24/28`: DOC incoming improves strongly
to `90.5487%`, but DOC outgoing becomes `10.62` times its initial loss;
free-to-adsorbed and advective reductions are `85.1634%` and `92.6142%`.
No checkpoint was written. The coordinate-aligned handoff is useful evidence,
but the joint signed endpoint head is rejected. See
[`coordinate_coupled_doc_transition_v4_design_20260820.md`](research/daily_coarse_graining/coordinate_coupled_doc_transition_v4_design_20260820.md)
and
[`coordinate_coupled_doc_transition_v4_memorization_result_20260820.md`](research/daily_coarse_graining/coordinate_coupled_doc_transition_v4_memorization_result_20260820.md).

The following free-CPU, zero-training endpoint geometry audit passes integrity
on all `326,144` coordinates. The joint inverse reconstructs every finite
target to `8.88e-16`, and incoming/outgoing median normalized Jacobians are
balanced at `0.4730` and `0.5340`. The decisive asymmetry is the supervised
loss: incoming supplies `97.6209%` of the two fields' combined zero-predictor
squared error and outgoing only `2.3791%`. This gives the shared scalar a
strong incentive to repair incoming while sacrificing outgoing. The joint
head also has 110 active Jacobians at or below `1e-6`. Reverting to v3's
separate transforms is not sufficient because they have 563 such active
coordinates. Status plus branch-native loss-scaled magnitude has finite
targets throughout and closely matched active medians (`0.01843` versus
`0.01838`), so it is the accepted basis for v5.

The frozen
`coordinate_coupled_doc_status_branch_magnitude_attention_v5` implementation
retains v4's coordinate-aligned Attention and changes only the final endpoint
decoder and DOC loss. It emits incoming/unchanged/outgoing logits and separate
incoming/outgoing magnitude coordinates, uses differentiable soft status
gates, and preserves the conservative updater and retained exact tail. The
model has `59,973,309` parameters. Focused local forward/loss/gradient,
canonical binding, one-day state/budget, inactive-PFT, and strict permutation
tests pass; the runner and launcher now select its loss and report schema
explicitly.

Paid job `14632388` completed the V5 10-epoch screen normally in `01:14:07`.
All 460 updates were finite, selected gradient/update coverage passed, counts
were unchanged, and every hard count was zero. Epoch 5 passed. Epoch 10 failed
the frozen absolute gate: litter/POC/DOC fell `83.1178%` instead of `90%`, DOC
incoming fell `29.6589%` instead of `50%`, and DOC outgoing fell `26.1806%`
instead of `50%`. Adsorption and advection improved `16.6932%` and `58.9182%`.
No checkpoint was written.

Status and the two branch-latent losses fall `75.4655%`, `78.4438%`, and
`54.2036%`, while the materialized physical endpoint term falls only
`29.2932%`. V5 therefore fixes v4's incoming/outgoing sacrifice but does not
convert its learned latent variables into sufficiently accurate physical
endpoints. The coordinate-aligned source bank remains accepted evidence; V5's
materialization head remains rejected. See
[`doc_status_branch_magnitude_attention_v5_early_screen_result_20260821.md`](research/daily_coarse_graining/doc_status_branch_magnitude_attention_v5_early_screen_result_20260821.md).

The following target/output audit replaces the old field-by-field gate as the
current operation authority. Of the 161 learned fields, 2/26 scalar
coordinates are direct primary science outputs, 93/3,682 are state/budget
essential, and 66/35,298 are auxiliary supervision. Twenty complete-boundary
fields remain exact/diagnostic. The frozen target-oriented objective combines
an equal-group one-day state anchor, independent-basis 7/30-day science
losses, and a 0.1-weight necessary-process term. It avoids double-counting
NPP, NEP/NEE, total live biomass, and ecosystem-carbon identities while still
requiring their absolute 7/30/365-day gates. Focused contract, hierarchy,
identity, objective, and weighting tests pass `6/6` without Teacher data or
training.

The user-approved successor is now implemented as
`target_oriented_hierarchical_coordinate_attention_v1`. It retains the
source-routed 384-wide encoder, replaces the old uniform label heads with 12
domain-refinement blocks, five cross-domain science tokens, separate
high-capacity primary/state coordinate decoders, and a 192-wide auxiliary
bottleneck with label-shared heads. The fresh path does not use v4's joint
signed endpoint or v5's status-times-magnitude materializer. It has exactly
`83,755,173` float32 parameters.

The focused local gate passes `10/10` with the frozen target contract: all 161
learned fields and 39,006 coordinates are finite in eager/JIT execution;
primary, state, auxiliary, domain, and science-token paths receive finite
nonzero gradients; inactive-PFT perturbations are exactly isolated; the full
learned boundary is PFT-permutation equivariant within float32 tolerance; and
canonical binding preserves the one-day water/carbon budget tolerances.

The following zero-update real-v6 GPU admission also passes on one free
`gln01` V100 at commit `7c6371c`. The complete 181-field forward is finite on
all 6,674 declared-defined values; all 83,755,173 gradient elements and the
seven required architecture families are finite and nonzero; the canonical
next-day transition is finite; and water/carbon budget residuals are exactly
zero. Hot JIT forward replay is `0.06197 s`. JAX peak live device memory is
approximately 2.49 GiB; the approximately 25 GiB seen in `nvidia-smi` is the
allocator's reserved pool. No update or checkpoint occurred. Report SHA256 is
`a92e9415...ad13803b`.

The first bounded training design is now frozen in machine contract
`target_oriented_bounded_training_v1` at SHA256 `91faf011...ea25b`. Each update
draws 28 distinct point-years and landpoints from one runtime year, while
eligible years rotate deterministically: 32 one-day samples from 16
point-years, eight seven-day windows, and four thirty-day windows, for 208
Teacher-day usages per update. GPU microbatches are `4/1/1`; exact sufficient
statistics preserve the full-batch MSE and signed-bias objective before one
AdamW update. The multi-point prepared reader gives every sample or window its
own retained-tail arrays, while a common static scaffold keeps compilation
valid. Synthetic full-batch versus accumulated gradients and focused prepared
artifact tests pass. Gradient clipping is calibrated from eight train/train
full-objective norms rather than assuming threshold 1.0. The bounded
milestones are 250/1,000/3,000 updates, and scientific promotion remains
absolute rather than control-relative.

The eight batches and first free-V100 full-step preflight are complete. Peak
live memory is about 8.15 GiB and a hot effective step is about 35.65 seconds,
but every full gradient is NaN. A component audit localizes the failure to the
one-day state anchor: one near-constant DOC coordinate has scale
`7.960966e-12`, creates loss around `1e14`, and is duplicated through the
complete `other` state group. Learned process outputs, 7/30-day science loss,
state transitions, budgets, and stocks are otherwise finite. The exact
`roughheight_pft` diagnostic supplies the sole structural nonfinite complete
output, and all state-mask mismatches are declared dynamic `rveget` status.

The frozen amendment makes `other` residual, applies the existing native-unit
fallback to the 46/3,854 state scales at or below `1e-6`, makes masked
arithmetic backward-safe, checks nonfinites only on learned outputs, and keeps
declared-dynamic `rveget` supervised/report-only. The representative real-v6
probe completed under job `14644566` at commit `d1eda1b`. It removed the
approximately `1e14` DOC domination and kept all learned outputs, states,
budgets, discrete state, and stocks valid, but all three gradient sources
remained NaN. The report SHA256 is `3575cfae...f3f2aee`.

The shared masked standard-deviation summary did contain a real `sqrt(0)` AD
defect, and commit `5a50a65` repaired its zero tangent without changing
forward values. Paid job `14650340` nevertheless reproduced the same loss,
hard counts, and three NaN gradient sources at that commit. Its report SHA256
is `c1bce9ed...fdbbdde`. This result corrects the earlier causal diagnosis:
the masked summary is part of the fixed input assembly, so a gradient with
respect to network parameters does not differentiate through it.

The single-sample learned-output-boundary diagnostic at commit `3d5a62f`
examined each of the seven state-group losses with respect to all 39,006
physical learned-output coordinates. Every group contained the same unique
nonfinite coordinate,
`inventory.water.qsintveg.storage.incoming_amount`; all other 39,005
coordinates were finite. In
`mechanistic_output_reconstruction.py::signed_endpoint_to_transfers`, bare
soil has `source == removed == 0`. The former expression
`jnp.where(source > 0, removed / source, 0)` therefore formed an internal
`0/0` that was hidden in the forward result but poisoned reverse mode.

Commit `c8cb497` selects a safe denominator before division and preserves the
forward result exactly. The local minimal reproduction now has finite
gradient `[[0, -5]]`, and the complete mechanistic-output regression passes
13 tests. The repaired free-`gln01` boundary VJP has SHA256
`4444e23b...b721ddc4` and all seven groups are finite. Commit `e8294b0` then
propagated the same real cotangent through the complete Attention model: all
83,755,173 parameter-gradient values are finite, with norm
`419576.1733` and finite nonzero gradients in the base encoder, domain
refinement, science cross-attention, science mixer, primary heads, and state
heads. Its report SHA256 is `ee2f1715...a5fba59`.

The required complete single-batch zero-update real-v6 representative probe
contains 32 one-day samples, eight 7-day windows, and four 30-day windows. It
was required after the boundary diagnostic because that diagnostic exercised
only the one-day state anchor.

Paid job `14651381` attempted that repaired probe at commit `2a6a5d5` but
reached the Slurm limit after `01:30:21`. It wrote no report, retained no
update, and produced no gradient acceptance or rejection. Its log contains no
Python exception or physical hard-gate failure; it shows that the repaired
finite path continued through expensive XLA compilation, including about
20 minutes for one backward executable, until Slurm terminated it. The
project-scoped persistent cache grew to about 262 MiB and retains the completed
executables.

The unchanged cache-reusing rerun completed under paid job `14652590` at
commit `50c2c14` in `00:53:12`. Its report SHA256 is
`aa195412...33ac861` and status is `passed`. The one-day/process, 7-day, and
30-day source-gradient norms are respectively `5202018.8299`, `1.4514`, and
`0.8519`; every value in all three 83,755,173-element source gradients is
finite. All seven combined parameter families are finite and nonzero. The
finite AdamW proposal has raw norm `5202019.0`, clip scale `1.0`, parameter
step norm `0.00345624`, and zero nonfinite parameter or optimizer-state
values. Peak live device memory is 8.180 GB. Every budget, discrete, stock,
unexpected-mask, prediction, and state hard-failure count is zero; the 147
mask mismatches are exclusively the declared dynamic `rveget` status. No
update or checkpoint was retained.

The gradient decomposition is now an explicit conditioning observation. The
7/30-day objectives dominate the primary science-head gradients but are about
seven orders of magnitude below the one-day state anchor in the shared
representation families at initialization. AdamW's coordinate-wise scaling
means this raw ratio is not itself an update ratio, and the long-horizon heads
do receive strong direct gradients, but one batch cannot prove that the shared
representation will learn the long-horizon objectives. The active operation
was the frozen eight-batch zero-update preflight.

Paid job `14659152` completed that preflight at commit `bca102c` in
`00:27:00`. Report SHA256 `7bddd58b...a688de5` has status `passed`. Every one
of the eight combined 83,755,173-element gradients is finite, all seven
connected parameter families are finite and nonzero in every batch, and the
frozen p90 clipping threshold is `5323655.8449`. The unclipped norms span
`4.2401e6` to `5.6075e6`. The finite diagnostic AdamW proposal has zero
nonfinite parameter or state values. Every physical hard-failure count is
zero, peak live JAX memory is 9.60 GB, Slurm peak host RSS is 17.48 GiB, and
the seven hot effective steps have median `36.8113` seconds. The report wrote
no checkpoint and retained no update.

The eight batches confirm that long-horizon gradients strongly reach primary
science heads but are initially about seven orders of magnitude below the
one-day anchor in shared representation families. This is now a required
warmup diagnostic rather than grounds for changing the frozen loss before
training.

The retained update-1-to-250 runner and free-GPU admission are complete at
commit `e363482`. The merged prepared manifest has SHA256
`0b1e2c97...dfcd3`, exactly covers updates `1..250`, and totals 840,481,740
bytes. The 96-window validation manifest has SHA256 `1c9081ef...5186a` and
passed independent per-file verification. Neither product reads v5 or the
sealed test and neither regenerates v6 Teacher data.

The passing one-update `gln01` admission report has SHA256
`51876122...a377`. It completed on GPU with gradient norm `5202019.0` below
the frozen `5323655.8449` threshold, clip scale `1.0`, finite parameter-step
norm `0.00345624`, zero contract-defined hard failures, verified recovery
reload, and six initialization plus six post-update 7/30-day cells. Its tiny
`0.004144%` macro improvement is not scientific evidence because admission
uses one window per cell and deliberately is not the full frozen panel.

Paid `gnall` job `14664525` completed normally at commit `e363482` in
`02:57:58`. It retained all 250 contiguous updates, both alternating recovery
generations, the update-250 stage checkpoint, the best-parameter checkpoint,
and both complete 96-window validation panels. All contract-defined hard
failures are zero. Only updates 5, 32, and 143 were clipped, for clip fraction
`0.012`; the last 25 updates had no clipping. Peak live JAX device use was
10.77 GB and Slurm MaxRSS was 19,590,408 KiB.

The full-panel science macro fell from `0.96405897` to `0.68948347`, a
`28.4812%` improvement. This passes the preregistered `5%` pacing rule but not
the absolute 7/30-day gate. AGB, BGB, and total live biomass pass all six
slice/horizon cells. GPP, NPP, NEE, and NEP improve materially but remain
above their absolute thresholds. The dominant unresolved errors are ecosystem
and nonliving ecosystem carbon, especially at 30 days where every slice
improves by less than `0.4%`; maintenance respiration is inconsistent and
worsens in several cells. The checkpoint is not promotion eligible.

The explicit update-251-to-1000 continuation is implemented locally. It loads
the exact stage checkpoint and AdamW step 250, admits a prepared partial range
without duplicating updates 1-250, and continues schedule and learning rate at
absolute update indices. It preserves the frozen objective, clipping threshold,
and validation panel. Focused continuation/checkpoint/range-manifest tests pass
`8/8`; the extended target-oriented group passes `37/38`, with the sole failure
an unchanged historical representative-probe wall-time assertion.

Free-node continuation admission now passes at commit `fee229e`. Update 251
loads the exact stage checkpoint and AdamW step 250, uses learning rate
`9.9999971e-5`, has finite gradient norm `284116.125`, is not clipped, and
writes/reloads recovery with zero physical hard failures. On the matched six
admission windows the macro changes from `0.552285` to `0.501974`, a `9.1095%`
improvement. This is not full-panel promotion evidence.

The exact update-251-to-1000 prepared range is accepted. Four free-CPU
chunks merged into 750 unique contiguous artifacts totaling 2,523,051,773
bytes. Every artifact hash and payload passed, the full schedule SHA256 is
`ee1ff380...9e2a5`, and the merged manifest SHA256 is `8205140d...8fc31`.
Teacher regeneration, historical v5 use, and sealed-test read are all false.
Paid job `14674142` then completed the exact continuation at commit `433d286`
in `08:12:26`. All 750 continuation updates are retained, continuation
clipping is `0/750`, cumulative clipping is `3/1000`, and every physical hard
gate passes. The frozen macro improves from `0.68948347` at update 250 to
`0.34085437` at update 1000, a further `50.5638%`, but the absolute gate still
fails.

The scientific failure has changed materially. Mean 30-day ecosystem and
nonliving carbon NRMSE collapse from about `2.56` to `0.0474` and both pass all
three slices; their 7-day counterparts and AGB/BGB/total-live biomass also
pass all cells. The remaining blockers are GPP, NPP, maintenance respiration,
growth respiration, and autotrophic respiration. NPP improves only `5.60%`
at 7 days and `0.132%` at 30 days from update 250. The subsequent zero-update
cross-batch audit completed over 32 train/train windows and all `83,755,173`
parameters. GPP stabilizes at 16 windows and NPP is already coherent at four,
but maintenance and total autotrophic respiration remain opposed between the
two 16-window estimates. Its decision is
`mixed_batch_and_objective_geometry`; a larger batch alone is not admitted.

The component-resolved long-horizon flux amendment and its free-`gln01`
zero-update geometry audit are complete. GPP, maintenance respiration, growth
respiration, and heterotrophic respiration are the independent flux basis;
autotrophic respiration, NPP, NEP, and NEE remain derived gates. Every one of
the five gradient sources is finite and nonzero over all `83,755,173`
parameters in all eight groups, with zero physical hard failures and no
retained update. Report SHA256 is `414aa543...ff2a05`.

The geometry does not admit raw averaged training. Seven-day flux gradients
have median cross-batch cosine `-0.4739`, `53.6%` negative pairs, and a norm
range from `0.515` to `1713.736`; 30-day flux gradients have median cosine
`+0.0199` and `50%` negative pairs. The mean is dominated by update 465 and
is nearly opposite the already successful 7-day stock gradient. The protected
virtual-step audit then completed on free `gln01` at commit `04c2b23`; report
SHA256 is `6a9ed9c9...f1fba`. A valid 16-constraint common direction exists,
with independently realized minimum cosine `0.04353145`, but no finite step
passes. Even norm `0.0001` raises exact 7-day component-flux loss by a median
`0.6359%` and at worst `15.7217%`, raises one-day state/process loss by a
median `11.5593%`, and raises the worst stock score `33.1712%`. All physical
hard and nonfinite counts are zero, and no model state is retained.

Direct realization of the raw component-resolved MSE-plus-squared-bias loss
is closed; the Attention architecture is not rejected. The subsequent
zero-training residual-tail audit completed on free `gln01` at commit
`b80494f`; report SHA256 is `e1c094e2...27d9e`. All 96 window gradients are
finite, complete over `83,755,173` parameters, and have zero physical hard
failures.

The 7-day loss effective sample size is `21.763/64`, but the gradient-norm
effective sample size is only `1.179/64`. The update-465, landpoint
`081.0-107.0`, row-90 window has loss `0.05285` and gradient norm `22489.30`:
about `2.12%` of total loss but `92.07%` of total gradient-norm mass. The
30-day maximum is also maintenance-respiration dominated. Frozen soft-L1 and
log1p loss weighting leaves the 7-day minimum cosine near `-0.97`; the best
30-day candidate improves but remains conflicted. Robust-loss-only repair is
therefore not established.

The focused gradient-path attribution completed on free `gln01` at commit
`53269af`; report SHA256 is `a2e9a885...ffa870d`. All `24/24` full gradients
are finite over all `83,755,173` parameters and all physical hard failures
are zero. Both extreme windows are recurrence dominated. Their maintenance
free-to-Teacher-state gradient ratios are `243184.83` at 7 days and `49.38`
at 30 days, and almost the entire extreme maintenance gradient enters through
the predicted `hydrol_previous_step_state.snowheat` tendency head. Matched
controls instead have larger Teacher-state maintenance gradients, so this is
a scenario-dependent feedback rather than a generic Teacher-forcing effect.

The snowheat-only intervention completed on free `gln01` at commit `159f672`;
report SHA256 is `e2a6934b...a05e0f`. All eight gradients are complete and
finite, free scores reproduce exactly, and physical hard failures are zero.
The 7-day extreme gradient falls by `10416.39`, confirming snowheat recurrence
there. The 30-day extreme snowheat-head gradient falls by `78.96`, but the full
gradient falls only by `1.56`; the remaining state path is dominated by
snowtemp with norm `272.77` of a full `276.37`.

The snowheat-plus-snowtemp recurrence intervention completed at commit
`975ae9b`; report SHA256 is `c4b80543...303810`. The 30-day extreme full
gradient falls from `276.37` to `52.87`, while its snowtemp head falls from
`272.77` to `0.551`. The full-gradient reduction of `5.23` misses the frozen
tenfold gate, and the remaining snowheat head is `40.71`.

The mirror-complete intervention completed at commit `51bb538`; report SHA256
is `5714d30d...4422168`. Both gradients are complete and finite, all four
slices and both mirror relations resolve, and physical hard failures are
zero. The snowheat head becomes exactly zero in both windows. The extreme
full gradient falls from `52.87` to `33.69`, only `1.5693` times, because the
direct maintenance head remains `33.47`. The matched control falls from
`31.70` to `15.68` and retains a `15.52` maintenance head. The alias path is
real but does not explain the full residual; the classification is mixed.

The persisted-maintenance intervention completed at commit `d08d6eb`; report
SHA256 is `a95ae129...f809b9`. Both complete gradients are finite and nonzero
over all `83,755,173` parameters, all five state slices resolve, and physical
hard failures are zero. The extreme full gradient changes from `33.69151431`
to `33.69151394`, while its direct maintenance head changes from
`33.46682558` to `33.46682500`. The matched control changes by similarly
negligible amounts. The frozen classification is
`maintenance_state_recurrence_not_explanatory`.

This closes the bounded Teacher-carry state investigation. It does not support
removing persisted `resp_maint` from the broad carbon input route. The
subsequent `target_oriented_maintenance_head_conditioning_audit_v1` completed
on free `gln01` at commit `79b2e36`; report SHA256 is
`85ab83a1...d27757`.

All 16 finite, nonzero gradients over the 769-parameter maintenance final head
admit a common direction with minimum cosine `0.056523`. Four in-bound steps
were evaluated, but none passes every exact protection gate. At norm
`0.0018462`, median 7/30-day maintenance scores improve by
`1.4415%/0.8188%`; NPP and live-biomass scores also improve. Small positive
median changes remain in 7-day GPP, growth respiration, and ecosystem carbon,
so the preregistered classification is `finite_head_step_not_protected`.
Every proposal was discarded, all hard/nonfinite counts are zero, and no
optimizer or checkpoint was used.

The source-shaped maintenance architecture discussion is complete. Contract
SHA256 is
`3c4d0c085c767f14b001837ccefb76aca291a56b8fd9d3a5e33279602af57fc4`.
The local implementation retains the parent Attention and non-maintenance
heads, removes the 769-parameter maintenance projection, and adds two bounded
part-group MLP branches for an exact total of `84,148,646` parameters. The
zero-correction source formula uses day-start biomass/LAI, retained-static
`coeff_maint_zero`, explicit slope parameters, daily air temperature, and
root temperature from predicted daily `tsoil`; reserve uses the tree-dependent
air/root branch. All six focused local tests pass, including eager/JIT,
complete loss, both new-branch and reused-Attention gradients, nonzero-head PFT
equivariance, one canonical day, and exact water/carbon budgets.

The zero-update real-v6 GPU admission passed on free `gln01` at commit
`cf1fead`; report SHA256 is `b5684fb4...b5119b46`. All 84,148,646 gradients
are finite, both new branches and every retained Attention family have nonzero
gradients, all 160 non-maintenance outputs remain exactly equal to the parent,
and the canonical next state and exact water/carbon budgets pass. The frozen
zero-update train/train panel then improves maintenance by `73.61%/87.35%` at
7/30 days without training, while failing protection because several 7-day
carbon scores regress. All hard counts are zero.

The free one-update branch-only admission then passed at commit `0ed926b`;
report SHA256 is `c2743cbc...34ea59`. The frozen p90 branch clip is
`0.1973867`; artifact 252 uses clip scale `0.86065`, changes both new branches,
keeps every retained coordinate unchanged, has zero hard failures before and
after, and slightly reduces the effective-batch objective. The proposal was
discarded without checkpoint or recovery.

The direct branch-subtree autodiff admission then passed at commit `d965bea`;
report SHA256 is `5852d996...e58538`. The full forward remains unchanged, but
only 394,242 branch coordinates are differentiated. Zero gradient coordinates
fail against the subtree extracted from the complete 84.1-million-parameter
gradient; the objective, three gradient sources, hard counts, fresh AdamW
proposal, and post-proposal objective also match. Hot effective-batch time is
`25.18 s` versus the admitted `36.85 s`, and JAX peak live memory is about
2.24 GiB versus 7.64 GiB. Physical gates pass and no training state is
retained. The paid runner and branch-only checkpoint/recovery were locally
qualified at commit `9224398`. They bind artifacts `252..351` to updates
`1..100`, reuse the accepted update-0 screen, evaluate the complete frozen
eight-group 7/30-day screen at update 100, and keep all retained parent
coordinates outside the optimizer.

The user approved that exact envelope and Slurm job `14701638` completed all
100 updates in `00:58:51` at commit `7fdce65`; report SHA256 is
`a5143c06...b2d4b22`. Only the 394,242 branch coordinates changed, clipping
occurred on `4/100` updates, and every final-screen physical hard-failure count
is zero. Median 7/30-day maintenance reductions increase from
`73.61%/87.35%` to `90.76%/96.50%`.

The scientific gate fails because protected carbon balance does not recover.
At update 100, median 7-day growth respiration, NPP, AGB, BGB, and total-live
biomass remain worse than the fixed parent, with several worsening relative
to update 0. No 7-day group and only one 30-day group passes every absolute
target-oriented key. The branch checkpoint is diagnostic only; continuing the
same branch-only training is closed. The active gate is a zero-training
source-coupling attribution across the fixed parent, update-0 source
candidate, and update-100 branch state on the same train/train panel. It must
separate direct carbon-identity effects from rollout-state recurrence before
any jointly trainable flux successor is designed. Validation/sealed-test
reads, broader training, 365-day rollout, Gate D2, and product work remain
closed.

That attribution completed on free `gln01` at commit `4c22a4c`; report SHA256
is `e2a0b90f...26a0d6b`. It creates no optimizer or checkpoint, all physical
hard failures are zero, and the two raw carbon-bias identities close to
`5.68e-14`. In free rollout, update 100 improves maintenance by
`62.93%/71.74%` at 7/30 days while worsening growth respiration, NPP, AGB,
BGB, and total live biomass. Under Teacher day starts, those protected
regressions all become improvements, GPP is exactly unchanged, and maintenance
itself instead worsens by `680%/792%`.

The accepted classification is `direct_daily_coupling_confirmed`, with every
material protected free-rollout regression marked recurrence-required and
`joint_flux_scope_required=false`. Immediate joint flux expansion is closed.
The zero-training source-owner state bridge then completed on free `gln01` at
commit `1c862a4`; report SHA256 is `7defdf47...642d5c`. It Teacher-carries only
`biomass`, `lai`, `veget_max`, and `t2m_longterm` while all other cross-day
state remains predicted. All hard failures are zero and the raw carbon
identities close to `3.02e-14`.

Only `4/10` required cells reverse: 7-day AGB, BGB, and total live biomass plus
30-day AGB. Both growth-respiration and NPP cells remain materially regressed,
as do 30-day BGB and total live biomass. The accepted classification is
`source_owner_state_drift_contributes_but_not_sufficient`.

The approved successor is now frozen and locally accepted under contract
SHA256 `f167943c...1a37d2d`. It retains the update-1000 Attention parent and
all 24 PFT/part maintenance hidden coordinates, adds a two-block
maintenance-specific Attention adapter over full canonical context, and uses
the existing exact source materializer. The complete candidate has
`93,504,933` parameters; the first stage owns only the `9,750,529` new adapter
parameters. All 11 focused candidate/predecessor tests pass, including
eager/JIT forward, complete loss, adapter gradients, exact non-maintenance
parent equality, permutation, canonical state, and water/carbon budgets. The
zero-update real-v6 admission then passed on free `gln01` at commit `54ada54`;
report SHA256 is `376d8016...c66ece5`. All adapter gradients and canonical
budgets pass, and all 160 non-maintenance outputs exactly equal the parent.
The eight-batch Teacher-guard calibration completed on free `gln01` at commit
`b980cde`; report SHA256 is `bb55388f...addf7a474`. Every free and guard
gradient covers all `9,750,529` adapter coordinates, all 36 leaves are
nonzero, and all physical hard failures are zero. Median norms are
`0.3181163` and `13.9532423`; their raw ratio `0.0227987` freezes the bounded
Teacher-guard coefficient at exactly `0.1`.

The adapter-only bounded training contract is frozen at SHA256
`0c60924e...210051`. It owns 250 prepared artifacts from
`252..511` after excluding calibration/evaluation updates, a fresh 250-update
AdamW schedule, and an immutable combined-gradient clip threshold
`2.692620468992339` derived without another run. The ephemeral artifact-252
update passed on free `gln01` at commit `5ea0033`; report SHA256 is
`a32b2c17...c7734e`. Its combined norm is `2.0161371`, no clipping occurs,
all adapter leaves change, every parent coordinate remains exactly fixed, and
all pre/post hard failures are zero. The exact 250-update runner, compact
adapter-only recovery/checkpoint, update-0/update-250 screen, and one-V100
Slurm wrapper were locally qualified at commit `ad5c11b`; the related suite
passes 22 tests. Paid job `14856147` completed all 250 updates at commit
`90c6636` in `03:38:12`; report SHA256 is `7f6d5686...9257`. No update was
clipped, all hard counts are zero, and all `83,754,404` parent coordinates
remain exactly fixed. Median maintenance reduction reaches
`90.7403%/95.7084%` at 7/30 days and maintenance passes both aggregate
absolute gates. The complete scientific gate fails because GPP, growth
respiration, and NPP remain above threshold and the protected no-regression
gate is not met. The adapter checkpoint SHA256 is `baae0c13...7da`, is
diagnostic only, and unchanged adapter-only continuation is closed. The
active gate is a zero-training Teacher-GPP causal bridge on the same eight
train/train groups before any architecture change. Validation, sealed test,
365-day rollout, Gate D2, and product work remain closed.
The bridge contract is frozen at SHA256 `ce2eefb1...28c5647`. Its local
implementation passes 29 distinct focused and related tests, including one
complete learned-output/typed-flux/state/budget day, plus Ruff, Python
compilation, Bash syntax, CLI import, and diff checks. Its free-`gln01`
execution completed in `934.80 s`; report SHA256 is `fa9bc445...531c52`.
Both modes have zero hard failures. Teacher GPP makes GPP exact, reduces
growth-respiration score by `99.6757%/99.5925%` and NPP by
`98.6264%/98.8312%` at 7/30 days, and makes all nine required aggregate keys
pass. Teacher-day-start instead severely worsens the learned fluxes. The
classification is `gpp_boundary_supported_as_next_scope`.

The subsequent user-approved structured GPP mechanism is frozen at SHA256
`ce211fa7...620ea` and locally accepted. It preserves the coordinate-aligned
Attention parent and accepted distribution-aware maintenance adapter, replaces
only the old 769-parameter GPP projection, and adds a 12,879,363-parameter
day-local Transformer over 45 forcing field/time tokens. Its potential,
environmental-limitation, and bounded-correction branches produce GPP through
the existing explicit `vcmax25 / 100` route. The complete candidate has
`106,383,527` parameters. Five focused tests pass exact counts, eager/JIT,
complete loss and gradients, bitwise non-GPP parent equality, parameter-route
isolation, inactive/bare/PFT permutation, and a complete canonical daily
state/budget transition. The free-`gln01` real-v6 zero-update admission then
passed at commit `dbc1373`; report SHA256 is `7a5d4310...04c488`. It uses the
GPU backend and frozen update-250 maintenance checkpoint, preserves all parent
parameter objects, matches all 160 non-GPP outputs bit for bit, covers all
12,879,363 specialist gradient elements with `129/129` nonzero leaves, and
finishes the canonical 181-output day with exact-zero water/carbon residuals.
JIT replay is `0.064 s`; peak live allocation is `1.47 GB`. No optimizer,
update, or checkpoint was created. The separate specialist-only training
preflight described below is now frozen. Paid training, validation,
7/30/365-day evaluation, Gate D2, sealed test, and product work remain closed
until their declared prerequisites pass. See
[`target_oriented_structured_gpp_v1.md`](research/daily_coarse_graining/target_oriented_structured_gpp_v1.md)
and
[`target_oriented_structured_gpp_local_acceptance_20260829.md`](research/daily_coarse_graining/target_oriented_structured_gpp_local_acceptance_20260829.md),
with real-v6 evidence in
[`target_oriented_structured_gpp_gpu_admission_20260829.md`](research/daily_coarse_graining/target_oriented_structured_gpp_gpu_admission_20260829.md).

The structured-GPP training preflight passed on free `gln01` at commit
`d852e69` in `1059.82 s`; report SHA256 is `d531f8dc...2cbccc`. All eight
train-only direct-GPP gradients are finite across all 12,879,363 specialist
elements and 129 leaves. Their p90 norm freezes the warm-stage clip at
`17.256861560753983`. One discarded AdamW proposal at learning rate `4e-6`
reduces same-artifact GPP loss from `4.04517568` to `4.04308744`, changes every
specialist component, preserves all 93,504,164 parent parameters by identity,
and passes the daily physical gate. Real 7/30-day forward compatibility passes
without horizon-gradient compilation. See
[`target_oriented_structured_gpp_training_preflight_result_20260829.md`](research/daily_coarse_graining/target_oriented_structured_gpp_training_preflight_result_20260829.md).

Paid job `14873550` completed the frozen 250-update direct-GPP warm start at
commit `1fb8916` in `00:21:25`; report SHA256 is `c0e23be4...500f`. Only the
12,879,363-parameter specialist trained, while all 93,504,164 parent
parameters remained exactly unchanged. The disjoint eight-artifact endpoint
panel improved `8/8`; aggregate direct-GPP loss fell from `1.11241519` to
`0.74803793`, a `32.7555%` reduction against the frozen `10%` requirement.
All physical hard failures are zero. The specialist checkpoint SHA256 is
`ebef94aa...3f9` and remains non-promotion-eligible.

The train-only 7/30-day joint-gradient calibration passed on free `gln01` at
commit `7c547d5`; report SHA256 is `6ff2d03e...389c0`. All three gradients
cover all `12,879,363` specialist parameters and 129 leaves on all eight
artifacts, and every physical hard count is zero. The frozen 7/30 coefficients
are `5.065197169020285` and `4.594433600967992`, and the combined-gradient p90
clip is `46.802861793665286`. Every pairwise source cosine is positive; the
minimum direct-to-30-day cosine is `0.4849`, while each source-to-combined
cosine is at least `0.7005`. A separately frozen discarded inherited-AdamW
joint proposal was defined at contract SHA256
`5d255e49...0c5d5`. It executes one ephemeral step 251 on artifact 512 and
must preserve descent for all three source losses and all physical gates.
The preflight completed at commit `e9aa41e` and is rejected; report SHA256 is
`c63d2d4e...ccfc6`. Direct GPP and 7-day loss fall `0.117%` and `0.192%`, but
30-day loss rises `0.256%`. Artifact 512's 7-versus-30 raw gradient cosine is
`-0.9134`, and the inherited AdamW step has cosine `-0.0167` with the 30-day
gradient. All engineering, finite-value, physical, parent-preservation, and
checkpoint gates pass, and no proposal is retained. The subsequent
zero-update gradient heterogeneity audit completed at commit `95322f8`;
report SHA256 is `15d993a2...80eea`. Batch sizes `1/2/4/8` pass
`7/8`, `3/4`, `2/2`, and `1/1` groups, so batch size 4 is the minimum
consistent effective batch. The two batch-four groups have minimum
source-to-combined cosines `0.7992` and `0.7543`. All execution and physical
gates pass, and no optimizer or update was created. The subsequent two-group
batch-four preflight passed at commit `42c0e4e`; report SHA256 is
`6a574c67...a0ac1`. Group-mean direct/7-day/30-day losses change by
`-0.1694%/-0.1079%/-0.0897%` and
`-0.2086%/-0.1038%/-0.1222%`. Every aggregate realized-step cosine is
positive, all physical gates pass, both proposals start from the same
step-250 state, and neither is retained. The bounded successor is now frozen
under contract SHA256 `168d44bd...98832`, with deterministic schedule SHA256
`770bae61...5f70`. It permits at most four epochs and 248 effective batch-four
updates, continues inherited AdamW from step 250, writes recovery every ten
updates, and selects only the last epoch whose train-only direct/7-day/30-day
mean losses all strictly improve. The runner directly checks finite specialist
and optimizer proposals and a nonzero parameter step. Its local contract,
negative-design, checkpoint, static, CLI, and shell gates pass. Paid job
`14878155` then completed all 248 batch-four updates and all four train-only
epoch panels. Direct GPP, 7-day science, and 30-day science panel losses fell
`49.6886%`, `50.7018%`, and `42.8952%`; all execution and physical gates pass,
and step 498 is the selected diagnostic checkpoint. Its independent 96-window
screen completed on free `gln01` at commit `d41cd0c`. Every physical hard
failure is zero, but no slice/horizon cell passes all absolute science keys.
The science macro is `0.4613409`, `35.3484%` worse than the inherited
step-1000 Attention checkpoint on the same panel. GPP fails all six cells with
negative bias and mean NRMSE `49.3998%` worse; maintenance respiration improves
`59.3330%` and passes five cells. AGB, BGB, total-live, ecosystem, and
nonliving-carbon stocks pass all six cells. Step 498 is not promoted; 365-day
rollout, Gate D2, sealed test, and product work remain closed. The active work
is a zero-update broad-v6 direct-GPP coverage and throughput audit before any
new training contract. The coverage half is now complete at commit `db9763b`.
It exactly reconstructs 103,399 unique exposed days, only `1.203494%` of the
8,591,565 accepted train/train transitions; the four joint epochs repeat a
51,878-day subset. All 535 train landpoints and 44 years occur, so the gap is
shallow day-level exposure rather than missing global IDs. A target-blind
monitor proposal reserves 128 previously unseen point-years on 128 distinct
landpoints, with 2,048 sampled monitor days and 46,589 fully reserved eligible
days. The direct-only streamer is now accepted at commit `c28a667`. It removes
typed-sidecar and complete-example assembly from the GPP path, reducing 208-day
host preparation from `84.2489 s` to `0.7304 s` (`115.34x`) and sustaining
284.76 host days/s. A contended free-GPU run gives a conservative 56.36
compute days/s and 47.05 end-to-end days/s, with finite loss statistics and all
12,879,363 specialist gradients finite. The active operation is now the broad
direct-GPP target-blind schedule and free-GPU preflight frozen by machine
contract `f0e50031...096a3`. It proposes 5,000 updates over 1,040,000 new days,
with 52 distinct landpoints and four days per point-year in every 208-day
update. It reserves the accepted monitor and excludes every previously exposed
day. The full schedule audit and free-GPU preflight have passed at commit
`853afa3`: schedule SHA256 `b4295558...d6b8b`; all 16 gradients were finite over
all 129 specialist leaves; raw gradient p95 is 20.98786557796449; and all four
independent `4e-6` AdamW steps reduced their batch loss while passing physical
and retained-parent gates. The active stage-1 contract is `c54233c1...340e1`,
bounded to 1,000 updates and 208,000 unique days. The two-update free admission
passed at commit `1573eb9`, report SHA256 `94a43cbe...cc70`: both chained
updates were finite and nonzero, the 93,504,164 retained parent parameters were
exactly unchanged, physical hard failures were zero, and the step-2 recovery
was written with SHA256 `eed4aca8...2e92`. Explicit paid resource approval is
now the active gate. The first paid launch, job `14881105` at `af5f426`, failed
before Python after two seconds because old Bash rejected an empty optional
argument array under `set -u`; it performed no training or checkpoint write.
Commit `c24f21b` removes that array and passed the former line in a free
production-branch launch smoke. Corrected job `14883081` then completed all
1,000 updates in `02:30:39`; report SHA256 is `5cfab5a6...bee9`. The fixed
monitor loss fell 89.63%, 125/128 point-years improved, aggregate normalized
bias magnitude fell from 0.432 to 0.092, all physical hard counts were zero,
and retained parent parameters were exactly unchanged. Update 1,000,
checkpoint SHA256 `3b173b89...4ec0`, is the accepted broad-stage selection.
The following fresh independent 7/30-day screen completed at commit `c422c37`
under contract SHA256 `894a0b2d...e6a7`; formal report SHA256 is
`a9b9f6dc...98b`. It excludes the consumed panel's point-years within each
slice and evaluates 96 new windows. Every execution and physical hard gate
passes, but the absolute science gate fails. Teacher-state daily GPP NRMSE is
`0.44496`; free-rollout GPP is `0.73409` to `1.00839` and fails all six cells.
Growth respiration, NPP, NEP, and NEE also fail every cell. Maintenance and
heterotrophic respiration plus all biomass and ecosystem-carbon stocks pass
all six cells. The checkpoint is rejected for promotion.

The same-panel three-mode attribution is now complete under contract SHA256
`2c93dbe5...112a`. Its corrected, hash-bound CPU final report SHA256 is
`fafd5111...f8037e`. Teacher-state cumulative GPP fails `5/6` cells; cross-day
state feedback is material in `6/6`. The Teacher-GPP bridge makes growth
respiration and NPP pass all cells, reducing their median acceptance scores by
`99.6822%` and `99.6525%`. The formal classification is
`direct_and_feedback_gpp_with_sufficient_downstream_cause`; all attribution
integrity and physical gates pass.

The following train-only source-component state-feedback attribution is also
complete at commit `15171cd`, under contract SHA256 `858b5907...c69ed` and
report SHA256 `1b2d1820...41c2a19b`. Carrying only
`slowproc_stomate_previous_step_state` captures median `96.0598%/81.9454%`
of the 7/30-day cumulative-GPP free-to-full-Teacher score gap and is positive
in all six update/horizon cases. It is the only component that passes the
frozen materiality rule. Water captures `9.1288%/5.1239%`; it remains a
monitored secondary path but does not enter the first selected scope. Canopy,
energy, thermal, and driver components are nonmaterial. All physical and
integrity gates pass, with no training, fresh-target read, Teacher
regeneration, v5 use, checkpoint write, 48-record reconstruction, or subdaily
scan.

The active gate is architecture discussion and contract freeze for a coupled
GPP/carbon-state candidate. Retain the day-local coordinate-Attention GPP
specialist and canonical retained exact tail; initially open only source-backed
inputs/process paths associated with `slowproc_stomate_previous_step_state`,
while protecting and reporting water. The train-only objective, matched
mechanism control, local/real-v6 zero-update admission, bounded training
budget, target-blind monitor, and stop rule must be approved before
implementation or any optimizer update. No unchanged specialist training,
fresh-target tuning, 365-day rollout, Gate D2, sealed test, or product
acceptance is authorized. See
[`target_oriented_structured_gpp_warm_start_result_20260829.md`](research/daily_coarse_graining/target_oriented_structured_gpp_warm_start_result_20260829.md)
and
[`target_oriented_structured_gpp_joint_gradient_calibration_result_20260829.md`](research/daily_coarse_graining/target_oriented_structured_gpp_joint_gradient_calibration_result_20260829.md),
with the rejected proposal in
[`target_oriented_structured_gpp_joint_update_preflight_result_20260829.md`](research/daily_coarse_graining/target_oriented_structured_gpp_joint_update_preflight_result_20260829.md),
the effective-batch result in
[`target_oriented_structured_gpp_joint_gradient_heterogeneity_result_20260829.md`](research/daily_coarse_graining/target_oriented_structured_gpp_joint_gradient_heterogeneity_result_20260829.md),
and the active preflight in
[`target_oriented_structured_gpp_joint_batch4_preflight_result_20260830.md`](research/daily_coarse_graining/target_oriented_structured_gpp_joint_batch4_preflight_result_20260830.md).
The completed training and screen results are in
[`target_oriented_structured_gpp_joint_training_result_20260830.md`](research/daily_coarse_graining/target_oriented_structured_gpp_joint_training_result_20260830.md)
and
[`target_oriented_structured_gpp_short_horizon_screen_result_20260830.md`](research/daily_coarse_graining/target_oriented_structured_gpp_short_horizon_screen_result_20260830.md).
The exact coverage evidence is in
[`target_oriented_structured_gpp_data_coverage_result_20260830.md`](research/daily_coarse_graining/target_oriented_structured_gpp_data_coverage_result_20260830.md).
The accepted direct-only throughput evidence is in
[`target_oriented_structured_gpp_streaming_throughput_result_20260830.md`](research/daily_coarse_graining/target_oriented_structured_gpp_streaming_throughput_result_20260830.md).
The active preflight contract is
[`target_oriented_structured_gpp_broad_preflight_v1.md`](research/daily_coarse_graining/target_oriented_structured_gpp_broad_preflight_v1.md).
Its accepted result is
[`target_oriented_structured_gpp_broad_preflight_result_20260830.md`](research/daily_coarse_graining/target_oriented_structured_gpp_broad_preflight_result_20260830.md),
and the active paid-stage design is
[`target_oriented_structured_gpp_broad_training_v1.md`](research/daily_coarse_graining/target_oriented_structured_gpp_broad_training_v1.md).
Its accepted runner admission is
[`target_oriented_structured_gpp_broad_training_admission_result_20260830.md`](research/daily_coarse_graining/target_oriented_structured_gpp_broad_training_admission_result_20260830.md).
The paid-launch failure and correction are in
[`target_oriented_structured_gpp_broad_training_launch_failure_20260831.md`](research/daily_coarse_graining/target_oriented_structured_gpp_broad_training_launch_failure_20260831.md).
The accepted broad-stage result is in
[`target_oriented_structured_gpp_broad_training_result_20260831.md`](research/daily_coarse_graining/target_oriented_structured_gpp_broad_training_result_20260831.md).
The completed fresh-screen contract is
[`target_oriented_structured_gpp_fresh_screen_v1.md`](research/daily_coarse_graining/target_oriented_structured_gpp_fresh_screen_v1.md).
Its completed result is
[`target_oriented_structured_gpp_fresh_screen_result_20260831.md`](research/daily_coarse_graining/target_oriented_structured_gpp_fresh_screen_result_20260831.md).
The completed attribution contract is
[`target_oriented_structured_gpp_fresh_attribution_v1.md`](research/daily_coarse_graining/target_oriented_structured_gpp_fresh_attribution_v1.md).
Its accepted result and predecessor design evidence are
[`target_oriented_structured_gpp_fresh_attribution_result_20260831.md`](research/daily_coarse_graining/target_oriented_structured_gpp_fresh_attribution_result_20260831.md).
The completed train-only state-feedback scope contract is
[`target_oriented_structured_gpp_state_feedback_attribution_v1.md`](research/daily_coarse_graining/target_oriented_structured_gpp_state_feedback_attribution_v1.md).
Its accepted result and current design-scope authority are
[`target_oriented_structured_gpp_state_feedback_attribution_result_20260831.md`](research/daily_coarse_graining/target_oriented_structured_gpp_state_feedback_attribution_result_20260831.md).
The user-approved architecture and accepted local implementation design are
[`target_oriented_coupled_gpp_carbon_feedback_architecture_review_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_carbon_feedback_architecture_review_v1.md).
The exact 2,386-coordinate layout, 3,080,832-parameter adapter, complete
109,464,359-parameter candidate, and local forward/loss/gradient/PFT/parameter-
route/daily-budget gates are recorded in
[`target_oriented_coupled_gpp_carbon_feedback_local_acceptance_20260831.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_carbon_feedback_local_acceptance_20260831.md).
The real-v6 zero-update admission passed under job `14897461`: all 15,960,195
trainable gradients and all 190 leaves are finite and nonzero, the complete
next state is finite, water/carbon budget residuals are zero, non-GPP parent
outputs are bit-exact, cache-hot forward is `0.057055 s/day`, and peak live JAX
allocation is about 1.55 GB. Report SHA256 is `cf8eb1d2...e623`; see
[`target_oriented_coupled_gpp_carbon_feedback_gpu_admission_20260901.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_carbon_feedback_gpu_admission_20260901.md).
The zero-training GPP-to-next-day slow-carbon reachability audit passed under
job `14899147` at commit `b730b99`; report SHA256 is `dc53765e...d020`. Across
12 train/train calendar positions and one restart boundary, the exact-nonzero
PFT14 union contains 41 coordinates in 16 fields. Every Jacobian and declared
next-state value is finite, every bare-soil derivative is exactly zero, and
all water/carbon budget residuals are zero. No optimizer or checkpoint was
created. See
[`target_oriented_gpp_state_reachability_result_20260901.md`](research/daily_coarse_graining/target_oriented_gpp_state_reachability_result_20260901.md).

The train-only field-grouped one-day state anchor and four-source objective
calibration passed under job `14899511` at commit `5582348` in `01:38:16`.
Report SHA256 is `4f209b77...b5138`. Every source gradient is finite and
nonzero over all 15,960,195 trainable parameters and all 190 leaves on all
eight artifacts; all 16 reachable fields are active and every true physical
hard failure is zero. The resolved direct/state/7-day/30-day coefficients are
`1.0/0.6524275555272667/0.4072260558556481/0.28249951035415655`; the p90 clip
threshold is `12.808193184117403`. The combined within-artifact direction
conflicts with 7-day science in `3/8` artifacts, 30-day science in `4/8`, and
state in `1/8`, though never with direct GPP. See
[`target_oriented_coupled_gpp_objective_calibration_result_20260901.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_objective_calibration_result_20260901.md).

The discarded macro-batch optimizer proposal completed as job `14902356` at
commit `b54c502` in `00:26:32`; report SHA256 is `dd43a960...d146ab`. Every
averaged source gradient has positive cosine with the realized negative step,
all parameters and optimizer values are finite, both trainable families and
all leaves change, and every physical/fixed/checkpoint gate passes. The finite
`4e-6` step nevertheless fails: state loss improves `30.0427%`, while direct
GPP, 7-day science, and 30-day science regress `11.9170%/0.7175%/0.0550%`.
The weighted total improves `1.1798%`, but every-source descent is required.
The raw norm is `6.41398` versus the `12.8082` clip threshold, excluding
clipping as the cause. See
[`target_oriented_coupled_gpp_optimizer_preflight_result_20260901.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_optimizer_preflight_result_20260901.md).

The independent fresh-AdamW half-rate proposal passed under job `14902878` at
commit `510a059` in `00:23:34`; report SHA256 is `6abb472d...f751c5`. Direct,
state, 7-day, 30-day, and weighted macro losses decrease
`4.08665%/16.3007%/0.003126%/0.039625%/4.37440%`. All direction, finite,
coverage, physical, fixed-parameter, and checkpoint-integrity gates pass. The
proposal, moments, and parameters were discarded. See
[`target_oriented_coupled_gpp_optimizer_backoff_result_20260901.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_optimizer_backoff_result_20260901.md).

The target-blind deterministic schedule and first bounded stage are now frozen
at contract SHA256 `5378fe99...699cca`. The stage contains eight macro updates,
each averaging eight prepared artifacts, while the eight calibration artifacts
and a fixed eight-artifact train-only monitor are excluded from training. All
64 training artifacts are unique. Every proposal must pass complete before/
after four-source and physical checks before retention; atomic recovery follows
each accepted update, and only passing update-4/update-8 monitor panels write
non-promotable checkpoints. Local runner qualification and a discarded
production-equivalent free-`gln01` admission are the active gate. Paid training
remains unauthorized until that admission and an exact resource approval.
See
[`target_oriented_coupled_gpp_bounded_training_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_bounded_training_v1.md).

The first free runner-admission attempt at commit `b0ad35d` stopped after
three artifact gradients because its schedule sampled missing artifact `173`
outside the prepared product's admitted `251-1000` range. No report, recovery,
retained update, or checkpoint was written. The corrected contract SHA256 is
`5378fe99...699cca`; its 64 training keys and eight monitor keys are now
checked against `prepared_records` before model construction or JIT.

The corrected free runner admission then passed at commit `885693a` in
`1,579.743` seconds; report SHA256 is `4aaac8ca...b8e23`. The first scheduled
macro proposal improves direct GPP, reachable state, 7-day science, 30-day
science, and weighted total by `1.60249%/10.41690%/1.02810%/1.32569%/2.52874%`.
All 15,960,195 gradients are finite, all 190 leaves are nonzero, both trainable
families change, clip scale is `1.0`, all true physical hard counts are zero,
and temporary recovery round-trips before deletion. No update or checkpoint is
retained. The active gate is the resource-approved paid eight-update stage.
See
[`target_oriented_coupled_gpp_bounded_training_admission_20260902.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_bounded_training_admission_20260902.md).

Paid job `14903328` completed the frozen runner at commit `6d1864b` in
`1,717.025` seconds; report SHA256 is `bf835017...be39c`. Update 1 was retained
after direct GPP, reachable state, 7-day science, 30-day science, and the
weighted objective improved `1.60249%/10.41690%/1.02810%/1.32569%/2.52874%`.
The update-2 proposal then improved the weighted objective `0.686293%`, state
`6.17157%`, 7-day science `1.00842%`, and 30-day science `0.550934%`, but
regressed direct GPP `2.63633%`; it was discarded by the frozen every-source
finite-step gate. Both gradient norms were below the clip threshold, every
true physical hard count was zero, and fixed parameters plus the source
checkpoint remained unchanged. The stage stopped scientifically before update
4, so no milestone monitor or checkpoint exists. The active gate is a
zero-training update-2 audit of raw-gradient versus inherited-AdamW direction
and bounded displacement scales at the exact retained recovery state. See
[`target_oriented_coupled_gpp_bounded_training_result_20260902.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_bounded_training_result_20260902.md).

The follow-up update-2 optimizer-geometry audit is implemented locally. It
uses the exact update-1 recovery and update-2 artifacts, computes the four
gradients once, compares
inherited AdamW against a zero-moment AdamW comparator, and evaluates the
fixed `1, 1/2, 1/4, 1/8, 1/16` displacement scales. It retains no proposal,
does not modify recovery, and reads no fresh target. Its first free execution
at commit `cc86e44` completed the full gradient and inherited scale-one
forward but stopped on an unnecessarily fatal `1e-10` parent-reference
tolerance. No report or retained state was written. The corrected contract at
SHA256 `7585cc5b...79f198`
makes that comparison reporting-only while preserving the frozen scientific
rejection outcome as mandatory. The active gate is corrected local acceptance
and one bounded free-`gln01` rerun. See
[`target_oriented_coupled_gpp_update2_geometry_audit_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_update2_geometry_audit_v1.md).

The corrected audit completed at commit `b7697f5` in `1,870.963` seconds;
report SHA256 is `09209273...b60956`. The inherited AdamW displacement has a
negative direct-GPP cosine `-0.00493710`, so every tested inherited scale
fails. Reset AdamW is a common descent direction and its full-scale proposal
improves direct GPP/state/7-day/30-day/weighted losses
`4.04281%/2.91495%/0.609099%/0.345835%/1.86634%`; all physical hard counts are
zero. Only full scale and `1/16` pass, excluding ordinary monotone
backtracking. Recovery, fixed parameters, and the source checkpoint remain
unchanged. The active gate is a fresh-moments-per-macro-update bounded-training
amendment plus one discarded consecutive-update free-GPU admission. See
[`target_oriented_coupled_gpp_update2_geometry_result_20260902.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_update2_geometry_result_20260902.md).

The fresh-moment amendment is now frozen and locally qualified at canonical
contract SHA256 `a8c362fe...31eda11`. It starts each scheduled macro update
with zero AdamW first/second moments and bias-correction step one while passing
accepted neural parameters in memory. The admission evaluates scheduled
updates 2 and 3 from the exact update-1 recovery, requires all four source
losses plus the existing physical and ownership gates to pass twice, writes
only a JSON report, and retains no proposal or checkpoint. The architecture,
objective coefficients, `2e-6` learning rate, clip threshold, and data schedule
are unchanged. Thirty-one related tests plus Python, Ruff, shell, and
whitespace checks pass. The active gate is the no-checkpoint two-update run on
free `gln01`; paid continuation and fresh validation remain closed. See
[`target_oriented_coupled_gpp_fresh_moment_admission_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_fresh_moment_admission_v1.md).

That free admission completed at commit `eb4eb19` in `1,519.041` seconds;
report SHA256 is `3e07fb8a...15f834`. Update 2 passes and improves direct
GPP/state/7-day/30-day/weighted losses by
`4.04281%/2.91495%/0.609017%/0.345835%/1.86632%`. After parameters advance in
memory and moments reset again, update 3 fails: GPP and 7-day science regress
`0.758192%/0.0372167%`, while state, 30-day science, and weighted loss improve
`4.32042%/1.01398%/0.766949%`. Its weighted raw gradient already has negative
cosine with the 7-day and 30-day gradients (`-0.274594/-0.0310940`), so this
is no longer explained solely by inherited AdamW moments. Both proposals are
finite, unclipped, physically valid, and fully discarded; recovery and fixed
parameters remain unchanged. The active gate is a zero-retention update-3
common-direction and finite-step audit at the in-memory post-update-2 state.
Paid continuation remains closed. See
[`target_oriented_coupled_gpp_fresh_moment_admission_result_20260902.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_fresh_moment_admission_result_20260902.md).

The zero-retention update-3 common-direction audit is now implemented and
frozen at contract SHA256 `956940a1...55cf09`. It reconstructs the accepted
fresh-moment update 2, computes the four update-3 source gradients and their
`4x4` Gram matrix, uses the existing tested minimum-norm unit-gradient simplex
solver, and compares direct versus fresh-AdamW realizations on the fixed
matched-distance scale grid `1,1/2,1/4,1/8,1/16`. Direct realization is
preferred; AdamW is admissible only if it preserves all four positive source
cosines. All proposals are discarded, numeric predecessor differences are
reporting-only, and fresh validation remains unread. Forty-three focused and
related tests plus static checks pass. The active gate is one free-`gln01`
execution. See
[`target_oriented_coupled_gpp_update3_common_direction_audit_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_update3_common_direction_audit_v1.md).

That free audit passed at commit `e704749` in `4,521.701` seconds; report
SHA256 is `642086af...ec8d`. The solved direct direction has minimum source
cosine `0.226096`, and every frozen scale from `1` through `1/16` improves
GPP, state, 7-day, and 30-day losses while passing every physical hard gate.
At full scale those improvements are
`0.9890%/6.5908%/3.7807%/6.5117%`, with weighted loss down `3.6710%`.
Fresh AdamW rotates the source cosines down to `0.02949`-`0.08313` and has a
nonmonotonic finite response: `1/2` and `1/4` regress 7-day loss. All proposals
were discarded and protected artifacts remain unchanged. The active gate is a
separate three-to-five-update free smoke that recomputes and directly realizes
the common direction at each accepted state. Paid training and validation
remain closed. See
[`target_oriented_coupled_gpp_update3_common_direction_result_20260902.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_update3_common_direction_result_20260902.md).

The direct-recalibration smoke is now frozen and locally qualified at contract
SHA256 `bb79be4c...76bb4`. From the accepted in-memory update-2 state, it tests
scheduled updates 3-5. At each state it recomputes the four source gradients
and minimum-norm direction, applies the direct full displacement norm
`0.00775897335105`, and retains the proposal only in memory when every source
and physical gate passes. It stops on the first scientific rejection and
writes no checkpoint. Thirty-six related tests plus Ruff, compilation, shell,
and whitespace checks pass. The active gate is one free-`gln01` execution;
paid training and validation remain closed. See
[`target_oriented_coupled_gpp_direct_recalibration_smoke_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_direct_recalibration_smoke_v1.md).

The corrected smoke completed at commit `a32db38` in `2,552.884` seconds;
report SHA256 is `0c2d3e21...2fd2`. Recalibrated direct updates 3 and 4 pass,
but update 5 rejects the same full displacement. Its common direction remains
strong (minimum cosine `0.495394`); state, 7-day, and 30-day losses improve
`28.6952%/9.9005%/10.6083%`, while GPP regresses `58.1509%` and weighted loss
regresses `9.1686%`. A derivative/full-response quadratic diagnostic places
the likely GPP crossing near scale `0.254` and local minimum near `0.127`.
The active gate is a zero-retention update-5 scale audit on
`1,1/2,1/4,1/8,1/16,1/32`; no architecture change, paid training, or
validation read is open. The earlier `40b66a2` launch is an engineering
timeout caused by float32-to-float64 proposal promotion; `a32db38` preserves
parameter dtype and removes that repeated compilation. See
[`target_oriented_coupled_gpp_direct_recalibration_result_20260902.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_direct_recalibration_result_20260902.md).

The bounded update-5 scale audit is frozen and locally qualified at contract
SHA256 `835a8ee8...69cbd`. It reconstructs the accepted update-3/update-4
states and rejected update-5 full proposal without an intermediate write,
then evaluates `1,1/2,1/4,1/8,1/16,1/32` on the same eight train/train
artifacts. The grid brackets the diagnostic GPP crossing/minimum at
`0.253654/0.126827`; only actual four-source and physical evaluations count.
No proposal or checkpoint is retained. Thirty-nine related tests and all
static checks pass. The active gate is one free-`gln01` execution; paid
training and validation remain closed. See
[`target_oriented_coupled_gpp_update5_scale_audit_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_update5_scale_audit_v1.md).

The free execution completed at commit `25ec5bc` in `2,696.660` seconds;
report SHA256 is `2199a0b7...5081`. The classification is
`bounded_direct_finite_step`: scales `1/4`, `1/8`, `1/16`, and `1/32` pass,
while `1/2` is rejected because GPP worsens `9.3058%`. At the largest accepted
scale `1/4`, GPP/state/7-day/30-day improve
`0.3069%/10.4394%/2.5014%/2.6021%`, and weighted loss improves `2.9544%`.
All physical and nonfinite gates pass; no parameters or checkpoint are
retained. The active gate is a frozen no-checkpoint scale-selecting
direct-recalibration smoke with per-update direction recomputation and
largest-passing-scale selection. Paid training and validation remain closed.
See
[`target_oriented_coupled_gpp_update5_scale_audit_result_20260902.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_update5_scale_audit_result_20260902.md).

The next scale-selecting smoke is frozen and locally qualified at contract
SHA256 `3b7109fc...8a5`. It reconstructs the selected update-5 `1/4` state and
continues through new updates 6 and 7. Each update recomputes the four-source
minimum-norm direction, evaluates the frozen scales in descending order, and
applies only the largest passing scale in memory. It stops on the first state
with no common direction or no accepted scale and writes no checkpoint.
Forty-two coupled-GPP/reachability tests plus all static checks pass. The
active gate is one free-`gln01` execution with a 120-minute timeout; paid
training and validation remain closed. See
[`target_oriented_coupled_gpp_scale_selecting_recalibration_smoke_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_scale_selecting_recalibration_smoke_v1.md).

The free execution passed at commit `aaeea61` in `3,402.216` seconds; report
SHA256 is `c8ae64e5...42a0b8`. Updates 5, 6, and 7 select scales `1/4`, `1`,
and `1/2`. Their GPP/state/7-day/30-day changes are respectively
`-0.3068%/-10.4393%/-2.5015%/-2.6018%`,
`-3.1720%/-4.3370%/-3.0809%/-4.5798%`, and
`-3.5180%/-4.4065%/-3.8337%/-3.2788%`. All physical gates pass, fixed
parameters remain exact, and nothing is retained. See
[`target_oriented_coupled_gpp_scale_selecting_recalibration_result_20260902.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_scale_selecting_recalibration_result_20260902.md).

The active gate is now the frozen no-checkpoint non-test 7/30-day transfer
audit at contract SHA256 `110d3b4a...ab571d`. It evaluates the state before
update 5 and the selected state after update 7 on the held-aside eight-artifact
train/train monitor. Both horizon means must improve and physical gates must
pass; per-artifact unanimity is not required. Local qualification and one free
`gln01` execution are next. Paid training and fresh validation remain closed.
See
[`target_oriented_coupled_gpp_scale_selected_transfer_audit_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_scale_selected_transfer_audit_v1.md).

The free transfer audit completed at commit `67a4d40` in `3,532.066` seconds;
report SHA256 is `b77e92fe...f625`. The selected updates improve independent
7-day and 30-day component-resolved science means by `8.3089%` and `13.4337%`;
`6/8` and `8/8` artifacts improve, respectively. Reachable state improves
`18.2362%`, and all physical/integrity gates pass. The frozen primary transfer
gate is therefore accepted. Direct teacher-state GPP nevertheless worsens
`39.2867%` and improves on only one artifact. See
[`target_oriented_coupled_gpp_scale_selected_transfer_result_20260902.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_scale_selected_transfer_result_20260902.md).

The persistent 100-update scale-selecting runner passed free `gln01` admission
at commit `7d57887`. The `4,669.897`-second run reconstructed update 7,
evaluated all 13 science keys, accepted scale `1` for one discarded proposal,
and removed its temporary recovery. Initial cumulative GPP scores remain
`0.417840` at 7 days and `0.497822` at 30 days; GPP, growth/autotrophic
respiration, NPP, NEP, and NEE fail the current absolute monitor thresholds at
both horizons. This is runner admission, not scientific promotion.

Paid job `14911154` was a zero-training Bash failure and was corrected at
`64185d8`. Corrected job `14911228` completed normally in `02:18:50` and
stopped at the frozen update-10 scientific gate. All ten updates decreased all
four sources on their own eight-artifact batches. On the fixed monitor,
reachable state and 7/30-day science improved `15.1638%`, `29.5626%`, and
`31.8176%`; cumulative 7/30-day GPP improved `30.2653%` and `33.9401%`.
Direct teacher-state GPP instead worsened `24.1562%`, with only `1/8`
artifacts improving. No checkpoint was selected; the retained recovery is
attribution-only.

The corrected zero-training direct-GPP audit completed on free `gln01` at
commit `52c60e7` in `7,117.794` seconds; report SHA256 is
`23327c3c...51cd9`. The first fixed-monitor regression is persistent update 2:
scheduled direct GPP improves `5.8362%`, while monitor direct GPP worsens
`2.2197%`. Their mean gradients have cosine `0.973434`, so a simple
scheduled-versus-monitor mean-direction reversal is ruled out. The applied
four-source descent has cosines only `0.154337` and `0.024148` with the two
GPP means, and artifact conflicts plus a positive first-order monitor cosine
followed by finite regression identify weak GPP protection and finite-step
nonlinearity. The ten-update monitor trajectory worsens `27.6762%`. Every
engineering gate passes and nothing is retained. See
[`target_oriented_coupled_gpp_direct_generalization_result_20260903.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_direct_generalization_result_20260903.md).

The update-2 direct-step transfer audit completed on free `gln01` at commit
`48eb1ae` in `3,935.020` seconds; report SHA256 is `df2593b...966c94`. The
largest train-only passing scale is `1/4`: all four scheduled sources and all
four disjoint guard-group direct-GPP means improve. The fixed monitor was read
only after selection and worsens by `0.5107%`, with `4/8` artifacts improving.
The classification is `train_only_scale_fails_monitor`. All engineering and
physical gates pass, allocator peak is about `4.10 GB`, and no state is
retained. See
[`target_oriented_coupled_gpp_direct_step_transfer_result_20260904.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_direct_step_transfer_result_20260904.md).

The multi-group direct-GPP direction audit completed on free `gln01` at commit
`c5c0a56` in `4,874.724` seconds; report SHA256 is `cf10f5e...663577`. The
minimum-norm solver finds a common eight-constraint direction with minimum
cosine `0.150436`. All six frozen scales improve all four scheduled sources,
all four train-only guard-group GPP means, and the combined guard mean, so the
train-only selector chooses scale `1/2`. The fixed monitor, read only after
selection, worsens by `0.4731%`; exactly `4/8` artifacts improve. The
classification is `multi_group_direction_fails_monitor`. All engineering and
scheduled physical gates pass, allocator peak is about `4.10 GB`, and no
state is retained. See
[`target_oriented_coupled_gpp_multi_group_direction_result_20260904.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_multi_group_direction_result_20260904.md).

The artifact-aware direction audit completed on free `gln01` at commit
`77d5a45` in `3,630.564` seconds; report SHA256 is `67bde6d8...09cbfe`. A
43-constraint common direction exists with minimum cosine `0.0422923`; its
21-constraint support includes all three scheduled non-GPP sources and 18 GPP
artifacts. Scale `1/2` improves all scheduled sources and every one of the 40
train-only GPP artifacts. Their combined mean improves `3.4706%`, and the
least-improved artifact still improves `0.4916%`. The fixed monitor, read only
after selection, improves `2.7567%`, with `8/8` artifacts improving. The
classification is `artifact_aware_direction_transfers_to_monitor`. All
engineering and physical gates pass, allocator peak is about `12.11 GB`, and
no state is retained. See
[`target_oriented_coupled_gpp_artifact_aware_direction_result_20260904.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_artifact_aware_direction_result_20260904.md).

The corrected three-update continuity smoke completed on free `gln01` at
commit `c345e64` in `5,467.939` seconds; report SHA256 is
`aa326bb4...52d69`. Updates 2, 3, and 4 select scales `1/32`, `1/16`, and
`1/2`. All three common directions pass, every selected step improves all
`40/40` train-only GPP artifacts and all four scheduled sources, and every
physical hard gate remains zero. The fixed-monitor mean remains below its
initial value after each step and finishes `6.2224%` lower; the final artifact
count is `7/8`. The classification is
`artifact_aware_three_update_continuity_passes_monitor`. See
[`target_oriented_coupled_gpp_artifact_aware_continuity_result_20260904.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_artifact_aware_continuity_result_20260904.md).

The subsequent artifact-aware persistent run completed on free `gln01` at
commit `714aa0d` in `5,434.218` seconds. Logical update 2 selected scale `1/2`
and was retained. Logical update 3 found a common first-order direction but no
tested scale passed all train-only gates, so the run stopped with recovery
through update 2 and no checkpoint. Its report SHA256 is
`ded2f4a5...7aa3a`; the rejected proposal SHA256 is `a8a983ca...de728`.

The exact-zero audit at commit `ca59ed5` then proves that the update-2
recovery reproduces all eight stored update-3 seven-day baselines exactly.
The historical `1/64` proposal worsens their aggregate by `0.5775%` solely
because artifact `550` worsens by `6.1243%`; all other artifacts improve and
all physical hard gates remain zero. Report SHA256 is
`c30f31a9...0795d`.

The localized trace completed at commit `e436b96` in `1,218.624` seconds;
report SHA256 is `cdca61cd...1f618e`. The fourth seven-day window is the sole
material regression. It has no discrete or defined-mask divergence. Two
mirror copies of snow-layer-2 `snowheat` first exceed `0.001` training scale
on Day 2, while the scientific discontinuity occurs on Day 5: GPP changes
from `2.767336` to `4.869900`, NPP by `+1.513847`, and AGB/BGB by
`+0.794635/+0.718573`. Both snowheat source families can reach every
structured-GPP branch through the inherited source bank, while the newer
carbon-coordinate adapter does not read snowheat. This proves reachability
and temporal precedence, not causality.

The matched snowheat intervention completed at commit `67fae75` in `890.625`
seconds; report SHA256 is `a19c6230...a15a2e23`. It passes trajectory, score,
factorization, and physical engineering gates. Replacing both treatment
snowheat mirrors with control same-day starts leaves the Day-5 GPP jump
unchanged: control is `2.767336`, free treatment and matched treatment are
both `4.869900`, for a retained fraction of `1.0`. The selected-window
displacement is also retained at `0.999999999997151`. The potential factor,
not limitation or correction, carries virtually the full GPP ratio.

The snowheat causal hypothesis is rejected. The subsequent Day-5 crossed
attribution completed on free `gln01` at commit `1c7dc9b` in `915.314`
seconds; report SHA256 is `c014ac09...7780c5`. All engineering gates pass.
The four cells are `CC=CT=2.7673363042` and `TC=TT=4.8698997235`, assigning
`100%` of the `+2.1025634193` GPP jump to the direct parameter map and exactly
zero to carried state or parameter-state interaction. All six state-owner
controls are null. At either Day-5 state, `structured_gpp` is `100%`
sufficient and necessary, while `coupled_gpp_carbon_feedback` contributes
zero. The parameter interpolation remains near `2.7673` through `1/2` and
jumps to `4.8699` at `3/4`; state interpolation is constant.

The subsequent zero-training structured-precision attribution completed on
free `gln01` at commit `587c3fd` in `935.288` seconds; report SHA256 is
`40c617ba...f5b693`, and all engineering gates pass. It inventories all
12,879,363 weights and 129 leaves. Of 8,974,854 changed coordinates, only
27.85% move one ULP; 72.15% are distinct intermediates at fraction `1/2`.
Float64 endpoint-weight interpolation retains the complete sharp jump in the
same `10/16` to `11/16` interval, with endpoint-jump magnitude ratio
`0.9999997364`. The accepted classification is
`structured_gpp_conditioning_persists_with_float64_weights`, rejecting
float32 endpoint quantization as the sampled cause.

Only `potential_blocks` is material: inserted alone it is `0.9999997732`
sufficient, and removing it captures `0.9999056398` of the effect. The final
potential output layer and all other top-level subtrees are immaterial. This
localizes the sampled pathology to the potential branch's existing residual
attention/FFN blocks, but not yet to one internal layer or operation. The
first internal audit then showed that the isolated jump moves from the full
tree's `10/16 -> 11/16` interval to `11/16 -> 12/16`; its component result is
therefore engineering-inconclusive.

The corrected internal v2 run passed on free `gln01` at commit `56ec8f2` in
`896.484` seconds; report SHA256 is `f781e4a9...07e0a3`. The full-float64
potential branch retains the sharp jump with magnitude ratio `0.9999999709`.
Block 0 is effectively fully sufficient and necessary, while block 1
parameters are immaterial. Block-0 attention Q and K are each sufficient but
neither singly necessary. The first discrete activation change is attention
head 7 switching from the `litterpart` state token at weight `0.459139` to the
`DOC` state token at weight `1.0`; the maximum float32 logit changes from
`8.1263` to `7796.5381`, and float64 repeats the collapse. The zero-training
head-slice and source-token causal audit is now frozen in
[`target_oriented_coupled_gpp_attention_head_token_attribution_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_attention_head_token_attribution_v1.md).
The head/token implementation then passed 35 related local tests at commit
`f885801`. Its free `gln01` execution passed at commit `7788d6b` in `904.470`
seconds; report SHA256 is `2c8a28d8...d94836`. Only block-0 head 7 is
sufficient (`1.0000001583`) and necessary (`0.9999992086`). Removing DOC only
from that head leaves lower GPP unchanged and reduces upper GPP from
`4.8699012434` to `2.7673204964`, capturing the full jump; removing
litterpart has zero effect. Q-only and K-only are redundantly sufficient, but
only paired Q+K removal is necessary.

The user-approved Q/K-conditioned candidate is now frozen and implemented. It
preserves the Transformer and changes only the two potential-attention layers:
projected Q/K vectors are independently normalized per 48-coordinate head and
their cosine score is multiplied by one bounded temperature for each of two
blocks and eight heads. The 16 new temperatures initialize at `sqrt(48)` and
remain below `12`. The complete candidate has `109,464,375` parameters, of
which `15,960,211` belong to the first-stage trainable boundary. It does not
hard-code head 7 or DOC, and all Value, residual, FFN, factorization,
conservative-update, and retained-tail paths remain unchanged.

The combined candidate and parent local regression passes `17` tests in
`804.14` seconds. Forward, complete loss, all temperature and inherited
potential Q/K/V/output gradients, fixed-reference agreement, non-GPP parent
identity, PFT behavior, canonical one-day state, and water/carbon budgets pass.
Ruff, Python compilation, shell syntax, and diff checks also pass.

The free-`gln01` real-v6 admission then passed at commit `812e851` in
`1753.274` seconds. It reproduces the parent jump from `2.7673221604` to
`4.8699012434`; candidate and fixed reference instead give `5.3164683410` at
both endpoints, eliminating the jump with maximum absolute logit `4.240962`.
The real-v6 complete loss, all `15,960,211` gradient coordinates, both
temperature blocks, both potential Q/K/V/output paths, all defined outputs and
next-state coordinates, and both budgets pass. No optimizer or checkpoint was
created. This proves the conditioning mechanism but does not establish that
the common endpoint is scientifically accurate.

The discarded update-3 preflight is now frozen and locally qualified at
contract SHA256 `b29e92b5...cb97d27d`. It loads the identity-exact accepted
update-2 recovery, attaches the 16 conditioned temperature coordinates, and
constructs a minimum-norm common direction from 40 individual direct-GPP
artifact gradients plus scheduled state, 7-day, and 30-day gradients. Its
step norm is calibrated from the conditioned architecture's own relative
gradient geometry rather than inherited from the parent. Descending scale
selection is train-only. If a scale passes, each of eight disjoint monitor
artifacts is loaded once and evaluated as an initial/candidate pair before
release. No proposal or optimizer state can be retained.

The corrected discarded update-3 preflight completed at commit `2adb422` in
`6,410.496` seconds. Its report SHA256 is `0aa30a0e...e9fe9b42a`, and the
classification is
`conditioned_discarded_update_transfers_but_absolute_science_remains_open`.
The common direction has minimum cosine `0.3265653`; its recalibrated full
delta L2 `0.0005178796` at scale `1` improves all `40/40` train-only direct-GPP
artifacts and all four scheduled sources. The train-only science macro falls
`1.3425%`.

All `8/8` excluded fixed-monitor artifacts also improve. Monitor direct GPP,
reachable state, 7-day science, and 30-day science fall `0.9373%`, `1.0276%`,
`0.7332%`, and `0.6816%`; the required science macro falls `0.7579%`. All
physical and engineering gates pass, reported allocator peak is about 12 GB,
and no parameter state is retained. Absolute GPP, NPP, growth respiration,
and autotrophic respiration still fail at 7 and 30 days, so model promotion
remains closed. See
[`target_oriented_coupled_gpp_conditioned_attention_discarded_update_result_20260906.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_conditioned_attention_discarded_update_result_20260906.md).

The current contract is a resumable three-update conditioned-Attention smoke,
SHA256 `ac65c5be...e5c9aff`. Updates 3-5 use 120 unique train-only artifacts,
recompute all 43 gradients and step calibration after every retained state,
and retain a candidate only after both step-local and cumulative fixed-monitor
gates pass. Failed candidates cannot overwrite the preceding recovery. The
absolute promotion gate is unchanged. Local/static qualification passes 57
related tests plus all static, CLI, contract, and shell checks. One
free-`gln01` execution completed at `45ab263` in 3162.530 seconds. Update 3
was retained; update 4 failed every scale because 30-day loss increased while
direct GPP, state, and 7-day losses improved. The subsequent original-panel
streamed integration completed; its remaining seven-day score discrepancy is
the active gate, as detailed below. The completed
`d9c4bd4` audit reproduces all eight forward loss pairs and all source norms;
seven artifacts agree with finite differences. Artifact 442 accounts for the
material mismatch (-0.019983 predicted versus +0.0000907 actual), with its
first window's gradient norm 1097.9 versus about 2 for the remaining windows.
Float32 gradient accumulation is not the material cause. Completed window
audit `2258c18` (3745.513 seconds) gives reverse -1097.9184 versus forward
tangent +2.6295 and central difference +2.6274. Its local-state summary had an
unmasked undefined NaN, so no physical process is yet identified. The active
same-scalar comparison reproduced the remat discrepancy but its no-remat
reverse requested 43.12 GiB of temporary buffers and failed on the 32 GiB GPU
after 5154.615 seconds. Full streamed BPTT run `0056141` completed in 1722.781
seconds: reverse +2.62694 and forward +2.62710 agree with canonical finite
differences +2.63857/+2.63083. Baseline hard failures are zero. One-day compiled
memory estimate is 1.50 GiB, not total measured VRAM. The largest local primal
absolute error (5.58793, day 2) and daily duality gap (0.000164441, day 16)
motivated saved-trace replay on days 2/15/16. That replay `4468c92` completed
in 502.559 seconds: snowheat relative difference is 5.43e-6, maximum GPP
difference 5.376e-7, and downstream linearized effects about 1e-9 to 1e-8.
Four-window discarded step `c726446` passed in 890.713 seconds. At norm 1e-4,
predicted change -0.000882413 matches actual -0.000882278; both magnitudes
improve all four windows, derivative errors are below 0.15%, and all hard
counts are zero. The original 40-artifact update-4 proposal with only its
7/30-day gradient callbacks replaced by full streamed backward has completed.
All original train-only gates pass at scale 1, and weights are discarded. Root cause
and general repair validity remain unproven; no model accuracy promotion.
The active contract is
`research/daily_coarse_graining/conditioned_attention_streamed_integration_20260908.md`;
25 local tests pass. Free `gln01` integration `1e6c521` completed in 2131.7955
seconds; wrapper PID `39230` has exited. Its complete remote report is
`runtime/outputs/diagnostics/conditioned-streamed-integration-1e6c521.json`.
All 40 direct-GPP artifacts improve (mean loss -0.95028%); scheduled state,
seven-day and thirty-day losses decrease 0.64089%, 0.64699% and 0.52441%.
Physical gates pass, but overall admission is false: seven of 64 seven-day
observation-score comparisons fail; all 32 thirty-day comparisons pass. The
largest difference is 1.83363843567e-5 at zero-based seven-day ordinal 37.
The active next task is source-level comparison and a bounded failing/passing
window replay if needed, tracing masks, fractions, trajectories and science
statistics before changing code. No tolerance relaxation, new architecture,
fixed-monitor read or retained continuation is implied. Accepted recovery
remains update 3. Absolute GPP/NPP/growth/total-respiration gates remain open.
The partial local SCP copy must not be used as complete evidence.
Source comparison finds the same extraction and loss formulas. The bounded
next replay is frozen in `conditioned_attention_seven_score_20260908.md`:
artifact 322/windows 5 and 1, compact/verbose trajectory statistics and
directional finite differences with the original eight-window totals.
Replay `c530f90` completed in 1153.55476 seconds. Compact trajectory scores
match production exactly; verbose scores reproduce the mismatch. Masks/counts
agree, all physical failures are zero, and directional derivatives agree
within 0.345% through production on both windows. Window-5 differences become
pronounced in day-5 maintenance, day-6 litter and day-7 GPP. The active
follow-up tests compact adjoint recording and source-mapped daily state
differences, with unchanged loss, tolerances and complete recurrent gradient.
Its completed report is `runtime/outputs/diagnostics/conditioned-seven-score-c530f90.json`.
Compact-tape replay `d31e93c` completed in 615.589 seconds. It is not admitted:
window 5 score drift is -7.37018e-6 and its smaller-displacement production/
observation derivative errors are 1.258%/1.828%. Window 1 passes; physical
failures are zero. The completed report is
`runtime/outputs/diagnostics/conditioned-compact-tape-d31e93c.json`.
The active source diagnosis is
`research/daily_coarse_graining/conditioned_attention_thermal_source_20260908.md`.
Its source inspection identifies learned float32 decoder outputs followed
by float64 affine snowheat/snowtemp updates, not a snow root solver. The
same-input replay separates fresh arithmetic differences from recurrence.
It completed on free gln01 at `ff9310c` in 1328.8 seconds, GPU 1, within its
30-minute bound and with no parameter retention. An earlier `8aa5331` launch stopped before model setup
because its progress file conflicted with the shared initializer; corrected
write ordering and the startup lifecycle test pass. Accepted recovery stays
at update 3; no production gradient change has been admitted. Same-input
one-day execution reproduces Day-1 snowheat drift. At Day 4, snow-only state
differences reproduce -211052.46 of the full -211069.69 layer-1 effect.
A free test01 read of existing artifact 322 shows both Teacher windows have
zero snow mass/depth/liquid/heat throughout; candidate heat reaches about
30--35 million by Day 7. The next priority is source-backed snow-state
consistency and conditioning, not another precision-only or layout trial.
The thermal report's fast-vector offsets were misindexed with compact PFT
widths; three fast-endpoint diagnostic fields are invalid. Independent
start-plus-decoded-delta checks reproduce all 84 actual retained snow
endpoints exactly. Canonical state and intervention results are unaffected.
The diagnostic offset correction has a full-versus-compact shape regression
test. See the named report for the scientific evidence and next boundaries.
The next correction is implemented at `e0fd6e3`, default off, under
`research/daily_coarse_graining/conditioned_attention_snow_free_boundary_20260908.md`.
It selects exactly snow-free, no-nobio, no-snowfall days with warm native
temperature endpoints; the warm condition excludes the source's freezing-dew
snow-creation path. The Attention body/weights and Teacher are unchanged.
Ten local state/budget/regime/thermal-gradient tests pass. At `c882529`, a
free-test01 read of 528 existing v6 day positions matches nine snow endpoint
families exactly (all selected, no real active-snow coverage). The bounded
zero-update 7/30-day paired screen completed on free gln01 GPU 1 in 1347.72
seconds, within its 45-minute timeout, report stem
`conditioned-snow-free-rollout-4d85256`.
The earlier `c882529` run stopped at report reduction after its first
reference window (278.52 s); host conversion of `HardCounts` and a complete
report-loop regression test correct that script error. All 18 related
mechanistic-output/thermal-source tests also pass.
All three repaired windows pass snow-state and production/recorded-score
checks, with no physical hard failure. Science composites change -2.79%,
+238.69% and +93.22%; the latter two GPP/biomass regressions preclude claiming
an accuracy gain. The next task is actual corrected-network streamed/JVP/FD
verification, then conditionally a discarded train-only descent check.
No new weights or gradient default are admitted. The correction remains
default off, and this forward-only bootstrap must not supply stale original
gradient helpers to training. No job from this screen remains running.
The successor corrected-head derivative runner and contract are implemented:
`research/daily_coarse_graining/conditioned_snow_free_gradient_20260908.md`.
All helper bindings receive the same corrected head; global bias totals are
recomputed with that head. Twenty-six targeted local tests pass, including
the complete mixed-precision three-window derivative/descent/report lifecycle
and actual HardCounts. The free gln01 GPU-1 run `e6a00b2` completed in
1402.1835 seconds, report/log stem `conditioned-snow-free-gradient-e6a00b2`.
All three cases and eight derivative pairs pass; six negative-gradient steps
lower their production linearized scores. Maximum relative derivative
discrepancy is 0.0650%, with no hard failures or mask changes. Reverse compiles
once (estimated buffers 1.50 GiB). This admits the corrected-head full
train-only discarded-step integration, not accuracy or retained training.
The local JSON is complete at `outputs/conditioned-snow-free-gradient-e6a00b2.json`.
The next contract is `research/daily_coarse_graining/conditioned_snow_free_integration_20260909.md`:
same 40 artifacts/43 constraints, all 96 horizon callbacks, corrected initial
and actual finite-step losses, unchanged science gates, one free GPU/90m.
Recovery remains at 3; paired fixed monitor is conditional on integration.
The corrected integration `90f6403` completed on free gln01 GPU 1 in 3363.2543
seconds. All 96 score checks pass and scale 1 is accepted, delta L2
0.000381141275313. All 40 direct-GPP artifacts improve (mean -0.94885%);
scheduled state/7-day/30-day losses change -0.64066%/-0.65962%/-0.58321%.
Detailed science macro decreases 0.64062%; all physical gates pass. Seven-day
ecosystem-carbon loss increases 0.009406%, inside its supporting cap. Absolute
GPP/NPP/growth/autotrophic-respiration remain unaccepted at both horizons.
Report/log stem: `conditioned-snow-free-integration-90f6403`, complete JSON
retrieved locally. No paid allocation or retained parameter update occurred.
Next is `research/daily_coarse_graining/conditioned_snow_free_monitor_20260909.md`:
reconstruct from the seven exactly nonzero source weights, verify against the
selected training result, save a separate diagnostic candidate, then evaluate
the existing eight-artifact monitor pair. No new tuning, independent-validation
claim or accepted-recovery mutation. This replay avoids recomputing zero-weight
gradients and repeating the direction solver/scale search.
The paired monitor `6ed00b0` completed on free gln01 GPU 1 in 1121.3835
seconds. Reconstruction and training-score replay pass, then all eight
monitor artifacts improve direct GPP (mean -0.89965%). All 18 science-key/
horizon losses decrease, macro -0.63715%; physical checks pass. Absolute
GPP/NPP/growth/autotrophic-respiration still fail at both horizons. This is
single-step development-monitor transfer, not final prediction accuracy.
The complete report is local at `outputs/conditioned-snow-free-monitor-6ed00b0.json`.
Its separate remote `.candidate.pkl` is saved and reusable; do not reconstruct
it again. Accepted recovery is still update 3. The next task is a frozen
bounded continuation with corrected-head reload/persistence and cumulative
monitor/absolute-science reporting; no continuation job is launched yet.
The user requested a training design. The proposal is now documented in
`research/daily_coarse_graining/conditioned_snow_free_continuation_plan_20260909.md`:
ten additional updates (logical 5--14), 400 distinct existing artifact IDs,
no fixed-monitor artifact overlap, same admitted optimizer/daily-gradient
method, automatic per-update monitor/recovery and 2/5/10 summaries. Estimated
hot update plus monitor is ~14 minutes from the previous report's first-use
overheads, not yet measured sustained throughput. The user subsequently approved
implementation, free-node checks and submission of the one-V100/8-CPU,
6-hour envelope, CNY 16.56 maximum. The continuation runner and exact 400-artifact
plan are implemented. Saved-candidate free-GPU admission passed in 518.64 s:
408 readable artifacts, unchanged reloaded parameters, reproduced corrected
monitor and all physical gates passing. Job **14979295** subsequently completed
normally in 02:49:07 at execution commit `7f13f10`: all ten new updates are
accepted and recovery is verified at logical update 14. Fixed development-monitor
direct GPP loss decreases 9.35793% and science macro decreases 7.34134%, with
every step improving and all 960 score comparisons passing. The final five
updates reduce direct loss another 5.47676%; there is no observed plateau.
The daily reverse compiles once. Absolute GPP/NPP/growth/autotrophic respiration
still fail at both horizons. Estimated additive actual-duration cost is CNY 7.78.
The longer continuation is now implemented under
`research/daily_coarse_graining/conditioned_continuation_v2_plan_20260909.md`:
20 new updates, 800 exposures / 678 distinct existing artifacts, stage/original
monitor protection, and paired 24-window independent evaluation. The growing
report-writing cost is measured on free test02 and bounded: late full event
write 0.4603s versus compact event 0.00244s. The local 38-test set passes.
Independent preparation on free test01 is complete (24 windows, 783.53 seconds).
Free-GPU admission at `b692915` passes in 1162.95 seconds, with update-14 monitor
reproduction, all 24 independent windows evaluated, zero physical hard failures
and zero new updates. Independent science macro normalized RMSE is 0.369501;
all six cells still fail absolute GPP/NPP/growth/autotrophic respiration gates.
This is a starting measurement, not paired transfer evidence. See
`research/daily_coarse_graining/conditioned_continuation_v2_admission_20260909.md`.
The user approved the 8h / CNY 22.08 run. Job `14984032` was submitted on
2026-09-09 at 22:32:51 Asia/Shanghai, initially PENDING / Priority. Slurm confirms
one GPU, eight CPUs, 32G host memory and no requeue. It completed on 2026-09-10
at 09:51:54, elapsed 5h08m, with all 20 updates accepted (recovery 34).
Development direct/science losses fall 16.8483%/12.5737%; independent macro
normalized RMSE falls 4.4875%. Paired transfer passes; joint/7 regresses 0.1962%,
the other five cells improve. All 1,920 score checks pass; daily reverse compiles
once. Absolute science still fails. See
`research/daily_coarse_graining/conditioned_continuation_v2_result_20260910.md`.
The checkout restriction is released; free inference at `3d94543` completed
all 24 Teacher/parent/current curves, reproducing 468 saved acceptance metrics
and counts. The initial shared GPU-1 export hit CUDA graph OOM; the unchanged
run on idle GPU 0 with 0.55 preallocation completed in 323.57 seconds. Actual
window-mean GPP Teacher/model is 3.51/4.53 and NPP 1.22/1.94 gC m-2 day-1;
maintenance is closer at 1.78/1.82. Stock levels are close but window changes
remain poorly fitted. Heterotrophic respiration is zero for both throughout
this panel. Curves and reading guide are in
`outputs/conditioned-science-fit-14984032/README.md`.
Results remain in `conditioned-snow-free-continuation-v2-14984032`.
Offline diagnosis on the same 444 validation days gives update-34 Teacher-state
GPP RMSE 1.74793 versus free-rollout 2.33039 gC m-2 day-1. Both improve from
update 14. Thirty-day NPP accumulated bias improves +33.1840 -> +29.1134 gC m-2;
AGB/BGB net-change RMSE improves 30.5028/19.8301 -> 29.4153/18.6226. Neither
direct fit nor recurrence is solved. Actual sampled training years are 1961-2004.
The user authorized preparing the next stage. The v3 plan proposes unchanged
updates 35-74 with continued sampling and embedded validation curves, under a
not-yet-approved 12h / CNY 33.12 paid envelope. See
`conditioned_gpp_feedback_diagnosis_20260910.md` and
`conditioned_continuation_v3_plan_20260910.md` in the daily research docs.
V3 admission completed on free gln01 GPU 1 at `ce7cd79`: 1236.23s, zero
updates/physical failures, no OOM, 24 embedded-curve windows. All 252 scalar
and 624 curve comparisons match update 34. Local 29 tests and real-parent
free CPU loading also pass. The server remains at this execution commit;
the user then approved the 12h/CNY 33.12 envelope. Gnall job **14991510** was
submitted on 2026-09-10 at 15:15:57 Asia/Shanghai, initially PENDING. Slurm
confirms one GPU, eight CPUs, 32GB host memory, 12h, no requeue and the project
working directory. The server must remain at `ce7cd79` while queued/running.
Results are in `runtime/outputs/training/conditioned-snow-free-continuation-v3-14991510/`.
It completed normally in 02:13:44 but stopped training early: updates 35-41
retained, update 42 rejected for one of 96 gradient-path score comparisons.
All 672 comparisons in retained updates pass. The failing 30-day pair differs
by 0.00020619 (0.8455%); root cause is unresolved, not proof of wrong gradients.
Development science macro improves 1.3702%; 30-day GPP/NPP cumulative RMSE
becomes 64.8228/44.7650, AGB/BGB net-change RMSE 29.0914/18.2379 gC m-2.
Physical hard gates pass; absolute science and planned-stage completion fail.
Update-41 recovery is preserved. Next isolate the one-window discrepancy on a
free node; no unchanged paid retry. See `conditioned_continuation_v3_result_20260910.md`.
Completed free gln01 diagnostics at unchanged ce7cd79 localize artifact 883/window
3's score failure to day-10 DOC input sensitivity, before reverse propagation.
First run (787.6s) reproduces both scores; full-state replay (604.8s) shows a
DOC-only state swap restores GPP 5.301189 from compact 5.590824. Discrete swaps
and the other 31 tested leaves do not change it. A free-CPU scale audit finds
DOC input about 0.048 standardized by 7.96e-12 into about 6.024e9; roughly
4.47e-9 raw path difference becomes 512 in float32. Attention input conditioning
for predicted off-Teacher-range states is the next gate; no repair/weight change
has been made. See `conditioned_score_discrepancy_20260910.md` for evidence and limits.
The user later reported that job 14991510 was running and approved parallel
coverage-expansion preparation. Exact train/train metadata counting finds the
678-package pool spans all 535 training locations/44 years but only 140,002
unique days out of 8,568,025 eligible (1.634%). Adding existing schedule entries
1001-3000 yields 539,171 unique days, a 3.851x increase; these counts do not audit
every inherited pretraining stage. Two new packages (1964/1990) were prepared
on free test03 and read back on test02, with canonical shapes and finite defined
GPP targets. Batch construction took 199.20/198.18s, total process 14m42s on a
shared node with 47% CPU availability. A single cnall job with four nodes and
ten one-CPU workers per node (40 total), 50 packages/worker and a 4h wall limit
was approved and submitted as **14992138** (maximum CNY 11.20). It started
2026-09-10 19:05:08 on `ibc11b08n[20-23]`; scheduler confirms 4 nodes/40 CPUs
and a hard end of 23:05:08. Projected worker memory is about 53 GiB per node.
Output: `runtime/outputs/training/conditioned-coverage-expanded-14992138/`;
log: `runtime/logs/data-expand_14992138.txt`. The job completed successfully
in 00:55:13, estimated CNY 2.58. All 40 workers and the free-test02 manifest
merge/readback pass: 2,000 artifacts (1001-3000), one structure, 6.726 GB.
Combined current-pool/extension availability is 539,171 unique days (6.293%);
the neural model has not trained on this extension yet. Local manifest is
`outputs/conditioned-coverage-expanded-14992138.json`.
No new-package neural forward or expanded-data GPU training has run. See
`conditioned_data_coverage_expansion_20260910.md` for the next concrete action.
The user's final-training reproducibility principle is now in `AGENTS.md` and
`NEXT_STEPS.md`: retain exploratory progress, but execute a documented complete
training recipe before final-method acceptance.
See `research/daily_coarse_graining/conditioned_snow_free_continuation_result_20260909.md` and
`research/daily_coarse_graining/conditioned_snow_free_continuation_execution_20260909.md`.
The completed preceding check is
`research/daily_coarse_graining/conditioned_attention_streamed_step_20260908.md`. See
`research/daily_coarse_graining/conditioned_attention_streamed_adjoint_20260907.md`;
the preceding result is in `research/daily_coarse_graining/conditioned_attention_derivative_modes_20260907.md`.
Previous evidence:
`research/daily_coarse_graining/conditioned_attention_update4_derivative_20260906.md`;
paid execution, fresh validation,
365-day rollout, Gate D2, sealed test, and product acceptance remain closed.
See
[`target_oriented_coupled_gpp_conditioned_attention_persistent_training_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_conditioned_attention_persistent_training_v1.md),
[`target_oriented_coupled_gpp_conditioned_attention_persistent_training_local_acceptance_20260906.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_conditioned_attention_persistent_training_local_acceptance_20260906.md),
[`target_oriented_coupled_gpp_conditioned_attention_discarded_update_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_conditioned_attention_discarded_update_v1.md),
[`target_oriented_coupled_gpp_zero_displacement_result_20260904.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_zero_displacement_result_20260904.md),
[`target_oriented_coupled_gpp_localized_jump_result_20260904.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_localized_jump_result_20260904.md),
[`target_oriented_coupled_gpp_snowheat_causal_intervention_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_snowheat_causal_intervention_v1.md),
and
[`target_oriented_coupled_gpp_snowheat_causal_result_20260904.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_snowheat_causal_result_20260904.md),
and
[`target_oriented_coupled_gpp_day5_crossed_attribution_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_day5_crossed_attribution_v1.md),
with the accepted result in
[`target_oriented_coupled_gpp_day5_crossed_attribution_result_20260904.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_day5_crossed_attribution_result_20260904.md),
the completed precision contract in
[`target_oriented_coupled_gpp_structured_precision_attribution_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_structured_precision_attribution_v1.md),
and its accepted result in
[`target_oriented_coupled_gpp_structured_precision_attribution_result_20260905.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_structured_precision_attribution_result_20260905.md),
the corrected internal contract in
[`target_oriented_coupled_gpp_potential_internal_attribution_v2.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_potential_internal_attribution_v2.md),
and its accepted result in
[`target_oriented_coupled_gpp_potential_internal_attribution_v2_result_20260905.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_potential_internal_attribution_v2_result_20260905.md),
the completed head/token contract in
[`target_oriented_coupled_gpp_attention_head_token_attribution_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_attention_head_token_attribution_v1.md),
and its accepted result in
[`target_oriented_coupled_gpp_attention_head_token_attribution_result_20260905.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_attention_head_token_attribution_result_20260905.md).
The candidate contract and local evidence are in
[`target_oriented_coupled_gpp_conditioned_attention_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_conditioned_attention_v1.md)
and
[`target_oriented_coupled_gpp_conditioned_attention_local_acceptance_20260906.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_conditioned_attention_local_acceptance_20260906.md).
The accepted free-GPU result is in
[`target_oriented_coupled_gpp_conditioned_attention_gpu_admission_20260906.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_conditioned_attention_gpu_admission_20260906.md).

The bounded persistent protocol remains recorded in
[`target_oriented_coupled_gpp_artifact_aware_persistent_training_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_artifact_aware_persistent_training_v1.md).
The first free launch at `d9f1238` stopped before GPU allocation because the
new contract transcribed the parent result SHA incorrectly. The repository's
existing authority and the server file both use `764b14e...81168`; the
contract now uses that value. No start, recovery, checkpoint, or report was
written by the failed preflight.
The completed smoke contract is
[`target_oriented_coupled_gpp_artifact_aware_continuity_smoke_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_artifact_aware_continuity_smoke_v1.md).
The completed one-step audit contract is
[`target_oriented_coupled_gpp_artifact_aware_direction_audit_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_artifact_aware_direction_audit_v1.md).
The completed predecessor contract is
[`target_oriented_coupled_gpp_multi_group_direction_audit_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_multi_group_direction_audit_v1.md).
The predecessor training evidence remains in
[`target_oriented_coupled_gpp_persistent_scale_training_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_persistent_scale_training_v1.md)
and
[`target_oriented_coupled_gpp_persistent_scale_training_admission_result_20260903.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_persistent_scale_training_admission_result_20260903.md),
with the launch correction in
[`target_oriented_coupled_gpp_persistent_scale_training_launch_failure_20260903.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_persistent_scale_training_launch_failure_20260903.md)
and the scientific result in
[`target_oriented_coupled_gpp_persistent_scale_training_result_20260903.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_persistent_scale_training_result_20260903.md),
with the completed attribution protocol in
[`target_oriented_coupled_gpp_direct_generalization_audit_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_direct_generalization_audit_v1.md).
The completed transfer contract is
[`target_oriented_coupled_gpp_direct_step_transfer_audit_v1.md`](research/daily_coarse_graining/target_oriented_coupled_gpp_direct_step_transfer_audit_v1.md).

The first free execution at `e755bb1` stopped after bootstrap because compact
direct-generalization replay records do not carry the persistent result's
`finite_step_accepted` field. No result or state was written. Commit
`4953939` switches to the accepted persistent records and performs metadata
validation before any GPU gradient computation. The completed scientific
contract was unchanged.

Job `14899347` at commit `8a1dffc` remains an engineering-failure record. Its
dictionary-order guard failed inside JAX `jit` before objective evaluation.
The exact-key-set repair and JIT regression were subsequently exercised by
the successful calibration; the failed job produced no report or checkpoint.
The completed training contract is
[`target_oriented_structured_gpp_joint_training_v1.md`](research/daily_coarse_graining/target_oriented_structured_gpp_joint_training_v1.md).
See
[`target_oriented_daily_operator_contract_v1.md`](research/daily_coarse_graining/target_oriented_daily_operator_contract_v1.md)
and
[`target_oriented_attention_v1.md`](research/daily_coarse_graining/target_oriented_attention_v1.md),
with the predecessor source-shaped contract in
[`target_oriented_source_shaped_maintenance_v1.md`](research/daily_coarse_graining/target_oriented_source_shaped_maintenance_v1.md),
with its accepted real-v6 execution evidence in
[`target_oriented_source_shaped_maintenance_gpu_admission_20260827.md`](research/daily_coarse_graining/target_oriented_source_shaped_maintenance_gpu_admission_20260827.md),
with its frozen train/train screen and zero-update result in
[`target_oriented_source_shaped_train_train_screen_v1.md`](research/daily_coarse_graining/target_oriented_source_shaped_train_train_screen_v1.md)
and
[`target_oriented_source_shaped_zero_panel_result_20260827.md`](research/daily_coarse_graining/target_oriented_source_shaped_zero_panel_result_20260827.md),
and
[`target_oriented_source_shaped_one_update_admission_20260827.md`](research/daily_coarse_graining/target_oriented_source_shaped_one_update_admission_20260827.md),
and its accepted direct-gradient engineering evidence in
[`target_oriented_source_shaped_branch_autodiff_admission_20260827.md`](research/daily_coarse_graining/target_oriented_source_shaped_branch_autodiff_admission_20260827.md),
with the completed paid-stage envelope in
[`target_oriented_source_shaped_branch_training_admission_20260827.md`](research/daily_coarse_graining/target_oriented_source_shaped_branch_training_admission_20260827.md),
and its completed scientific result in
[`target_oriented_source_shaped_branch_training_result_20260827.md`](research/daily_coarse_graining/target_oriented_source_shaped_branch_training_result_20260827.md),
with the frozen coupling-attribution contract in
[`target_oriented_source_shaped_coupling_attribution_v1.md`](research/daily_coarse_graining/target_oriented_source_shaped_coupling_attribution_v1.md),
and its accepted result in
[`target_oriented_source_shaped_coupling_attribution_result_20260828.md`](research/daily_coarse_graining/target_oriented_source_shaped_coupling_attribution_result_20260828.md),
with the frozen bridge method in
[`target_oriented_source_owner_state_bridge_v1.md`](research/daily_coarse_graining/target_oriented_source_owner_state_bridge_v1.md),
and its accepted result in
[`target_oriented_source_owner_state_bridge_result_20260828.md`](research/daily_coarse_graining/target_oriented_source_owner_state_bridge_result_20260828.md),
with the active locally accepted successor in
[`target_oriented_distribution_aware_maintenance_v1.md`](research/daily_coarse_graining/target_oriented_distribution_aware_maintenance_v1.md),
with its accepted real-v6 evidence in
[`target_oriented_distribution_aware_maintenance_gpu_admission_20260828.md`](research/daily_coarse_graining/target_oriented_distribution_aware_maintenance_gpu_admission_20260828.md),
with the completed bounded-training result in
[`target_oriented_distribution_aware_training_result_20260829.md`](research/daily_coarse_graining/target_oriented_distribution_aware_training_result_20260829.md),
with the accepted GPP causal result in
[`target_oriented_distribution_aware_gpp_bridge_result_20260829.md`](research/daily_coarse_graining/target_oriented_distribution_aware_gpp_bridge_result_20260829.md),
with the active zero-update coefficient method in
[`target_oriented_distribution_aware_guard_calibration_v1.md`](research/daily_coarse_graining/target_oriented_distribution_aware_guard_calibration_v1.md),
with GPU evidence in
[`target_oriented_attention_gpu_admission_20260821.md`](research/daily_coarse_graining/target_oriented_attention_gpu_admission_20260821.md)
and the training design in
[`target_oriented_bounded_training_v1.md`](research/daily_coarse_graining/target_oriented_bounded_training_v1.md)
and
[`target_oriented_state_anchor_amendment_20260821.md`](research/daily_coarse_graining/target_oriented_state_anchor_amendment_20260821.md).
The completed probe and derivative diagnosis are in
[`target_oriented_representative_gradient_probe_20260823.md`](research/daily_coarse_graining/target_oriented_representative_gradient_probe_20260823.md).
The accepted eight-batch evidence is
[`target_oriented_full_step_preflight_acceptance_20260824.md`](research/daily_coarse_graining/target_oriented_full_step_preflight_acceptance_20260824.md).
The active admission and submission evidence is
[`target_oriented_formal_training_admission_20260825.md`](research/daily_coarse_graining/target_oriented_formal_training_admission_20260825.md).
The formal scientific result is
[`target_oriented_formal_training_250_result_20260825.md`](research/daily_coarse_graining/target_oriented_formal_training_250_result_20260825.md).
The frozen continuation contract is
[`target_oriented_update_1000_continuation_v1.md`](research/daily_coarse_graining/target_oriented_update_1000_continuation_v1.md).
The accepted free continuation admission is
[`target_oriented_update_251_admission_20260825.md`](research/daily_coarse_graining/target_oriented_update_251_admission_20260825.md).
The accepted complete preparation is
[`target_oriented_update_1000_preparation_acceptance_20260825.md`](research/daily_coarse_graining/target_oriented_update_1000_preparation_acceptance_20260825.md).
The update-250 loss/gradient diagnosis and frozen update-1000 result analyzer
are in
[`target_oriented_update_250_training_dynamics_diagnosis_20260825.md`](research/daily_coarse_graining/target_oriented_update_250_training_dynamics_diagnosis_20260825.md).
The formal update-1000 result is
[`target_oriented_formal_training_1000_result_20260826.md`](research/daily_coarse_graining/target_oriented_formal_training_1000_result_20260826.md).
The completed zero-update diagnostic and result are
[`target_oriented_flux_gradient_variance_audit_v1.md`](research/daily_coarse_graining/target_oriented_flux_gradient_variance_audit_v1.md)
and
[`target_oriented_flux_gradient_variance_audit_result_20260826.md`](research/daily_coarse_graining/target_oriented_flux_gradient_variance_audit_result_20260826.md).
The completed component-resolved contract is
[`target_oriented_component_resolved_flux_basis_v1.md`](research/daily_coarse_graining/target_oriented_component_resolved_flux_basis_v1.md).
The current geometry result is
[`target_oriented_component_resolved_geometry_result_20260826.md`](research/daily_coarse_graining/target_oriented_component_resolved_geometry_result_20260826.md).
The completed virtual-step contract and result are
[`target_oriented_flux_protected_virtual_step_v1.md`](research/daily_coarse_graining/target_oriented_flux_protected_virtual_step_v1.md)
and
[`target_oriented_flux_protected_virtual_step_result_20260826.md`](research/daily_coarse_graining/target_oriented_flux_protected_virtual_step_result_20260826.md).
The completed residual-tail contract and result are
[`target_oriented_flux_residual_tail_audit_v1.md`](research/daily_coarse_graining/target_oriented_flux_residual_tail_audit_v1.md)
and
[`target_oriented_flux_residual_tail_result_20260826.md`](research/daily_coarse_graining/target_oriented_flux_residual_tail_result_20260826.md).
The completed gradient-path diagnostic and result are
[`target_oriented_flux_gradient_path_attribution_v1.md`](research/daily_coarse_graining/target_oriented_flux_gradient_path_attribution_v1.md)
and
[`target_oriented_flux_gradient_path_result_20260826.md`](research/daily_coarse_graining/target_oriented_flux_gradient_path_result_20260826.md).
The completed snowheat-only intervention is
[`target_oriented_snowheat_recurrence_intervention_v1.md`](research/daily_coarse_graining/target_oriented_snowheat_recurrence_intervention_v1.md).
Its accepted result is
[`target_oriented_snowheat_recurrence_result_20260827.md`](research/daily_coarse_graining/target_oriented_snowheat_recurrence_result_20260827.md).
The completed coupled intervention and result are
[`target_oriented_snow_thermal_recurrence_intervention_v1.md`](research/daily_coarse_graining/target_oriented_snow_thermal_recurrence_intervention_v1.md)
and
[`target_oriented_snow_thermal_recurrence_result_20260827.md`](research/daily_coarse_graining/target_oriented_snow_thermal_recurrence_result_20260827.md).
The completed mirror-aware intervention and accepted result are
[`target_oriented_snow_thermal_alias_recurrence_intervention_v1.md`](research/daily_coarse_graining/target_oriented_snow_thermal_alias_recurrence_intervention_v1.md)
and
[`target_oriented_snow_thermal_alias_recurrence_result_20260827.md`](research/daily_coarse_graining/target_oriented_snow_thermal_alias_recurrence_result_20260827.md).
The completed persisted-maintenance intervention and accepted result are
[`target_oriented_maintenance_state_recurrence_intervention_v1.md`](research/daily_coarse_graining/target_oriented_maintenance_state_recurrence_intervention_v1.md)
and
[`target_oriented_maintenance_state_recurrence_result_20260827.md`](research/daily_coarse_graining/target_oriented_maintenance_state_recurrence_result_20260827.md).
The completed non-recurrence audit and accepted result are
[`target_oriented_maintenance_head_conditioning_audit_v1.md`](research/daily_coarse_graining/target_oriented_maintenance_head_conditioning_audit_v1.md)
and
[`target_oriented_maintenance_head_conditioning_result_20260827.md`](research/daily_coarse_graining/target_oriented_maintenance_head_conditioning_result_20260827.md).

The update-1000 `source_routed_process_attention_v2` checkpoint has a strong
common first-order direction across 10 train/train groups and the four
one-day-anchor/NPP/AGB-tendency/BGB-tendency objectives. The group-aware
minimum-norm simplex audit has minimum cosine `0.050356` over all 40
constraints. Every tested global norm through `0.0003` improves every
component in all `10/10` groups, with zero hard or nonfinite failures.

The bounded step extension passes at norm `0.002`. All 40 group-component
losses decrease, with medians `-1.2176e-4` anchor, `-1.0193e-3` NPP,
`-1.3361e-3` AGB tendency, and `-3.4462e-4` BGB tendency. The least negative
individual change is `-1.8028e-5`; hard and nonfinite counts are zero. See
[`successor_group_aware_step_extension_20260815.md`](research/daily_coarse_graining/successor_group_aware_step_extension_20260815.md).

The efficient parent-frozen scalarization exactly reproduces the admitted
direction and applies one `0.002` update. A second fixed `0.002` proposal
regresses several constraints, worst group-5 NPP by `0.0861%`, and is stopped
before application. After the single accepted step, all six independent
7/30-day cells and all four NPP/AGB/BGB/total-live aggregates improve, but the
combined improvement is only `0.00922%`. See
[`successor_group_aware_scalarization_smoke_20260815.md`](research/daily_coarse_graining/successor_group_aware_scalarization_smoke_20260815.md).

Backtracking closes parent-frozen weights. At the second proposal, norms
`0.002, 0.001, 0.0005` regress the worst constraint by `8.6109e-4,
4.3088e-4, 2.1552e-4`. This near-linear scaling shows that the direction, not
just the finite step, has ceased to be common. See
[`successor_group_aware_backtracking_smoke_20260815.md`](research/daily_coarse_graining/successor_group_aware_backtracking_smoke_20260815.md).

The second-state exact recalibration audit completed at commit `1902936`.
The first update is reproduced to `4.96e-15` relative direction-norm error.
At that state, all 40 objectives retain a common direction with minimum cosine
`0.0486966`; norm `0.002` improves every loss and passes all component-effect
gates. The recalibrated support has 13 constraints, 12 overlapping the
parent's 14. Hard and nonfinite counts are zero, no validation values are
read, and no parameters are retained. Report SHA256 is
`c4ddb39b...10e97`. See
[`successor_second_state_recalibration_audit_20260815.md`](research/daily_coarse_graining/successor_second_state_recalibration_audit_20260815.md).

The bounded active-set implementation audit completed at commit `217876d`.
Recalibrating the 14 parent-support gradients closes all 40 directional checks
in one round with minimum relative derivative `0.0107166`; no constraint is
added. Norm `0.002` passes all ten groups and closely reproduces the exact
reference medians. Component reverse passes fall from 40 to 14, while hard
and nonfinite counts remain zero. Report SHA256 is `e2a9e0eb...daf97`. See
[`successor_active_set_recalibration_audit_20260815.md`](research/daily_coarse_graining/successor_active_set_recalibration_audit_20260815.md).

The bounded repeated active-set smoke completed at commit `b730731`. It
reconstructs the admitted first update and applies one new recalibrated norm-
`0.002` update. At the following state, the recalibrated direction remains
common across all 40 objectives with minimum relative derivative `0.0105908`.
Norms `0.001`-`0.002` regress at least one objective, while norm `0.0005`
decreases all 40 but misses the per-step `1e-4` useful-effect gate for anchor
and BGB medians. No hard or nonfinite failures occur. The two-update endpoint
improves all `6/6` non-test 7/30-day cells and every primary aggregate, but the
combined improvement is only `0.01822%`, below the `0.5%` promotion threshold.
Report SHA256 is `04c35474...613d7`. See
[`successor_repeated_active_set_smoke_20260815.md`](research/daily_coarse_graining/successor_repeated_active_set_smoke_20260815.md).

The active design task is a cumulative-microstep gate. It will recalibrate the
active set after every safe norm-`0.0005` microstep, preserve all-40
nonincrease at each microstep, and apply the unchanged four-component
`1e-4` effect gate to the cumulative macro-update. The four-microstep maximum,
early success stop, and unsafe-step stop are frozen under contract SHA256
`be46303e...2c545a`. Local admission passes 74 related tests and all static,
contract, and launcher checks. The free-`gln01` execution completed at commit
`b3d2f5f` and rejects the mechanism. The first microstep is safe and applied.
After it, the next recalibrated direction remains first-order common with
minimum relative derivative `0.0105646`, but its norm-`0.0005` finite step
regresses up to `28.7222%`, dominated by BGB tendency across multiple groups.
The step is not applied; no validation values or parameters are retained.
Report SHA256 is `7316749e...b674aa`. See
[`successor_cumulative_microstep_smoke_20260815.md`](research/daily_coarse_graining/successor_cumulative_microstep_smoke_20260815.md).

The active task is a zero-retention parameter-geometry attribution at this
stopped state. It must use existing architecture ownership to locate
direction-norm and relative-update concentration, then test a bounded set of
block projections only after they preserve all-40 first-order descent. No
checkpoint, architecture change, paid run, or further raw-direction
subdivision is authorized. The seven-block leave-one-out contract SHA256 is
`fa046011...c87153`; local admission passes 77 related tests plus static,
contract, and launcher checks. The free-`gln01` execution completed at commit
`c51dd87`. The output decoder contains `96.1526%` of squared direction norm.
Excluding it breaks common descent with minimum relative derivative
`-0.285257`; excluding each other block preserves first-order descent but
leaves the full `28.72%` finite-step regression essentially unchanged. No
top-level block is safely attributed. Report SHA256 is
`a7a170c7...c1a003`. See
[`successor_parameter_geometry_attribution_20260815.md`](research/daily_coarse_graining/successor_parameter_geometry_attribution_20260815.md).

The active design task is a four-leaf decoder attribution for the first and
second decoder kernels and biases. It must preserve the same stopped state,
all-40 first-order prefilter, and one norm-`0.0005` finite-step gate. A failure
of every leaf exclusion closes raw parameter-space direct steps and opens a
function/output-space trust-region design; it does not justify changing the
Attention architecture. The contract SHA256 is `08783a37...3cf721`; local
admission passes 79 related tests plus static, contract, and launcher checks.
The free-`gln01` execution completed at commit `39cd8ef` and finds no safe
leaf exclusion. The second decoder kernel carries `68.4533%` of full squared
direction norm; the second bias has the largest relative step at `0.645335%`
of its parameter norm. Both are required for common descent. Removing either
first decoder leaf preserves first-order descent but leaves the `28.72%`
finite regression. Report SHA256 is `513e4337...9b6c4d`. See
[`successor_decoder_leaf_attribution_20260816.md`](research/daily_coarse_graining/successor_decoder_leaf_attribution_20260816.md).

Raw Euclidean parameter-space direct steps are closed. The active design task
is now frozen as an output-locality audit comparing the 30-day per-field JVP
and exact norm-`0.0005` change at the same stopped state. All 181 fields use
their accepted train/train scales and day-start-delta representation. A
three-seed output VJP decomposes the NPP/AGB/BGB finite mismatch into decoded
output nonlinearity and downstream conservative-trajectory nonlinearity,
without per-field intervention rollouts. The contract SHA256 is
`edaebad5...7e9852`; 97 directly related local tests plus static, contract, and
launcher checks pass. See
[`successor_output_locality_audit_20260816.md`](research/daily_coarse_graining/successor_output_locality_audit_20260816.md).

The free-`gln01` execution completed at commit `2a29c2c`; report SHA256 is
`93ebbff8...1cfa85`. All ten groups, output-tape replay, JVP/VJP chain-rule,
state, budget, and nonfinite gates pass. Decoded-output nonlinearity accounts
for `96.08%` of NPP and `79.79%` of BGB finite mismatch.
`daily.fast_maintenance_respiration` carries `94.29%` of absolute NPP and
`95.11%` of BGB output-nonlinearity contribution. Its exact normalized RMS is
`0.08206` versus first-order RMS `0.000414`, and its maximum absolute change is
`4.71565` output scales. GPP contributes with the opposite sign and offsets
the regression.

The maintenance-local normalized output trust region is frozen under contract
SHA256 `84b96d4f...e35b72`. It uses the raw failed proposal as matched control,
the preceding safe microstep's exact maintenance change times `1.25` as the
cap, and one shared maintenance-displacement gradient for correction norms
`0.000125`, `0.00025`, and `0.0005`. GPP is explicitly uncapped. Local
admission passes 99 related tests plus static, contract, and launcher checks.
The first four-point-year reverse implementation exited before any candidate
or report after a `39,772,984,576`-byte XLA allocation request. The exact
gradient mean is now accumulated over 40 one-point-year reverse microbatches,
with a zero-displacement memory preflight before stopped-state reconstruction.
The free-`gln01` execution completed at commit `629238a`; formal report SHA256
is `64c68938...ce98c5`. All evidence checks pass with zero hard failures and
zero nonfinite values. The safe-reference maintenance RMS/maximum are
`0.0004166623`/`0.00402458`, defining caps `0.0005208279`/`0.00503073`; the
raw control reproduces RMS `0.08206115`, maximum `4.71565188`, and worst loss
regression `28.7222%`.

All three single-gradient corrections fail. Norm `0.00025` is the informative
result: maintenance RMS falls to `0.00363294` and the NPP/AGB/BGB medians all
improve, but RMS remains `6.98x` above its cap, maximum remains `15.33x` above
its cap, the anchor regresses `0.0177%`, and the worst loss regresses
`0.2733%`. The fixed maintenance gradient is useful but cannot jointly restore
the frozen constraints. The single-gradient projection is closed; do not
subdivide it. The active task is to freeze a bounded nonlinear constrained
solver that relinearizes violated constraints in a low-dimensional parameter
subspace and verifies each correction with exact finite evaluation. No
architecture change, optimizer update, checkpoint, validation read, sealed
test, or paid run is authorized. See
[`successor_maintenance_output_projection_protocol_20260816.md`](research/daily_coarse_graining/successor_maintenance_output_projection_protocol_20260816.md).

The successor low-dimensional nonlinear projection is frozen under contract
SHA256 `bb0f0718...d67ba07`. Its matched control is the informative
norm-`0.00025` correction. The candidate subspace has at most eight ordered
directions spanning raw progress, raw and relinearized maintenance repair,
the worst maintenance scalar, aggregate loss/effect repair, and two leading
individual loss violations. The subspace is fixed before at most four
sequential relinearizations and eight exact trials. Exact acceptance retains
the prior all-40, NPP/AGB/BGB, maintenance, state, budget, and nonfinite gates
and adds total-displacement bounds `[0.000125, 0.00075]`. Local implementation
and admission pass 95 related tests plus all static, contract-hash, and
launcher checks. The active operation is one free-`gln01` zero-retention
execution. See
[`successor_low_dimensional_nonlinear_projection_protocol_20260816.md`](research/daily_coarse_graining/successor_low_dimensional_nonlinear_projection_protocol_20260816.md).

The free execution completed at commit `85f1633` in `4905.58 s`, with report
SHA256 `4af8ddaf...5447ce`. Evidence is valid, the eight-dimensional basis and
matched-control reconstruction pass, and all hard/nonfinite counts are zero.
The matched point still violates 12 constraints. The bounded hard/slack solve
returned no admissible nonzero step, so no exact candidate trial was opened.
Fixed low-dimensional post-training projection is closed.

The Attention-only mixed-point annealed continuation completed as approved
`gnall` job `14569673` at commit `f702696` in `00:35:11`. All 250 updates and
1,000 unique point-years completed with zero hard failures. Median training
loss fell 4.7% from the first to last 50-update window and maintenance
aggregate validation RMSE improved 22.3%. The research-primary
NPP/AGB/BGB/total-live composite nevertheless became 6.98% worse; all four
primary aggregates and all six 7/30-day cells regressed. The final state is
retained but not promotion eligible, and update 1000 remains a diagnostic
reference only. See
[`successor_attention_mixed_annealed_continuation_result_20260816.md`](research/daily_coarse_graining/successor_attention_mixed_annealed_continuation_result_20260816.md).

The user rejected further local ranking around the weak update-1000 model as
the main research strategy. It is diagnostic evidence, not an incumbent that
defines success. The route-local-decoder A/B is superseded before
implementation.

The user approved `axis_aligned_process_expert_operator_v1`. The audit
behind it records 23,540 train/train point-years and 8,591,565 days versus only
500 point-years and 8,000 one-day anchors in the first fit. It also identifies
a deeper architecture bottleneck: all source/process Attention finishes at one
query per field before scalar coordinates are expanded, so large carbon
tensors cannot condition one output coordinate on its matching local input
depth/pool/PFT state. The proposal preserves dynamic axes through seven
process-owned expert paths and axis-local heads, targets 35-50 million
parameters, trains complete v6 one-day epochs before a 7/30-day curriculum,
and uses only absolute scientific gates. It has no matched control during
baseline qualification. The refined proposal keeps neural algebra and AdamW
in float32 but casts head outputs into the existing float64 constrained
physics, retained-tail, loss, and budget path. It also stops successively at
the structural, one-point-year, 64-point-year, 1%/10% epoch, one-day, 7-day,
30-day, and 365-day gates; every expert is scored independently so aggregate
loss cannot hide an unlearned process. See
[`strong_baseline_axis_aligned_process_expert_review_v1.md`](research/daily_coarse_graining/strong_baseline_axis_aligned_process_expert_review_v1.md).

Its contract is frozen at SHA256 `baf5791f...a96351d`. The executable has
49,750,681 float32 parameters, seven independent experts, source-restricted
forcing Attention, one shared factorized axial block, axis-aligned
static/process-static/state inputs, per-label heads, and an
equal-expert/equal-label loss. Fifteen local tests pass exact
schema and capacity checks, eager/JIT full forward, finite loss and JVP/VJP,
full/compact PFT mapping, forcing-source isolation, inactive-PFT isolation,
and a synthetic bound one-day conservative update with zero water/carbon
budget residual. The combined strong-baseline and relevant successor/parameter
routing regression run passes 39 tests.
See
[`strong_baseline_local_acceptance_20260816.md`](research/daily_coarse_graining/strong_baseline_local_acceptance_20260816.md).

The real-v6 zero-update admission passed. The subsequent one-point-year run at
commit `4b849fb` used all 364 valid train/train days for 100 epochs and 4,600
finite unclipped AdamW updates. Its report SHA256 is
`ada70c37...b4b399b`. GPP (`0.03391`), maintenance (`0.00697`), NPP
(`0.04041`), AGB tendency (`1.80e-5`), and BGB tendency (`3.60e-5`) meet their
declared NRMSE thresholds, with zero nonfinite, negative-inventory, or budget
hard counts. Complete process loss fell `97.60%`, below the `99%` gate, and
four of seven experts missed their `95%` reduction gates. The stock-scale
AGB/BGB metric was already below threshold at initialization, so it is not a
discriminating one-day gate. The qualification failed, retained no checkpoint,
and blocks the 64-point-year and paid stages. See
[`strong_baseline_one_point_qualification_result_20260816.md`](research/daily_coarse_graining/strong_baseline_one_point_qualification_result_20260816.md).

The bounded one-point loss-floor diagnostic contract and local implementation
are complete. The runner preserves the original 100-epoch recipe, adds at most
25 exact all-364-day updates through 46 size-8 gradient microbatches, reports
per-expert/per-label/per-field loss floors, loads the accepted train-only
one-day AGB/BGB tendency scales, and writes only a diagnostic parameter state.
Scientific rejection now exits normally and remains distinct from execution
failure. The related local suite passes 94 tests plus static checks.

The real-v6 preflight and full run then completed on free `gln01` at commit
`04b0c2e`. The final total reduction is `95.4744%`; the 25 exact-gradient
updates improve their starting loss by only `3.3112%` and do not cross the
original gates. Corrected GPP, maintenance, and AGB NRMSE pass; BGB
(`0.05064`) and NPP (`0.05694`) fail narrowly. All hard counts are zero. Five
labels account for `91.06%` of the final process loss, led by
`fast_owner.diffuco` and `fast_owner.driver` at `76.18%` together. Report
SHA256 is `2f338354...7ad5661`. See
[`strong_baseline_one_point_loss_floor_diagnostic_result_20260817.md`](research/daily_coarse_graining/strong_baseline_one_point_loss_floor_diagnostic_result_20260817.md).

The active task is a source/representation audit of the dominant endpoint
labels before any new architecture is implemented. No repeated stochastic
fit, paid job, 64-point-year fit, full epoch, 7/30/365-day screen, long-held
point, Gate D2, sealed test, or product acceptance is authorized.

That bounded audit is complete and subsequently corrected after tracing the
actual output transform and loss path. The original `39,019/260,988` claim was
wrong because it tested signed deltas against absolute endpoint ranges. The
corrected inverse-decoder audit finds zero hard range violations, 24,532 exact
0/1 boundary endpoints, and only 34 interior targets requiring absolute raw
above 20. True-v6 boundary calls still verify exact reconstruction of `z0m`,
`roughheight*`, `drysoil_frac`, `gtemp`, `stempdiag`, `tdeep`, and `hsdeep`;
`z0h` remains unresolved, while coupled energy primitives and canopy storage
remain learned. See
[`dominant_endpoint_source_representation_audit_20260817.md`](research/daily_coarse_graining/dominant_endpoint_source_representation_audit_20260817.md).

The exact-reconstruction claim now passes a 20-point-year non-test panel:
eight spatial-train points and two spatial-validation points in 1961 and 2006.
All eight admitted field families pass every point-year, with maximum error
`2.22e-15` or lower. `z0h` fails `20/20` and remains learned. Report SHA256 is
`0b906b0c...7925f74`.

The same panel closes all ten thermal coefficients from the complete daily
boundary with zero failed point-years under `1e-10 + 1e-12*abs(target)`;
report SHA256 is `adbdb831...6e579e8`. They are now verified exact derived
fields. A separate DOC panel corrects the capture semantics: both advective
and diffusive interface labels store nonnegative magnitudes, with direction
owned by water-flow sign or concentration ordering. The endpoint-trapezoid
DOC diffusion proxy reaches correlation `0.9833`, NRMSE `0.2119`, and residual
variance fraction `0.0450` overall and passes all four non-test slices. It is a
physical baseline plus learned residual, not an exact retained process. Report
SHA256 is `e200f2d7...8b69cbd`.

The user approved the complete `149/16/1/15` learned-primary, exact-derived,
physical-baseline/residual, and auxiliary/status classification. The formal
mechanistic-output implementation retains the axis-aligned seven-expert
Attention backbone, adds coupled surface-energy routing and capacity-aware
canopy storage, keeps `z0h` learned, and exposes all 181 fields after the
conservative update. It has 49,744,537 float32 parameters and 165 neural
fields/39,153 neural scalar outputs.

Eight local tests pass schema/capacity, eager/JIT primitive and post-update
forward, DOC identity/nonnegativity/gradient, thermal-kernel identity, complete
loss, all-parameter VJP/JVP, bound one-day state, and water/carbon budgets. The
bounded real-v6 `gln01` admission also passed. A fresh one-epoch 364-day smoke
then completed 46 finite AdamW updates; all prediction/state/budget/inventory
hard counts were zero and no checkpoint was written. Its report SHA256 is
`725130c4122801f56b9f22c6a0d7b6c2cf0794978d4586eb4b13f6c5bc73d304`.
The first formal 100-epoch run reduces total loss `99.1046%` and passes all
five primary-science and hard gates. It fails only the independent expert
reduction gates for canopy GPP (`93.9165%`), hydrology/snow (`83.0539%`),
litter/POC/DOC (`80.3503%`), and surface energy (`92.2124%`). Its report SHA256
is `b72ac45f...58daa0b`. A second detailed run localizes the residual to
hydrology endpoints, canopy debt, coupled energy endpoints, and DOC/litter,
and exposes sensitivity to the unclipped optimization trajectory. Its report
SHA256 is `86e7f64e...c7b539`. See
[`mechanistic_output_one_point_qualification_result_20260817.md`](research/daily_coarse_graining/mechanistic_output_one_point_qualification_result_20260817.md).

The 20-point-year non-test hydrology/canopy boundary audit is complete. Final
`nroot` is history dependent and its one-boundary reconstruction passes only
`5/20` point-years. With true final `nroot`, final `us`, `humrelv`, and
`humrel` pass `20/20` at maximum error `2.22e-16`; `evap_bare_lim` is exact
from tile `evap_bare_lim_ns`. `temp_hydro` fails every point-year with maximum
error `9.88667 K` and remains learned. The panel includes 101 under-`mcr`
events and therefore covers that active mask behavior.

Canopy storage stays within capacity on all 7,290 endpoints. Signed canopy
debt is positive on `41.51%`, and daily `vevapwet` includes negative
condensation values, so the old rare-debt wording and wet-evaporation hard
scale are rejected. See
[`hydrology_canopy_boundary_audit_result_20260817.md`](research/daily_coarse_graining/hydrology_canopy_boundary_audit_result_20260817.md).

The user approved and the code now implements the hydrology/canopy v2 output
mechanism. The frozen v2 contract learns 161 fields/39,006 scalar coordinates,
derives 20 fields exactly, and contains 49,743,001 float32 parameters. Final
`nroot` is a support-aware sparse simplex coordinate; final `us`, `humrelv`,
`humrel`, and grid `evap_bare_lim` are materialized after the conservative
update. Canopy uses one signed final endpoint plus storage/zero/debt status
logits and a unique deterministic conversion to the original conservative
transfers. The old four-transfer MSE is replaced by normalized endpoint MSE
and categorical status loss.

The v2 local gate passes 60 focused and related regression tests, including
eager/JIT forward, complete loss, Attention-path VJP/JVP, endpoint conversion,
one-day state, and water/carbon budgets. Ruff, compileall, and diff checks also
pass. The subsequent zero-update real-v6 `gln01` admission also passes. Its
complete 49,743,001-element gradient is finite with L2 norm `208.5265`; all
181 outputs and declared next-state values are finite, and water/carbon budget
residuals are exactly zero. JAX peak bytes in use are 1.441 GB despite the
roughly 25 GiB allocator reservation visible in `nvidia-smi`. Report SHA256 is
`cd376e48...ccd0d8`.

The one-epoch training smoke then passes all preregistered gates. All 46
batch-8 AdamW updates are finite, complete process loss decreases `7.3818%`,
all seven experts improve, masks/counts are unchanged, and all state/budget/
inventory hard counts remain zero. GPP, maintenance, and NPP NRMSE improve to
`0.1390`, `0.2301`, and `0.1253`. Report SHA256 is
`61de8c4c...a48b5c0`.

The formal 100-epoch/4,600-update run then completes with no execution failure.
Total loss falls from `4.5707906` to `0.0498893` (`98.9085%`). All five
primary NRMSE and all hard gates pass, but hydrology/snow (`93.7780%`),
canopy/GPP (`93.4923%`), surface energy (`89.3930%`), and litter/POC/DOC
(`80.2522%`) miss their independent `95%` reduction gates.

The following zero-training attainability audit passes all 364 days. Maximum
learned-field normalized MSE is `2.0612e-9`, maximum derived exact-field
normalized MSE is `9.8733e-22`, and DOC plus every hard count are zero. The
thermal source panel finds no physical-tolerance source failure; finite
conservative arithmetic in `mc/mcl` explains the old coefficient-level
physical-tolerance failures.

The subsequent isolated diagnostics complete all 4,600 updates per expert.
Surface energy reaches `97.9259%` aggregate reduction and passes every one of
its 11 dominant-field gates. Litter/POC/DOC reaches only `91.8014%`; 12 of 28
dominant fields fail, led by litter-above incoming amount, coupled OK_LEAK
auxiliary endpoints, DOC incoming amount, adsorption, and advective transfer.
Coverage, count identity, state, budget, finite-value, and nonnegative-inventory
gates pass, and no checkpoint is written. The active gate is an architecture
discussion for that litter/POC/DOC mapping. Another network, complete training
run, 64-point-year, full-v6, rollout, Gate D2, sealed-test, and product stages
remain closed. See
[`mechanistic_output_v2_isolated_expert_memorization_result_20260818.md`](research/daily_coarse_graining/mechanistic_output_v2_isolated_expert_memorization_result_20260818.md).

## First Formal Neural Candidate

`axial_process_operator_v1` and its exactly parameter-matched
`factorized_process_control_v1` are implemented at 5,464,321 float64
parameters. The candidate uses native-record forcing attention and four
scientific-domain axial process blocks; the control replaces interactions
with token-local transforms and masked pooling while retaining the same
parameter tree. Neither arm inherits a rejected model or checkpoint.

Local regression and bounded real-v6 `gln01` forward, loss, JVP/VJP, PFT
permutation, inactive-slot, conservative budget, retained-tail, and
100-update tiny-fit gates pass. The retained tail uses native interval
closed-form reductions and does not construct 48 forcing records or execute a
48-step state scan.

The first formal matched seed pair completed. The axial candidate job
`14507225_0` reached its best score `0.8219892971` at update 8750; the
factorized control job `14508831_1` reached `0.8433795294` at update 9000.
The candidate improved the overall runner macro-MSE score by `2.54%`, temporal
by `5.08%`, and spatial by `1.83%`, while regressing the joint slice by
`4.41%`. Both trained stably with nearly identical final training loss.

The user stopped the experiment for practical equivalence after this first
pair. Seeds `20260808/20260809` and the pair's 7/30-day rollout will not be
run; neither arm is promoted. Because the original contract named NRMSE while
the runner reported macro MSE and did not define three-seed aggregation, this
is not a formal equivalence claim or a complete-protocol rejection. See
[`first_neural_process_head_first_pair_decision_20260808.md`](research/daily_coarse_graining/first_neural_process_head_first_pair_decision_20260808.md).

The full train/train process statistics artifact remains accepted. A
vectorized, deterministic, partition-mergeable fitter
covers all 101 named process outputs plus semantic water/carbon/thermal input
scales, with PFT-shared moments and hash-bound loading. Approved `cnmix` job
`14506391` completed all eight fit steps over 8,591,565 rows and 23,540 source
shards. The batch failed after 31:22 only because its merge CLI passed one
unsupported argument; commit `9e8d4e9` fixed that wiring, after which the
retained partials merged and independently reloaded on `test01` without a
rescan. The final JSON SHA256 is
`9bc2cab43b3279e39a0fc2876ca298b2b371d6890986d44ef035859ca925eaab` and
the NPZ SHA256 is
`addf957afbd8338f4aa250155dcd299d568c1070b644e30279f9bf0ea37f9bf8`.

The final `gln01` no-checkpoint training preflight passed at commit `492f2c0`.
Both arms completed 100 real changing train/train updates at about
`0.170-0.171 s/update`, with finite loss, gradient norm, and clip scale. Each
of the temporal, spatial, and joint non-test validation slices completed its
bounded four-batch probe; sealed test data remained unread. The report SHA256
is `127ef3289af776d5c5bb994f9129f5d220aa8b8574b18c50e7e8a77aba739277`.
The source-backed long-horizon objective foundation and ownership-aware
parameter boundary are now implemented. The latter separates named
`vcmax25`/maintenance fast factors from retained `alloc_min`/`residence_time`
and also routes `vcmax25` to retained daily `vmax_step`. The closed first A/B
did not consume its packed fast-process parameter columns and cannot support
parameter-conditioned or Gate-D2 claims. See
[`daily_operator_parameter_routing_v1.md`](research/daily_coarse_graining/daily_operator_parameter_routing_v1.md).
The user selected source-routed field-level attention against an exactly
matched source-graph MLP control. The v1 contract is frozen at SHA256
`bb8bb349...92b5c`; both implemented arms contain exactly `8,679,681`
parameters and share the 20-family source registry, 101 field queries,
process graph, decoder, parameter routing, and loss boundary. Local eager/JIT
forward, loss/JVP/VJP, PFT permutation, inactive isolation, fast/retained
parameter gradients, one-day conservation, synthetic free rollout, 5+7
trajectory extraction, and bounded train/train coefficient calibration pass.
The inference audit reports zero violations. See
[`successor_process_head_local_foundation_20260809.md`](research/daily_coarse_graining/successor_process_head_local_foundation_20260809.md).
The real-v6 lifecycle smoke is not accepted. On Day 2 it loses 37
Teacher-defined values from the active vegetation slot. Initialization and
baseline probes do not repair the failure, and replacing all 101 named outputs
with Teacher values still reconstructs only `87/160` canonical fast-target
leaves and loses the same 37 next-state masks. The failure is therefore in the
named-output-to-fast-boundary adapter, not evidence against either attention or
the matched graph control.

Commit `e705b68` adds a 160-leaf ownership ledger. It identifies 14 leaves/113
columns where daily-integrated typed fluxes are written into same-named
instantaneous state, and 106 leaves/1,204 columns still carried from day start.
A read-only `test01` scan of 18 existing v6 annual shards and 6,564 transitions
shows 62 of those carried leaves, totaling 1,007 columns, actually change
within the day. Gate C2 had required non-inventory fast fields to remain
explicit endpoint reconstruction targets; the current 101-output schema does
not implement that layer. See
[`successor_fast_boundary_coverage_20260809.md`](research/daily_coarse_graining/successor_fast_boundary_coverage_20260809.md).

The revised successor fast-owner reconstruction schema is accepted at machine
contract SHA256
`4db5f828442212c58903096c36015f9d769c19aeaaa4eb9a4326fbea2fdfc4c9`.
It assigns all 160 leaves/2,855 columns exactly once to 21 existing endpoints,
19 derivations, 30 direct mirrors, 11 guarded/static carries, and 79 new neural
value heads. The added neural width is 839 values plus two `rveget` status
logits; the 14 typed-integral/state collisions are explicitly forbidden. The
contract and full 79-row field table are in
[`successor_fast_owner_reconstruction_contract_v1.md`](research/daily_coarse_graining/successor_fast_owner_reconstruction_contract_v1.md).

The non-neural JAX materializer is now wired as an opt-in stage inside the
canonical retained-tail adapter. Local daily-operator/successor regression is
`79/79`. Free-`gln01` detached commit `4648a00` reached `160/160` exact leaves
for cold Day 2, ordinary Day 3, and restart 1962 Day 1, with exact masks,
discrete state, and `rveget` status plus passing retained lifecycle and
water/carbon budgets. Its report SHA256 is
`f3ae02d08c950bbf7269c8ba618b3fb34e4ed88d14c3d264a9751cd9f360e084`.
No training, checkpoint, Teacher regeneration, or historical v5 data was used.

The user approved the identical candidate/control decoder extension. It adds
79 value queries and one `rveget` logit query while preserving the original
101 fields. Both matched arms now contain exactly `8,722,689` parameters and
share a 181-field/48-label schema, physical constraints, normalization,
dynamic mask loss, and materializer. Sixty-five fast-state targets are
day-end-minus-safe-day-start deltas; 14 daily/OK_LEAK targets are absolute.
Loss group, process domain, and source route are explicit independent
metadata. The attention candidate and fixed-graph control both pass eager/JIT
forward, loss/JVP/VJP, inactive-PFT, one-day conservation, 160-leaf
materialization, and inference-boundary audits. Full local evidence is in
[`successor_fast_owner_decoder_local_acceptance_20260809.md`](research/daily_coarse_graining/successor_fast_owner_decoder_local_acceptance_20260809.md).

The added-head train/train statistics artifact is accepted. Approved `cnmix`
job `14515519` completed eight partitions and merged 8,591,565 train/train
rows from 23,540 source shards in 8 minutes 57 seconds. A fresh `test01`
process independently reloaded 79 added fields with zero never-defined
columns and finite moments, then assembled complete 181-field normalization
and 180-field value-loss scales. `rveget` was defined for 4,771,338 of
17,183,130 eligible slots. The final JSON SHA256 is
`b2e61a171156071d404d12b833130f2d0e98064564d1c3e007e16f0210734719`
and the NPZ SHA256 is
`890d1fd6110912a835c760b4fe58e9320481a929797858cad5b70b1bf03eb99f`.
See
[`successor_fast_owner_statistics_acceptance_20260809.md`](research/daily_coarse_graining/successor_fast_owner_statistics_acceptance_20260809.md).
The free `gln01` no-checkpoint GPU probe is also accepted at commit `7c555d1`.
Both matched arms produced 181 finite constrained fields and finite fitted
losses; all 155 parameter leaves and 8,722,689 gradient elements were finite
and nonzero. All 78 PFT-axis outputs were exactly zero in the disabled slot.
The canonical transition passed with water/carbon residual magnitudes below
`1.1e-14`. The report SHA256 is
`5bb1f885367b3fc5183216c476a6bbe8d37122cb698d990bd5ca346006c0aaa7`.
See
[`successor_fast_owner_gpu_probe_acceptance_20260809.md`](research/daily_coarse_graining/successor_fast_owner_gpu_probe_acceptance_20260809.md).
The free bounded-small-fit preflight is accepted at commit `d1ae419`. Both
real-v6 Teacher replay days pass all state, mask, representation, discrete and
budget gates. The candidate coefficient preflight has finite nonzero gradients
and bounded coefficients; both matched arms complete one update with all
8,722,689 gradient elements finite, all 155 parameter leaves nonzero, finite
updated parameters and zero rollout hard failures. The report SHA256 is
`1e79bc76b97098ecf0725d94edfe144363420dba18792f05105e8f0707d543b6`.
The earlier production runner and two-update probe proved same-process
cross-shard reuse but spent 86.57 seconds per hot entry on host preparation.
Its 20-hour paid request is rejected and its Slurm script is removed.

Commit `1c63494` replaces per-update GPU-host preparation with hash-bound CPU
artifacts. The accepted two-entry manifest totals 66,868,764 bytes and binds
the source HEAD, contract, schedule, trace signature and per-file SHA256. The
fixed-seed cold `gln01` report passes both arms and all hard gates. Its hot
switched entry loads in 0.235 seconds; replay, attention and control take
1.311, 1.138 and 0.879 seconds. Report SHA256 is
`481581d1648699e140fa12c1b6a8e342f154ecb2216473f01a6916bd95c31f60`.

Cross-process persistent cache reuse is not accepted. A new process with the
same fixed hash seed and completed cache still recompiled `jit_train_step` and
was stopped by its 10-minute hard limit. See
[`successor_small_fit_prepared_batch_acceptance_20260810.md`](research/daily_coarse_graining/successor_small_fit_prepared_batch_acceptance_20260810.md).

Commit `507cb61` then froze the four accepted train/train objective
coefficients behind a source-report and first-artifact identity contract. A
fresh two-entry manifest binds the new HEAD while retaining byte-identical
artifact payloads. The cold `gln01` probe completed in 1,442.46 seconds, down
465.92 seconds (24.4%) from the 1,908.38-second baseline. Coefficient loading
took 0.018 seconds with no recalibration. Final zero-rate evaluations reused
the compiled training steps in 1.055/0.894 seconds and added no executable.
Both remaining cold train-step compile/updates took about 581 seconds; hot
attention/control updates remained 1.208/0.894 seconds. All gradients were
finite and all hard gates were zero. Report SHA256 is
`a9ab43d855a4c035cbc52f978717de9737f801ef349981a19b50e22d2056bdc9`;
see
[`successor_small_fit_compile_reduction_acceptance_20260810.md`](research/daily_coarse_graining/successor_small_fit_compile_reduction_acceptance_20260810.md).

Commit `50bc99a` implements a four-worker resumable CPU preparation boundary
and a fail-closed finalizer. The two-worker `test01` smoke passes with
byte-identical artifact hashes, about 4.8 GB peak RSS per worker, approximately
70 seconds of artifact work per worker and exact merged coverage. Its final
manifest SHA256 is
`f73c714a7f9e24699f7775b5c23f221a7eda4e1b1fdb81621f927bede26b0f31`.
The bounded resource plan requests four one-core `cnmix` tasks for four hours
(CNY 1.12 ceiling), then only after manifest admission requests 16 CPU cores
and one `gnall` GPU for two hours (additive CNY 6.64 ceiling). See
[`successor_small_fit_production_resource_plan_20260810.md`](research/daily_coarse_graining/successor_small_fit_production_resource_plan_20260810.md).

Approved `cnmix` job `14517346` completed in `01:58:09`; all four preparation
workers and the finalizer exited zero. The final product contains exactly 500
artifacts totaling 16,717,190,876 bytes. A separate `test01` process rehashed
every artifact and passed exact coverage, size, SHA256, path, HEAD, schedule
and policy checks. Manifest SHA256 is
`0182b7f7a6d5eae09025a336f26bc67ba5fb54706c3529a846d3885c16d80462`.
See
[`successor_small_fit_prepared_500_acceptance_20260810.md`](research/daily_coarse_graining/successor_small_fit_prepared_500_acceptance_20260810.md).

The pre-existing runner would have discarded both trained parameter trees at
process exit, making its named later 7/30-day screen impossible without
retraining. A checkpoint handoff amendment now preserves the immutable
producer identity at `d493699` while recording a distinct consumer commit.
Only a complete, passing 500-update fit may atomically publish both arms,
including parameters, optimizer state, hashes and tree schemas; failed and
submaximal runs publish nothing. Local contract, round-trip, tamper, incomplete
fit and injected second-arm failure tests pass `23/23`. See
[`successor_small_fit_checkpoint_handoff_v1.md`](research/daily_coarse_graining/successor_small_fit_checkpoint_handoff_v1.md).

The free `gln01` GPU-array serialization/reload preflight passed at consumer
commit `c15cd86`. Both arm trees originated on GPU, survived atomic host
persistence and verified reload exactly, and were removed with the temporary
directory. The persistent report SHA256 is
`a923d6371473bc19bef9ea89e3a390a4798b02ac13e23782d2520ab22616cf5a`.
It records zero training updates and no Teacher/v5 access; see
[`successor_checkpoint_handoff_gpu_probe_acceptance_20260810.md`](research/daily_coarse_graining/successor_checkpoint_handoff_gpu_probe_acceptance_20260810.md).

Approved paid fit job `14517771` started but failed closed after `00:02:52`
with exit code `1:0`, before the training loop and before any checkpoint
publication. The output directory is empty. The loader incorrectly required
the two-update coefficient-source schedule hash to equal the complete
500-update consuming schedule hash even though their coefficient-defining
first artifact is byte-identical. The schedule-ownership repair and a new
CPU-only exact-production admission probe pass locally; no coefficient,
objective, data or gate changed. See
[`successor_small_fit_paid_admission_failure_20260811.md`](research/daily_coarse_graining/successor_small_fit_paid_admission_failure_20260811.md).

While that job waits, the complete successor 7/30-day acceptance package was
frozen before any fit result. Contract SHA256 is
`99ec6eb0...0b6fc07`. It hash-admits both successful 500-update checkpoints,
selects a deterministic non-test panel, prepares host artifacts on free CPU
test nodes, runs same-boundary matched 7/30-day free rollouts, and applies
biomass/productivity, bias and exact hard gates. Focused local tests pass
`12/12`. At package-freeze time no rollout screen had run and no production
checkpoint existed.
See
[`successor_7_30_rollout_screen_package_20260810.md`](research/daily_coarse_graining/successor_7_30_rollout_screen_package_20260810.md).

The repaired zero-update production-asset admission has now passed on
`test01` at commit `0ed9891` against the exact 500-entry manifest, formal
schedule, first artifact, coefficient source and checkpoint handoff. Report
SHA256 is `50d090e6...18ea089`; no training, checkpoint, v5, sealed-test or
Teacher-regeneration path was used. See
[`successor_production_fit_asset_admission_20260811.md`](research/daily_coarse_graining/successor_production_fit_asset_admission_20260811.md).

The replacement fit is complete. Paid job `14520379` resumed from update 155
and completed both arms through update 500 with every update applied, finite
gradients, and zero hard counts. Its final process exit was a runner-only JIT
reuse misclassification: recovery host arrays legitimately created one second
device specialization at update 157, followed by zero new executables through
update 500. The complete update-500 recovery was retained.

Commit `d834d5d` accepts only that bounded post-recovery specialization and
supports finalization from a completed recovery without another training
update. The free-`gln01` finalizer published both checkpoints atomically. Fit
report SHA256 is `ea4b66cd...a259a`; checkpoint manifest SHA256 is
`f8eed186...75e40`. A separate `test01` reader rehashed and loaded both
checkpoints, verifying 500 updates, 8,722,689 parameters and 17,445,379
optimizer elements per arm, exact tree schemas, identities and finite values.
See
[`successor_small_fit_500_checkpoint_acceptance_20260811.md`](research/daily_coarse_graining/successor_small_fit_500_checkpoint_acceptance_20260811.md).

The frozen three-window 7/30-day free smoke passed at evaluation commit
`753db81`. Both arms completed one temporal, spatial, and joint non-test
window, all hard counts were zero, and the report SHA256 is
`966313db...458e22`. The smoke is explicitly not promotion eligible and no arm
is classified. Its single temporal window is substantially worse for
attention, while the spatial and joint windows are closer; the frozen formal
panel must determine whether that signal generalizes. See
[`successor_7_30_free_smoke_acceptance_20260811.md`](research/daily_coarse_graining/successor_7_30_free_smoke_acceptance_20260811.md).

The deterministic 96-window formal panel is prepared and independently
rehash-verified on free `test01`. It contains 24 artifacts totaling
`212,275,046` bytes. Prepared-manifest SHA256 is `6f4c4a30...23d0074` and
window-plan file SHA256 is `71a1a1ac...f397965`. See
[`successor_7_30_formal_prepared_acceptance_20260811.md`](research/daily_coarse_graining/successor_7_30_formal_prepared_acceptance_20260811.md).

Approved `gnall` job `14520704` exhausted its 45-minute limit before its first
formal artifact completed and wrote no report. The evaluator was blocked in
shared-filesystem synchronization while writing many tiny JAX persistent-cache
entries under `/WORK`; the GPU held the model but remained idle. This is an
infrastructure failure and provides no scientific classification. The local
launcher now uses process-local JIT caching only; see
[`successor_7_30_formal_cache_timeout_20260811.md`](research/daily_coarse_graining/successor_7_30_formal_cache_timeout_20260811.md).

The repair completed the full frozen formal screen on free `gln01` in
`312.730` seconds, making another paid submission unnecessary. The report
SHA256 is `7e8dc04f...a5b9d52b`. Attention is rejected: all six primary
composites are `60.61-272.25%` worse than control, 46/72 key-RMSE gates and
42/72 bias gates fail, while all 30 hard gates pass. NPP, maintenance
respiration, AGB/BGB, and total live biomass are the repeated dominant
failures. See
[`successor_7_30_formal_screen_result_20260812.md`](research/daily_coarse_graining/successor_7_30_formal_screen_result_20260812.md).

The bounded no-training attribution is complete. The formal fit consumed 500
of 23,540 available train/train point-years and was still improving at update
500; every update in both arms exceeded the global clip threshold. The old
rollout reduction also made each MSE component identical to its bias-squared
component. Commit `70c2f62` repairs bias reduction across rollout windows.

A 16-batch, two-arm coefficient probe found candidate coefficients spanning
more than 25 orders of magnitude. Free CPU activity inspection ruled out
inactive vegetation or empty Teacher targets on the near-zero-gradient
batches. A one-day `gln01` path probe then found order-one gradients at the
named-output boundary but parameter gradients around `1e-24` to `1e-20`:
initialized GPP and maintenance outputs had entered the shared nonnegative
decoder's negative saturation tail. Across all 16 batches, both arms have the
same mix of near-zero and extreme-positive initial outputs. This shared output
conditioning also explains persistent gradient clipping; it is not evidence
against attention specifically. See
[`successor_training_sufficiency_loss_gradient_audit_20260812.md`](research/daily_coarse_graining/successor_training_sufficiency_loss_gradient_audit_20260812.md).

The rejected attention checkpoint remains rejected and the control is not
promoted. The user approved a matched repair that changes only the common
final decoder kernel initialization to `1e-3` of its former scale, retaining
zero bias, both bodies, parameter count, routing, retained tail and operator
boundary. Implementation and 59 related local tests pass. The free `gln01`
16-batch re-audit passes at `0ef543b`: near-zero active GPP and maintenance
cases are eliminated, GPP >10x cases are eliminated, maintenance >10x cases
fall from 41/60 to 3/60, and the minimum parameter-gradient norm rises from
about `1e-26` to `0.6077`, with the exact parameter count unchanged.

Commit `b800db9` closes the follow-on asset gate without another GPU probe.
The immutable v2 report's 32 raw measurements are recalculated once under the
v3 lower-10%-quantile rule, producing coefficients `0.088284572329923`,
`0.13244689760336273`, `0.019877774130136576`, and
`0.029825749684563974`; all state/budget hard counts remain zero. Free
`test01` froze and independently reloaded those coefficients plus a 48-window
temporal/spatial/joint validation panel with eight distinct landpoints per
slice, nonoverlapping within-shard windows, zero exact overlap with the old
96-window panel, and no sealed-test use. See
[`successor_staged_training_assets_acceptance_20260812.md`](research/daily_coarse_graining/successor_staged_training_assets_acceptance_20260812.md).
Commit `9b4a1ce` completes the free `gln01` staged-runner smoke. Both matched
arms applied two finite-gradient updates, and one fixed validation artifact
from each temporal/spatial/joint slice passed for both arms with zero state,
mask, discrete, nonnegative-stock, or budget hard failures. An independent
`test01` process re-read the report. Wall time was `26:35.37`, maximum host RSS
was `11,648,856` KiB, observed GPU allocation peaked near `21.9` GiB, and one
XLA train-step compile took `5:57.59`. Report SHA256 is
`67b4ba0748e4f2fdc918de68587fa5ea4834271152940aba011c60fd06839f4c`.
This is pipeline evidence, not evidence that either architecture is accurate
or promotable. See
[`successor_staged_training_gpu_smoke_acceptance_20260812.md`](research/daily_coarse_graining/successor_staged_training_gpu_smoke_acceptance_20260812.md).

The first staged milestone is now accepted at production commit `9856dae`.
Paid `cnmix` job `14524758` prepared and finalized exactly 500 gzip artifacts
in `03:26:53`; an independent `test01` process decompressed and rehashed every
artifact. Manifest SHA256 is
`19d0cfd699c2535a101edbe245fffc23f49a6c514f565bb069b60e5683355375`.
Paid `gnall` job `14525122` then completed both matched arms through update 500
and the full frozen 48-window validation in `00:52:51`, with every update
applied and all training/validation hard counts zero. Report SHA256 is
`5d2ee5b19143f0d5662e6cad0818b5d82de91f859b84140ad33942da52a5019e`.

The control validation composite is `1.0457081708` and attention is
`1.0523784009`; attention is `0.64%` worse overall and slightly worse in all
six slice/horizon cells. Neither arm is promoted. The checkpoint
`improved=true` flags only establish the first staged baseline, not improvement
over zero training. Pre-clip gradient norm still exceeds one on 482/500
attention and 474/500 control updates. Independent `test01` reload verified
the recovery, best checkpoint, both 8,722,689-parameter trees, all update
records, and all hard counts. See
[`successor_staged_500_milestone_acceptance_20260812.md`](research/daily_coarse_graining/successor_staged_500_milestone_acceptance_20260812.md).

The second staged milestone is accepted at production commit `ea21f8a`.
Preparation job `14527233` timed out after four hours with 56 resumable
artifacts; free-node reproductions showed that its extreme per-sample delays
were execution anomalies and reproduced two artifacts bit for bit. Approved
`cnall` continuation `14532991` reused those artifacts and finished the 500
incremental `501-1000` products in `01:47:12`. A separate `test01` process
rebuilt the frozen schedule and rehashed and deserialized all 500 artifacts.
Manifest SHA256 is
`fe39325a3c18d6a7240e0fe5c2c23c14324b5abd9a88377a1ba380112196f076`.

Paid `gnall` job `14535746` resumed the accepted update-500 recovery and
completed update 1000 plus the unchanged 48-window panel in `00:52:54`. Report,
recovery, and best-checkpoint SHA256 values are respectively
`e379e3e7...bd68ce1`, `d49222d4...7cae7`, and `8a3967e3...f50d0a`.
Independent `test01` loading verifies both complete 8,722,689-parameter trees,
the optimizer state, all 500 incremental records, finite scalar values, and
zero training/validation hard counts.

Control improves from `1.0457081708` to `1.0060782200` (`3.79%`); attention
improves from `1.0523784009` to `1.0145714634` (`3.59%`). All six fixed
slice/horizon cells improve for each arm. Control remains `0.84%` better
overall, so neither arm is promoted and attention has not established a
decisive advantage. Incremental clipping falls to 402/500 control and 290/500
attention updates, but gradient spikes remain. Temporal 30-day normalized
GPP/NPP RMSE remains about `2.27/2.34-2.36`, while total-live-biomass RMSE is
about `0.061`; productivity remains the main scientific error. See
[`successor_staged_1000_milestone_acceptance_20260813.md`](research/daily_coarse_graining/successor_staged_1000_milestone_acceptance_20260813.md).

The approved update-2000 continuation completed under GPU job `14540920`.
Control improved only `0.31%`, from `1.0060782200` to `1.00298315`, below the
frozen 1% gate. Attention worsened `3.08%`, from `1.0145714634` to
`1.04584784`; its maintenance-respiration errors were especially unstable.
All hard gates passed, but the best checkpoint remains update 1000 and the
unchanged schedule must not continue to update 5000.

A free-`gln01` equal-sample gradient probe at commit `14b9420` then compared
`1 x (32 anchors + 4 rollouts)` against
`4 x (8 anchors + 1 rollout)`. Both arms reproduced the ordinary loss and
full gradient to about `1e-17`, with zero hard failures and no sampled
gradient above the clipping threshold. Mixed point-year batches changed the
gradient direction but did not consistently reduce cross-batch conflict.
Report SHA256 is
`eec3fccc372ad72f1818c86f7867c1eb63fc39d8a463bfe7b883196f3e88925a`;
see
[`successor_batch_diversity_probe_20260814.md`](research/daily_coarse_graining/successor_batch_diversity_probe_20260814.md).

The 100-update equal-sample A/B completed on free `gln01` at commit `aa602ab`
in `4130.87` seconds, including the unchanged 48-window validation panel.
Report SHA256 is `6c3f16ff...161154b`; all hard counts are zero and no
checkpoint or recovery was written. Attention mixed sampling scores
`1.00714567`, 4.35% better than attention single at `1.05300199` and 0.73%
better than the update-1000 baseline. Control mixed scores `1.00539728`,
0.25% worse than control single at `1.00288890`. The preregistered both-arm
gate fails, so mixed sampling is not adopted and no paid continuation is
authorized.

The report-only loss attribution is complete. Across six cells,
control-single changes maintenance/NPP/total-live-biomass RMSE by
`-12.94%/+2.39%/+2.38%`; attention-single changes them by
`-30.32%/+14.53%/+14.50%`. Attention-mixed reverses the pattern to
`+12.32%/-2.68%/-2.66%`. The signs are consistent across all six cells for
the principal tradeoffs. See
[`successor_loss_conflict_attribution_20260814.md`](research/daily_coarse_graining/successor_loss_conflict_attribution_20260814.md).

The attention-specific zero-update train/train gradient audit completed on
free `gln01` at commit `4f35e6a` in `779.454` seconds. Report SHA256 is
`a476b332624a83515555e5532229ca51b265c6ff7eab1c349156c1a3d1410b49`;
all hard and nonfinite-gradient counts are zero, and no update, checkpoint,
validation value, or sealed-test value was used. Maintenance has stable
negative gradient cosine with NPP, AGB, and total live biomass in at least
three of four single groups. Mixed sampling mitigates none of those pairs and
makes all four median maintenance pairings less aligned, so the proposed
Attention mixed-conflict mechanism is rejected.

The frozen objective also yields median Attention gradients around `1e-1` for
NPP and `1e-2` for maintenance, versus only `1e-6` to `1e-7` for AGB, BGB,
and total live biomass. This four-to-six-order imbalance supports a loss-
objective design review, not another unchanged continuation. The active task
is to specify a train-only biomass increment/tendency mechanism, maintenance
guard, matched current-loss Attention control, graph-control role, and bounded
preflight/short-fit gates before implementation or training. See
[`successor_per_output_gradient_audit_20260814.md`](research/daily_coarse_graining/successor_per_output_gradient_audit_20260814.md).
The discussion draft is
[`successor_loss_objective_design_review_v1.md`](research/daily_coarse_graining/successor_loss_objective_design_review_v1.md).
It explicitly does not reopen the rejected all-state `process_increment_v2`:
the proposed signal is limited to source-partitioned AGB/BGB tendencies after
the conservative operator, with unchanged one-day process supervision.
The 365-day rollout, long-held points, Gate D2, paid training, and product
acceptance remain closed.

## Stable Teacher

The canonical user-facing PFT14 Teacher core is on `main` at `5910ba9`.
The corresponding research-branch core commit is `10030eb`; their complete
`jax_orchidee/` trees are byte-identical. The branches remain separate only
because the research branch also contains experimental dataset and neural
code.

Gate A cross-branch parity is complete. Seven detached-worktree cases cover
cold start, later day, restart, two 365-day cold runs, a 365-day restart run,
three landpoints, and a true 1961-to-1962 year handoff. Six cases have no
differing state or modelout leaf. One low-stock case activates the research
branch's source `min_stomate` correction on day 274; its first fields,
magnitude, continuous-only behavior, and source-extracted Fortran Oracle are
hash-bound in the historical admission evidence. The accepted six-file core
was then synchronized into `main`, so the current drift gate requires raw
parity and has no source-correction exception. See
[`teacher-branch-parity-acceptance.md`](teacher-branch-parity-acceptance.md).

Current evidence:

- cold-start, restart, 365-day, and cross-landpoint executable-reuse gates
  pass;
- seven previously unused landpoints completed 1961-2010 validation;
- their largest 2010 AGB/BGB/GPP/NPP relative error is `8.5e-5`;
- the full 669-landpoint acceptance run remains a release gate.

The project must not claim equivalence for other PFTs or unsupported
ORCHIDEE configurations.

Gate B1 is complete. The versioned PFT catalog records
stable semantic IDs, canonical Fortran PFT/MTC identities, traits, parameter
row ownership, and explicit process capabilities. Runtime parameter loading,
modelout selection, state checkpoints, and newly created SECHIBA/STOMATE
restart files retain this identity. Restart reads now parse stable metadata or
require an explicit legacy paper layout, then remap all declared PFT axes by
stable ID. The paper layout still activates only bare soil and
`mangrove_pft14`; PFT2-PFT13 are `structural_only` and fail closed if activated.
Generic orchestration now resolves PFT-indexed run.def rows, PFT14 capability
slots, daily allocation inputs, and production/reference modelout selection
through stable identities or explicitly named legacy layouts. State/layout
axis mismatches fail closed. A compact bare-soil-plus-PFT14 lifecycle now
passes cold start, one complete day, three-file restart write/readback, and a
restart-backed continuation day while retaining a two-slot PFT axis.

Gate B2 is complete. The frozen machine contract
`manifests/coarse_graining/daily_neural_pft_interface_v1.json` defines the
leading variable PFT axis, named parameter and trait channels, exact mask
semantics, forbidden identity shortcuts, and physical-parameter ownership.
Tests cover PFT permutation equivariance, inactive-slot isolation, named
channel completeness, and variable legal `n_pft`. `VCMAX25` and the
maintenance-temperature polynomial remain explicit fast-process factors;
`ALLOC_MIN` and `RESIDENCE_TIME` remain in exact daily processes. Constant
parameters cannot be claimed as learnably invertible without controlled
Teacher perturbations. This closes structural extensibility only; PFT2-PFT13
remain scientifically unsupported.

## Daily Coarse-Graining Research

The `research/daily-coarse-graining` branch contains the complete Teacher plus
research-only capture, dataset, and neural-surrogate code. The production
Teacher defaults are unchanged because all capture hooks default to off.

### Active architecture decision

The final surrogate target is now a true conservative daily process operator:

```text
day-start state + native 6-hour forcing + parameters/static conditions
  -> daily fluxes, transfer fractions, and bounded tendencies
  -> one constrained water/carbon/energy update
  -> retained source-backed daily processes
  -> next-day state
```

Final inference must not reconstruct 48 interpolated forcing steps or execute
a 48-step state scan. The accepted 96-day `OK_LEAK` driver capture and exact
48-step replay remain Teacher/Oracle evidence, not the final neural boundary.
Direct neural ownership of `carbon_32l`, DOC, or `deepC_peat` remains rejected.

PFT14 is the current training and acceptance scope, but the architecture must
use shared PFT-axis processing conditioned on source-backed traits, parameters,
fractions, and masks. Supporting another PFT later will require Teacher branch
coverage and training data, but must not require redesigning the neural
interface. See
[`research/daily_coarse_graining/conservative_daily_process_operator_v1.md`](research/daily_coarse_graining/conservative_daily_process_operator_v1.md).

Gate C is complete. Its source-backed inventory classifies 50 required daily
labels against the real v5 Teacher contract: 10 are present, 11 are exactly
derivable, and 29 are supplied by three daily-reduced compiled capture
families. Capture is default-off and stores no 48-step trajectory. The
true-label constrained updater represents ordinary inventories with bounded
outgoing fractions and nonnegative inputs; source-signed `qsintveg` is
represented as nonnegative storage minus nonnegative evaporation debt.

The Gate C2 lifecycle matrix passes cold-start continuation, an ordinary
later day, and a 1962 restart-year day. All 349 day-end state leaves, 26
modelout fields, and four modelout leaves pass; maximum state error is
`2.78e-17`. Maximum water, carbon, and flux-side energy residuals are
`4.70e-13`, `8.05e-9`, and `6.98e-9`, respectively. Defined masks and inactive
PFT slots are explicit, no post-hoc clipping is used, and the accepted exact
three-file restart roundtrip composes with the restart-day constrained replay.
Evidence is described in
[`gate_c2_constrained_replay_v1.md`](research/daily_coarse_graining/gate_c2_constrained_replay_v1.md).
The six prerequisite sequence is now `6/6`.

Gate E1 is **accepted**. The
`research/daily_coarse_graining/daily_operator/` package consumes a masked
variable-length native forcing sequence, retains explicit PFT inputs, predicts
constrained process quantities, applies one pure-JAX conservative
water/carbon/thermal update, and composes through the reusable canonical
retained-tail adapter. `DailyOperatorInput` contains only canonical day-start
state and allowed forcing/PFT/static/annual/calendar inputs; the unique
`CanonicalDailyOperatorInputAssembler` derives every overlapping grouped,
litter/lignin, mask, and process-static view. Its fail-closed cache gate rejects
all tested contradictory copies.

Thirty E1 unit tests and the 70-test B2/C/D1/registry/E1 suite pass. The real
adapter, not the identity fixture, passes eager/JIT agreement to `5.69e-14`, a
finite forward JVP, and a finite reverse VJP. A real-Teacher Oracle matrix
passes cold continuation, ordinary later day, restart-year, and restart
composition through `daily_operator_transition`. Every per-case canonical
continuous/discrete next state passes with maximum error below `3.94e-13`;
maximum water, carbon, and thermal flux residuals are `4.70e-13`, `8.69e-9`,
and `6.99e-9`. The graph contains no callback and no length-48 scan. All 47
frozen labels have exactly one owner binding.

The frozen data-readiness matrix is `10` labels already present, `8` exactly
derivable, and `29` requiring 66 full-coverage typed fields. Gate E2's active
design now generates those fields and the base Markov rows in the same
continuous Teacher execution. The formal sixth dataset release
`pft14-daily-teacher-v6` binds Teacher source SHA256
`ffae1ec0...457223`, configuration, PFT catalog, Markov contract, coherent
typed contract, and producer sources. A
point-year is complete only when its base shard, lossless typed shard plus
exact masks, year-end checkpoint, and common metadata all exist and hash
verify. The release SHA256 is
`1affefd9b3166005cdd868477ba25e68b682c7394438e5a79c4fcf087c8496a8`;
the unified dataset schema is `daily_teacher_dataset_manifest_v6`, the typed
packaging contract is v3, and the underlying Markov state contract remains v5.

A real three-day 1961 cold-start instrumentation A/B passed: capture-on and
capture-off compiled base arrays and final state were elementwise equal,
discrete state was exact, and all 66 typed fields passed the fixed compiled
production gate `atol=1e-8, rtol=1e-10`. The complete v5/v6 compatibility and
data/operator regression is `102 passed`; Ruff and Python compilation pass.
The six-point 1961 production
pilot passed on Explore1000 as job `14486923`: all six workers exited zero,
and unified aggregation plus the formal coherent reader accepted six shards,
2,184 transitions, all exact masks, and all year-end checkpoint metadata. The
dataset manifest SHA256 is
`a7e44cc536a1c1bae4f707a5097d379c8019fe1e1fbe146fad48084ef0042d88`.
Peak process RSS was 31.4 GiB; base, typed, and checkpoint assets total
28,221,594 bytes. The provisional full storage projection is 146.5 GiB.
This cold-year pilot did not measure a later hot year or execute a real
1961-to-1962 chain. That gap is now closed by Explore1000 job `14490208`, a
ten-landpoint, two-year v6 reuse benchmark at commit `19087b1`. All five
workers and `20/20` point-years passed hash verification and unified
aggregation. First-process cold entries averaged 2148.84 seconds total;
same-point hot capture averaged 82.73 seconds; after switching landpoints,
capture averaged 91.07 seconds with no compiled-cache growth; and the second
point's hot year averaged 81.97 seconds. Peak RSS was about 31.2 GiB per
worker, and five staggered workers completed safely on one node. The accepted
global shape is 100 workers on 20 nodes, with about 8.6-10 hours projected
wall time and about CNY 60 expected CPU charge. See
[`teacher_v6_reuse_benchmark_20260806.md`](research/daily_coarse_graining/teacher_v6_reuse_benchmark_20260806.md).
The pilot details remain in
[`coherent_teacher_pilot_20260806.md`](research/daily_coarse_graining/coherent_teacher_pilot_20260806.md).

Full v6 production and admission are now complete. The accepted aggregate has
669 landpoints, 50 noleap years, 33,450 point-year shards, and 12,208,581 daily
transitions. Its dataset-manifest SHA256 is
`5fc2c0691bfa53252f875f7646f3c1a5bea203c2759a6e947b26bb51dc13aa99`.
All base and typed shards, metadata, input identities, checkpoints, and
year-to-year links passed aggregation. Finalizer job `14500801` completed in
`02:17:57` with exit code zero. Target representation has zero persistence
mismatches; statistics use 8,591,565 train/train samples only; and a real
batch-8 JAX smoke produced finite predictions, loss, and gradients. The
acceptance-report SHA256 is
`8784ed42dc64ef77c9642f613eb2e5940635edc6e744a1af4ea8634293baa8e3`.
See
[`teacher_v6_full_production_acceptance_20260807.md`](research/daily_coarse_graining/teacher_v6_full_production_acceptance_20260807.md).

Explore1000 runtime cleanup subsequently archived the 100 historical v5 worker
directories as individually verified tar files while leaving the formal v6
product unchanged. The operation reduced the project runtime by 153,803 files
and 21.475 GiB. Historical v5 scripts must restore the required worker tar
before reading its old loose-file path; v5 remains prohibited as an input or
join source for formal v6 training. Archive paths, hashes, restoration, and
post-cleanup v6 evidence are recorded in
[`storage_cleanup_20260807.md`](research/daily_coarse_graining/storage_cleanup_20260807.md).

The former immutable-parent Gate E2 attempts below are historical negative
evidence. They explain why old v5 state rows cannot be joined to labels from a
new Teacher execution. Job `14481274_0` stopped in preflight
because the historical generation-plan configuration predates the current PFT
catalog; execution is now correctly bound to the current repository
configuration while the historical plan owns only the immutable input
inventory. Replacement job `14481314_0` passed preflight and cold start but
stopped after the first captured transition: current-Teacher Day 2 end state
is not bit-exact to the parent shard generated at Teacher commit `7397d1e`.
It ran for `00:10:02`, peaked at about 9.22 GB RSS, and wrote no sidecar. The
bounded diagnostic subsequently classified the difference in job
`14481448_0`: only `cn_ind` and `ind` differ, both by one ULP; the global
maximum absolute and relative errors are `2.842170943040401e-14` and
`2.063503050227843e-16`, and all discrete state is exact. This is safely
inside Gate A's existing `atol=1e-12, rtol=1e-12` policy and is not a
scientific Teacher regression. The historical design then reused that
declared continuous policy, retained exact discrete comparisons, and kept
sidecar encoding itself bit-exact. Its planned task-0 continuation was later
abandoned in favor of coherent regeneration.
The first full task-0 submission, `14482172_0`, stopped before Python execution
because the cluster's older Bash treats an empty array expansion as unbound
under `set -u`. The launcher now uses explicit diagnostic and full-generation
branches and has a regression guard against reintroducing the empty-array
dispatch. No Teacher transition or new sidecar was produced by that job.
Replacement `14482331_0` entered Teacher execution and stopped at Day 21. The
only examples outside the Gate A combined tolerance were three mirrors of
`soilflx`, with absolute error `4.5773e-11` and relative error `1.3124e-12`;
all discrete state remained exact. This is accumulated free-run roundoff from
joining a current-Teacher trajectory to an immutable old-Teacher parent, not
an admissible sidecar transition policy. The tolerance was not widened.
That teacher-forced workaround was not promoted. Historical v5 remains
experiment evidence only; the coherent release regenerates both sides of the
training record together. See
[`coherent_teacher_dataset_generation_v1.md`](research/daily_coarse_graining/coherent_teacher_dataset_generation_v1.md).

Completed infrastructure:

- Daily Fast-Day Teacher Contract v5:
  `S[d] (3,854 continuous values plus 12 exact discrete values) + native
  6-hour forcing[d] + P -> B_fast[d] (2,855 values)`, followed by retained
  source-backed daily season/STOMATE to produce `S[d+1]`;
- only the true cross-day subset of `sechiba_finalize_state`, including
  `leaf_ci`, is carried in `S[d]`; the 93-field finalize packet is no longer a
  duplicated learned output;
- canonical PFT14 state trajectories with PFT1/PFT14 axis compaction,
  exact discrete state, deterministic mirror reconstruction and year-start
  `nroot` handling;
- exact native-forcing window reconstruction of all 48 Teacher input steps;
- named condition slices for 84 parameter values, 735 physical/static
  landpoint values, annual CO2, year, and day index;
- a machine-audited ownership ledger covering every compiled complete-day
  Teacher argument;
- a hash-verifying training reader that enforces frozen splits, excludes
  landpoint identity, computes train-only streaming statistics, excludes
  NaN/inf and ORCHIDEE `+/-1e20` sentinels, and uses bounded prefetch;
- a persistence-centered residual target representation with exact
  restoration of persistent undefined values and a dedicated classifier for
  source-dynamic `diffuco.rveget` defined/undefined transitions;
- pure-JAX gradient and checkpoint plumbing;
- a hash-bound post-generation acceptance gate that requires a complete v4
  aggregate, audits target representation, fits sentinel-aware train-only
  statistics, and passes a real-batch finite forward/loss/gradient smoke
  before GPU training;
- compiled Teacher training capture with reduced host transfers;
- restartable, atomic landpoint-year NPZ shard generation with provenance,
  frozen spatial/temporal splits, resume, and aggregate validation;
- a frozen 669-point production specification and a first five-point,
  1961-2010 architecture-development subset with two train, two validation,
  and one test landpoint;
- Linux CPU and V100 compatibility tests.

Current scientific status:

- the neural path is technically connected but is not a validated daily
  surrogate;
- the contract-v4 nine-point V100 baselines established
  nontrivial one-step learnability: its equal-split validation score was
  0.812 versus 1.064 for the matched persistence baseline, but spatial and
  joint scores remained 0.855 and 1.438;
- that run exposed an absolute-state `rveget` undefined classifier which
  damaged a stronger persistence prior; the accepted follow-up predicts only
  defined/undefined flips and reduced temporal/spatial/joint classification
  errors to `18/0/0`;
- a fresh 15-epoch run reached its best equal-split score of `0.772` at epoch
  10, improving temporal/spatial/joint scores over matched persistence by
  `67.0%/28.0%/18.0%`; additional epochs did not resolve the dominant spatial
  DIFFUCO/ENERBIL and HYDROL errors;
- no user-facing `daily-surrogate` run mode exists;
- the first seven-day v4 free rollout reached `0.921` normalized state RMSE,
  while teacher-forced daily error stayed near `0.11-0.15`; this identifies
  recursive state-distribution drift rather than a retained-tail handoff error;
- contract v4 was then found to omit the fast-owned cross-day `leaf_ci` output.
  Contract v5 appends it losslessly from stored `S[d+1]`, so no Teacher rerun
  is required, but all v4 statistics and checkpoints are superseded;
- the pure-JAX retained-tail transition passes a real Day 2 forward and
  reverse-mode gate: maximum next-state error `5.68e-14`, zero mask/discrete
  mismatch, and finite gradients for all 2,847 defined target values;
- a real three-day `lax.scan` gate also passes: maximum trajectory error
  `1.71e-13`, zero mask/discrete mismatch, and finite gradients for all 8,541
  defined day-target values;
- the complete nine-point v4 asset has been losslessly migrated to v5 without
  rerunning Teacher: 450/450 landpoint-year shards and about 1.4 GiB passed
  source/output hash verification;
- the v5 dataset acceptance gate passed on 96,354 train/train transitions with
  zero target-representation persistence mismatches and finite forward, loss,
  and gradients for the 1,963,369-parameter canonical model;
- a real V100 one-window multistep smoke passed at research commit `f68ea59`:
  one horizon-1 update completed in 104.84 seconds including first compile,
  with loss `0.0495625`, gradient norm `0.389538`, and a finite 31.4 MB
  checkpoint; this also closed canonical generation-plan hashing and external
  server data-root wiring in the multistep launchers;
- the first complete v5 V100 curriculum finished on `gln01` in 35:03. Its
  ten-epoch one-step initialization selected epoch 10 at validation score
  `0.680439`; the subsequent `1:64,3:64,7:64` stages remained finite;
- on the joint spatial-and-temporal validation window at landpoint
  `215.0-119.0`, year 2005, days 100-106, the seven-day free-rollout final
  state RMSE improved from `0.473270` for the one-step checkpoint to
  `0.429058` after multistep training, with zero defined-status or discrete
  mismatches. Teacher-state feedback kept all daily RMSE values below
  `0.086930` and ended at `0.068236`, so recursive state-distribution drift,
  rather than an invalid retained-tail handoff, remains the dominant error;
- a complete field-level drift diagnostic at research commit `18471aa`
  reproduced every daily rollout RMSE exactly. The current multistep state
  loss is uniform over 3,854 scalar values: slowproc/STOMATE owns 2,386 values
  and 61.91% nominal weight, but already contributes 76.89% of the first-day
  realized Huber state loss. Biomass alone contributes 36.39% on Day 1 and its
  normalized RMSE grows from `0.660380` to `4.841768` by Day 7; growth and
  maintenance respiration, NPP, litter partitioning, and surface energy state
  are also early divergence leaders. This rules out the simplistic diagnosis
  that important carbon state is merely invisible to the loss. The next
  objective must combine process-balanced weighting with explicit recursive
  stability/state-change supervision, then be tested under a controlled A/B;
- a four-way split diagnosis at research commit `825935b` compared
  train/train, train/validation-year, validation-point/train-year, and joint
  validation rollouts on the same Day 100-106 window. Day-7 RMSE was
  `0.145471`, `0.134233`, `0.519886`, and `0.429058`, respectively. The
  controlled temporal holdout did not degrade the training landpoint, while
  the spatial holdout failed even in a training year. Spatial-condition
  coverage is therefore a demonstrated limitation of the current six
  training landpoints. Recursive objective failure also remains: train/train
  `litterpart` grows from `0.541` to `3.652`, proving that the global `<=0.29`
  gate can hide a scientifically important low-dimensional failure. Do not
  add more years of the same points; first run one process-balanced/state-
  increment objective A/B, then add a bounded, source-selected spatial pilot;
- the bounded `process_increment_v2` equal-budget A/B completed and is
  rejected. On train/train Day 7, global normalized state RMSE changed from
  `0.145471` to `0.146456`, biomass from `0.014259` to `0.016716`, and
  litterpart from `3.651536` to `4.008877`. The candidate retained zero
  defined-status and discrete mismatches but failed all required seen-condition
  improvement gates. `canonical_multistep_v1` remains the baseline; do not
  continue tuning process/increment loss weights;
- the counterfactual Teacher-at-model-state gate passed at research commit
  `87e1132`: exact Teacher re-entry completed at model-generated Days 2, 4,
  and 7 with zero defined-status or discrete mismatch. Mean matched operator
  error was `0.085390`, mean state sensitivity was `0.073494`, and their ratio
  was `1.161865`. This supports stop-gradient pushforward with same-state
  on-policy Teacher labels;
- the real detached-prefix pushforward smoke passed at research commit
  `c3e2d79`. After a three-day model prefix, one exact same-state Teacher query
  and final-transition update produced finite loss `0.0836542`, gradient norm
  `0.1333402`, and a finite checkpoint;
- the corresponding 192-update, batch-4 `on_policy_pushforward_v1` A/B is
  complete and rejected. On train/train Day 7, global RMSE improved from
  `0.145471` to `0.092764` and litterpart from `3.651536` to `0.854632`, but
  NPP and growth respiration regressed about 9.6%. Spatial-validation global
  RMSE worsened from `0.519886` to `0.550696` and joint validation from
  `0.429058` to `0.459871`; spatial litterpart also worsened by more than 60%.
  All mask/discrete gates passed, but the predeclared named-state and spatial
  gates did not. Keep `canonical_multistep_v1`; do not tune this candidate or
  inspect the sealed test split;
- deterministic replay of that run's single extreme batch proved a separate
  physical-domain defect in neural rollout reconstruction: the unconstrained
  network produced negative `carbon_32l`, `DOC`, and `deepC_peat` stocks before
  exact Teacher re-entry, causing NaNs and a `7.27e8` soil-carbon artifact.
  Commit `5414d8c` now applies the source-backed nonnegative-stock projection
  at the shared NumPy/JAX physical reconstruction boundary. The exact four
  replayed losses changed from `0.0997/0.5658/0.0984/562889` to
  `0.0999/0.0946/0.0970/0.1165`, with defined-status mismatches falling from
  980 to zero;
- the frozen four-way seven-day baseline was rerun after that semantic fix.
  Day-7 RMSE is now `0.144765`, `0.133391`, `0.519286`, and `0.428353` for
  train/train, train/validation-year, validation-point/train-year, and joint
  validation. The small changes leave the scientific diagnosis intact:
  spatial generalization remains dominant and train/train `litterpart` still
  reaches `3.651537`;
- the matched `structured_process_film_v1` architecture A/B is complete and
  rejected. Relative to an identically continued flat control, structured
  Day-7 RMSE improves 4.36% and 4.24% on the two spatial-validation cases but
  regresses 7.68% and 11.32% at the seen landpoint. Its maximum useful
  parameter/static validation permutation effect is only `5.26e-5`, below the
  predeclared `1e-4` condition-use threshold; parameter permutation slightly
  improves loss, showing a wrongly learned association rather than a dead
  channel. The flat continuation itself worsens all four frozen global scores,
  so neither checkpoint is promoted. See
  `structured_architecture_ab_result_20260726.md`;
- the reusable 669-point Teacher data-product admission now passes at the
  contract level. The versioned machine gate binds the frozen 535/67/67
  spatial split and 1961-2010 temporal split to Markov Contract v5, verifies
  all state/target/diagnostic/forcing/condition inventories, and computes
  33,450 point-years with 12,208,581 transitions. Its final mode additionally
  requires all 33,450 aggregate shards and exact split assignments. The next
  data operation is therefore full 669-point baseline generation after
  staging/plan/resource approval, not another architecture-specific spatial
  pilot. Parameter perturbations and new forcing products remain separate
  parent-bound extension datasets;
- the first real bounded 669 production probe generated and hash-verified the
  complete `001.0-071.0:1961` v5 shard at contract SHA256
  `6813065b...9d442`. It took 3084.5 seconds for the entry and reached about
  29.5 GiB peak process RSS. Slurm's exit code 2 was an orchestration defect:
  the old CLI reported an intentionally partial `--max-new-entries` worker as
  failed after successfully writing its resumable manifest. That exit
  semantics now has a regression fix. The `07fe2f9` probe remains immutable
  admission evidence; full production must use one fresh output root bound to
  the accepted post-fix commit rather than mixing commit identities;
- the post-fix admission at commit `7397d1e` passed: Slurm task `14390213_0`
  completed with exit code zero in 42:43, the entry took 2381.0 seconds and
  reached about 29.5 GiB peak process RSS. Its NPZ SHA256 exactly matches the
  original probe, proving the orchestration-only change did not alter
  scientific arrays. This five-worker output remains immutable admission
  evidence and is not mixed into the later 100-worker aggregate;
- the accepted full-production topology is now 20 `cnall` nodes with five
  one-CPU workers per node, for 100 concurrent workers. The launcher commit is
  `74bd4eb`, while the separately validated Teacher worktree remains fixed at
  `7397d1e`. Five-minute within-node startup staggering reduces overlapping
  cold-compilation memory peaks. The validated `w100` plan SHA256 is
  `9ba2caf5...5fa17e`; 69 workers own 350 point-years each and 31 own 300 each;
- the first two-node admission job `14391204` exposed default `srun`
  first-task wait semantics, not a Teacher failure. The explicit `--wait=0`
  fix was then accepted by rerun `14391639`: all ten ranks exited zero in
  55:46, all 11 accumulated point-years and hashes passed, no active lock or
  manifest error remained, and worker 0 closed the 1961-to-1962 restart chain.
  Cold workers peaked near 31 GB RSS and the resumed worker near 26.2 GB;
- consecutive-entry calibration `14391864` passed in 40:03. Its first fresh
  process entry took 1743.348 seconds, while the next three years in the same
  process took 158.455, 158.915, and 160.493 seconds. Mean steady-state
  production is therefore 159.288 seconds per point-year, including capture,
  compression, hashes, and checkpoint output, or about 0.436 seconds per
  simulated day. The post-run hash audit found 15 valid point-years and no
  active lock or manifest error;
- production recovery and consumption are prepared: a progress auditor
  identifies running/partial/missing/invalid workers, arbitrary worker IDs can
  be resumed without changing the canonical 100-worker assignment, and final
  aggregation now requires complete worker inventories, no active lock, all
  shard/metadata/checkpoint hashes, and exact checkpoint chains;
- the complete-data training protocol is frozen at
  `manifests/coarse_graining/daily_teacher_669_training_protocol.json`.
  It specifies exact-once balanced landpoint/year/day streaming with at most
  eight open shards, train/train-only normalization, four non-test
  model-selection slices, sealed final testing, and one-step through
  complete-chain promotion gates. A local real-shard streaming preflight
  passed; it is an I/O gate, not neural accuracy evidence;
- the literature-backed post-admission architecture decision is frozen in
  `long_rollout_architecture_review_20260727.md`: first compare a matched flat
  control with a process- and axis-aware operator, then add mixed-horizon and
  explicit tendency-bias training only if the architecture passes. Long-term
  acceptance targets bounded physical drift and unbiased budgets, not an
  impossible promise of mathematically zero accumulated error;
- the provisional seven-day free-rollout gate of `<=0.29` therefore did not
  pass. No v5 30-, 365-day, or 50-year neural rollout has passed;
- generated labels remain `provisional_teacher` until the 669-point Teacher
  acceptance gate is complete.
- the first five-epoch V100 run is rejected as architecture evidence because
  it used contract v3 and statistics v1, which treated `rveget=1e20` as an
  ordinary regression value and duplicated finalize mirrors in `B_fast`;
- the accepted five-point v3 Teacher dataset remains provenance evidence, but
  it and its v1 statistics/checkpoint must not be reused for v4 training.
- old v1/v2 shards are audit evidence only and must not be expanded into the
  current v3 dataset.

## CPU and GPU Decision

One-landpoint V100 Teacher capture reached about `0.263 s/day` hot for a
28-day block, with about `922.6 s` cold compilation. This does not establish
that GPU is faster than CPU because the retained CPU measurement was not a
matching same-process hot benchmark. It does establish that one landpoint
does not provide enough parallelism to justify GPU Teacher generation.

Current policy:

- use persistent CPU workers for Teacher shard generation unless a measured
  batched-landpoint GPU implementation changes the result;
- use the accepted seven-day complete-day block for CPU Teacher generation;
  the 28-day capture result is a historical V100-specific experiment;
- use GPU for neural-network training;
- keep inference backend-neutral and benchmark CPU latency versus batched GPU
  throughput after the network architecture is frozen;
- do not describe the research branch as a delivered GPU model.

The project-owned Explore1000 `orcjax_gpu` environment passed its real-V100
runtime gate on 2026-07-23. JAX 0.4.38 selected backend `gpu` and executed a
finite JIT loss and gradient after explicit PJRT plugin discovery. The
canonical verification and training entry points now share that fail-fast
initialization and reject CPU fallback. This validates the training runtime;
it is not evidence that the neural surrogate itself is scientifically valid.

## Repository and Release Architecture

The intended user-facing release is one codebase with two independent runtime
choices:

- execution backend: `cpu` or `gpu`;
- transition implementation: `teacher_half_hour` or `neural_daily`.

CPU and GPU support must not become permanent model forks. They share the
same state, forcing, parameter, restart, output, and acceptance contracts;
only batching, compilation, and device execution may differ. The stable
`main` branch contains accepted user-facing implementations. The
`research/daily-coarse-graining` branch remains a reproducible development
line for dataset generation, network training, experiments, and promotion
evidence, not a separate product.

A released neural checkpoint cannot be the sole training artifact. Its
provenance must bind the Teacher commit, dataset and split manifests, state
and input schema hashes, train-only normalization statistics, architecture,
optimizer and schedule, random seeds, environment lock, checkpoint hash, and
rollout acceptance report. Reproduction code and compact manifests belong in
Git; large datasets and weights remain external assets with stable identifiers
and SHA256 hashes. Accepted training and inference code is merged into
`main`, while ongoing experiments continue on the research branch.

Canonical-Teacher physical-parameter gradient Gate D1 is accepted. Its matrix
covers seven source-backed process pairs, seven complete Day-2 pairs, three
two-day propagation pairs, four restart-year pairs, and two source-extracted
Fortran finite-difference pairs. JAX forward and reverse AD agree with centered
`h`/`h2` Richardson finite differences under the frozen 1% active-pair policy;
inactive and threshold-adjacent cases are classified separately. The gate
found and repaired dormant-branch reverse-AD pollution across half-hour state
and a traced `altmax` host cache conversion at the yearly HYDROL first step.
The accepted evidence and exact scope are documented in
[`gate_d1_physical_parameter_gradients_v1.md`](research/daily_coarse_graining/gate_d1_physical_parameter_gradients_v1.md).

Gate D1 does not validate physical-parameter gradients of a future daily
surrogate. Before neural parameter calibration, sensitivities, or Jacobian
claims, Gate D2 must compare surrogate AD against canonical Teacher/Fortran
finite differences over frozen parameter ranges and held-out combinations.
Network-weight and cross-day state gradients remain training checks and cannot
satisfy Gate D2.

The source-driven physical-parameter candidate registry is accepted. It binds
34 parameter families and 91 scalar components to hashed Fortran source and
classifies them as 9 inversion candidates, 17 sensitivity-only families, 2
discrete controls, and 6 currently non-identifiable families. The first wave
remains `VCMAX25`, `MAINT_RESP_SLOPE_C`, `ALLOC_MIN`, and `RESIDENCE_TIME`.
Five second-wave PFT14 families require priors, controlled perturbations, and
new gradient evidence before promotion. Registry membership does not expand
the public parameter interface. See
[`physical_parameter_candidate_registry_v1.md`](research/daily_coarse_graining/physical_parameter_candidate_registry_v1.md).

## Teacher Production Evidence

Explore1000 admission measurements established the current production policy:

- a first eight-day landpoint compiled and captured in 1,406.85 seconds;
- a second landpoint in the same process captured in 3.35 seconds and took
  40.15 seconds including preparation, proving executable reuse across
  equal-shape PFT14 landpoints;
- a cold 1961 full year took 2,395.32 seconds;
- the following in-process 1962 hot year took 102.30 seconds to capture and
  139.20 seconds including compression, hashing, checkpoints, and shared IO;
- peak RSS reached about 29.4 GiB, so the initial total concurrency cap is five
  persistent workers per approximately 187 GiB node;
- persistent compilation cache did not eliminate compilation in a new Slurm
  process, so production relies on same-process reuse rather than cross-job
  cache reuse.

The first bounded production target was the frozen five-point, 1961-2010 v3
dataset: 250 landpoint-year shards and 91,245 daily transitions. Its first
submission was rejected before acceptance because the production planner
incorrectly inserted Gregorian leap days into the paper model's fixed 365-day
noleap calendar. The planner, plan validator, and training calendar features
now enforce noleap semantics. The clean replacement completed and passed its
aggregate gate on 2026-07-23: all five 1961-2010 chains, 250 shards, 91,245
transitions, source/input hashes, checkpoint links, splits, and the single
Markov contract are valid. Its operational evidence is recorded in
[`research/daily_coarse_graining/HANDOFF.md`](research/daily_coarse_graining/HANDOFF.md).

## Historical Neural Milestones

The material below records completed v3-v5 dataset and rejected-model
evidence. It is not the active work queue. Gate E1 is accepted; the coherent
Gate E2 coherent production and formal v6 dataset admission have passed. The
first formal neural pair is closed without promotion. The long-horizon and
parameter-routing foundations are complete. The successor options analysis is
complete, the pair is selected, and the local synthetic foundation is
accepted. The real-v6 `gln01` smoke is the active stop point before fitting.

The full 669-point baseline is generated and admitted. Finalization job
`14400343` completed in 4:08:54 with exit code zero. Its fail-closed gates
accepted all 33,450 point-years and 12,208,581 transitions, found zero target
persistence mismatches, fitted statistics from the 8,591,565 train/train
samples only, and passed a finite real-batch JAX forward/loss/gradient smoke.
The dataset manifest SHA256 is
`89ea2f24bad8f4b39129937f4106285dd53936c35d8dc610810f865b02d4e508`;
The statistics JSON SHA256 is
`c9ed1c6ea4cfca6c0da9504a36d440a2b388375f01acf8bc78edc7645a4ffb37`;
the statistics NPZ SHA256 is
`33865286021c39bbdb6511a115da55143789e857c31e701c7417ff1a9c50cdbd`;
the acceptance report SHA256 is
`0b13da5c7f8664f9b3b8fe70f6b8dc8e3aca513ecbf15d4b3d879a07d663b596`.

The matched `canonical_flat_v1` versus `axis_process_coupled_v1` architecture
A/B is complete and passed. Both arms used the same admitted samples, update
budget, optimizer, schedule, seed, physical reconstruction, and canonical
objective. Teacher shards were not regenerated and the sealed test split was
not inspected.

The architecture screen was frozen in
`canonical_669_axis_process_architecture_ab.json`, canonical SHA256
`39b94a28e51dba67017ac3f06973b7f4fbbfa4a560b249eec49420f4baed8099`.
It uses one exact-once train/train epoch, batch size 256, learning rate
`1e-4`, seed `20260728`, full temporal/spatial/joint one-step validation, and
no test samples. The production trainer now supports both registered
architectures through the same loss and balanced reader, performs one shared
asset verification for the two-arm run, and avoids per-batch host loss
synchronization.

The prepared real-shard V100 compile/update smoke passed on `gln01` at commit
`0992178`. It hash-verified the train/train shard `001.0-071.0:1961`, used
zero-based sample indices 0-255 as one batch of 256, and consumed no
validation or test sample.
Both arms retained float32 parameters and finite losses. Flat compile/hot
updates took `2.2506/0.00862 s`; axis/process compile/hot updates took
`8.7199/0.00757 s`. The report SHA256 is
`3975eb93000b355b51c565f421f04fdc19342032360c1610e1823dce607f3d16`.

Full job `14403674` completed in `03:00:49` with exit code zero from clean
commit `4c1fe0c`. The final report SHA256 is
`1995fb7f9284bde56c8ac5a9cd411d1bf08fc94f795a0ab1fdad69b42814c9a7`.
Axis/process passed every predeclared gate and produced decision
`advance_axis_process_to_rollout_stability_experiment`. Its
temporal/spatial/joint score ratios relative to flat are
`0.943530/0.960916/0.952866`; the maximum process-family ratio is `0.990947`,
the parameter ratio is `0.995249`, and test evaluation is false. The selected
axis/process checkpoint SHA256 is
`a119999606b063ac9f9ed47e4d1bb664cdfe32b9459e7e7ee24e96a0e944db2e`.

The production screen now performs the full acceptance/hash preflight once,
runs the flat and axis/process arms concurrently on two isolated V100 devices,
and then applies the unchanged aggregate gates. The original single-process
runner remains a sequential fallback. A real-shard two-process smoke passed on
both free `gln01` V100s at commit `ef3ddda`: both reports retained float32
parameters and reproduced identical compile/hot losses for both architectures.
The GPU 0/GPU 1 report SHA256 values are
`ec66dac4be68cc0afc12dac5cd59efc2b3b6932377c999fa6ccc6665ed9a680a`
and
`4001937f2d22916755d9646cef7227cdb26496b5a6375246d8349070a0322351`.
The launcher is restart-safe under the same output root: an existing preflight
must match every frozen input exactly, completed arm checkpoints/reports are
resumed without another epoch, incomplete arms rerun, and worker logs append.
The 12-hour Slurm limit is retained as a safety ceiling.

The conditional post-architecture rollout-stability experiment was frozen
before the parent result in
[`canonical_669_rollout_stability_protocol.json`](../manifests/coarse_graining/canonical_669_rollout_stability_protocol.json),
Its v1 canonical SHA256 was
`370011f6d8edc447bd0ebf037249df1a18f3bfb30071204001366a2aecde310b`.
Its auditor binds the admitted 669-point assets and architecture-screen hash,
keeps the test split sealed, and rejects drift in the matched arms,
`1/3/7/30`-day sampling, loss inventory, train-only coefficient calibration,
hard constraints, screening thresholds, promotion ladder, or stop rules.
Experiment B runs only if the parent decision is exactly
`advance_axis_process_to_rollout_stability_experiment`; a rejected architecture
returns to a named architecture hypothesis rather than silently applying the
stability objective to the flat model.

The Experiment B training implementation and real-shard GPU gate now pass.
Horizon `1/3/7/30` protocol shapes completed on a V100 with two distinct train
landpoints sharing one executable, finite first and second updates, zero hard
counts, and exact checkpoint resume. Horizon-7 and rematerialized horizon-30
hot updates take about `0.10 s` and `0.20 s`, with peak allocator use below
`0.55 GiB`.

Host preparation was then reduced without changing any measured loss,
component value, gradient norm, or hard count. Each shard is loaded once,
runtime/static state is cached by landpoint, and the five forcing leaves
actually consumed after the learned fast-day boundary are reconstructed in
one vectorized batch operation. Steady forcing preparation fell from seconds
to about `0.015 s`; a complete warm-cache horizon-7 preparation is `0.245 s`.
A 16-landpoint cache gate measured about `7.8 MiB` RSS growth per added
landpoint, making the 535-train-landpoint cache safe on a gnall node. Details
are in
[`rollout_stability_smoke_20260729.md`](research/daily_coarse_graining/rollout_stability_smoke_20260729.md).

Preflight, calibration, execution, and arm checkpoint identities now bind the
training Git HEAD in addition to protocol, data, schedule, model, and
environment.

The first paid Experiment B attempt, job `14410820`, failed after `00:07:47`
during train-only coefficient-calibration ordinal 4, before either training
arm started and before any checkpoint was written. The exact batch was
landpoint `087.0-105.0`, year 1961, horizon 1, batch size 256. It contained
three and only three defined-status errors, all in the PFT14 column of
`diffuco_previous_step_state.rveget`, on Days 2, 294, and 358. There were zero
discrete mismatches, nonfinite defined values, or negative source-constrained
carbon stocks. The sealed test split was not used. The diagnostic SHA256 is
`f24eeef2f937c2f828b4b7449cdadbaa5d60ef8cf581d186564eb622e6751a2e`.

This exposed a protocol contradiction rather than a new retained-tail defect:
`rveget` defined/undefined is an explicitly declared learned classification
target, but v1 also treated every classification mistake as an exact-zero
optimizer veto. Protocol v2, canonical SHA256
`a0ecfd4c89ff3c5691f153168f3ba80939582774689b92ff04b0cf77e699cb40`,
separates declared dynamic `rveget` status errors from unexpected structural
status errors. Declared errors remain supervised, counted, reported, and
screened against the matched control; they no longer veto an optimizer update.
Unexpected status errors, discrete mismatches, nonfinite defined values, and
negative source-constrained carbon stocks remain exact hard failures.

The required free `gln01` gate passed at commit `13f09c9`. It reproduced
calibration ordinal 4 with exactly three declared dynamic errors, zero
unexpected/discrete/nonfinite/negative-stock hard counts, finite loss and
gradient norm, and `update_applied=true`. The sealed test split was not used.
The smoke report SHA256 is
`66dc952abef5804eabe43ca8a9e9144828caba9649452557ded274f2bf8146a5`.

A later paid attempt reached mixed-horizon update 226 before failing on two
rollout samples with nonfinite gradients. Source-owner isolation proved that
the failure came from the NPP negative-stock add-back landing one ULP above
`min_stomate`, which activated a strict leaf-fraction division gate. The same
pathology was reproduced from the original Fortran procedure under both
`gfortran -O0` and `-O3`. Commit `f9554c6` preserves the source carbon budget
while pinning the corrected stock exactly to the intended threshold.

After that stabilization, both failing samples had finite gradients, the
complete update-226 audit passed 64/64 rollout samples with all hard counts
zero, and the candidate update was applied. A resumable sequential prefix gate
then passed all 256 mixed-horizon updates at commit `452b743`, including an
exact checkpoint restart at update 128 and coverage of horizons 1, 3, 7, and
30. The sealed test split remains unused.

Formal Experiment B rerun `14419242` completed the full 8,192-update control
but failed closed at candidate update 302. Exact checkpoint replay and
horizon-prefix bisection localized the first bad transition to 1973 Day 355.
The forward model remained finite; reverse-mode AD failed in the STOMATE
leaf-age turnover quotient at a `3.7058e-311` leaf stock. Commit `90113c0`
preserves the source-equivalent quotient primal and masks only quotient
tangents below the float64 representable-gradient boundary. The complete
update-302 audit then passed with zero bad samples, finite independent
component gradients, all hard counts zero, and an applied update.

An identity-checked diagnostic checkpoint fork at commit `b8c8dee` resumed the
formal update-256 state and passed every mixed-horizon update through 1024.
All 768 post-fix updates were applied; horizon counts were `431/317/182/94`
for 1/3/7/30 days, all exact hard counts stayed zero, and the sealed test split
was not used. The failed formal root remains rejected. See
[`rollout_turnover_ratio_gradient_20260730.md`](research/daily_coarse_graining/rollout_turnover_ratio_gradient_20260730.md).

The clean formal Experiment B rerun is complete. Slurm job `14425151` ran on
one V100 from training commit
`d6e43cc79f590e494f79fe54adba4ac2c28779b6`, completed in `04:16:53`, and
returned exit code zero. Both matched arms completed all 8,192 updates with
identical horizon counts `3277/2458/1638/819` for 1/3/7/30 days and did not
access the sealed test split. The immutable output root is
`runtime/outputs/training/canonical-669-rollout-stability-v2-d6e43cc`.
Control/candidate checkpoint SHA256 values are
`8a099c85f8c869398ffe2203b0fa1d558735b0f819ac606d112d452db2fb2ec1`
and
`4f36bb39839f8fba7a9cbd0283610bced3bd45c0688891bcdb4d051917320300`;
their training-report SHA256 values are
`6ca7985edd5b5b61beecd534a7f341a7173a6ef549b39ddccca3b93be2237069`
and
`ffd25be74ab739d15c45f66d04fc79d304e4877821b91f23f12d85da5749184c`.
This proves successful matched training, not promotion.

Commit `a9d2390` adds the fail-closed full-data screening evaluator. Local
validation passes 64 tests, Ruff, `py_compile`, shell syntax, and diff checks.
Real V100 smokes cover temporal, spatial, and joint model-selection shards,
large batches `512/512/256`, and a bit-exact 30-day versus `15+15`
restart-split probe with maximum difference zero. The evaluator covers every
valid 1/7/30-day teacher-forced and free-rollout window, the complete state,
process families, tendency bias, 12 named science fields, dynamic status and
hard constraints; incomplete evidence cannot be classified.

The formal matched screen completed as Slurm job `14430377` in `04:21:26`
with exit code zero. All 4,754 point-year references and every valid
teacher-forced/free 1/7/30-day window were evaluated; the report SHA256 is
`1387448b0b0831661daaaffc8985ef22602540e0c7e91fea7bfe96c085828bb0`.
The candidate is rejected: 114 of 223 relative gates fail. All one-step global
no-regression gates and all three 30-day global rollout gates pass, with
30-day temporal/spatial/joint error ratios `0.9181/0.7539/0.8047`. However,
none of the 7-day global gates reaches the required 5% improvement, 46/54
tendency-bias gates fail, and 57/108 named science-field gates fail. NPP,
growth and maintenance respiration, biomass, LAI, `carbon_32l`, and
`deepC_peat` are the repeated failures.

All exact hard and structural constraints pass, including zero unexpected
status/discrete/nonfinite/negative-stock errors and a bit-exact 30-day
`15+15` restart split. This is therefore an objective/architecture failure,
not a numerical-integrity or orchestration failure. Do not run confirmation
seeds, inspect the sealed test split, or begin 365-day validation for this
candidate. See
[`rollout_stability_screening_20260731.md`](research/daily_coarse_graining/rollout_stability_screening_20260731.md).

Experiment C is now frozen as a bounded causal-carbon hypothesis, but has not
yet run on a real GPU shard. It keeps the Experiment B one-step control as a
frozen 1,954,041-parameter parent and attaches an exact-zero
76,877-parameter adapter. The adapter can alter only 141 PFT14 columns:
same-day GPP, 12-part maintenance respiration, 32-layer carbon, and deep peat
carbon. The other 2,714 fast-day columns and the dynamic undefined head remain
parent-owned and bit-exact. The objective directly supervises this causal
`B_fast` interface, then separately supervises downstream carbon state,
signed flux bias, and stock-tendency bias; reset day-end `gpp_daily` is
explicitly excluded as a proxy for same-day GPP. DOC and litter are
no-regression guards.

The protocol SHA256 is
`35ede512bbc5f55881317cb700d7512d71993391c65f36ed8a5de087c4d4c3e7`.
Local architecture/objective/protocol regression and a complete synthetic
two-day retained-transition reverse-mode path pass. Paid matched training is
forbidden until an eight-update, real train/train-shard V100 feasibility gate
passes all 1/3/7-day, frozen-parent, protected-column, gradient, and hard-state
checks. This preparation does not establish that Experiment C improves
rollout quality. See
[`causal_carbon_adapter_experiment_20260731.md`](research/daily_coarse_graining/causal_carbon_adapter_experiment_20260731.md).

The Experiment C real-shard feasibility gate subsequently passed at commit
`79258fd`. Eight matched train/train updates covered 1/3/7-day horizons on a
free `gln01` V100. Every update was applied with finite gradients; all
unexpected/dynamic status, discrete, nonfinite, and negative-stock counts were
zero. The frozen 1,954,041-parameter parent stayed outside the optimizer, all
2,714 protected columns and the dynamic undefined head remained bit-exact,
and the maximum protected-column difference was `0.0`. Report SHA256 is
`ca8cacdccb42f20b94b5cd64796118cb170f7fbc7813c036cbc6b1edc8aa7fa8`.
The smoke checkpoints are not training parents. Formal train-only
independent-component gradient calibration must precede a fresh matched A/B
from the exact-zero adapter. See
[`causal_carbon_adapter_feasibility_20260731.md`](research/daily_coarse_graining/causal_carbon_adapter_feasibility_20260731.md).

The formal Experiment C lifecycle completed as paid job `14434127` at training
commit `8d24fd0`. It ran for `02:33:11` on one `gnall` V100 and exited zero.
All 48 deterministic train/train calibration records and both matched
4,096-update arms completed. Each arm has exact horizon counts
`2048/1229/819` for 1/3/7 days. Both final parent-invariance checks report
bit-exact protected columns and dynamic undefined head with maximum protected
absolute difference `0.0`; neither arm used sealed test data. Evidence hashes:

```text
control checkpoint:
4ee68de505f0163a62231e0310369882d46a0b72663b87d2a394c0e4ffeef720
control training report:
7400144a2f7143d5591e7a1bf2b49cfcd0c50d55385cee71dcd481c554e2170c
candidate checkpoint:
ea7193453e279bd94eb55984c629a0f530d0275466c1cd0daccf142754497892
candidate training report:
0a358a32f9b156c2d2434a0d11b6e0a8a35ceae47f3f6f0fdd7f3b78d7dc0eaf
```

This closes formal training execution only; it does not establish that the
candidate improves model-selection rollout quality.

The post-training screen was frozen before any candidate validation output was
available. It evaluates all temporal, spatial, and joint model-selection
windows at Day 1 teacher-forced and Day 7/30 free rollout. Classification
requires all 165 relative, 108 exact-zero, and six structural gates to pass;
the sealed test remains unread. After training, run a one-shard-per-slice free
`gln01` smoke before requesting the paid all-sample screen. The smoke
subsequently completed on GPU 1 in `286.46` seconds at evaluation commit
`375b8d2`. It covered one complete temporal, spatial, and joint shard for both
arms, with all unexpected-status, discrete, nonfinite, and negative-stock
counts zero. Both restart-split probes passed, and sealed test data remained
unread. The top-level smoke report SHA256 is
`f05d88d97ec93152487f22697e50361de705a6e1223964971bdfccdcd777b7cb`.
As preregistered, this limited smoke is not promotion-eligible; it authorizes
the paid all-sample screen.

The first paid submission, job `14436672`, exited after two seconds before
starting the evaluator because the Slurm wrapper directly executed a
non-executable checkout file. Commit `7091e1d` changes only that invocation to
explicit `bash` and adds a regression test; it does not alter the evaluator or
frozen gates. Replacement job `14436716` completed all 4,754 references in
`04:25:50` with exit code zero from the immutable `7091e1d` checkout.

Experiment C is rejected with decision
`stop_and_attribute_declared_screening_failure`. All 108 hard and six
structural gates passed, but 73/165 scientific relative gates failed:
11/12 causal-interface, 46/63 primary-state, 3/27 flux-bias, and 13/36
stock-tendency gates. All nine global terminal-state and all 18 litter/DOC
guard gates passed. The dominant failure is direct damage to the carbon-stock
boundary: `carbon_32l` interface error worsened by `7.33-9.32x` and
`deepC_peat` by `5.66-7.45x` across all three model-selection slices. The
matched report SHA256 is
`2b636981a5aeb73479c354558c09ed99cf398559c7f315bd56774f7f9822b5dd`.
Do not run confirmation seeds, 365-day validation, complete-chain validation,
or sealed test for this candidate. See
[`causal_carbon_adapter_screening_20260731.md`](research/daily_coarse_graining/causal_carbon_adapter_screening_20260731.md)
and
[`causal_carbon_adapter_screening_protocol_20260731.md`](research/daily_coarse_graining/causal_carbon_adapter_screening_protocol_20260731.md).

A single post-rejection attribution is now frozen before any successor is
built. It evaluates the rejected checkpoint with only
`carbon_stock_interface` set to exact zero while retaining the trained flux
adapter. The one-shard-per-slice smoke must reproduce the prior control,
recover `carbon_32l` and `deepC_peat`, retain at least 24/27 flux-bias gates,
and keep every hard and litter/DOC guard gate. It does not retrain or revive
Experiment C. See
[`causal_carbon_adapter_stock_ablation_protocol_20260731.md`](research/daily_coarse_graining/causal_carbon_adapter_stock_ablation_protocol_20260731.md).

That one-shard-per-slice ablation is complete at commit `7c53f79` and fails
its predeclared gate. It exactly preserves the original candidate's direct
flux interface, and its 27 downstream flux-bias metrics change by only about
`-0.23%` to `+0.31%`. It strongly repairs temporal and joint carbon-stock
error, but spatial `carbon_32l` and `deepC_peat` remain `13.71x` and `16.20x`
worse than the matched control. Thus neither independent stock residuals nor
an unconditional return to parent-owned stocks is acceptable. The next
candidate requires source-backed learned transfers with a conservative stock
update. No paid training is authorized yet. See
[`causal_carbon_adapter_stock_ablation_result_20260731.md`](research/daily_coarse_graining/causal_carbon_adapter_stock_ablation_result_20260731.md).

The subsequent source audit closes the ownership ambiguity. Contract v5's
1,376 compact `OK_LEAK` values contain 1,222 independent inventory values,
64 derived `deepC_peat` values, and 90 ratio or overlapping tracer/partition
values. `deepC_peat` is reset from `carbon_32l` before decomposition and must
not be a neural stock owner. Existing shards provide endpoint supervision but
not aggregate transfer labels or the 13 half-hour driver series required by
the exact source-backed scan. At that historical decision point, a bounded
successor predicted those drivers and executed the existing 48-step
`_paper_compiled_ok_leak_fold`. Its capture and replay evidence remains valid,
but the 2026-08-02 true-daily decision supersedes it as the final surrogate
design. See
[`carbon_budget_ownership_audit_20260731.md`](research/daily_coarse_graining/carbon_budget_ownership_audit_20260731.md).

The bounded train-only auxiliary asset and persisted replay gate are now
complete. The accepted plan selects 16 days from each of six training
landpoints across years and process conditions. All `96/96` target-day
captures pass the exact capture interface with no failed record; the
hash-bound reader verifies every source shard, state/target row, report, and
driver NPZ. Replaying each persisted 13-driver sequence through the current
exact 48-step scan reproduces all 14 `ok_leak.*` endpoints within `1e-12` for
`96/96` days. The global maximum absolute error is `4.5401182813264995e-14`;
six records are fully bit-exact and 77 have 13/14 bit-exact endpoints. The
remaining differences are float64 evaluation-order noise below the frozen
tolerance, not relaxed acceptance. The sealed test split was not used.

This closes capture-interface and persisted-scan replay sufficiency. It does
not assert equivalence of a reconstructed full outer seven-day Teacher block:
historical shards were produced in seven-day compiled blocks, and one-day
outer replay is not the accepted verification boundary. The next action is to
close conservation, feasible-perturbation/nonnegative-stock, real
forward/reverse finite-gradient, and bounded tiny-fit parent-no-regression
micro-gates before any paid neural A/B. See
[`ok_leak_auxiliary_capture_replay_20260801.md`](research/daily_coarse_graining/ok_leak_auxiliary_capture_replay_20260801.md).

The source-backed combined carbon-inventory diagnostic is now connected to a
diagnostic-only full-step output of the same 48-step scan. The production
return boundary remains unchanged. A real persisted-driver smoke for
`001.0-071.0`, 1961 Day 2 passes all 48 steps: all values are finite, all five
independent stock families are nonnegative, the diagnostic final carry is
bit-exact to ordinary replay, and maximum absolute closure is `2.9293e-9`.
The closure policy uses the source `min_stomate=1e-8` verdict from
`constantes_var.f90`; relative closure (`1.1884e-8` at the worst low-stock
step) is retained as a diagnostic and is not an invented source gate. Job
`14446767` completed in `00:06:38` with exit code zero. Feasible perturbation,
real forward/reverse gradient, and bounded tiny-fit gates remain.

A deterministic bounded selection tool is now ready for the existing
server-side v5 shards. It admits only the six spatial-train/temporal-train
landpoints and selects 16 days per point: cold-start anchor, low/high extrema
of six reachable dynamic process metrics and rank-space farthest fill.
`shumdiag_peat` and `runoff2peat` remain exact-zero contract fields because
the audited HYDROL-local SAVE branch is statically
`peat_hydro=F/branch_peat=F`; changing forcing or landpoint cannot activate
it. `PERMA_PEAT` carbon redistribution remains in the exact scan. Static
`fpeat` is report-only. The intended 96
days require about 44.58 MiB of uncompressed driver labels. Source shard,
day-start state, and endpoint hashes are mandatory, and sealed references are
filtered before file access. The first real plan
`500979ab...d9166a3` is rejected because its old static-`fpeat` proxy was zero.
The second `f7967d70...09d0f92` plan is also rejected because it required an
unreachable peat-hydrology branch and ranked tied zeros by date. Neither is a
capture asset. The accepted replacement is frozen at
`manifests/coarse_graining/ok_leak_auxiliary_capture_96day_v1.json`, canonical
SHA256 `8ebe5345...3b922c7`. It passes all 19 independent checks over 96 unique
train-only days, including exact selected-row state and target hashes. The
complete capture and replay result is recorded in
[`ok_leak_auxiliary_capture_replay_20260801.md`](research/daily_coarse_graining/ok_leak_auxiliary_capture_replay_20260801.md).
See also
[`ok_leak_auxiliary_capture_selection_protocol_20260731.md`](research/daily_coarse_graining/ok_leak_auxiliary_capture_selection_protocol_20260731.md).

The restartable one-process capture and persisted-replay runners are now
validated on all 96 accepted records. They verify the source generation plan,
stage each day atomically, and resume only hash-identical passed records.
Capture-interface failures and persisted-replay failures are both zero.

The 2026-08-14 gradient-balanced Attention smoke is complete and rejected.
At commit `e22a235`, a free-`gln01` run continued the accepted update-1000
Attention state for 10 matched updates. The candidate used a per-update
minimum-norm convex combination of unit gradients for one-day anchor, NPP,
AGB tendency, and BGB tendency; the control retained the staged-v3 objective.
All 10 candidate raw directions pass the `0.05` minimum-cosine gate, candidate
clipping is `0/10`, and all hard failures are zero. On the frozen six-artifact
7/30-day subset, however, the candidate worsens the four-key NPP/AGB/BGB/live
biomass composite by `8.39%`, worsens every key, and improves `0/6` cells.
Maintenance improves `30.80%`. The report SHA256 is
`d84a6a764efdbd3b4d8cb74d5ddd0965c301b1ea162671a3db1b7a465bb83834`.

This rejects raw-gradient minimum-norm combination followed by inherited
AdamW state; it does not reject the Attention architecture. The conditional
100-update A/B is closed. The next active task is to freeze a zero-update
realized-step audit of whether AdamW momentum/preconditioning preserves the
raw common-descent direction. No further optimizer update, paid job, sealed
test, or architecture change is authorized by this result. See
[`successor_gradient_balanced_smoke_20260814.md`](research/daily_coarse_graining/successor_gradient_balanced_smoke_20260814.md).

The follow-up zero-update realized-step audit completed on free `gln01` at
commit `6cec975`. The raw minimum-norm direction passes `4/4` groups. After
the actual inherited-AdamW transform, minimum component cosines fall to
`0.00039-0.00464` and no group passes the `0.01` direction gate. Reset AdamW
also passes `0/4`; two groups have a negative component cosine. Exact virtual
all-component loss checks pass `3/4` inherited groups and only `2/4` reset
groups. Hard and nonfinite counts are zero, and no proposed state is retained.
The report SHA256 is
`4b987902e4baf40e5223ae9201be416ff8f61f64080309056302ba947bf47037`.

This rejects both inherited and reset AdamW as realizations of the global
common-descent direction. The active gate is a zero-update shared
direct-direction trust-region audit; no optimizer-reset smoke or further
training is authorized. See
[`successor_realized_step_audit_20260815.md`](research/daily_coarse_graining/successor_realized_step_audit_20260815.md).

The direct-direction trust-region audit then completed on free `gln01` at
commit `55c30a3`. With AdamW and all coordinate-wise optimizer transforms
removed, global step norm `0.001` decreases all four component losses in all
four train/train groups. Median relative changes are `-0.0116%` for the
one-day anchor, `-0.1652%` for NPP, `-0.3081%` for AGB tendency, and
`-0.0317%` for BGB tendency. Step norm `0.003` is rejected because one group
regresses by `19.43%`; all larger tested norms are also outside the shared
trust region. Hard and nonfinite counts are zero, no update is retained, and
the report SHA256 is
`ccc4e1fd29ddebf2d55976aec0282c62a14c814c6b148e1d862f9513379e1fbd`.

This is positive local evidence for directly realizing the common-descent
direction, not a promotion result. The active gate is a no-checkpoint
10-update direct-direction smoke at fixed global norm `0.001`, followed by
the accepted non-test 7/30-day subset. See
[`successor_direct_trust_region_audit_20260815.md`](research/daily_coarse_graining/successor_direct_trust_region_audit_20260815.md).

The 10-update direct-direction smoke completed on free `gln01` at commit
`621cf57` and is rejected. All ten proposed updates pass finite and process
hard gates, and every component has a negative median relative change, but
only `6/10` updates decrease every component. The maximum single
batch-component regression is `19.11%`. On the accepted non-test 7/30-day
subset, the NPP/AGB/BGB/total-live composite worsens `1.4348%`, every primary
key worsens, and `0/6` cells improve. Maintenance-respiration RMSE improves
`5.7566%`. No final parameters or checkpoint are retained. Report SHA256 is
`48e8e7da98a5b2c52e045af5653be5aea0bb85f970cb34b6f7ac40a65b50abf7`.

Removing AdamW therefore fixes direction realization but not direction
transfer. The active gate is a zero-update robust multi-batch audit using ten
mixed batches and all 40 batch-component gradients at the frozen update-1000
parameters. See
[`successor_direct_direction_smoke_20260815.md`](research/daily_coarse_graining/successor_direct_direction_smoke_20260815.md).

The accumulated multi-batch audit completed on free `gln01` at commit
`ab7f41a`. One effective update now covers 40 point-years, 320 one-day anchors,
and 40 30-day windows. The direction is clear for the four group-mean
objectives (minimum cosine `0.2736`) and sharply reduces instability: at global
norm `0.0003`, the worst group-component regression is `0.0323%` rather than
the per-batch smoke's `19.11%`. However, only `4/10` groups improve every
component, so no frozen step passes. Norm `0.001` is nonlinear and regresses up
to `28.89%`. Hard and nonfinite counts are zero, and the report SHA256 is
`0601fcd6bc9f51b638bb2de14087be152320e65fa6430e5ddcab254990c0d847`.

This confirms that effective batch size matters but ordinary averaging is not
enough. The active gate is a zero-update group-aware 40-objective common
direction audit. See
[`successor_multibatch_direction_audit_20260815.md`](research/daily_coarse_graining/successor_multibatch_direction_audit_20260815.md).

The group-aware 40-objective audit then completed on free `gln01` at commit
`b0ebda9`. A common direction exists with minimum constraint cosine `0.050356`.
Every tested norm through `0.0003` improves all four components in all ten
groups, and all hard/nonfinite counts are zero. The formal gate is not passed
because the one-day anchor and BGB tendency median decreases at `0.0003` are
`1.6217e-5` and `5.1778e-5`, below the required `1e-4`; NPP and AGB tendency
already pass. No parameters or checkpoint are retained. Report SHA256 is
`f9c55f15dbd4744097dc5bc5089a35447d2cc63fcad41a00650622e3d809a4e3`.

This solves the observed direction-sign conflict and leaves a bounded
finite-step question. The active gate is a zero-update extension at norms
`0.0005, 0.001, 0.0015, 0.002`, frozen before execution. No architecture
change, training continuation, validation read, or sealed-test access is
authorized. See
[`successor_group_aware_direction_audit_20260815.md`](research/daily_coarse_graining/successor_group_aware_direction_audit_20260815.md).

The bounded finite-step extension then completed on free `gln01` at commit
`f0b2235`. Norm `0.002` passes all ten groups and all four component median
effect gates. The one-day anchor, NPP, AGB tendency, and BGB tendency medians
are `-1.2176e-4`, `-1.0193e-3`, `-1.3361e-3`, and `-3.4462e-4`; no individual
component regresses. Hard/nonfinite counts are zero and no parameters are
retained. Report SHA256 is
`a8b448488a25e924a36dc356a0bdba6f6cc213c6509e7e21ac3e4af0635edf9a`.

The active gate is a five-update frozen-scalarization smoke that realizes the
14 positive constraints through eight group-level reverse passes per update.
It must preserve all ten groups locally and improve the established non-test
7/30-day NPP/biomass composite before any rotating-group training design is
opened. See
[`successor_group_aware_step_extension_20260815.md`](research/daily_coarse_graining/successor_group_aware_step_extension_20260815.md).

The parent-frozen scalarization smoke then completed on free `gln01` at
commit `1bbb163`. Update 1 reproduces the accepted direction and improves all
40 constraints. Update 2 at fixed norm `0.002` regresses multiple constraints,
worst group-5 NPP by `0.0861%`, so it is not applied. The one accepted update
improves all six independent 7/30-day cells and each NPP/AGB/BGB/total-live
aggregate, but only by `0.00922%` in the combined primary score. Report SHA256
is `1aeb93ae49e29496c43dfd9589f7e87f6e6be20c214a23a6d266aaca57360ca6`.

The active gate is a bounded backtracking smoke at norms
`0.002, 0.001, 0.0005`. It keeps the eight-VJP realization, stops if no useful
step preserves all 40 constraints, and retains no checkpoint. See
[`successor_group_aware_scalarization_smoke_20260815.md`](research/daily_coarse_graining/successor_group_aware_scalarization_smoke_20260815.md).

The matched backtracking smoke completed on free `gln01` at commit `4118c72`.
At the second proposal, worst regressions scale from `8.6109e-4` at norm
`0.002` to `4.3088e-4` at `0.001` and `2.1552e-4` at `0.0005`. No second
update is applied, and all hard/nonfinite counts remain zero. This closes
smaller fixed-weight steps and requires direction recalibration. Report
SHA256 is
`c2691824b6548be05337620ffb91cab1994b58bd83293a56968d89574a074261`.

The zero-retention exact 40-gradient audit at the state after the first
admitted update completed with minimum cosine `0.0486966`, passing norm
`0.002`, and support overlap `12/14`. Its efficient active-set realization
then passes with 14 component reverse passes and one all-group JVP round. See
[`successor_second_state_recalibration_audit_20260815.md`](research/daily_coarse_graining/successor_second_state_recalibration_audit_20260815.md).
