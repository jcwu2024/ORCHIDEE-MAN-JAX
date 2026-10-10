# Original full-state schedule continuation results — 2026-10-05

Four additional real paired updates completed, with independent old/new chains
advancing step 1→5 and all requested timing samples saved. The three post-first
complete-update medians are **old 9.810149s / new 7.647413s = 1.282806×**.
This is a useful bounded schedule gain: 28.28% higher throughput for the same
declared update workload. The cumulative10× research target remains unmet.
**The route remains strict_failed and cannot be adopted:** both
artifact 275 evaluations produced severe daily biomass initial-state derivative
instability at roughly 10^116–10^118 scales. Finite state gradients and passing
parameter/AdamW comparisons do not establish valid state AD.

The CPU follow-up below closes the tiny-biomass local numerical mechanism:
four real step 1/row domains reproduce the original source boundary, a guarded
algebraic correction preserves stocks and removes the spurious leaf-scale
derivative, and one real public physical-tail loss drops its extreme gradient
from2.52e117 to0.02186. The correction is integrated locally; the
historical GPU strict failures remain recorded. The independent corrected
full-network/CUDA leaf validation below now passes its four target domains;
residual DOC/control state drift prevents all-state acceptance. Real T7 two-day feedback is measured, rather than inferred from Teacher
states; it is not complete T7/T30 acceptance.

## Completed execution and unchanged scope

User approved the prepared free two-V100 900-second/1800 GPU-second window with
“批准”. Launch 2026-10-05T13:44:45+0800 on free gln01/ibc13b02n04, parent physical
GPU0/worker GPU1, V100-SXM2-32GB; affinity 0-55, load 0.00/0.01/0.05, both GPUs
0MiB/0% with no compute processes. Workdir immutable c599236 checkout with the
declared staged overlay. Old parent daily 32/worker T7+T30; new parent daily 32+T7/
worker T30. Eight distinct micro4 slices, full T7/T30, public losses/AdamW,
64,884,481 FP32 parameters, full native parameter and initial-state AD, FP32
neural/FP64 physics, original strict rtol 1e-5 / atol 1e-6 and 24GiB gates remained.

Each chain restored its own actual step 1 parameters and AdamW moments, whose
CPU admission and runtime array identity passed. All branch versions matched
the input step 1/2/3/4, and each public optimizer reached 2/3/4/5. Cumulative
attempts are 5 each, retaining the original first attempt and 16-attempt cap.
Artifacts 275/742/275/742 follow the true [742,275] cursor. Zero warmup proposals.
Zero model-gradient compilations; all four saved executable instances report
model_compiles=0 and validated full-state ABI. No paid job/formal training ran.

## First boundary and three post-first measurements

|Chain step|Artifact|Execution order|Old complete seconds|New complete seconds|
|---|---|---|---:|---:|
|2|275|new → old|10.371270850|15.362782503|
|3|742|old → new|9.671098562|7.641205700|
|4|275|new → old|9.897219468|7.647413101|
|5|742|old → new|9.810149368|7.844491290|

Step 2 is the restart/load/helper/weight boundary and is excluded from the
post-first result. The old run's first 6.997× is also excluded. The new run
reached ready_no_model_compile at 307.982464s, with
actual stage records rather than reconstructed old timestamps.

|Post-first median seconds|Old|New|
|---|---:|---:|
|Complete update|9.810149368|7.647413101|
|Branches including synchronization/IPC|9.683838045|7.555750892|
|Public AdamW/sync|0.016132953|0.015236164|
|Daily32|4.315901473|4.321137302|
|T7|1.593687402|1.605112007|
|T30|6.752701640|6.752937373|
|Recurrent synchronized branch|8.347005147|6.753334841|
|Worker total|8.718680380|6.943592789|
|Component merge/sync|0.010600256|0.010296116|
|Merge/check/optimizer|0.116755560|0.082490670|
|Parent parameter device_get|0.047307976|0.013254028|
|Request pickle write|0.331171706|0.329931790|
|Response wait/read/unpickle|9.253679138|7.211597314|
|Worker result device_get|0.299722580|0.135798819|

Medians of different intervals are not an additive decomposition. New T30
remains the longer synchronized card branch (~6.75s); moving T7 beside daily
reduces the old serialized T7+T30 critical path. AdamW is now ~0.015s, so the
large first-call AdamW contrast does not identify a continuing bottleneck.
The new complete branch interval includes parameter transport/IPC and waits;
it is not pure GPU compute or pure transfer. Even eliminating all branch
overhead while holding T30 unchanged limits this schedule's further gains;
the recurrent T30 compute and verified transport/execution structure are the
next performance candidates. This result does not authorize changed capacity,
loss/sampling, recurrent gradient length, precision or scientifically active
outputs. No additional GPU run is needed to conclude this bounded speed screen.

Full comparison/download/preservation is outside each complete update and
inside the occupied window. Three post-first combined outside-update median
was 11.453381s per pair. Actual
stage intervals give:

|Post-first per-pair stage interval|Median seconds|
|---|---:|
|state_artifact_save|0.034986|
|full_comparison_after_state_save|8.695123|
|parameter_arrays_download|0.896950|
|parameter_arrays_save|1.607844|

Comparison and preservation are diagnostic costs, not discarded or silently
included in the complete-update speed ratio. All lifecycle time is charged.

## Severe state AD limitation and strict comparisons

All four pairs passed full mixed parameter-gradient 64,884,481 coordinates,
updated parameters 64,884,481 and public AdamW 129,768,963, with no strict
failures. Every daily/T7/T30 loss/auxiliary/component parameter-gradient
comparison passed. All physical/science hard counters were 0 and all retained
parameters/optimizer/state gradients finite. These gates cannot certify the
scientific validity of extreme state derivatives.

|Chain step|Artifact|Daily batch-mean failures/123328|Raw micro4 mean failures/123328|T7/3854|T30/3854|
|---|---|---:|---:|---:|---:|
|2|275|145|240|1|1|
|3|742|99|141|11|1|
|4|275|124|182|1|2|
|5|742|139|206|3|2|

These are new changing-chain comparisons. Inherited original step 1 failures
daily 34/raw 46/T7 2/T30 0 and earlier micro4 self-replay 4/1/3 remain separate,
unresolved and sticky. Daily batch-mean state derivatives are each micro4
mean derivative divided by 8, unrelated to input-state standardization.

Independent CPU NumPy inspection reproduced every fail count, all failed
coordinates and each arm/field extrema. Both artifact 275 daily calls show the
largest error at sample 17, state 2243, actual contract
slowproc_stomate_previous_step_state.biomass[2231,2255), local flat 12; raw
address micro4[4,1,2243]. Step 2 old=-3.743087738404725e116,
new=1.5760369424862002e116, error5.319124680890925e116; step 4
old=2.99447019072378e117, new=7.0921662411879e116,
error2.28525356660499e117. Raw mean errors are 8× larger, reaching
1.828202853283992e118. This is not near-tolerance numerical drift and affects
both schedules. Across all four pairs and every contract field, only the two
artifact 275 daily biomass fields exceeded10^50 absolute derivative magnitude;
the strict-failed entries above10^100 were sample 16 and 17 at state 2243.
Artifact 742 daily max differences were 0.364185295/0.247681748,
with field magnitudes roughly 10^5; no same extreme occurred in those two calls.

CPU inspection of the actual compact input bundle found artifact 275's
sample 17/state 2243 biomass=5.805124529028718e-139, versus 122.39629961844614
for artifact 742. Other same-row biomass values include 3.2323629572321526e-139
and 9.684152147898288e-154. All biomass inputs are finite/nonnegative. The
near-zero inputs are a source-diagnostic lead, not proof of a denominator,
min/max/tie, compiler or scheduling cause. Independent chains already have
small parameter differences from previous accepted numerical updates; this
comparison does not isolate identical-weight schedule replay. Parameter AD
passing does not demonstrate that the unstable state AD path is harmless.

The full state/field extrema, strict rows and actual inputs are retained in
original-full-state-schedule-followup-state-coordinate-analysis.json and
original-full-state-schedule-followup-input-biomass.json. No tolerance
relaxation, masking away the coordinate or production adoption occurred.

## Bounded source audit and next local mechanism question

The source owner completed a bounded read-only audit. No ordinary wrapper
state/sample misbinding or daily÷8 merge error was found. The executed wrapper
[FullStateADEvaluator](E:/ORCHIDEE-MAN-JAX/scripts/hpc/shared_full_state_ad_execution.py:203)
uses the same micro4 sample for explicit continuous_state and samples;
[microbatch slicing](E:/ORCHIDEE-MAN-JAX/scripts/hpc/shared_daily_microbatch.py:211)
slices the whole sample tree at 0/4/…/28;
[state merge](E:/ORCHIDEE-MAN-JAX/scripts/hpc/probe_shared_original_full_state_schedule.py:294)
preserves eight raw mean4 derivatives and concatenates/divides by 8. That
operation cannot manufacture a10^116 amplification. Shape/tree/dtype ABI
checks admit calling convention; they do not certify contents or dynamic275
state-gradient scientific validity. Original admission covered only 742's
first micro4, rather than all32 rows of both declared artifacts.

Physical base carbon_kernels.py/ad_primitives.py and shared_training.py/model.py
were compared with git show c599236 and verified unchanged before the scoped
numerical correction described below. Concrete sensitive
leads in that base are
[maintenance respiration ratios](E:/ORCHIDEE-MAN-JAX/jax_orchidee/stomate/carbon_kernels.py:4002)
in npp_closed_update and
[leaf-age turnover ratios](E:/ORCHIDEE-MAN-JAX/jax_orchidee/stomate/carbon_kernels.py:4568)
in turnover_leaf_age_fall. Positive but extremely small denominators can enter
these divisions. The
[ratio tangent](E:/ORCHIDEE-MAN-JAX/jax_orchidee/ad_primitives.py:26)
screens denominators below approximately 1.49e-154 in FP64; a 1e-139 positive
quantity is above that screen. Nearby threshold/min(0.99,…) predicates are
also source questions. Bounded min/max/tie subgradients alone do not explain
the huge result. None of these statements proves that the actual275 row16/17
post-NPP/turnover denominator or branch hits a particular operation.

Historical compilation correspondence is incomplete. Of five recorded
admission-overlay SHAs, current backbone/v4 match, while dual_helper,
dual_probe and field_expert do not. Thus current dirty overlay source cannot
be presented as the exact historic binary graph. The c599236 physical base
is a verified source lead; no complete historical overlay/HLO causality was
established. Graph MCP/coverage transport was unavailable for this audit;
exact-source fallback supports these bounded positive findings without an
exhaustive graph claim.

## Actual CPU physical-tail diagnosis

The old step 1 snapshot and actual artifact 275 row17 now have a reconstructed
CPU forward and public daily scalar-loss physical-tail VJP. The neural
context/raw outputs were computed once from that exact snapshot/sample and
replayed with identity, source and ABI checks; their derivatives with respect
to state are deliberately omitted. This is physical-tail evidence, not full
network state AD or historical CUDA-binary replay. CPU execution used the
immutable c599236 checkout, four CPU cores, JAX CPU only, no `--nv`, unchanged
FP32 neural/FP64 physical arithmetic, masks, losses and gates. No GPU was used.

The captured maintenance denominator is1.809325500401829e-4. Actual leaf
allocation is zero and NPP's `min_stomate=1e-8` disables its tiny-leaf age
update; that boundary forwards the existing age fractions. Turnover receives
leaf mass5.804965023785553e-139 and fractions
[0.002545695295720209,0.011313425835462453,0.0434304139324322,0.9427104649363851].
Its sequential post-stock denominator is3.272368041512384e-139, positive and
above the existing FP64 ratio tangent cutoff1.4916681462400413e-154.
These are captured intermediate operands, rather than initial-stock guesses.

Original physical-tail scalar loss is0.14390864878732995; all hard
counts are zero. State coordinate2243 derivative is
2.5216591079779203e+117
(its mean32 row contribution is
7.8801847124310009e+115).
Both same-process VJPs reproduce all3854 gradient coordinates and loss exactly;
the first costs85.166213s, warm repeat
25.066982s. Forward/capture costs
82.439283s and the
whole process197.274482s. This stable enormous derivative can
arise with the network-to-state path absent; finiteness alone is insufficient.

Independent local checks execute explicit frozen function text in separate
globals, rather than importing a changing workspace callable. Original
function SHA is `bb8674f3d2ceadb28df907346d18f6300dd87f46d09ff6010c795f72dbee6485`; guarded candidate SHA
is `f6240cf5ce64a47b9bbe0647b5e5715305e9a9230ea77f2bd9f42981ae35a93c`. The early unbound full-kernel zero
result was withdrawn after source-version isolation; it is not old-source
evidence. At the actual captured boundary, original raw leaf-mass fraction
JVP has maximum absolute6.785441066170827e122; four basis VJPs reach
3.305279843951243e122 and a nonuniform output-cotangent VJP is
-6.610559687902486e+122. The quotient primitive
alone agrees with its analytic derivative. The defect is numerical cancellation
in the composite sequential stock/fraction calculation, whose exact fraction
derivative with respect to the common leaf-mass scale is zero.

The diagnostic candidate preserves the source-order stock/turnover updates,
rates and branch masks. It computes fractions from the sequential relative
identity: p0=1, delta_rel_i=-p_(i-1)*f_i*rate_i,
p_i=p_(i-1)+delta_rel_i, frac_i=(f_i+delta_rel_i)/p_N.
It applies only when actual post-stock is positive and at least the original
ratio cutoff, with finite nonzero p_N; original ratio fallback remains for
zero/negative/below-cutoff/degenerate cases. No epsilon, input clamp,
stop-gradient or precision change is introduced. Fortran provenance is
stomate_turnover.f90, turnover leaf fall569–665, fraction update650–665.
Actual local candidate leaf-mass JVP/basis/nonuniform VJPs are zero;
fractions differ by at most3.3306690738754696e-16.
Stocks/turnover are bitwise equal. Their leaf-mass derivatives remain
0.56371882140616181 and
0.43628117859383803; fraction and age directions
agree at FP64 rounding scale. Zero, negative and below-cutoff fallback
values/derivatives match. Local correction evidence is scoped to this source
boundary and does not waive full training or scientific gates.

The same immutable CPU checkout then applies only the candidate function
through an in-memory overlay after verifying the original function SHA;
the remote checkout remains unchanged. Exact snapshot, sample, neural cache,
loss and precision identities agree. Candidate physical-tail scalar loss is
0.14390864878732995; loss and all captured next-state values/masks
pass unchanged strict rtol 1e-5 / atol 1e-6 comparisons and hard counts stay zero.
Leaf-mass scalar derivative changes from2.5216591079779203e+117 to
0.02186144081328182 (mean32 row0.00068317002541505687).
Only that one coordinate changes beyond the original strict gate; the other
3853 state-gradient coordinates pass. Candidate same-process VJPs reproduce
all coordinates/loss exactly, at 85.176896s
and warm26.205508s; candidate total
198.645590s. Original and candidate stock/turnover
values remain bitwise equal. This is a
controlled source-formula intervention at the reconstructed physical tail;
it does not establish exact historical CUDA-binary causation or independent
full-network/scheduling parity. The original GPU strict failures remain sticky.

This narrowly guarded algebraic correction is accepted as an ordinary source
numerical fix and is integrated in the local `turnover_leaf_age_fall` function.
Production formula identity is checked against the executed candidate source.
The immutable remote checkout and previously compiled CUDA executables are
unchanged. The CPU evidence alone did not establish corrected full-network/GPU
validation; the independent bounded CUDA result below now does, without adoption.

All four requested step 1 snapshot/row combinations now have actual source
boundaries. The additional three forwards completed in one bounded300-second
CPU window: old row16 cold138.816831s, new row17 warm10.735579s and new row16
warm10.075548s (sum159.627958s). Prepared objects and the two neural forward
executables were reused, with current parameters and inputs remaining dynamic;
neural outputs were recomputed for each own snapshot/sample. All scalar
forwards are finite and hard counts zero. No extra public-loss VJP was run.

| Snapshot | Row | Incoming leaf mass | Actual post-stock denominator | Original raw-mass fraction JVP max | Guarded derivative |
| --- | ---: | ---: | ---: | ---: | ---: |
| old | 17 | 5.804965024e-139 | 3.272368042e-139 | 6.785441066e+122 | zero |
| old | 16 | 1.696730560e-136 | 9.659805365e-137 | 1.149322354e+120 | zero |
| new | 17 | 5.804965024e-139 | 3.272368042e-139 | 6.785441066e+122 | zero |
| new | 16 | 1.696730560e-136 | 9.659805365e-137 | 1.149322354e+120 | zero |

The frozen original source reproduces all captured stock/turnover/fraction/
critical-age outputs bitwise in all four domains. Every actual post-stock
branch is positive and above the existing cutoff; candidate raw-mass fraction
JVP, basis VJPs and nonuniform VJP are zero in each. Stock/turnover values and
mass derivatives remain exact. Fraction forward differences are≤3.33e-16,
normal fraction/age directions differ≤1.67e-16/2.04e-18, and original fallback
policies remain. This is four-domain local source coverage, not all Fortran
branches or complete neural AD. Local production/capture/loader tests pass21
focused checks in18.14s, with179 deselected; source function SHA matches the
executed guarded candidate. Exact cutoff and adjacent representable-value
policy checks also pass (8 conditioning tests,13.43s); no broad rerun follows.
The existing pytest cache_dir warning is benign.

Daily rows16/17 are days143/153 of1970; both PFT14-present masks are true.
Their initial states do not match the actual T7 day179–185 or T30 day69–98
Teacher inputs/initial states. Teacher leaf biomass is38.7226–38.8263 on T7
and 36.4596–37.2336 on T30, with all biomass targets defined. These input
facts do not establish predicted recurrent trajectories. Existing recurrent
auxiliary artifacts retain scalar losses/hard counts without per-day predicted
states. A warm scalar VJP duration is not a recurrent per-day measurement;
do not infer37-day safety or cost from it.

A separate CPU window now measures actual T7 days179/180 with the old step 1
snapshot and corrected in-memory source overlay: total 144.471487s,
cold first day123.365745s and warm second day16.808618s,
under the300-second external cap. NPZ inspection confirms day0's prediction
equals day1's state input exactly. Both backbone and decoder execute every
day with dynamic inputs; day2 contexts and raw outputs actually change.
No Teacher state injection or frozen daily neural-output replay is used.
Predicted leaf biomass equals the actual post-turnover denominator,
38.460361882863594 then
38.208096229096853; both positive and represented.
All public physical/mask/discrete/budget and shared-stock-negative counts are
zero for both days. This establishes this two-day forward feedback/cost and
its measured boundary values only. Remaining T7/T30 predictions, recurrent
derivatives and science accuracy have not been tested by this prefix.

The completed original repeat, candidate comparison, three-domain batch and
two-day feedback launches exited0 and are closed. Earlier deployment-repair
and timeout attempts are preserved as incomplete evidence; no CPU diagnostic
process remains.
The authorized bounded corrected full-network/CUDA check is now completed
below, with four verified target derivatives and residual DOC/control repeat
differences disclosed. It does not authorize optimizer execution or establish
complete recurrent-state AD acceptance. Resume the speed work using this
bounded evidence; do not make every historical gradient discrepancy or all37
predicted days a prerequisite for the main speed research.

CPU evidence under outputs/diagnostics/shared-latent-reader-20261004:
biomass-amplification-cpu-old-row17-first6.json/.npz/.neural.pkl;
biomass-amplification-cpu-old-row17-vjp.json/.npz;
biomass-amplification-cpu-old-row17-repeat.json/.npz;
turnover-actual-boundary-candidate-check.json;
turnover_leaf_age_fall.original-source.py/.guarded-source.py;
biomass-daily-recurrent-metadata.cpu.json/.compact.json;
biomass-amplification-cpu-old-row17-candidate.json/.npz;
biomass-physical-tail-candidate-comparison.cpu.json;
compare_biomass_physical_tail_cpu.py.
Four-domain source checks: turnover-four-actual-domains-check.json and each
capture's .turnover-source-check.json; production tests:
turnover-source-fix-local-checks.json. Feedback evidence:
biomass-feedback-cpu-t7-prefix2.json/.npz/.acceptance.json and
inspect_biomass_recurrent_prefix_cpu.py. Launcher argument expansion receipts
cover VJP-repeat, candidate overlay, three-domain batch and true prefix2.
Final read-only close receipt biomass-cpu-diagnosis-final-close-confirmed.txt
at remote2026-10-05T17:51:15+08:00 confirms no diagnostic process, both GPUs
0MiB/0%, and an unchanged immutable physical source. The remote older Git
requires changing directory rather than git -C; its first status-only check
error is preserved separately and the corrected close check exited0.
Source-bound loader and capture tests retain immutable runtime safety;
no paid job, full optimizer update, new GPU window or tolerance relaxation
belongs to this CPU diagnosis.

## Corrected full original neural CUDA leaf validation — independent grant

The corrected leaf-age fraction now has bounded **complete original neural
CUDA validation** for frozen old/new step1 snapshots, artifact275 rows16:20.
The four required old/new × rows16/17 coordinate2243 gradients pass independent
complete-public-loss CPU JVP at unchanged rtol1e-5/atol1e-6. This validates the
specific leaf correction and its full neural state dependence; it does not
accept every initial-state gradient or T7/T30, and it is not model accuracy or
formal training acceptance.

Exactly one corrected micro4 native parameter+state AD graph was lowered and
compiled, with 64,884,481 dynamic parameters and original FP32 neural/FP64
physics. Its source is the guarded f6240c leaf function and exact historical
5d7714 field / ebf730 backbone adapters from their historical transfer directory,
using the unchanged immutable c599236 checkout with the function overlay.
The public scalar loss and mean4 reducer, physical masks, cutoff/fallback,
tolerances and capacities are preserved. No neural outputs were frozen; no
stop-gradient, optimizer update, loss/precision change or paid job ran.
Lowering took 123.385051s;
backend compilation and immediate binary preservation took
823.700683s. The sole
binary and exact identity sidecar were saved before ABI checks or model calls.
No second model graph was compiled or needed for recovery.

Old/new each ran twice on identical inputs, with complete loss/auxiliary,
parameter/state gradient arrays preserved before checks. Every call is finite,
all eleven physical hard counts are zero, and 64,861,825 parameter plus 15,332
state gradient entries are nonzero. Repeated complete parameter gradients,
ordinary forward/gates and all defined state diagnostic values/masks pass.
The previously admitted original full-state binary was loaded without compiling
and evaluated with both exact step1 snapshots and the same micro4. Its complete
parameter gradients and shared forward/gates pass strict regression against
the corrected binary. Full state-gradient comparisons retain34/38 strict
differences for old/new, including the corrected leaf targets and other
unproven differences. The correct target derivative is checked independently;
it is not required to reproduce the old erroneous huge derivative.

Independent CPU evaluation recomputes the historical neural backbone/head and
the complete scalar public loss. Native JVP component jit remains transparent
to AD; there is no neural-output replay, custom VJP or GPU execution. All four
domains passed finite/physical checks in 324.488158s. Dividing
the scalar derivative by4 is the exact mean4 row contribution; further mean32
scaling would divide by8, and no mean32 acceptance is inferred.
The earlier0.0218614408 scalar derivative held neural outputs fixed and omitted
their state dependence. Here the complete old-row17 scalar JVP is
0.0218610684583, with neural state paths and numerical execution included.
Equality to the earlier partial derivative is not the correctness test;
the independent complete JVP and CUDA VJP comparison below is that test.

|Snapshot:artifact row|CUDA state VJP2243|Complete scalar CPU JVP /4|Absolute difference|Strict gate|
|---|---:|---:|---:|---|
|old:16|0.00535889106801421|0.00535889470336838|3.63535417e-09|pass|
|old:17|0.00546526436130684|0.00546526711457632|2.75326948e-09|pass|
|new:16|0.00535889138180588|0.00535889436539323|2.98358736e-09|pass|
|new:17|0.00546526405789278|0.00546526667009053|2.61219775e-09|pass|

Coordinate2243 repeat differences also pass for all four micro4 rows, including
controls18/19. Full initial-state repeat parity remains false: old has 3
strict failures in DOC; new has those 3 DOC failures plus control row19,
coordinate2250 in biomass[0,1,7,0]. Maximum complete-state repeat differences
are 12.7964744605/5.48420334049, occurring in DOC; the table lists only actual
strict failures, since relative tolerances can admit larger absolute drift.
This is a separate residual state-AD issue, not evidence against the four
verified leaf target derivatives. The raw probe additionally marks two
diagnostic .other leaves false because their identical undefined NaNs were
compared without equal_nan. CPU inspection checks exact masks and every defined
value and discloses undefined observations; it does not discard active failures.

|Snapshot|Artifact row|Coordinate|Field|Repeat absolute difference|Strict allowance|
|---|---:|---:|---|---:|---:|
|new|17|1778|slowproc_stomate_previous_step_state.DOC|0.00345960184|0.00269069954|
|new|19|1834|slowproc_stomate_previous_step_state.DOC|8.22515925e-06|1.37659575e-06|
|new|19|1841|slowproc_stomate_previous_step_state.DOC|0.00645976211|0.000198401779|
|new|19|2250|slowproc_stomate_previous_step_state.biomass|0.000779151917|0.000736448734|
|old|17|1778|slowproc_stomate_previous_step_state.DOC|0.0108112557|0.00269059374|
|old|19|1834|slowproc_stomate_previous_step_state.DOC|1.12161263e-05|1.37670541e-06|
|old|19|1841|slowproc_stomate_previous_step_state.DOC|0.00861301615|0.000198256436|

Compiled estimated live bytes are 11,348,524,788;
observed JAX live peak is 11,368,941,056B (10.588151GiB), below 24GiB.
Independent NVML sampling inside the charged process group recorded
18,613MiB (18.176758GiB),
including allocator/device occupancy, also below 24GiB. JAX allocator pool peak
18,251,513,856B is distinct from its live-byte peak.

The new isolated 7200 GPU-second ledger settled four one-card sessions:
runtime preflight 27.957239s, an ordinary missing-helper startup repair 1.772803s
(before lowering), corrected compile/repeats/persistence/cleanup 968.850916s,
and saved-original control 27.307186s. Actual total
**1025.888144 GPU-seconds / 0.284969 card-hours**,
remaining 6174.111856; no historical funds were used or
reset. The chosen physical GPU1 was idle before each launch. Final read-only
close at 2026-10-05T19:46:25.365427+08:00 confirms no GPU1 compute process and 0MiB/0%,
unchanged immutable tracked source, and 1,681,170,901B across
21 consolidated assets/receipts, plus four small CPU JVP
artifacts. Large binary and full gradients remain remote; compact JSON, identity
sidecar, memory CSV and CPU JVP arrays are mirrored locally.

Primary evidence in outputs/diagnostics/shared-latent-reader-20261004:
leaf-full-state-cuda-20261005-full-state.json;
leaf-full-state-cuda-20261005.acceptance.json;
leaf-full-neural-cpu-jvp-20261005.json and four .jvp.pkl artifacts;
leaf-full-state-cuda-20261005-saved-original-control.json;
leaf-full-state-cuda-20261005.gpu-budget.json;
leaf-full-state-cuda-20261005.settlement.json;
leaf-full-state-cuda-20261005-full-state.executable.pkl.identity.json;
leaf-full-state-cuda-20261005-full-state.json.memory.csv.

## Memory, preservation and settled budget

Observed parent peak14,029,131,776B, worker
peak3,775,201,536B; both below 24GiB. Pool
peaks19,325,255,680B/5,366,611,968B. All four .pair-002 through005-gradients.pkl
files retain old/new full mixed and component parameter/state arrays; independent
.old-diagnostic.pkl/.new-diagnostic.pkl retain full step5 parameters/AdamW.
The 16 MiB state artifact is mirrored locally. Large gradients/chains remain
remote in runtime/outputs/diagnostics. No new parameter/AdamW strict failures
required additional failed-update snapshots. Actual preservation:
9,895,762,755B across 11 consolidated files,
inside the approved approximately 17GB/<20-file envelope. No data deleted.

The unique persistent ledger contains16 settled sessions. Additional source
original_schedule_additional_1800_20261005 recorded one approved 25200→27000
increment and charged **897.671305 GPU-seconds**
(0.249353 card-hours), from
448.835653s×2. Its remaining 902.328695 is
not an automatic new window. Total 22570.320281/27000,
global remaining 4429.679719. Original 18,000 remains
17444.261045 used/555.738955 remaining and its fixed route 199.181077 remaining;
daily-only 7,200 remains4228.387930 used/2971.612070 remaining. No balances
transferred or historical charges reset. Worst-case1800, actual897.671305,
paid cost0 CNY.

Normal completion saved both chains, completed resource close at
444.017111s, and exited1 for retained strict failure.
No deadline/resource/runtime exception occurred. Session UUID
90f65afc-dfdc-4e72-9674-6090b5f3bcef settled within its 900-second cap. Wrapper
PID is gone. Initial release13:52:14 showed 0MiB/no processes (GPU0 transient 1%);
readonly13:52:53 confirmed both0MiB/0%. Final settlement receipt at
2026-10-05T14:03:15.003694+08:00 also confirms 0MiB/no compute processes and completed assets.
No subsequent GPU work belongs to this historical funding scope.

Primary evidence in outputs/diagnostics/shared-latent-reader-20261004:
shared-original-full-state-schedule-resume-20261005.json/.json.log/
.json.launch.txt/.json.release.txt/.state-diagnostics.pkl;
original-full-state-schedule-followup-summary.json;
original-full-state-schedule-followup-settlement.remote.json;
original-full-state-schedule-followup-state-coordinate-analysis.json;
original-full-state-schedule-followup-input-biomass.json;
shared-latent-reader-20261004.gpu-budget.json. Preparation hashes and prior CPU
asset admission remain valid; no preparation/preflight was repeated for this run.
