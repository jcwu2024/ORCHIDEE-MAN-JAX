# Normalized attention-key reuse: real-input and real-loss gates

Status: real daily micro4 gate passed with1.0720x warm native-AD ratio. Whole
normalized T7 failed strict initial-state derivative parity despite parameter,
loss, physical and memory passes. Saved-binary repeats establish persistent
cross-arm discrepancies beyond observed within-arm drift; T30/full updates are
not admitted. A bounded original-head-order check also failed to resolve the
strict component discrepancy and has no native-VJP benefit; no T7 retry followed.
No model default or formal training has changed.

## Treatment and actual component schedule

The opt-in adapter moves per-head stable FP32 L2 normalization to projected
family/process/carbon and distinct field axis/local banks, before coordinate
gather. The inactive dynamic coordinate uses normalized trainable key bias;
values, head-query normalization, concatenated segmented softmax, masks and
all-masked behavior remain source formulas. AD is native JAX. Source owners:
`shared_transformer.py` lines376-448,451-534,537-607;
`strong_baseline_architecture.py` lines901-910. No Fortran process is modified.

Capture used immutable c599236 update0, accepted prepared context, artifact742,
and the original four distinct rows. It stopped at decoder inputs, retaining
raw banks so screening recomputes K/V and differentiates their parameters.
Seven expert groups have14/28/722/146/24/7542/72 rows; GPP has2 actual gathered
rows and its carbon bank. Each group keeps chunk=min(128,rows), original padding,
two384-wide decoder blocks/eight heads and native `lax.map`. Micro1 uses actual
row0; micro4 vmaps the four independent captured rows. No whole-field giant
GEMM replaces the chunk loop. Cotangent is declared synthetic sin(0.11*i) over
all outputs, without scaling. Gates are unchanged rtol1e-5/atol1e-6.

## Component results

All output and captured-input VJPs pass for normalized micro1/micro4. All values
and all3,548,928 decoder-block parameter VJPs are finite, but strict parameter
parity fails at the following coordinates. These component results are not
public model-loss or physical-state gradient admission.

| Route | Micro1 parameter failures | Micro1 native-VJP ratio | Micro4 parameter failures | Micro4 native-VJP ratio |
| --- | ---: | ---: | ---: | ---: |
| expert0 |1165|0.9542x|0|0.9686x|
| expert1 |0|0.9824x|0|0.9680x|
| expert2 |16|1.0958x|315|1.0677x|
| expert3 |3|1.0507x|11|1.0728x|
| expert4 |3031|0.9676x|0|0.9837x|
| expert5 |2065|1.1370x|10048|1.1782x|
| expert6 |0|0.9792x|0|0.9762x|
| GPP |0|0.9780x|0|0.9958x|

Expert5 micro1 forward medians24.035/20.862ms and native forward+VJP
99.021/87.093ms; micro4 forward39.970/34.943ms and native VJP
197.687/167.792ms. Estimated live executable memory decreases2155.33/1889.52MiB
for micro1 and11396.41/7414.00MiB for micro4. The two-arm process cumulative
active peaks are2.284/11.465GiB. Other routes can use slightly more memory or
run slower. Timing is seven alternating warm pairs with all results synchronized;
these ratios do not establish full-update speed. Every reference/candidate
result, native-VJP executable and optimized HLO is saved before timing.

## Independent discrepancy evidence

Five saved-binary repeats on expert0 and expert5 used no new model compilation.
Expert0 micro1 both arms are individually stable across repeats and processes,
but the normalized arm differs by1165 parameter coordinates on every comparison.
Tiny forward differences (max8.34e-7) propagate into FFN and attention gradients;
this is a deterministic compiled-graph difference, not baseline random noise.
Expert0 micro4 both arms strictly pass and are stable.

Expert5's *original arm itself* fails strict parameter replay by1359-1680
coordinates against its own first micro1 result; micro4 has7183-7922 failures.
The normalized arm likewise drifts. These failures are in gathered-bank
key/value projection kernels/biases. Both optimized HLOs contain native
transpose gather `scatter-add` operations with duplicated indices. Thus native
bank-gradient aggregation is non-deterministic at this synthetic sensitivity;
the cross-arm failures cannot all be assigned to normalization placement.
Exact failed-leaf paths, values, errors and allowed errors are in the leaf/replay
JSONs. No tolerance, cotangent magnitude, architecture or precision is changed.

## Original-formula chunk unroll screen (separate candidate)

`probe_shared_decoder_chunk_unroll.py` tested explicit native scan unroll2
against original `lax.map`, preserving128-row chunks/padding and both blocks.
It reused the captured raw banks and runtime preflight. Only expert2/expert5
micro1 was run. Native-VJP ratios are1.1285x/1.2177x, but strict output,
parameter and input gates all fail: output17/201, parameter136745/896271,
input149/2064 coordinates. Expert5 parameter max error1.5926e-4 exceeds the
observed same-arm scatter drift; forward/input failures are additional evidence.
Do not expand this candidate to micro4 or the model. Its artifacts retain a
separate `shared-decoder-chunk-unroll2` stem and do not enter the normalized
treatment.

## Real daily gate and assets

The real daily diagnostic binds public `make_shared_joint_daily_objective` and
`make_microbatch_objective`, original FP32 field expert128 and source-reuse
backbone, plus only normalized-key patches. The legacy staged decoder has no
`dynamic_projection_execution` argument and is source-verified field-only;
there is no union branch. Exact decoder/backbone/adapter sources and SHA256s
are in the provenance sidecar. Admission verifies original four rows,
update0/method/source identities,300 parameter leaves/64,884,481 FP32 parameters,
saved baseline executable signature, runtime JAX0.4.38/GPU, physical/finite
results and unchanged strict gates. Baseline replay compiles zero model graphs.
Before candidate compilation it exactly matches saved loss/160 auxiliary
coordinates and passes all parameter gradients (max2.98e-8).

The actual daily candidate passed all64,884,481 parameter gradients/300 leaves,
public loss and160 auxiliary coordinates at rtol1e-5/atol1e-6. Parameter max
absolute error is1.3411e-7; loss max error2.8658e-8; no coordinates fail.
All gradients are finite,64,862,209 coordinates are nonzero, gradient norm is
10.3940, and all11 physical hard counts are zero. This actual public-loss
admission does not erase the component synthetic-cotangent failures above.

Seven alternating warm pairs give field median0.374224536s and normalized
median0.349093990s, a1.071987907x ratio. Estimated live executable memory is
11,344,311,092 bytes for field and9,227,919,372 bytes for normalized. The
two-binary process cumulative active peak is11,362,754,816 bytes, below24GiB;
it is not an independent per-arm measured peak. Lowering took122.934565s,
backend compilation1007.899032s and complete Python elapsed1170.492390s.
The original240/1200/1800s bounded process completed with exit0; no daily
retry was needed. Candidate executable/result were saved before comparison.

T7 candidate was launched2026-10-01T02:33:41+0800 on idle free GPU0 using
unchanged default native checkpoint and compiler settings. It replays the
durable default dual-gradient T7 reference before compiling and compares all
parameters plus3854 initial-state gradient coordinates,14 physical hard counts
and the24GiB memory gate. Its budgets are360s lowering,5400s backend and7200s
process, informed by the prior2545s T7 default backend compile. The reference
replay compiled zero model graphs:20 loss/auxiliary coordinates are exact and
all parameter gradients pass(max2.4680e-8). One of3854 initial-state coordinates
fails against its cross-process saved result; that existing baseline drift is
recorded separately. A free-test01 host diagnosis(no model compilation) locates
it at flat index33:0.3685392439365387 versus0.368509441614151, error2.980232e-5
versus allowed4.685094e-6. The aggregate max absolute drift24.6328 belongs to
a much larger state gradient that passes its relative tolerance; it is not the
failed-coordinate error. Prepared-contract mapping identifies the failure as
`diffuco_previous_step_state.snowgrain`, local coordinate(0,0) of shape(1,3),
source_ref=`explicit_snow`. Mapping reuses current contract metadata and introduces
no physical-process or AD change. Candidate comparison uses the same-process replay and
unchanged strict tolerance. Candidate lowering completed170.9972s; backend
compilation3363.9938s and Python elapsed3682.9457s. Executable/result were saved
before comparison. The launcher exited1 for numerical parity, not a resource
failure or timeout.

Actual T7 loss/auxiliary20 coordinates and all64,884,481 parameter gradients
strictly pass(parameter max9.3132e-8), but7/3854 initial-state coordinates fail.
All initial-state gradients are finite and3833 are nonzero; all14 physical hard
counts are zero. Candidate live estimate2,947,419,389 bytes versus default
3,488,289,277 bytes; two-binary cumulative active peak3,507,481,856 bytes passes
24GiB. Seven matched warm medians are1.575990463s default/1.516120825s normalized,
ratio1.0394887x. The first default timed call1.815s is slower than its remaining
six1.572-1.578s calls; normalization is1.512-1.524s. Warm timing alone does not
admit this candidate.

Exact first-result failures are state indices33/34(snowgrain),199(canopy2ground),
385(nroot),416/422/428(profil_froz_hydro_ns). The largest failed-coordinate error
is2.299845e-4 at33, allowed4.685392e-6; the aggregate max47.00595 includes much
larger gradients that pass relative tolerance. Both main-process arms have1-2
within-arm state failures in each of seven repeats, retained independently.

The subsequent saved-binary diagnostic completed227.519s/exit0 with zero model
compilation and every full3854 state vector saved. Seven paired failures are
[5,6,5,5,5,5,4]; default self failures[0,1,1,1,1,1,2], normalized
[0,1,0,1,1,1,1]. Indices33/199/385/428 fail in every pair. At199/385/428 both
arms independently pass all self comparisons; their observed value ranges do
not overlap. Range gaps are1.84484e-6/2.55704e-5/3.31080e-5, versus default
ranges1.74586e-7/2.62260e-6/1.96397e-6 and normalized
2.97435e-7/4.58956e-6/1.55930e-6. At33 the two noisy ranges are still separated
by1.46598e-4. Thus the whole normalized strategy's strict state failure cannot
be assigned only to baseline noise. These are bounded observed ranges, not
universal error bounds. Index34 fails six pairs; snowdz index182 one pair.

The persistent failure semantics are:

| Initial-state index | Contract state leaf | Leaf coordinate | Strict self stability in both arms | Gap between observed arm ranges |
| ---: | --- | --- | --- | ---: |
| 33 | diffuco_previous_step_state.snowgrain | (0,0) | fails | 1.46598e-4 |
| 199 | hydrol_previous_step_state.canopy2ground | (0,0) | passes | 1.84484e-6 |
| 385 | hydrol_previous_step_state.nroot | (0,0,10) | passes | 2.55704e-5 |
| 428 | hydrol_previous_step_state.profil_froz_hydro_ns | (0,4,3) | passes | 3.31080e-5 |

Prepared metadata identifies endpoint outputs with these names as hydrology_snow
expert2. Output ownership alone cannot attribute the initial-state derivative
change to that expert: all decoders consume shared input banks and recurrent
physical feedback. No expert-subset T7 rerun is justified by this mapping alone.

An independent synthetic LLVM module compilation parallelism runtime check
failed in4.924s because no available compilation provider supports the flag.
This deployment has NvJitLink/NvPtxCompiler support disabled at build time and
no matching nvlink found in the bounded software-path inspection. A separate
owner then minimally extracted official matching nvlink12.6.85 into isolated
`runtime/toolchains/cuda-nvlink-12.6.85` and passed the synthetic compatibility
check in25.375s/exit0 using an independent PATH, LLVM parallelism and forced
parallelism4. Canonical environments remain unchanged. T7 used default settings;
the synthetic compatibility pass does not establish real-model compile speed or
normalized-key numerical correctness.

## Bounded original head-order hypothesis

Original attention reshapes keys and swaps the head/token axes before stable L2
normalization. The normalized-bank adapter had normalized before this swap.
`probe_shared_normalized_key_head_order.py` tests the original head-first order,
then applies the inverse swap to restore bank shape. Five actual captured bank
shapes(20/1/12/13/14 by384) include native VJPs for the real key projection
parameters and raw inputs. Their values and gradients are bitwise identical
between the two bank-normalization orders, although optimized HLO text after
metadata/identifier normalization differs.

Only expert0/expert4 micro1 were consequently screened with original128 chunk
semantics/two blocks/all3,548,928 decoder parameters and the unchanged synthetic
cotangent. Both output/input VJP gates pass, but parameter parity still fails
1209/2014 coordinates. Native-VJP ratios are0.9446x/0.9815x, so this control
provides no useful repair or speed evidence for another whole T7 compile. It
completed54.069s/exit0 under a300s cap, with original compiler settings for both
arms. The bank HLOs, complete component result/executable and timings are saved
under a separate `shared-normalized-key-head-order-micro1-20261001` stem.

Current route decision: retain the admitted daily diagnostic and all evidence;
reject whole normalized recurrent/full-update adoption. Do not launch T30 or a
guessed expert-subset T7 retry. A future repair needs independent component or
source evidence that targets the actual derivative discrepancy. No tolerance,
architecture, precision, data mix or checkpoint policy is changed.

## Follow-up execution

`probe_shared_normalized_keys_recurrent_replay.py` executed the saved dual-gradient
diagnostic above, without lowering or compilation. Raw vectors, exact failed
values and contract semantics are retained; `analyze_shared_normalized_key_recurrent_replay.py`
recomputes all paired/self failures and observed ranges with host NumPy only.

`probe_shared_normalized_keys_full_update.py` prepares a matched comparison using
the public `execute_learning_update`, whole-tree daily slicing and unchanged
AdamW. Both arms use dual AD for T7/T30, synchronize the complete dual result and
then extract parameter gradients. Three pairs from two independent changing-weight
chains preserve32 daily rows(eight micro4), oneT7 and oneT30, every
physical/finite gate, full combined parameter gradients, optimizer/parameter/loss
parity, and additional initial-state derivative parity. Timed execution rejects
all new JAX compilation after helper warm-up. Failed full proposals are saved
for diagnosis; no training checkpoint is retained. Its paired baseline includes
extra state AD and is not directly the previous11.57s parameter-only reference.
Execution requires completed recurrent admission; unresolved self drift must be
diagnosed from the saved binaries before deciding the next bounded step.

The actual shared runner/slicer support is
`runtime/transfer/shared-update-changing-slice-20260930`(three existing modules).
`shared-update-tree-slice-20260929` also exists but is not this launcher's support
selection. `run_shared_normalized_keys_followup.sh` has replay420s/full1080s
budgets, free-node GPU occupancy guards and exit sidecars. Python syntax, actual
daily admission-reader verification, target Bash syntax and actual full/replay
dry-run argument expansion pass. Replay executed; full-update integration has
not run because recurrent admission failed.

Local evidence: `outputs/diagnostics/shared-normalized-keys-20261001/` holds
runtime, capture, micro1/micro4 result JSONs/logs, exact failed-leaf diagnostics,
five-repeat replay JSONs/logs, unroll2 result/log, provenance sidecar and the
actual daily/T7 result/log/exit, raw T7 saved-state replay, failed state semantics,
repeat ranges and current prepared expert/field mapping.
The detailed native assets stay under
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/` with
`shared-normalized-keys-*20261001*` stems. Reproducer overlay:
`runtime/transfer/shared-normalized-keys-20261001`; admitted decoder/backbone:
`runtime/transfer/shared-combined-recurrent-20260926`; original daily loader:
`runtime/transfer/shared-combined-real-daily-20260925`.

Five local ORCJAX tests pass (bias fallback, all-masked behavior, all local
parameters/input native VJP, normalized/unroll formulas, patch restoration).
GPU runtime preflight passed. Target shell syntax and actual dry-run argument
expansion passed. T7 main gate exited1 for the strict state discrepancy; its
saved-binary replay exited0. The first independent LLVM compatibility probe
exited1 and its isolated-nvlink follow-up exited0. There is no paid submission or
formal training. Free GPU0 is Tesla V100-SXM2-32GB; this owner uses one card
at a time. Completed preflight/capture/component/replay/unroll/daily processes
plus T7 launcher3698s and saved replay235s total approximately5918.547s, or
1.6440 GPU card-hours including startup/compile before the head-order control.
Including its57s launcher process gives5975.547s or approximately1.660 GPU
card-hours for this owner's whole normalized route. This excludes the parent's
independent4.924s/25.375s GPU1 compiler compatibility probes. Warm execution
times alone are not the resource accounting; no paid jobs were submitted.
