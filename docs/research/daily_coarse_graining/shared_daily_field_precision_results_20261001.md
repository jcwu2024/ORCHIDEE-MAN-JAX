# Original-field four-sample neural precision screen, 2026-10-01

Status: **engineering screening failed** after the broad selected-neural
policy and the authorized native-AD conditioning diagnostics. The original
unscaled repaired screen failed14/16 parameter and663/1028 state-field groups.
S256 and the single same-binary S4096 replay passed all16 parameter groups
but each failed76 state-field groups in initial + all seven paired comparisons.
Finite/physical/mask/actual-undefined-state-zero gates passed. Further scaling
is stopped; no complete-update candidate, training, adoption or paid job ran.
The narrower FFN-only screen was approved and completed on2026-10-02. All16
parameter groups pass, but47/1028 state fields fail in the initial and all
seven comparisons; its1.058350x daily component ratio admits no full update.

## Declared candidate and comparison

The original field128 decoder plus shared-source backbone execution is retained;
there is no union projection, vmap or G2 grouping change. Native
`jax.value_and_grad(..., argnums=(0,1), has_aux=True)` returns all64,884,481
parameter gradients and all15,416 initial-state gradients. The dynamic ABI is
`(parameters, states[4,3854], daily_samples)`, retaining the original891 leaves.
The independently saved admitted original F4S binary is loaded and freshly
replayed in the same process, with zero baseline model compilations. The public
objective, four samples, losses, counts and sampling mix are unchanged.

In the earlier broad policy, only `_linear`/`_ffn` aliases in `shared_transformer`,
`shared_backbone_execution`, `shared_backbone_projection_reuse` and
`shared_decoder_macro_packed` receive the existing native
`DotAlgorithmPreset.F16_F16_F32` learned-neural dot policy. Parameters remain
FP32 in storage, and linear outputs/bias and dot accumulation remain FP32.
The macro module's captured readout `_linear` aliases for readout_first and
readout_second are covered. Strong-baseline tokenizer helpers, carbon encoders
using original helpers, attention score and value-aggregation einsums retain
their original precision; physical modules retain FP64. This is not a claim
that every kind of neural dot is FP16 or that hardware TensorCore utilization
has been measured. The candidate combines this precision policy with the
required compiler repair below, so a measured difference will not establish
precision-only causality. Compiled dtype evidence was recorded without recompiling
the model or exporting full source-metadata HLO.

Initial and seven alternating paired original/candidate comparisons report the
unchanged strict rtol1e-5/atol1e-6 gate separately from the previously declared
engineering gate: loss relative error≤1e-3; each of16 semantic parameter groups
and each of257 continuous fields separately for each of4 samples has
relativeL2≤1e-2 and cosine≥.9999. Reference norms≤1e-12 use maximum absolute
error≤1e-6 instead. A zero candidate norm against an active reference fails
cosine. There is no sample stacking that can cancel opposite errors.

Finite loss/auxiliary/gradients, all11 public physical hard counts zero, actual
per-sample undefined input-state gradients zero, unchanged masks and estimated
and observed24GiB bounds remain hard gates. Engineering passage is diagnostic
and does not authorize production adoption or training.

## Mask evidence and limits

The source audit verifies the PFT active mask is copied from assembled inputs
by the unchanged tokenizer and source-reuse backbone. `_field_active_and_pft_owner`
uses layout, PFT booleans and token shape, with no neural helper or token-value
arithmetic. The field adapter restores these full masks, including the GPP
source path; the macro decoder sets inactive readout to zero. Every sample's
source-derived raw mask coordinates, active counts and hashes are recorded.
Source raw-mask precision invariance is audited; raw neural numeric outputs
are not separately returned by this unchanged public-loss ABI. That limitation
is explicitly recorded as `raw_mask_numeric_outputs_observed=false`.

Each real sample is passed through `head.input_assembler.assemble`, producing
its own actual3854-coordinate boolean defined mask. Candidate and original
state gradients must be exactly zero at its undefined coordinates. Arbitrary
reference zero gradients are descriptive; they do not establish source-defined
structural inactivity. A full structural-inactive parameter/output gradient
audit remains incomplete. Graph transport was closed throughout; evidence is
bounded exact-source fallback and no graph-completeness claim is made.

## Execution and artifacts

Source runner: `scripts/hpc/probe_shared_daily_field_precision.py`;
metrics: `scripts/hpc/shared_daily_precision_metrics.py`;
launcher: `scripts/hpc/run_shared_daily_field_precision.sh`.
The executed snapshot is isolated in
`runtime/transfer/shared-daily-field-precision-20261001` and copied locally
under `outputs/diagnostics/shared-daily-field-precision-20261001/executed-snapshot`.
Existing overlays and production defaults were not edited.

Lower/backend/process budgets are240/1500/4200s, based on the prior real daily
lower108s/backend879s measurement. Original-stem target preflight and candidate
`--dry-run` expansions passed before launch. Saved dry-run text uses only a
separate output stem because the launcher correctly rejects rerender over
executed output records. Local metric checks cover independent sample slices,
16-group full tree partition, zero/near-zero gradient behavior and actual mask
shape/dtype/undefined-coordinate handling.

Actual runtime preflight passed GPU0 on ibc13b02n04, JAX/JAXlib0.4.38,
V100-SXM2-32GB, canonical `orcjax_gpu` environment in foundationpose. Checker
22.9653815s. Candidate start10:00:29 CST: load1.11/1.06/.95, affinity0–111,
GPU0 empty. Parallel GPU1 original T30 reference PID33296 used1512MiB and is
separately accounted; it does not count as a candidate speedup.

## Compiler rejection, evidence and compatibility repair

The original model compile and a rank-one matrixization retry both failed with
`ALG_DOT_F16_F16_F32`, before any candidate model values existed. All 86 actual
primal helper shapes passed separate synthetic native-VJP tests; those tests
did not reproduce the full scalar-loss graph. Full lowering showed 5,201
algorithm-tagged dots with F16 operands and F32 results. The GPU-pass capture
then showed 12 scalar-readout AD contractions promoted to F32 operands while
retaining the F16 algorithm tag. This matches the pinned XLA OperandUpcaster
source behavior and the later algorithm checker. Source pin evidence is not
independent proof of installed-binary build provenance.

Six exact readout shapes `[4,N,256]` for N=128,14,28,72,24,2 with kernel
`[256,1]`, a nonlinear masked FP64 scalar loss, and native parameter+FP64-state
AD reproduced the rejection for every shape. Supplying a repeated DebugOptions
field through JAX `compiler_options` failed at the protobuf API, before testing
the repair. A fresh-process flag with `operand-upcaster` also did not repair the
error: the pinned header defines the pass name as `operand_upcaster`.

The exact fresh-process flag
`--xla_disable_hlo_passes=operand_upcaster`, together with the unchanged
`--xla_gpu_enable_llvm_module_compilation_parallelism=false`, supported all six
reproducers. All results were finite; all 257 readout parameter coordinates
and meaningful state coordinates had nonzero gradients. Compiled cuBLAS calls
show F16 operands and F32 outputs for the forward and both derivative paths.
The repair uses the unchanged native F16_F16_F32 preset, with no FP32 readout
exception and no custom AD. The flag disables this compiler pass globally; the
full repaired candidate is therefore measured as a precision+compatibility
combination against the unchanged saved original binary.

The current immutable full snapshot is
`runtime/transfer/shared-daily-field-precision-upcaster-20261001`;
the local source copy is `executed-snapshot-upcaster`. It reuses the passed
original preflight, F4S binary, prepared inputs and source checkout. Target
`bash -n` and actual `--dry-run` expansion passed. Contract metadata excludes
the inherited arbitrary-reference-zero inactivity gate; actual undefined
state and source/public masks remain hard gates, and the structural audit gap
remains explicit. Future launchers register exit-record traps only after the
unique-output guard; old executed snapshots/records were retained.

## Actual numerical disposition

The repaired run exited1 with `screening_failed`, after completing all seven
paired replays. The engineering loss-relative error was0.0003371588443,
below0.001, at initial and every repeat. All values were finite. Both arms
had exactly zero gradients at all actual undefined input-state coordinates,
for all four samples. All11 physical hard counts were zero and the source
mask invariance audit passed throughout. Candidate returned300 full parameter
leaves/64,884,481 coordinates with64,855,680 nonzeros, and15,416 state
coordinates with15,334 nonzeros.

All14 failed parameter groups persisted on all seven repeats. The two passing
groups were `readout_first` (relativeL2 .00502591, cosine .999987373) and
`readout_second` (.000755357, .999999878). Largest parameter-group error was
`carbon`: relativeL2 .13638054, cosine .990738971. Backbone blocks failed
relativeL2 at .013955406 although their cosine .999907527 passed; the
tokenizer also failed (.021467654, .999779558) despite retaining its original
linear helpers. This demonstrates changed downstream/shared gradients, not
that the tokenizer itself was directly assigned the F16 policy.

| Sample | Failed state fields, of257 | DOC relativeL2 | DOC cosine |
| --- | ---: | ---: | ---: |
| 0 | 174 | .471220236 | .972683011 |
| 1 | 163 | .320418458 | .988085734 |
| 2 | 158 | .198513531 | .997068688 |
| 3 | 168 | 1.931120118 | .705613258 |

Each sample's field-failure count was identical at initial and all seven
repeats. The largest absolute state-gradient discrepancy came from DOC;
its reference norms were168,135–799,987, so this is not solely a small-norm
cosine issue. Across all fields the largest relativeL2 was11.1147133 for
sample2 `frac_nobio`, with reference norm4.9024e-6 and sign-reversed cosine;
the smallest measured cosine was-1. No near-zero criterion was altered.

The unchanged strict `rtol=1e-5, atol=1e-6` comparison also failed:

| Output | Initial failed coordinates | Seven paired failures | Initial maximum absolute error | Initial relativeL2 |
| --- | ---: | ---: | ---: | ---: |
| Loss | 1/1 | 1 each | .000168894470 | .000337158844 |
| Auxiliary | 44/160 | 44 each | .000876599860 | .000051044681 |
| Full parameters | 46,488,948/64,884,481 | 46,488,943–46,488,995 | .00421310705 | .0163494695 |
| Full input states | 5,746/15,416 | 5,746–5,747 | 313,722.810129 | .414038897 |

Original state self-repeat strict failures were[6,5,5,7,5,5,7]; candidate
self-repeat failures were[162,144,172,201,172,101,32]. The saved-original vs
fresh-original state drift had7 strict failures, global relativeL2
8.81403e-7. These self diagnostics do not become an added all-self-pass gate
or waive the measured cross-arm engineering failures. The complete fresh
original/candidate pairs, group metrics and raw states are retained.

## Timing, memory and compiled precision

| Measure | Original field | Repaired F16 candidate |
| --- | ---: | ---: |
| Seven-pair median full micro4 loss+parameter+state AD, s | .384208759 | .336207662 |
| Estimated live bytes | 11,348,361,292 | 11,633,413,900 |

The paired evaluator ratio was1.142772x, or12.493494% less wall time. This is
four-sample evaluator evidence from a numerically rejected combination. It
does not establish complete daily32/T7/T30/AdamW throughput or a usable speed
improvement. Both arms synchronize full AD; host transfer, metrics and raw
failure serialization occur outside the timed executable calls.

Candidate lower101.893961s; backend825.560600s; first valid model result and
initial comparison around985.6s after process start; internal completion
1019.722089s. Launcher1027.993086s, one assigned GPU,
0.285553634895 card-hours. Observed process active-buffer peak was
11,911,560,448bytes (11.09GiB), below24GiB; pool peak was
18,251,513,856bytes (17.00GiB). Active-buffer peak is cumulative for the
original/candidate paired process, not an isolated candidate peak. The run
started11:15:32 CST on ibc13b02n04, affinity0–111, load .26/.12/.17;
both GPUs were empty. No paid job was submitted.

Saved-binary inspection completed without model compilation/evaluation. Of718
compiled cuBLAS calls,532 have F16 matrix operands and F32 output (399 with
two operands,133 additionally accumulating an F32 operand);186 retain F32
matrix operands. Examples include backend `ALG_DOT_F16_F16_F32`. Full shapes,
native preset, output storage and algorithm metadata are available in the
compact evidence JSON. This proves actual compiled dtype/algorithm use,
not device-level TensorCore utilization. The82.19MiB compact HLO representation
stayed in memory with metadata and large constants omitted; no full HLO
artifact was exported.

The first inspection guard caught module-import table initialization and
failed before loading the model. A separate immutable v2 inspector recorded23
small CPU import/table helper compilations, then forbade every compilation
during serialized model loading and inspection. Its launcher was28.859309s
(.008016474843 card-hours); the failed inspection2.581253s
(.000717014803). These are included in accounting, separately from model
compile and numerical rejection.

## Accounting and artifact retention

| Diagnostic | Launcher seconds | Assigned-GPU card-hours | Disposition |
| --- | ---: | ---: | --- |
| Original runtime preflight | 24.305870 | .006751630505 | Passed; reused |
| Original full compile | 258.870458 | .071908460657 | Backend algorithm rejection |
| Tiny FFN policies | 5.541782 | .001539383796 | Capability measured |
| Rank-one/linear shapes | 6.255948 | .001737763219 | Mixed capability |
| Matrixized rank-one shapes | 6.415626 | .001782118413 | Capability passed |
| Full rank-one matrixization retry | 258.559632 | .071822119885 | Same backend rejection; actual graph had no rank-one helpers |
| All86 primal helper shapes | 30.515916 | .008476643364 | Native synthetic VJP passed |
| Full lowering only | 157.275019 | .043687505192 | Dtype evidence; no backend compile |
| Targeted GPU-pass capture | 275.946878 | .076651910477 | Same rejection; bad AD dots captured |
| Six scalar readout AD/plumbing test | 7.172289 | .001992302405 | Algorithm and repeated-option failures recorded |
| Unmatched hyphen pass flag | 5.817095 | .001615859800 | Algorithm failures persisted |
| Exact underscore pass flag | 10.009681 | .002780467073 | All six scalar AD cases supported |
| Repaired full four-sample screen | 1027.993086 | .285553634895 | All comparisons measured; engineering rejected |
| Failed saved-binary inspection | 2.581253 | .000717014803 | Import helper blocked |
| Repaired saved-binary inspection | 28.859309 | .008016474843 | Compiled dtype evidence saved |
| Conditioning runtime preflight | 38.850127 | .010791702006 | Passed; reused for same-binary S4096 |
| Six scaled readout AD shapes | 26.177713 | .007271586921 | Native same-binary S1/S256 evidence |
| S256 full screen | 1088.481975 | .302356104122 | All16 parameter groups pass;76 state fields fail |
| Scaled saved-binary inspection | 45.650045 | .012680568033 | Dynamic scale dataflow evidence; no model compilation/evaluation |
| S4096 saved-binary replay | 105.862653 | .029406292571 | All16 parameter groups pass;76 state fields fail; no new compilation |
| **Assigned-GPU total** | **3411.142355** | **.947539542980** | **20 invocations; one GPU each; paid0** |

CPU-only proto inspection took19.349539s internally with zero GPU card-hours;
launcher wall was not recorded. The accounting total excludes the separately
closed input-only family and separately owned original T30 reference. Assigned
card-hours include host/lowering/backend/diagnostic time, not measured device
utilization. Exact records are in `launcher-accounting.json`.

Evidence is under
`outputs/diagnostics/shared-daily-field-precision-20261001/`:

- `shared-daily-field-precision-upcaster-20261001.json`, `.log`, launcher record,
  `screen-summary.json`, `launcher-accounting.json`, `artifact-manifest.json`;
- local candidate executable119,498,244bytes, candidate initial result
  259,764,048bytes, complete fresh-original replay and all14 state-repeat
  arrays in `.state-replays.pkl`;
- seven `.failed-pair-0.pkl` through `.failed-pair-6.pkl`, each containing
  both complete loss/auxiliary/parameter/state results and comparisons. These
  519,599,368byte paired gradient artifacts were preserved remotely and
  **all downloaded and size-verified locally**; exact paths and copy status
  are recorded in the manifest;
- `...compiled-dtypes-v2.json`/log/launcher record, prior scalar/readout and
  lowering/GPU-pass JSONs, bad-dot dependencies and pinned XLA source note;
- source snapshots for original, rank-one retry and final repaired execution.
  Immutable remote overlays and executed records remain intact.

The prepared `daily-only-precision-diagnostic` full-update interface and target
shell expansion were produced while compilation ran. It preserves strict
updated-parameter/AdamW and original recurrent-state gates, independently
checks eight dynamic micro4 batches with the declared engineering gradient
and mask gates, and explicitly labels diagnostic results as strict-not-admitted.
It was **not executed**, because the prerequisite four-sample engineering gate
failed. No complete-update speed or optimizer-parity claim follows from it.

## Error localization and authorized equivalent-AD conditioning

Independent raw state-tree energy exactly matches the sum over all1028
semantic groups. Four DOC fields contribute **99.9992755%** of squared
state-gradient discrepancy. No state coordinate became newly zero. The
parameter tree has6,529 newly zero coordinates, but those explain only
3.2938e-10 of squared parameter error; their reference magnitudes span
8.96e-31–3.083e-6. Reference parameter gradients span8.96e-31–.627728,
with16,184,492 nonzero coordinates below2^-14. These final-output ranges
and newly-zero counts do not establish intermediate F16 underflow.
Full group norms/cosines/zero changes are in `gradient-distribution.json`.

The authorized next repair retains the exact architecture, public losses,
loss weights, masks, preset and gates: native AD differentiates `S*loss`,
where dynamic FP64 scalar S=256, then divides every parameter/state gradient
by S cast to that leaf's original dtype. The same timed executable returns
the original public loss/auxiliary and all unscaled gradients. It uses no
custom AD, clipping or gradient truncation. Adding an actual scalar argument
changes the candidate ABI891→892; the saved original reference remains891.

A new actual GPU runtime preflight passed (38.850127s launcher,
.010791702006 card-hours). The six actual readout shapes were tested with
dynamic loss coefficients .01 and1e-8, with originalFP32/F16, S=1/256,
using each same compiled program across S values. Every result was finite,
FP32 unscale was correct, original public values were exact across F16 S
values, and state gradients/S were FP64. The executed launcher enabled x64;
the current source also enables it explicitly. For the small-cotangent
coefficient1e-8, F16 state relativeL2 improved from1 to .005957–.007910,
and parameter relativeL2 from .993862–.999986 to .005211–.006939. The
normal coefficient .01 showed nearly unchanged rounding discrepancies.
Compiled tiny patterns retain an external FP64 S parameter, multiplication
of S into the cotangent before the F16 conversion, and final FP32/FP64
gradient divides. This supports the conditioning diagnostic, without
establishing why DOC fails in the complete public graph.

Small-probe launcher26.177713s, .007271586921 card-hours. The independent
source review and lightweight conditioning tests passed. The actual S256 model
lowered in127.692561s and compiled in851.130895s; internal time1080.189024s.

## Complete conditioning results and limits

| Original-vs-candidate result | S256 | Same-binary S4096 |
| --- | ---: | ---: |
| Loss relative discrepancy | .000311862322 | .000311862322 |
| Failed parameter groups, initial and every pair | 0/16 | 0/16 |
| Failed sample/state fields, initial and every pair | 76/1028 | 76/1028 |
| Initial failed fields by sample |17,21,24,14 |17,16,22,21 |
| Global parameter gradient relativeL2 | .000886544701 | .000884187019 |
| Global state gradient relativeL2 | .001294608168 | .001013028898 |
| Carbon parameter group relativeL2 | .000854292426 | .000852276073 |
| DOC sample0 relativeL2 | .001036953 | .001271176 |
| DOC sample1 relativeL2 | .001103520 | .001145171 |
| DOC sample2 relativeL2 | .000499091 | .000744172 |
| DOC sample3 relativeL2 | .006816708 | .001413808 |
| Original median seconds | .390356908 | .377238916 |
| Candidate median seconds, including g/S | .333752425 | .323290763 |
| Original/candidate median ratio |1.169600212 |1.166871927 |
| Estimated candidate bytes |13,122,214,452 |13,122,214,452 |
| Observed process peak bytes |13,400,360,704 |13,400,360,960 |

All16 parameter groups and all four DOC fields pass the declared engineering
gate at both scales; all initial/seven finite, eleven physical hard counts,
mask and actual undefined-state-zero checks pass. Both fail the complete state
gate. The evaluator wall savings14.500700%/14.300792% are evidence from rejected
daily evaluators, not same-workload full-update throughput. Strict initial
failures are loss1/aux44/parameters8,905,544/states2129 at S256 and
loss1/aux44/parameters8,847,630/states2128 at S4096; full strict repeat and
self-drift diagnostics are retained in each summary.

The S1 scaled-wrapper binary differs from the earlier unscaled candidate:
loss relative2.5288e-5, parameter relativeL2 .000629689 and state relativeL2
.011341189. This material compiler/wrapper drift prevents assigning every
old-to-new improvement to scaling alone. Conditioning evidence uses S1→S256
and S256→S4096 in the **same serialized scaled executable**; original F4S
remains the independent engineering reference. Of the76 S256 residual fields,
65 have reference norm<1e-3 and61 show provable S1→S256 improvement by
triangle bounds. The largest fields have ordinary-sized norms, so the residual
cannot be described as purely small cotangents or proven intermediate underflow.

S4096 changed error distribution without reducing the failure count:59 fields
remain failed,17 S256 failures resolve and17 previously passing fields newly
fail. Across their93-field union,55 delta norms decrease and38 increase.
For example sample0 hydrol.profil_froz_hydro_ns improves .01438146→.01278195
relativeL2 but still fails; sample2 hydrol.snowdz worsens .01397021→.01552440;
sample1 enerbil.snowdz improves .01057877→.00740474 and passes. Original public
loss and all160 auxiliary coordinates are exactly unchanged between S256/S4096,
while state/parameter gradients change. Scaling repairs much of the backward
error but neither tested scale closes every field; residual operation attribution
and forward/backward interaction remain unproven. No further factor sweep or
tolerance change follows from this result.

The compiled scale inspector identifies dynamic FP64 scalar argument676 in the
pruned backend entry (public candidate ABI remains892),573 S-dependent F16
conversions,140 S-dependent FP32 divides and3 FP64 divides through508
computation calls. Evidence follows parsed operand/fusion/tuple dependencies;
it is not exhaustive algebraic or device instruction proof. Actual staged
module file hashes are recorded by the S4096 replay, and the imported
expert_decode_fields globals are the actual staged macro module dictionary.
These are source provenance records; source loading did not recompile the
saved model. Native algorithm compatibility still depends on the global
operand_upcaster disabling flag; comparisons concern a precision/compiler
combination, not precision-only causality.

S4096 reused the exact S256120,491,610byte executable, prepared daily4 inputs,
update0 parameters, original F4S891 binary and the completed runtime preflight.
Its public ABI adds only dynamic scalar S; all per-leaf dtype divisions stay
inside the timed binary. Lower/backend model compilation counts are zero;
internal101.627463s and launcher105.862653s use their own clocks. The actual
node snapshot was GPU0 V100-SXM2-32GB, zero other GPU processes, host load
.01/.17/.45, affinity0-111, OMP4. Node timestamps are not mixed with app clocks.

S256 complete initial/replays/seven failed-pair trees are local and size
verified. S4096 complete initial/original replay/state repeats are local;
all seven complete failed-pair trees remain remote and their actual sizes and
copy status are in artifact-manifest.json. Executed S4096 source snapshots
remain separate from later output-uniqueness guard repairs; these repairs did
not trigger another broad-policy run.

## Approved FFN-only experiment; completed and state-gradient rejected

The independent shared_daily_ffn_precision_policy.py locks the native base
helper SHA256 f2e4233fd3d4c8bbe7a81da75b54d98dc7c2d0dd71bcaf65dde5b8ffeb39f45d.
Only shared_transformer._ffn, shared_backbone_projection_reuse._ffn and the
actual staged shared_decoder_macro_packed._ffn receive F16_F16_F32. Every
_linear alias is verified to be the original FP32 helper, including all readout,
attention and input/bridge/local projections; physics stays FP64. The policy
records each actual module file/hash and macro globals identity, and restores
all original aliases on normal/exception exits. This preserves the architecture,
loss, four samples, parameter/state AD scope, masks and all current gates.

The runner retains native S4096 conditioning, candidate892/reference891
ABI, a distinct FFN-only identity/schema, same-binary S1 diagnostic, initial + seven comparisons, physical/finite/
undefined/mask/count and24GiB checks. Local actual-module alias isolation and
restoration plus native AD dtype/algorithm tracing passed; these checks establish
no GPU numerical accuracy. Target bash syntax and actual --dry-run expansion
passed, using the immutable source checkout and separately staged next overlay.
An FFN S4096-vs-broad S4096 comparison isolates the changed precision scope
at the same scale; FFN S1-vs-FFN S4096 isolates conditioning in its own binary.
FFN S1-vs-broad S4096 changes both policy and scale and is explicitly labelled
diagnostic, with no wrapper-only causality claim. Original F4S remains admission.
The user approved this new FFN-only precision choice on2026-10-02. Its own
actual-environment GPU preflight passed exit0, JAX0.4.38, internal70.997853s,
launcher80.283481s (.022300967044 card-hours). The candidate completed all
initial/seven paired numerical evaluations and returned `screening_failed`.
Exit1 records this failed engineering screen, without a runtime/compiler error.
Source/public masks and candidate892/original891 interfaces passed.

Exact executed commands:

```bash
bash /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/shared-daily-ffn-precision-scale4096-20261001/run_shared_daily_ffn_precision.sh preflight
bash /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/shared-daily-ffn-precision-scale4096-20261001/run_shared_daily_ffn_precision.sh candidate
```

Working directory: runtime/checkouts/shared-split-width-v1 at commit
c599236bcf57b745e9bafed678298010f4c3b58f, under the canonical Explore1000 root.
Resources: free gln01 GPU0, one V10032GB, OMP4/OpenBLAS1, observed/estimated
24GiB gate, no Slurm allocation and no paid job. Budgets: lower240s,
backend1800s, internal process4200s, candidate launcher4210s plus300s preflight,
worst-case cash charge0.
The frozen source/local-check/expanded-command records are retained in the
family directory. A separate bounded full-update launcher is prepared at
runtime/transfer/shared-daily-ffn-precision-full-update-20261002 and passed
actual target-shell expansion. It uses the exact screened FFN S4096 binary,
eight distinct micro4 batches (daily32), and completed original T7/T30 binaries.
It was not run because the component state gate failed. Updated
parameters/AdamW and recurrent state checks would remain strict;
daily/mixed gradient engineering checks remain separate.
Warm/update1 compare the same inputs; updates2/3 compare separate trajectories.
Available arms are preserved on runtime/comparison/gate failure. This remains
a diagnostic, with no training checkpoint or production precision adoption.

### Complete original-reference result

| Metric | FFN-only S4096 |
| --- | ---: |
| Loss relative discrepancy, initial and every pair | .000154567925 |
| Parameter groups failed, initial and every pair | 0/16 |
| Sample/state fields failed, initial and every pair | 47/1028 |
| Failed fields by sample | 10,12,14,11 |
| Global parameter/state gradient relativeL2 | .000498977385 / .000877239479 |
| Maximum parameter-group relativeL2 / minimum cosine | .000940126815 / .999999567758 |
| Maximum state-field relativeL2 / minimum cosine | .243206975 / .999290329475 |
| Original / candidate warm medians, seconds | .381142199 / .360128524 |
| Original/candidate ratio | 1.058350488 |
| Estimated live / observed active peak, bytes | 11,419,851,356 / 11,697,997,568 |

All initial/seven complete results are finite; all eleven physical counts are
zero, actual undefined-state gradients are exactly zero per sample, and the
source/public-mask gate passes. Raw numerical neural mask outputs remain
unobserved by this public ABI; source dependency evidence is explicitly scoped.
All16 parameter groups and all four DOC fields pass engineering gates. DOC
relativeL2 by sample is .001425403, .000116693, .000251720 and .004050690.
This does not waive the47 failed fields or establish trained scientific accuracy.
The unchanged strict initial gate fails loss1/aux40/parameters4,148,043/
states1,940; all seven strict diagnostics are retained separately.

Against the broad S4096 candidate,35 failed fields persist,41 resolve and12
new fields fail. This is a changed error distribution at the same scale, with
global compatibility flags retained, not a precision-only compiler attribution.
All47 fail relativeL2; four also fail cosine. Of47 reference field norms,46
are at most1e-3 and none are at most1e-12. Forty-four have maximum absolute
error at most1e-6; the declared gate still requires relative accuracy for
these reference norms and rejects them. No tolerance was changed.

The largest relative failure is sample3 hydrol_previous_step_state.run2peat:
reference norm2.968740629e-6, delta7.220184285e-7, relativeL2 .243206975.
The one failed norm above1e-3 is sample1 sechiba_finalize_state.q_sol_pot:
reference norm .001259468432, relativeL2 .011349426578, maximum absolute
error1.429424450e-5. The failed set is identical in all seven comparisons;
maximum change from initial relativeL2 is1.725132258e-6 and maximum change
in failed-field delta norm is2.910383046e-11. Within-arm strict state drift
still exists (original2–5, candidate2–7 coordinates), but it does not explain
away this stable failed set. Small norms alone do not prove intermediate
underflow, identify the responsible FFN, or authorize ignoring these gradients.
Operation attribution and forward/backward interaction remain unproven.
Further scale tuning and full-update execution stop at this result.

### Runtime, preservation and throughput limits

Lowering took125.512516s and backend compilation854.697433s. Internal
process1096.399937s and candidate launcher1106.484582s use separate clocks.
Candidate .307356828385 assigned card-hours plus preflight .022300967044
totals .329657795429, one free V100 and cash0. The entire field-precision/
compatibility/conditioning/FFN family now records22 GPU launchers,
4597.910418s /1.277197338409 assigned card-hours; this excludes original
T30 reference construction and the separate gather-hoisting experiment.

The exact executed six-source snapshot is `executed-snapshot-ffn-only/`.
Primary candidate, S1 diagnostic, original replay, executable and fourteen
state repeat arrays are local and size verified. The three complete raw trees
independently retain300 FP32 parameter leaves/64,884,481 coordinates and
FP64 states[4,3854], all finite. Seven complete failed-pair trees remain at
the canonical remote path,519,596,299bytes each; their exact preservation and
local copy status are in `artifact-manifest.json`. The local machine report,
screen summary, `ffn-residuals.json` and `ffn-evidence-validation.json` record
all47 field names, norms/errors, repeated sets and evidence checks.

The rejected daily evaluator saves5.5133% wall time. For ranking subsequent
work only, the completed three-update whole-tree report from2026-09-30 has
daily32 median3.219553s within total11.657018s (27.619%); T7/T30 medians
are1.624638/6.738458s. Removing that entire daily branch would have an
optimistic1.381578x ceiling. Applying the FFN component ratio to the entire
daily branch predicts only1.015463x under unchanged other costs. These are
historical parameter-AD update estimates, not measured current dual-AD
FFN full updates or admitted same-workload gains. Recurrent work therefore
needs attention to approach the10x research target.

An existing raw-bank capture check also found no wholly inactive original
chunks in any of32 route/sample records: each expert5 sample has7542 active
rows and0/59 inactive chunks. A whole-inactive-chunk skip has no measured
opportunity on these four samples; this does not cover every recurrent day.
No such implementation or new GPU extraction ran. A separate native scan
`prevent_cse=False` component investigation is owned by the active next task;
it retains FP32/native AD and does not reopen the failed selective-save policy.
This report claims no result for that route.
