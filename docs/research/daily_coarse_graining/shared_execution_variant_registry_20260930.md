# Shared execution variant registry (2026-09-30)

Updated: 2026-10-02.

This registry identifies the saved experiments, not production model versions.
The common model has 64,884,481 parameters (768/eight-block backbone and
384/two-block decoder), artifact 742, fresh update-0 checkpoint, FP64 physical
calculations, and the canonical daily objective. The immutable Explore1000 code
checkout used by the union diagnostics is `runtime/checkouts/shared-split-width-v1`
at `c599236bcf57b745e9bafed678298010f4c3b58f`. Remote paths below are under
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/`.

| Version / execution graph | Saved executable or reference | Input identity and actual measurement |
| --- | --- | --- |
| **F: FP32 field**; combined source-reuse backbone + expert128 field decoder, default route | `shared-combined-real-daily-20260925.candidate.executable.pkl` and `.candidate.pkl`; original canonical truth `shared-expert-real-daily-20260925.reference.pkl` | Artifact 742 original rows 0-3, update-0 checkpoint `shared-split-profile-c599236.updates.update-000000.checkpoint.pkl`; public mean daily loss, 160 auxiliary coordinates, 300 parameter-gradient leaves. The free-node combined admission passed strict `1e-5/1e-6` against canonical truth. In job 15461046 the saved field binary was reloaded (zero new field graph compiles), its result saved as `shared-union-micro4-paid-20260929-15461046.field-replay.pkl`, and its warm median measured as 0.374004 s. |
| **U4: FP32 union**, opt-in dynamic projection bank with the same source-reuse backbone | Job 15461046 `shared-union-micro4-paid-20260929-15461046.candidate.executable.pkl` and `.candidate.pkl`; compared with F replay and canonical truth | The same four distinct rows (590 prepared leaves match; row hashes are in the [job report](../../../outputs/diagnostics/shared-union-micro4-paid-20260929-15461046.json)), checkpoint, loss and 64,884,481 parameter-gradient coordinates. Strict loss/auxiliary/parameter gates and all 11 physical hard-count gates passed. Three interleaved warm pairs gave median 0.369003 s versus F's 0.374004 s (1.01355x); estimated live and active-buffer peaks rose about 12.8%. Union lowering/backend compilation took 100.252/579.554 s, separately from the warm comparison. |
| **U0: FP32 union staged full AD reference**; checkpointed neural/physical/loss six-unit graph | `shared-union-integrated-20260927.pkl` and `.json`, completed FP32 staged cache `shared-neural-reference-20260927-staged.stages-cache/` | Artifact 742 **row 0**, update 0, canonical daily loss, all 300 parameter leaves and 3,854 continuous-state gradient coordinates. This is the saved numerical reference for the precision screen, with native full parameter/state AD. It is neither an independent field-route state-gradient reference nor an original-four timing reference. |
| **M0: union neural TensorCore candidate**, U0 graph with scoped native `F16_F16_F32` neural dots; physical path remains FP64 | Jobs 15469456 and repaired 15469965; [repaired report](../../../outputs/diagnostics/shared-union-neural-precision-paid-20260929-15469965.json), unit-0/1 StableHLO and private `.stages-cache/`. No candidate result or full executable exists. | Same saved U0 reference, row 0 and checkpoint; multiplier 1. The first job stopped on a seven-leaf mask-schema error before candidate compilation. The repaired job validated the saved reference and compiled neural-forward unit 0 (77.483 s backend), then stopped at mandatory physical-forward unit-1 cache miss before its backend entry. No candidate loss, gradients, physical screen or warm timing was measured. |
| **U0 cache-only replay**, original FP32 union graph, not a precision candidate | Job 15479295 [report](../../../outputs/diagnostics/shared-union-cache-compatibility-paid-20260929-15479295.json), old/new unit-0 IR and job-specific cache | The saved U0 reference validated; unit-0 ordered ABI matched (301 inputs/1,798 outputs), but the required native cache lookup missed and blocked the named model backend. No new model result, gradients or timing. The [cache diagnosis](../../../outputs/diagnostics/shared_union_cache_compatibility_results_20260929.md) finds equal saved IR text/parsed structure, while original GPU native key components are unavailable; the miss cause remains unknown. |
| **N4: FP32 field with normalized-key banks**, field-only expert128 and admitted backbone; no union or precision change | `shared-normalized-keys-real-daily-micro4-20261001.candidate.executable.pkl`, `.candidate.pkl` and `.reference.pkl`; [result](shared_normalized_key_results_20261001.md) | Same artifact742, update0 and original four distinct rows. Loss, all160 auxiliary values, all64,884,481 parameter gradients and11 physical counts pass. Seven warm pairs give0.374225/0.349094s (1.071988x). Estimated live memory10.566/8.595GiB; observed10.582GiB is cumulative across both binaries. This admits daily parameter AD only. Component failures remain separate; the corresponding T7 variant fails below. |
| **N7: same whole-bank normalization, T7 native checkpoint dual AD** | `shared-normalized-keys-recurrent-T7-20261001-candidate.candidate.executable.pkl` and `.candidate.pkl`, compared with the same-process default dual replay; [result](shared_normalized_key_results_20261001.md) | Artifact742/update0, one unchanged seven-day case, all parameter and3,854 initial-state derivatives. Loss/20auxiliary values/all64,884,481 parameter gradients/14physical counts pass;7state coordinates fail. Seven warm pairs give1.575990/1.516121s (1.039489x), live3.249/2.745GiB. Seven further saved-binary pairs confirm four persistent state differences beyond within-arm drift. N7 is not admitted to T30 or full updates. |
| **C2: original FP32 field expert5 component, FUSION minimum group 2 versus 5** | `shared-command-buffer-loops-threshold-component-default-20261001.{default,fusion2}.native_vjp.executable.pkl` and complete initial `.result.pkl` trees; [result](shared_fusion_capture_threshold_results_20261001.md) | Real artifact742/update0 row0 raw-bank capture, original expert128/two-block decoder, synthetic sin cotangent. Scope is 3,548,928 decoder parameters, not all model parameters or the public loss. Seven pairs give 0.100610/0.101692s (0.989368x), live +33.37%. Outputs/all raw-input VJPs pass; strict parameter parity fails with self-drift in both arms. Candidate stops; public-loss/physics/recurrent/full-update checks were not run. |
| **C3: original FP32 field expert5 component, CUDA12.3 default versus WHILE capture** | `shared-command-buffer-loops-runtime-component-{default,while}-20261001-compat.*.native_vjp.executable.pkl`; complete initial trees and seven raw parameter-VJP repeats; [result](shared_cuda_capture_runtime_results_20261001.md) | Same real row0 capture and synthetic cotangent as C2, unchanged 3,548,928 decoder parameters/formulas/native AD. Isolated libcuda545.23.08/libcudart12.3.101 actually enable capture. Seven same-runtime pairs give 0.101039/0.112683s (0.896667x), live -22.30%. Parameter/raw-input parity fails far beyond self-drift; stop model integration. Public loss, physics, T7/T30 and full updates were not run. |

**C4: original FP32 field expert5, whole chunk-axis vmap.** Saved under
`shared-command-buffer-loops-chunk-vmap-default-20261001-vmap2.*`;
see [chunk batching](shared_chunk_batch_results_20261001.md). Same real row0,
original 3,548,928 decoder parameters, chunk128/padding and unscaled synthetic
cotangent as C2, with the legacy default runtime/executable reused. Seven pairs
give native VJP 0.102476/0.022538 s (4.5469x), estimated live 2.260/1.015 GB.
Output/parameter/raw-input gates fail beyond self drift; C4 is not a public-loss
or full-model variant. R4 learned-dot repair and V4 public-loss execution are
separate tests below, retaining this failed synthetic evidence.

**R4: original field, expert5 preserve128 learned execution.** Public-loss
assets are `shared-command-buffer-loops-shape-preserved-daily-default-20261001-vmap.*`
under the `shared-command-buffer-loops-20261001` local evidence directory.
Same original four samples and all 64,884,481 parameters; 128-row learned
dots/norm/readout, batched remaining coordinate operations, original runtime.
Loss/all 160 auxiliary values/all parameter gradients/11 physical counts pass
strict gates. Seven pairs give 0.374888/0.267035 s (1.403892x); estimated live
10.565/7.701 GiB and two-binary active peak 10.582 GiB. R4 is daily parameter AD
only, with no returned state-gradient or complete-update admission. Its separate
synthetic component still fails parameter/input VJP gates. See
[chunk batching](shared_chunk_batch_results_20261001.md).

**R7: R4 preserve128 repair, T7 native checkpoint dual AD.** Stem
`shared-command-buffer-loops-shape-preserved-T7-default-20261001-vmap`;
raw state repeats and interval analysis are preserved in the R4 evidence
directory. Same artifact742/update0 and original seven-day public loss;
all 64,884,481 parameter gradients/loss/20 auxiliary values/14 physical counts
pass, but 4/3,854 initial-state coordinates and every one of seven cross-arm
state checks fail (3–5 coordinates). Median 1.572075/1.218308 s (1.290376x)
does not admit the candidate. Canopy2ground coordinate199 has separated
within-arm ranges beyond the original strict allowance; self drift does not
explain it. Stop this multi-day variant before T30/full updates. The largest
error among initially failed coordinates is 6.678700e-5; global max24.811663
belongs to a passing large-gradient coordinate and is not that failure metric.

**V4: original field, complete expert5 chunk call mapped with native vmap.**
Stem `shared-chunk-vmap-daily-expert5-micro4-20261001-v2`; unique adapter
`runtime/transfer/shared-chunk-vmap-daily-20261001/shared_chunk_vmap_execution.py`
and starting metadata/SHA identify its immutable executed source. Same original
four samples, complete 59×128 chunk call including readout, padding10, original
public loss and all parameters. Loss/160 auxiliary values/64,884,481 parameter
gradients/11 physical counts pass unchanged strict gates. Seven pairs give
0.375510/0.221913 s (1.692144x); estimated live7.446GB, cumulative active
peak10.836GiB. Lower/backend102/720s, launcher873.501s (0.242639 single-card
hours, free). This is parameter AD only; the separate V4S state gate fails
below. V1 failed before lowering on module load order, repaired in V2
with the failed record preserved. C4 synthetic failures are not erased.
See [V4 daily report](shared_chunk_vmap_daily_results_20261001.md).

**V7: V4 T7 native checkpoint dual AD, rejected.** Stem
`shared-command-buffer-loops-whole-vmap-T7-default-20261001-v4`, original
default dual reference reused with zero model compiles. Median1.578532/
0.906024s (1.742263x); loss/20 auxiliary values/all64,884,481 parameter
gradients/14 physical counts/finite state/memory pass. Initial state6/3854
fails; seven pair failure counts7,7,7,5,7,5,6. Coordinates33,199,385,428
have disjoint repeat ranges beyond strict allowance; this is persistent bias,
not noise-only. Initial failed-coordinate max1.589656e-4; global67.595623 is
a passing large-coordinate difference. Lower146.184s/backend2527.958s,
launcher2961.715s (0.822699 free card-hours). No V4 T30/adoption replay.
See [V4 recurrent report](shared_chunk_vmap_recurrent_results_20261001.md).

**F4S: original field daily micro4, parameter plus initial-state AD.** Stem
`shared-chunk-vmap-daily-dual-v4-20261001-reference`; expert128 decoder and
backbone source reuse, original four samples. Dynamic ABI is
`(parameters, initial_states[4,3854], daily_samples)` with891 input leaves;
only parameters/explicit states are differentiated. Reference loss/auxiliary/
all parameter gradients bridge to the admitted field binary and saved field
replay. All15,416 state coordinates finite,15,334 nonzero;11 physical counts0.
Estimated live11.348GB, observed active peak11.367GB. Saved original result
and executable are retained; this is an independent four-row field reference,
not U0 union row0 or a two-process "dual" benchmark.

**V4S: V4 with the same daily full-AD interface, rejected.** Stem
`shared-chunk-vmap-daily-dual-v4-20261001-candidate`; unique overlay
`runtime/transfer/shared-chunk-vmap-daily-dual-20261001`, free GPU1. It replays
F4S in-process at the original tolerance. Median0.382541/0.228810s
(1.671871x); loss/160aux/all64,884,481parameters/11physics counts pass
initially and in all seven pairs. Initial27/15,416 state coordinates fail;
paired counts26,29,27,25,28,26,26, with34 unique and20 always-failing
coordinates. Stable self ranges with persistent cross offsets confirm actual
state mismatch. Estimated live7.437GB, active peak11.367GB; lower112.827s,
backend845.197s. Actual historical launcher exit0 with JSON`parity_failed`
does not imply admission. No V4 daily/recurrent full-update integration.
Extra state derivatives are not AdamW inputs. See [V4 daily report](shared_chunk_vmap_daily_results_20261001.md).

**G2C: native map batch_size2, original expert5 component.** Stem
`shared-chunk-group-component-20261001-g2`; complete128 chunk body,
59chunks, ten padding rows. Synthetic cotangent is unchanged. FullVJP
0.100129/0.070345s (1.423414x) but initial and all seven complete paired
output/parameter/raw-input gates fail. Preserve raw repeats and self evidence;
component speed grants no public-loss admission.

**G2S: native two-chunk map, public daily full AD, rejected.** Stem
`shared-chunk-group-daily-dual-g2-20261001`, dedicated overlay
`runtime/transfer/shared-chunk-group-20261001`. Reuse original F4S binary
with zero original compilations and identical dynamic three-argument ABI.
Original0.380717/G2 0.331364s (1.148938x); initial and seven paired public
loss/auxiliary/all parameter/physical gates pass. Initial25 state coordinates
fail; paired counts22,24,25,20,23,24,22. Seventeen coordinates fail everypair,
fifteen repeat ranges are disjoint beyond strict allowance. Initial failed
coordinate max0.535738; global1.842353 is not that metric. Lower108.236s,
backend878.829s; launcher1045.422s/0.290395 free card-hours. Historicalexit0
with JSON`parity_failed` is preserved. No G2 T7/T30/full-update admission.
See [G2 report](shared_chunk_group_results_20261001.md).

**GH1C: original FP32 expert5 raw-gather-only hoist, closed at component.**
Stem `shared-gather-hoist-20261001-component`, exact original raw-bank sample0
and sin0.11 synthetic cotangent, saved original forward/full-VJP binaries reused
with zero original compilations. Pad indices first; move only raw axis/local
K/V gathers outside59×128 map; keep concat/active bias substitution and every
learned/attention/norm operation inside. Initial and all seven output strict
gates pass; parameter/raw-input VJPs fail (initial720360/1420 coordinates).
Forward/full-VJP medians0.024373/0.024385 and0.099313/0.099659s give
ratios0.999535/0.996527; first/second-position timing groups also show no
benefit. Estimated live2.260→2.366GB. HLO confirms entry gather/scatter
relocation, removing bank-shaped reverse carries but replacing them with full-row
residuals;237→235 carry slots and247 dot_general metadata annotations retain
128 learned dimensions without implying algorithm/layout/bitwise identity.
Free GPU runtime preflight plus component0.025887799 assigned card-hours,
paid0. Actual transformation achieved; no further component GPU probe or public
four-sample/state/physical/full-update expansion. This is separate from G2/R4/V4
and does not reject broader indexing strategies. See
[gather-only report](shared_gather_hoist_results_20261001.md).

**M4: original-four field selected neural F16_F16_F32 screen, gradient-rejected.**
Reuse F4S and its dynamic ABI; FP32 neural storage/accumulation/results, FP64
physical work, original expert128/backbone and public loss. This is distinct
from M0's union-row0 staged graph. First stem
`shared-daily-field-precision-20261001` stops before candidate results on
`ALG_DOT_F16_F16_F32` unsupported-device error; lower105.108s, launcher258.870s,
0.07190846 free card-hours. All 86 traced helper shapes pass independent native
VJPs, which does not establish complete-model admission. A subsequent rank-one
matrixization has zero matches in the actual full trace, so its causal role is
unproven; the second full compile also fails. The subsequent exact
`operand_upcaster` compatibility repair succeeds. Stem
`shared-daily-field-precision-upcaster-20261001` completes initial plus seven
full-gradient pairs: **0.384209/0.336208 s (1.142772x)**, but all pairs fail
14/16 parameter groups and 663/1028 sample-state fields. Loss relative error
0.000337159 passes; global parameter/state relative L2 is 0.01635/0.41404.
Finite/physical/undefined-state checks pass. Keep this as a precision-plus-
compatibility measurement, not an admitted update improvement. An equivalent
dynamic native-AD scaling repair is the next diagnostic under unchanged gates.
See [field precision report](shared_daily_field_precision_results_20261001.md).

**M4S256: same selected-alias precision with equivalent dynamic native AD scaling.**
Stem `shared-daily-field-precision-scale256-20261001`, separate 892-leaf binary;
differentiate S*loss and unscale all gradients inside the timed executable,
returning original loss/aux. Initial and seven paired results pass all 16
parameter groups and four DOC fields; 76/1028 state fields still fail. All
finite/physical/mask/undefined-state gates pass. Stable full AD is
0.390357/0.333752 s (1.169600x), with no full-update admission. Lower/backend
127.693/851.131 s; one free GPU, 0.302356 card-hours. S1 wrapper drift is
separately recorded; conditioning attribution uses same-binary S1/S256.
The later S4096 replay below uses this dynamic binary while preserving
the objective, precision, full gates and separate result identities.

**M4S4096: replay of M4S256's same dynamic binary, no new model compilation.**
Stem `shared-daily-field-precision-scale4096-20261001`; initial and seven pairs
give 0.377239/0.323291 s (1.166872x). All 16 parameter groups, four DOC fields
and finite/physical/mask/undefined-state gates pass, but 76/1028 state fields
fail. Against S256, 17 failures resolve and 17 newly fail; 59 persist. Do not
describe the two 76-field sets as identical. Global parameter/state relative
L2 is 0.000884187/0.001013029; unchanged strict gates still fail separately.
Active peak 12.48 GiB; one free GPU, 0.029406 card-hours. No full update;
stop further factor tuning. The separately approved FFN-only scope completed
below and remains state-gradient rejected.

**M4FFN4096: FFN-only native scaled full daily AD, rejected.** Stem
`shared-daily-ffn-precision-scale4096-20261001`, new892-input binary. Only three
active FFN aliases use F16_F16_F32; all linear aliases remain original FP32,
physics FP64, native dynamic S4096/full-tree unscale remains in the timed
executable. Approved2026-10-02, completed initial/seven complete paired results.
All16 parameter groups, four DOC fields, finite/physics/mask/undefined-state
and24GiB gates pass;47/1028 sample-state fields fail in everypair (10/12/14/11).
Compared with broad S4096,35 failures persist,41 resolve and12 newly fail.
Strict errors remain reported separately. Original/candidate medians
0.381142/0.360129s (1.058350x), global parameter/state relativeL2
.000498977/.000877239. Lower125.513s/backend854.697s, estimated11.420GB,
active peak11.698GB, preflight+candidate .329658 free assigned card-hours,
cash0. Raw results/repeats/executable/source preserved. No complete update,
training or adoption; stop further factor tuning. See
[precision result and47-field evidence](shared_daily_field_precision_results_20261001.md).

**F30S: original field T30, parameter plus initial-state AD reference.** Stem
`shared-original-t30-dual-reference-20261001`; unchanged native checkpointing,
FP32 neural/FP64 physics, artifact742/update0/public T30 objective. Saved
`.reference.pkl` and `.reference.executable.pkl` bridge loss/auxiliary/all
64,884,481 parameter gradients to the admitted T30 parameter binary. All3,854
initial-state coordinates finite,3,833 nonzero;14 physical counts zero.
Lower172.664s/backend2545.616s, estimated live3,490,683,231B, observed active
peak3,514,478,592B. This is a reusable baseline, not a candidate speed result
or the older dual-process parameter-only benchmark. [Validation](../../../outputs/diagnostics/shared-original-t30-dual-reference-20261001/shared-original-t30-dual-reference-20261001.validation.json).

**Comparison boundary.** F versus U4, N4, R4 and V4 are separate matched
original-four warm parameter-AD comparisons. They omit returned state-gradient validation,
optimizer work and T7/T30; their ratios cannot be combined as a direct paired
comparison between candidates. N7 and R7 are separate matched T7 dual-AD comparisons,
including initial-state derivatives; their speed results do not override the
failed correctness gate. U0 and M0 use one row and a different staged output
interface; U0's repeated-row0 or staged timing must not be compared to U4's
four-distinct-row warm timing. M0 and the cache-only replay stopped before full
candidate execution, so neither supplies a precision or speed ratio. The older
isolated union component's 27 strict parameter-gradient failures remain separate
from U4's actual-loss pass. None of these versions demonstrates unchanged-mix
complete-update throughput, scientific accuracy, production precision adoption
or the 10x target.

Sources: [combined field admission](shared_combined_execution_admission_20260925.md),
[original-four union result](shared_union_micro4_results_20260929.md),
[union precision preparation](shared_union_neural_precision_preparation_20260929.md),
and the current [work order](../../NEXT_STEPS.md). Full remote binary/cache
contents were not reloaded for this documentation audit; identities and outcomes
above come from the saved reports and their explicit artifact paths.
