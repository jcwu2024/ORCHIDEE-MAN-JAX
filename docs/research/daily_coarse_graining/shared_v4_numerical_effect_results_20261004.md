# V4 saved-array numerical effects — 2026-10-04

The combined V4 daily/V7 T7 execution fails the original strict parameter/state-gradient parity contract (`rtol=1e-5`, `atol=1e-6`). The saved mixed gradient has 977 failed coordinates across 12 leaves; fixed-input T7 cross comparisons have 1,357 and 1,348 failures across 17 leaves. This establishes disagreement with the reference under that contract. It does **not** establish mathematical derivative error, learning damage, or harmlessness. The reference is not an independently established derivative truth.

The earlier complete same-input proposal medians remain **13.103010 → 11.186190 s, 1.171356×**. Updated parameters, stored Adam moments, objective/auxiliary values, gradient norm and clipping passed the original proposal comparisons across all three pairs. B state-gradient failures persist across all three. This report neither waives the strict failures nor adopts V4.

## Scope and artifacts

This is an offline float64 reduction of saved actual arrays on free CPU `test01` (`ibtest01`), with no model import/call, GPU call, AD compilation, optimizer execution or learning execution. The numerical analysis took **43.016442 s** after Python/NumPy imports, including raw IO, reductions and source HEAD inspection. It covers 300 numeric parameter leaves, 64,884,481 coordinates and two static `None` nodes.

Local artifact directory: `outputs/diagnostics/shared-v4-numerical-effect-20261004/`; remote mirror: `/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/outputs/diagnostics/shared-v4-numerical-effect-20261004/`. `summary.json` records exact inputs and metadata. The nine metric JSON files provide global, module and leaf norms, direction/parallel/perpendicular errors, coordinate allowance ratios, zero/sign/rounding classes and squared-error concentration. `source-scope.json` retains 11 exact bounded source snippets. `analyze_saved_numerical_effect.py` and `record_source_scope.py` are executable analysis records.

Inputs are actual pair2 reference/candidate full proposals from `shared-v4-same-input-update-20261003/model/`. The common optimizer/parameter input is `pair-1-reference.full-proposal.pkl`, with optimizer step 2 → 3. Common inputs are source-bound to that actual reference output; unsaved per-arm input arrays were not byte-compared. T7 uses branch-attribution calls 09/18 and 37/28 from `shared-v4-branch-attribution-20261003/model/`.

The first CPU attempt stopped before metric output because saved `update["loss"]` is a mapping. Metadata handling was repaired without changing arithmetic or tolerance; `cpu-attempt-0-repair.json` records this. A source-inspection attempt then found that the macro helper was absent from immutable HEAD: it is a staged overlay. `source-inspection-repair.json` records resolution through the actual field adapter's relative import. The immutable checkout was not modified.

## Global directions and actual stored increments

All deltas are formed **after casting operands to float64**. Actual increments are the differences between saved proposal values and their common saved input, including stored rounding. They are not reconstructed unrounded AdamW updates.

| Quantity | Reference L2 norm | Cross-error L2 norm | Error/reference norm | Cosine | Direction angle (rad) |
|---|---:|---:|---:|---:|---:|
| Mixed parameter gradient | 10.580103490 | 2.535834699e-4 | 2.396795741e-5 | 0.9999999997207973 | 2.363059760e-5 |
| T7 parameter gradient, repeat0 | 0.699587180 | 2.523835639e-4 | 3.607607047e-4 | 0.9999999645175568 | 2.663923554e-4 |
| T7 parameter gradient, repeat1 | 0.699586730 | 2.520376748e-4 | 3.602665174e-4 | 0.9999999646174825 | 2.660169825e-4 |
| Actual parameter increment | 0.0382356964 | 4.479208481e-6 | 1.171472970e-4 | 0.9999999931552506 | 1.170021328e-4 |
| Actual first-moment increment | 0.115884479 | 2.363054549e-6 | 2.039146716e-5 | 0.9999999997921984 | 2.038637161e-5 |
| Actual second-moment increment | 1.239481319e-5 | 2.030681339e-10 | 1.638331541e-5 | 0.9999999998711657 | 1.605203200e-5 |

Mixed gradient norm changes by 4.006944234e-6 relative to reference; the signed parallel error is 4.239093067e-5 and perpendicular error 2.500151699e-4. T7 repeat0 norm changes by −2.433118700e-4, with signed parallel error −1.702426819e-4 and perpendicular error 1.863193296e-4. Direction agreement alone does not establish local-coordinate or learning acceptance.

Comparing the updated weights with their large existing norm (301.825694572) yields relative error 1.484038159e-8. Comparing the same error with the actual step yields 1.171472970e-4. The latter makes the update-scale discrepancy visible. Stored-weight cosine rounds to 1.0 although its angle is 1.484038156e-8 rad and perpendicular error is 4.479208473e-6; this is floating statistic rounding, not exact array equality.

Stored first- and second-moment relative errors are 2.398252809e-5 and 6.334274479e-6. `optimizer.step` is recorded separately and excluded from moment norms. Applying `1e-5/1e-6` to increment arrays in the detailed JSONs is a **descriptive diagnostic**, not the original formal updated-parameter/optimizer gate and not a new learning admission gate. In particular, tiny moments can be covered by the absolute allowance while retaining measurable relative differences.

## Local readout and concentration evidence

Indices below identify **parameter coordinates**, not physical state fields.

Mixed gradient's largest allowance ratio occurs at `shared_transformer/readout_second/kernel[229,0]`: reference −0.05103332921862602, candidate −0.05104856193065643, absolute error 1.523271203e-5 versus allowance 1.510333292e-6, or **10.085663×** the strict allowance. Its local relative error is 2.984855635e-4. Of the 977 failed mixed coordinates, 56 exceed twice the allowance and two exceed ten times.

T7 repeat0's largest ratio is `shared_transformer/readout_second/kernel[164,0]`: reference 0.020180895924568176, candidate 0.020166657865047455, error 1.423805952e-5 versus allowance 1.201808959e-6, or **11.847190×**. Its local relative error is 7.055216762e-4.

| Module | Mixed failures | Mixed relative L2 error | Mixed squared-error share | T7 repeat0 failures | T7 repeat0 relative L2 error |
|---|---:|---:|---:|---:|---:|
| `readout_first` | 482 | 2.279210623e-5 | 7.2761% | 682 | 3.618385843e-4 |
| `readout_second` | 78 | 2.013940569e-5 | 5.3908% | 120 | 3.798101378e-4 |

Other leading mixed squared-error shares are decoder bridge 16.4118%, tokenizer axial block0 15.8488%, coordinate block1 12.6773%, input projection 10.0774%, and coordinate block0 9.8122%. Modules use actual parameter path prefixes, including block index. Counts and energy shares are not scientific importance rankings.

For actual parameter increments, readout_first reference norm is 0.001682214084 and cross-error norm 2.072355034e-7 (relative 1.231921105e-4); readout_second reference norm is 8.509192502e-5 and error 1.320868496e-8 (relative 1.552284186e-4). At these modules' largest allowance-ratio coordinates, local increment relative differences are 1.1111% and 0.67114%, each one stored-input ULP; these are not maxima over all local relative ratios.

For actual second-moment increments, **97.2752%** of squared cross-error is in readout_second. That module's reference increment norm is **1.230428183e-5**, error norm **2.002824278e-10**, and relative error **1.627745776e-5**. At `[14,0]`, reference increment is 3.538164492e-6 and candidate 3.538260898e-6, differing by 9.640643839e-11. Concentration does not establish learning importance or damage.

Actual parameter increments contain 484 reference-zero/candidate-nonzero coordinates, 515 with the converse, 36,024 both-zero, 56,504,716 equal nonzero increments, 8,342,683 changed nonzero increments of the same sign and 59 changed-sign increments. In total, 8,343,741 coordinates differ. The stored subtraction classes expose rounding effects; they do not establish their downstream learning consequences.

A local small-step example is `shared_transformer/backbone_blocks/7/ffn/first/kernel[233,216]`: reference step −1.235865057e-6, candidate −1.394189894e-6, error 1.583248377e-7, or **12.81085% of the local reference step**. Common input is 0.0108502097427845; the difference is 170 input ULPs. It fits a descriptive absolute allowance of about 1e-6, but that fact alone does not certify increment accuracy or harmlessness.

## First-order directional scalars only

| Stored gradient · actual stored step | Reference step | Candidate step |
|---|---:|---:|
| Reference gradient | 0.004585676558932274 | 0.004585743992963880 |
| Candidate gradient | 0.004586953926334472 | 0.004587021167780818 |

The candidate/reference step difference under the reference gradient is 6.743403161e-8. These positive scalars are only first-order directional measurements of saved arrays. They are not observed loss changes, accepted predictions of learning change, or curvature evidence; no ascent/damage conclusion is drawn from them. Stored input joint objectives are 0.6371876799063789 and 0.6371876653937641. The saved optimizer norms/clip scales are 10.5801029205/0.09451703727 and 10.5801458359/0.09451665729.

## Bounded source interpretation and unresolved learning effect

Fresh source reads use immutable checkout HEAD `c599236bcf57b745e9bafed678298010f4c3b58f` and the actual staged adapters named by the original candidate report. The exact snippets and remote paths are in `source-scope.json`:

- V4 `decode` lines 32–72 and `decode_chunk` 59–69: expert5 has 7,542 rows, 128-row chunks, 59 chunks/7,552 padded rows and ten zero pads. It uses the same projected coordinate-block helper, two coordinate blocks, FP32 linear/GELU readouts and active masks. Only chunk `lax.map` becomes `vmap`; other row shapes use the original helper.
- Original staged macro `_decode_mapped_rows` lines 25–76: the same packing/readout and mask math with chunk `lax.map`. The canonical file is the sibling of the staged field adapter under `runtime/transfer/shared-combined-recurrent-20260926`, not a nonexistent file in immutable HEAD.
- Original daily dual wrapper lines 11–42: replaces only explicit initial states and applies native `value_and_grad` to parameters and initial states. The reviewed adapter/dual contain no `custom_vjp`, `custom_jvp` or `stop_gradient` call.
- Immutable shared transformer coordinate helper lines 537–607, `_linear` lines 886–887, norm/FFN lines 890–898 and functional optimizer lines 371–426: `_linear` explicitly casts its input to FP32 before matmul; the optimizer returns fresh parameters and first/second moments.

These are bounded source facts, not exhaustive branch coverage or verification of every local Jacobian. Changed FP32 batching, reduction and rounding order are plausible mechanisms; source review does not establish the unique compiled backward cause. Absence of an added AD omission does not prove derivative correctness. No source-demonstrated fast repair preserving the measured vmap speed gain exists in this evidence. Ordered shared-parameter adjoint accumulation remains speculative and would require new AD and performance evidence.

The completed discriminator continues to localize the parameter cross discrepancy to T7: daily parameter crosses pass, both T7 parameter self repeats pass, and T30 parameter repeats pass. Fresh mixed failures (960/954, all among the original 977) and assigned-T30's 960 failures preserve that bounded localization; they do not resolve the learning effect. T30 state has one self-repeat failure. Prior GPU accounting remains in the owned reports: one physical GPU1 was used in each of two windows and released after each; no GPU time was added here.

This task supplies numerical scale, direction, local-coordinate and source evidence for a separate learning-sensitive decision. It creates or executes no learning driver and changes no defaults, architecture, precision, loss, sampling or adoption status. Neither a blanket harmlessness claim nor rejection of the scientific route follows from the strict parity failure alone.
