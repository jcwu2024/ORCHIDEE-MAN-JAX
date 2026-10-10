# Fixed-local-S4096 parameter-gradient precision proposal

Status: scientifically approved free-node screen completed on2026-10-02.
The user approved the command and3500s bound after accepting preparation.
This is one approximate reverse parameter-gradient diagnostic, with no
training or production adoption. [Actual results](shared_parameter_gradient_precision_results_20261002.md)
reject the screen: all 16 parameter engineering groups and hard admission
passed, but strict state parity failed and the real F4S median ratio 1.039317
missed the 1.05 speed gate. All three bounded warm pairs completed. The route
used 1508.347398 seconds / 0.418985388 free GPU card-hours, CNY0; both backend
attempts and the sole known compatibility repair are closed. No expansion or
adoption is implied.

## Method and scope

Forward, input/state derivative, and bias-gradient operations remain original
FP32, with FP64 physics. Only the kernel tangent term uses
`S * dot_F16_F16_F32(x, W_t / S)`, with fixed local `S=4096`. Native
`custom_jvp` transposition gives the intended kernel pullback
`dot_F16_F16_F32(x.T, S*g) / S`. Both activation and scaled cotangent operands
are quantized; accumulation/output are FP32. This is an approximation to the
original FP32 derivative, without an exact finite-precision adjoint claim.
Symbolic-zero branches exclude the low-precision term from forward, state-only,
and bias-only differentiation. Rank-one matrixization applies only to the
kernel tangent term.

The overlay patches `_linear` aliases in `shared_transformer`,
`shared_backbone_execution`, `shared_backbone_projection_reuse`, and
`shared_decoder_macro_packed`, and `_ffn` aliases in the same set except
`shared_backbone_execution`. The actual expert decoder must resolve the macro
module globals. Canonical tokenizer helpers and other unpatched neural dots
retain their source implementations. This is not all-neural coverage.
Neural provenance is `strong_baseline_architecture.py::_linear` lines886–887
and `_ffn` lines897–898; no Fortran process, physical equations, architecture,
loss, sampling, parameter capacity, or recurrent horizon is changed.

There is no public loss scaling, clipping, automatic scaling, FP32 fallback,
custom-VJP replacement, factor sweep, or new precision variant. Candidate
parameter JVP must never serve as the scientific correctness oracle. Use the
original FP32 VJP or original FP32 direction JVP for parameter acceptance.
Native state JVP remains usable on the original isolated state path.

## Completed bounded evidence

Local ORCJAX JAX0.10.1 CPU and staged canonical JAX0.4.38 CPU both passed
20 native JAXpr/CUDA-targeted StableHLO cases and13 explicitly emulated FP32
runtime interface cases. CPU cannot execute the native F16 preset and rejects
it explicitly; the runtime emulator verifies plumbing only. Target CUDA IR is
structural evidence without a GPU backend compilation. Both reports record
zero real-model calls.

The CUDA-targeted IR shows FP32 primal/input-gradient dots, the local FP32
cotangent multiply4096, FP16 conversion of both low-dot operands, FP32
accumulation, and FP32 divide4096. Forward/state-only/bias-only cases have zero
F16 dots; parameter-only/dual cases have one. Dimension-specific artifact
names preserve all20 cases. A canonical-version lowering API difference
(`backend_or_name`, `name_stack`) was caught on test01 and repaired; the
earlier failed report remains remote alongside the passing `-v2` report.

Host rounding illustrations are not native V100 measurements. At cotangent
scale1e-7, local4096 reduces illustrated weight-gradient relativeL2 from
0.125126 to0.0002571; at1e-9 it gives0.0040428 and still loses one of24
cotangents. Large cotangents overflow: scale32 gives17 nonfinite scaled
cotangents. The policy stops on nonfinite results. Normalized parameter
directions are especially unsuitable: scale1e-4 loses97 of128 entries after
division4096 and FP16 conversion, with illustrated JVP relative error0.977689.
The observed tangent-additivity error also rules out exact linearity claims.

The optimized-HLO gate resolves symbols within each computation and requires
the first two actual dot/GEMM operands to be FP16 with FP32 output. Host
fixtures passed for typed shapes/layouts, local-name scope collisions, and
bare-name tuple GEMM output. A source-only Luna/high independent review found
no remaining parser/launcher blocker. Native V100 optimized HLO now confirms
FP16 operands and FP32 output for the illustrative and two source-backed F4S
shape classes; opaque or unfamiliar HLO remains an incomplete outcome.

test01 (`ibtest01`) passed `bash -n`, actual `--dry-run` expansion for GPU0/1,
four original/repaired public CLI checks, four invalid-argument rejections,
and actual staged policy/helper imports with a tiny host metric. The helper
loads from the overlay sibling by file spec; immutable checkout package paths
need not discover the flat overlay. These checks read no Teacher data and
invoke no GPU or real model.

Evidence is under
`outputs/diagnostics/shared-parameter-gradient-precision-20261002/`:
`cpu-interface-final-v3/interface.json`, `cpu-interface-canonical-v2/interface.json`,
`draft-host-checks.json`, and `preparation-final-v2/target-shell-checks.json`
plus the two expanded dry-run command files and dimension-specific IR.

## Proposed bounded execution and gates

The single-GPU free-node route uses the immutable c599236 checkout, foundationpose
container, canonical GPU environment, staged overlay, saved original F4S
binary/results, and unchanged update-0 inputs. It records node, CPU affinity,
host load, visible GPU processes, flags, wall time, and card-hours. It requires
an empty selected GPU on `gln01`/`ibc13b02n04`, outside a Slurm paid job. There
is no paid submission or paid fallback.

```bash
bash /WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/transfer/shared-parameter-gradient-precision-20261002/run_shared_parameter_gradient_precision.sh --gpu 1 route
```

The declared working directory is
`/WORK/liwei_work/jcwu/ORCHIDEE-MAN-JAX/runtime/checkouts/shared-split-width-v1`.
Outputs go under the corresponding `runtime/outputs/diagnostics/` family.

| Stage | Budget and stopping rule |
|---|---|
| Fresh runtime preflight |300s maximum; no Teacher inputs |
| Synthetic component interface |600s shared across first attempt and any repair retest |
| Conditional public F4S |2400s shared across the first attempt, any necessary repair retest, and repaired attempt |
| Whole launcher |3500s cumulative cap including preparation/retries; at most0.97222 free GPU card-hours |

The initial GPU interface verified tiny synthetic plumbing and illustrative shapes
`[384]→384`, `[128,384]→1536`, `[2,128,384]→1536`, and the known masked
FP64 scalar-loss readout `[4,128,256]×[256,1]`. The readout reproduces the
shape/plumbing class of the historical `operand_upcaster` rejection; it does
not certify the full graph. These four are retained as illustrative component
evidence, not claimed as the actual F4S FFN batch layout. Their speed ratios
were 0.774733, 0.986845, 1.072654 and 0.874752 respectively; the aggregate
component speed screen failed. Strict primal/input/bias and kernel engineering
comparisons passed for these shape pairs.

Source inspection and the existing real F4S forward StableHLO identified the
two actual shape classes `[4,128,1,384]×[384,1536]` (decoder) and
`[4,7,1,768]×[768,3072]` (backbone). The corrected fixed-policy interface
measured speed ratios 1.095207 and 0.990471 over five interleaved pairs;
both strict and engineering comparisons and native codegen passed, but the
aggregate speed≥1.05 component gate remained false. Independent NumPy
verification retained both failed speed decisions. No precision factor,
architecture, loss, sampling, or numerical gate was changed.

The lead's execution decision, following the user's standing instruction to
reach a real-model conclusion within existing authorization, continues the
real F4S diagnostically despite component speed failure. The narrow continuation
guard still requires all interface correctness, alias restoration, native JVP,
shape, policy hash, JAX version, flag, codegen and memory evidence. Only the
component speed prerequisite is bypassed; its failed result stays false.

F4S retains the original891-leaf input ABI, full300-leaf/64,884,481-parameter
gradient tree, and four3854-coordinate state derivatives. Its interface
prerequisite must match policy identity/hash, JAX version, compiler flags,
the two exact source-backed representative shape classes and either the
passed component speed screen or the explicit diagnostic continuation. The
saved original full-AD binary is loaded and replayed; it is never recompiled.
Original loss/aux/parameter replay bridge remains strict. Original state
self-drift is fully saved and diagnostic, without an added all-self-pass gate.
It must still be finite, physical, mask-valid, undefined-zero, and within memory.

Acceptance requires all16 semantic parameter groups: reference norm≤1e-12
uses maxabs≤1e-6; otherwise relativeL2≤0.01 and cosine≥0.9999. Loss relative
error≤0.001 is retained alongside strict loss, auxiliary, and every state
coordinate (`rtol=1e-5`, `atol=1e-6`). Strict all-parameter results are retained
separately. State engineering summaries are descriptive and never waive a
strict state failure or enlarge tolerance based on original self-drift.
Finite values, all11 physical hard counts zero, exact source-mask invariance,
actual per-sample undefined-state gradients exactly zero, unchanged ABI, and
both estimated and observed memory≤24GiB are required.

After valid hard admission, three interleaved F4S warm pairs run even if the
initial or a paired strict/engineering comparison fails. Full first results,
failed pair results, all state replays, and comparison metrics are preserved.
Nonfinite, physical, mask, undefined-zero, or memory failures stop immediately.
Numerical failures remain rejection and prevent expansion; timings are
diagnostic until all gates pass. F4S median speed≥1.05x is an additional screen
gate, with all three timings reported. This is neither a full optimizer update
nor recurrent validation.

One exact known compiler compatibility repair is allowed:
`--xla_disable_hlo_passes=operand_upcaster`, only after
`UNIMPLEMENTED: Unsupported algorithm on the current device(s): ALG_DOT_F16_F16_F32`.
The first JSON/log remains intact. If the rejection occurs in the synthetic
stage, its repair retest shares600s. If it first occurs in F4S, a small
repaired interface retest and one repaired F4S attempt share the original2400s
window. There is at most one repair total and at most two public backend
attempts. The repaired result is explicitly the precision+compiler-compatibility
combination. No other flags or numerical failures trigger automatic retry.
An exhausted budget is incomplete compatibility evidence, not proof that the
precision strategy is ineffective.

## Limits and follow-up decision

The eligible kernel-gradient wall-time fraction is unknown. Whole-step speed
cannot be inferred from component improvements or an81% historical AD share.
Only the conditional bound `R=1/(1-p+p/r)` applies, with unknown eligible
fraction `p`; there is no measured forecast or10x claim.

A passing screen does not authorize T7/T30, full-update/AdamW validation,
training, new paid jobs, or adoption. Those require a concrete later budget
and consequential-loss native AD/physical/state/update checks. Full-update
parameter/AdamW acceptance remains strict. The original FP32 state parity
is not independently certified by this proposal or the earlier FP64 reference.
